import torch
from torch import nn
from torch.optim.lr_scheduler import LRScheduler
import logging
import time
from torch.optim import Optimizer
from torch import amp
import os
import itertools
import json
from src import utils
from src.dataloader import resume_at_minibatch


class WarmupInverseSquareRootLR(LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        warm_up_steps: int,
        last_epoch: int = -1,
    ) -> None:

        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.warm_up_steps = warm_up_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        step_no = self.last_epoch + 1
        if self.last_epoch < self.warm_up_steps:
            lrs = [base_lr * (step_no) / self.warm_up_steps for base_lr in self.base_lrs]
        else:
            lrs = [
                base_lr * (self.warm_up_steps**0.5) / ((step_no) ** 0.5)
                for base_lr in self.base_lrs
            ]
        return lrs


def validation_step(
    model, dataloader, criterion, device, tokenizer, step_no, max_length, validation_minibatches
):

    model.eval()
    start_time = time.time()

    total_loss_ce = 0.0
    total_loss_aux = 0.0
    total_tokens = 0
    num_batches = 0

    for minibatch in itertools.islice(dataloader, validation_minibatches):

        minibatch = {
            k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in minibatch.items()
        }

        with torch.no_grad():
            logits, loss_aux = model(
                minibatch["input_ids"], padding_mask=minibatch["padding_mask"]
            )
            loss = criterion(
                logits.reshape(-1, logits.shape[2]), minibatch["output_ids"].reshape(-1)
            )

        total_loss_ce += loss.item()
        total_loss_aux += loss_aux.item()
        total_tokens += minibatch["input_ids"].ne(tokenizer.pad_token_id).sum().item()
        num_batches += 1

    elapsed_time = time.time() - start_time

    return {
        "type": "validation",
        "step_no": step_no,
        "num_tokens": total_tokens,
        "tokens_per_sec": total_tokens / elapsed_time,
        "loss_ce": total_loss_ce / num_batches,
        "loss_aux": total_loss_aux / num_batches,
    }


def train_loop(stage, model, dataloaders, tokenizer, run, config):

    train_config = config["train"][stage]
    device = torch.device(train_config["device"])
    grad_accum_steps = (
        train_config["effective_batch_token_size"] // train_config["minibatch_token_size"]
    )
    num_steps = train_config["num_steps"]
    checkpoint_steps = train_config["checkpoint_steps"]
    keep_checkpoints = train_config.get("keep_checkpoints")
    validation_steps = train_config["validation_steps"]
    validation_batches = train_config["validation_batches"]
    max_length = config["model"]["max_length"]
    cache_clear_steps = train_config.get("cache_clear_steps")
    router_aux_loss_coef = train_config["router_aux_loss_coef"]

    criterion = run["criterion"]
    optimiser = run["optimiser"]
    lr_scheduler = run["lr_scheduler"]
    scaler = run["scaler"]
    results = run["results"]
    step_no = run["step_no"]
    run_path = run["run_path"]

    # Initialise values - need to do learning rate, optimiser state, scaler state loading here too
    batch_tokens = 0
    start_time = time.time()
    total_loss_ce = 0.0
    total_loss_aux = 0.0
    # Resume exactly where we left off: skip the sampler forward in index space
    # (no batches are read from disk for skipped positions). Counted in minibatches, the
    # unit both stages share — pretrain's sampler converts to rows, SFT's already batches.
    resume_at_minibatch(dataloaders["train"], step_no * grad_accum_steps)
    optimiser.zero_grad(set_to_none=True)

    model.train()
    for accum_idx, batch in enumerate(dataloaders["train"]):

        # Move batch to device
        batch = {
            k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        with amp.autocast(device_type=device.type):

            # Forward pass
            # The mask keeps pad tokens out of the MoE's seat assignment: they are not
            # data, and letting them take seats would drop real tokens instead.
            logits, loss_aux = model(
                batch["input_ids"], padding_mask=batch["padding_mask"]
            )
            loss_ce = criterion(
                logits.reshape(-1, logits.shape[2]), batch["output_ids"].reshape(-1)
            )
            loss = loss_ce + loss_aux * router_aux_loss_coef

        # Compute loss and gradients
        minibatch_tokens = batch["input_ids"].ne(tokenizer.pad_token_id).sum().item()
        batch_tokens += minibatch_tokens

        scaled_loss = loss / grad_accum_steps
        scaler.scale(scaled_loss).backward()
        total_loss_ce += loss_ce.item() / grad_accum_steps
        total_loss_aux += loss_aux.item() / grad_accum_steps

        # Gradient accumulation
        if (accum_idx + 1) % grad_accum_steps != 0:
            continue

        # Step optimiser and scheduler
        scaler.unscale_(optimiser)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimiser)
        scaler.update()
        lr_scheduler.step()
        optimiser.zero_grad(set_to_none=True)

        # Increment step counter FIRST
        step_no += 1

        # Logging with the completed step number
        elapsed_time = time.time() - start_time
        result = {}
        result["type"] = "train"
        result["step_no"] = step_no
        result["num_tokens"] = batch_tokens
        result["tokens_per_sec"] = batch_tokens / elapsed_time
        result["learning_rate"] = lr_scheduler.get_last_lr()[0]
        result["loss_ce"] = total_loss_ce
        result["loss_aux"] = total_loss_aux
        result["token_length"] = batch["input_ids"].shape[1]
        result["grad_norm"] = grad_norm.item()
        results.append(result)
        logging.info(result)

        # Validation step
        if step_no % validation_steps == 0:
            validation_result = validation_step(
                model,
                dataloaders["test"],
                criterion,
                device,
                tokenizer,
                step_no,
                max_length,
                validation_batches * grad_accum_steps,
            )
            logging.info(validation_result)
            results.append(validation_result)
            model.train()

        # Checkpointing
        if step_no % checkpoint_steps == 0:
            save_checkpoint(
                model, optimiser, lr_scheduler, scaler, run_path, step_no, results, keep_checkpoints
            )

        # Delete tensors to free up memory
        del batch, logits, loss, loss_ce, loss_aux, scaled_loss
        if cache_clear_steps and step_no % cache_clear_steps == 0:
            torch.cuda.empty_cache()

        # Reset counters
        batch_tokens = 0
        start_time = time.time()
        total_loss_ce = 0.0
        total_loss_aux = 0.0

        # Stop after num_steps
        if step_no >= num_steps:
            break

    # A final save when num_steps is not a multiple of checkpoint_steps. Without it every
    # step since the last checkpoint is discarded — the weights, and the metrics too, since
    # save_checkpoint is also what writes results.json. Guarded so a run that ends exactly
    # on a checkpoint boundary does not write the same step twice.
    if step_no % checkpoint_steps != 0:
        save_checkpoint(
            model, optimiser, lr_scheduler, scaler, run_path, step_no, results, keep_checkpoints
        )

    return None


def checkpoint_steps_on_disk(checkpoints_path):
    """Step numbers of the checkpoints in a run, ascending. Filenames are "<step>.pt"."""
    if not os.path.isdir(checkpoints_path):
        return []
    steps = []
    for name in os.listdir(checkpoints_path):
        stem, ext = os.path.splitext(name)
        if ext == ".pt" and stem.isdigit():
            steps.append(int(stem))
    return sorted(steps)


def _prune_checkpoints(checkpoints_path, keep):
    """Delete all but the `keep` most recent checkpoints.

    Called after the new checkpoint is written, never before, so an interrupted prune leaves
    more checkpoints than asked for rather than none. Ordering is by step number rather than
    mtime, so a resumed run that rewrites an existing step does not reorder history.
    """
    if keep is None:
        return
    if keep < 1:
        raise ValueError(f"keep_checkpoints must be >= 1, got {keep}")

    steps = checkpoint_steps_on_disk(checkpoints_path)
    for step in steps[:-keep]:
        os.remove(f"{checkpoints_path}/{step}.pt")
    if len(steps) > keep:
        logging.info(
            f"Pruned {len(steps) - keep} checkpoint(s), keeping steps {steps[-keep:]}"
        )


def save_checkpoint(
    model, optimiser, lr_scheduler, scaler, run_path, step_no, results, keep_checkpoints=None
):
    checkpoints_path = f"{run_path}/checkpoints"
    os.makedirs(checkpoints_path, exist_ok=True)

    unwrapped = model._orig_mod if hasattr(model, "_orig_mod") else model
    checkpoint = {
        "model_state_dict": unwrapped.state_dict(),
        "optimizer_state_dict": optimiser.state_dict(),
        "scheduler_state_dict": lr_scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "step_no": step_no,
    }

    torch.save(checkpoint, f"{checkpoints_path}/{step_no}.pt")
    with open(f"{run_path}/results.json", "w") as f:
        json.dump(results, f)
    logging.info(f"Checkpoint saved at step {step_no}")
    _prune_checkpoints(checkpoints_path, keep_checkpoints)
    return None


def load_checkpoint(run_path, step_no, device=None):
    checkpoints_path = f"{run_path}/checkpoints"
    # Load checkpoint with device mapping if device is specified
    if device is not None:
        checkpoint = torch.load(f"{checkpoints_path}/{step_no}.pt", map_location=device)
    else:
        checkpoint = torch.load(f"{checkpoints_path}/{step_no}.pt")

    logging.info(f"Checkpoint loaded from step {step_no}")
    return checkpoint


def create_training_objects(model, train_config, tokenizer):

    device = torch.device(train_config["device"])
    criterion = nn.CrossEntropyLoss(
        reduction="mean",
        label_smoothing=train_config["label_smoothing"],
        ignore_index=tokenizer.pad_token_id,
    ).to(device)

    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=train_config["learning_rate"],
        betas=train_config["adam_betas"],
        eps=train_config["adam_eps"],
        weight_decay=0.01,
    )

    lr_scheduler = WarmupInverseSquareRootLR(optimiser, train_config["warm_up_steps"])

    scaler = amp.GradScaler()

    return criterion, optimiser, lr_scheduler, scaler


def create_run(model, train_config, tokenizer):
    results = []

    device = torch.device(train_config["device"])
    model.to(device)

    criterion, optimiser, lr_scheduler, scaler = create_training_objects(
        model, train_config, tokenizer
    )

    return {
        "model": model,
        "criterion": criterion,
        "optimiser": optimiser,
        "lr_scheduler": lr_scheduler,
        "scaler": scaler,
        "results": results,
        "step_no": 0,
    }


def load_run(run_path, model, train_config, tokenizer):
    steps = checkpoint_steps_on_disk(f"{run_path}/checkpoints")
    if not steps:
        raise FileNotFoundError(f"No checkpoints found in {run_path}")

    checkpoint_latest_step = steps[-1]

    device = torch.device(train_config["device"])
    model.to(device)

    checkpoint = load_checkpoint(run_path, checkpoint_latest_step, device)

    criterion, optimiser, lr_scheduler, scaler = create_training_objects(
        model, train_config, tokenizer
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    optimiser.load_state_dict(checkpoint["optimizer_state_dict"])
    lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    scaler.load_state_dict(checkpoint["scaler_state_dict"])
    step_no = checkpoint["step_no"]
    results_path = f"{run_path}/results.json"
    results = json.load(open(results_path)) if os.path.exists(results_path) else []

    return {
        "model": model,
        "criterion": criterion,
        "optimiser": optimiser,
        "lr_scheduler": lr_scheduler,
        "scaler": scaler,
        "results": results,
        "step_no": step_no,
    }


def load_model_weights(model, from_path, device, step=None):
    """Start a stage from another stage's weights: parameters only, nothing else.

    Not a resume. The optimiser moments are estimates fitted to the previous stage's loss
    surface, the scheduler position would hand SFT a decayed learning rate with no warmup,
    and the step counter would report the run already finished. Only the parameters carry
    over; everything else is built fresh for the new objective.

    `step` pins which checkpoint to take. Leaving it None takes the latest, which is
    convenient but means the stage config no longer records what actually loaded — prefer
    pinning it so the saved config is an honest account of the run's ancestry.
    """
    steps = checkpoint_steps_on_disk(f"{from_path}/checkpoints")
    if not steps:
        raise FileNotFoundError(
            f"No checkpoints in {from_path} to initialise from — run that stage first."
        )
    if step is None:
        step = steps[-1]
        logging.warning(
            f"init_from_step not set; taking the latest checkpoint ({step}) from {from_path}. "
            "Pin it in the config so this run's ancestry is recorded."
        )
    elif step not in steps:
        raise FileNotFoundError(
            f"init_from_step {step} not found in {from_path}/checkpoints (have {steps})."
        )

    model.to(device)
    checkpoint = load_checkpoint(from_path, step, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    logging.info(f"Initialised model weights from {from_path} step {step}")
    return step


def get_run(run_path, model, train_config, tokenizer, init_from=None):
    if checkpoint_steps_on_disk(f"{run_path}/checkpoints"):
        # This stage has its own history, so it resumes from that and ignores init_from
        # entirely — the earlier stage only ever seeds the very first run.
        run = load_run(run_path, model, train_config, tokenizer)
    else:
        if init_from is not None:
            load_model_weights(
                model,
                init_from,
                torch.device(train_config["device"]),
                train_config.get("init_from_step"),
            )
        run = create_run(model, train_config, tokenizer)

    run["run_path"] = run_path
    return run


def train(stage, model, dataloaders, tokenizer, config, init_from=None):
    """`init_from` is another stage's directory, whose weights seed this one's first run."""

    train_config = config["train"][stage]
    run_path = utils.get_stage_path(config, stage)
    run = get_run(run_path, model, train_config, tokenizer, init_from)

    # A zero step budget trains nothing but still writes step 0, so a later stage has
    # weights to initialise from and needs no special case of its own. That makes "build
    # the tokenizer but do not pretrain" a config choice rather than a separate code path,
    # and it is checked before the already-complete branch below, which a zero budget
    # would otherwise satisfy immediately and return without saving anything.
    if train_config["num_steps"] == 0:
        if checkpoint_steps_on_disk(f"{run_path}/checkpoints"):
            logging.info(f"{stage}: num_steps is 0 and a checkpoint already exists")
            return None
        save_checkpoint(
            model, run["optimiser"], run["lr_scheduler"], run["scaler"],
            run_path, 0, run["results"], train_config.get("keep_checkpoints"),
        )
        logging.info(f"{stage}: num_steps is 0 — saved the untrained model as step 0")
        return None

    if run["step_no"] >= train_config["num_steps"]:
        logging.info("Training already complete.")
        return None

    if train_config.get("compile_model", False):
        model = torch.compile(model, fullgraph=True)

    train_loop(stage, model, dataloaders, tokenizer, run, config)

    return None
