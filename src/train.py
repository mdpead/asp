import torch
from torch import nn
from torch.optim.lr_scheduler import LRScheduler
import logging
import time
from torch.optim import Optimizer
from torch import amp
import sacrebleu
from src import generation
import os
import itertools
import json
from src import utils

MODEL_INPUTS = [
    "src_input_ids",
    "tgt_input_ids",
    "src_padding_mask",
    "tgt_padding_mask",
    "tgt_output_ids",
]


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


def generate_metrics(model, batch, max_length, tokenizer, device):

    pred_text = generation.generate_texts(
        model, tokenizer, input_texts=batch["src_text"], max_length=max_length, device=device
    )

    sacrebleu_score = sacrebleu.corpus_bleu(
        pred_text,
        [batch["tgt_text"]],
        smooth_method="exp",
        smooth_value=0.0,
        lowercase=False,
        tokenize="intl",
    )

    metrics = {
        "bleu": sacrebleu_score.score,
    }

    return metrics


def validation_step(
    model, dataloader, criterion, device, tokenizer, step_no, max_length, validation_minibatches
):

    model.eval()
    start_time = time.time()

    total_loss = 0.0
    total_tokens = 0
    num_batches = 0
    bleu_batch = None

    for minibatch in itertools.islice(dataloader, validation_minibatches):

        minibatch = {
            k: v.to(device, non_blocking=True) if k in MODEL_INPUTS else v
            for k, v in minibatch.items()
        }

        with torch.no_grad():
            logits = model(
                minibatch["src_input_ids"],
                minibatch["tgt_input_ids"],
                minibatch["src_padding_mask"],
                minibatch["tgt_padding_mask"],
            )
            loss = criterion(
                logits.reshape(-1, logits.shape[2]), minibatch["tgt_output_ids"].reshape(-1)
            )

        total_loss += loss.item()
        total_tokens += minibatch["src_input_ids"].ne(tokenizer.pad_token_id).sum().item()
        num_batches += 1
        if bleu_batch is None:
            bleu_batch = minibatch

    elapsed_time = time.time() - start_time
    validation_metrics = generate_metrics(model, bleu_batch, max_length, tokenizer, device)

    return {
        "type": "validation",
        "step_no": step_no,
        "num_tokens": total_tokens,
        "tokens_per_sec": total_tokens / elapsed_time,
        "loss": total_loss / num_batches,
        "validation_metrics": validation_metrics,
    }


def train_loop(stage, model, dataloaders, tokenizer, run, config):

    train_config = config["train"][stage]
    device = torch.device(train_config["device"])
    grad_accum_steps = train_config["effective_batch_token_size"] // train_config["minibatch_token_size"]
    num_steps = train_config["num_steps"]
    checkpoint_steps = train_config["checkpoint_steps"]
    validation_steps = train_config["validation_steps"]
    validation_batches = train_config["validation_batches"]
    max_length = config["model"]["max_length"]
    cache_clear_steps = train_config.get("cache_clear_steps")

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
    total_loss = 0.0
    minibatch_idx = step_no * grad_accum_steps
    train_dataloader = itertools.islice(dataloaders["train"], minibatch_idx, None)
    optimiser.zero_grad(set_to_none=True)

    model.train()
    for minibatch_idx, batch in enumerate(train_dataloader, start=minibatch_idx):

        # Move batch to device
        batch = {
            k: v.to(device, non_blocking=True) if k in MODEL_INPUTS else v for k, v in batch.items()
        }

        with amp.autocast(device_type=device.type):

            # Forward pass
            logits = model(
                batch["src_input_ids"],
                batch["tgt_input_ids"],
                batch["src_padding_mask"],
                batch["tgt_padding_mask"],
            )
            loss = criterion(
                logits.reshape(-1, logits.shape[2]), batch["tgt_output_ids"].reshape(-1)
            )

        # Compute loss and gradients
        minibatch_tokens = batch["src_input_ids"].ne(tokenizer.pad_token_id).sum().item()
        batch_tokens += minibatch_tokens

        scaled_loss = loss / grad_accum_steps
        scaler.scale(scaled_loss).backward()
        total_loss += scaled_loss.item()

        # Gradient accumulation
        if (minibatch_idx + 1) % grad_accum_steps != 0:
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
        result["loss"] = total_loss
        result["token_length"] = batch["src_input_ids"].shape[1]
        result["grad_norm"] = grad_norm.item()
        results.append(result)
        logging.info(result)

        # Reset counters
        batch_tokens = 0
        start_time = time.time()
        total_loss = 0.0

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
            save_checkpoint(model, optimiser, lr_scheduler, scaler, run_path, step_no, results)

        # Delete tensors to free up memory
        del batch, logits, loss, scaled_loss
        if cache_clear_steps and step_no % cache_clear_steps == 0:
            torch.cuda.empty_cache()

        # Stop after num_steps
        if step_no >= num_steps:
            break

    return None


def save_checkpoint(model, optimiser, lr_scheduler, scaler, run_path, step_no, results):
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


def create_training_objects(model, config):

    # Define loss function, optimiser, and scheduler
    device = torch.device(config["train"]["device"])
    criterion = nn.CrossEntropyLoss(
        reduction="mean",
        label_smoothing=config["train"]["label_smoothing"],
        ignore_index=config["tokenizer"]["pad_token_id"],
    ).to(device)

    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=config["train"]["learning_rate"],
        betas=config["train"]["adam_betas"],
        eps=config["train"]["adam_eps"],
        weight_decay=0.01,
    )

    lr_scheduler = WarmupInverseSquareRootLR(optimiser, config["train"]["warm_up_steps"])

    scaler = amp.GradScaler()

    return criterion, optimiser, lr_scheduler, scaler


def create_run(model, config):
    results = []

    device = torch.device(config["train"]["device"])
    model.to(device)

    criterion, optimiser, lr_scheduler, scaler = create_training_objects(model, config)

    return {
        "model": model,
        "criterion": criterion,
        "optimiser": optimiser,
        "lr_scheduler": lr_scheduler,
        "scaler": scaler,
        "results": results,
        "step_no": 0,
    }


def load_run(run_path, model, config):
    checkpoints = [f for f in os.listdir(run_path + "/checkpoints") if f.endswith(".pt")]
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {run_path}")

    checkpoint_latest_step = max([int(checkpoint.split(".")[0]) for checkpoint in checkpoints])

    device = torch.device(config["train"]["device"])
    model.to(device)

    checkpoint = load_checkpoint(run_path, checkpoint_latest_step, device)

    criterion, optimiser, lr_scheduler, scaler = create_training_objects(model, config)

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


def get_run(config, model):
    run_path = utils.get_run_path(config)
    checkpoints_path = f"{run_path}/checkpoints"
    has_checkpoint = os.path.isdir(checkpoints_path) and any(
        f.endswith(".pt") for f in os.listdir(checkpoints_path)
    )

    if has_checkpoint:
        run = load_run(run_path, model, config)
    else:
        run = create_run(model, config)

    run["run_path"] = run_path
    return run


def train(stage, model, dataloaders, tokenizer, config):

    run = get_run(config, model)
    train_config = config["train"][stage]

    if run["step_no"] >= train_config["num_steps"]:
        logging.info("Training already complete.")
        return run["results"]

    if train_config.get("compile_model", False):
        model = torch.compile(model, fullgraph=True)

    train_loop(stage, model, dataloaders, tokenizer, run, config)

    return None
