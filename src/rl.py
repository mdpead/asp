import torch
import torch.nn.functional as F
from torch import amp
import logging
import time

from src import train, generation, synth
from src.dataloader import resume_at_minibatch

# Which checker scores a rollout, keyed the way data.get_dataset_rl labels the row. Mirrors
# data._TASK_FORMATTERS, which built the prompt from the same record: the task decides both
# what was asked and what counts as having answered it.
_TASK_CHECKERS = {
    "output": synth.check_output_answer,
    "input": synth.check_input_answer,
}


def score(rows, completions):
    """Reward each rollout 1.0 if its checker accepts the answer, else 0.0."""
    return [float(_TASK_CHECKERS[row["task"]](row, c)) for row, c in zip(rows, completions)]


def validation_step(
    model,
    dataloader,
    tokenizer,
    device,
    step_no,
    max_new_tokens,
    max_batches,
    samples_per_prompt,
    temperature,
    seed,
):
    """Score the test prompts two ways: sampled reward and greedy pass@1.

    `sampled_reward` is the fraction of `samples_per_prompt` rollouts per prompt, drawn at
    the training temperature, that the checker accepts. It is the quantity the policy
    gradient actually raises, so it is the one that shows whether RL is working: an update
    that lifts a right answer from 0.2 to 0.35 is real progress, but greedy decoding still
    picks a wrong one held at 0.4, and pass@1 does not move until the right one overtakes.

    `pass_rate` is greedy pass@1, the model's single best guess — the number that matters
    in the end, and one that lags the sampled reward.

    Sampling runs under a fixed seed, restored afterwards so training's own draws are left
    alone. Every validation then sees the same random stream on the same prompts, so a
    change between steps is the weights moving rather than a luckier draw.
    """
    sampled, passed, finished_count, total = 0.0, 0.0, 0, 0
    with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
        torch.manual_seed(seed)
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= max_batches:
                break

            # Sampled: each prompt repeated, as training builds its groups
            rows = [row for row in batch for _ in range(samples_per_prompt)]
            token_ids, _, _, _, _ = generation.generate(
                model,
                tokenizer,
                [row["prompt"] for row in rows],
                device,
                max_new_tokens,
                temperature,
            )
            sampled += sum(score(rows, generation.decode_rollouts(tokenizer, token_ids)))

            # Greedy
            texts, finished = generation.generate_texts(
                model, tokenizer, [row["prompt"] for row in batch], device, max_new_tokens
            )
            passed += sum(score(batch, texts))
            finished_count += sum(finished)
            total += len(batch)

    return {
        "type": "validation",
        "step_no": step_no,
        "sampled_reward": sampled / max(total * samples_per_prompt, 1),
        "pass_rate": passed / max(total, 1),
        "finished_rate": finished_count / max(total, 1),
        "num_prompts": total,
    }


def token_budget_chunks(starts, ends, budget):
    """Split rows into chunks whose padded size, rows x width, fits `budget` tokens.

    `starts` and `ends` are each row's first real column and one past its last, so a
    chunk's width is the span from its earliest start to its latest end — what remains
    once the columns that are padding for every row in it are trimmed.

    A rollout's length is not known until it has been generated, so this cannot live in
    the sampler the way SFT's token budget does. One completion that runs to its limit
    pads its whole minibatch out to that length, and the activations held for backward
    scale with exactly that padded size.

    Rows are taken in order of `ends`, so short rollouts share a chunk and are not padded
    out to a long one. A row wider than the budget on its own still gets a chunk: it
    cannot be split, and dropping it would bias the update towards short completions.
    `budget` None means one chunk.
    """
    order = sorted(range(len(ends)), key=ends.__getitem__)
    if budget is None:
        return [order]

    chunks, current = [], []
    for row in order:
        candidate = current + [row]
        width = max(ends[i] for i in candidate) - min(starts[i] for i in candidate)
        if current and len(candidate) * width > budget:
            chunks.append(current)
            current = [row]
        else:
            current = candidate
    chunks.append(current)
    return chunks


def policy_gradient_backward(
    model,
    tokenizer,
    scaler,
    token_ids,
    completion_mask,
    old_logprobs,
    seq_starts,
    advantages,
    blocked,
    temperature,
    router_aux_loss_coef,
    loss_scale,
    rescore_token_size=None,
):
    """Rescore rollouts with gradients on and backpropagate the policy-gradient loss.

    Per rollout the loss is -(advantage x mean logprob of its own tokens), plus the
    router's load-balancing term. What is backpropagated is the sum over the rollouts given,
    times `loss_scale`, so the caller decides what the gradient is averaged over — it can
    only know how many rollouts a step trained on once the whole step has been generated.
    The sum is accumulated over chunks that each fit `rescore_token_size`, which changes
    the peak memory and not the gradient. Advantages arrive already computed, so a group
    can be split across chunks freely.

    Returns (loss_pg_sum, loss_aux_sum, logprob_gap_sum, completion_tokens) as floats: the
    two losses summed over rollouts and unscaled, the gap summed over completion tokens
    against the logprobs recorded at sampling.
    """
    device = token_ids.device
    length = token_ids.shape[1]

    # One past each row's last real token. Right padding only follows a finished row, and
    # sampling cannot emit <pad>, so the trailing pads are exactly the filler.
    is_pad = token_ids == tokenizer.pad_token_id
    ends = (length - is_pad.flip(1).int().cumprod(1).sum(1)).tolist()
    starts = seq_starts.tolist()

    loss_pg_total, loss_aux_total, gap_total, tokens_total = 0.0, 0.0, 0.0, 0
    for chunk in token_budget_chunks(starts, ends, rescore_token_size):
        left = min(starts[i] for i in chunk)
        right = max(ends[i] for i in chunk)
        rows = torch.tensor(chunk, device=device)

        ids = token_ids[rows, left:right]
        chosen = completion_mask[rows, left:right]
        chunk_starts = seq_starts[rows] - left

        # seq_starts gives the left-padded rows the same context generation saw;
        # padding_mask keeps pad out of the MoE's expert seats. Position t predicts token
        # t+1, so the positions worth projecting are the ones whose next token the model
        # chose. output_mask restricts the vocabulary projection to those: the prompt and
        # padding columns of a (rows, seq, vocab) logits tensor are never read.
        predicts_completion = F.pad(chosen[:, 1:], (0, 1), value=False)
        with amp.autocast(device_type=device.type):
            logits, loss_aux = model(
                ids,
                seq_starts=chunk_starts,
                padding_mask=ids != tokenizer.pad_token_id,
                output_mask=predicts_completion,
            )  # (completion tokens, vocab), row-major

        # fp32, for the reason generate samples in it. Divided by the temperature because
        # generate was: the rewards describe the policy as it was sampled, so that is the
        # distribution whose logprobs the gradient needs. Scoring the untempered one against
        # tempered samples is not the gradient of the expected reward, and it reads as a
        # logprob gap that is not a bug. The ids sampling blocks are removed here too, for
        # the same reason: with them left in, the rescore spreads probability over tokens
        # the sampler could not draw.
        step_logits = logits.float() / temperature
        step_logits[:, blocked] = -float("inf")
        # Boolean indexing is row-major on both sides, so these line up with the logits
        targets = ids[:, 1:][chosen[:, 1:]]
        new_logprobs = -F.cross_entropy(step_logits, targets, reduction="none")  # (tokens,)

        # Mean over each rollout's own tokens, so a long completion does not outweigh a
        # short one with the same advantage. Negated: minimising this raises the logprob of
        # rollouts that beat their group. Summed, not averaged: see the docstring.
        row_of_token = predicts_completion.nonzero()[:, 0]
        tokens_per_row = predicts_completion.sum(-1)
        per_seq = torch.zeros(len(chunk), device=device).index_add_(0, row_of_token, new_logprobs)
        per_seq = per_seq / tokens_per_row.clamp(min=1)
        loss_pg = -(per_seq * advantages[rows]).sum()
        # The aux term is a per-pass mean, so it is weighted by the rows that pass held
        loss = loss_pg + loss_aux * len(chunk) * router_aux_loss_coef

        scaler.scale(loss * loss_scale).backward()

        with torch.no_grad():
            old = old_logprobs[rows, left:right][:, 1:][chosen[:, 1:]]
            gap_total += (new_logprobs - old).abs().sum().item()
        loss_pg_total += loss_pg.item()
        loss_aux_total += loss_aux.item() * len(chunk)
        tokens_total += int(tokens_per_row.sum().item())

    return loss_pg_total, loss_aux_total, gap_total, tokens_total


def train_loop(stage, model, dataloaders, tokenizer, run, config):

    train_config = config["train"][stage]
    device = torch.device(train_config["device"])
    grad_accum_steps = (
        train_config["effective_prompts_size"] // train_config["minibatch_prompts_size"]
    )
    num_steps = train_config["num_steps"]
    checkpoint_steps = train_config["checkpoint_steps"]
    keep_checkpoints = train_config.get("keep_checkpoints")
    validation_steps = train_config["validation_steps"]
    validation_batches = train_config["validation_batches"]
    cache_clear_steps = train_config.get("cache_clear_steps")
    router_aux_loss_coef = train_config["router_aux_loss_coef"]
    rollouts_per_group = train_config["rollouts_per_group"]
    max_new_tokens = train_config["max_new_tokens"]
    temperature = train_config["temperature"]
    rescore_token_size = train_config.get("rescore_token_size")

    optimiser = run["optimiser"]
    lr_scheduler = run["lr_scheduler"]
    scaler = run["scaler"]
    results = run["results"]
    step_no = run["step_no"]
    run_path = run["run_path"]

    blocked = generation.blocked_token_ids(tokenizer, device)
    # Rollouts a step generates. Gradients are accumulated against this and corrected to
    # the number actually trained on once the step is complete, which keeps the loss going
    # into backward at the scale the GradScaler has adapted to.
    nominal_rollouts = train_config["effective_prompts_size"] * rollouts_per_group

    def validate():
        result = validation_step(
            model,
            dataloaders["test"],
            tokenizer,
            device,
            step_no,
            max_new_tokens,
            # In validation batches, which have their own size
            validation_batches,
            train_config.get("validation_samples", rollouts_per_group),
            temperature,
            config["seed"],
        )
        logging.info(result)
        results.append(result)
        model.train()

    # The untrained starting point, so a run's own log holds what its later validations
    # are compared against. Only on a fresh start: a resumed run already recorded it.
    if step_no == 0 and not any(r["type"] == "validation" for r in results):
        validate()

    start_time = time.time()
    total_loss_pg = 0.0
    total_loss_aux = 0.0
    total_reward = 0.0
    total_finished = 0.0
    total_mixed_groups = 0.0
    total_logprob_gap = 0.0
    total_completion_tokens = 0
    total_rescored_tokens = 0
    trained_groups = 0
    resume_at_minibatch(dataloaders["train"], step_no * grad_accum_steps)
    optimiser.zero_grad(set_to_none=True)

    model.train()
    for accum_idx, batch in enumerate(dataloaders["train"]):

        # One row per rollout, each prompt's group contiguous so the reshape below lines
        # every group up with the prompt that produced it. Interleaving instead would take
        # each group statistic across different prompts, with no error to show for it.
        rows = [row for row in batch for _ in range(rollouts_per_group)]
        prompts = [row["prompt"] for row in rows]

        # Sample the rollouts. No gradient comes back from here: generate runs under
        # no_grad, so its logprobs are a record of the sampling policy, not something to
        # differentiate.
        token_ids, completion_mask, old_logprobs, seq_starts, finished = generation.generate(
            model, tokenizer, prompts, device, max_new_tokens, temperature
        )

        # The whole row travels with the rollout rather than just its answer: scoring means
        # executing the record that produced the prompt, and the checkers read result,
        # source and args straight off it.
        completions = generation.decode_rollouts(tokenizer, token_ids)
        rewards = torch.tensor(score(rows, completions), dtype=torch.float32, device=device)
        rewards = rewards.reshape(len(batch), rollouts_per_group)

        # Group-relative advantage
        mean = rewards.mean(dim=-1, keepdim=True)
        sd = rewards.std(dim=-1, keepdim=True)
        advantages = ((rewards - mean) / (sd + 1e-8)).reshape(-1)  # (prompts * group,)

        # A group whose rollouts all scored the same has zero advantage everywhere, so its
        # policy gradient is exactly zero. Those rollouts are dropped here rather than
        # rescored: the forward and backward they would cost buys nothing, and counting
        # them in the average shrinks the gradient by however many there happened to be.
        mixed = sd.squeeze(-1) > 0  # (prompts,)
        keep = mixed.repeat_interleave(rollouts_per_group).nonzero().squeeze(-1)

        # Rescore with gradients on and backpropagate, in chunks that fit the token budget
        if len(keep) > 0:
            loss_pg, loss_aux, gap, rescored_tokens = policy_gradient_backward(
                model,
                tokenizer,
                scaler,
                token_ids[keep],
                completion_mask[keep],
                old_logprobs[keep],
                seq_starts[keep],
                advantages[keep],
                blocked,
                temperature,
                router_aux_loss_coef,
                1 / nominal_rollouts,
                rescore_token_size,
            )
            # The logprob gap is the rescore against the sampling pass on the same tokens:
            # it should sit near zero, and drifting from it means generation and training
            # no longer compute the same policy (padding, routing, precision).
            total_loss_pg += loss_pg
            total_loss_aux += loss_aux
            total_logprob_gap += gap
            total_rescored_tokens += rescored_tokens
            trained_groups += int(mixed.sum().item())

        # Over every group, dropped or not: these describe what the policy produced
        total_reward += rewards.mean().item() / grad_accum_steps
        total_finished += finished.float().mean().item() / grad_accum_steps
        total_mixed_groups += mixed.float().mean().item() / grad_accum_steps
        total_completion_tokens += int(completion_mask.sum().item())

        # Gradient accumulation
        if (accum_idx + 1) % grad_accum_steps != 0:
            continue

        # Step optimiser and scheduler. The gradients were accumulated as a sum over the
        # rollouts trained on, scaled by the nominal count; this turns that into their mean.
        # A step where no group was mixed has nothing to apply and leaves the weights alone,
        # but still counts, so the step number keeps tracking the sampler's position.
        trained_rollouts = trained_groups * rollouts_per_group
        if trained_rollouts > 0:
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.mul_(nominal_rollouts / trained_rollouts)
            scaler.unscale_(optimiser)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).item()
            scaler.step(optimiser)
            scaler.update()
        else:
            grad_norm = 0.0
        lr_scheduler.step()
        optimiser.zero_grad(set_to_none=True)

        step_no += 1

        elapsed_time = time.time() - start_time
        result = {}
        result["type"] = "train"
        result["step_no"] = step_no
        result["num_completion_tokens"] = total_completion_tokens
        result["tokens_per_sec"] = total_completion_tokens / elapsed_time
        result["learning_rate"] = lr_scheduler.get_last_lr()[0]
        # Losses and the gap are over the rollouts trained on; the rest over all of them
        result["loss_pg"] = total_loss_pg / max(trained_rollouts, 1)
        result["loss_aux"] = total_loss_aux / max(trained_rollouts, 1)
        result["reward"] = total_reward
        result["finished_rate"] = total_finished
        result["mixed_group_rate"] = total_mixed_groups
        result["trained_groups"] = trained_groups
        result["logprob_gap"] = total_logprob_gap / max(total_rescored_tokens, 1)
        result["grad_norm"] = grad_norm
        results.append(result)
        logging.info(result)

        # Validation step
        if step_no % validation_steps == 0:
            validate()

        # Checkpointing
        if step_no % checkpoint_steps == 0:
            train.save_checkpoint(
                model, optimiser, lr_scheduler, scaler, run_path, step_no, results, keep_checkpoints
            )

        # Delete tensors to free up memory
        del token_ids, completion_mask, old_logprobs
        if cache_clear_steps and step_no % cache_clear_steps == 0:
            torch.cuda.empty_cache()

        # Reset counters
        start_time = time.time()
        total_loss_pg = 0.0
        total_loss_aux = 0.0
        total_reward = 0.0
        total_finished = 0.0
        total_mixed_groups = 0.0
        total_logprob_gap = 0.0
        total_completion_tokens = 0
        total_rescored_tokens = 0
        trained_groups = 0

        # Stop after num_steps
        if step_no >= num_steps:
            break

    # A final save when num_steps is not a multiple of checkpoint_steps, as in train.train_loop
    if step_no % checkpoint_steps != 0:
        train.save_checkpoint(
            model, optimiser, lr_scheduler, scaler, run_path, step_no, results, keep_checkpoints
        )

    return None


def train_rl(stage, model, dataloaders, tokenizer, config, init_from=None):
    """Run the RL stage: train.train's bookkeeping, this module's loop.

    Kept as a wrapper rather than folded into train.train so that module stays free of RL
    specifics, and rather than moved into it so scripts/rl.py reads like scripts/sft.py.
    """
    return train.train(stage, model, dataloaders, tokenizer, config, init_from, loop=train_loop)
