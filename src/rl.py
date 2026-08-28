import torch
import logging

from src import train, generation, synth

# Which checker scores a rollout, keyed the way data.get_dataset_rl labels the row. Mirrors
# data._TASK_FORMATTERS, which built the prompt from the same record: the task decides both
# what was asked and what counts as having answered it.
_TASK_CHECKERS = {
    "output": synth.check_output_answer,
    "input": synth.check_input_answer,
}


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
    max_length = config["model"]["max_length"]
    cache_clear_steps = train_config.get("cache_clear_steps")
    router_aux_loss_coef = train_config["router_aux_loss_coef"]
    rollouts_per_group = train_config["rollouts_per_group"]
    max_new_tokens = train_config["max_new_tokens"]
    temperature = train_config["temperature"]

    optimiser = run["optimiser"]
    lr_scheduler = run["lr_scheduler"]
    scaler = run["scaler"]
    results = run["results"]
    step_no = run["step_no"]
    run_path = run["run_path"]

    for batch in dataloaders["train"]:

        # One row per rollout, each prompt's group contiguous so the reshape below lines
        # every group up with the prompt that produced it. Interleaving instead would take
        # each group statistic across different prompts, with no error to show for it.
        rows = [row for row in batch for _ in range(rollouts_per_group)]
        prompts = [row["prompt"] for row in rows]

        # Generate the token sequence
        token_ids, completion_mask, logprobs, seq_starts, finished = generation.generate(
            model, tokenizer, prompts, device, max_new_tokens, temperature
        )

        # The whole row travels with the rollout rather than just its answer: scoring means
        # executing the record that produced the prompt, and the checkers read result,
        # source and args straight off it.
        completions = generation.decode_rollouts(tokenizer, token_ids)
        rewards = torch.tensor(
            [
                _TASK_CHECKERS[row["task"]](row, completion)
                for row, completion in zip(rows, completions)
            ],
            dtype=torch.float32,
            device=device,
        )
        rewards = rewards.reshape(len(batch), rollouts_per_group)

        # Calculate per group statistics
        mean = rewards.mean(dim=-1, keepdim=True)
        sd = rewards.std(dim=-1, keepdim=True)
        advantages = (rewards - mean) / (sd + 1e-8)  # (prompts, groups)
        advantages = advantages.reshape(-1)  # (prompts * groups,)

        # Calculate loss
        advantages_per_token = advantages[:, None]  # (prompt * group, seq)
        loss_per_token = logprobs * advantages_per_token * completion_mask  # (prompt * group, seq)
        loss = loss_per_token.sum()

    return None


def train_rl(stage, model, dataloaders, tokenizer, config, init_from=None):
    """Run the RL stage: train.train's bookkeeping, this module's loop.

    Kept as a wrapper rather than folded into train.train so that module stays free of RL
    specifics, and rather than moved into it so scripts/rl.py reads like scripts/sft.py.
    """
    return train.train(stage, model, dataloaders, tokenizer, config, init_from, loop=train_loop)
