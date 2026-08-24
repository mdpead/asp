"""Starting a stage from an earlier stage's weights."""

import copy

import pytest
import torch

from src import train, utils

DEV = "cuda"


@pytest.fixture
def staged(tmp_path, make_model, tokenizer):
    """A pretrain stage with two checkpoints, and a config pointing an sft stage at it."""
    model = make_model(len(tokenizer)).float()
    cfg = {
        "name": "t", "seed": 0,
        "locations": {"models_dir": str(tmp_path)},
        "train": {"pretrain": {}, "sft": {
            "device": DEV, "learning_rate": 1e-4, "adam_betas": [0.9, 0.95],
            "adam_eps": 1e-8, "label_smoothing": 0.0, "warm_up_steps": 5,
        }},
    }
    pre = utils.get_stage_path(cfg, "pretrain")
    tc = cfg["train"]["sft"]
    _, opt, sched, scaler = train.create_training_objects(model, tc, tokenizer)
    for step in (100, 200):
        with torch.no_grad():                       # make the two checkpoints differ
            next(model.parameters()).add_(float(step))
        train.save_checkpoint(model, opt, sched, scaler, pre, step, [])
    return cfg, pre, model


def _first_param(m):
    return next(m.parameters()).detach().float().clone()


def test_seeds_a_fresh_stage_from_the_earlier_one(staged, make_model, tokenizer):
    cfg, pre, source = staged
    fresh = make_model(len(tokenizer)).float()
    assert not torch.equal(_first_param(fresh), _first_param(source))

    run = train.get_run(utils.get_stage_path(cfg, "sft"), fresh,
                        {**cfg["train"]["sft"], "init_from_step": 200}, tokenizer,
                        init_from=pre)

    assert torch.equal(_first_param(fresh), _first_param(source)), "weights did not transfer"
    assert run["step_no"] == 0, "step counter must not come across"
    assert run["results"] == [], "history belongs to the source stage"
    # a fresh optimiser has no accumulated moments
    assert all(not st for st in run["optimiser"].state.values())


def test_init_from_step_selects_the_pinned_checkpoint(staged, make_model, tokenizer):
    cfg, pre, _ = staged
    a, b = make_model(len(tokenizer)).float(), make_model(len(tokenizer)).float()
    tc = cfg["train"]["sft"]
    train.get_run(utils.get_stage_path(cfg, "sft"), a, {**tc, "init_from_step": 100},
                  tokenizer, init_from=pre)
    train.get_run(utils.get_stage_path(cfg, "sft"), b, {**tc, "init_from_step": 200},
                  tokenizer, init_from=pre)
    assert not torch.equal(_first_param(a), _first_param(b))


def test_missing_pinned_step_is_an_error(staged, make_model, tokenizer):
    cfg, pre, _ = staged
    with pytest.raises(FileNotFoundError, match="init_from_step 999 not found"):
        train.get_run(utils.get_stage_path(cfg, "sft"), make_model(len(tokenizer)).float(),
                      {**cfg["train"]["sft"], "init_from_step": 999}, tokenizer, init_from=pre)


def test_missing_source_stage_is_an_error(tmp_path, make_model, tokenizer):
    cfg = {"name": "t", "seed": 0, "locations": {"models_dir": str(tmp_path)},
           "train": {"sft": {"device": DEV}}}
    with pytest.raises(FileNotFoundError, match="No checkpoints in"):
        train.load_model_weights(make_model(len(tokenizer)).float(),
                                 utils.get_stage_path(cfg, "pretrain"), torch.device(DEV))


def test_resume_ignores_the_source_stage(staged, make_model, tokenizer):
    """Once the stage has its own history it must resume from that, not re-seed."""
    cfg, pre, _ = staged
    sft = utils.get_stage_path(cfg, "sft")
    own = make_model(len(tokenizer)).float()
    with torch.no_grad():
        next(own.parameters()).fill_(-7.0)          # unmistakably not the pretrain weights
    tc = cfg["train"]["sft"]
    _, opt, sched, scaler = train.create_training_objects(own, tc, tokenizer)
    train.save_checkpoint(own, opt, sched, scaler, sft, 50, [])

    target = make_model(len(tokenizer)).float()
    run = train.get_run(sft, target, {**tc, "init_from_step": 200}, tokenizer, init_from=pre)

    assert torch.equal(_first_param(target), _first_param(own)), "re-seeded instead of resuming"
    assert run["step_no"] == 50, "resume must restore the step counter"


# --- final checkpoint on exit ---


def _loop_config(tmp_path, num_steps, checkpoint_steps):
    return {
        "name": "t", "seed": 0,
        "locations": {"models_dir": str(tmp_path)},
        "model": {"max_length": 64},
        "train": {"sft": {
            "device": DEV, "learning_rate": 1e-4, "adam_betas": [0.9, 0.95],
            "adam_eps": 1e-8, "label_smoothing": 0.0, "warm_up_steps": 2,
            "effective_batch_token_size": 256, "minibatch_token_size": 128,
            "num_steps": num_steps, "checkpoint_steps": checkpoint_steps,
            "keep_checkpoints": 10, "validation_steps": 10_000,
            "validation_batches": 1, "router_aux_loss_coef": 0.01,
            "num_workers": 0,
        }},
    }


def _run_loop(cfg, make_model, tokenizer):
    from src import dataloader as dl

    rows = [
        {"ids": [1] + [5 + (i % 7)] * (10 + i % 5) + [2], "prompt_len": 3,
         "length": 11 + i % 5}
        for i in range(24)
    ]
    model = make_model(len(tokenizer)).float()
    tc = cfg["train"]["sft"]
    loaders = dl.create_dataloaders_sft({"train": rows, "test": rows[:6]}, tokenizer, cfg)
    run = train.create_run(model, tc, tokenizer)
    run["run_path"] = utils.get_stage_path(cfg, "sft")
    train.train_loop("sft", model, loaders, tokenizer, run, cfg)
    return run["run_path"]


def test_final_step_is_checkpointed_when_it_is_not_a_multiple(tmp_path, make_model, tokenizer):
    """num_steps need not divide by checkpoint_steps. Without a save on exit the steps
    since the last checkpoint are lost, and so are their metrics — save_checkpoint is also
    what writes results.json."""
    cfg = _loop_config(tmp_path, num_steps=3, checkpoint_steps=2)
    run_path = _run_loop(cfg, make_model, tokenizer)

    assert train.checkpoint_steps_on_disk(f"{run_path}/checkpoints") == [2, 3]


def test_a_run_ending_on_a_boundary_is_not_checkpointed_twice(
    tmp_path, make_model, tokenizer, monkeypatch
):
    """The in-loop save already covers the final step; saving again rewrites the same file.

    Counting calls rather than reading the directory, because a duplicate save lands on the
    same "<step>.pt" filename — the checkpoints on disk look identical either way, and only
    the wasted write distinguishes them.
    """
    saved_steps = []
    real_save = train.save_checkpoint

    def spy(*args, **kwargs):
        saved_steps.append(args[5])
        return real_save(*args, **kwargs)

    monkeypatch.setattr(train, "save_checkpoint", spy)

    cfg = _loop_config(tmp_path, num_steps=4, checkpoint_steps=2)
    run_path = _run_loop(cfg, make_model, tokenizer)

    assert saved_steps == [2, 4]
    assert train.checkpoint_steps_on_disk(f"{run_path}/checkpoints") == [2, 4]


def test_final_checkpoint_carries_the_results_for_its_steps(tmp_path, make_model, tokenizer):
    """results.json is written by save_checkpoint, so an unsaved tail loses its metrics."""
    import json

    cfg = _loop_config(tmp_path, num_steps=3, checkpoint_steps=2)
    run_path = _run_loop(cfg, make_model, tokenizer)

    results = json.load(open(f"{run_path}/results.json"))
    assert [e["step_no"] for e in results if e["type"] == "train"] == [1, 2, 3]
