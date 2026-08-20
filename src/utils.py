import argparse
import os
import json
import logging
import shutil
import random
import numpy as np
import torch
import yaml


def get_run_path(config):
    """Run-level artifacts shared by every stage: the tokenizer."""
    return f"{config['locations']['models_dir']}/{config['name']}"


def get_stage_path(config, stage):
    """Per-stage artifacts: checkpoints, results, and that stage's config snapshot.

    One directory per stage, so SFT does not find pretrain's checkpoints and inherit its
    step counter and optimiser state, and so restarting one stage leaves the others alone.
    """
    return f"{get_run_path(config)}/{stage}"


def parse_config():
    """--config <name> -> the parsed configs/<name>.yaml. Shared by the stage scripts."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Config name (e.g. base_5070ti, test)")
    args = parser.parse_args()
    with open(f"configs/{args.config}.yaml") as f:
        return yaml.safe_load(f)


# Sections every stage's checkpoints depend on, whichever stage is running.
SHARED_SECTIONS = ("name", "seed", "model", "tokenizer")

# Keys that can change without invalidating a checkpoint.
RESUMABLE_KEYS = {
    "num_steps",
    "checkpoint_steps",
    "keep_checkpoints",
    "validation_steps",
    "validation_batches",
    "minibatch_token_size",
    "cache_clear_steps",
    "num_workers",
    "prefetch_factor",
    "compile_model",
}


def _stage_fingerprint(config, stage):
    """The part of a config that this stage's checkpoints actually depend on.

    Scoped to one stage so that editing data.sft restarts SFT and not pretraining, and
    stripped of the keys that can change mid-run — step budgets, worker counts — without
    invalidating a checkpoint.
    """
    fingerprint = {key: config.get(key) for key in SHARED_SECTIONS}
    train_config = config.get("train", {}).get(stage) or {}
    fingerprint["train"] = {k: v for k, v in train_config.items() if k not in RESUMABLE_KEYS}
    fingerprint["data"] = config.get("data", {}).get(stage)
    return fingerprint


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_run(config, stage):
    """Prepare this stage's directory, clearing it if the config it depends on moved."""
    set_seed(config["seed"])
    run_path = get_run_path(config)
    stage_path = get_stage_path(config, stage)
    os.makedirs(stage_path, exist_ok=True)

    # The tokenizer outlives any single stage's restart, and get_tokenizer loads it from disk
    # whenever the directory exists — so without its own check a changed vocab_size would be
    # silently ignored rather than retrained.
    if os.path.exists(f"{run_path}/config.json"):
        existing = load_config(run_path)
        if existing.get("tokenizer") != config.get("tokenizer"):
            logging.warning("Tokenizer config differs from saved config — clearing tokenizer.")
            shutil.rmtree(f"{run_path}/tokenizer", ignore_errors=True)
    save_config(run_path, config)

    if os.path.exists(f"{stage_path}/config.json"):
        existing = load_config(stage_path)
        if _stage_fingerprint(existing, stage) != _stage_fingerprint(config, stage):
            logging.warning(f"{stage} config differs from saved config — clearing {stage_path}.")
            shutil.rmtree(stage_path)
            os.makedirs(stage_path)
    save_config(stage_path, config)
    return stage_path


def save_config(run_path, config):
    filepath = f"{run_path}/config.json"
    json.dump(config, open(filepath, "w"))
    return filepath


def load_config(run_path):
    filepath = f"{run_path}/config.json"
    with open(filepath) as f:
        config = json.load(f)
    return config
