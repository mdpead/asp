import os
import json
import logging
import shutil
import random
import numpy as np
import torch


def get_run_path(config):
    return f"{config['locations']['models_dir']}/{config['name']}"


# Keys that can change without invalidating a checkpoint. Sections are keyed by
# their top-level name, but the keys themselves live one level down under a stage
# (train.pretrain.num_steps), so stripping has to descend into each stage.
RESUMABLE_KEYS = {
    "train": {
        "num_steps",
        "checkpoint_steps",
        "validation_steps",
        "validation_batches",
        "minibatch_token_size",
        "cache_clear_steps",
        "num_workers",
        "prefetch_factor",
        "compile_model",
    }
}


def _strip_resumable(config):
    stripped = dict(config)
    for section, keys in RESUMABLE_KEYS.items():
        section_config = stripped.get(section)
        if not isinstance(section_config, dict):
            continue
        stripped[section] = {
            stage: {k: v for k, v in stage_config.items() if k not in keys}
            if isinstance(stage_config, dict)
            else stage_config
            for stage, stage_config in section_config.items()
        }
    return stripped


def _config_requires_restart(existing, current):
    return _strip_resumable(existing) != _strip_resumable(current)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_run(config):
    set_seed(config["seed"])
    run_path = get_run_path(config)
    os.makedirs(run_path, exist_ok=True)
    config_path = f"{run_path}/config.json"
    if os.path.exists(config_path):
        existing = load_config(run_path)
        if _config_requires_restart(existing, config):
            logging.warning("Config differs from saved config — clearing run directory.")
            shutil.rmtree(run_path)
            os.makedirs(run_path)
    save_config(run_path, config)
    return run_path


def save_config(run_path, config):
    filepath = f"{run_path}/config.json"
    json.dump(config, open(filepath, "w"))
    return filepath


def load_config(run_path):
    filepath = f"{run_path}/config.json"
    with open(filepath) as f:
        config = json.load(f)
    return config
