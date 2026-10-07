from src import data, model, tokenizer, utils, rl, train, dataloader
import logging


# Reinforcement learning on synth tasks with verifiable rewards: sample rollouts from a
# prompt, score them by executing the generated function, update on the group-relative
# advantage (GRPO). See src/rl.py for the loop.
def main():

    logging.basicConfig(level=logging.INFO)

    config = utils.parse_config()
    utils.init_run(config, "rl")

    token = tokenizer.load_tokenizer(utils.get_run_path(config))

    ds_raw = data.get_dataset_rl(config)

    ds = data.prepare_rl(ds_raw, token, config)

    logging.info(f"rl prompts: {  {split: len(rows) for split, rows in ds.items()} }")

    dataloaders = dataloader.create_dataloaders_rl(ds, token, config)

    transformer = model.build_transformer(config)

    rl.train_rl(
        "rl",
        transformer,
        dataloaders,
        token,
        config,
        # From SFT, not pretrain: a pretrained model never emits the answer marker, so
        # every reward is 0, every advantage is 0, and there is no gradient to follow.
        init_from=utils.get_stage_path(config, "sft"),
    )


if __name__ == "__main__":
    main()
