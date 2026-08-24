from src import data, model, tokenizer, train, utils, dataloader
import logging


# Supervised fine-tuning on synth execution-reasoning tasks: prompt in, trace and answer
# out, loss on the completion only.
def main():

    logging.basicConfig(level=logging.INFO)

    config = utils.parse_config()
    utils.init_run(config, "sft")

    # Run-level artifact: pretraining trained it, this only loads it. Fails loudly if
    # pretraining has not run for this config yet, which is the right order of operations.
    token = tokenizer.load_tokenizer(utils.get_run_path(config))

    ds_raw = data.get_dataset_sft(config)

    ds = data.prepare_sft(ds_raw, token, config)

    logging.info(f"sft rows: {  {split: len(rows) for split, rows in ds.items()} }")

    transformer = model.build_transformer(config)

    # Rows are ragged here, so batches come from a TokenSampler against a token budget
    # rather than a fixed row count.
    dataloaders = dataloader.create_dataloaders_sft(ds, token, config)

    # Starting from the pretrained weights: train.load_model_weights takes model_state_dict
    # only, with a fresh optimiser, scheduler and step counter, and train.sft.init_from_step
    # pins which checkpoint so the saved config records it.
    train.train(
        "sft", transformer, dataloaders, token, config,
        init_from=utils.get_stage_path(config, "pretrain"),
    )


if __name__ == "__main__":
    main()
