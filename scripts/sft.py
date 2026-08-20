from src import data, model, tokenizer, utils
import logging


# Supervised fine-tuning on synth execution-reasoning tasks: prompt in, trace and answer
# out, loss on the completion only. Runs the data path today; the training call is not
# wired up yet — see the TODOs at the bottom.
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

    model.build_transformer(config)

    # TODO dataloaders. Needs, in dataloader.py:
    #   - TokenSampler rewritten for single sequences (it currently assumes the en/cy pairs
    #     of the translation project) so rows batch by length against a token budget
    #   - an SFT collate that pads to the batch maximum and writes pad_token_id over the
    #     prompt positions of output_ids, which criterion's ignore_index then masks
    # dataloaders = dataloader.sft_dataloaders(ds, token, config)

    # TODO training. train.get_run only knows how to resume a stage's own run, so it cannot
    # yet start from the pretrained weights: SFT needs model_state_dict from
    # models/<name>/pretrain/checkpoints/<latest>.pt with a fresh optimiser, scheduler and
    # step counter. train.py:107 and :111 also assume fixed-length blocks and an
    # InfiniteRandomSampler, neither of which holds once batches come from a batch sampler.
    # train.train("sft", transformer, dataloaders, token, config)

    logging.warning("SFT data path ran; training is not wired up yet (see TODOs in this file)")


if __name__ == "__main__":
    main()
