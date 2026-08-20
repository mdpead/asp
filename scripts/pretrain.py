from src import data, tokenizer, model, train, utils, dataloader
import logging


# Everything runs under a __main__ guard: DataLoader workers re-import this module
# (Python 3.14 defaults to the forkserver start method), so module-level work would
# be repeated in every worker.
def main():

    logging.basicConfig(level=logging.INFO)

    config = utils.parse_config()
    utils.init_run(config, "pretrain")

    ds_raw = data.get_dataset_pretrain(config)

    # Trains the BPE on first run and loads it after, which is why the raw dataset has to
    # exist before this line: it is the corpus the vocabulary is built from.
    token = tokenizer.get_tokenizer(ds_raw, config)

    ds = data.prepare_pretrain(ds_raw, token, config)

    dataloaders = dataloader.create_dataloaders_pretrain(ds, token, config)

    transformer = model.build_transformer(config)

    train.train("pretrain", transformer, dataloaders, token, config)


if __name__ == "__main__":
    main()
