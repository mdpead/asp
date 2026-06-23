from src import data, tokenizer, model, train, utils, dataloader
import logging
import argparse
import yaml

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True, help="Config name (e.g. base, test)")
args = parser.parse_args()

with open(f"configs/{args.config}.yaml") as f:
    config = yaml.safe_load(f)

utils.init_run(config)



# Pretrain

ds_raw = data.get_dataset("pretrain", config)

token = tokenizer.get_tokenizer(ds_raw, config)

ds = data.prepare_dataset("pretrain", ds_raw, token, config)

dataloaders = dataloader.create_dataloaders("pretrain", ds, config)

transformer = model.build_transformer(config)

train.train("pretrain", transformer, dataloaders, token, config)

# SFT



# RL

