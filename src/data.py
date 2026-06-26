import itertools
import datasets as hf_datasets
from torch.utils.data import Dataset


def _load_starcoder(ds_config, seed):
    ds = hf_datasets.load_dataset("bigcode/starcoderdata", data_dir="python", split="train", streaming=False)
    num_rows = len(ds)
    if "sample_size" in ds_config:
        num_rows = min(ds_config["sample_size"], num_rows)
        ds = ds.select(range(num_rows))
    test_size = int(num_rows * ds_config["test_split_ratio"])
    return {"test": ds.select(range(test_size)), "train": ds.select(range(test_size, num_rows))}


LOADERS = {
    "starcoder": _load_starcoder,
}


def get_dataset(stage, config):
    ds_config = config["data"][stage]
    name = ds_config["name"]
    loader = LOADERS.get(name)
    if loader is None:
        raise ValueError(f"Unknown dataset: {name}")
    return loader(ds_config, config["seed"])


def _tokenize_batch(batch, tokenizer):
    eos_id = tokenizer.eos_token_id
    encoded = tokenizer(batch["content"], add_special_tokens=False)["input_ids"]
    return {"ids": [ids + [eos_id] for ids in encoded]}


def _chunk_batch(batch, max_length):
    # +1 token per chunk so the input/target shift in collate still leaves
    # max_length tokens for the model to see.
    block = max_length + 1
    concatenated = list(itertools.chain.from_iterable(batch["ids"]))
    total = (len(concatenated) // block) * block
    return {"ids": [concatenated[i : i + block] for i in range(0, total, block)]}


class TokenizedDataset(Dataset):
    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        return self.ds[idx]["ids"]


def prepare_dataset(stage, ds_raw, tokenizer, config):
    max_length = config["model"]["max_length"]
    prepared = {}
    for split, ds in ds_raw.items():
        tokenized = ds.map(
            _tokenize_batch,
            batched=True,
            remove_columns=ds.column_names,
            fn_kwargs={"tokenizer": tokenizer},
            desc=f"Tokenizing {split}",
        )
        chunked = tokenized.map(
            _chunk_batch,
            batched=True,
            fn_kwargs={"max_length": max_length},
            desc=f"Chunking {split}",
        )
        prepared[split] = TokenizedDataset(chunked.with_format("torch"))
    return prepared
