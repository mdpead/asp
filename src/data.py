import datasets as hf_datasets
import torch
from torch.utils.data import IterableDataset


def _load_starcoder(ds_config, seed):
    ds = hf_datasets.load_dataset("bigcode/starcoderdata", data_dir="python", split="train", streaming=True)
    num_rows = ds.info.splits["train"].num_examples
    if "sample_size" in ds_config:
        num_rows = min(ds_config["sample_size"], num_rows)
        ds = ds.take(num_rows)
    test_size = int(num_rows * ds_config["test_split_ratio"])
    return {"test": ds.take(test_size), "train": ds.skip(test_size)}



class StreamingChunkedDataset(IterableDataset):
    def __init__(self, ds_stream, tokenizer, max_length):
        self.ds_stream = ds_stream
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        buffer = []
        eos_id = self.tokenizer.eos_token_id
        for i, example in enumerate(self.ds_stream):
            if worker_info is not None and i % worker_info.num_workers != worker_info.id:
                continue
            tokens = self.tokenizer.encode(example["content"], add_special_tokens=False)
            buffer.extend(tokens)
            buffer.append(eos_id)
            while len(buffer) >= self.max_length:
                yield torch.tensor(buffer[:self.max_length], dtype=torch.long)
                buffer = buffer[self.max_length:]


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


def prepare_dataset(stage, ds_raw, tokenizer, config):
    max_length = config["model"]["max_length"]
    return {split: StreamingChunkedDataset(ds_raw[split], tokenizer, max_length) for split in ds_raw}
