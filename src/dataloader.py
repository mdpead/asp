from torch.utils.data import DataLoader, IterableDataset
import torch
from torch.utils.data.sampler import BatchSampler
import random
from functools import partial

class TokenSampler(BatchSampler):
    def __init__(self, ds, token_batch_size, seed):
        self.token_batch_size = token_batch_size
        self.seed = seed
        self.batches = self.generate_batches(ds)

    def generate_batches(self, ds):

        # Create batches based on token counts
        lengths = list(zip(ds["idx"], ds["en_token_length"], ds["cy_token_length"]))
        lengths_sorted = sorted(lengths, key=lambda x: max(x[1], x[2]))
        batches = []
        batch = []
        batch_token_count = 0
        for idx, en_len, cy_len in lengths_sorted:
            token_count = max(en_len, cy_len)
            if batch_token_count + token_count > self.token_batch_size and batch:
                batches.append(batch)
                batch = []
                batch_token_count = 0
            batch.append(idx)
            batch_token_count += token_count
        if batch:
            batches.append(batch)

        return batches

    def __iter__(self):
        # Shuffle batches to introduce randomness
        rng = random.Random(self.seed)
        while True:
            batches = rng.sample(self.batches, len(self.batches))
            for batch in batches:
                yield batch

    def __len__(self):
        return len(self.batches)


SAMPLERS = {
    "token": TokenSampler,
}


def collate_batch(batch, pad_token_id):

    output = {}
    for type in ["src", "tgt"]:
        lang = "en" if type == "src" else "cy"
        input_tokens = [item[f"text_{lang}_tokenized"] for item in batch]
        max_len = max(len(ids) for ids in input_tokens)
        input_ids = torch.tensor(
            [ids + [pad_token_id] * (max_len - len(ids)) for ids in input_tokens], dtype=torch.long
        )
        padding_mask = (input_ids != pad_token_id).bool()
        output[f"{type}_input_ids"] = input_ids
        output[f"{type}_padding_mask"] = padding_mask

    output["tgt_output_ids"] = output["tgt_input_ids"][:, 1:].contiguous()
    output["tgt_input_ids"] = output["tgt_input_ids"][:, :-1].contiguous()
    output["tgt_padding_mask"] = output["tgt_padding_mask"][:, :-1].contiguous()

    output["src_text"] = [item["text_en"] for item in batch]
    output["tgt_text"] = [item["text_cy"] for item in batch]

    return output



def create_dataloaders(
    stage,
    ds,
    config,
):

    train_config = config["train"][stage]
    sampler_name = train_config.get("sampler")
    num_workers = train_config.get("num_workers", 0)
    prefetch_factor = train_config.get("prefetch_factor", 2) if num_workers > 0 else None

    dataloaders = {}
    for split in ds:
        if sampler_name:
            sampler_cls = SAMPLERS.get(sampler_name)
            if sampler_cls is None:
                raise ValueError(f"Unknown sampler: {sampler_name}")
            dataloaders[split] = DataLoader(
                ds[split],
                batch_sampler=sampler_cls(
                    ds[split], train_config["minibatch_token_size"], config["seed"]
                ),
                collate_fn=partial(collate_batch, pad_token_id=config["tokenizer"]["pad_token_id"]),
                pin_memory=True,
                num_workers=num_workers,
                prefetch_factor=prefetch_factor,
            )
        else:
            batch_size = train_config["minibatch_token_size"] // config["model"]["max_length"]
            dataloaders[split] = DataLoader(
                ds[split],
                batch_size=batch_size,
                pin_memory=True,
                num_workers=num_workers,
                prefetch_factor=prefetch_factor,
            )
    return dataloaders
