from torch.utils.data import DataLoader
import torch
from torch.utils.data.sampler import Sampler
import numpy as np
import random
from functools import partial


class InfiniteRandomSampler(Sampler):
    """Endless reshuffled row indices, resumable to an exact position.

    `batch_size` is only needed to resume: this yields rows, the training loop counts
    minibatches, and a fixed batch size is the bridge between them. It is fixed only
    because pretrain blocks are all one length; leave it None where resume is not wanted
    and start_minibatch will refuse rather than compute a wrong offset.
    """

    def __init__(self, data_source, seed, start_index=0, batch_size=None):
        self.n = len(data_source)
        self.seed = seed
        self.start_index = start_index
        self.batch_size = batch_size

    @property
    def start_minibatch(self):
        if self.batch_size is None:
            raise ValueError("sampler built without batch_size; cannot resume by minibatch")
        return self.start_index // self.batch_size

    @start_minibatch.setter
    def start_minibatch(self, n):
        if self.batch_size is None:
            raise ValueError("sampler built without batch_size; cannot resume by minibatch")
        self.start_index = n * self.batch_size

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        # Fast-forward whole epochs without yielding, keeping the RNG in sync
        skip = self.start_index
        while skip >= self.n:
            rng.permutation(self.n)
            skip -= self.n
        first = True
        while True:
            order = rng.permutation(self.n)
            if first:
                order = order[skip:]
                first = False
            yield from (int(i) for i in order)


class TokenSampler(Sampler):
    """Group rows of similar length into batches that fit a token budget.

    Padding makes a batch cost len(batch) * its longest row, not the sum of its rows, so
    that product is what the budget is checked against. Sorting by length first keeps the
    two close, which is what keeps the padding waste low in the first place.

    Deliberately no __len__: iteration never ends, so any length this could return would
    describe one pass and not the thing being iterated. Callers that want a bounded number
    of batches take them explicitly, as validation_step does with islice; a __len__ here
    would let `for batch in loader` look finite and hang instead. InfiniteRandomSampler
    omits one for the same reason.

    Sampler rather than BatchSampler, despite being passed as batch_sampler=. DataLoader
    only requires an iterable of index lists, and BatchSampler brings a __len__ that reads
    self.drop_last and self.sampler — attributes its __init__ would set and this one does
    not — so inheriting it turns len(loader) into a confusing AttributeError instead of
    the TypeError that says no length exists.
    """

    def __init__(self, ds, token_batch_size, seed, start_minibatch=0):
        self.token_batch_size = token_batch_size
        self.seed = seed
        # Each yielded item is one minibatch, so no conversion is needed here — unlike
        # InfiniteRandomSampler, which yields rows and needs a batch size to translate.
        self.start_minibatch = start_minibatch
        self.batches = self.generate_batches(ds)

    def generate_batches(self, ds):
        # Read `length` as a whole column where the dataset offers one. Row-at-a-time
        # access on an arrow-backed dataset materialises each row through Python — about
        # 70us apiece, so tens of seconds over a corpus this runs over twice. Plain lists
        # of dicts have no column to read, hence the fallback.
        if hasattr(ds, "column_names"):
            lengths = list(ds["length"])
        else:
            lengths = [row["length"] for row in ds]

        # Position in the dataset is the index the DataLoader will ask for. `length` is
        # the model input length prepare_sft recorded for exactly this purpose.
        order = sorted(range(len(lengths)), key=lengths.__getitem__)

        batches = []
        batch = []
        longest = 0
        for idx in order:
            length = lengths[idx]
            # A row longer than the whole budget still gets its own batch rather than
            # being dropped — prepare_sft already dropped what does not fit the context.
            if max(longest, length) * (len(batch) + 1) > self.token_batch_size and batch:
                batches.append(batch)
                batch, longest = [], 0
            batch.append(idx)
            longest = max(longest, length)
        if batch:
            batches.append(batch)

        return batches

    def __iter__(self):
        # Shuffle batches to introduce randomness
        rng = random.Random(self.seed)
        skip = self.start_minibatch
        while True:
            batches = rng.sample(self.batches, len(self.batches))
            # Fast-forward whole passes without yielding, drawing from the RNG as we go so
            # a resumed run continues the shuffle rather than replaying the first pass.
            if skip >= len(batches):
                skip -= len(batches)
                continue
            for batch in batches[skip:]:
                yield batch
            skip = 0


def collate_sft_batch(batch, pad_token_id):
    """Pad to the batch maximum and mask the prompt out of the targets.

    Rows are variable length here, unlike pretrain's fixed blocks, so padding is per batch
    rather than absent. Prompt positions are overwritten in output_ids only: the model
    still reads the prompt through input_ids, it is just not scored on predicting it,
    since criterion is built with ignore_index=pad_token_id.
    """
    max_len = max(len(row["ids"]) for row in batch)
    ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    for i, row in enumerate(batch):
        ids[i, : len(row["ids"])] = torch.tensor(row["ids"], dtype=torch.long)

    input_ids = ids[:, :-1]
    # A view would alias input_ids, so the prompt masking below would blank the model's
    # own input as well as the targets.
    output_ids = ids[:, 1:].clone()

    # The shift puts the first completion target at prompt_len - 1; everything before it
    # is prompt. Trailing pad positions already hold pad_token_id and need no masking.
    for i, row in enumerate(batch):
        output_ids[i, : row["prompt_len"] - 1] = pad_token_id

    return {
        "input_ids": input_ids,
        "padding_mask": input_ids != pad_token_id,
        "output_ids": output_ids,
    }


def collate_lm_batch(batch, pad_token_id):
    input_ids = torch.stack(batch)
    padding_mask = (input_ids != pad_token_id).bool()
    return {
        "input_ids": input_ids[:, :-1],
        "padding_mask": padding_mask[:, :-1],
        # CrossEntropyLoss requires Long targets. A no-op while with_format("torch") hands
        # back int64, but the ids are stored int32 and this is the only place that would
        # break if that ever changed.
        "output_ids": input_ids[:, 1:].long(),
    }


def create_dataloaders_pretrain(
    ds,
    tokenizer,
    config,
):

    train_config = config["train"]["pretrain"]
    num_workers = train_config.get("num_workers", 0)
    prefetch_factor = train_config.get("prefetch_factor", 2) if num_workers > 0 else None

    dataloaders = {}
    for split in ds:
        split_num_workers = 0 if split == "test" else num_workers
        split_prefetch_factor = None if split_num_workers == 0 else prefetch_factor

        batch_size = train_config["minibatch_token_size"] // config["model"]["max_length"]
        # batch_size reaches the sampler only so resume_at_minibatch can convert the training
        # loop's minibatch count into the row offset this sampler indexes by.
        sampler = (
            InfiniteRandomSampler(ds[split], config["seed"], batch_size=batch_size)
            if split == "train"
            else None
        )
        dataloaders[split] = DataLoader(
            ds[split],
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=partial(collate_lm_batch, pad_token_id=tokenizer.pad_token_id),
            pin_memory=True,
            num_workers=split_num_workers,
            prefetch_factor=split_prefetch_factor,
        )
    return dataloaders


def create_dataloaders_sft(
    ds,
    tokenizer,
    config,
):

    train_config = config["train"]["sft"]
    num_workers = train_config.get("num_workers", 0)
    prefetch_factor = train_config.get("prefetch_factor", 2) if num_workers > 0 else None

    dataloaders = {}
    for split in ds:
        split_num_workers = 0 if split == "test" else num_workers
        split_prefetch_factor = None if split_num_workers == 0 else prefetch_factor

        sampler = TokenSampler(ds[split], train_config["minibatch_token_size"], config["seed"])
        dataloaders[split] = DataLoader(
            ds[split],
            batch_sampler=sampler,
            collate_fn=partial(collate_sft_batch, pad_token_id=tokenizer.pad_token_id),
            pin_memory=True,
            num_workers=split_num_workers,
            prefetch_factor=split_prefetch_factor,
        )
    return dataloaders


def resume_at_minibatch(dataloader, n_minibatches):
    """Fast-forward a train loader past n_minibatches without reading any data.

    Both samplers expose start_minibatch in the training loop's own unit and convert to
    their native indexing themselves, so this does not need to know which stage it is
    handed. The lookup matters: when a batch_sampler is set, DataLoader still builds a
    plain sampler and never reads it, so writing resume state onto `.sampler` is silently
    discarded — which is exactly how this failed before. `batch_size is None` is torch's
    own signal for "a custom batch_sampler owns the grouping here".

    The check is for a resumable sampler, not for the train split. It catches pretrain's
    test loader, whose sampler is a stock SequentialSampler, but not SFT's, which uses the
    same TokenSampler class as its train split. Resuming a validation loader would skip
    batches and make validation loss incomparable across restarts, so call this on the
    train split only.
    """
    sampler = dataloader.batch_sampler if dataloader.batch_size is None else dataloader.sampler
    if not hasattr(sampler, "start_minibatch"):
        raise TypeError(
            f"cannot resume a {type(sampler).__name__}: it has no start_minibatch"
        )
    sampler.start_minibatch = n_minibatches
