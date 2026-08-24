"""Data loading for both trained stages: the input/target shift, the samplers, the masks.

Built on synthetic rows rather than the real corpus. The script this replaced downloaded a
StarCoder shard and timed batches per second, which measures the disk and the worker count,
not this code; what is worth pinning is the shift, the mask, the batch-size arithmetic and
the fact that a resumed sampler picks up exactly where it left off.

The SFT half pins two things that fail quietly rather than loudly: the prompt/completion
mask boundary, which is a shift off-by-one, and the fact that output_ids is a copy of the
shifted ids rather than a view of them. Both produce a plausible-looking loss curve when
wrong, so neither would show up as a crash during training.
"""

import pytest
import torch

from src.dataloader import (
    InfiniteRandomSampler,
    TokenSampler,
    collate_lm_batch,
    collate_sft_batch,
    create_dataloaders_pretrain,
    create_dataloaders_sft,
)

MAX_LENGTH = 16
BLOCK = MAX_LENGTH + 1  # prepare_pretrain stores one extra token for the shift
MINIBATCH_TOKENS = 64
NUM_BLOCKS = 20


class _Blocks(torch.utils.data.Dataset):
    """Stands in for data.TokenizedDataset: one fixed-length int32 block per item."""

    def __init__(self, n, pad_id, block=BLOCK):
        # Distinct, non-pad ids so any mis-shift shows up as a value mismatch. Row r holds
        # r*block + [0..block), and pad_id is pushed out of that range.
        self.rows = [
            torch.arange(r * block, (r + 1) * block, dtype=torch.int32) + pad_id + 1
            for r in range(n)
        ]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


@pytest.fixture
def config():
    return {
        "seed": 0,
        "model": {"max_length": MAX_LENGTH},
        "train": {
            "pretrain": {
                "minibatch_token_size": MINIBATCH_TOKENS,
                "num_workers": 2,
                "prefetch_factor": 2,
            }
        },
    }


@pytest.fixture
def splits(tokenizer):
    return {
        "train": _Blocks(NUM_BLOCKS, tokenizer.pad_token_id),
        "test": _Blocks(NUM_BLOCKS // 2, tokenizer.pad_token_id),
    }


# --- collate ---


def test_collate_shifts_inputs_and_targets_by_one(tokenizer):
    rows = _Blocks(3, tokenizer.pad_token_id)
    batch = collate_lm_batch([rows[i] for i in range(3)], tokenizer.pad_token_id)

    assert batch["input_ids"].shape == (3, MAX_LENGTH)
    assert batch["output_ids"].shape == (3, MAX_LENGTH)
    assert batch["padding_mask"].shape == (3, MAX_LENGTH)

    for i in range(3):
        # Target at position t is the input at position t+1: both are the same block,
        # offset by one, so comparing the overlap catches an off-by-one in either.
        assert torch.equal(batch["output_ids"][i, :-1], batch["input_ids"][i, 1:].long())


def test_collate_output_ids_are_long_for_cross_entropy(tokenizer):
    """The ids are stored int32; CrossEntropyLoss requires Long targets."""
    rows = _Blocks(2, tokenizer.pad_token_id)
    assert rows[0].dtype == torch.int32

    batch = collate_lm_batch([rows[0], rows[1]], tokenizer.pad_token_id)
    assert batch["output_ids"].dtype == torch.int64


def test_collate_padding_mask_marks_exactly_the_pad_positions(tokenizer):
    pad = tokenizer.pad_token_id
    row = torch.arange(BLOCK, dtype=torch.int32) + pad + 1
    row[-4:] = pad  # a tail of padding, as a short final block would have

    batch = collate_lm_batch([row], pad)

    # The mask is taken before the shift, so it lines up with input_ids.
    assert torch.equal(batch["padding_mask"][0], batch["input_ids"][0] != pad)
    assert batch["padding_mask"][0, :-3].all()
    assert not batch["padding_mask"][0, -3:].any()


# --- infinite sampler ---


def _take(sampler, n):
    it = iter(sampler)
    return [next(it) for _ in range(n)]


def test_sampler_is_deterministic_for_a_seed():
    a = _take(InfiniteRandomSampler(range(NUM_BLOCKS), seed=7), 3 * NUM_BLOCKS)
    b = _take(InfiniteRandomSampler(range(NUM_BLOCKS), seed=7), 3 * NUM_BLOCKS)
    assert a == b
    assert a != _take(InfiniteRandomSampler(range(NUM_BLOCKS), seed=8), 3 * NUM_BLOCKS)


def test_sampler_visits_every_index_once_per_epoch():
    """Reshuffled permutations, not sampling with replacement: no block is seen twice
    before every block has been seen once."""
    drawn = _take(InfiniteRandomSampler(range(NUM_BLOCKS), seed=0), 2 * NUM_BLOCKS)
    assert sorted(drawn[:NUM_BLOCKS]) == list(range(NUM_BLOCKS))
    assert sorted(drawn[NUM_BLOCKS:]) == list(range(NUM_BLOCKS))
    assert all(isinstance(i, int) for i in drawn)


@pytest.mark.parametrize("start_index", [0, 1, NUM_BLOCKS - 1])
def test_sampler_resumes_mid_epoch(start_index):
    full = _take(InfiniteRandomSampler(range(NUM_BLOCKS), seed=0), 2 * NUM_BLOCKS)
    resumed = _take(InfiniteRandomSampler(range(NUM_BLOCKS), seed=0, start_index=start_index), 5)
    assert resumed == full[start_index : start_index + 5]


@pytest.mark.parametrize("epochs,offset", [(1, 0), (1, 3), (2, 5)])
def test_sampler_resumes_after_whole_epochs(epochs, offset):
    """Fast-forwarding past an epoch boundary must consume the RNG, not just skip counts —
    otherwise a restart mid-run replays the first epoch's order."""
    start_index = epochs * NUM_BLOCKS + offset
    full = _take(InfiniteRandomSampler(range(NUM_BLOCKS), seed=0), start_index + 5)
    resumed = _take(InfiniteRandomSampler(range(NUM_BLOCKS), seed=0, start_index=start_index), 5)
    assert resumed == full[start_index:]


# --- dataloader construction ---


def test_batch_size_is_the_token_budget_divided_by_context(splits, tokenizer, config):
    dls = create_dataloaders_pretrain(splits, tokenizer, config)
    for split, dl in dls.items():
        assert dl.batch_size == MINIBATCH_TOKENS // MAX_LENGTH, split


def test_train_split_samples_infinitely_and_test_split_does_not(splits, tokenizer, config):
    dls = create_dataloaders_pretrain(splits, tokenizer, config)
    assert isinstance(dls["train"].sampler, InfiniteRandomSampler)
    assert not isinstance(dls["test"].sampler, InfiniteRandomSampler)


def test_test_split_runs_in_process(splits, tokenizer, config):
    """Validation is a handful of batches; paying worker startup for it would dominate."""
    dls = create_dataloaders_pretrain(splits, tokenizer, config)
    assert dls["train"].num_workers == 2
    assert dls["train"].prefetch_factor == 2
    assert dls["test"].num_workers == 0
    assert dls["test"].prefetch_factor is None


def test_prefetch_factor_is_dropped_when_workers_are_disabled(splits, tokenizer, config):
    """DataLoader rejects a prefetch_factor alongside num_workers=0."""
    config["train"]["pretrain"]["num_workers"] = 0
    dls = create_dataloaders_pretrain(splits, tokenizer, config)
    for dl in dls.values():
        assert dl.num_workers == 0
        assert dl.prefetch_factor is None


def test_batches_have_the_shape_the_model_expects(splits, tokenizer, config):
    config["train"]["pretrain"]["num_workers"] = 0  # keep the test in-process
    dls = create_dataloaders_pretrain(splits, tokenizer, config)
    batch_size = MINIBATCH_TOKENS // MAX_LENGTH

    for i, batch in enumerate(dls["train"]):
        assert set(batch) == {"input_ids", "padding_mask", "output_ids"}
        assert batch["input_ids"].shape == (batch_size, MAX_LENGTH)
        assert batch["output_ids"].shape == (batch_size, MAX_LENGTH)
        assert batch["padding_mask"].shape == (batch_size, MAX_LENGTH)
        if i == 2:
            break
    else:
        pytest.fail("train loader ended; InfiniteRandomSampler should never exhaust it")


def test_test_loader_covers_the_split_once(splits, tokenizer, config):
    config["train"]["pretrain"]["num_workers"] = 0
    dls = create_dataloaders_pretrain(splits, tokenizer, config)
    batch_size = MINIBATCH_TOKENS // MAX_LENGTH

    batches = list(dls["test"])
    seen = sum(batch["input_ids"].shape[0] for batch in batches)

    # drop_last defaults to False, so the ragged final batch is kept and validation sees
    # the whole split rather than a truncated one.
    assert seen == len(splits["test"])
    assert len(batches) == -(-len(splits["test"]) // batch_size)


# --- sft rows ---


def _sft_row(n, prompt_len, pad_id, tag=0):
    """A prepare_sft row: <bos> + prompt + completion + <eos>, flattened to `n` ids.

    `prompt_len` is where the completion starts, matching prepare_sft. Ids are distinct
    within and across rows and never equal pad_id, so a mis-shift or a stray mask shows up
    as a value mismatch rather than passing by coincidence.
    """
    return {
        "ids": [tag * 1000 + i + pad_id + 1 for i in range(n)],
        "prompt_len": prompt_len,
        "length": n - 1,
    }


def _rows_of_length(lengths, pad_id=0):
    """Rows that only the sampler will look at: it reads `length` and nothing else."""
    return [_sft_row(L + 1, prompt_len=1, pad_id=pad_id, tag=t) for t, L in enumerate(lengths)]


# --- sft collate ---


def test_sft_collate_masks_the_prompt_and_scores_the_completion(tokenizer):
    """The shift puts the first completion target at prompt_len - 1.

    Masking prompt_len positions instead would blank that target too, and the model would
    never be taught to emit the first token of its own answer. Masking one fewer would
    score the last prompt token, teaching it to generate part of its own input.
    """
    pad = tokenizer.pad_token_id
    row = _sft_row(n=7, prompt_len=4, pad_id=pad)
    out = collate_sft_batch([row], pad)["output_ids"][0]

    assert (out[: row["prompt_len"] - 1] == pad).all()
    assert out[row["prompt_len"] - 1 :].tolist() == row["ids"][row["prompt_len"] :]


def test_sft_collate_leaves_the_prompt_intact_in_the_inputs(tokenizer):
    """output_ids must be a copy, not a view.

    It overlaps input_ids in the same storage, so masking a view would blank the prompt
    the model is meant to condition on — it would be asked to produce the completion from
    nothing, which trains to a bad loss rather than raising.
    """
    pad = tokenizer.pad_token_id
    row = _sft_row(n=7, prompt_len=4, pad_id=pad)
    batch = collate_sft_batch([row], pad)

    assert batch["input_ids"][0].tolist() == row["ids"][:-1]
    assert pad not in batch["input_ids"][0].tolist()


def test_sft_collate_scores_the_stop_token(tokenizer):
    """<eos> is the last target: nothing else teaches the model to stop generating."""
    pad = tokenizer.pad_token_id
    row = _sft_row(n=7, prompt_len=4, pad_id=pad)
    out = collate_sft_batch([row], pad)["output_ids"][0]

    assert out[-1].item() == row["ids"][-1]


def test_sft_collate_pads_ragged_rows_to_the_batch_maximum(tokenizer):
    pad = tokenizer.pad_token_id
    rows = [
        _sft_row(n=9, prompt_len=3, pad_id=pad, tag=0),
        _sft_row(n=5, prompt_len=2, pad_id=pad, tag=1),
    ]
    batch = collate_sft_batch(rows, pad)

    # One narrower than the longest row, since the shift consumes a position.
    assert batch["input_ids"].shape == (2, 8)
    assert batch["output_ids"].shape == (2, 8)

    # The long row fills the width; the short one is real up to its own length, then pad.
    assert batch["padding_mask"][0].all()
    assert batch["padding_mask"][1, :5].all()
    assert not batch["padding_mask"][1, 5:].any()


def test_sft_collate_masks_padding_and_prompt_with_the_same_id(tokenizer):
    """criterion is built with a single ignore_index, so both must be pad_token_id."""
    pad = tokenizer.pad_token_id
    rows = [
        _sft_row(n=9, prompt_len=3, pad_id=pad, tag=0),
        _sft_row(n=5, prompt_len=2, pad_id=pad, tag=1),
    ]
    out = collate_sft_batch(rows, pad)["output_ids"]

    scored = out != pad
    # Row 0: 9 ids -> 8 targets, 2 of them prompt. Row 1: 5 ids -> 4 targets, 1 prompt,
    # then 4 positions of padding out to the batch width.
    assert scored[0].sum().item() == 8 - (3 - 1)
    assert scored[1].sum().item() == 4 - (2 - 1)


def test_sft_collate_targets_are_long_for_cross_entropy(tokenizer):
    pad = tokenizer.pad_token_id
    batch = collate_sft_batch([_sft_row(n=6, prompt_len=3, pad_id=pad)], pad)
    assert batch["output_ids"].dtype == torch.int64


def test_sft_collate_returns_the_keys_the_train_loop_reads(tokenizer):
    """Same keys as collate_lm_batch, so train_loop and validation_step are stage-agnostic."""
    pad = tokenizer.pad_token_id
    batch = collate_sft_batch([_sft_row(n=6, prompt_len=3, pad_id=pad)], pad)
    assert set(batch) == {"input_ids", "padding_mask", "output_ids"}


# --- token sampler ---


def test_token_sampler_covers_every_row_exactly_once_per_pass():
    lengths = [7, 31, 12, 45, 3, 28, 19, 22, 9, 50]
    sampler = TokenSampler(_rows_of_length(lengths), token_batch_size=200, seed=0)

    drawn = [i for batch in sampler.batches for i in batch]
    assert sorted(drawn) == list(range(len(lengths)))


@pytest.mark.parametrize("budget", [64, 200, 1000])
def test_token_sampler_respects_the_padded_token_budget(budget):
    """The budget bounds len(batch) * the longest row, not the sum of the rows.

    Padding is what the GPU actually processes, so summing true lengths would let a batch
    overrun minibatch_token_size at a bucket boundary.
    """
    lengths = [7, 31, 12, 45, 3, 28, 19, 22, 9, 50]
    sampler = TokenSampler(_rows_of_length(lengths), budget, seed=0)

    for batch in sampler.batches:
        cost = max(lengths[i] for i in batch) * len(batch)
        assert cost <= budget or len(batch) == 1


def test_token_sampler_groups_rows_of_similar_length():
    """Batches are contiguous runs of the length-ordered rows.

    That is the property keeping padding waste low — measured at 0.6% on real SFT data
    against 28.6% for randomly composed batches.
    """
    lengths = [17, 3, 9, 40, 12, 3, 25, 8]
    sampler = TokenSampler(_rows_of_length(lengths), token_batch_size=80, seed=0)

    flat = [i for batch in sampler.batches for i in batch]
    assert flat == sorted(range(len(lengths)), key=lambda i: lengths[i])


def test_token_sampler_keeps_a_row_larger_than_the_whole_budget():
    """prepare_sft already dropped what will not fit the context; dropping more here would
    silently shrink the training set."""
    sampler = TokenSampler(_rows_of_length([5, 500, 5]), token_batch_size=50, seed=0)

    assert sorted(i for batch in sampler.batches for i in batch) == [0, 1, 2]
    assert [1] in sampler.batches


def test_token_sampler_never_exhausts():
    """The train loop runs to num_steps, not to the end of the data."""
    sampler = TokenSampler(_rows_of_length([10] * 6), token_batch_size=40, seed=0)
    wanted = 3 * len(sampler.batches) + 1

    it = iter(sampler)
    assert len([next(it) for _ in range(wanted)]) == wanted


def test_token_sampler_reshuffles_order_but_keeps_batch_membership():
    """Membership is frozen deliberately: re-packing each pass would break the length
    grouping, which measured roughly three times the padding waste on real SFT data."""
    sampler = TokenSampler(_rows_of_length(list(range(5, 45))), token_batch_size=100, seed=0)
    n = len(sampler.batches)
    assert n >= 4, "need several batches for the ordering assertion to mean anything"

    it = iter(sampler)
    first = [tuple(next(it)) for _ in range(n)]
    second = [tuple(next(it)) for _ in range(n)]

    assert sorted(first) == sorted(second)
    assert first != second


def test_token_sampler_reports_no_length():
    """len() must raise rather than describe a single pass.

    Iteration never ends, so a number here would let `for batch in loader` look finite and
    hang. Inheriting BatchSampler would supply one that reads attributes this class never
    sets, turning the TypeError into a confusing AttributeError.
    """
    sampler = TokenSampler(_rows_of_length([10] * 6), token_batch_size=40, seed=0)
    with pytest.raises(TypeError):
        len(sampler)


def test_token_sampler_is_deterministic_for_a_seed():
    rows = _rows_of_length(list(range(5, 45)))

    def take(seed):
        return [tuple(batch) for batch in _take(TokenSampler(rows, 100, seed=seed), 12)]

    assert take(7) == take(7)
    assert take(7) != take(8)


# --- sft dataloader construction ---


@pytest.fixture
def sft_config():
    return {
        "seed": 0,
        "model": {"max_length": MAX_LENGTH},
        "train": {
            "sft": {
                "minibatch_token_size": MINIBATCH_TOKENS,
                "num_workers": 2,
                "prefetch_factor": 2,
            }
        },
    }


@pytest.fixture
def sft_splits(tokenizer):
    pad = tokenizer.pad_token_id
    return {
        "train": [_sft_row(n=6 + (t % 5), prompt_len=3, pad_id=pad, tag=t) for t in range(12)],
        "test": [_sft_row(n=6 + (t % 3), prompt_len=2, pad_id=pad, tag=t) for t in range(5)],
    }


def test_sft_test_split_runs_in_process(sft_splits, tokenizer, sft_config):
    dls = create_dataloaders_sft(sft_splits, tokenizer, sft_config)
    assert dls["train"].num_workers == 2
    assert dls["test"].num_workers == 0
    assert dls["test"].prefetch_factor is None


def test_sft_loader_batches_by_token_budget_not_row_count(sft_splits, tokenizer, sft_config):
    """batch_size must stay None: a batch_sampler owns the grouping, and setting both
    is the error DataLoader raises on."""
    dls = create_dataloaders_sft(sft_splits, tokenizer, sft_config)
    for dl in dls.values():
        assert dl.batch_size is None
        assert isinstance(dl.batch_sampler, TokenSampler)


def test_sft_loader_reports_no_length(sft_splits, tokenizer, sft_config):
    """The pretrain test loader is finite and `list(dl)` works on it; the SFT one is not,
    so it must refuse a length rather than imply the same is safe here."""
    dls = create_dataloaders_sft(sft_splits, tokenizer, sft_config)
    for split, dl in dls.items():
        with pytest.raises(TypeError):
            len(dl)


def test_sft_batches_have_the_shape_the_model_expects(sft_splits, tokenizer, sft_config):
    sft_config["train"]["sft"]["num_workers"] = 0  # keep the test in-process
    dls = create_dataloaders_sft(sft_splits, tokenizer, sft_config)
    pad = tokenizer.pad_token_id

    for i, batch in enumerate(dls["train"]):
        assert set(batch) == {"input_ids", "padding_mask", "output_ids"}
        rows, width = batch["input_ids"].shape
        assert batch["output_ids"].shape == (rows, width)
        assert batch["padding_mask"].shape == (rows, width)
        # Rows vary in length, so unlike pretrain the width is per batch, not the context.
        assert width <= MAX_LENGTH
        assert (batch["output_ids"] != pad).any()
        if i == 2:
            break
    else:
        pytest.fail("train loader ended; TokenSampler should never exhaust it")
