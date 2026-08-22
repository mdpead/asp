"""Pretrain data loading: the input/target shift, and the infinite sampler's resume maths.

Built on a synthetic block dataset rather than the real corpus. The script this replaced
downloaded a StarCoder shard and timed batches per second, which measures the disk and the
worker count, not this code; what is worth pinning is the shift, the mask, the batch-size
arithmetic and the fact that a resumed sampler picks up exactly where it left off.
"""

import pytest
import torch

from src.dataloader import InfiniteRandomSampler, collate_lm_batch, create_dataloaders_pretrain

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
