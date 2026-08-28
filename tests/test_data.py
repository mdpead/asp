"""SFT data preparation: streaming into arrow, the split, and the chunk boundary.

These exist because the corpus stopped fitting in host memory. prepare_sft used to hold
every row's token ids as Python objects — roughly 5KB a row — so a million-row run died
before reaching the GPU. Both functions now work in arrow, which changes the return types
from lists to memory-mapped datasets and moves the split behind a filter.
"""

import datasets
import pytest

from src import data, dataloader


@pytest.fixture(autouse=True)
def _no_arrow_cache():
    """Disable the datasets cache for these tests.

    map() and from_generator() cache on a fingerprint of their inputs, and a cached result
    is returned without running the function. That silently masks changes to the mapper:
    a test asserting on drop behaviour passed against a stale cache entry while the code
    under test had been mutated to drop nothing.
    """
    datasets.disable_caching()
    yield
    datasets.enable_caching()


def _config(tokenizer, num_records, max_length=512, ratio=0.1):
    return {
        "seed": 0,
        "model": {"max_length": max_length},
        "data": {
            "sft": {
                "num_records": num_records,
                "tier": "easy",
                "tasks": ["output"],
                "inputs_per_fn": 4,
                "with_trace": True,
                "test_split_ratio": ratio,
            }
        },
    }


def _prepared(tokenizer, num_records, **kw):
    cfg = _config(tokenizer, num_records, **kw)
    return cfg, data.prepare_sft(data.get_dataset_sft(cfg), tokenizer, cfg)


def test_prepared_rows_are_memory_mapped_not_resident(tokenizer):
    """The point of the change: rows live on disk and are addressed by index.

    A plain list would pass every other test here while reintroducing the memory ceiling.
    """
    _, ds = _prepared(tokenizer, 200)

    for split in ("train", "test"):
        assert ds[split].cache_files, f"{split} is not backed by an arrow file"
        assert not isinstance(ds[split], list)


def test_rows_stay_randomly_indexable(tokenizer):
    """Length bucketing needs random access, which an IterableDataset could not give.

    This is the property that decides whether streaming is usable at all here.
    """
    _, ds = _prepared(tokenizer, 200)
    train = ds["train"]

    assert train[0]["length"] == train[0]["length"]
    assert len(train["length"]) == len(train)
    assert train[len(train) - 1]["ids"]


def test_no_function_appears_in_both_splits(tokenizer):
    """The leak the source-hash split exists to prevent.

    Routing moved from a post-hoc list partition to an is_test flag set at generation and
    applied by an arrow filter, so the guarantee is worth re-asserting on the new path.
    """
    _, ds = _prepared(tokenizer, 300)

    train_sources = set(ds["train"]["source"])
    test_sources = set(ds["test"]["source"])
    assert train_sources and test_sources
    assert not (train_sources & test_sources)


def test_tokenising_keeps_every_row_that_fits(tokenizer):
    """The batched map must neither drop a row that fits nor duplicate one.

    A partial final batch is the easy case to lose, and losing it silently shortens the
    corpus rather than raising.
    """
    cfg, ds = _prepared(tokenizer, 120)

    assert len(ds["train"]) + len(ds["test"]) == len(data.get_dataset_sft(cfg))


def test_token_sampler_batches_arrow_and_lists_identically(tokenizer):
    """The sampler reads a column when offered one and falls back to per-row access.

    Both paths must pack the same batches, or a result measured on one is not comparable
    to a result measured on the other.
    """
    _, ds = _prepared(tokenizer, 200)
    train = ds["train"]
    as_list = [dict(row) for row in train]

    from_arrow = dataloader.TokenSampler(train, 2048, seed=0).batches
    from_list = dataloader.TokenSampler(as_list, 2048, seed=0).batches

    assert from_arrow == from_list


def test_rows_longer_than_the_context_are_dropped(tokenizer):
    """prepare_sft drops rather than truncates, since truncating cuts off the answer."""
    cfg = _config(tokenizer, 200)
    raw = data.get_dataset_sft(cfg)
    loose = data.prepare_sft(raw, tokenizer, cfg)
    # ds[col] is an arrow Column, not a list — it does not concatenate with +.
    lengths = sorted(list(loose["train"]["length"]) + list(loose["test"]["length"]))
    # A bound inside the observed spread, so some rows fit and some do not; an over-tight
    # bound drops every row and leaves an empty split rather than testing the drop.
    bound = lengths[len(lengths) // 2]

    tight_cfg = _config(tokenizer, 200, max_length=bound)
    tight = data.prepare_sft(data.get_dataset_sft(tight_cfg), tokenizer, tight_cfg)

    kept_tight = len(tight["train"]) + len(tight["test"])
    assert 0 < kept_tight < len(lengths)
    assert all(n <= bound for n in tight["train"]["length"])
