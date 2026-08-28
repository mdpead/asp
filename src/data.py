import hashlib
import itertools
import logging
import re

import datasets as hf_datasets
from torch.utils.data import Dataset

from src import synth


# StarCoder's training format prefixes ~48% of files with metadata markers, sometimes
# several concatenated onto the first line ("<filename>x.py<gh_stars>1-10\ncode..."). They
# were special tokens in StarCoder's own tokenizer; ours splits them into literal subwords,
# and half the corpus does not even parse as Python until they are gone. Anchored to the
# start of the file, so the few legitimate occurrences deeper in real source survive, and
# requiring the trailing newline, so a marker-only file is left alone rather than emptied.
_MARKER_PREFIX = re.compile(r"^<(?:reponame|filename|gh_stars)>[^\n]*\n")


def strip_markers(text):
    """Drop StarCoder's leading metadata line. Also applied when training the tokenizer."""
    return _MARKER_PREFIX.sub("", text, count=1)


def _hash_position(text):
    """Stable position in [0, 1) for a string, fixed across processes and runs.

    blake2b rather than hash(): str hashing is salted per process unless PYTHONHASHSEED is
    set, so hash() would move a train/test boundary on every run.
    """
    digest = hashlib.blake2b((text or "").encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") / 2**64


# The python subset ships as 59 parquet shards, and the count is baked into the filenames
# ("train-00000-of-00059.parquet"), so naming a subset of them means hardcoding the total.
PYTHON_SHARDS = 59


def _shard_files(num_shards):
    if not 1 <= num_shards <= PYTHON_SHARDS:
        raise ValueError(f"num_shards must be 1..{PYTHON_SHARDS}, got {num_shards}")
    return [
        f"python/train-{i:05d}-of-{PYTHON_SHARDS:05d}.parquet" for i in range(num_shards)
    ]


def get_dataset_pretrain(config):
    """Raw StarCoder python files, split by repo. Rows carry a `content` column."""
    ds_config = config["data"]["pretrain"]

    num_shards = ds_config.get("num_shards")
    if num_shards is not None:
        # Download only the shards the step budget needs — whole shards, because a row slice
        # would come after load_dataset had already fetched everything. This assumes the
        # corpus was shuffled before sharding: shard 0 holds 165,528 repos across 218,079
        # rows with only one adjacent pair sharing a repo, which is consistent with that,
        # but it has not been checked across shards.
        logging.info(f"pretrain: loading {num_shards} of {PYTHON_SHARDS} shards")
        ds = hf_datasets.load_dataset(
            "bigcode/starcoderdata", data_files=_shard_files(num_shards), split="train"
        )
    else:
        ds = hf_datasets.load_dataset(
            "bigcode/starcoderdata", data_dir="python", split="train", streaming=False
        )

    # Cap the row count for the toy configs. An index slice is fine here in a way it is not
    # for the train/test split below: this is "give me a small sample of the corpus", not a
    # held-out set, and the repo hash split still runs on whatever it returns.
    sample_size = ds_config.get("sample_size")
    if sample_size is not None:
        ds = ds.select(range(min(sample_size, len(ds))))
        logging.info(f"pretrain: capped at {len(ds)} files")

    # Split on a hash of the repo, not on row index. Rows arrive shuffled corpus-wide (only
    # 1 in 218k consecutive pairs shares a repo), so an index cut sprays each repo's files
    # across both sides: taking the first 0.1% of rows left 36.7% of test files in a repo
    # that also appears in train, and the shared imports, helpers and licence headers that
    # follow flatter the validation loss. Files per repo vary (82% of repos contribute one,
    # the largest contributes 143), so the realised test fraction only approximates the
    # configured ratio.
    ratio = ds_config["test_split_ratio"]

    def is_test(batch):
        return [_hash_position(name) < ratio for name in batch["max_stars_repo_name"]]

    def is_train(batch):
        return [not held_out for held_out in is_test(batch)]

    return {
        "test": ds.filter(is_test, batched=True, desc="Selecting test repos"),
        "train": ds.filter(is_train, batched=True, desc="Selecting train repos"),
    }


_TASK_FORMATTERS = {
    "output": synth.format_output_task,
    "input": synth.format_input_task,
}


def _synth_records(ds_config, seed, with_trace):
    return synth.generate(
        ds_config["num_records"],
        tier=ds_config["tier"],
        seed=seed,
        inputs_per_fn=ds_config["inputs_per_fn"],
        with_trace=with_trace,
    )


def _split_on_source(rows, ratio):
    """Partition synth rows on the generated function, not the row.

    generate() emits several input variants per function, and one function can appear as
    both an output-prediction and an input-prediction task, so splitting per row would put
    the same function on both sides — the leak the repo-level split fixes for StarCoder.
    Plain lists rather than hf_datasets.Dataset: an `args` field holds a mix of ints and
    lists, which arrow cannot type, and regenerating is cheap and deterministic given seed.
    """
    return {
        "test": [row for row in rows if _hash_position(row["source"]) < ratio],
        "train": [row for row in rows if _hash_position(row["source"]) >= ratio],
    }


def _is_test(source, ratio):
    """Which side of the split a generated function falls on. See _split_on_source."""
    return _hash_position(source) < ratio


def _sft_rows(config):
    """Yield prompt/completion rows from the synth generator, one per (record, task)."""
    ds_config = config["data"]["sft"]
    with_trace = ds_config["with_trace"]
    ratio = ds_config["test_split_ratio"]

    for record in _synth_records(ds_config, config["seed"], with_trace):
        is_test = _is_test(record["source"], ratio)
        for task in ds_config["tasks"]:
            prompt, completion = _TASK_FORMATTERS[task](record, with_trace)
            yield {
                "prompt": prompt,
                "completion": completion,
                "task": task,
                "source": record["source"],
                "n_executed": record["n_executed"],
                "is_test": is_test,
            }


def get_dataset_rl(config):
    """Prompts only, plus the fields synth's checkers need to score a rollout.

    The trace is part of the completion, not the prompt, so generation runs untraced here:
    the reward comes from executing the function against the model's answer.
    """
    ds_config = config["data"]["rl"]

    rows = []
    for record in _synth_records(ds_config, config["seed"], with_trace=False):
        for task in ds_config["tasks"]:
            prompt, _ = _TASK_FORMATTERS[task](record, False)
            rows.append(
                {
                    "prompt": prompt,
                    "task": task,
                    "source": record["source"],
                    "args": record["args"],
                    "result": record["result"],
                    "n_executed": record["n_executed"],
                }
            )
    return _split_on_source(rows, ds_config["test_split_ratio"])


def get_dataset_sft(config):
    """Prompt/completion rows from the synth generator, one per (record, task).

    An arrow dataset, so it can be mapped and filtered without ever being resident. Rows
    carry `is_test` rather than being partitioned here: prepare_sft applies the split
    after tokenising, which keeps this a single generation pass — the expensive half,
    since every generated function is executed under a tracer to produce its trace.
    """
    return hf_datasets.Dataset.from_generator(_sft_rows, gen_kwargs={"config": config})


def _tokenise_sft_batch(batch, tokenizer, max_length):
    """Map one batch of prompt/completion rows to model-ready sequences.

    A batched map rather than a hand-rolled chunk loop: arrow feeds fixed-size batches in
    and writes results straight back out, so nothing accumulates and there is no sentinel
    or boundary condition to get wrong. Returning fewer rows than it was given is how an
    over-long task is dropped.
    """
    bos_id, eos_id = tokenizer.bos_token_id, tokenizer.eos_token_id
    prompts = tokenizer(batch["prompt"], add_special_tokens=False)["input_ids"]
    completions = tokenizer(batch["completion"], add_special_tokens=False)["input_ids"]

    carried = ("task", "source", "n_executed", "is_test")
    out = {k: [] for k in ("ids", "prompt_len", "length") + carried}
    for i, (prompt, completion) in enumerate(zip(prompts, completions)):
        ids = [bos_id] + prompt + completion + [eos_id]
        # One token longer than the context, since the input/target shift consumes one.
        # Truncating would cut the answer off the end, so an over-long task is dropped.
        if len(ids) > max_length + 1:
            continue
        out["ids"].append(ids)
        out["prompt_len"].append(1 + len(prompt))
        out["length"].append(len(ids) - 1)
        for key in carried:
            out[key].append(batch[key][i])
    return out


def prepare_sft(ds_raw, tokenizer, config):
    """Tokenise prompt/completion pairs into single sequences with a mask boundary.

    `prompt_len` is where the completion starts in `ids`. The collate shift makes
    output_ids = ids[1:], so the first completion target sits at output_ids[prompt_len - 1]
    and everything before it is masked. `length` is the model input length, which the
    length-bucketed sampler batches on.

    Arrow in, arrow out, the same shape as prepare_pretrain. Rows stay memory-mapped
    rather than resident: holding the corpus as Python token lists cost roughly 5KB a row
    and capped it well below what the GPU could consume. They stay randomly indexable
    too, which the length-bucketed sampler needs and a streaming dataset could not give.
    """
    max_length = config["model"]["max_length"]

    prepared = ds_raw.map(
        _tokenise_sft_batch,
        batched=True,
        batch_size=1000,
        remove_columns=ds_raw.column_names,
        fn_kwargs={"tokenizer": tokenizer, "max_length": max_length},
        desc="Tokenizing sft",
    )
    dropped = len(ds_raw) - len(prepared)
    if dropped:
        logging.warning(
            f"sft: dropped {dropped}/{len(ds_raw)} rows over max_length {max_length}"
        )

    # Split here rather than at generation: partitioning one pass into two collections
    # means buffering one of them, which an arrow filter does not.
    return {
        "test": prepared.filter(lambda b: b["is_test"], batched=True),
        "train": prepared.filter(lambda b: [not v for v in b["is_test"]], batched=True),
    }


def prepare_rl(ds_raw, tokenizer, config):
    """Annotate rollout prompts with their token length.

    The prompt stays text: generate_texts tokenises and left-pads it itself. Only the length
    is added here, so prompts of similar size batch together and left-padding — and so
    wasted rollout compute — stays small. The reward fields ride along untouched, since
    scoring a rollout means executing the function that produced the prompt.
    """
    max_length = config["model"]["max_length"]
    prepared = {}
    for split, rows in ds_raw.items():
        encoded = tokenizer([row["prompt"] for row in rows], add_special_tokens=False)

        kept, dropped = [], 0
        for row, prompt in zip(rows, encoded["input_ids"]):
            length = 1 + len(prompt)  # +1 for the <bos> generate_texts prepends
            # A prompt that fills the context leaves nothing to generate. How much headroom
            # a rollout actually needs is an RL-loop decision, so only the impossible cases
            # are dropped here.
            if length >= max_length:
                dropped += 1
                continue
            kept.append({**row, "length": length})
        if dropped:
            logging.warning(
                f"rl {split}: dropped {dropped}/{len(rows)} prompts with no room to generate"
            )
        prepared[split] = kept
    return prepared


def _tokenize_and_chunk_batch(batch, tokenizer, max_length):
    """Tokenise, join with <eos> separators, and cut fixed blocks — all in one pass.

    One map rather than two: the intermediate variable-length token lists are a full copy of
    the corpus on disk (~128GB at 16B tokens), and nothing reads them except the chunking
    step. Blocks are packed across file boundaries, so no token is spent on padding; the
    remainder of each map batch, under one block, is dropped (0.002% of tokens measured).
    """
    # +1 token per chunk so the input/target shift in collate still leaves
    # max_length tokens for the model to see.
    block = max_length + 1
    eos_id = tokenizer.eos_token_id
    cleaned = [strip_markers(text) for text in batch["content"]]
    encoded = tokenizer(cleaned, add_special_tokens=False)["input_ids"]

    concatenated = list(itertools.chain.from_iterable(ids + [eos_id] for ids in encoded))
    total = (len(concatenated) // block) * block
    return {"ids": [concatenated[i : i + block] for i in range(0, total, block)]}


class TokenizedDataset(Dataset):
    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        return self.ds[idx]["ids"]


def _check_token_budget(n_blocks, config):
    """Report whether the loaded shards cover the configured step budget.

    Undersizing does not raise: InfiniteRandomSampler reshuffles and loops, so training
    quietly begins a second pass over the same blocks — which contradicts the single-pass
    assumption behind dropout: 0.0.
    """
    train_config = config["train"]["pretrain"]
    needed = (
        train_config["num_steps"]
        * train_config["effective_batch_token_size"]
        // config["model"]["max_length"]
    )
    if n_blocks < needed:
        logging.warning(
            f"pretrain: {n_blocks} blocks available but {needed} needed for "
            f"{train_config['num_steps']} steps — training will repeat data "
            f"({needed / n_blocks:.2f} epochs). Raise data.pretrain.num_shards."
        )
    else:
        logging.info(
            f"pretrain: {n_blocks} blocks available, {needed} needed "
            f"({100 * needed / n_blocks:.0f}% of one epoch)"
        )


def prepare_pretrain(ds_raw, tokenizer, config):
    max_length = config["model"]["max_length"]

    # int32, not the int64 arrow infers from Python ints. A 32000-token vocabulary needs 15
    # bits, so the extra four bytes per token buy nothing and cost ~64GB over the corpus.
    # This is a storage win only: with_format("torch") materialises the list column through
    # Python and hands back int64 tensors either way.
    features = hf_datasets.Features({"ids": hf_datasets.Sequence(hf_datasets.Value("int32"))})

    prepared = {}
    for split, ds in ds_raw.items():
        chunked = ds.map(
            _tokenize_and_chunk_batch,
            batched=True,
            remove_columns=ds.column_names,
            features=features,
            fn_kwargs={"tokenizer": tokenizer, "max_length": max_length},
            desc=f"Tokenizing and chunking {split}",
        )
        prepared[split] = TokenizedDataset(chunked.with_format("torch"))

    _check_token_budget(len(prepared["train"]), config)
    return prepared
