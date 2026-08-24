import os
import itertools
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, processors, decoders
from transformers import PreTrainedTokenizerFast
from src import data, utils


def create_tokenizer(ds, tokenizer_config):
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))

    # digit_split puts every digit in its own pre-token, so BPE can never merge a number
    # into one atomic id. Without it, 99% of the integers appearing in synth traces are a
    # single token bearing no relation to their digits, and the model has to memorise each
    # arithmetic fact per value rather than learn the operation. Read from the config
    # rather than hardcoded because config["tokenizer"] is what the stage fingerprint
    # compares: a change made only here would leave the saved tokenizer in place and the
    # pretrained checkpoints valid, silently keeping the old scheme.
    byte_level = pre_tokenizers.ByteLevel(add_prefix_space=False)
    if tokenizer_config.get("digit_split"):
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
            [pre_tokenizers.Digits(individual_digits=True), byte_level]
        )
    else:
        tokenizer.pre_tokenizer = byte_level
    tokenizer.decoder = decoders.ByteLevel()

    # Reasoning and answer delimiters, pipe-guarded so they cannot collide with the angle
    # brackets that occur naturally in Python reprs ("<class 'int'>"). One token each
    # rather than the four that "# Trace:\n" costs, and unlike a "#" comment they cannot
    # appear in generated source, so extract_answer's split is unambiguous. Reserved here
    # rather than added later because a post-hoc add_tokens would grow the vocabulary past
    # vocab_size and so the embedding matrix past the model's.
    #
    # The spare slots exist so the next format idea does not cost another pretraining run:
    # the vocabulary is fixed at training time, and adding a token later means retraining.
    special_tokens = [
        "<bos>", "<eos>", "<pad>", "<unk>",
        "<|think|>", "<|/think|>",
        "<|answer|>", "<|/answer|>",
        *[f"<|reserved_{i}|>" for i in range(16)],
    ]
    trainer = trainers.BpeTrainer(
        vocab_size=tokenizer_config["vocab_size"],
        special_tokens=special_tokens,
    )

    train_iter = itertools.islice(
        (data.strip_markers(example["content"]) for example in ds["train"]),
        tokenizer_config["training_size"],
    )
    tokenizer.train_from_iterator(train_iter, trainer)

    tokenizer.post_processor = processors.TemplateProcessing(
        single="<bos> $A <eos>",
        special_tokens=[
            ("<bos>", tokenizer.token_to_id("<bos>")),
            ("<eos>", tokenizer.token_to_id("<eos>")),
        ],
    )

    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<bos>",
        eos_token="<eos>",
        pad_token="<pad>",
        unk_token="<unk>",
    )


def save_tokenizer(tokenizer, dir):
    tokenizer.save_pretrained(f"{dir}/tokenizer")


def load_tokenizer(dir):
    return PreTrainedTokenizerFast.from_pretrained(f"{dir}/tokenizer")


def get_tokenizer(ds, config):
    tokenizer_config = config["tokenizer"]
    run_path = utils.get_run_path(config)
    tokenizer_path = f"{run_path}/tokenizer"

    if os.path.isdir(tokenizer_path):
        return load_tokenizer(run_path)

    tokenizer = create_tokenizer(ds, tokenizer_config)
    save_tokenizer(tokenizer, run_path)

    return tokenizer
