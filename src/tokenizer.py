import os
import itertools
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, processors, decoders
from transformers import PreTrainedTokenizerFast
from src import utils


def create_tokenizer(ds, tokenizer_config):
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    special_tokens = [
        "<bos>", "<eos>", "<pad>", "<unk>",
        "<user>", "<assistant>",
    ]
    trainer = trainers.BpeTrainer(
        vocab_size=tokenizer_config["vocab_size"],
        special_tokens=special_tokens,
    )

    train_iter = itertools.islice(
        (example["content"] for example in ds["train"]),
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
