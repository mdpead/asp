import pytest
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

from src.model import Transformer


def pytest_addoption(parser):
    parser.addoption(
        "--benchmark",
        action="store_true",
        default=False,
        help="also run benchmark tests (slow; they print timing tables)",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "benchmark: timing comparison, only collected with --benchmark"
    )


# The Triton kernels assert is_cuda, and every test path reaches them, so skip the
# whole suite rather than failing confusingly on a CPU-only machine.
def pytest_collection_modifyitems(config, items):
    no_cuda = pytest.mark.skip(reason="requires CUDA")
    not_asked = pytest.mark.skip(reason="benchmark: pass --benchmark to run")
    for item in items:
        if not torch.cuda.is_available():
            item.add_marker(no_cuda)
        elif "benchmark" in item.keywords and not config.getoption("--benchmark"):
            item.add_marker(not_asked)


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


@pytest.fixture(scope="session")
def tokenizer():
    """Tiny throwaway BPE tokenizer with the same special tokens as src.tokenizer."""
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(
        ["hello world foo bar baz qux " * 60, "the quick brown fox jumps over " * 60],
        trainers.BpeTrainer(
            vocab_size=400, special_tokens=["<bos>", "<eos>", "<pad>", "<unk>"]
        ),
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=tok,
        bos_token="<bos>",
        eos_token="<eos>",
        pad_token="<pad>",
        unk_token="<unk>",
    )


@pytest.fixture(scope="session")
def make_model():
    """Factory for small models. Session-scoped so weights are reused where possible."""

    def _make(
        vocab_size,
        max_length=64,
        num_heads=4,
        num_kv_heads=2,
        num_layers=2,
        capacity_factor=4.0,
    ):
        torch.manual_seed(0)
        model = Transformer(
            d_model=64,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            d_h=16,
            d_ff=128,
            num_experts=4,
            top_k=2,
            # Default high enough that no token can ever be dropped: these tests assert on
            # shapes, masking and cache equivalence, none of which should turn on how the
            # router happened to load an expert. Tests about capacity pass their own value.
            capacity_factor=capacity_factor,
            num_layers=num_layers,
            vocab_size=vocab_size,
            max_length=max_length,
            dropout=0.0,
        )
        return model.to("cuda").to(torch.float16).eval()

    return _make


@pytest.fixture(scope="session")
def model(tokenizer, make_model):
    return make_model(vocab_size=len(tokenizer))
