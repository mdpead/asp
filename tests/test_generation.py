"""Generation budget rules and batch independence.

Weights are random, so the generated TEXT is meaningless. These assert the contract:
the prompt is never modified, max_new_tokens only ever reduces output, and a row's
result does not depend on what it was batched with.
"""

import pytest
import torch

from src.generation import generate_texts

DEV = "cuda"


def n_tokens(tokenizer, text):
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def test_returns_one_text_per_prompt(model, tokenizer):
    out = generate_texts(model, tokenizer, ["hello", "the quick brown"], DEV, max_new_tokens=4)
    assert len(out) == 2


def test_respects_max_new_tokens(model, tokenizer):
    prompt = "hello"
    base = n_tokens(tokenizer, prompt)
    for new in (1, 5, 12):
        out = generate_texts(model, tokenizer, [prompt], DEV, max_new_tokens=new)
        # eos may stop it early, so this is an upper bound
        assert n_tokens(tokenizer, out[0]) <= base + new


def test_output_is_independent_of_batchmates(model, tokenizer):
    """A short prompt gets a leading pad block when batched with a long one; with an
    explicit budget its result must be identical to running it alone."""
    short, long = "hello", "the quick brown fox jumps over foo bar baz"
    alone = generate_texts(model, tokenizer, [short], DEV, max_new_tokens=15)[0]
    batched = generate_texts(model, tokenizer, [long, short], DEV, max_new_tokens=15)[1]
    assert alone == batched


def test_uncapped_budget_depends_on_batch(model, tokenizer):
    """Documented consequence of the shared frontier: with no cap, the longest prompt
    in the batch sets everyone's budget. Pass max_new_tokens to avoid it."""
    short, long = "hello", "the quick brown fox jumps over foo bar baz"
    alone = generate_texts(model, tokenizer, [short], DEV)[0]
    batched = generate_texts(model, tokenizer, [long, short], DEV)[1]
    assert n_tokens(tokenizer, alone) > n_tokens(tokenizer, batched)


def test_prompt_is_never_truncated(model, tokenizer):
    """An over-long prompt is refused rather than silently trimmed."""
    too_long = "hello world foo bar baz qux " * 20
    assert n_tokens(tokenizer, too_long) > model.max_length
    with pytest.raises(ValueError, match="exceed"):
        generate_texts(model, tokenizer, [too_long], DEV)


def test_prompt_filling_context_generates_nothing(model, tokenizer):
    """Legal but has no room left; returns the prompt and says so."""
    filler = "hello world foo bar baz qux " * 20
    ids = tokenizer(filler, add_special_tokens=False)["input_ids"][: model.max_length - 1]
    exact = tokenizer.decode(ids)

    with pytest.warns(UserWarning, match="generating nothing"):
        out = generate_texts(model, tokenizer, [exact], DEV)
    assert n_tokens(tokenizer, out[0]) == len(ids)


def test_max_new_tokens_clamped_with_warning(model, tokenizer):
    with pytest.warns(UserWarning, match="reduced"):
        generate_texts(model, tokenizer, ["hello"], DEV, max_new_tokens=10_000)


def test_training_mode_restored(model, tokenizer):
    model.train()
    try:
        generate_texts(model, tokenizer, ["hello"], DEV, max_new_tokens=2)
        assert model.training, "generate_texts left the model in eval mode"
    finally:
        model.eval()


def test_no_grad_leaks(model, tokenizer):
    """Generation must not build a graph or leave gradients behind."""
    for p in model.parameters():
        p.grad = None
    generate_texts(model, tokenizer, ["hello"], DEV, max_new_tokens=3)
    assert all(p.grad is None for p in model.parameters())
