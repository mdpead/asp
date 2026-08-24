"""Generation budget rules and batch independence.

Weights are random, so the generated TEXT is meaningless. These assert the contract:
the prompt is never modified, max_new_tokens only ever reduces output, and a row's
result does not depend on what it was batched with.
"""

import pytest
import torch
import torch.nn.functional as F

from src.generation import generate_texts, make_kv_cache

DEV = "cuda"


def n_tokens(tokenizer, text):
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def test_returns_one_text_and_one_flag_per_prompt(model, tokenizer):
    out, finished = generate_texts(
        model, tokenizer, ["hello", "the quick brown"], DEV, max_new_tokens=4
    )
    assert len(out) == 2
    assert len(finished) == 2
    assert all(isinstance(f, bool) for f in finished)


def test_respects_max_new_tokens(model, tokenizer):
    prompt = "hello"
    base = n_tokens(tokenizer, prompt)
    for new in (1, 5, 12):
        out, _ = generate_texts(model, tokenizer, [prompt], DEV, max_new_tokens=new)
        # eos may stop it early, so this is an upper bound
        assert n_tokens(tokenizer, out[0]) <= base + new


def test_output_is_independent_of_batchmates(model, tokenizer):
    """A short prompt gets a leading pad block when batched with a long one; with an
    explicit budget its result must be identical to running it alone."""
    short, long = "hello", "the quick brown fox jumps over foo bar baz"
    alone = generate_texts(model, tokenizer, [short], DEV, max_new_tokens=15)[0][0]
    batched = generate_texts(model, tokenizer, [long, short], DEV, max_new_tokens=15)[0][1]
    assert alone == batched


def test_uncapped_budget_depends_on_batch(model, tokenizer):
    """Documented consequence of the shared frontier: with no cap, the longest prompt
    in the batch sets everyone's budget. Pass max_new_tokens to avoid it."""
    short, long = "hello", "the quick brown fox jumps over foo bar baz"
    alone = generate_texts(model, tokenizer, [short], DEV)[0][0]
    batched = generate_texts(model, tokenizer, [long, short], DEV)[0][1]
    assert n_tokens(tokenizer, alone) > n_tokens(tokenizer, batched)


def test_framing_tokens_are_stripped_from_the_text(model, tokenizer):
    """bos, padding and eos are this function's framing, not model output.

    A caller cannot remove them safely once they are in the string: scoring a rollout
    means ast.literal_eval on the answer, and an <eos> stuck to it is a SyntaxError.

    All three are forced into the stream rather than hoped for — bos is always prepended,
    batching a short prompt with a long one produces the left padding, and <eos> is put in
    a prompt, which the returned text echoes. Waiting for the model to emit <eos> on its
    own would pass whether or not it is stripped, since random weights rarely reach it.
    """
    short = f"hello {tokenizer.eos_token} world"
    long = "the quick brown fox jumps over foo bar baz"
    assert tokenizer.eos_token_id in tokenizer(short, add_special_tokens=False)["input_ids"]

    texts, _ = generate_texts(model, tokenizer, [long, short], DEV, max_new_tokens=15)

    for text in texts:
        for token in (tokenizer.bos_token, tokenizer.eos_token, tokenizer.pad_token):
            assert token not in text, f"{token} survived into {text!r}"


def test_other_special_tokens_survive_generation(model, tokenizer):
    """Reasoning markers ride in the text and callers split on them, so generation must
    not skip special tokens wholesale — only the three framing ones go, dropped by id.

    Driven through the prompt because the returned text echoes it: a decode that skipped
    special tokens would erase the marker from that echo, whatever the model then emits.
    """
    marker = "<|think|>"
    assert len(tokenizer(marker, add_special_tokens=False)["input_ids"]) == 1

    texts, _ = generate_texts(
        model, tokenizer, [f"hello {marker} world"], DEV, max_new_tokens=3
    )
    assert marker in texts[0]


def test_finished_reports_eos_not_budget_exhaustion(model, tokenizer):
    """A one-token budget cannot reach <eos>, so nothing is reported as finished."""
    _, finished = generate_texts(model, tokenizer, ["hello"], DEV, max_new_tokens=1)
    assert finished == [False]


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
        out, _ = generate_texts(model, tokenizer, [exact], DEV)
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


def test_cached_logits_match_uncached_forward(model, tokenizer):
    """The kv cache is an optimization, not a model change: at every step, decoding
    with the cache must produce the same logits as a full no-cache forward over the
    same prefix. The behavioral tests above cannot see a cache that is consistently
    wrong (bad RoPE offset, misaligned mask, stale entries); this one can."""
    prompts = ["hello", "the quick brown fox jumps over"]
    prompt_ids = tokenizer(prompts, add_special_tokens=False)["input_ids"]
    prompt_ids = [[tokenizer.bos_token_id] + ids for ids in prompt_ids]
    encoded = tokenizer.pad({"input_ids": prompt_ids}, return_tensors="pt", padding_side="left")

    input_ids = encoded["input_ids"].to(DEV)
    batch_size, input_length = input_ids.shape
    seq_starts = (input_length - encoded["attention_mask"].sum(dim=-1)).to(
        device=DEV, dtype=torch.int32
    )

    new_tokens = 8
    token_ids = F.pad(input_ids, (0, new_tokens), value=tokenizer.pad_token_id)
    kv_cache = make_kv_cache(model, batch_size, input_length + new_tokens, DEV)

    num_cached = 0
    with torch.no_grad():
        for cur_length in range(input_length, input_length + new_tokens):
            cached, _, new_kv = model(
                token_ids[:, num_cached:cur_length], seq_starts, kv_cache[..., 0:num_cached, :]
            )
            kv_cache[..., num_cached:cur_length, :] = new_kv
            num_cached = cur_length

            full, _ = model(token_ids[:, 0:cur_length], seq_starts)

            a, b = cached[:, -1].float(), full[:, -1].float()
            err = (a - b).abs().max().item()
            scale = b.abs().max().item()
            step = cur_length - input_length
            assert err / scale < 1e-2, f"step {step}: cached logits diverge, rel {err / scale:.4g}"

            # Both paths continue from the reference's greedy token, so any divergence
            # is attributable to this step alone rather than a drifted prefix.
            token_ids[:, cur_length] = b.argmax(dim=-1)
