"""Kernel correctness against a PyTorch reference.

Tolerance strategy: rather than a hand-picked atol, each comparison asserts the kernel
is no less accurate than computing the *same reference maths in the same precision*.
That keeps the bound meaningful as head dims and sequence lengths change.
"""

import pytest
import torch

from src.kernels.flash_attention import flash_attention

DEV = "cuda"
DTYPE = torch.float16

# (num_q_heads, num_kv_heads) -> ratio 1 (MHA), 4, 2, and 4 with a single KV head (MQA)
HEAD_CONFIGS = [(4, 4), (8, 2), (8, 4), (4, 1)]
PAD_CONFIGS = [[0, 0], [0, 17], [5, 40]]


def _attention(q, k, v, seq_starts, causal, compute_dtype):
    """Reference attention. GQA is expressed with repeat_interleave, so autograd's
    backward through it performs exactly the group sum the dkdv kernel must reproduce."""
    seq = q.shape[2]
    head_dim = q.shape[3]
    ratio = q.shape[1] // k.shape[1]

    qq = q.to(compute_dtype)
    kk = k.repeat_interleave(ratio, dim=1).to(compute_dtype)
    vv = v.repeat_interleave(ratio, dim=1).to(compute_dtype)

    scores = (qq @ kk.transpose(-1, -2)) / (head_dim**0.5)

    pos = torch.arange(seq, device=q.device)
    mask = (pos[None, :] >= seq_starts[:, None].to(pos.dtype))[:, None, None, :]
    if causal:
        mask = mask & (pos[None, :] <= pos[:, None])[None, None]
    scores = scores.masked_fill(~mask, float("-inf"))

    probs = torch.softmax(scores, dim=-1)
    # A fully-masked row softmaxes to NaN; the kernel yields zeros for those rows.
    probs = torch.nan_to_num(probs, nan=0.0)
    return probs @ vv


def _max_err(a, b):
    return (a.float() - b.float()).abs().max().item()


def assert_no_worse_than_reference(actual, exact, same_precision, label, slack=2.0, rel_floor=2e-2):
    """Pass if the kernel is either no further from `exact` than the same maths in the
    same precision (within slack), or within rel_floor of the tensor's scale.

    The second clause matters because when the fp16 reference lands very close to fp32 the
    ratio has a near-zero denominator and explodes on errors that are objectively tiny.
    Measured worst-case relative error across all configs here is 5e-3, so rel_floor=2e-2
    leaves ~4x headroom while a genuine bug (wrong head index, missing mask) produces
    order-1 relative error and is still caught.
    """
    err = _max_err(actual, exact)
    baseline = _max_err(same_precision, exact)
    scale = max(exact.float().abs().max().item(), 1e-9)
    limit = max(slack * baseline, rel_floor * scale)
    assert err <= limit, (
        f"{label}: error {err:.4g} exceeds limit {limit:.4g} "
        f"(fp16 baseline {baseline:.4g}, scale {scale:.4g}, rel {err / scale:.2e})"
    )


def _inputs(batch, q_heads, kv_heads, seq, head_dim, pads, requires_grad=False):
    q = torch.randn(batch, q_heads, seq, head_dim, device=DEV, dtype=DTYPE, requires_grad=requires_grad)
    k = torch.randn(batch, kv_heads, seq, head_dim, device=DEV, dtype=DTYPE, requires_grad=requires_grad)
    v = torch.randn(batch, kv_heads, seq, head_dim, device=DEV, dtype=DTYPE, requires_grad=requires_grad)
    seq_starts = torch.tensor(pads, device=DEV, dtype=torch.int32)
    return q, k, v, seq_starts


@pytest.mark.parametrize("q_heads,kv_heads", HEAD_CONFIGS)
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("pads", PAD_CONFIGS)
def test_forward_matches_reference(q_heads, kv_heads, causal, pads):
    seq, head_dim = 64, 32
    q, k, v, seq_starts = _inputs(len(pads), q_heads, kv_heads, seq, head_dim, pads)

    out = flash_attention(q, k, v, seq_starts, causal=causal)
    exact = _attention(q, k, v, seq_starts, causal, torch.float32)
    same_precision = _attention(q, k, v, seq_starts, causal, DTYPE)

    assert not out.isnan().any(), "kernel produced NaN"
    assert not out.isinf().any(), "kernel produced inf"

    # Only rows the caller cares about: pad-position queries are discarded downstream.
    for b, start in enumerate(pads):
        assert_no_worse_than_reference(
            out[b, :, start:], exact[b, :, start:], same_precision[b, :, start:], f"out row {b}"
        )


@pytest.mark.parametrize("q_heads,kv_heads", HEAD_CONFIGS)
@pytest.mark.parametrize("pads", [[0, 0], [5, 40]])
def test_backward_matches_reference(q_heads, kv_heads, pads):
    seq, head_dim, causal = 64, 32, True
    q, k, v, seq_starts = _inputs(len(pads), q_heads, kv_heads, seq, head_dim, pads, requires_grad=True)

    # Weight the loss so pad-position queries contribute nothing, as a real loss with
    # ignore_index on pad targets would.
    weights = torch.zeros(len(pads), 1, seq, 1, device=DEV)
    for b, start in enumerate(pads):
        weights[b, :, start:] = 1.0

    def grads_of(fn):
        qq, kk, vv = (x.detach().clone().requires_grad_(True) for x in (q, k, v))
        (fn(qq, kk, vv).float() * weights).pow(2).sum().backward()
        return qq.grad, kk.grad, vv.grad

    got = grads_of(lambda a, b, c: flash_attention(a, b, c, seq_starts, causal=causal))
    exact = grads_of(lambda a, b, c: _attention(a, b, c, seq_starts, causal, torch.float32))
    same_precision = grads_of(lambda a, b, c: _attention(a, b, c, seq_starts, causal, DTYPE))

    for name, g, e, s in zip(("dq", "dk", "dv"), got, exact, same_precision):
        assert not g.isnan().any(), f"{name} contains NaN"
        assert not g.isinf().any(), f"{name} contains inf"
        assert_no_worse_than_reference(g, e, s, name)


@pytest.mark.parametrize("q_heads,kv_heads", [(4, 4), (8, 2)])
def test_padding_is_invisible(q_heads, kv_heads):
    """The same real tokens must give the same output with or without a leading pad block."""
    seq, head_dim, pad = 48, 32, 12
    real = seq - pad

    q = torch.randn(1, q_heads, real, head_dim, device=DEV, dtype=DTYPE)
    k = torch.randn(1, kv_heads, real, head_dim, device=DEV, dtype=DTYPE)
    v = torch.randn(1, kv_heads, real, head_dim, device=DEV, dtype=DTYPE)

    def pad_front(x):
        filler = torch.randn(1, x.shape[1], pad, head_dim, device=DEV, dtype=DTYPE)
        return torch.cat([filler, x], dim=2)

    unpadded = flash_attention(q, k, v, torch.zeros(1, device=DEV, dtype=torch.int32), causal=True)
    padded = flash_attention(
        pad_front(q), pad_front(k), pad_front(v),
        torch.full((1,), pad, device=DEV, dtype=torch.int32), causal=True,
    )

    err = _max_err(padded[:, :, pad:], unpadded)
    scale = unpadded.float().abs().max().item()
    assert err / scale < 5e-3, f"padding changed the result: rel {err / scale:.4g}"


def test_fully_masked_rows_are_zero_not_nan():
    """Pad-position queries under causal masking see no valid keys at all."""
    pads = [0, 20]
    q, k, v, seq_starts = _inputs(2, 4, 2, 64, 32, pads)
    out = flash_attention(q, k, v, seq_starts, causal=True)

    masked_rows = out[1, :, : pads[1]]
    assert not out.isnan().any()
    assert masked_rows.abs().max().item() == 0.0, "fully-masked rows should be exactly zero"


def test_rejects_mismatched_seq_starts_shape():
    q, k, v, _ = _inputs(2, 4, 2, 32, 32, [0, 0])
    bad = torch.zeros(3, device=DEV, dtype=torch.int32)  # batch is 2, not 3
    with pytest.raises(AssertionError):
        flash_attention(q, k, v, bad, causal=True)
