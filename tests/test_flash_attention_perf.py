"""Throughput of the Triton kernel against torch's fused SDPA and unfused eager attention.

Opt-in (`pytest tests/test_flash_attention_perf.py --benchmark -s`): timings depend on the
GPU and on what else is running, so this reports rather than gates. The one assertion is a
loose floor — it catches a kernel that has stopped being a flash attention (materialising
the full score matrix, losing its autotuned config) rather than a few percent of drift.
"""

import pytest
import torch
import triton

from src.kernels.flash_attention import flash_attention_forward

DEV = "cuda"
BATCHES, HEADS, HEAD_DIMS = 4, 4, 64
SEQS = [128, 256, 512, 1024]

# The eager path materialises seq x seq scores, so it is the loser by construction; SDPA is
# the real bar. Allow the kernel to be this many times slower than SDPA before failing.
MAX_SLOWDOWN_VS_SDPA = 3.0


def _bench_ms(fn) -> float:
    # Short warmup/rep windows keep GPU load low; enough to be stable to a few percent.
    ms = triton.testing.do_bench(fn, warmup=10, rep=30)
    assert isinstance(ms, float)
    return ms


def _sdpa(q, k, v, causal):
    return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=causal)


def _eager(q, k, v, causal):
    """Unfused reference: softmax(QK^T / sqrt(d)) @ V, scores held in memory."""
    scores = (q @ k.transpose(-2, -1)) / q.shape[-1] ** 0.5
    if causal:
        seq = q.shape[-2]
        mask = torch.ones(seq, seq, device=q.device, dtype=torch.bool).tril()
        scores = scores.masked_fill(~mask, float("-inf"))
    return torch.softmax(scores, dim=-1) @ v


@pytest.mark.benchmark
@pytest.mark.parametrize("causal", [False, True])
def test_forward_throughput(causal):
    print(f"\nGPU: {torch.cuda.get_device_name(0)}, causal={causal}, dtype=float16")
    print(
        f"{'seq':>6} | {'triton(ms)':>10} | {'sdpa(ms)':>9} | {'eager(ms)':>9} | "
        f"{'TFLOPS':>7} | {'x sdpa':>7} | {'x eager':>7}"
    )
    print("-" * 74)

    slowdowns = {}
    for seq in SEQS:
        shape = (BATCHES, HEADS, seq, HEAD_DIMS)
        q, k, v = (torch.randn(shape, device=DEV, dtype=torch.float16) for _ in range(3))
        seq_starts = torch.zeros(BATCHES, device=DEV, dtype=torch.int32)

        ms_triton = _bench_ms(lambda: flash_attention_forward(q, k, v, seq_starts, causal))
        ms_sdpa = _bench_ms(lambda: _sdpa(q, k, v, causal))
        ms_eager = _bench_ms(lambda: _eager(q, k, v, causal))

        # QK^T and probs @ V each cost 2*B*H*seq^2*d flops; causal masking roughly halves
        # the useful work (ignoring the diagonal).
        flops = 4 * BATCHES * HEADS * seq * seq * HEAD_DIMS
        if causal:
            flops //= 2
        tflops = flops / (ms_triton * 1e-3) / 1e12
        slowdowns[seq] = ms_triton / ms_sdpa

        print(
            f"{seq:>6} | {ms_triton:>10.3f} | {ms_sdpa:>9.3f} | {ms_eager:>9.3f} | "
            f"{tflops:>7.2f} | {ms_sdpa / ms_triton:>6.2f}x | {ms_eager / ms_triton:>6.2f}x"
        )

    regressed = {s: r for s, r in slowdowns.items() if r > MAX_SLOWDOWN_VS_SDPA}
    assert not regressed, (
        f"kernel is more than {MAX_SLOWDOWN_VS_SDPA}x slower than SDPA at "
        f"{ {s: round(r, 2) for s, r in regressed.items()} }"
    )
