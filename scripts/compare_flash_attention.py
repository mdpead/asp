import torch

import triton

from src.kernels import flash_attention, flash_attention_forward


def bench_ms(fn, warmup=10, rep=30) -> float:
    # Light benchmarking: short warmup/rep windows (ms) keep GPU load low.
    ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    assert isinstance(ms, float)
    return ms


def sdpa_attention(q, k, v, causal=False):
    """Torch's built-in fused scaled-dot-product attention."""
    return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=causal)


def eager_attention(q, k, v, causal=False):
    """Manual, unfused reference: softmax(QK^T / sqrt(d)) @ V."""
    scale = 1.0 / q.shape[-1] ** 0.5
    s = torch.matmul(q, k.transpose(-2, -1)) * scale
    if causal:
        seqs = q.shape[-2]
        mask = torch.tril(torch.ones(seqs, seqs, device=q.device, dtype=torch.bool))
        s = s.masked_fill(~mask, float("-inf"))
    p = torch.softmax(s, dim=-1)
    return torch.matmul(p, v)


def compare(dtype: torch.dtype = torch.float32):
    BATCHES, HEADS, HEAD_DIMS = 4, 4, 64
    seqs = [128, 256, 512, 1024]

    print(f"GPU: {torch.cuda.get_device_name(0)}, dtype: {dtype}")
    print(
        f"{'seq':>6} | {'causal':>6} | {'triton(ms)':>10} | {'sdpa(ms)':>9} | {'eager(ms)':>9} | "
        f"{'tri TFLOPS':>10} | {'x sdpa':>7} | {'x eager':>7} | "
        f"{'diff sdpa':>9} | {'diff eager':>10} | match"
    )
    print("-" * 125)

    for causal in (False, True):
        for s in seqs:
            q = torch.randn((BATCHES, HEADS, s, HEAD_DIMS), device="cuda", dtype=dtype)
            k = torch.randn((BATCHES, HEADS, s, HEAD_DIMS), device="cuda", dtype=dtype)
            v = torch.randn((BATCHES, HEADS, s, HEAD_DIMS), device="cuda", dtype=dtype)

            o_triton, _ = flash_attention_forward(q, k, v, causal=causal)
            o_sdpa = sdpa_attention(q, k, v, causal)
            o_eager = eager_attention(q, k, v, causal)

            diff_sdpa = (o_triton - o_sdpa).abs().max().item()
            diff_eager = (o_triton - o_eager).abs().max().item()
            match = (
                "✅"
                if torch.allclose(o_triton, o_sdpa, atol=1e-2, rtol=1e-2)
                and torch.allclose(o_triton, o_eager, atol=1e-2, rtol=1e-2)
                else "❌"
            )

            ms_triton = bench_ms(
                lambda q=q, k=k, v=v: flash_attention_forward(q, k, v, causal=causal)
            )
            ms_sdpa = bench_ms(lambda q=q, k=k, v=v: sdpa_attention(q, k, v, causal))
            ms_eager = bench_ms(lambda q=q, k=k, v=v: eager_attention(q, k, v, causal))

            # QK^T and softmax @ V each cost 2*BATCHES*HEADS*s*s*HEAD_DIMS flops;
            # causal masking halves the useful work (roughly, ignoring the diagonal)
            flops = 4 * BATCHES * HEADS * s * s * HEAD_DIMS
            if causal:
                flops //= 2
            tflops_triton = flops / (ms_triton * 1e-3) / 1e12

            print(
                f"{s:>6} | {str(causal):>6} | {ms_triton:>10.3f} | {ms_sdpa:>9.3f} | {ms_eager:>9.3f} | "
                f"{tflops_triton:>10.2f} | {ms_sdpa / ms_triton:>6.2f}x | {ms_eager / ms_triton:>6.2f}x | "
                f"{diff_sdpa:>9.2e} | {diff_eager:>10.2e} | {match}"
            )

    # Backward correctness: our autograd.Function vs both references
    print("\nBackward (grad max-abs diff)")
    print(
        f"{'seq':>6} | {'causal':>6} | {'dq sdpa':>9} | {'dk sdpa':>9} | {'dv sdpa':>9} | "
        f"{'dq eager':>9} | {'dk eager':>9} | {'dv eager':>9} | match"
    )
    print("-" * 105)

    for causal in (False, True):
        for s in seqs:
            q = torch.randn(
                (BATCHES, HEADS, s, HEAD_DIMS), device="cuda", dtype=dtype, requires_grad=True
            )
            k = torch.randn(
                (BATCHES, HEADS, s, HEAD_DIMS), device="cuda", dtype=dtype, requires_grad=True
            )
            v = torch.randn(
                (BATCHES, HEADS, s, HEAD_DIMS), device="cuda", dtype=dtype, requires_grad=True
            )
            do = torch.randn((BATCHES, HEADS, s, HEAD_DIMS), device="cuda", dtype=dtype)

            # Triton grads via the public wrapper
            o_triton = flash_attention(q, k, v, causal=causal)
            dq_t, dk_t, dv_t = torch.autograd.grad(o_triton, (q, k, v), do)

            # Reference grads (same leaves, separate graphs)
            dq_s, dk_s, dv_s = torch.autograd.grad(sdpa_attention(q, k, v, causal), (q, k, v), do)
            dq_e, dk_e, dv_e = torch.autograd.grad(eager_attention(q, k, v, causal), (q, k, v), do)

            def md(a, b):
                return (a - b).abs().max().item()

            ok = all(
                torch.allclose(a, b, atol=1e-2, rtol=1e-2)
                for a, b in (
                    (dq_t, dq_s),
                    (dk_t, dk_s),
                    (dv_t, dv_s),
                    (dq_t, dq_e),
                    (dk_t, dk_e),
                    (dv_t, dv_e),
                )
            )
            match = "✅" if ok else "❌"
            print(
                f"{s:>6} | {str(causal):>6} | "
                f"{md(dq_t, dq_s):>9.2e} | {md(dk_t, dk_s):>9.2e} | {md(dv_t, dv_s):>9.2e} | "
                f"{md(dq_t, dq_e):>9.2e} | {md(dk_t, dk_e):>9.2e} | {md(dv_t, dv_e):>9.2e} | {match}"
            )


if __name__ == "__main__":
    compare()
