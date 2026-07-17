import torch

import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_Q_SEQ": 32, "BLOCK_SIZE_KV_SEQ": 32}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_SIZE_Q_SEQ": 64, "BLOCK_SIZE_KV_SEQ": 32}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_SIZE_Q_SEQ": 64, "BLOCK_SIZE_KV_SEQ": 64}, num_warps=4, num_stages=4),
        triton.Config(
            {"BLOCK_SIZE_Q_SEQ": 128, "BLOCK_SIZE_KV_SEQ": 64}, num_warps=8, num_stages=3
        ),
        triton.Config(
            {"BLOCK_SIZE_Q_SEQ": 64, "BLOCK_SIZE_KV_SEQ": 128}, num_warps=8, num_stages=3
        ),
        triton.Config(
            {"BLOCK_SIZE_Q_SEQ": 128, "BLOCK_SIZE_KV_SEQ": 128}, num_warps=8, num_stages=3
        ),
    ],
    key=["BATCHES", "HEADS", "SEQS", "HEAD_DIMS", "causal"],
)
@triton.jit
def flash_attention_forward_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    l_ptr,
    causal: tl.constexpr,
    BATCHES,
    HEADS,
    SEQS,
    HEAD_DIMS: tl.constexpr,
    stride_qb,
    stride_qh,
    stride_qs,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_ks,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vs,
    stride_vd,
    stride_ob,
    stride_oh,
    stride_os,
    stride_od,
    stride_lb,
    stride_lh,
    stride_ls,
    BLOCK_SIZE_Q_SEQ: tl.constexpr,
    BLOCK_SIZE_KV_SEQ: tl.constexpr,
):
    # Calculate constant offsets and masks
    pid_b = tl.program_id(axis=0) // HEADS
    pid_h = tl.program_id(axis=0) % HEADS
    pid_s = tl.program_id(axis=1)

    # Load Q tile
    offsets_qb = pid_b
    offsets_qh = pid_h
    offsets_qs = pid_s * BLOCK_SIZE_Q_SEQ + tl.arange(0, BLOCK_SIZE_Q_SEQ)
    offsets_qd = tl.arange(0, HEAD_DIMS)

    q_tile_ptrs = q_ptr + (
        offsets_qb * stride_qb
        + offsets_qh * stride_qh
        + offsets_qs[:, None] * stride_qs
        + offsets_qd[None, :] * stride_qd
    )
    q_tile_mask_s = offsets_qs < SEQS
    q_tile_mask = q_tile_mask_s[:, None]
    q_tile = tl.load(q_tile_ptrs, q_tile_mask, other=0.0)

    # Loop over KV tiles
    s_max = tl.full((BLOCK_SIZE_Q_SEQ, 1), float("-inf"), dtype=tl.float32)
    s_sum = tl.zeros((BLOCK_SIZE_Q_SEQ, 1), dtype=tl.float32)
    o_tile_weights = tl.zeros((BLOCK_SIZE_Q_SEQ, HEAD_DIMS), dtype=tl.float32)
    q_tile = q_tile * (1.0 / HEAD_DIMS**0.5)  # Scale q by the root dim once

    # Figure out how many kv tiles are needed depending on whether it's causal
    if causal:
        kv_tiles = min(
            tl.cdiv((pid_s + 1) * BLOCK_SIZE_Q_SEQ, BLOCK_SIZE_KV_SEQ),
            tl.cdiv(SEQS, BLOCK_SIZE_KV_SEQ),
        )
    else:
        kv_tiles = tl.cdiv(SEQS, BLOCK_SIZE_KV_SEQ)
    for tile_idx in range(0, kv_tiles):
        # Load K tile
        offsets_kb = pid_b
        offsets_kh = pid_h
        offsets_ks = tile_idx * BLOCK_SIZE_KV_SEQ + tl.arange(0, BLOCK_SIZE_KV_SEQ)
        offsets_kd = tl.arange(0, HEAD_DIMS)

        k_tile_ptrs = k_ptr + (
            offsets_kb * stride_kb
            + offsets_kh * stride_kh
            + offsets_ks[:, None] * stride_ks
            + offsets_kd[None, :] * stride_kd
        )
        k_tile_mask = (offsets_ks < SEQS)[:, None]
        k_tile = tl.load(k_tile_ptrs, k_tile_mask, other=0.0)

        # Load V tile
        offsets_vb = pid_b
        offsets_vh = pid_h
        offsets_vs = tile_idx * BLOCK_SIZE_KV_SEQ + tl.arange(0, BLOCK_SIZE_KV_SEQ)
        offsets_vd = tl.arange(0, HEAD_DIMS)

        v_tile_ptrs = v_ptr + (
            offsets_vb * stride_vb
            + offsets_vh * stride_vh
            + offsets_vs[:, None] * stride_vs
            + offsets_vd[None, :] * stride_vd
        )
        v_tile_mask = (offsets_vs < SEQS)[:, None]
        v_tile = tl.load(v_tile_ptrs, v_tile_mask, other=0.0)

        # Compute scores for q,k,v tiles
        s_tile_raw = tl.dot(q_tile, k_tile.T)
        if causal:
            causal_mask = offsets_ks[None, :] <= offsets_qs[:, None]
            s_tile_raw = tl.where(causal_mask, s_tile_raw, float("-inf"))
        else:
            s_tile_raw = tl.where((offsets_ks < SEQS)[None, :], s_tile_raw, float("-inf"))
        s_tile_max = tl.max(s_tile_raw, axis=-1, keep_dims=True)

        # Scale old scores
        s_max_new = tl.maximum(s_max, s_tile_max)
        weight_correction = tl.exp(s_max - s_max_new)
        o_tile_weights = o_tile_weights * weight_correction
        s_sum = s_sum * weight_correction

        # Update accumulators
        s_tile_weights = tl.exp(s_tile_raw - s_max_new)
        s_max = s_max_new
        s_sum = s_sum + tl.sum(s_tile_weights, axis=-1, keep_dims=True)

        # Output
        o_tile_partial_weights = tl.dot(s_tile_weights.to(v_tile.dtype), v_tile)
        o_tile_weights = o_tile_weights + o_tile_partial_weights

    # Apply final normalisation
    o_tile = (o_tile_weights / s_sum).to(o_ptr.dtype.element_ty)

    # Write out o_tile
    offsets_ob = pid_b
    offsets_oh = pid_h
    offsets_os = pid_s * BLOCK_SIZE_Q_SEQ + tl.arange(0, BLOCK_SIZE_Q_SEQ)
    offsets_od = tl.arange(0, HEAD_DIMS)

    o_tile_ptrs = o_ptr + (
        offsets_ob * stride_ob
        + offsets_oh * stride_oh
        + offsets_os[:, None] * stride_os
        + offsets_od[None, :] * stride_od
    )
    o_tile_mask_s = offsets_os < SEQS
    o_tile_mask = o_tile_mask_s[:, None]
    tl.store(o_tile_ptrs, o_tile, o_tile_mask)

    # Write out row-wise log s_sum
    l_tile = s_max + tl.log(s_sum)

    offsets_lb = pid_b
    offsets_lh = pid_h
    offsets_ls = pid_s * BLOCK_SIZE_Q_SEQ + tl.arange(0, BLOCK_SIZE_Q_SEQ)

    l_tile_ptrs = l_ptr + (
        offsets_lb * stride_lb + offsets_lh * stride_lh + offsets_ls[:, None] * stride_ls
    )
    l_tile_mask = (offsets_ls < SEQS)[:, None]

    tl.store(l_tile_ptrs, l_tile, l_tile_mask)


def flash_attention_forward(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    assert q.is_cuda and k.is_cuda and v.is_cuda

    BATCHES, HEADS, SEQS, HEAD_DIMS = q.shape
    o = torch.empty_like(q)
    l = torch.empty((BATCHES, HEADS, SEQS), dtype=torch.float32, device=q.device)

    stride_qb, stride_qh, stride_qs, stride_qd = q.stride()
    stride_kb, stride_kh, stride_ks, stride_kd = k.stride()
    stride_vb, stride_vh, stride_vs, stride_vd = v.stride()
    stride_ob, stride_oh, stride_os, stride_od = o.stride()
    stride_lb, stride_lh, stride_ls = l.stride()

    grid = lambda meta: (BATCHES * HEADS, triton.cdiv(SEQS, meta["BLOCK_SIZE_Q_SEQ"]))
    flash_attention_forward_kernel[grid](
        q,
        k,
        v,
        o,
        l,
        causal,
        BATCHES,
        HEADS,
        SEQS,
        HEAD_DIMS,
        stride_qb,
        stride_qh,
        stride_qs,
        stride_qd,
        stride_kb,
        stride_kh,
        stride_ks,
        stride_kd,
        stride_vb,
        stride_vh,
        stride_vs,
        stride_vd,
        stride_ob,
        stride_oh,
        stride_os,
        stride_od,
        stride_lb,
        stride_lh,
        stride_ls,
    )

    return o, l


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_Q_SEQ": 32}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_SIZE_Q_SEQ": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE_Q_SEQ": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE_Q_SEQ": 256}, num_warps=8, num_stages=2),
    ],
    key=["BATCHES", "HEADS", "SEQS", "HEAD_DIMS"],
)
@triton.jit
def flash_attention_backward_kernel_d(
    o_ptr,
    do_ptr,
    d_ptr,
    BATCHES,
    HEADS,
    SEQS,
    HEAD_DIMS: tl.constexpr,
    stride_ob,
    stride_oh,
    stride_os,
    stride_od,
    stride_dob,
    stride_doh,
    stride_dos,
    stride_dod,
    stride_db,
    stride_dh,
    stride_ds,
    BLOCK_SIZE_Q_SEQ: tl.constexpr,
):
    # Calculate constant offsets and masks
    pid_b = tl.program_id(axis=0) // HEADS
    pid_h = tl.program_id(axis=0) % HEADS
    pid_s = tl.program_id(axis=1)

    # Load O tile
    offsets_ob = pid_b
    offsets_oh = pid_h
    offsets_os = pid_s * BLOCK_SIZE_Q_SEQ + tl.arange(0, BLOCK_SIZE_Q_SEQ)
    offsets_od = tl.arange(0, HEAD_DIMS)

    o_tile_ptrs = o_ptr + (
        offsets_ob * stride_ob
        + offsets_oh * stride_oh
        + offsets_os[:, None] * stride_os
        + offsets_od[None, :] * stride_od
    )
    o_tile_mask_s = offsets_os < SEQS
    o_tile_mask = o_tile_mask_s[:, None]
    o_tile = tl.load(o_tile_ptrs, o_tile_mask, other=0.0)

    # Load dO tile
    offsets_dob = pid_b
    offsets_doh = pid_h
    offsets_dos = pid_s * BLOCK_SIZE_Q_SEQ + tl.arange(0, BLOCK_SIZE_Q_SEQ)
    offsets_dod = tl.arange(0, HEAD_DIMS)

    do_tile_ptrs = do_ptr + (
        offsets_dob * stride_dob
        + offsets_doh * stride_doh
        + offsets_dos[:, None] * stride_dos
        + offsets_dod[None, :] * stride_dod
    )
    do_tile_mask_s = offsets_dos < SEQS
    do_tile_mask = do_tile_mask_s[:, None]
    do_tile = tl.load(do_tile_ptrs, do_tile_mask, other=0.0)

    # Calculate d tile
    d_tile = tl.sum(o_tile * do_tile, axis=-1)

    # Store d tile
    offsets_db = pid_b
    offsets_dh = pid_h
    offsets_ds = pid_s * BLOCK_SIZE_Q_SEQ + tl.arange(0, BLOCK_SIZE_Q_SEQ)

    d_tile_mask = offsets_ds < SEQS

    d_ptrs = d_ptr + offsets_db * stride_db + offsets_dh * stride_dh + offsets_ds * stride_ds
    tl.store(d_ptrs, d_tile, d_tile_mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_Q_SEQ": 32, "BLOCK_SIZE_KV_SEQ": 32}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_SIZE_Q_SEQ": 64, "BLOCK_SIZE_KV_SEQ": 32}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_SIZE_Q_SEQ": 64, "BLOCK_SIZE_KV_SEQ": 64}, num_warps=4, num_stages=4),
        triton.Config(
            {"BLOCK_SIZE_Q_SEQ": 128, "BLOCK_SIZE_KV_SEQ": 64}, num_warps=8, num_stages=3
        ),
        triton.Config(
            {"BLOCK_SIZE_Q_SEQ": 64, "BLOCK_SIZE_KV_SEQ": 128}, num_warps=8, num_stages=3
        ),
        triton.Config(
            {"BLOCK_SIZE_Q_SEQ": 128, "BLOCK_SIZE_KV_SEQ": 128}, num_warps=8, num_stages=3
        ),
    ],
    key=["BATCHES", "HEADS", "SEQS", "HEAD_DIMS", "causal"],
)
@triton.jit
def flash_attention_backward_kernel_dq(
    q_ptr,
    k_ptr,
    v_ptr,
    l_ptr,
    do_ptr,
    dq_ptr,
    d_ptr,
    causal: tl.constexpr,
    BATCHES,
    HEADS,
    SEQS,
    HEAD_DIMS: tl.constexpr,
    stride_qb,
    stride_qh,
    stride_qs,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_ks,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vs,
    stride_vd,
    stride_lb,
    stride_lh,
    stride_ls,
    stride_dob,
    stride_doh,
    stride_dos,
    stride_dod,
    stride_dqb,
    stride_dqh,
    stride_dqs,
    stride_dqd,
    stride_db,
    stride_dh,
    stride_ds,
    BLOCK_SIZE_Q_SEQ: tl.constexpr,
    BLOCK_SIZE_KV_SEQ: tl.constexpr,
):
    # Calculate constant offsets and masks
    pid_b = tl.program_id(axis=0) // HEADS
    pid_h = tl.program_id(axis=0) % HEADS
    pid_s = tl.program_id(axis=1)

    # Load Q tile
    offsets_qb = pid_b
    offsets_qh = pid_h
    offsets_qs = pid_s * BLOCK_SIZE_Q_SEQ + tl.arange(0, BLOCK_SIZE_Q_SEQ)
    offsets_qd = tl.arange(0, HEAD_DIMS)

    q_tile_ptrs = q_ptr + (
        offsets_qb * stride_qb
        + offsets_qh * stride_qh
        + offsets_qs[:, None] * stride_qs
        + offsets_qd[None, :] * stride_qd
    )
    q_tile_mask_s = offsets_qs < SEQS
    q_tile_mask = q_tile_mask_s[:, None]
    q_tile = tl.load(q_tile_ptrs, q_tile_mask, other=0.0)

    # Load L tile
    offsets_lb = pid_b
    offsets_lh = pid_h
    offsets_ls = pid_s * BLOCK_SIZE_Q_SEQ + tl.arange(0, BLOCK_SIZE_Q_SEQ)

    l_tile_ptrs = l_ptr + (
        offsets_lb * stride_lb + offsets_lh * stride_lh + offsets_ls[:, None] * stride_ls
    )
    l_tile_mask = (offsets_ls < SEQS)[:, None]
    l_tile = tl.load(l_tile_ptrs, l_tile_mask, other=0.0)

    # Load dO tile
    offsets_dob = pid_b
    offsets_doh = pid_h
    offsets_dos = pid_s * BLOCK_SIZE_Q_SEQ + tl.arange(0, BLOCK_SIZE_Q_SEQ)
    offsets_dod = tl.arange(0, HEAD_DIMS)

    do_tile_ptrs = do_ptr + (
        offsets_dob * stride_dob
        + offsets_doh * stride_doh
        + offsets_dos[:, None] * stride_dos
        + offsets_dod[None, :] * stride_dod
    )
    do_tile_mask = (offsets_ls < SEQS)[:, None]
    do_tile = tl.load(do_tile_ptrs, do_tile_mask, other=0.0)

    # Load d tile
    offsets_db = pid_b
    offsets_dh = pid_h
    offsets_ds = pid_s * BLOCK_SIZE_Q_SEQ + tl.arange(0, BLOCK_SIZE_Q_SEQ)

    d_tile_ptrs = d_ptr + (
        offsets_db * stride_db + offsets_dh * stride_dh + offsets_ds[:, None] * stride_ds
    )
    d_tile_mask = (offsets_ds < SEQS)[:, None]

    d_tile = tl.load(d_tile_ptrs, d_tile_mask, other=0.0)

    # Loop over KV tiles
    q_tile = q_tile * (1.0 / HEAD_DIMS**0.5)  # Scale q by the root dim once
    dq_tile = tl.zeros((BLOCK_SIZE_Q_SEQ, HEAD_DIMS), dtype=tl.float32)

    # Figure out how many kv tiles are needed depending on whether it's causal
    if causal:
        kv_tiles = min(
            tl.cdiv((pid_s + 1) * BLOCK_SIZE_Q_SEQ, BLOCK_SIZE_KV_SEQ),
            tl.cdiv(SEQS, BLOCK_SIZE_KV_SEQ),
        )
    else:
        kv_tiles = tl.cdiv(SEQS, BLOCK_SIZE_KV_SEQ)
    for tile_idx in range(0, kv_tiles):
        # Load K tile
        offsets_kb = pid_b
        offsets_kh = pid_h
        offsets_ks = tile_idx * BLOCK_SIZE_KV_SEQ + tl.arange(0, BLOCK_SIZE_KV_SEQ)
        offsets_kd = tl.arange(0, HEAD_DIMS)

        k_tile_ptrs = k_ptr + (
            offsets_kb * stride_kb
            + offsets_kh * stride_kh
            + offsets_ks[:, None] * stride_ks
            + offsets_kd[None, :] * stride_kd
        )
        k_tile_mask_s = offsets_ks < SEQS
        k_tile_mask = k_tile_mask_s[:, None]
        k_tile = tl.load(k_tile_ptrs, k_tile_mask, other=0.0)

        # Load V tile
        offsets_vb = pid_b
        offsets_vh = pid_h
        offsets_vs = tile_idx * BLOCK_SIZE_KV_SEQ + tl.arange(0, BLOCK_SIZE_KV_SEQ)
        offsets_vd = tl.arange(0, HEAD_DIMS)

        v_tile_ptrs = v_ptr + (
            offsets_vb * stride_vb
            + offsets_vh * stride_vh
            + offsets_vs[:, None] * stride_vs
            + offsets_vd[None, :] * stride_vd
        )
        v_tile_mask_s = offsets_vs < SEQS
        v_tile_mask = v_tile_mask_s[:, None]
        v_tile = tl.load(v_tile_ptrs, v_tile_mask, other=0.0)

        # Compute scores for q,k,v tiles
        s_tile_raw = tl.dot(q_tile, k_tile.T)
        if causal:
            causal_mask = offsets_ks[None, :] <= offsets_qs[:, None]
            s_tile_raw = tl.where(causal_mask, s_tile_raw, float("-inf"))
        else:
            s_tile_raw = tl.where((offsets_ks < SEQS)[None, :], s_tile_raw, float("-inf"))

        # Compute through to dq_tile
        p_tile = tl.exp(s_tile_raw - l_tile)  # Use l_tile to get p directly
        dp_tile = tl.dot(do_tile, v_tile.T)
        ds_tile = p_tile * (dp_tile - d_tile)
        dq_tile_partial = (1.0 / HEAD_DIMS**0.5) * tl.dot(ds_tile.to(k_tile.dtype), k_tile)

        # Accumulate
        dq_tile = dq_tile + dq_tile_partial

    # Store dq tile
    offsets_dqb = pid_b
    offsets_dqh = pid_h
    offsets_dqs = pid_s * BLOCK_SIZE_Q_SEQ + tl.arange(0, BLOCK_SIZE_Q_SEQ)
    offsets_dqd = tl.arange(0, HEAD_DIMS)

    dq_mask = (offsets_dqs < SEQS)[:, None]

    dq_ptrs = dq_ptr + (
        offsets_dqb * stride_dqb
        + offsets_dqh * stride_dqh
        + offsets_dqs[:, None] * stride_dqs
        + offsets_dqd[None, :] * stride_dqd
    )
    tl.store(dq_ptrs, dq_tile, dq_mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_Q_SEQ": 32, "BLOCK_SIZE_KV_SEQ": 32}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_SIZE_Q_SEQ": 64, "BLOCK_SIZE_KV_SEQ": 32}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_SIZE_Q_SEQ": 64, "BLOCK_SIZE_KV_SEQ": 64}, num_warps=4, num_stages=4),
        triton.Config(
            {"BLOCK_SIZE_Q_SEQ": 128, "BLOCK_SIZE_KV_SEQ": 64}, num_warps=8, num_stages=3
        ),
        triton.Config(
            {"BLOCK_SIZE_Q_SEQ": 64, "BLOCK_SIZE_KV_SEQ": 128}, num_warps=8, num_stages=3
        ),
        triton.Config(
            {"BLOCK_SIZE_Q_SEQ": 128, "BLOCK_SIZE_KV_SEQ": 128}, num_warps=8, num_stages=3
        ),
    ],
    key=["BATCHES", "HEADS", "SEQS", "HEAD_DIMS", "causal"],
)
@triton.jit
def flash_attention_backward_kernel_dkdv(
    q_ptr,
    k_ptr,
    v_ptr,
    l_ptr,
    do_ptr,
    dk_ptr,
    dv_ptr,
    d_ptr,
    causal: tl.constexpr,
    BATCHES,
    HEADS,
    SEQS,
    HEAD_DIMS: tl.constexpr,
    stride_qb,
    stride_qh,
    stride_qs,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_ks,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vs,
    stride_vd,
    stride_lb,
    stride_lh,
    stride_ls,
    stride_dob,
    stride_doh,
    stride_dos,
    stride_dod,
    stride_dkb,
    stride_dkh,
    stride_dks,
    stride_dkd,
    stride_dvb,
    stride_dvh,
    stride_dvs,
    stride_dvd,
    stride_db,
    stride_dh,
    stride_ds,
    BLOCK_SIZE_Q_SEQ: tl.constexpr,
    BLOCK_SIZE_KV_SEQ: tl.constexpr,
):

    # Calculate constant offsets and masks
    pid_b = tl.program_id(axis=0) // HEADS
    pid_h = tl.program_id(axis=0) % HEADS
    pid_s = tl.program_id(axis=1)

    # Load K tile
    offsets_kb = pid_b
    offsets_kh = pid_h
    offsets_ks = pid_s * BLOCK_SIZE_KV_SEQ + tl.arange(0, BLOCK_SIZE_KV_SEQ)
    offsets_kd = tl.arange(0, HEAD_DIMS)

    k_tile_ptrs = k_ptr + (
        offsets_kb * stride_kb
        + offsets_kh * stride_kh
        + offsets_ks[:, None] * stride_ks
        + offsets_kd[None, :] * stride_kd
    )
    k_tile_mask_s = offsets_ks < SEQS
    k_tile_mask = k_tile_mask_s[:, None]
    k_tile = tl.load(k_tile_ptrs, k_tile_mask, other=0.0)

    # Load V tile
    offsets_vb = pid_b
    offsets_vh = pid_h
    offsets_vs = pid_s * BLOCK_SIZE_KV_SEQ + tl.arange(0, BLOCK_SIZE_KV_SEQ)
    offsets_vd = tl.arange(0, HEAD_DIMS)

    v_tile_ptrs = v_ptr + (
        offsets_vb * stride_vb
        + offsets_vh * stride_vh
        + offsets_vs[:, None] * stride_vs
        + offsets_vd[None, :] * stride_vd
    )
    v_tile_mask_s = offsets_vs < SEQS
    v_tile_mask = v_tile_mask_s[:, None]
    v_tile = tl.load(v_tile_ptrs, v_tile_mask, other=0.0)

    # Loop over Q tiles
    dk_tile = tl.zeros((BLOCK_SIZE_KV_SEQ, HEAD_DIMS), dtype=tl.float32)
    dv_tile = tl.zeros((BLOCK_SIZE_KV_SEQ, HEAD_DIMS), dtype=tl.float32)

    # Figure out how many q tiles are needed depending on whether it's causal
    if causal:
        q_tile_start_idx = pid_s * BLOCK_SIZE_KV_SEQ // BLOCK_SIZE_Q_SEQ
    else:
        q_tile_start_idx = 0
    q_tile_end_idx = tl.cdiv(SEQS, BLOCK_SIZE_Q_SEQ)

    for tile_idx in range(
        q_tile_start_idx,
        q_tile_end_idx,
    ):
        # Load Q tile
        offsets_qb = pid_b
        offsets_qh = pid_h
        offsets_qs = tile_idx * BLOCK_SIZE_Q_SEQ + tl.arange(0, BLOCK_SIZE_Q_SEQ)
        offsets_qd = tl.arange(0, HEAD_DIMS)

        q_tile_ptrs = q_ptr + (
            offsets_qb * stride_qb
            + offsets_qh * stride_qh
            + offsets_qs[:, None] * stride_qs
            + offsets_qd[None, :] * stride_qd
        )
        q_tile_mask = (offsets_qs < SEQS)[:, None]
        q_tile = tl.load(q_tile_ptrs, q_tile_mask, other=0.0)

        # Load L tile
        offsets_lb = pid_b
        offsets_lh = pid_h
        offsets_ls = tile_idx * BLOCK_SIZE_Q_SEQ + tl.arange(0, BLOCK_SIZE_Q_SEQ)

        l_tile_ptrs = l_ptr + (
            offsets_lb * stride_lb + offsets_lh * stride_lh + offsets_ls[:, None] * stride_ls
        )
        l_tile_mask = (offsets_ls < SEQS)[:, None]
        l_tile = tl.load(l_tile_ptrs, l_tile_mask, other=0.0)

        # Load dO tile
        offsets_dob = pid_b
        offsets_doh = pid_h
        offsets_dos = tile_idx * BLOCK_SIZE_Q_SEQ + tl.arange(0, BLOCK_SIZE_Q_SEQ)
        offsets_dod = tl.arange(0, HEAD_DIMS)

        do_tile_ptrs = do_ptr + (
            offsets_dob * stride_dob
            + offsets_doh * stride_doh
            + offsets_dos[:, None] * stride_dos
            + offsets_dod[None, :] * stride_dod
        )
        do_tile_mask = (offsets_ls < SEQS)[:, None]
        do_tile = tl.load(do_tile_ptrs, do_tile_mask, other=0.0)

        # Load d tile
        offsets_db = pid_b
        offsets_dh = pid_h
        offsets_ds = tile_idx * BLOCK_SIZE_Q_SEQ + tl.arange(0, BLOCK_SIZE_Q_SEQ)

        d_tile_ptrs = d_ptr + (
            offsets_db * stride_db + offsets_dh * stride_dh + offsets_ds[:, None] * stride_ds
        )
        d_tile_mask = (offsets_ds < SEQS)[:, None]
        d_tile = tl.load(d_tile_ptrs, d_tile_mask, other=0.0)

        # Compute scores for q,k,v tiles
        s_tile_raw = (1.0 / HEAD_DIMS**0.5) * tl.dot(q_tile, k_tile.T)
        if causal:
            causal_mask = offsets_ks[None, :] <= offsets_qs[:, None]
            s_tile_raw = tl.where(causal_mask, s_tile_raw, float("-inf"))
        else:
            s_tile_raw = tl.where((offsets_ks < SEQS)[None, :], s_tile_raw, float("-inf"))

        # Compute through to dk/dv tiles
        p_tile = tl.exp(s_tile_raw - l_tile)  # Use l_tile to get p directly
        dp_tile = tl.dot(do_tile, v_tile.T)
        ds_tile = p_tile * (dp_tile - d_tile)
        dk_tile_partial = (1.0 / HEAD_DIMS**0.5) * tl.dot(ds_tile.T.to(q_tile.dtype), q_tile)
        dv_tile_partial = tl.dot(p_tile.T.to(do_tile.dtype), do_tile)  # dV = P^T dO, no tau

        # Accumulate
        dk_tile = dk_tile + dk_tile_partial
        dv_tile = dv_tile + dv_tile_partial

    # Store dk tile
    offsets_dkb = pid_b
    offsets_dkh = pid_h
    offsets_dks = pid_s * BLOCK_SIZE_KV_SEQ + tl.arange(0, BLOCK_SIZE_KV_SEQ)
    offsets_dkd = tl.arange(0, HEAD_DIMS)

    dk_mask = (offsets_dks < SEQS)[:, None]

    dk_ptrs = dk_ptr + (
        offsets_dkb * stride_dkb
        + offsets_dkh * stride_dkh
        + offsets_dks[:, None] * stride_dks
        + offsets_dkd[None, :] * stride_dkd
    )
    tl.store(dk_ptrs, dk_tile, dk_mask)

    # Store dv tile
    offsets_dvb = pid_b
    offsets_dvh = pid_h
    offsets_dvs = pid_s * BLOCK_SIZE_KV_SEQ + tl.arange(0, BLOCK_SIZE_KV_SEQ)
    offsets_dvd = tl.arange(0, HEAD_DIMS)

    dv_mask = (offsets_dvs < SEQS)[:, None]

    dv_ptrs = dv_ptr + (
        offsets_dvb * stride_dvb
        + offsets_dvh * stride_dvh
        + offsets_dvs[:, None] * stride_dvs
        + offsets_dvd[None, :] * stride_dvd
    )
    tl.store(dv_ptrs, dv_tile, dv_mask)


def flash_attention_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    l: torch.Tensor,
    do: torch.Tensor,
    causal: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert q.is_cuda and k.is_cuda and v.is_cuda

    BATCHES, HEADS, SEQS, HEAD_DIMS = q.shape
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    d = torch.empty((BATCHES, HEADS, SEQS), dtype=torch.float32, device=dq.device)

    stride_qb, stride_qh, stride_qs, stride_qd = q.stride()
    stride_kb, stride_kh, stride_ks, stride_kd = k.stride()
    stride_vb, stride_vh, stride_vs, stride_vd = v.stride()
    stride_ob, stride_oh, stride_os, stride_od = o.stride()
    stride_lb, stride_lh, stride_ls = l.stride()
    stride_dob, stride_doh, stride_dos, stride_dod = do.stride()
    stride_dqb, stride_dqh, stride_dqs, stride_dqd = dq.stride()
    stride_dkb, stride_dkh, stride_dks, stride_dkd = dk.stride()
    stride_dvb, stride_dvh, stride_dvs, stride_dvd = dv.stride()
    stride_db, stride_dh, stride_ds = d.stride()

    # Preprocess d
    grid_d = lambda meta: (BATCHES * HEADS, triton.cdiv(SEQS, meta["BLOCK_SIZE_Q_SEQ"]))
    flash_attention_backward_kernel_d[grid_d](
        o,
        do,
        d,
        BATCHES,
        HEADS,
        SEQS,
        HEAD_DIMS,
        stride_ob,
        stride_oh,
        stride_os,
        stride_od,
        stride_dob,
        stride_doh,
        stride_dos,
        stride_dod,
        stride_db,
        stride_dh,
        stride_ds,
    )

    # Calculate dq, parallelising over seqs
    grid_dq = lambda meta: (BATCHES * HEADS, triton.cdiv(SEQS, meta["BLOCK_SIZE_Q_SEQ"]))
    flash_attention_backward_kernel_dq[grid_dq](
        q,
        k,
        v,
        l,
        do,
        dq,
        d,
        causal,
        BATCHES,
        HEADS,
        SEQS,
        HEAD_DIMS,
        stride_qb,
        stride_qh,
        stride_qs,
        stride_qd,
        stride_kb,
        stride_kh,
        stride_ks,
        stride_kd,
        stride_vb,
        stride_vh,
        stride_vs,
        stride_vd,
        stride_lb,
        stride_lh,
        stride_ls,
        stride_dob,
        stride_doh,
        stride_dos,
        stride_dod,
        stride_dqb,
        stride_dqh,
        stride_dqs,
        stride_dqd,
        stride_db,
        stride_dh,
        stride_ds,
    )

    # Calculate dk, dv parallelising over kv seq tiles
    grid_dkdv = lambda meta: (BATCHES * HEADS, triton.cdiv(SEQS, meta["BLOCK_SIZE_KV_SEQ"]))
    flash_attention_backward_kernel_dkdv[grid_dkdv](
        q,
        k,
        v,
        l,
        do,
        dk,
        dv,
        d,
        causal,
        BATCHES,
        HEADS,
        SEQS,
        HEAD_DIMS,
        stride_qb,
        stride_qh,
        stride_qs,
        stride_qd,
        stride_kb,
        stride_kh,
        stride_ks,
        stride_kd,
        stride_vb,
        stride_vh,
        stride_vs,
        stride_vd,
        stride_lb,
        stride_lh,
        stride_ls,
        stride_dob,
        stride_doh,
        stride_dos,
        stride_dod,
        stride_dkb,
        stride_dkh,
        stride_dks,
        stride_dkd,
        stride_dvb,
        stride_dvh,
        stride_dvs,
        stride_dvd,
        stride_db,
        stride_dh,
        stride_ds,
    )

    return dq, dk, dv


class FlashAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal):
        o, l = flash_attention_forward(q, k, v, causal)
        ctx.save_for_backward(q, k, v, o, l)
        ctx.causal = causal
        return o

    @staticmethod
    def backward(ctx, *grad_outputs):
        (do,) = grad_outputs
        q, k, v, o, l = ctx.saved_tensors
        causal = ctx.causal
        dq, dk, dv = flash_attention_backward(q, k, v, o, l, do, causal)
        return dq, dk, dv, None


def flash_attention(q, k, v, causal=False):
    return FlashAttentionFunction.apply(q, k, v, causal)
