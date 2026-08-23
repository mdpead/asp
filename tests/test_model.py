"""Model-level integration: GQA widths, seq_starts plumbing, gradient flow."""

import pytest
import torch

DEV = "cuda"


def test_forward_shapes(model, tokenizer):
    ids = torch.randint(0, len(tokenizer), (2, 16), device=DEV)
    logits, loss_aux = model(ids)
    assert logits.shape == (2, 16, len(tokenizer))
    assert loss_aux.ndim == 0
    assert not logits.isnan().any()


def test_kv_projection_is_not_expanded(make_model, tokenizer):
    """K/V must stay at num_kv_heads: the kernel maps query heads to groups itself."""
    num_heads, num_kv_heads, d_h = 8, 2, 16
    m = make_model(len(tokenizer), num_heads=num_heads, num_kv_heads=num_kv_heads)
    attn = m.decoder.decoder_layers[0].attention
    assert attn.qkv_proj.out_features == (num_heads + 2 * num_kv_heads) * d_h


def test_max_length_exposed(model):
    """generate_texts reads the context bound off the model rather than being told."""
    assert model.max_length == model.decoder.rope_layer.cos_thetas.shape[0]


def test_default_seq_starts_equals_explicit_zeros(model, tokenizer):
    ids = torch.randint(0, len(tokenizer), (2, 16), device=DEV)
    zeros = torch.zeros(2, dtype=torch.int32, device=DEV)
    a, _ = model(ids)
    b, _ = model(ids, zeros)
    assert torch.equal(a, b)


def test_seq_starts_changes_only_the_padded_row(model, tokenizer):
    ids = torch.randint(0, len(tokenizer), (2, 16), device=DEV)
    base, _ = model(ids, torch.zeros(2, dtype=torch.int32, device=DEV))
    masked, _ = model(ids, torch.tensor([0, 5], dtype=torch.int32, device=DEV))

    assert torch.equal(base[0], masked[0]), "row with seq_start=0 should be untouched"
    assert not torch.equal(base[1], masked[1]), "row with seq_start=5 should differ"


def test_padding_is_invisible_end_to_end(model, tokenizer):
    """A prompt preceded by pad tokens must produce the same final-position logits."""
    pad_id = tokenizer.pad_token_id
    real = torch.randint(0, len(tokenizer), (1, 10), device=DEV)
    pad = 6
    padded = torch.cat(
        [torch.full((1, pad), pad_id, device=DEV, dtype=torch.long), real], dim=1
    )

    a, _ = model(real, torch.zeros(1, dtype=torch.int32, device=DEV))
    b, _ = model(padded, torch.tensor([pad], dtype=torch.int32, device=DEV))

    err = (a[0, -1].float() - b[0, -1].float()).abs().max().item()
    scale = a[0, -1].float().abs().max().item()
    assert err / scale < 1e-2, f"padding shifted logits by rel {err / scale:.4g}"


@pytest.mark.parametrize("num_heads,num_kv_heads", [(4, 4), (8, 2), (4, 1)])
def test_gradients_flow_without_nan(make_model, tokenizer, num_heads, num_kv_heads):
    m = make_model(len(tokenizer), num_heads=num_heads, num_kv_heads=num_kv_heads)
    m.train()
    ids = torch.randint(0, len(tokenizer), (2, 16), device=DEV)

    logits, loss_aux = m(ids)
    (logits.float().pow(2).mean() + loss_aux).backward()

    with_grad = [p for p in m.parameters() if p.grad is not None]
    assert with_grad, "no parameter received a gradient"
    assert not any(p.grad.isnan().any() for p in with_grad)
    assert not any(p.grad.isinf().any() for p in with_grad)


def _reference_moe(moe, x):
    """The obvious implementation: for each expert, mask out its tokens and run them.

    Deliberately naive and slow -- it is the oracle the capacity dispatch has to match, and
    it is the code that dispatch replaced. Only valid where nothing is dropped.
    """
    import torch.nn.functional as F

    router_logits = moe.router(x)
    route_l, route_ind = torch.topk(router_logits, moe.top_k, dim=-1)
    route_p = F.softmax(route_l, dim=-1)

    out = torch.zeros_like(x)
    for i in range(moe.num_experts):
        chose_i = route_ind == i
        probs = (chose_i * route_p).sum(dim=-1)
        mask = torch.any(chose_i, dim=-1)
        out[mask] += probs[mask].unsqueeze(-1) * moe.experts[i](x[mask])
    return out


def test_moe_matches_naive_dispatch_when_nothing_is_dropped(make_model, tokenizer):
    """With capacity above the worst case, seating tokens must change nothing at all."""
    m = make_model(len(tokenizer), capacity_factor=4.0)  # == num_experts, so no drop is possible
    moe = m.decoder.decoder_layers[0].feedforward
    x = torch.randn(2, 16, 64, device=DEV, dtype=torch.float16)

    got, _ = moe(x)
    want = _reference_moe(moe, x)

    assert got.shape == x.shape
    assert torch.equal(got, want), f"max diff {(got - want).abs().max().item():.3e}"


def test_moe_capacity_drops_only_the_overflow(make_model, tokenizer):
    """At capacity 1.0 some assignments are dropped, and only those lose their expert."""
    tight = make_model(len(tokenizer), capacity_factor=1.0)
    moe_tight = tight.decoder.decoder_layers[0].feedforward
    moe_loose = make_model(len(tokenizer), capacity_factor=4.0).decoder.decoder_layers[0].feedforward
    moe_loose.load_state_dict(moe_tight.state_dict())

    x = torch.randn(2, 16, 64, device=DEV, dtype=torch.float16)
    got, _ = moe_tight(x)
    full, _ = moe_loose(x)

    differing = (got != full).any(dim=-1).sum().item()
    assert differing > 0, "capacity 1.0 dropped nothing; the test is not exercising overflow"
    assert differing < got.shape[0] * got.shape[1], "capacity 1.0 dropped every token"
    assert not got.isnan().any()


class _FixedRouter(torch.nn.Module):
    """Router with hand-written logits, so a test can choose the load exactly."""

    def __init__(self, num_experts, collapsed):
        super().__init__()
        self.num_experts, self.collapsed = num_experts, collapsed

    def forward(self, x):
        n, e = x.shape[0], self.num_experts
        logits = torch.full((n, e), -10.0, device=x.device, dtype=x.dtype)
        # Collapsed: every token picks the same two experts. Balanced: the pair each token
        # picks cycles, so all e experts carry an equal share.
        first = torch.zeros(n, dtype=torch.long, device=x.device) if self.collapsed \
            else torch.arange(n, device=x.device) % e
        rows = torch.arange(n, device=x.device)
        logits[rows, first] = 10.0
        logits[rows, (first + 1) % e] = 9.0
        return logits


def test_moe_aux_loss_tracks_imbalance(make_model, tokenizer):
    """Guards the reduction axes: a collapsed router must score worse than a balanced one.

    Reducing over the wrong axes yields a constant, which silently disables load balancing
    for a whole training run rather than failing.
    """
    moe = make_model(len(tokenizer)).decoder.decoder_layers[0].feedforward
    x = torch.randn(4, 32, 64, device=DEV, dtype=torch.float16)

    moe.router = _FixedRouter(moe.num_experts, collapsed=False)
    balanced = moe(x)[1].item()
    moe.router = _FixedRouter(moe.num_experts, collapsed=True)
    collapsed = moe(x)[1].item()

    # Perfect balance scores 1.0; piling top_k experts' worth of load onto top_k experts
    # scores num_experts / top_k.
    assert balanced == pytest.approx(1.0, abs=0.05), f"balanced routing scored {balanced:.4f}"
    assert collapsed == pytest.approx(moe.num_experts / moe.top_k, abs=0.05), \
        f"collapsed routing scored {collapsed:.4f}"
    assert collapsed > balanced
