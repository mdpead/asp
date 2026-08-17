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
