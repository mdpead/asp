from torch import nn
import torch
import math
from src.kernels import flash_attention
from torch.nn import functional as F
from torch import Tensor


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff, bias=False)
        self.linear2 = nn.Linear(d_ff, d_model, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


class SwiGLUFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.up = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        g = F.silu(self.gate(x))
        u = self.up(x)
        u_g = g * u
        o = self.down(u_g)
        return o


class MixtureOfExpertsFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, num_experts, top_k, capacity_factor):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        # Rows each expert reserves, as a multiple of its balanced share. Above 1.0 it buys
        # tolerance to router imbalance with padding it computes and throws away; at 1.0 any
        # imbalance at all costs a token its expert.
        self.capacity_factor = capacity_factor
        self.router = nn.Linear(d_model, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [SwiGLUFeedForward(d_model, d_ff) for _ in range(0, num_experts)]
        )

    def forward(self, x):

        n, d = x.shape[0] * x.shape[1], x.shape[-1]

        x_flat = x.reshape(-1, x.shape[-1]) # (batch * seq = n, d)
        router_logits = self.router(x_flat)
        route_l, route_ind = torch.topk(router_logits, self.top_k, dim=-1)
        route_p = F.softmax(route_l, dim=-1) # (n, k)
        full_p = F.softmax(router_logits, dim=-1, dtype=torch.float32)  # (n, e)

        # Assign seats to each token
        token_expert_oh = F.one_hot(route_ind, self.num_experts) # (n, k, e)
        rank_km = token_expert_oh.permute(1,0,2).reshape(route_ind.shape[0] * self.top_k, -1).cumsum(0) - 1 # (n * k, e)
        token_expert_seat = rank_km.reshape(self.top_k, route_ind.shape[0], self.num_experts).permute(1, 0, 2) # (n, k, e)
        token_choice_seat = token_expert_seat.gather(-1, route_ind.unsqueeze(-1)).squeeze(-1) # (n, k)

        # Capacity
        capacity = int(self.capacity_factor * n * self.top_k/self.num_experts)
        keep = token_choice_seat < capacity
        gate = route_p * keep
        seat_safe = torch.where(keep, token_choice_seat, capacity) # (n, k)

        # Dispatch
        token_idx = torch.arange(n, device=x.device).unsqueeze(1).expand(n, self.top_k) # (n, k)
        t_flat, e_flat, s_flat = token_idx.reshape(-1), route_ind.reshape(-1), seat_safe.reshape(-1) # (n * k,)

        buf = x_flat.new_zeros(self.num_experts, capacity + 1, d)
        buf[e_flat, s_flat] = x_flat[t_flat] # (e, c + 1, d)


        # Every expert gets the same fixed (capacity + 1, d) slice, so no shape below
        # depends on how the router happened to distribute tokens. Row `capacity` is the
        # overflow bin: seat_safe sends dropped assignments there and their gate is zero,
        # so whatever it computes is discarded.
        y = torch.stack(
            [self.experts[i](buf[i]) for i in range(0, self.num_experts)], dim=0
        )  # (e, c + 1, d)

        # Combine. index_add_ rather than assignment because each token collects top_k
        # results, and it is the transpose of the gather that dispatched them.
        # Accumulate in the residual stream's dtype, which under autocast is fp32 while the
        # experts return fp16: index_add_ demands both match, and compiling happens to paper
        # over the mismatch, so eager runs would break alone.
        out = x_flat.new_zeros(n, d)
        weighted = y[e_flat, s_flat] * gate.reshape(-1, 1).to(y.dtype)
        out.index_add_(0, t_flat, weighted.to(out.dtype))

        # Load balance is measured on the routing decision, not on what survived capacity:
        # dropping is the symptom this loss exists to prevent, so it must not hide from it.
        expert_fracs = token_expert_oh.float().mean(dim=(0, 1))  # (e), sums to 1 so balance reads 1.0
        expert_probs = full_p.mean(dim=0)  # (e)

        loss_aux = self.num_experts * (expert_fracs * expert_probs).sum()

        return out.reshape(x.shape), loss_aux


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        x_prec = x.to(torch.float32)
        ms = (x_prec**2).mean(-1, keepdim=True)
        x_scaled = x_prec * torch.rsqrt(ms + self.eps)
        out = x_scaled * self.alpha
        out = out.to(x.dtype)
        return out


class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads, d_h, rope_layer):
        super().__init__()
        assert num_heads % num_kv_heads == 0
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.d_h = d_h
        self.qkv_proj = nn.Linear(d_model, (num_heads + 2 * num_kv_heads) * d_h, bias=False)
        self.o_proj = nn.Linear(num_heads * d_h, d_model, bias=False)

        self.rope = rope_layer

    def forward(self, x: Tensor, seq_starts: Tensor, kv_cache: Tensor | None = None):

        # Create Q, K, V tensors in a fused manner
        qkv = self.qkv_proj(x)  # (batch, seq, (num_heads + 2 * num_kv_heads) * d_h)

        qkv = qkv.reshape(
            qkv.shape[0], qkv.shape[1], self.num_heads + 2 * self.num_kv_heads, self.d_h
        )  # (batch, seq, num_heads + 2 * num_kv_heads, d_h)

        q, k_new, v_new = torch.split(
            qkv, [self.num_heads, self.num_kv_heads, self.num_kv_heads], dim=2
        )

        q = q.permute(0, 2, 1, 3)  # (batch, num_heads, seq_len, d_h)
        k_new = k_new.permute(0, 2, 1, 3)  # (batch, num_kv_heads, seq_len, d_h)
        v_new = v_new.permute(0, 2, 1, 3)  # (batch, num_kv_heads, seq_len, d_h)

        # Apply RoPE
        if kv_cache is not None:
            offset = kv_cache.shape[-2]
        else:
            offset = 0
        q, k_new = self.rope(q, k_new, offset)

        # Append onto the kv cache if using
        if kv_cache is not None:
            k = torch.concat((kv_cache[0], k_new), dim=2)
            v = torch.concat((kv_cache[1], v_new), dim=2)
        else:
            k = k_new
            v = v_new

        # K and V stay at num_kv_heads; the kernel maps each query head to its group
        attn_mh = flash_attention(q, k, v, seq_starts, causal=True)  # (batch, num_heads, seq, d_h)

        # Project out
        attn_flat = attn_mh.permute(0, 2, 1, 3).reshape(attn_mh.shape[0], attn_mh.shape[2], -1)
        O = self.o_proj(attn_flat)  # (batch, seq_len, d_model)

        # Wrap up the new kv cache entries; training has no cache to append them to
        kv_new = None if kv_cache is None else torch.stack((k_new, v_new), dim=0)

        return O, kv_new


class RotaryPositionalEncoding(nn.Module):
    def __init__(self, d_h, max_length):
        super().__init__()
        cos_thetas, sin_thetas = self.create_ang_trigs(d_h, max_length)
        self.register_buffer("cos_thetas", cos_thetas, persistent=False)
        self.register_buffer("sin_thetas", sin_thetas, persistent=False)

    @staticmethod
    def create_ang_trigs(d_h, max_length):
        base = 10000

        ws = []
        for i in range(0, d_h // 2):
            w_i = base ** (-2 * i / d_h)
            ws.append(w_i)
        ws = torch.tensor(ws, dtype=torch.float32)

        # Calculate the rotation angle for each pair
        positions = torch.arange(max_length)
        thetas = positions[:, None] * ws[None, :]  # (seq, d_h/2)

        # Calculate the trig components needed
        cos_thetas = torch.cos(thetas)
        sin_thetas = torch.sin(thetas)

        return cos_thetas, sin_thetas

    @staticmethod
    def rotate(x, cos_thetas, sin_thetas):
        cos_thetas = cos_thetas.to(x.dtype)
        sin_thetas = sin_thetas.to(x.dtype)

        # Apply the rotations to each pair
        x1 = x[..., 0::2]  # (b, h, seq, d_h/2)
        x2 = x[..., 1::2]  # (b, h, seq, d_h/2)

        out1 = x1 * cos_thetas - x2 * sin_thetas  # (b, h, seq, d_h/2)
        out2 = x1 * sin_thetas + x2 * cos_thetas  # (b, h, seq, d_h/2)

        # Interleave back together
        out = torch.stack([out1, out2], dim=-1).flatten(-2)  # (b, h, seq, d_h)

        return out

    def forward(self, Q, K, offset):

        seq_length = Q.shape[-2]

        if offset + seq_length > self.cos_thetas.shape[0]:
            raise ValueError(
                f"offset {offset} + seq_length {seq_length} exceeds "
                f"max_length {self.cos_thetas.shape[0]}"
            )

        cos_thetas = self.cos_thetas[offset : offset + seq_length]
        sin_thetas = self.sin_thetas[offset : offset + seq_length]

        Q_rot = self.rotate(Q, cos_thetas, sin_thetas)
        K_rot = self.rotate(K, cos_thetas, sin_thetas)

        return Q_rot, K_rot


class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.E = nn.Parameter(torch.randn(vocab_size, d_model) * 0.02)
        self.d_model = d_model

    def forward(self, x):
        x = self.E[x] * math.sqrt(self.d_model)
        return x


class DecoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        num_kv_heads,
        d_h,
        d_ff,
        num_experts,
        top_k,
        capacity_factor,
        dropout,
        rope_layer,
    ):
        super().__init__()
        self.attention = GroupedQueryAttention(d_model, num_heads, num_kv_heads, d_h, rope_layer)
        self.feedforward = MixtureOfExpertsFeedForward(
            d_model, d_ff, num_experts, top_k, capacity_factor
        )
        self.layer_norms = nn.ModuleList([RMSNorm(d_model) for _ in range(0, 2)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, seq_starts: Tensor, kv_cache: Tensor | None = None):
        attn_out, new_kv = self.attention(self.layer_norms[0](x), seq_starts, kv_cache)
        x = x + self.dropout(attn_out)
        ff_out, loss_aux = self.feedforward(self.layer_norms[1](x))
        x = x + self.dropout(ff_out)
        return x, loss_aux, new_kv


class Decoder(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        num_kv_heads,
        d_h,
        d_ff,
        num_experts,
        top_k,
        capacity_factor,
        num_dec_layers,
        dropout,
        max_length,
    ):
        super().__init__()
        self.rope_layer = RotaryPositionalEncoding(d_h, max_length)
        self.decoder_layers = nn.ModuleList(
            [
                DecoderLayer(
                    d_model,
                    num_heads,
                    num_kv_heads,
                    d_h,
                    d_ff,
                    num_experts,
                    top_k,
                    capacity_factor,
                    dropout,
                    self.rope_layer,
                )
                for _ in range(num_dec_layers)
            ]
        )
        self.final_norm = RMSNorm(d_model)

    def forward(self, x: Tensor, seq_starts: Tensor, kv_cache: Tensor | None = None):
        loss_aux_sum = 0
        new_kvs = []
        for i, layer in enumerate(self.decoder_layers):
            layer_cache = None if kv_cache is None else kv_cache[i]
            x, loss_aux, new_kv = layer(x, seq_starts, layer_cache)
            loss_aux_sum += loss_aux
            new_kvs.append(new_kv)

        loss_aux = loss_aux_sum / len(self.decoder_layers)
        x = self.final_norm(x)
        new_kv = None if kv_cache is None else torch.stack(new_kvs)  # (layers, 2, b, kv_h, q, d_h)
        return x, loss_aux, new_kv


class Transformer(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        num_kv_heads,
        d_h,
        d_ff,
        num_experts,
        top_k,
        capacity_factor,
        num_layers,
        vocab_size,
        max_length,
        dropout,
    ):
        super().__init__()
        self.max_length = max_length  # context bound: RoPE table size and training seq length
        self.num_kv_heads = num_kv_heads
        self.d_h = d_h
        self.num_layers = num_layers
        self.embedding = Embedding(vocab_size, d_model)
        self.decoder = Decoder(
            d_model,
            num_heads,
            num_kv_heads,
            d_h,
            d_ff,
            num_experts,
            top_k,
            capacity_factor,
            num_layers,
            dropout,
            max_length,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, seq_starts: Tensor | None = None, kv_cache: Tensor | None = None):
        # Training right-pads, where causal masking already excludes pad keys, so no
        # leading padding to skip. Left-padded generation passes real offsets.
        if seq_starts is None:
            seq_starts = torch.zeros(x.shape[0], dtype=torch.int32, device=x.device)

        x_emb = self.dropout(self.embedding(x))
        x_dec, loss_aux, new_kv = self.decoder(x_emb, seq_starts, kv_cache)
        out = torch.matmul(x_dec, self.embedding.E.T)
        # Cache-free callers (training) keep the two-tuple contract they already use
        if kv_cache is None:
            return out, loss_aux
        return out, loss_aux, new_kv


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.Embedding, Embedding)):
        nn.init.normal_(m.E, mean=0.0, std=0.02)


def build_transformer(config):

    model_config = config["model"]

    transformer = Transformer(
        model_config["d_model"],
        model_config["num_heads"],
        model_config["num_kv_heads"],
        model_config["d_h"],
        model_config["d_ff"],
        model_config["num_experts"],
        model_config["top_k"],
        model_config["capacity_factor"],
        model_config["num_layers"],
        config["tokenizer"]["vocab_size"],
        model_config["max_length"],
        model_config["dropout"],
    )
    transformer = transformer.apply(init_weights)
    return transformer
