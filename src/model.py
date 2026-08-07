from torch import nn
import torch
import math
from src.kernels import flash_attention


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


class MixtureOfExpertsFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, num_experts):
        super().__init__()
        self.num_experts = num_experts
        self.relu = nn.ReLU()

    def forward(self, x):
        # Placeholder implementation
        return x


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


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, d_h, rope_layer):
        super().__init__()
        self.num_heads = num_heads
        self.d_h = d_h
        self.W_qkv = nn.Linear(d_model, num_heads * 3 * d_h)
        self.W_o = nn.Linear(num_heads * d_h, d_model)
        self.rope = rope_layer

    def forward(self, x):

        # Create Q, K, V tensors in a fused manner
        QKV = (
            self.W_qkv(x)
            .reshape(x.shape[0], x.shape[1], self.num_heads, 3, self.d_h)
            .permute(0, 2, 3, 1, 4)
        )  # (batch, num_heads, 3, seq_len, d_h)
        Q, K, V = QKV.unbind(dim=2)  # (batch, num_heads, seq_len, d_h)

        # Apply RoPE
        Q, K = self.rope(Q, K)

        # Apply flash attention kernel
        attn_mh = flash_attention(Q, K, V, causal=True)  # (batch, num_heads, seq_len, d_h)

        # Project out
        attn_flat = attn_mh.permute(0, 2, 1, 3).reshape(attn_mh.shape[0], attn_mh.shape[2], -1)
        O = self.W_o(attn_flat)  # (batch, seq_len, d_model)

        return O


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

    def forward(self, Q, K):

        seq_length = Q.shape[-2]

        if seq_length > self.cos_thetas.shape[0]:
            raise ValueError(
                f"seq_length {seq_length} exceeds max_length {self.cos_thetas.shape[0]}"
            )

        cos_thetas = self.cos_thetas[0:seq_length]
        sin_thetas = self.sin_thetas[0:seq_length]

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
    def __init__(self, d_model, num_heads, d_h, d_ff, dropout, rope_layer):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, d_h, rope_layer)
        self.feedforward = PositionWiseFeedForward(d_model, d_ff)
        self.layer_norms = nn.ModuleList([RMSNorm(d_model) for _ in range(0, 2)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out = self.dropout(self.attention(self.layer_norms[0](x)))
        x = x + attn_out
        ff_out = self.dropout(self.feedforward(self.layer_norms[1](x)))
        x = x + ff_out
        return x


class Decoder(nn.Module):
    def __init__(self, d_model, num_heads, d_h, d_ff, num_dec_layers, dropout, max_length):
        super().__init__()
        self.rope_layer = RotaryPositionalEncoding(d_h, max_length)
        self.decoder_layers = nn.ModuleList(
            [
                DecoderLayer(d_model, num_heads, d_h, d_ff, dropout, self.rope_layer)
                for _ in range(num_dec_layers)
            ]
        )
        self.final_norm = RMSNorm(d_model)

    def forward(self, x):
        for layer in self.decoder_layers:
            x = layer(x)
        x = self.final_norm(x)
        return x


class Output(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, tgt_dec):
        x = self.linear(tgt_dec)
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        d_h,
        d_ff,
        num_layers,
        vocab_size,
        max_length,
        dropout,
    ):
        super().__init__()
        self.embedding = Embedding(vocab_size, d_model)
        self.decoder = Decoder(d_model, num_heads, d_h, d_ff, num_layers, dropout, max_length)
        self.output = Output(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x_emb = self.dropout(self.embedding(x))
        x_dec = self.decoder(x_emb)
        out = self.output(x_dec)
        return out


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
        model_config["d_h"],
        model_config["d_ff"],
        model_config["num_layers"],
        config["tokenizer"]["vocab_size"],
        model_config["max_length"],
        model_config["dropout"],
    )
    transformer = transformer.apply(init_weights)
    return transformer
