from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import CANONICAL_CONFIG, CoreConfig


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.register_buffer("multiplicity_weight", torch.ones(width), persistent=True)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = (x.pow(2) * self.multiplicity_weight).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, base: float) -> None:
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, length: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(length, device=device, dtype=torch.float32)
        angles = torch.outer(positions, self.inv_freq.to(device=device))
        angles = torch.repeat_interleave(angles, 2, dim=-1)
        return angles.cos().to(dtype), angles.sin().to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    even, odd = x[..., ::2], x[..., 1::2]
    return torch.stack((-odd, even), dim=-1).flatten(-2)


class GroupedQueryAttention(nn.Module):
    def __init__(self, config: CoreConfig, *, full_attention: bool) -> None:
        super().__init__()
        self.config = config
        self.full_attention = full_attention
        self.q_proj = nn.Linear(config.d_model, config.n_heads * config.head_dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.n_kv_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.n_kv_heads * config.head_dim, bias=False)
        self.out_proj = nn.Linear(config.n_heads * config.head_dim, config.d_model, bias=False)
        self.rope = RotaryEmbedding(config.head_dim, config.rope_base)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        q = self.q_proj(x).view(batch, length, self.config.n_heads, self.config.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, length, self.config.n_kv_heads, self.config.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, length, self.config.n_kv_heads, self.config.head_dim).transpose(1, 2)
        if self.config.qk_norm:
            q = F.rms_norm(q, (self.config.head_dim,))
            k = F.rms_norm(k, (self.config.head_dim,))
        cos, sin = self.rope(length, x.device, q.dtype)
        cos, sin = cos[None, None, :, :], sin[None, None, :, :]
        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin
        limit = 0.8 * math.sqrt(65_504.0 / self.config.head_dim)
        q, k = limit * torch.tanh(q / limit), limit * torch.tanh(k / limit)
        mask = None
        is_causal = True
        if not self.full_attention and length > self.config.sliding_window:
            positions = torch.arange(length, device=x.device)
            mask = (positions[:, None] >= positions[None, :]) & (
                positions[None, :] > positions[:, None] - self.config.sliding_window
            )
            mask = mask[None, None, :, :]
            is_causal = False
        attended = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=is_causal, enable_gqa=True
        )
        attended = attended.transpose(1, 2).contiguous().view(batch, length, -1)
        return self.out_proj(attended)


class SwiGLU(nn.Module):
    def __init__(self, config: CoreConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.up_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.down_proj = nn.Linear(config.d_ff, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DenseBlock(nn.Module):
    def __init__(self, config: CoreConfig, layer: int) -> None:
        super().__init__()
        self.norm_1 = RMSNorm(config.d_model, config.rms_norm_eps)
        self.attn = GroupedQueryAttention(
            config, full_attention=((layer + 1) % config.full_attention_every == 0)
        )
        self.norm_2 = RMSNorm(config.d_model, config.rms_norm_eps)
        self.mlp = SwiGLU(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm_1(x))
        return x + self.mlp(self.norm_2(x))


class AnRaCore(nn.Module):
    def __init__(self, config: CoreConfig = CANONICAL_CONFIG) -> None:
        super().__init__()
        self.config = config
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.d_model)
        self.register_buffer(
            "embedding_input_scale", torch.ones(config.d_model), persistent=True
        )
        self.blocks = nn.ModuleList(DenseBlock(config, index) for index in range(config.n_layers))
        self.norm_f = RMSNorm(config.d_model, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding_table.weight

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [batch, sequence]")
        if token_ids.shape[1] > self.config.block_size:
            raise ValueError(f"sequence exceeds {self.config.block_size} token context")
        x = self.token_embedding_table(token_ids) * self.embedding_input_scale
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.norm_f(x))
