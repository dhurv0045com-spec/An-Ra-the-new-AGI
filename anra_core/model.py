"""Canonical An-Ra V4 Dense Neural Architecture.

Pure mathematical formulation of the 18-layer, width-896 decoder transformer.
Owns learned parameters, normalization, rotary embeddings, attention scheduling,
and next-token logit computation.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import CANONICAL_CONFIG, CoreConfig
from .errors import ContextOverflowError

if TYPE_CHECKING:
    from .state import CoreState


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

    def forward(
        self,
        length: int,
        device: torch.device,
        dtype: torch.dtype,
        start_pos: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(start_pos, start_pos + length, device=device, dtype=torch.float32)
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

    def forward(
        self,
        x: torch.Tensor,
        *,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        start_pos: int = 0,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        batch, length, _ = x.shape
        q = self.q_proj(x).view(batch, length, self.config.n_heads, self.config.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, length, self.config.n_kv_heads, self.config.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, length, self.config.n_kv_heads, self.config.head_dim).transpose(1, 2)

        if self.config.qk_norm:
            q = F.rms_norm(q, (self.config.head_dim,))
            k = F.rms_norm(k, (self.config.head_dim,))

        cos, sin = self.rope(length, x.device, q.dtype, start_pos=start_pos)
        cos, sin = cos[None, None, :, :], sin[None, None, :, :]
        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin

        limit = 0.8 * math.sqrt(65_504.0 / self.config.head_dim)
        q, k = limit * torch.tanh(q / limit), limit * torch.tanh(k / limit)

        if kv_cache is not None:
            prev_k, prev_v = kv_cache
            k = torch.cat([prev_k, k], dim=2)
            v = torch.cat([prev_v, v], dim=2)

        updated_kv = (k, v)
        total_len = k.shape[2]

        if length == 1:
            # Incremental single-token decode
            if not self.full_attention and total_len > self.config.sliding_window:
                k_attn = k[:, :, -self.config.sliding_window :, :]
                v_attn = v[:, :, -self.config.sliding_window :, :]
            else:
                k_attn = k
                v_attn = v

            attended = F.scaled_dot_product_attention(
                q, k_attn, v_attn, attn_mask=None, dropout_p=0.0, is_causal=False, enable_gqa=True
            )
        elif kv_cache is None:
            # Full uncached forward pass (L_q == L_k == length)
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
        else:
            # Chunked prefill with prior KV cache (L_q = length, L_k = total_len)
            q_pos = torch.arange(start_pos, start_pos + length, device=x.device)[:, None]
            k_pos = torch.arange(0, total_len, device=x.device)[None, :]
            mask = q_pos >= k_pos
            if not self.full_attention:
                mask = mask & (k_pos > q_pos - self.config.sliding_window)
            mask = mask[None, None, :, :]
            attended = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False, enable_gqa=True
            )

        attended = attended.transpose(1, 2).contiguous().view(batch, length, -1)
        return self.out_proj(attended), updated_kv


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

    def forward(
        self,
        x: torch.Tensor,
        *,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        start_pos: int = 0,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        attn_out, updated_kv = self.attn(self.norm_1(x), kv_cache=kv_cache, start_pos=start_pos)
        x = x + attn_out
        x = x + self.mlp(self.norm_2(x))
        return x, updated_kv


class AnRaCore(nn.Module):
    """The canonical V4 dense core language model."""

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

    def forward(
        self,
        token_ids: torch.Tensor,
        state: CoreState | None = None,
    ) -> torch.Tensor:
        """Full sequence or stateful incremental forward pass.

        If state is None: executes standard full-sequence forward pass.
        If state is provided: executes incremental decode step and updates state.
        """
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [batch, sequence]")

        batch_size, seq_len = token_ids.shape

        if state is None:
            if seq_len > self.config.block_size:
                raise ContextOverflowError(
                    f"sequence exceeds {self.config.block_size} token context",
                    details={"seq_len": seq_len, "block_size": self.config.block_size},
                )
            x = self.token_embedding_table(token_ids) * self.embedding_input_scale
            for block in self.blocks:
                x, _ = block(x, kv_cache=None, start_pos=0)
            return self.lm_head(self.norm_f(x))

        # Stateful incremental execution
        state.check_capacity(seq_len)
        start_pos = state.current_length
        x = self.token_embedding_table(token_ids) * self.embedding_input_scale

        for idx, block in enumerate(self.blocks):
            layer_kv = state.get_layer_kv(idx)
            x, updated_kv = block(x, kv_cache=layer_kv, start_pos=start_pos)
            state.set_layer_kv(idx, updated_kv)

        state.advance(seq_len)
        return self.lm_head(self.norm_f(x))
