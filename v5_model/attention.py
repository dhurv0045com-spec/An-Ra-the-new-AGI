"""Grouped-query attention with RoPE positions and affine QK normalization."""

from __future__ import annotations

from typing import Any


def build_attention(config: Any, *, torch_module: Any) -> Any:
    """Build one full-causal GQA layer with pairwise RoPE and QK norm."""

    torch = torch_module
    nn = torch.nn
    functional = torch.nn.functional
    kv_width = config.kv_heads * config.head_dimension

    class Attention(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.query = nn.Linear(config.width, config.width, bias=False)
            self.key = nn.Linear(config.width, kv_width, bias=False)
            self.value = nn.Linear(config.width, kv_width, bias=False)
            self.output = nn.Linear(config.width, config.width, bias=False)
            if config.qk_norm and config.qk_norm_affine:
                self.query_scale = nn.Parameter(
                    torch.ones(config.query_heads, config.head_dimension)
                )
                self.key_scale = nn.Parameter(
                    torch.ones(config.kv_heads, config.head_dimension)
                )
            else:
                self.register_parameter("query_scale", None)
                self.register_parameter("key_scale", None)

        def normalize(self, value: Any, scale: Any) -> Any:
            if not config.qk_norm:
                return value
            normalized = value.float() * torch.rsqrt(
                value.float().square().mean(-1, keepdim=True) + config.qk_norm_epsilon
            )
            if scale is not None:
                normalized = normalized * scale.float()[None, :, None, :]
            return normalized.to(value.dtype)

        def rope(self, value: Any, positions: Any) -> Any:
            inverse = config.rope_base ** (
                -torch.arange(0, config.head_dimension, 2,
                              device=value.device, dtype=torch.float32)
                / config.head_dimension
            )
            phase = positions.float()[:, None, :, None] * inverse[None, None, None, :]
            cosine, sine = phase.cos().to(value.dtype), phase.sin().to(value.dtype)
            even, odd = value[..., 0::2], value[..., 1::2]
            return torch.stack(
                (even * cosine - odd * sine, even * sine + odd * cosine), -1
            ).flatten(-2)

        def forward(self, hidden: Any, positions: Any, mask: Any) -> Any:
            batch, length, _ = hidden.shape
            query = self.query(hidden).view(
                batch, length, config.query_heads, config.head_dimension).transpose(1, 2)
            key = self.key(hidden).view(
                batch, length, config.kv_heads, config.head_dimension).transpose(1, 2)
            value = self.value(hidden).view(
                batch, length, config.kv_heads, config.head_dimension).transpose(1, 2)
            query = self.rope(self.normalize(query, self.query_scale), positions)
            key = self.rope(self.normalize(key, self.key_scale), positions)
            repeats = config.query_heads // config.kv_heads
            key = key.repeat_interleave(repeats, dim=1)
            value = value.repeat_interleave(repeats, dim=1)
            attended = functional.scaled_dot_product_attention(
                query, key, value, attn_mask=mask, dropout_p=0.0)
            return self.output(attended.transpose(1, 2).contiguous().view(
                batch, length, config.width))

    return Attention()


__all__ = ["build_attention"]
