"""Grouped-query attention with RoPE positions and affine QK normalization."""

from __future__ import annotations

import math
from typing import Any


def build_rope_cache(
    positions: Any, *, head_dimension: int, rope_base: float,
    torch_module: Any,
) -> tuple[Any, Any]:
    """Build reusable FP32 RoPE angles for every layer in one forward pass."""

    torch = torch_module
    if positions.ndim != 2:
        raise ValueError("RoPE positions must be rank two")
    if type(head_dimension) is not int or head_dimension <= 0 or head_dimension % 2:
        raise ValueError("RoPE head dimension must be positive and even")
    if (isinstance(rope_base, bool) or not isinstance(rope_base, (int, float))
            or not math.isfinite(float(rope_base)) or float(rope_base) <= 0.0):
        raise ValueError("RoPE base must be finite and positive")
    inverse = float(rope_base) ** (
        -torch.arange(0, head_dimension, 2,
                      device=positions.device, dtype=torch.float32)
        / head_dimension
    )
    phase = positions.float()[:, None, :, None] * inverse[None, None, None, :]
    return phase.cos(), phase.sin()


def apply_rope(value: Any, cosine: Any, sine: Any, *, torch_module: Any) -> Any:
    """Apply shared RoPE angles to one Q or K tensor without rebuilding them."""

    torch = torch_module
    if value.ndim != 4 or cosine.ndim != 4 or sine.shape != cosine.shape:
        raise ValueError("RoPE values and cached angles must have rank-four shapes")
    if (cosine.shape[0] != value.shape[0]
            or cosine.shape[2] != value.shape[2]
            or cosine.shape[3] * 2 != value.shape[3]
            or cosine.shape[1] not in (1, value.shape[1])):
        raise ValueError("RoPE cache shape does not match the attention tensor")
    cosine = cosine.to(value.dtype)
    sine = sine.to(value.dtype)
    even, odd = value[..., 0::2], value[..., 1::2]
    return torch.stack(
        (even * cosine - odd * sine, even * sine + odd * cosine), -1
    ).flatten(-2)


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

        def forward(self, hidden: Any, positions: Any, mask: Any,
                    rope_cosine: Any = None, rope_sine: Any = None) -> Any:
            batch, length, _ = hidden.shape
            if (rope_cosine is None) != (rope_sine is None):
                raise ValueError("RoPE cosine and sine caches must be supplied together")
            if rope_cosine is None:
                rope_cosine, rope_sine = build_rope_cache(
                    positions,
                    head_dimension=config.head_dimension,
                    rope_base=config.rope_base,
                    torch_module=torch,
                )
            query = self.query(hidden).view(
                batch, length, config.query_heads, config.head_dimension).transpose(1, 2)
            key = self.key(hidden).view(
                batch, length, config.kv_heads, config.head_dimension).transpose(1, 2)
            value = self.value(hidden).view(
                batch, length, config.kv_heads, config.head_dimension).transpose(1, 2)
            query = apply_rope(
                self.normalize(query, self.query_scale),
                rope_cosine, rope_sine, torch_module=torch,
            )
            key = apply_rope(
                self.normalize(key, self.key_scale),
                rope_cosine, rope_sine, torch_module=torch,
            )
            repeats = config.query_heads // config.kv_heads
            key = key.repeat_interleave(repeats, dim=1)
            value = value.repeat_interleave(repeats, dim=1)
            attended = functional.scaled_dot_product_attention(
                query, key, value, attn_mask=mask, dropout_p=0.0)
            return self.output(attended.transpose(1, 2).contiguous().view(
                batch, length, config.width))

    return Attention()


__all__ = ["apply_rope", "build_attention", "build_rope_cache"]
