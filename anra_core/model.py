"""Canonical An-Ra V4 Dense Neural Architecture.

Pure mathematical formulation of the 18-layer, width-896 decoder transformer.
Owns learned parameters, normalization, rotary embeddings, attention scheduling,
and next-token logit computation.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from .config import CANONICAL_CONFIG, CoreConfig
from .errors import ContextOverflowError


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
        self.memory_efficient_chunk_size: int | None = None

    def enable_memory_efficient_attention(self, chunk_size: int | None) -> None:
        if chunk_size is not None and int(chunk_size) <= 0:
            raise ValueError("attention chunk size must be positive")
        self.memory_efficient_chunk_size = None if chunk_size is None else int(chunk_size)

    def _tiled_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        start_pos: int,
    ) -> torch.Tensor:
        """Apply the exact V4 attention mask in bounded query tiles."""
        tile = int(self.memory_efficient_chunk_size or q.shape[2])
        key_positions = torch.arange(k.shape[2], device=q.device)[None, :]
        pieces: list[torch.Tensor] = []
        for offset in range(0, q.shape[2], tile):
            end = min(offset + tile, q.shape[2])
            query_positions = torch.arange(
                start_pos + offset, start_pos + end, device=q.device
            )[:, None]
            mask = key_positions <= query_positions
            if not self.full_attention:
                mask = mask & (key_positions > query_positions - self.config.sliding_window)
            pieces.append(
                F.scaled_dot_product_attention(
                    q[:, :, offset:end, :],
                    k,
                    v,
                    attn_mask=mask[None, None, :, :],
                    dropout_p=0.0,
                    is_causal=False,
                    enable_gqa=True,
                )
            )
        return torch.cat(pieces, dim=2)

    def forward(
        self,
        x: torch.Tensor,
        *,
        cache_buffer: tuple[torch.Tensor, torch.Tensor] | None = None,
        start_pos: int = 0,
    ) -> torch.Tensor:
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

        if cache_buffer is not None:
            key_buffer, value_buffer = cache_buffer
            end_pos = start_pos + length
            if end_pos > key_buffer.shape[2] or end_pos > value_buffer.shape[2]:
                raise ContextOverflowError(
                    "incremental attention exceeds cache capacity",
                    details={"end_pos": end_pos, "capacity": key_buffer.shape[2]},
                )
            # Writes occur outside the state's committed logical prefix. If a
            # later layer fails, current_length is unchanged and retries safely
            # overwrite these uncommitted slots.
            key_buffer[:, :, start_pos:end_pos].copy_(k.detach())
            value_buffer[:, :, start_pos:end_pos].copy_(v.detach())
            k = key_buffer[:, :, :end_pos]
            v = value_buffer[:, :, :end_pos]
        total_len = k.shape[2]

        if self.memory_efficient_chunk_size is not None and length > 1:
            attended = self._tiled_attention(q, k, v, start_pos=start_pos)
        elif cache_buffer is not None and length == 1:
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
        elif cache_buffer is None:
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

    def forward(
        self,
        x: torch.Tensor,
        *,
        cache_buffer: tuple[torch.Tensor, torch.Tensor] | None = None,
        start_pos: int = 0,
    ) -> torch.Tensor:
        attn_out = self.attn(
            self.norm_1(x), cache_buffer=cache_buffer, start_pos=start_pos
        )
        x = x + attn_out
        return x + self.mlp(self.norm_2(x))


class AnRaCore(nn.Module):
    """The canonical V4 dense core language model."""

    def __init__(self, config: CoreConfig = CANONICAL_CONFIG) -> None:
        super().__init__()
        self.config = config
        self.gradient_checkpointing = False
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.d_model)
        self.register_buffer(
            "embedding_input_scale", torch.ones(config.d_model), persistent=True
        )
        self.blocks = nn.ModuleList(DenseBlock(config, index) for index in range(config.n_layers))
        self.norm_f = RMSNorm(config.d_model, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding_table.weight

    def enable_gradient_checkpointing(self, enabled: bool = True) -> None:
        """Rematerialize block activations during training without changing weights."""
        self.gradient_checkpointing = bool(enabled)

    def enable_memory_efficient_attention(self, chunk_size: int | None = 128) -> None:
        """Bound attention workspace without changing the model state dictionary."""
        for block in self.blocks:
            block.attn.enable_memory_efficient_attention(chunk_size)

    def forward(
        self,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Pure differentiable full-sequence V4 forward computation."""
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [batch, sequence]")
        seq_len = token_ids.shape[1]
        if seq_len <= 0:
            raise ValueError("token_ids cannot contain an empty sequence")
        if seq_len > self.config.block_size:
            raise ContextOverflowError(
                f"sequence exceeds {self.config.block_size} token context",
                details={"seq_len": seq_len, "block_size": self.config.block_size},
            )
        x = self.token_embedding_table(token_ids) * self.embedding_input_scale
        for block in self.blocks:
            if self.training and self.gradient_checkpointing:
                if x.device.type == "xla":
                    # XLA needs an optimization barrier around recomputation and
                    # currently supports only the reentrant implementation.
                    from torch_xla.utils.checkpoint import checkpoint as xla_checkpoint

                    x = xla_checkpoint(
                        block, x, use_reentrant=True, preserve_rng_state=False
                    )
                else:
                    x = torch_checkpoint(
                        block, x, use_reentrant=False, preserve_rng_state=False
                    )
            else:
                x = block(x, cache_buffer=None, start_pos=0)
        return self.lm_head(self.norm_f(x))

    def forward_incremental(
        self,
        token_ids: torch.Tensor,
        *,
        cache_buffers: list[tuple[torch.Tensor, torch.Tensor]],
        start_pos: int,
    ) -> torch.Tensor:
        """Executor primitive for an exact incremental V4 forward.

        Cache storage belongs to the Executor. This method defines how V4 uses
        that storage without importing, owning, or mutating a CoreState handle.
        """
        if token_ids.ndim != 2 or token_ids.shape[1] <= 0:
            raise ValueError("token_ids must have non-empty shape [batch, sequence]")
        if len(cache_buffers) != len(self.blocks):
            raise ValueError("incremental cache layer count does not match the model")
        end_pos = start_pos + token_ids.shape[1]
        if start_pos < 0 or end_pos > self.config.block_size:
            raise ContextOverflowError(
                f"incremental sequence exceeds {self.config.block_size} token context",
                details={"start_pos": start_pos, "end_pos": end_pos},
            )
        x = self.token_embedding_table(token_ids) * self.embedding_input_scale
        for block, cache_buffer in zip(self.blocks, cache_buffers, strict=True):
            x = block(x, cache_buffer=cache_buffer, start_pos=start_pos)
        return self.lm_head(self.norm_f(x))
