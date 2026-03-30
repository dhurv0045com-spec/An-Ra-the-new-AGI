"""
model/attention.py — Steps 13–15
Scaled dot-product attention → Multi-head attention → Grouped Query Attention.
Supports: causal masking, KV cache (inference), Flash Attention, RoPE.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ─────────────────────────────────────────────
# STEP 13 & 14 — Scaled dot-product attention
# ─────────────────────────────────────────────

def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    scale: Optional[float] = None,
    training: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Scaled dot-product attention.
    Q, K, V: (batch, heads, seq, head_dim)

    Returns: (attended_values, attention_weights)

    The scale factor 1/sqrt(d_k) prevents the softmax from saturating
    into regions of zero gradient for large d_k.
    """
    d_k = q.size(-1)
    scale = scale or (1.0 / math.sqrt(d_k))

    # (B, H, T_q, T_k)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))

    weights = F.softmax(scores, dim=-1)

    # Replace NaN from all-masked rows (can happen with padding)
    weights = torch.nan_to_num(weights)

    if dropout_p > 0.0 and training:
        weights = F.dropout(weights, p=dropout_p)

    out = torch.matmul(weights, v)
    return out, weights


def make_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Lower-triangular causal mask. 1 = attend, 0 = mask."""
    return torch.tril(torch.ones(seq_len, seq_len, device=device)).unsqueeze(0).unsqueeze(0)


# ─────────────────────────────────────────────
# STEP 15 — Multi-Head Attention
# ─────────────────────────────────────────────

class MultiHeadAttention(nn.Module):
    """
    Standard multi-head attention (Vaswani et al., 2017).

    Supports:
    - Self-attention and cross-attention
    - Causal masking for autoregressive decoding
    - KV cache for fast inference
    - RoPE positional encoding
    - PyTorch 2.x scaled_dot_product_attention (Flash Attention path)
    - Grouped Query Attention (n_kv_heads < n_heads) — memory efficient
    """

    def __init__(self, d_model: int, n_heads: int,
                 dropout: float = 0.1,
                 n_kv_heads: Optional[int] = None,
                 bias: bool = False,
                 rope: Optional[nn.Module] = None):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Grouped Query Attention: fewer KV heads than Q heads
        self.n_kv_heads = n_kv_heads or n_heads
        self.n_kv_groups = n_heads // self.n_kv_heads

        self.dropout = dropout
        self.rope = rope

        # Projections
        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=bias)
        self.out_proj = nn.Linear(n_heads * self.head_dim, d_model, bias=bias)

        self._init_weights()

    def _init_weights(self):
        for proj in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.normal_(proj.weight, mean=0, std=0.02)
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)

    def _split_heads(self, x: torch.Tensor, n_heads: int) -> torch.Tensor:
        """(B, T, n_heads*head_dim) → (B, n_heads, T, head_dim)"""
        B, T, _ = x.shape
        x = x.view(B, T, n_heads, self.head_dim)
        return x.transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, n_heads, T, head_dim) → (B, T, n_heads*head_dim)"""
        B, H, T, D = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, H * D)

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        causal: bool = True,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cache_offset: int = 0,
    ) -> Tuple[torch.Tensor, Optional[Tuple]]:
        """
        x:       (B, T, d_model) — query source
        context: (B, S, d_model) — key/value source (None = self-attention)
        kv_cache: (k_cache, v_cache) for autoregressive decoding
        Returns: (output, new_kv_cache)
        """
        B, T, _ = x.shape
        src = context if context is not None else x

        q = self._split_heads(self.q_proj(x),   self.n_heads)
        k = self._split_heads(self.k_proj(src),  self.n_kv_heads)
        v = self._split_heads(self.v_proj(src),  self.n_kv_heads)

        # Apply RoPE if provided
        if self.rope is not None:
            q, k = self.rope(q, k, offset=cache_offset)

        # KV cache: append new k,v and use full sequence
        new_cache = None
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)
            new_cache = (k, v)

        # Grouped Query Attention: repeat KV heads
        if self.n_kv_groups > 1:
            k = k.repeat_interleave(self.n_kv_groups, dim=1)
            v = v.repeat_interleave(self.n_kv_groups, dim=1)

        # Build causal mask
        if causal and kv_cache is None:
            causal_mask = make_causal_mask(T, x.device)
            mask = causal_mask if mask is None else mask & causal_mask

        # Use PyTorch 2.x flash attention when available and no explicit weights needed
        use_flash = (hasattr(F, "scaled_dot_product_attention")
                     and mask is None and not self.training)

        if use_flash:
            # Flash Attention path — O(N) memory, faster on hardware
            attn_out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=causal and kv_cache is None,
            )
        else:
            attn_out, _ = scaled_dot_product_attention(
                q, k, v, mask=mask,
                dropout_p=self.dropout if self.training else 0.0,
                training=self.training,
            )

        out = self.out_proj(self._merge_heads(attn_out))
        return out, new_cache


# ─────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("Steps 13–15: Attention mechanism checks")
    B, T, d = 2, 32, 256
    H = 8

    x = torch.randn(B, T, d)

    # Standard MHA
    mha = MultiHeadAttention(d_model=d, n_heads=H, dropout=0.0)
    out, _ = mha(x, causal=True)
    assert out.shape == (B, T, d)
    print(f"  MHA output shape:  {tuple(out.shape)}  ✓")

    # GQA — 8 Q heads, 2 KV heads (4x reduction in KV cache size)
    gqa = MultiHeadAttention(d_model=d, n_heads=8, n_kv_heads=2, dropout=0.0)
    out_gqa, _ = gqa(x, causal=True)
    assert out_gqa.shape == (B, T, d)
    print(f"  GQA output shape:  {tuple(out_gqa.shape)}  ✓")

    # KV cache — single token decoding step
    mha.eval()
    context_len = 10
    ctx = torch.randn(B, context_len, d)
    _, cache = mha(ctx, causal=True)

    new_tok = torch.randn(B, 1, d)
    out_cached, new_cache = mha(new_tok, causal=False,
                                kv_cache=cache, cache_offset=context_len)
    assert out_cached.shape == (B, 1, d)
    print(f"  KV cache decode:   {tuple(out_cached.shape)}  ✓")

    # Scaled dot-product directly
    q = torch.randn(B, H, T, d // H)
    k = torch.randn(B, H, T, d // H)
    v = torch.randn(B, H, T, d // H)
    mask = make_causal_mask(T, q.device)
    att, weights = scaled_dot_product_attention(q, k, v, mask=mask, training=False)
    assert att.shape == (B, H, T, d // H)
    print(f"  SDPA output shape: {tuple(att.shape)}  ✓")

    print("✓ Steps 13–15 verified")
