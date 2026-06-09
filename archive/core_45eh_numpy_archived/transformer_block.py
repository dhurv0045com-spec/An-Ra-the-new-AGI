"""
================================================================================
FILE: transformer_block.py
PROJECT: Transformer Language Model — 45E v2
STEP: 18 — Full Transformer Block
================================================================================

LLaMA/Mistral-style transformer block — the current production standard:

  PRE-NORM PATTERN:
    x = x + self_attn( RMSNorm(x) )       ← attention sublayer
    x = x + cross_attn( RMSNorm(x), mem ) ← cross-attention (decoder only)
    x = x + ffn( RMSNorm(x) )             ← feed-forward sublayer

  COMPONENT CHOICES (all deliberate, all state-of-art):
    Norm:      RMSNorm  (LLaMA, Mistral) — faster, no β needed
    Attention: GQA with RoPE             — compressed KV, relative positions
    FFN:       SwiGLU                   — better quality, learned gating
    Residuals: Pre-norm (not post-norm)  — stable at depth

  OUTPUT PROJECTION SCALING (GPT-2 trick):
    Depth N → scale output projections by 1/√(2N).
    This prevents the residual stream from growing as layers accumulate.
    Without it, deep models need very careful LR warmup.

  GRADIENT FLOW:
    Residual connections provide a "highway" for gradients to flow
    directly from loss to early layers. The normalization ensures
    each sublayer's output stays in a stable range before being added.
================================================================================
"""

import numpy as np
from typing import Optional, Tuple

from attention   import KVCache, make_causal_mask
from multihead   import MultiHeadAttention
from feedforward import SwiGLUFeedForward, GELUFeedForward
from layernorm   import RMSNorm


class TransformerBlock:
    """
    One transformer block: pre-RMSNorm + GQA attention + SwiGLU FFN + residuals.

    Supports:
      - Encoder mode: bidirectional self-attention (no causal mask)
      - Decoder mode: causal self-attention (+ optional cross-attention)
      - KV-cache for efficient autoregressive inference

    Args:
        d_model:         Model dimension
        num_heads:       Number of query attention heads
        num_kv_heads:    KV heads for GQA (default: num_heads = standard MHA)
        d_ff:            FFN hidden dim (None → SwiGLU auto: ~2.67× d_model)
        dropout_rate:    Dropout on attention weights and FFN hidden
        use_cross_attn:  Add cross-attention sublayer (encoder-decoder only)
        ffn_type:        "swiglu" (default) or "gelu"
        max_seq_len:     For RoPE table (default: 4096)
        rope_base:       RoPE base frequency (10000 default)
        layer_idx:       Index in the stack — used for output projection scaling
        num_layers:      Total layers in stack — used for output projection scaling
        seed:            RNG seed
    """

    def __init__(
        self,
        d_model:        int,
        num_heads:      int,
        num_kv_heads:   Optional[int]   = None,
        d_ff:           Optional[int]   = None,
        dropout_rate:   float           = 0.1,
        use_cross_attn: bool            = False,
        ffn_type:       str             = "swiglu",
        max_seq_len:    int             = 4096,
        rope_base:      float           = 10000.0,
        layer_idx:      int             = 0,
        num_layers:     int             = 1,
        seed:           int             = 0,
    ):
        self.d_model        = d_model
        self.num_heads      = num_heads
        self.num_kv_heads   = num_kv_heads or num_heads
        self.use_cross_attn = use_cross_attn
        self.layer_idx      = layer_idx

        # GPT-2 depth scaling: deeper layers get smaller output projections
        # Prevents residual stream variance from growing with depth
        out_proj_scale = 1.0 / np.sqrt(2.0 * num_layers) if num_layers > 1 else 1.0

        # ── Pre-norm for self-attention ──────────────────────────────────
        self.norm1 = RMSNorm(d_model)

        # ── Self-attention (GQA + RoPE) ──────────────────────────────────
        self.self_attn = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            num_kv_heads=self.num_kv_heads,
            dropout_rate=dropout_rate,
            max_seq_len=max_seq_len,
            rope_base=rope_base,
            out_proj_scale=out_proj_scale,
            seed=seed,
        )

        # ── Optional cross-attention (seq2seq decoder) ───────────────────
        if use_cross_attn:
            self.norm_cross  = RMSNorm(d_model)
            self.cross_attn  = MultiHeadAttention(
                d_model=d_model,
                num_heads=num_heads,
                num_kv_heads=self.num_kv_heads,
                dropout_rate=dropout_rate,
                max_seq_len=max_seq_len,
                rope_base=rope_base,
                out_proj_scale=out_proj_scale,
                seed=seed + 100,
            )

        # ── Pre-norm for FFN ─────────────────────────────────────────────
        self.norm2 = RMSNorm(d_model)

        # ── Feed-forward ─────────────────────────────────────────────────
        if ffn_type == "swiglu":
            self.ffn = SwiGLUFeedForward(
                d_model=d_model,
                d_ff=d_ff,
                dropout_rate=dropout_rate,
                seed=seed + 200,
            )
        elif ffn_type == "gelu":
            self.ffn = GELUFeedForward(
                d_model=d_model,
                d_ff=d_ff or 4 * d_model,
                dropout_rate=dropout_rate,
                seed=seed + 200,
            )
        else:
            raise ValueError(f"ffn_type must be 'swiglu' or 'gelu', got '{ffn_type}'")

        self.ffn_type = ffn_type

    def forward(
        self,
        x:              np.ndarray,
        enc_memory:     Optional[np.ndarray] = None,
        self_mask:      Optional[np.ndarray] = None,
        cross_mask:     Optional[np.ndarray] = None,
        kv_cache:       Optional[KVCache]    = None,
        rope_offset:    int                  = 0,
        training:       bool                 = False,
        chunk_size:     Optional[int]        = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Forward pass through one transformer block.

        Pre-norm residual pattern:
          x = x + self_attn( norm1(x) )
          x = x + cross_attn( norm_cross(x), enc_memory )   ← if use_cross_attn
          x = x + ffn( norm2(x) )

        Args:
            x:            (batch, seq, d_model)
            enc_memory:   (batch, enc_seq, d_model) encoder states for cross-attn
            self_mask:    Additive mask for self-attention (causal mask for decoder)
            cross_mask:   Additive mask for cross-attention
            kv_cache:     KVCache for this layer (inference only)
            rope_offset:  Starting position index for RoPE (incremental decode)
            training:     Enable dropout
            chunk_size:   Chunked attention for memory efficiency

        Returns:
            output:       (batch, seq, d_model)
            attn_weights: (batch, heads, seq_q, seq_k) or None if chunked
        """
        attn_weights = None

        # ── Sublayer 1: Self-Attention ────────────────────────────────────
        # Pre-norm: normalize x before attention, add result back to x
        attn_out, attn_weights = self.self_attn.forward(
            x_q=self.norm1(x),   # normalized query
            mask=self_mask,
            kv_cache=kv_cache,
            rope_offset=rope_offset,
            training=training,
            chunk_size=chunk_size,
        )
        x = x + attn_out   # residual connection

        # ── Sublayer 2: Cross-Attention (optional) ────────────────────────
        if self.use_cross_attn and enc_memory is not None:
            cross_out, _ = self.cross_attn.forward(
                x_q=self.norm_cross(x),   # normalized decoder state
                x_k=enc_memory,            # encoder keys
                x_v=enc_memory,            # encoder values
                mask=cross_mask,
                training=training,
                chunk_size=chunk_size,
            )
            x = x + cross_out   # residual

        # ── Sublayer 3: Feed-Forward ──────────────────────────────────────
        x = x + self.ffn.forward(self.norm2(x), training=training)

        return x, attn_weights

    def count_parameters(self) -> int:
        """Total parameters in this block."""
        total = (
            self.norm1.count_parameters()     +
            self.self_attn.count_parameters() +
            self.norm2.count_parameters()     +
            self.ffn.count_parameters()
        )
        if self.use_cross_attn:
            total += self.norm_cross.count_parameters() + self.cross_attn.count_parameters()
        return total

    def __repr__(self) -> str:
        mode = "Decoder" if self.use_cross_attn else "Encoder"
        attn = f"GQA(q={self.num_heads}, kv={self.num_kv_heads})"
        return (
            f"TransformerBlock[{self.layer_idx}]({mode}, {attn}, "
            f"FFN={self.ffn_type}, d={self.d_model}, "
            f"params={self.count_parameters():,})"
        )


# ──────────────────────────────────────────────────────────────────────────────
# SELF-TEST
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 68)
    print("  transformer_block.py — Step 18 — Self-Test")
    print("=" * 68)

    rng = np.random.default_rng(42)
    B, S, D, H = 2, 16, 128, 8

    x = rng.standard_normal((B, S, D)).astype(np.float32)

    # ── Encoder block (bidirectional, no mask) ────────────────────────────
    print(f"\n[1] Encoder block — SwiGLU + GQA")
    enc_blk = TransformerBlock(
        d_model=D, num_heads=H, num_kv_heads=2,
        ffn_type="swiglu", layer_idx=0, num_layers=6, seed=0
    )
    out1, w1 = enc_blk.forward(x, training=False)
    print(f"  {enc_blk}")
    print(f"  Input:   {x.shape}  Output: {out1.shape}")
    assert out1.shape == x.shape

    # ── Encoder block with causal mask ────────────────────────────────────
    print(f"\n[2] Causal mask — no future leakage")
    causal = make_causal_mask(S)
    out2, w2 = enc_blk.forward(x, self_mask=causal, training=False)
    future_attn = np.triu(w2[0, 0], k=1).max()
    print(f"  Max future-attending weight: {future_attn:.2e}  (target: ≈ 0)")
    assert future_attn < 1e-6, "Causal mask broken!"

    # ── Full decoder block (self + cross-attention) ───────────────────────
    print(f"\n[3] Full decoder block (SwiGLU + cross-attn + GQA)")
    dec_blk = TransformerBlock(
        d_model=D, num_heads=H, num_kv_heads=2,
        ffn_type="swiglu", use_cross_attn=True,
        layer_idx=1, num_layers=6, seed=10
    )
    enc_mem = rng.standard_normal((B, 20, D)).astype(np.float32)
    out3, w3 = dec_blk.forward(x, enc_memory=enc_mem, self_mask=causal, training=False)
    print(f"  {dec_blk}")
    print(f"  Encoder memory: {enc_mem.shape}  Output: {out3.shape}")
    assert out3.shape == x.shape

    # ── GELU FFN variant ──────────────────────────────────────────────────
    print(f"\n[4] GELU FFN variant")
    gelu_blk = TransformerBlock(d_model=D, num_heads=H, ffn_type="gelu", seed=20)
    out_g, _ = gelu_blk.forward(x, training=False)
    assert out_g.shape == x.shape
    print(f"  GELU block output: {out_g.shape}  ✓")

    # ── KV-cache consistency ──────────────────────────────────────────────
    print(f"\n[5] KV-cache — incremental decode matches full pass")
    test_blk = TransformerBlock(D, num_heads=H, num_kv_heads=2, ffn_type="swiglu", seed=0)

    # Full sequence forward (no cache)
    full_x    = rng.standard_normal((1, 8, D)).astype(np.float32)
    full_mask = make_causal_mask(8)
    out_full, _ = test_blk.forward(full_x, self_mask=full_mask, training=False)

    # Cached: process tokens one at a time
    d_head = D // H
    kv_cache = KVCache(batch_size=1, num_kv_heads=2, max_seq_len=64, d_head=d_head)
    out_cached_last = None
    for i in range(8):
        tok = full_x[:, i:i+1, :]
        out_step, _ = test_blk.forward(
            tok, kv_cache=kv_cache, rope_offset=i, self_mask=None, training=False
        )
        out_cached_last = out_step

    diff = abs(out_full[0, -1] - out_cached_last[0, 0]).max()
    print(f"  Max diff (full vs cached at last pos): {diff:.2e}  (target: < 1e-4)")
    assert diff < 1e-4, f"KV-cache block consistency failed: {diff}"

    # ── Depth scaling ─────────────────────────────────────────────────────
    print(f"\n[6] Depth scaling — output projections scale with layer count")
    blk_shallow = TransformerBlock(D, H, num_layers=2,  layer_idx=0, seed=0)
    blk_deep    = TransformerBlock(D, H, num_layers=24, layer_idx=0, seed=0)
    # Deeper model should have smaller W_O norms (scaled by 1/√(2N))
    norm_shallow = np.abs(blk_shallow.self_attn.W_O).mean()
    norm_deep    = np.abs(blk_deep.self_attn.W_O).mean()
    print(f"  W_O mean |val| — 2-layer:  {norm_shallow:.5f}")
    print(f"  W_O mean |val| — 24-layer: {norm_deep:.5f}  (should be smaller)")
    assert norm_deep < norm_shallow

    # ── Training vs eval ──────────────────────────────────────────────────
    print(f"\n[7] Dropout — stochastic in training, deterministic in eval")
    drop_blk = TransformerBlock(D, H, dropout_rate=0.2, seed=0)
    o_t1 = drop_blk.forward(x, training=True)[0]
    o_t2 = drop_blk.forward(x, training=True)[0]
    o_e1 = drop_blk.forward(x, training=False)[0]
    o_e2 = drop_blk.forward(x, training=False)[0]
    print(f"  Training passes differ:  {not np.allclose(o_t1, o_t2)}")
    print(f"  Eval passes identical:   {np.allclose(o_e1, o_e2)}")

    print("\n  ✓ All transformer block tests passed")
    print("=" * 68)
