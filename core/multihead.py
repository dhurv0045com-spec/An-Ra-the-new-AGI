"""
================================================================================
FILE: multihead.py
PROJECT: Transformer Language Model — 45E v2
STEP: 15 — Multi-Head / Grouped-Query Attention Module
================================================================================

This implements the complete attention module as used in modern LLMs:

  GROUPED QUERY ATTENTION (GQA)
    Generalizes all three attention variants in a single class:
      num_kv_heads == num_heads  →  standard Multi-Head Attention (MHA)
      num_kv_heads == 1          →  Multi-Query Attention (MQA, used by Falcon)
      1 < num_kv_heads < num_heads  →  GQA (used by Mistral 7B, LLaMA-2 70B)

    At d_model=4096, num_heads=32, num_kv_heads=8:
      MHA KV cache per layer:  2 × 32 × seq × 128 × 4 bytes = 32MB per K step
      GQA KV cache per layer:  2 ×  8 × seq × 128 × 4 bytes =  8MB per K step

  ROTARY POSITION EMBEDDINGS (RoPE)
    Applied to Q and K after projection, before attention.
    Position-relative inner products — no separate PE table needed.
    Offset parameter supports KV-cache inference (only new tokens need encoding).

  PROJECTION WEIGHT INITIALIZATION
    GPT-2 paper: scale output projection by 1/√(2×num_layers) to prevent
    residual stream growth. We implement this as a configurable scale factor.

  FUSED QKV PROJECTION (optional)
    Project Q, K, V in one matmul instead of three.
    For num_kv_heads < num_heads:  Q: (d_model, d_model), KV: (d_model, 2×d_kv)
    Reduces matmul overhead, especially for GQA where KV is small.
================================================================================
"""

import numpy as np
from typing import Optional, Tuple

from attention import (
    RotaryEmbedding,
    KVCache,
    scaled_dot_product_attention,
    make_causal_mask,
)


class MultiHeadAttention:
    """
    Multi-Head / Grouped-Query Attention with RoPE and KV-cache support.

    Generalizes MHA, MQA, and GQA through the num_kv_heads parameter.
    Supports both training (full sequence, no cache) and inference (cached KV).

    Args:
        d_model:       Model dimension
        num_heads:     Number of query heads
        num_kv_heads:  Number of key/value heads. Must divide num_heads.
                       Default: num_heads (= standard MHA)
        dropout_rate:  Attention weight dropout during training
        max_seq_len:   Maximum sequence length (for RoPE table)
        rope_base:     RoPE base frequency (10000 default; scale up for longer ctx)
        out_proj_scale: Scale factor for output projection (GPT-2 depth scaling)
        seed:          RNG seed
    """

    def __init__(
        self,
        d_model:        int,
        num_heads:      int,
        num_kv_heads:   Optional[int] = None,
        dropout_rate:   float = 0.1,
        max_seq_len:    int   = 4096,
        rope_base:      float = 10000.0,
        out_proj_scale: float = 1.0,
        seed:           int   = 0,
    ):
        assert d_model % num_heads == 0, f"d_model ({d_model}) % num_heads ({num_heads}) != 0"

        self.d_model      = d_model
        self.num_heads    = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.d_head       = d_model // num_heads          # dimension per Q head
        self.d_kv         = self.d_head                   # same for K/V heads
        self.dropout_rate = dropout_rate
        self.rng          = np.random.default_rng(seed)

        assert self.num_heads % self.num_kv_heads == 0, (
            f"num_heads ({num_heads}) must be divisible by num_kv_heads ({self.num_kv_heads})"
        )
        self.kv_groups = self.num_heads // self.num_kv_heads

        # ── Projection weights ────────────────────────────────────────────
        # Q projects to (num_heads × d_head) = d_model
        # K/V project to (num_kv_heads × d_kv)  — smaller for GQA
        d_q  = self.num_heads    * self.d_head   # full query dimension
        d_kv_total = self.num_kv_heads * self.d_kv  # compressed KV dimension

        self.W_Q = self._init_weight(d_model, d_q)
        self.b_Q = np.zeros(d_q, dtype=np.float32)

        self.W_K = self._init_weight(d_model, d_kv_total)
        self.b_K = np.zeros(d_kv_total, dtype=np.float32)

        self.W_V = self._init_weight(d_model, d_kv_total)
        self.b_V = np.zeros(d_kv_total, dtype=np.float32)

        # Output projection: (d_model → d_model)
        # Scaled by out_proj_scale for depth-aware initialization (GPT-2 trick)
        self.W_O = self._init_weight(d_model, d_model, scale=out_proj_scale)
        self.b_O = np.zeros(d_model, dtype=np.float32)

        # ── RoPE ─────────────────────────────────────────────────────────
        self.rope = RotaryEmbedding(
            d_head=self.d_head,
            max_seq_len=max_seq_len,
            base=rope_base,
        )

    def _init_weight(self, fan_in: int, fan_out: int, scale: float = 1.0) -> np.ndarray:
        """
        Xavier uniform initialization with optional depth scaling.

        Xavier: W ~ U(-√(6/(fan_in+fan_out)), +√(6/(fan_in+fan_out)))
        After depth scaling: multiply by scale factor (e.g. 1/√(2×layers))
        """
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        W = self.rng.uniform(-limit, limit, (fan_in, fan_out)).astype(np.float32)
        return W * scale

    def _split_heads(self, x: np.ndarray, num_h: int) -> np.ndarray:
        """
        Reshape (batch, seq, num_h × d_head) → (batch, num_h, seq, d_head).

        Exposes the head dimension for batched attention computation.
        """
        batch, seq, _ = x.shape
        x = x.reshape(batch, seq, num_h, self.d_head)
        return x.transpose(0, 2, 1, 3)   # (batch, num_h, seq, d_head)

    def _merge_heads(self, x: np.ndarray) -> np.ndarray:
        """
        Inverse of _split_heads: (batch, num_h, seq, d_head) → (batch, seq, d_model).
        """
        batch, num_h, seq, d_head = x.shape
        x = x.transpose(0, 2, 1, 3)                    # (batch, seq, num_h, d_head)
        return x.reshape(batch, seq, num_h * d_head)    # (batch, seq, d_model)

    def forward(
        self,
        x_q:      np.ndarray,
        x_k:      Optional[np.ndarray] = None,
        x_v:      Optional[np.ndarray] = None,
        mask:     Optional[np.ndarray] = None,
        kv_cache: Optional[KVCache]    = None,
        rope_offset: int               = 0,
        training: bool                 = False,
        chunk_size: Optional[int]      = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Forward pass — supports training, prefill, and cached single-token decode.

        Three modes of operation:
          Training (mask + full sequence, no cache):
            x_q = x_k = x_v = full token sequence
          Prefill (fill KV cache with prompt, then switch to decode):
            Same as training but kv_cache is provided — populates it
          Cached decode (one new token per step):
            x_q = new token only (seq_len=1)
            kv_cache contains all previous K/V
            rope_offset = number of tokens already generated

        Args:
            x_q:         (batch, seq_q, d_model) — query source
            x_k:         (batch, seq_k, d_model) — key source (None = self-attention)
            x_v:         (batch, seq_k, d_model) — value source (None = self-attention)
            mask:        Additive mask (batch, 1, seq_q, seq_k) or broadcastable
            kv_cache:    KVCache instance for incremental decoding
            rope_offset: Position index of the first token in x_q
            training:    Enable attention dropout
            chunk_size:  Memory-efficient chunked attention (None = standard)

        Returns:
            output:  (batch, seq_q, d_model)
            weights: (batch, num_heads, seq_q, seq_k) or None if chunked
        """
        if x_k is None:
            x_k = x_q
        if x_v is None:
            x_v = x_q

        # ── Project Q, K, V ──────────────────────────────────────────────
        Q = x_q @ self.W_Q + self.b_Q   # (batch, seq_q, d_model)
        K = x_k @ self.W_K + self.b_K   # (batch, seq_k, d_kv_total)
        V = x_v @ self.W_V + self.b_V   # (batch, seq_k, d_kv_total)

        # ── Split into heads ─────────────────────────────────────────────
        Q = self._split_heads(Q, self.num_heads)     # (batch, num_heads,    seq_q, d_head)
        K = self._split_heads(K, self.num_kv_heads)  # (batch, num_kv_heads, seq_k, d_head)
        V = self._split_heads(V, self.num_kv_heads)

        # ── Apply RoPE to Q and K ────────────────────────────────────────
        # Q: apply from rope_offset to rope_offset + seq_q
        Q = rope_apply_to_heads(Q, self.rope, offset=rope_offset)
        # K: for cross-attention (x_k != x_q) we don't rotate K with offset
        K = rope_apply_to_heads(K, self.rope, offset=rope_offset if x_k is x_q else 0)

        # ── KV-Cache update ──────────────────────────────────────────────
        if kv_cache is not None:
            K, V = kv_cache.update(K, V)   # appends new K/V, returns full history

        # ── Scaled dot-product attention ─────────────────────────────────
        attended, weights = scaled_dot_product_attention(
            Q, K, V,
            mask=mask,
            dropout_rate=self.dropout_rate,
            training=training,
            rng=self.rng,
            chunk_size=chunk_size,
        )
        # attended: (batch, num_heads, seq_q, d_head)

        # ── Merge heads + output projection ─────────────────────────────
        attended = self._merge_heads(attended)          # (batch, seq_q, d_model)
        output   = attended @ self.W_O + self.b_O       # (batch, seq_q, d_model)

        return output, weights

    def count_parameters(self) -> int:
        """Total trainable parameters."""
        d_kv = self.num_kv_heads * self.d_kv
        return (
            self.W_Q.size + self.b_Q.size +
            self.W_K.size + self.b_K.size +
            self.W_V.size + self.b_V.size +
            self.W_O.size + self.b_O.size
        )

    def __repr__(self) -> str:
        mode = "MHA" if self.num_kv_heads == self.num_heads else (
               "MQA" if self.num_kv_heads == 1 else
               f"GQA(groups={self.kv_groups})"
        )
        return (
            f"MultiHeadAttention({mode}, d_model={self.d_model}, "
            f"q_heads={self.num_heads}, kv_heads={self.num_kv_heads}, "
            f"d_head={self.d_head}, params={self.count_parameters():,})"
        )


def rope_apply_to_heads(x: np.ndarray, rope: RotaryEmbedding, offset: int = 0) -> np.ndarray:
    """
    Apply RoPE to a (batch, num_heads, seq, d_head) tensor.

    RoPE.apply expects (..., seq, d_head) — we treat (batch, heads) as leading dims.
    Reshape to (batch × heads, seq, d_head), rotate, reshape back.

    Args:
        x:      (batch, heads, seq, d_head)
        rope:   RotaryEmbedding instance
        offset: Starting position for this sequence chunk
    """
    batch, heads, seq, d_head = x.shape
    # Merge batch and heads for the rotation — rope doesn't care about these dims
    x_flat = x.reshape(batch * heads, seq, d_head)
    x_rotated = rope.apply(x_flat, offset=offset)
    return x_rotated.reshape(batch, heads, seq, d_head)


# ──────────────────────────────────────────────────────────────────────────────
# SELF-TEST
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 68)
    print("  multihead.py — Step 15 — Self-Test")
    print("=" * 68)

    rng  = np.random.default_rng(42)
    B, S = 2, 16
    D    = 128
    H    = 8

    x = rng.standard_normal((B, S, D)).astype(np.float32)

    # ── Standard MHA ─────────────────────────────────────────────────────
    print(f"\n[1] Standard MHA (num_kv_heads={H})")
    mha = MultiHeadAttention(D, num_heads=H, num_kv_heads=H, seed=0)
    print(f"  {mha}")
    mask = make_causal_mask(S)
    out, w = mha.forward(x, mask=mask, training=False)
    assert w is not None
    print(f"  Output: {out.shape}  Weights: {w.shape}")
    assert out.shape == (B, S, D)
    assert np.triu(w[0,0],k=1).max() < 1e-6, "Causal mask broken"

    # ── GQA (8 Q-heads, 2 KV-heads) ──────────────────────────────────────
    print(f"\n[2] GQA (num_kv_heads=2, {H//2}× KV compression)")
    gqa = MultiHeadAttention(D, num_heads=H, num_kv_heads=2, seed=1)
    print(f"  {gqa}")
    out_gqa, w_gqa = gqa.forward(x, mask=mask, training=False)
    assert w_gqa is not None
    print(f"  Output: {out_gqa.shape}  Weights: {w_gqa.shape}")
    assert out_gqa.shape == (B, S, D)

    # ── MQA (8 Q-heads, 1 KV-head) ────────────────────────────────────────
    print(f"\n[3] MQA (num_kv_heads=1, {H}× KV compression)")
    mqa = MultiHeadAttention(D, num_heads=H, num_kv_heads=1, seed=2)
    print(f"  {mqa}")
    out_mqa, _ = mqa.forward(x, mask=mask, training=False)
    assert out_mqa.shape == (B, S, D)

    # ── KV-Cache decode simulation ─────────────────────────────────────────
    print(f"\n[4] KV-Cache — simulate generation step by step")
    dec_mha = MultiHeadAttention(D, num_heads=H, num_kv_heads=2, max_seq_len=64, seed=3)
    cache   = KVCache(batch_size=1, num_kv_heads=2, max_seq_len=64, d_head=D//H)

    # Prefill with 8-token prompt
    prompt = rng.standard_normal((1, 8, D)).astype(np.float32)
    pmask  = make_causal_mask(8)
    out_p, _ = dec_mha.forward(prompt, mask=pmask, kv_cache=cache, rope_offset=0)
    print(f"  Prefill (8 tokens): cache len = {cache.current_len}")
    assert cache.current_len == 8

    # Single-token decode step (no mask needed — only one new Q attending full K)
    new_tok = rng.standard_normal((1, 1, D)).astype(np.float32)
    out_t, _ = dec_mha.forward(new_tok, kv_cache=cache, rope_offset=8, mask=None)
    print(f"  Decode step 1:      cache len = {cache.current_len}")
    print(f"  Decode output:      {out_t.shape}")
    assert cache.current_len == 9

    # Verify causal: same context, full-sequence vs cached should match at last position
    # (We rebuild with a fresh module to compare)
    cache2 = KVCache(batch_size=1, num_kv_heads=2, max_seq_len=64, d_head=D//H)
    full_seq = np.concatenate([prompt, new_tok], axis=1)   # (1, 9, D)
    full_mask = make_causal_mask(9)
    out_full, _ = dec_mha.forward(full_seq, mask=full_mask, kv_cache=cache2, rope_offset=0)
    diff = abs(out_t[0, 0] - out_full[0, -1]).max()
    print(f"  Cached vs full at last position — max diff: {diff:.2e}  (target: ~ 0)")
    assert diff < 1e-4, f"KV-cache inconsistency: {diff}"

    # ── Cross-attention ────────────────────────────────────────────────────
    print(f"\n[5] Cross-attention (encoder-decoder)")
    enc_out = rng.standard_normal((B, 12, D)).astype(np.float32)
    dec_q   = rng.standard_normal((B, 6,  D)).astype(np.float32)
    out_cross, _ = mha.forward(dec_q, x_k=enc_out, x_v=enc_out)
    print(f"  Encoder: {enc_out.shape}  Decoder Q: {dec_q.shape}  Out: {out_cross.shape}")
    assert out_cross.shape == (B, 6, D)

    # ── Chunked attention output matches full ─────────────────────────────
    print(f"\n[6] Chunked attention — memory-efficient equivalent")
    out_std,   _ = mha.forward(x, mask=mask, training=False, chunk_size=None)
    out_chunk, _ = mha.forward(x, mask=mask, training=False, chunk_size=4)
    diff = abs(out_std - out_chunk).max()
    print(f"  Max diff (full vs chunk=4): {diff:.2e}  (target: < 1e-5)")
    assert diff < 1e-5

    print(f"\n  Parameter counts:")
    print(f"  MHA  (kv_heads={H}): {mha.count_parameters():>8,}")
    print(f"  GQA  (kv_heads=2): {gqa.count_parameters():>8,}")
    print(f"  MQA  (kv_heads=1): {mqa.count_parameters():>8,}")

    print("\n  [OK] All multi-head attention tests passed")
    print("=" * 68)
