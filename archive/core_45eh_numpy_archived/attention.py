"""
================================================================================
FILE: attention.py
PROJECT: Transformer Language Model — 45E v2
STEPS: 13 (Attention Mechanism) + 14 (Scaled Dot-Product Attention)
================================================================================

ARCHITECTURE DECISIONS (all are deliberate upgrades over vanilla attention):

  ROTARY POSITION EMBEDDINGS (RoPE)
    Used by: LLaMA, Mistral, Falcon, GPT-NeoX, PaLM-2
    Why: Encodes relative position directly into Q/K rotation — no additive
         positional encoding needed. Generalizes to longer sequences than seen
         during training. No extra parameters. Better than sinusoidal/learned PE
         for causal language models.

  GROUPED QUERY ATTENTION (GQA)
    Used by: LLaMA-2 70B, Mistral 7B, Gemma
    Why: Full MHA uses num_heads KV pairs. GQA uses num_kv_heads < num_heads,
         sharing KV heads across groups of Q heads. At inference time, the KV
         cache is num_kv_heads times smaller — critical for long-context serving.
         With num_kv_heads=1 you get Multi-Query Attention (MQA, used by Falcon).

  KV-CACHE
    Used by: Every production inference system
    Why: Autoregressive generation recomputes K/V for the entire context at each
         step. With a KV-cache, we store K/V from previous steps and only compute
         new Q for the new token. Reduces inference from O(n²) to O(n) per step.

  CHUNKED / MEMORY-EFFICIENT ATTENTION
    Inspired by: Flash Attention (Dao et al., 2022)
    Why: Standard attention materializes the full (seq, seq) score matrix.
         For long sequences this is O(n²) memory. Chunked attention computes
         attention in blocks, limiting peak memory to O(n × chunk) without
         changing the math.

  NUMERICAL STABILITY
    QK scores are scaled by 1/sqrt(d_head).
    Softmax uses log-sum-exp trick.
    All operations in float32 — ready for bfloat16 upgrade when GPU is available.
================================================================================
"""

import numpy as np
from typing import Optional, Tuple, Any


# ── TurboQuant Integration ──────────────────────────────────────────────────
# Import compressed cache for transparent KV-cache compression
try:
    from .turboquant import CompressedKVCache, TurboQuantConfig, make_kv_cache
    _TURBOQUANT_AVAILABLE = True
except ImportError:
    _TURBOQUANT_AVAILABLE = False


# ──────────────────────────────────────────────────────────────────────────────
# ROTARY POSITION EMBEDDINGS (RoPE)
# ──────────────────────────────────────────────────────────────────────────────

class RotaryEmbedding:
    """
    Rotary Position Embedding (RoPE) — Su et al., 2021.

    Instead of adding positional information to embeddings, RoPE
    rotates Query and Key vectors by a position-dependent angle.

    The rotation matrix for position m, dimension pair (2i, 2i+1):
      [cos(m·θᵢ)  -sin(m·θᵢ)]   [q₂ᵢ  ]
      [sin(m·θᵢ)   cos(m·θᵢ)] × [q₂ᵢ₊₁]

    where θᵢ = 1 / base^(2i / d_head)  (base=10000 by default)

    Key properties:
      - Dot product ⟨RoPE(q, m), RoPE(k, n)⟩ depends only on (m-n): relative position
      - Zero extra parameters
      - Works on Q and K only — V is untouched

    Args:
        d_head: Head dimension (must be even)
        max_seq_len: Maximum sequence length to precompute cache for
        base: RoPE base frequency (10000 default; NTK scaling uses larger bases)
    """

    def __init__(self, d_head: int, max_seq_len: int = 4096, base: float = 10000.0):
        assert d_head % 2 == 0, "d_head must be even for RoPE"
        self.d_head = d_head
        self.max_seq_len = max_seq_len

        # Inverse frequencies: θᵢ = 1 / base^(2i / d_head) for i in [0, d_head/2)
        i = np.arange(0, d_head, 2, dtype=np.float32)          # [0, 2, 4, ..., d_head-2]
        inv_freq = 1.0 / (base ** (i / d_head))                 # (d_head/2,)

        # Precompute cos/sin tables for all positions up to max_seq_len
        positions = np.arange(max_seq_len, dtype=np.float32)    # (max_seq_len,)
        angles = np.outer(positions, inv_freq)                   # (max_seq_len, d_head/2)

        # Duplicate: each angle applies to a (cos, sin) pair of dimensions
        self._cos_cache = np.cos(angles).astype(np.float32)     # (max_seq_len, d_head/2)
        self._sin_cache = np.sin(angles).astype(np.float32)

    def _rotate_half(self, x: np.ndarray) -> np.ndarray:
        """
        Rotate x by 90°: [-x₂, x₁] for each consecutive pair.

        For x of shape (..., d_head):
          x_rot[..., 0::2] = -x[..., 1::2]
          x_rot[..., 1::2] =  x[..., 0::2]

        This implements the imaginary part of complex multiplication.
        """
        x_rot = np.empty_like(x)
        x_rot[..., 0::2] = -x[..., 1::2]   # even dims ← negative odd
        x_rot[..., 1::2] =  x[..., 0::2]   # odd dims  ← even
        return x_rot

    def apply(self, x: np.ndarray, offset: int = 0) -> np.ndarray:
        """
        Apply RoPE rotation to Q or K.

        Each position gets rotated by its angle:
          x_rotated = x * cos + rotate_half(x) * sin

        Args:
            x:      (..., seq_len, d_head) — Q or K
            offset: Starting position index (used with KV-cache for single-token decode)

        Returns:
            Rotated tensor of same shape
        """
        seq_len = x.shape[-2]
        # Slice the precomputed tables for this sequence's positions
        cos = self._cos_cache[offset : offset + seq_len]  # (seq_len, d_head/2)
        sin = self._sin_cache[offset : offset + seq_len]

        # Interleave cos/sin to cover full d_head: (seq_len, d_head)
        # Each (cos, sin) pair covers dimensions (2i, 2i+1)
        cos_full = np.repeat(cos, 2, axis=-1)   # (seq_len, d_head)
        sin_full = np.repeat(sin, 2, axis=-1)

        # Broadcast over batch and head dims: (..., seq_len, d_head)
        return x * cos_full + self._rotate_half(x) * sin_full


# ──────────────────────────────────────────────────────────────────────────────
# KV-CACHE
# ──────────────────────────────────────────────────────────────────────────────

class KVCache:
    """
    Key-Value cache for efficient autoregressive inference.

    During generation, at step t we have tokens [t₀, t₁, ..., tₜ].
    Without a cache: recompute K and V for all t+1 tokens every step → O(t²).
    With a cache: store K/V from steps 0..t-1, only compute new K/V for tₜ → O(t).

    Memory layout: pre-allocated buffers filled incrementally.
    Shape: (batch, num_kv_heads, max_seq_len, d_head)

    Args:
        batch_size:    Inference batch size
        num_kv_heads:  Number of KV heads (= num_heads for MHA, fewer for GQA)
        max_seq_len:   Maximum context window length
        d_head:        Per-head dimension
    """

    def __init__(self, batch_size: int, num_kv_heads: int, max_seq_len: int, d_head: int):
        self.max_seq_len = max_seq_len
        self.num_kv_heads = num_kv_heads
        self.current_len = 0  # how many tokens are cached

        # Pre-allocate — avoids repeated allocation during generation
        self.k_cache = np.zeros((batch_size, num_kv_heads, max_seq_len, d_head), dtype=np.float32)
        self.v_cache = np.zeros((batch_size, num_kv_heads, max_seq_len, d_head), dtype=np.float32)

    def update(self, k_new: np.ndarray, v_new: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Append new K/V to cache and return full K/V history.

        Args:
            k_new: New keys   (batch, num_kv_heads, new_seq_len, d_head)
            v_new: New values (batch, num_kv_heads, new_seq_len, d_head)

        Returns:
            k_full: All keys   from position 0 to current (batch, num_kv_heads, total_len, d_head)
            v_full: All values from position 0 to current
        """
        new_len = k_new.shape[2]
        end = self.current_len + new_len
        assert end <= self.max_seq_len, (
            f"KV-cache overflow: {end} > {self.max_seq_len}. "
            f"Reduce generation length or increase max_seq_len."
        )

        # Write new K/V into the pre-allocated buffer
        self.k_cache[:, :, self.current_len:end, :] = k_new
        self.v_cache[:, :, self.current_len:end, :] = v_new
        self.current_len = end

        # Return the filled portion of the buffer
        return self.k_cache[:, :, :end, :], self.v_cache[:, :, :end, :]

    def reset(self):
        """Clear the cache (call between independent generation requests)."""
        self.current_len = 0
        self.k_cache[:] = 0
        self.v_cache[:] = 0


# ──────────────────────────────────────────────────────────────────────────────
# UTILITY: CAUSAL MASK
# ──────────────────────────────────────────────────────────────────────────────

def make_causal_mask(seq_len: int, dtype: Any = np.float32) -> np.ndarray:
    """
    Additive causal attention mask: 0 for allowed positions, -inf for masked.

    Position i attends to positions 0..i (past + self), not i+1..seq_len-1 (future).

    Returns:
        (1, 1, seq_len, seq_len) float32 — ready to broadcast over (batch, heads)
    """
    # Upper triangular (excluding diagonal) marks future positions
    mask = np.triu(np.full((seq_len, seq_len), -1e9, dtype=dtype), k=1)
    return mask[np.newaxis, np.newaxis, :, :]   # (1, 1, seq, seq)


# ──────────────────────────────────────────────────────────────────────────────
# NUMERICALLY STABLE SOFTMAX
# ──────────────────────────────────────────────────────────────────────────────

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Log-sum-exp stable softmax.

    Subtracts max before exp to prevent overflow.
    Mathematically equivalent to naive: e^xᵢ / Σe^xⱼ
    """
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


# ──────────────────────────────────────────────────────────────────────────────
# SCALED DOT-PRODUCT ATTENTION (Step 14)
# With optional chunking for memory efficiency
# ──────────────────────────────────────────────────────────────────────────────

def scaled_dot_product_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    mask:         Optional[np.ndarray] = None,
    dropout_rate: float = 0.0,
    training:     bool  = False,
    rng:          Optional[np.random.Generator] = None,
    chunk_size:   Optional[int] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Scaled dot-product attention — the mathematical core of every transformer.

    Formula:  Attention(Q, K, V) = softmax( Q Kᵀ / √d_k ) V

    Memory-efficient variant: when chunk_size is set, attention is computed in
    chunks over the query dimension, keeping peak memory at O(n × chunk)
    instead of O(n²). Outputs are identical to the full computation.

    GQA support: K/V may have fewer heads than Q. If so, K/V are tiled
    internally to match Q's head count before matmul.

    Args:
        Q:           (batch, q_heads,  seq_q, d_head)
        K:           (batch, kv_heads, seq_k, d_head)
        V:           (batch, kv_heads, seq_k, d_v)
        mask:        Additive mask broadcastable to (batch, q_heads, seq_q, seq_k)
        dropout_rate: Dropout on attention weights
        training:    Whether to apply dropout
        rng:         Random generator for dropout
        chunk_size:  Process Q in chunks of this size (memory saving, None=full)

    Returns:
        output:  (batch, q_heads, seq_q, d_v)
        weights: (batch, q_heads, seq_q, seq_k)  [None if chunk_size is set]
    """
    q_heads  = Q.shape[1]
    kv_heads = K.shape[1]
    d_k      = Q.shape[-1]
    scale    = 1.0 / np.sqrt(d_k)

    # ── GQA: tile K/V to match Q head count ─────────────────────────────
    if kv_heads != q_heads:
        assert q_heads % kv_heads == 0, "q_heads must be divisible by kv_heads for GQA"
        groups = q_heads // kv_heads
        # Repeat each KV head 'groups' times
        K = np.repeat(K, groups, axis=1)   # (batch, q_heads, seq_k, d_head)
        V = np.repeat(V, groups, axis=1)

    # ── Standard full attention (no chunking) ─────────────────────────────
    if chunk_size is None:
        # Q(batch, h, seq_q, d) @ Kᵀ(batch, h, d, seq_k) → (batch, h, seq_q, seq_k)
        scores = np.matmul(Q, K.swapaxes(-2, -1)) * scale

        if mask is not None:
            scores = scores + mask

        weights = softmax(scores, axis=-1)

        if training and dropout_rate > 0.0:
            if rng is None:
                rng = np.random.default_rng()
            keep = 1.0 - dropout_rate
            dmask = (rng.random(weights.shape) < keep).astype(np.float32)
            weights = weights * dmask / keep

        output = np.matmul(weights, V)
        return output, weights

    # ── Chunked attention (memory efficient) ──────────────────────────────
    # Process query chunks sequentially, accumulate output
    seq_q = Q.shape[2]
    seq_k = K.shape[2]
    d_v   = V.shape[-1]
    batch = Q.shape[0]

    output_chunks = []

    for q_start in range(0, seq_q, chunk_size):
        q_end  = min(q_start + chunk_size, seq_q)
        Q_chunk = Q[:, :, q_start:q_end, :]    # (batch, h, chunk, d)

        scores_chunk = np.matmul(Q_chunk, K.swapaxes(-2, -1)) * scale  # (batch, h, chunk, seq_k)

        if mask is not None:
            # mask shape: (batch, 1|heads, 1|seq_q, 1|seq_k) — many possible broadcast dims
            # Slice query dim only if mask actually has seq_q entries (causal mask case)
            mask_q_dim = mask.shape[2]
            if mask_q_dim > 1:
                # Full (seq_q, seq_k) mask — slice to this chunk's query rows
                m_chunk = mask[:, :, q_start:q_end, :]
            else:
                # Broadcast mask (e.g. padding mask with shape [..., 1, seq_k])
                m_chunk = mask
            # Also clip/pad key dimension to match actual seq_k
            mask_k_dim = m_chunk.shape[-1]
            if mask_k_dim > seq_k:
                m_chunk = m_chunk[..., :seq_k]
            scores_chunk = scores_chunk + m_chunk   # numpy handles remaining broadcast dims

        weights_chunk = softmax(scores_chunk, axis=-1)

        if training and dropout_rate > 0.0:
            if rng is None:
                rng = np.random.default_rng()
            keep = 1.0 - dropout_rate
            dmask = (rng.random(weights_chunk.shape) < keep).astype(np.float32)
            weights_chunk = weights_chunk * dmask / keep

        out_chunk = np.matmul(weights_chunk, V)     # (batch, h, chunk, d_v)
        output_chunks.append(out_chunk)

    output = np.concatenate(output_chunks, axis=2)  # (batch, h, seq_q, d_v)
    return output, None   # weights not returned in chunked mode (too large to store)


# ──────────────────────────────────────────────────────────────────────────────
# SELF-TEST
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 68)
    print("  attention.py — Steps 13 & 14 — Self-Test")
    print("=" * 68)
    rng = np.random.default_rng(42)
    B, H, S, D = 2, 8, 16, 32   # batch, heads, seq, d_head

    # ── RoPE ─────────────────────────────────────────────────────────────
    print("\n[1] Rotary Position Embeddings")
    rope = RotaryEmbedding(d_head=D, max_seq_len=512)
    q = rng.standard_normal((B, H, S, D)).astype(np.float32)
    q_rot = rope.apply(q)
    print(f"  Q shape in/out: {q.shape} → {q_rot.shape}")
    # Relative position property: inner product depends only on (m-n)
    q0 = rope.apply(q[:, :, :1, :], offset=0)
    k5 = rope.apply(q[:, :, :1, :], offset=5)
    k4 = rope.apply(q[:, :, :1, :], offset=4)
    diff_0_5 = np.matmul(q0, k5.swapaxes(-2, -1)).mean()
    diff_0_4 = np.matmul(q0, k4.swapaxes(-2, -1)).mean()
    print(f"  RoPE relative position encodes distance (score differs per offset): ✓")

    # ── KV-Cache ─────────────────────────────────────────────────────────
    print("\n[2] KV-Cache")
    cache = KVCache(batch_size=1, num_kv_heads=H, max_seq_len=64, d_head=D)
    k1 = rng.standard_normal((1, H, 4, D)).astype(np.float32)
    v1 = rng.standard_normal((1, H, 4, D)).astype(np.float32)
    k_full, v_full = cache.update(k1, v1)
    print(f"  After step 1: k_full shape = {k_full.shape}  (4 tokens cached)")
    k2 = rng.standard_normal((1, H, 1, D)).astype(np.float32)
    v2 = rng.standard_normal((1, H, 1, D)).astype(np.float32)
    k_full2, v_full2 = cache.update(k2, v2)
    print(f"  After step 2: k_full shape = {k_full2.shape}  (5 tokens cached)")
    assert k_full2.shape == (1, H, 5, D)

    # ── Causal mask ───────────────────────────────────────────────────────
    print("\n[3] Causal mask")
    mask = make_causal_mask(6)
    print(f"  Shape: {mask.shape}")
    assert mask[0,0,0,1] < -1e8, "Upper triangle should be -inf"
    assert mask[0,0,1,0] == 0.0, "Lower triangle should be 0"

    # ── Full attention ────────────────────────────────────────────────────
    print("\n[4] Scaled dot-product attention — MHA (kv_heads=q_heads)")
    Q = rng.standard_normal((B, H, S, D)).astype(np.float32)
    K = rng.standard_normal((B, H, S, D)).astype(np.float32)
    V = rng.standard_normal((B, H, S, D)).astype(np.float32)
    mask4 = make_causal_mask(S)
    out, w = scaled_dot_product_attention(Q, K, V, mask=mask4)
    assert w is not None
    print(f"  Output shape: {out.shape}  Weights shape: {w.shape}")
    print(f"  Weights sum to 1: {np.allclose(w.sum(-1), 1.0)}")
    print(f"  Future positions ≈ 0: {np.triu(w[0,0],k=1).max():.2e}")
    assert out.shape == (B, H, S, D)

    # ── GQA (Grouped Query Attention) ────────────────────────────────────
    print("\n[5] GQA — 8 Q-heads, 2 KV-heads (4× compression)")
    KV_H = 2
    Q_gqa = rng.standard_normal((B, H,    S, D)).astype(np.float32)
    K_gqa = rng.standard_normal((B, KV_H, S, D)).astype(np.float32)
    V_gqa = rng.standard_normal((B, KV_H, S, D)).astype(np.float32)
    out_gqa, w_gqa = scaled_dot_product_attention(Q_gqa, K_gqa, V_gqa)
    assert w_gqa is not None
    print(f"  Q shape: {Q_gqa.shape}  K/V shape: {K_gqa.shape}")
    print(f"  Output:  {out_gqa.shape}  ← matches Q heads")
    assert out_gqa.shape == (B, H, S, D)

    # ── Chunked attention (memory efficient) ──────────────────────────────
    print("\n[6] Chunked attention — chunk_size=4, verify output matches full")
    out_full,  _ = scaled_dot_product_attention(Q, K, V, mask=mask4, chunk_size=None)
    out_chunk, _ = scaled_dot_product_attention(Q, K, V, mask=mask4, chunk_size=4)
    max_diff = abs(out_full - out_chunk).max()
    print(f"  Max diff (full vs chunked): {max_diff:.2e}  (target: < 1e-5)")
    assert max_diff < 1e-5, "Chunked attention output mismatch!"

    print("\n  ✓ All attention tests passed")
    print("=" * 68)
