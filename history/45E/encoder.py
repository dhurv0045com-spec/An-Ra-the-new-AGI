"""
================================================================================
FILE: encoder.py
PROJECT: Transformer Language Model — 45E v2
STEP: 19 — Encoder Stack
================================================================================

Bidirectional transformer encoder stack.

ARCHITECTURE: LLaMA/Mistral-style (no separate PE — RoPE is in attention)
  Input token IDs
    → Token embedding lookup (scaled by √d_model)
    → Embedding dropout
    → N × TransformerBlock (RMSNorm + GQA/RoPE + SwiGLU, pre-norm)
    → Final RMSNorm
    → Encoder output: (batch, seq, d_model)

KEY DECISION — NO SEPARATE POSITIONAL ENCODING:
  Because RoPE is applied inside the attention mechanism (in multihead.py),
  the encoder doesn't need a separate positional embedding table.
  This simplifies the encoder and reduces parameters.
  Token embeddings remain position-agnostic — only attention is position-aware.

USE CASES:
  - BERT-style masked language modeling (bidirectional)
  - Text classification (encode → pool → classify)
  - Source encoder in seq2seq (translation, summarization)
  - Sentence embeddings (mean/CLS pooling of encoder output)

PADDING MASK:
  Variable-length batches require masking out padding tokens.
  We build a padding mask from the input token IDs:
    mask = -1e9 where token_id == pad_token_id, else 0
  This mask is added to the attention scores before softmax.
================================================================================
"""

import numpy as np
from typing import Optional, List

from transformer_block import TransformerBlock
from layernorm         import RMSNorm


def build_padding_mask(
    token_ids:    np.ndarray,
    pad_token_id: int = 0,
) -> np.ndarray:
    """
    Build an additive attention mask that blocks padding positions.

    Padding tokens (id == pad_token_id) should not be attended to.
    We set their mask value to -1e9, which becomes ≈ 0 after softmax.

    Args:
        token_ids:    (batch, seq_len) integer token IDs
        pad_token_id: Token ID used for padding (typically 0)

    Returns:
        mask: (batch, 1, 1, seq_len) — ready to broadcast over (batch, heads, seq_q, seq_k)
              0.0 for real tokens, -1e9 for padding tokens
    """
    # True where tokens are padding
    is_pad = (token_ids == pad_token_id).astype(np.float32)   # (batch, seq)
    # Scale to large negative to kill after softmax
    mask = is_pad * -1e9                                       # (batch, seq)
    # Add dims for broadcasting over (head, seq_q) dimensions
    return mask[:, np.newaxis, np.newaxis, :]                  # (batch, 1, 1, seq)


class Encoder:
    """
    Transformer encoder: token embedding + N pre-norm GQA/SwiGLU blocks + RMSNorm.

    All positions attend to all positions (no causal masking).
    Positional information comes from RoPE inside each attention layer.

    Args:
        vocab_size:    Token vocabulary size
        d_model:       Embedding / hidden dimension
        num_layers:    Number of stacked transformer blocks
        num_heads:     Query attention heads per block
        num_kv_heads:  KV heads (GQA). None = num_heads (standard MHA)
        d_ff:          FFN hidden dim. None → SwiGLU auto (≈2.67× d_model)
        max_seq_len:   Maximum input sequence length (for RoPE table)
        dropout_rate:  Dropout probability (embeddings + attention + FFN)
        ffn_type:      "swiglu" (default) or "gelu"
        rope_base:     RoPE base frequency (10000; increase for longer context)
        pad_token_id:  Token ID for padding (used to auto-build padding mask)
        seed:          RNG seed
    """

    def __init__(
        self,
        vocab_size:   int,
        d_model:      int,
        num_layers:   int,
        num_heads:    int,
        num_kv_heads: Optional[int] = None,
        d_ff:         Optional[int] = None,
        max_seq_len:  int   = 2048,
        dropout_rate: float = 0.1,
        ffn_type:     str   = "swiglu",
        rope_base:    float = 10000.0,
        pad_token_id: int   = 0,
        seed:         int   = 0,
    ):
        self.d_model      = d_model
        self.num_layers   = num_layers
        self.vocab_size   = vocab_size
        self.max_seq_len  = max_seq_len
        self.dropout_rate = dropout_rate
        self.pad_token_id = pad_token_id
        self.rng          = np.random.default_rng(seed)

        # ── Token embedding table ────────────────────────────────────────
        # d_model^-0.5 initialization: keeps embedding norms consistent
        # with the attention scale factor (which divides by √d_head)
        self.token_embedding = (
            self.rng.standard_normal((vocab_size, d_model)).astype(np.float32)
            * (d_model ** -0.5)
        )

        # ── Stacked transformer blocks ───────────────────────────────────
        # Each block gets a unique seed to ensure weight diversity
        # Layer index passed for depth-aware output projection scaling
        self.blocks: List[TransformerBlock] = [
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                d_ff=d_ff,
                dropout_rate=dropout_rate,
                use_cross_attn=False,     # encoder: self-attention only
                ffn_type=ffn_type,
                max_seq_len=max_seq_len,
                rope_base=rope_base,
                layer_idx=i,
                num_layers=num_layers,
                seed=seed + i * 37,       # deterministic, diverse seeds
            )
            for i in range(num_layers)
        ]

        # ── Final normalization ──────────────────────────────────────────
        # Applied after all blocks — standard in LLaMA/Mistral
        self.final_norm = RMSNorm(d_model)

    def embed(self, token_ids: np.ndarray) -> np.ndarray:
        """
        Map token IDs to embedding vectors, scaled by √d_model.

        Scaling keeps embedding magnitudes comparable to the attention scale
        factor (1/√d_head), preventing the residual stream from shrinking
        relative to the positional information from RoPE.

        Args:
            token_ids: (batch, seq_len) integer token IDs in [0, vocab_size)

        Returns:
            (batch, seq_len, d_model) float32
        """
        embeddings = self.token_embedding[token_ids]    # (batch, seq, d_model)
        return embeddings * np.sqrt(self.d_model)       # scale up

    def forward(
        self,
        token_ids:    np.ndarray,
        mask:         Optional[np.ndarray] = None,
        auto_mask:    bool                 = True,
        training:     bool                 = False,
        chunk_size:   Optional[int]        = None,
    ) -> np.ndarray:
        """
        Full encoder forward pass: tokens → contextual representations.

        Args:
            token_ids:  (batch, seq_len) integer token IDs
            mask:       Optional custom additive mask (batch, 1, 1, seq_len)
                        If None and auto_mask=True, builds padding mask automatically
            auto_mask:  If True, auto-detect padding tokens and build mask
            training:   Enable dropout
            chunk_size: Memory-efficient attention chunk size

        Returns:
            (batch, seq_len, d_model) — contextual token representations
        """
        # ── Step 1: Embed ────────────────────────────────────────────────
        x = self.embed(token_ids)   # (batch, seq, d_model)

        # ── Step 2: Build padding mask if not provided ───────────────────
        if mask is None and auto_mask:
            mask = build_padding_mask(token_ids, self.pad_token_id)

        # ── Step 3: Embedding dropout ────────────────────────────────────
        if training and self.dropout_rate > 0.0:
            keep = 1.0 - self.dropout_rate
            dmask = (self.rng.random(x.shape) < keep).astype(np.float32)
            x = x * dmask / keep

        # ── Step 4: Pass through N transformer blocks ────────────────────
        # Each block: pre-RMSNorm + GQA (with RoPE) + SwiGLU + residuals
        for block in self.blocks:
            x, _ = block.forward(
                x,
                self_mask=mask,
                training=training,
                chunk_size=chunk_size,
            )

        # ── Step 5: Final normalization ──────────────────────────────────
        x = self.final_norm(x)   # (batch, seq, d_model)

        return x.astype(np.float32)

    def encode_mean(self, token_ids: np.ndarray, training: bool = False) -> np.ndarray:
        """
        Encode and mean-pool over non-padding positions → sentence embedding.

        Useful for classification, retrieval, and semantic similarity.

        Args:
            token_ids: (batch, seq_len)

        Returns:
            (batch, d_model) — one vector per sequence
        """
        hidden = self.forward(token_ids, training=training)   # (batch, seq, d_model)

        # Build a mask of non-padding positions (1 = real, 0 = pad)
        not_pad = (token_ids != self.pad_token_id).astype(np.float32)  # (batch, seq)
        not_pad = not_pad[:, :, np.newaxis]   # (batch, seq, 1) for broadcasting

        # Sum over non-pad positions, then divide by count
        sum_hidden = (hidden * not_pad).sum(axis=1)    # (batch, d_model)
        count      = not_pad.sum(axis=1).clip(min=1)   # (batch, 1) — avoid /0
        return sum_hidden / count                       # (batch, d_model)

    def count_parameters(self) -> int:
        """Total learnable parameters."""
        total = self.token_embedding.size
        total += sum(b.count_parameters() for b in self.blocks)
        total += self.final_norm.count_parameters()
        return total

    def __repr__(self) -> str:
        ffn = self.blocks[0].ffn_type if self.blocks else "?"
        kv_h = self.blocks[0].num_kv_heads if self.blocks else "?"
        return (
            f"Encoder(\n"
            f"  vocab={self.vocab_size:,}  d_model={self.d_model}\n"
            f"  layers={self.num_layers}  q_heads={self.blocks[0].num_heads}  kv_heads={kv_h}\n"
            f"  ffn={ffn}  pos=RoPE  norm=RMSNorm\n"
            f"  params={self.count_parameters():,}\n"
            f")"
        )


# ──────────────────────────────────────────────────────────────────────────────
# SELF-TEST
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 68)
    print("  encoder.py — Step 19 — Self-Test")
    print("=" * 68)

    rng = np.random.default_rng(42)
    VOCAB, D, LAYERS, HEADS, KV_HEADS = 1000, 128, 4, 8, 2
    B, S = 2, 20

    encoder = Encoder(
        vocab_size=VOCAB,
        d_model=D,
        num_layers=LAYERS,
        num_heads=HEADS,
        num_kv_heads=KV_HEADS,
        max_seq_len=256,
        dropout_rate=0.1,
        ffn_type="swiglu",
        pad_token_id=0,
    )

    token_ids = rng.integers(1, VOCAB, size=(B, S)).astype(np.int64)

    # ── Forward pass ──────────────────────────────────────────────────────
    print(f"\n[1] Encoder forward pass")
    out = encoder.forward(token_ids, training=False)
    print(f"  Tokens:  {token_ids.shape}  Output: {out.shape}")
    print(f"  dtype:   {out.dtype}")
    print(f"\n{encoder}")
    assert out.shape == (B, S, D)
    assert out.dtype == np.float32

    # ── No NaN/Inf ────────────────────────────────────────────────────────
    print(f"\n[2] Numerical health — no NaN or Inf")
    assert np.isfinite(out).all(), "Output contains NaN or Inf!"
    print(f"  All outputs finite: ✓")

    # ── Padding mask auto-detection ───────────────────────────────────────
    print(f"\n[3] Padding mask — pad_token=0 attended less")
    tokens_padded = token_ids.copy()
    tokens_padded[:, 15:] = 0   # last 5 tokens are padding
    out_padded = encoder.forward(tokens_padded, training=False)
    print(f"  Padded sequence output: {out_padded.shape}  ✓")

    # ── Padding mask function ─────────────────────────────────────────────
    pad_mask = build_padding_mask(tokens_padded, pad_token_id=0)
    print(f"  Padding mask shape: {pad_mask.shape}")
    print(f"  Real token masks to 0: {pad_mask[0, 0, 0, 0] == 0.0}")
    print(f"  Pad token masks to -1e9: {pad_mask[0, 0, 0, 15] < -1e8}")

    # ── Different inputs → different outputs ──────────────────────────────
    print(f"\n[4] Contextual encoding quality")
    token_ids_b = rng.integers(1, VOCAB, size=(B, S)).astype(np.int64)
    out_b = encoder.forward(token_ids_b, training=False)
    diff = abs(out - out_b).mean()
    print(f"  Mean diff (different inputs): {diff:.4f}  (should be > 0)")
    assert diff > 0.1

    # ── Same input always same output (deterministic) ─────────────────────
    print(f"\n[5] Determinism in eval mode")
    out_e1 = encoder.forward(token_ids, training=False)
    out_e2 = encoder.forward(token_ids, training=False)
    print(f"  Two eval passes identical: {np.allclose(out_e1, out_e2)}")
    assert np.allclose(out_e1, out_e2)

    # ── Mean pooling (sentence embedding) ─────────────────────────────────
    print(f"\n[6] Mean-pool sentence embedding")
    sent_emb = encoder.encode_mean(token_ids)
    print(f"  Token output:     {out.shape}")
    print(f"  Sentence emb:     {sent_emb.shape}  ← one vector per sequence")
    assert sent_emb.shape == (B, D)

    # ── Chunked attention ─────────────────────────────────────────────────
    print(f"\n[7] Chunked attention — memory-efficient encoding")
    out_std   = encoder.forward(token_ids, training=False, chunk_size=None)
    out_chunk = encoder.forward(token_ids, training=False, chunk_size=5)
    diff_c = abs(out_std - out_chunk).max()
    print(f"  Max diff (standard vs chunk=5): {diff_c:.2e}  (target: < 1e-5)")
    assert diff_c < 1e-5

    print("\n  ✓ All encoder tests passed")
    print("=" * 68)
