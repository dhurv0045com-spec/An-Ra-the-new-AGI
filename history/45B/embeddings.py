"""
model/embeddings.py — Steps 11–12
Token embeddings + positional encodings.
Implements: learned embeddings, sinusoidal PE, RoPE (Rotary Position Embedding).
RoPE is the modern choice — used in LLaMA, Mistral, Gemma.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


# ─────────────────────────────────────────────
# STEP 11 — Token embeddings
# ─────────────────────────────────────────────

class TokenEmbedding(nn.Module):
    """
    Learnable token embedding table.
    Optionally scales by sqrt(d_model) as in the original transformer paper.
    """
    def __init__(self, vocab_size: int, d_model: int, pad_idx: int = 0,
                 scale: bool = True):
        super().__init__()
        self.d_model = d_model
        self.scale = scale
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, mean=0, std=self.d_model**-0.5)
        if self.embedding.padding_idx is not None:
            with torch.no_grad():
                self.embedding.weight[self.embedding.padding_idx].fill_(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x)
        if self.scale:
            emb = emb * math.sqrt(self.d_model)
        return emb


# ─────────────────────────────────────────────
# STEP 12 — Positional encodings
# ─────────────────────────────────────────────

class SinusoidalPositionalEncoding(nn.Module):
    """
    Fixed sinusoidal PE from "Attention Is All You Need".
    Deterministic, no parameters, infinite extrapolation in theory.
    """
    def __init__(self, d_model: int, max_seq_len: int = 4096,
                 dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_seq_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, d_model)"""
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    """
    Learned absolute position embeddings — used in BERT, GPT-2.
    Simple and effective for fixed-length contexts.
    """
    def __init__(self, d_model: int, max_seq_len: int = 2048,
                 dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.pe = nn.Embedding(max_seq_len, d_model)
        nn.init.normal_(self.pe.weight, mean=0, std=0.02)

    def forward(self, x: torch.Tensor,
                offset: int = 0) -> torch.Tensor:
        """x: (batch, seq_len, d_model)"""
        seq_len = x.size(1)
        positions = torch.arange(offset, offset + seq_len,
                                 device=x.device).unsqueeze(0)
        return self.dropout(x + self.pe(positions))


class RotaryPositionalEncoding(nn.Module):
    """
    RoPE — Rotary Position Embedding (Su et al., 2021).
    Encodes position by rotating Q and K before attention.
    Used in: LLaMA, Mistral, Gemma, Falcon, Qwen.

    Advantages over sinusoidal/learned PE:
    - Relative position information baked into attention
    - Extends naturally to longer sequences (with scaling tricks)
    - No extra parameters
    - Better length generalization
    """
    def __init__(self, head_dim: int, max_seq_len: int = 4096,
                 base: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute frequencies
        inv_freq = 1.0 / (base ** (
            torch.arange(0, head_dim, 2).float() / head_dim
        ))
        self.register_buffer("inv_freq", inv_freq)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)          # (seq, head_dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)        # (seq, head_dim)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate the second half of the last dimension."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor,
                offset: int = 0) -> tuple:
        """
        q, k: (batch, heads, seq_len, head_dim)
        Returns rotated q and k.
        """
        seq_len = q.shape[2]
        if offset + seq_len > self.max_seq_len:
            self._build_cache(offset + seq_len)

        cos = self.cos_cached[:, :, offset:offset+seq_len, :]
        sin = self.sin_cached[:, :, offset:offset+seq_len, :]

        q_rot = q * cos + self._rotate_half(q) * sin
        k_rot = k * cos + self._rotate_half(k) * sin
        return q_rot, k_rot


# ─────────────────────────────────────────────
# Combined input embedding module
# ─────────────────────────────────────────────

class InputEmbedding(nn.Module):
    """
    Full input embedding: tokens → d_model vectors.
    Uses either learned or sinusoidal PE.
    For RoPE models, PE is applied inside the attention module.
    """
    def __init__(self, vocab_size: int, d_model: int,
                 max_seq_len: int = 2048, dropout: float = 0.1,
                 pad_idx: int = 0, pe_type: str = "learned"):
        super().__init__()
        self.token_emb = TokenEmbedding(vocab_size, d_model, pad_idx)

        if pe_type == "sinusoidal":
            self.pos_enc = SinusoidalPositionalEncoding(d_model, max_seq_len, dropout)
        elif pe_type == "learned":
            self.pos_enc = LearnedPositionalEncoding(d_model, max_seq_len, dropout)
        elif pe_type == "none":
            # RoPE — applied inside attention, not here
            self.pos_enc = nn.Dropout(dropout)
        else:
            raise ValueError(f"Unknown pe_type '{pe_type}'")

        self.pe_type = pe_type

    def forward(self, x: torch.Tensor,
                offset: int = 0) -> torch.Tensor:
        """
        x: (batch, seq_len) token ids
        Returns: (batch, seq_len, d_model)
        """
        emb = self.token_emb(x)
        if self.pe_type == "learned":
            return self.pos_enc(emb, offset=offset)
        else:
            return self.pos_enc(emb)


# ─────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("Steps 11–12: Embedding + positional encoding checks")
    B, T, d = 2, 16, 128
    V = 1000

    tokens = torch.randint(0, V, (B, T))

    # Test all PE types
    for pe_type in ["learned", "sinusoidal"]:
        ie = InputEmbedding(V, d, max_seq_len=64, dropout=0.0, pe_type=pe_type)
        out = ie(tokens)
        print(f"  {pe_type:12s} → shape {tuple(out.shape)}")
        assert out.shape == (B, T, d)

    # Test RoPE
    rope = RotaryPositionalEncoding(head_dim=d, max_seq_len=64)
    q = torch.randn(B, 4, T, d)
    k = torch.randn(B, 4, T, d)
    q_r, k_r = rope(q, k)
    assert q_r.shape == q.shape
    print(f"  RoPE            → shape {tuple(q_r.shape)}")

    print("✓ Steps 11–12 verified")
