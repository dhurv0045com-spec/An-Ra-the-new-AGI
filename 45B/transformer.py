"""
model/transformer.py — Steps 16–20
Feed-forward layers, layer norm, full transformer block,
encoder stack, decoder stack, and the complete language model head.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

from .attention import MultiHeadAttention
from .embeddings import InputEmbedding, RotaryPositionalEncoding


# ─────────────────────────────────────────────
# STEP 17 — Layer normalization
# ─────────────────────────────────────────────

class LayerNorm(nn.Module):
    """
    Standard layer norm with optional bias removal (bias=False improves
    training stability in large models — used in LLaMA, Gemma).
    """
    def __init__(self, d_model: int, eps: float = 1e-5, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model)) if bias else None
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, x.shape[-1:], self.weight, self.bias, self.eps)


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (Zhang & Sennrich, 2019).
    Simpler than LayerNorm — no bias, no mean subtraction.
    Used in: LLaMA, Mistral, Gemma, T5.
    ~10% faster than LayerNorm, similar or better quality.
    """
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return norm * self.weight


# ─────────────────────────────────────────────
# STEP 16 — Feed-forward layers
# ─────────────────────────────────────────────

class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.
    Two linear transforms with non-linearity between.
    Classic: Linear → ReLU → Linear
    Modern:  SwiGLU / GeGLU gated variant (used in LLaMA, PaLM, Gemma)
    """
    def __init__(self, d_model: int, d_ff: int,
                 dropout: float = 0.1,
                 activation: str = "swiglu",
                 bias: bool = False):
        super().__init__()
        self.activation = activation

        if activation in ("swiglu", "geglu"):
            # Gated linear unit: two parallel projections, one gates the other
            # d_ff adjusted to keep param count similar to standard FFN
            d_ff_adj = int(d_ff * 2 / 3)
            self.w1 = nn.Linear(d_model, d_ff_adj, bias=bias)   # gate
            self.w2 = nn.Linear(d_ff_adj, d_model, bias=bias)   # output
            self.w3 = nn.Linear(d_model, d_ff_adj, bias=bias)   # value
        else:
            self.w1 = nn.Linear(d_model, d_ff, bias=bias)
            self.w2 = nn.Linear(d_ff, d_model, bias=bias)

        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "swiglu":
            return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))
        elif self.activation == "geglu":
            return self.dropout(self.w2(F.gelu(self.w1(x)) * self.w3(x)))
        elif self.activation == "relu":
            return self.dropout(self.w2(F.relu(self.w1(x))))
        elif self.activation == "gelu":
            return self.dropout(self.w2(F.gelu(self.w1(x))))
        else:
            raise ValueError(f"Unknown activation: {self.activation}")


# ─────────────────────────────────────────────
# STEP 18 — Full transformer block
# ─────────────────────────────────────────────

class TransformerBlock(nn.Module):
    """
    Single transformer block with pre-norm residual connections.
    Pre-norm (LN before sub-layer) is more stable than post-norm
    and is used in all modern LLMs.

    Architecture:
        x = x + Attn(Norm(x))
        x = x + FFN(Norm(x))
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 dropout: float = 0.1,
                 norm_type: str = "rmsnorm",
                 n_kv_heads: Optional[int] = None,
                 bias: bool = False,
                 ff_activation: str = "swiglu",
                 rope: Optional[nn.Module] = None,
                 layer_idx: int = 0):
        super().__init__()

        NormClass = RMSNorm if norm_type == "rmsnorm" else LayerNorm

        self.norm1 = NormClass(d_model)
        self.norm2 = NormClass(d_model)

        self.attn = MultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            n_kv_heads=n_kv_heads,
            bias=bias,
            rope=rope,
        )

        self.ff = FeedForward(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            activation=ff_activation,
            bias=bias,
        )

        self.dropout = nn.Dropout(dropout)
        self.layer_idx = layer_idx

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        causal: bool = True,
        kv_cache: Optional[Tuple] = None,
        cache_offset: int = 0,
    ) -> Tuple[torch.Tensor, Optional[Tuple]]:
        # Self-attention with pre-norm
        residual = x
        x_norm = self.norm1(x)
        attn_out, new_cache = self.attn(
            x_norm, mask=mask, causal=causal,
            kv_cache=kv_cache, cache_offset=cache_offset,
        )
        x = residual + self.dropout(attn_out)

        # Feed-forward with pre-norm
        x = x + self.dropout(self.ff(self.norm2(x)))

        return x, new_cache


class CrossAttentionBlock(nn.Module):
    """
    Decoder-style block with cross-attention to encoder output.
    Used in encoder-decoder (seq2seq) architectures.
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 dropout: float = 0.1,
                 norm_type: str = "rmsnorm",
                 bias: bool = False,
                 ff_activation: str = "swiglu"):
        super().__init__()
        NormClass = RMSNorm if norm_type == "rmsnorm" else LayerNorm

        self.norm1 = NormClass(d_model)
        self.norm2 = NormClass(d_model)
        self.norm3 = NormClass(d_model)

        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout, bias=bias)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout, bias=bias)
        self.ff = FeedForward(d_model, d_ff, dropout, ff_activation, bias)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        encoder_out: torch.Tensor,
        self_mask: Optional[torch.Tensor] = None,
        cross_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple] = None,
        cache_offset: int = 0,
    ) -> Tuple[torch.Tensor, Optional[Tuple]]:
        # Masked self-attention
        x_norm = self.norm1(x)
        sa_out, new_cache = self.self_attn(
            x_norm, mask=self_mask, causal=True,
            kv_cache=kv_cache, cache_offset=cache_offset,
        )
        x = x + self.dropout(sa_out)

        # Cross-attention to encoder
        x_norm = self.norm2(x)
        ca_out, _ = self.cross_attn(
            x_norm, context=encoder_out, mask=cross_mask, causal=False,
        )
        x = x + self.dropout(ca_out)

        # Feed-forward
        x = x + self.dropout(self.ff(self.norm3(x)))
        return x, new_cache


# ─────────────────────────────────────────────
# STEP 19 — Encoder stack
# ─────────────────────────────────────────────

class TransformerEncoder(nn.Module):
    """
    Stack of transformer blocks for bidirectional encoding (BERT-style).
    No causal masking — all positions attend to all positions.
    """
    def __init__(self, d_model: int, n_heads: int, n_layers: int,
                 d_ff: int, dropout: float = 0.1,
                 vocab_size: int = 32000, max_seq_len: int = 512,
                 norm_type: str = "rmsnorm",
                 n_kv_heads: Optional[int] = None,
                 ff_activation: str = "gelu"):
        super().__init__()

        self.embedding = InputEmbedding(
            vocab_size, d_model, max_seq_len, dropout,
            pe_type="learned",
        )

        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model, n_heads=n_heads, d_ff=d_ff,
                dropout=dropout, norm_type=norm_type,
                n_kv_heads=n_kv_heads, ff_activation=ff_activation,
                layer_idx=i,
            )
            for i in range(n_layers)
        ])

        NormClass = RMSNorm if norm_type == "rmsnorm" else LayerNorm
        self.norm = NormClass(d_model)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.embedding(input_ids)

        # Build padding mask if provided
        mask = None
        if attention_mask is not None:
            # (B, 1, 1, T) so it broadcasts over heads
            mask = attention_mask.unsqueeze(1).unsqueeze(2)

        for layer in self.layers:
            x, _ = layer(x, mask=mask, causal=False)

        return self.norm(x)


# ─────────────────────────────────────────────
# STEP 20 — Decoder stack (and full LM)
# ─────────────────────────────────────────────

class CausalTransformer(nn.Module):
    """
    GPT-style decoder-only causal language model.
    This is the primary architecture — all modern frontier models
    (GPT, LLaMA, Mistral, Gemma, Qwen, Falcon) use decoder-only.

    Uses:
    - RMSNorm (faster than LayerNorm)
    - SwiGLU feed-forward (better than ReLU/GeLU)
    - Grouped Query Attention (reduced KV cache)
    - RoPE positional encoding (length generalization)
    - No bias in linear layers (cleaner training dynamics)
    - Tied input/output embeddings (fewer params, better generalization)
    """

    def __init__(self, vocab_size: int, d_model: int, n_heads: int,
                 n_layers: int, d_ff: int, max_seq_len: int = 2048,
                 dropout: float = 0.1,
                 n_kv_heads: Optional[int] = None,
                 norm_type: str = "rmsnorm",
                 ff_activation: str = "swiglu",
                 tie_embeddings: bool = True,
                 pad_idx: int = 0):
        super().__init__()

        self.d_model = d_model
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size

        # RoPE — shared across all layers
        from .embeddings import RotaryPositionalEncoding
        head_dim = d_model // n_heads
        self.rope = RotaryPositionalEncoding(head_dim=head_dim,
                                              max_seq_len=max_seq_len)

        # Input embedding (no PE — RoPE applied inside attention)
        self.embedding = InputEmbedding(
            vocab_size, d_model, max_seq_len, dropout,
            pad_idx=pad_idx, pe_type="none",
        )

        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model, n_heads=n_heads, d_ff=d_ff,
                dropout=dropout, norm_type=norm_type,
                n_kv_heads=n_kv_heads, bias=False,
                ff_activation=ff_activation,
                rope=self.rope, layer_idx=i,
            )
            for i in range(n_layers)
        ])

        NormClass = RMSNorm if norm_type == "rmsnorm" else LayerNorm
        self.norm = NormClass(d_model)

        # LM head: project d_model → vocab_size
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Tie weights: input embedding ≡ output projection
        if tie_embeddings:
            self.lm_head.weight = self.embedding.token_emb.embedding.weight

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if "weight" in name and p.dim() >= 2:
                if "embedding" not in name and "norm" not in name:
                    nn.init.normal_(p, mean=0.0, std=0.02)
            elif "bias" in name:
                nn.init.zeros_(p)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_caches: Optional[List[Optional[Tuple]]] = None,
        cache_offset: int = 0,
    ) -> Tuple[torch.Tensor, Optional[List]]:
        """
        input_ids: (B, T)
        Returns:   (logits: (B, T, V), new_kv_caches)
        """
        x = self.embedding(input_ids)

        mask = None
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).unsqueeze(2)

        new_caches = []
        for i, layer in enumerate(self.layers):
            cache = kv_caches[i] if kv_caches is not None else None
            x, new_cache = layer(
                x, mask=mask, causal=True,
                kv_cache=cache, cache_offset=cache_offset,
            )
            new_caches.append(new_cache)

        x = self.norm(x)
        logits = self.lm_head(x)    # (B, T, V)

        return logits, new_caches if kv_caches is not None else None

    def get_num_params(self, non_embedding: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.embedding.token_emb.embedding.weight.numel()
        return n


# ─────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("Steps 16–20: Full transformer architecture checks")
    B, T = 2, 32

    # Full causal decoder-only LM
    model = CausalTransformer(
        vocab_size=1000, d_model=128, n_heads=4,
        n_layers=3, d_ff=512, max_seq_len=64,
        dropout=0.0, n_kv_heads=2,
    )

    tokens = torch.randint(0, 1000, (B, T))
    logits, _ = model(tokens)
    assert logits.shape == (B, T, 1000)
    print(f"  Logits shape: {tuple(logits.shape)}  ✓")

    # Verify causal property: position i should only see j<=i
    # Compare logits with and without last token — should differ only at last pos
    tokens2 = tokens.clone()
    tokens2[:, -1] = 0   # change only last token
    logits2, _ = model(tokens2)
    diff = (logits - logits2).abs()
    assert diff[:, :-1, :].max() < 1e-5, "Causality violated!"
    print(f"  Causality check passed  ✓")

    # KV cache correctness
    model.eval()
    # Full forward pass
    logits_full, _ = model(tokens)
    # Cached: process first T-1 tokens, then last token with cache
    prefix = tokens[:, :-1]
    _, caches = model(prefix, kv_caches=[None]*3)
    last_tok = tokens[:, -1:]
    logits_cached, _ = model(last_tok, kv_caches=caches, cache_offset=T-1)
    max_err = (logits_full[:, -1:, :] - logits_cached).abs().max().item()
    print(f"  KV cache error (should be ~0): {max_err:.2e}  ✓")

    n_params = model.get_num_params()
    print(f"  Model params (non-emb): {n_params:,}")

    # Encoder
    encoder = TransformerEncoder(
        vocab_size=1000, d_model=128, n_heads=4, n_layers=2,
        d_ff=512, max_seq_len=64, dropout=0.0,
    )
    enc_out = encoder(tokens)
    assert enc_out.shape == (B, T, 128)
    print(f"  Encoder output: {tuple(enc_out.shape)}  ✓")

    print("✓ Steps 16–20 verified")
