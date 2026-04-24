"""An-Ra mainline model and tokenizer definitions.

Canonical exports:
- CausalTransformer          -> V2 mainline decoder
- CausalTransformerV2        -> explicit V2 class name
- CharTokenizer             -> legacy char tokenizer (kept for compatibility)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import functional as F

from tokenizer.char_tokenizer import CharTokenizer


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * scale * self.weight


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: int = 10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cached_seq_len = 0
        self._cached_cos: torch.Tensor | None = None
        self._cached_sin: torch.Tensor | None = None

    def _build_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> None:
        if self._cached_cos is not None and self._cached_seq_len >= seq_len:
            return
        positions = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(positions, self.inv_freq.to(device))
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos()[None, None, :, :]
        sin = emb.sin()[None, None, :, :]
        self._cached_cos = cos.to(dtype=dtype)
        self._cached_sin = sin.to(dtype=dtype)
        self._cached_seq_len = seq_len

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = q.size(-2)
        self._build_cache(seq_len, q.device, q.dtype)
        assert self._cached_cos is not None and self._cached_sin is not None
        cos = self._cached_cos[..., :seq_len, :]
        sin = self._cached_sin[..., :seq_len, :]
        q = (q * cos) + (_rotate_half(q) * sin)
        k = (k * cos) + (_rotate_half(k) * sin)
        return q, k


class MultiHeadAttentionV2(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float = 0.0):
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError(f"n_embd={n_embd} must be divisible by n_head={n_head}")
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.q_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.k_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.v_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.out_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.rope = RotaryEmbedding(self.head_dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        q = self.q_proj(x).view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        q, k = self.rope(q, k)

        if hasattr(F, "scaled_dot_product_attention"):
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
            mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1)
            att = att.masked_fill(mask, float("-inf"))
            att = F.softmax(att, dim=-1)
            out = att @ v

        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.out_proj(out)


class SwiGLU(nn.Module):
    def __init__(self, n_embd: int, hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(n_embd, hidden_dim, bias=False)
        self.up_proj = nn.Linear(n_embd, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class BlockV2(nn.Module):
    def __init__(self, n_embd: int, n_head: int, *, eps: float = 1e-5, dropout: float = 0.0):
        super().__init__()
        hidden_dim = 4 * n_embd
        self.norm_1 = RMSNorm(n_embd, eps=eps)
        self.attn = MultiHeadAttentionV2(n_embd, n_head, dropout=dropout)
        self.norm_2 = RMSNorm(n_embd, eps=eps)
        self.mlp = SwiGLU(n_embd, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm_1(x))
        x = x + self.mlp(self.norm_2(x))
        return x


class CausalTransformerV2(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_embd: int,
        n_head: int,
        n_layer: int,
        block_size: int,
        *,
        rms_norm_eps: float = 1e-5,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.d_model = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.blocks = nn.ModuleList(
            [BlockV2(n_embd, n_head, eps=rms_norm_eps, dropout=dropout) for _ in range(n_layer)]
        )
        self.norm_f = RMSNorm(n_embd, eps=rms_norm_eps)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding_table.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def model_config(self) -> dict[str, int]:
        return {
            "vocab_size": self.vocab_size,
            "n_embd": self.n_embd,
            "n_head": self.n_head,
            "n_layer": self.n_layer,
            "block_size": self.block_size,
        }

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        _, seq_len = idx.shape
        if seq_len > self.block_size:
            raise ValueError(f"sequence length {seq_len} exceeds block size {self.block_size}")
        x = self.token_embedding_table(idx)
        for block in self.blocks:
            x = block(x)
        x = self.norm_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            bsz, time_steps, channels = logits.shape
            loss = F.cross_entropy(logits.view(bsz * time_steps, channels), targets.view(bsz * time_steps))
        return logits, loss


CausalTransformer = CausalTransformerV2

