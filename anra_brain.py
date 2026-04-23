"""An-Ra architecture and tokenizer definitions.

This module provides stable import paths for:
- CausalTransformer
- CharTokenizer
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import functional as F

from tokenizer.char_tokenizer import CharTokenizer


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int):
        super().__init__()
        self.n_head = n_head
        self.head_size = n_embd // n_head
        self.key = nn.Linear(n_embd, self.head_size * n_head, bias=False)
        self.query = nn.Linear(n_embd, self.head_size * n_head, bias=False)
        self.value = nn.Linear(n_embd, self.head_size * n_head, bias=False)
        self.proj = nn.Linear(n_embd, n_embd)
        self.register_buffer(
            "tril", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, t, channels = x.shape
        k = self.key(x).view(bsz, t, self.n_head, self.head_size).transpose(1, 2)
        q = self.query(x).view(bsz, t, self.n_head, self.head_size).transpose(1, 2)
        v = self.value(x).view(bsz, t, self.n_head, self.head_size).transpose(1, 2)

        wei = q @ k.transpose(-2, -1) * (self.head_size ** -0.5)
        wei = wei.masked_fill(self.tril[:, :, :t, :t] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        out = wei @ v
        out = out.transpose(1, 2).contiguous().view(bsz, t, channels)
        return self.proj(out)


class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class CausalTransformer(nn.Module):
    def __init__(self, vocab_size: int, n_embd: int, n_head: int, n_layer: int, block_size: int):
        super().__init__()
        self.block_size = block_size
        self.d_model = n_embd
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.apply(self._init_weights)
        self.lm_head.weight = self.token_embedding_table.weight

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            std = 0.02 / (2 * len(self.blocks)) ** 0.5 if hasattr(self, "blocks") else 0.02
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            std = 0.02 / (2 * len(self.blocks)) ** 0.5 if hasattr(self, "blocks") else 0.02
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        _, t = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos = torch.arange(t, device=idx.device)
        pos_emb = self.position_embedding_table(pos)
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            b, tt, c = logits.shape
            loss = F.cross_entropy(logits.view(b * tt, c), targets.view(b * tt))
        return logits, loss
