"""Causal associative copy memory over the current packed context."""

import math

import torch
from torch import nn
from torch.nn import functional as F


WINDOW = 3
ADDRESS_WIDTH = 16


class AssociativeDecoder(nn.Module):
    """Add a learned read gate over token-addressable positions to the dense decoder."""

    def __init__(self, core: nn.Module, seed: int):
        super().__init__()
        self.core = core
        width = core.config.width
        # Leave the base model's initialization and caller RNG identical across arms.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.address_query = nn.Linear(width, ADDRESS_WIDTH, bias=False)
            self.address_key = nn.Linear(width, ADDRESS_WIDTH, bias=False)
            self.read_gate = nn.Linear(width, 1)
        nn.init.zeros_(self.read_gate.weight)
        nn.init.constant_(self.read_gate.bias, -2)
        self.lexical_weights = nn.Parameter(torch.zeros(WINDOW, WINDOW))

    def _lexical_scores(self, tokens: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        padded = F.pad(tokens, (WINDOW - 1, 0))
        windows = padded.unfold(1, WINDOW, 1).flip(-1)
        valid = (positions.unsqueeze(-1) >= torch.arange(WINDOW, device=tokens.device)) & (windows >= 3)
        lexical = windows[:, :, None, :, None] == windows[:, None, :, None, :]
        valid_pairs = valid[:, :, None, :, None] & valid[:, None, :, None, :]
        return ((lexical & valid_pairs) * self.lexical_weights).sum(dim=(-2, -1))

    def forward(self, tokens: torch.Tensor, positions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        hidden = self.core.forward_hidden(tokens, positions, mask)
        base = F.linear(hidden, self.core.embedding.weight).log_softmax(dim=-1)
        context = torch.arange(tokens.size(1), device=tokens.device)
        allowed = mask[:, 0] & (context[:, None] > context[None, :]) & (tokens[:, None, :] >= 3)
        affinity = self.address_query(hidden) @ self.address_key(hidden).transpose(-1, -2)
        affinity = affinity / math.sqrt(ADDRESS_WIDTH) + self._lexical_scores(tokens, positions)
        affinity = affinity.masked_fill(~allowed, torch.finfo(affinity.dtype).min)
        attention = affinity.softmax(dim=-1).masked_fill(~allowed, 0)
        source_ids = tokens[:, None, :].expand_as(attention)
        copied = torch.zeros_like(base).scatter_add(-1, source_ids, attention)
        gate = self.read_gate(hidden)
        # Finite log scores protect backward at vocabulary positions with zero copy mass.
        mixture = torch.logaddexp(
            base + F.logsigmoid(-gate), copied.clamp_min(torch.finfo(copied.dtype).tiny).log()
            + F.logsigmoid(gate))
        return torch.where(allowed.any(dim=-1, keepdim=True), mixture, base)
