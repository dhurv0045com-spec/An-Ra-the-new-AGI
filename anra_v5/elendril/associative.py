"""Causal associative copy memory over the current packed context."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .codec import TOKEN_IDS


WINDOW = 3
ADDRESS_WIDTH = 16
RECURRENT_MODES = frozenset({"recurrent", "recurrent-dense", "three-read"})


class AssociativeDecoder(nn.Module):
    """Add a learned read gate over token-addressable positions to the dense decoder."""

    def __init__(self, core: nn.Module, seed: int, *, mode: str = "hybrid"):
        super().__init__()
        if mode not in ("hybrid", "lexical", "context", "two-read", *RECURRENT_MODES):
            raise ValueError("unknown associative addressing mode")
        self.core = core
        self.mode = mode
        width = core.config.width
        # Leave the base model's initialization and caller RNG identical across arms.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            if mode != "lexical":
                self.address_query = nn.Linear(width, ADDRESS_WIDTH, bias=False)
                self.address_key = nn.Linear(width, ADDRESS_WIDTH, bias=False)
            self.read_gate = nn.Linear(width, 1)
        nn.init.zeros_(self.read_gate.weight)
        nn.init.constant_(self.read_gate.bias, -2)
        if mode != "context":
            self.lexical_weights = nn.Parameter(torch.zeros(WINDOW, WINDOW))
        if mode == "two-read" or mode in RECURRENT_MODES:
            self.join_offsets = nn.Parameter(torch.zeros(WINDOW))
            self.join_strength = nn.Parameter(torch.tensor(2.0))

    def _lexical_scores(self, tokens: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        padded = F.pad(tokens, (WINDOW - 1, 0))
        windows = padded.unfold(1, WINDOW, 1).flip(-1)
        valid = (positions.unsqueeze(-1) >= torch.arange(WINDOW, device=tokens.device)) & (windows >= 3)
        lexical = windows[:, :, None, :, None] == windows[:, None, :, None, :]
        valid_pairs = valid[:, :, None, :, None] & valid[:, None, :, None, :]
        return ((lexical & valid_pairs) * self.lexical_weights).sum(dim=(-2, -1))

    @staticmethod
    def _copy(affinity: torch.Tensor, allowed: torch.Tensor,
              tokens: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
        affinity = affinity.masked_fill(~allowed, torch.finfo(affinity.dtype).min)
        attention = affinity.softmax(dim=-1).masked_fill(~allowed, 0)
        source_ids = tokens[:, None, :].expand_as(attention)
        return torch.zeros_like(base).scatter_add(-1, source_ids, attention)

    def _join_scores(self, first: torch.Tensor, tokens: torch.Tensor,
                     positions: torch.Tensor) -> torch.Tensor:
        windows = F.pad(tokens, (WINDOW - 1, 0)).unfold(1, WINDOW, 1).flip(-1)
        weights = self.join_offsets.softmax(dim=0)
        scores = torch.zeros((*first.shape[:2], tokens.size(1)), dtype=first.dtype, device=first.device)
        for offset in range(WINDOW):
            symbols = windows[:, :, offset]
            valid = (positions >= offset) & (symbols >= 3)
            indices = symbols[:, None, :].expand(-1, first.size(1), -1)
            scores = scores + weights[offset] * first.gather(-1, indices) * valid[:, None, :]
        return F.softplus(self.join_strength) * torch.log1p(10 * scores)

    @staticmethod
    def _attention(affinity: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
        return affinity.masked_fill(~allowed, torch.finfo(affinity.dtype).min).softmax(
            dim=-1).masked_fill(~allowed, 0)

    @staticmethod
    def _token_groups(tokens: torch.Tensor) -> torch.Tensor:
        """Give equal token IDs equal contiguous group IDs, with at most L groups."""
        symbols, order = tokens.sort(dim=-1, stable=True)
        starts = torch.cat((torch.ones_like(symbols[:, :1], dtype=torch.bool),
                            symbols[:, 1:] != symbols[:, :-1]), dim=-1)
        groups = starts.long().cumsum(dim=-1) - 1
        return torch.empty_like(groups).scatter(1, order, groups)

    def _group_join_scores(self, attention: torch.Tensor, groups: torch.Tensor,
                           tokens: torch.Tensor, positions: torch.Tensor, vocab_size: int) -> torch.Tensor:
        """Match candidate source windows without materializing vocabulary-wide mass."""
        width = min(vocab_size, tokens.size(1))
        mass = attention.new_zeros((*attention.shape[:2], width)).scatter_add(
            -1, groups[:, None, :].expand_as(attention), attention)
        symbols = F.pad(tokens, (WINDOW - 1, 0)).unfold(1, WINDOW, 1).flip(-1)
        windows = F.pad(groups, (WINDOW - 1, 0)).unfold(1, WINDOW, 1).flip(-1)
        scores = torch.zeros_like(attention)
        weights = self.join_offsets.softmax(dim=0)
        for offset in range(WINDOW):
            indices = windows[:, :, offset, None].transpose(1, 2).expand_as(attention)
            valid = (positions >= offset) & (symbols[:, :, offset] >= 3)
            scores = scores + weights[offset] * mass.gather(-1, indices) * valid[:, None, :]
        return F.softplus(self.join_strength) * torch.log1p(10 * scores)

    @staticmethod
    def _requested_depth(tokens: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """The digit after each segment BOS is the public read-count instruction."""
        now = torch.arange(tokens.size(1), device=tokens.device)
        # Never inspect a future token, including at BOS and padding.
        control = (now - positions + 1).clamp_min(0).minimum(now)
        requested = tokens.gather(1, control) - TOKEN_IDS["1"] + 1
        valid = (positions >= 1) & (requested >= 1) & (requested <= 4)
        return torch.where(valid, requested, 1)

    def _recurrent_copy(self, first, contextual, allowed, tokens, positions, base):
        if self.mode == "three-read":
            depth, cap = torch.full_like(positions, 3), 3
        else:
            depth, cap = self._requested_depth(tokens, positions), 4
        groups = self._token_groups(tokens) if self.mode != "recurrent-dense" else None
        attention = self._attention(first, allowed)
        selected = attention
        token_ids = tokens[:, None, :].expand_as(attention)
        for iteration in range(2, cap + 1):
            if groups is None:
                mass = torch.zeros_like(base).scatter_add(-1, token_ids, attention)
                join = self._join_scores(mass, tokens, positions)
            else:
                join = self._group_join_scores(attention, groups, tokens, positions, base.size(-1))
            attention = self._attention(contextual + join, allowed)
            selected = torch.where((depth == iteration)[:, :, None], attention, selected)
        return torch.zeros_like(base).scatter_add(-1, token_ids, selected)

    def forward(self, tokens: torch.Tensor, positions: torch.Tensor, mask: torch.Tensor,
                *, remove_second_read: bool = False) -> torch.Tensor:
        hidden = self.core.forward_hidden(tokens, positions, mask)
        base = F.linear(hidden, self.core.embedding.weight).log_softmax(dim=-1)
        context = torch.arange(tokens.size(1), device=tokens.device)
        allowed = mask[:, 0] & (context[:, None] > context[None, :]) & (tokens[:, None, :] >= 3)
        if self.mode == "lexical":
            affinity = self._lexical_scores(tokens, positions)
        else:
            contextual = self.address_query(hidden) @ self.address_key(hidden).transpose(-1, -2)
            contextual = contextual / math.sqrt(ADDRESS_WIDTH)
            affinity = contextual
            if self.mode in ("hybrid", "two-read", *RECURRENT_MODES):
                affinity = affinity + self._lexical_scores(tokens, positions)
        if self.mode in RECURRENT_MODES and not remove_second_read:
            copied = self._recurrent_copy(affinity, contextual, allowed, tokens, positions, base)
        else:
            copied = self._copy(affinity, allowed, tokens, base)
            if self.mode == "two-read" and not remove_second_read:
                copied = self._copy(contextual + self._join_scores(copied, tokens, positions),
                                    allowed, tokens, base)
        gate = self.read_gate(hidden)
        # Finite log scores protect backward at vocabulary positions with zero copy mass.
        mixture = torch.logaddexp(
            base + F.logsigmoid(-gate), copied.clamp_min(torch.finfo(copied.dtype).tiny).log()
            + F.logsigmoid(gate))
        return torch.where(allowed.any(dim=-1, keepdim=True), mixture, base)



class DenseCapacityControl(nn.Module):
    """Add a tokenwise residual MLP with six more parameters than the hybrid read head."""

    def __init__(self, core: nn.Module, seed: int):
        super().__init__()
        self.core = core
        width = core.config.width
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.up = nn.Linear(width, ADDRESS_WIDTH)
            self.down = nn.Linear(ADDRESS_WIDTH, width)
            nn.init.normal_(self.down.weight, std=0.02)
            nn.init.zeros_(self.down.bias)

    def forward(self, tokens: torch.Tensor, positions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        hidden = self.core.forward_hidden(tokens, positions, mask)
        return F.linear(hidden + self.down(F.silu(self.up(hidden))), self.core.embedding.weight)
