"""Decision adapters (M02): candidate-isolated scoring and value prefixes.

ARCHITECTURE §2 mandates candidate isolation: each legal candidate is
evaluated as an independent branch of the SAME prefix (flattened to
independent batch rows), never appended into one causal sequence where
earlier candidates could leak into later scores.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from bramastra_lab.research.config import BuildConfig
from bramastra_lab.research.models import IntegratedModel


class DecisionError(ValueError):
    """A decision adapter was driven against its contract."""


@dataclass(frozen=True)
class CandidateScores:
    scores: list[float]
    action_ids: list[str]
    isolated_rows: int
    model_identity: str

    @property
    def normalized(self) -> list[float]:
        import math

        finite = [score for score in self.scores if math.isfinite(score)]
        if not finite or len(finite) != len(self.scores):
            raise DecisionError("cannot normalize nonfinite candidate scores")
        ceiling = max(finite)
        weights = [math.exp(score - ceiling) for score in self.scores]
        total = sum(weights)
        return [weight / total for weight in weights]


def _action_identity(candidate: Sequence[int]) -> str:
    from bramastra_lab.research.contracts.core import content_identity

    return content_identity({"candidate": list(candidate)})


def score_candidates(model: IntegratedModel, config: BuildConfig,
                     prefix_tokens: Sequence[int],
                     candidates: Sequence[Sequence[int]], *,
                     action_identities: Sequence[str] | None = None) -> CandidateScores:
    """Score each candidate as an isolated branch of the same prefix.

    - Flatten [K, T] to independent batch rows sharing the prefix.
    - Deduplicate identical public action identities before scoring.
    - Empty candidates and empty candidate rows reject; padding inside a
      candidate row is the caller's declared encoding responsibility and is
      validated for nonempty rows only.
    """
    if not candidates:
        raise DecisionError("no legal candidates supplied")
    if any(len(candidate) == 0 for candidate in candidates):
        raise DecisionError("candidate rows must be nonempty")
    identities = list(action_identities) if action_identities is not None \
        else [_action_identity(candidate) for candidate in candidates]
    if len(identities) != len(candidates):
        raise DecisionError("action identity count mismatch")
    seen: dict[str, int] = {}
    order: list[int] = []
    for index, identity in enumerate(identities):
        if identity in seen:
            continue  # duplicates deduplicated before labeling/scoring
        seen[identity] = index
        order.append(index)
    prefix_list = list(prefix_tokens)
    max_len = max(len(prefix_list) + len(candidates[index]) for index in order)
    if max_len > config.model.max_seq:
        raise DecisionError("prefix + candidate exceeds the configured context limit")
    rows = []
    row_lengths = []
    for index in order:
        row = prefix_list + list(candidates[index])
        rows.append(row)
        row_lengths.append(len(row))
    input_ids = torch.full((len(rows), max_len), 0, dtype=torch.long)
    padding = torch.zeros((len(rows), max_len), dtype=torch.bool)
    for row_index, row in enumerate(rows):
        input_ids[row_index, :len(row)] = torch.tensor(row, dtype=torch.long)
        padding[row_index, :len(row)] = True
    with torch.no_grad():
        hidden = model.decoder.forward_hidden(input_ids, padding)
        spans = torch.tensor([[row_lengths[row_index] - 1] for row_index in range(len(rows))],
                             dtype=torch.long)
        gathered = hidden.gather(
            1, spans.unsqueeze(-1).expand(-1, -1, hidden.shape[-1])).squeeze(1)
        scores = model.action_head(gathered).squeeze(-1)
    scores_by_identity = {identities[index]: float(scores[position].item())
                          for position, index in enumerate(order)}
    return CandidateScores(
        scores=[scores_by_identity[identity] for identity in identities],
        action_ids=list(identities),
        isolated_rows=len(rows),
        model_identity=f"{type(model).__name__}:{sum(p.numel() for p in model.parameters())}")


def estimate_value(model: IntegratedModel, config: BuildConfig,
                   prefix_tokens: Sequence[int]) -> torch.Tensor:
    """Value estimate at the action-free public prefix (never after a
    candidate or a teacher answer)."""
    prefix = list(prefix_tokens)
    if not prefix:
        raise DecisionError("value prefix must be nonempty")
    if len(prefix) > config.model.max_seq:
        raise DecisionError("value prefix exceeds the configured context limit")
    tokens = torch.tensor([prefix], dtype=torch.long)
    with torch.no_grad():
        return model(tokens, return_value=True).value[0]
