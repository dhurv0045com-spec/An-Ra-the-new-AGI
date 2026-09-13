"""Differentiable training scoring APIs (I02/K8).

The inference helpers (`score_candidates`, `estimate_value`,
`predict_step_finite`) return detached values at an inference boundary.
These training twins return live differentiable tensors so the router can
build real loss terms; using an inference float as a learned objective
silently severs the gradient and is exactly the defect this module closes.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch

from bramastra_lab.research.config import BuildConfig
from bramastra_lab.research.models import IntegratedModel
from bramastra_lab.research.models.world import json_key


class ScoringError(ValueError):
    """A training-scoring call violated its contract."""


def score_candidates_trainable(model: IntegratedModel, config: BuildConfig,
                               prefix_tokens: Sequence[int],
                               candidates: Sequence[Sequence[int]],
                               *, teacher_distribution: torch.Tensor | None = None,
                               legal_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    """Batched candidate scoring with a live gradient path.

    Each candidate is an isolated branch of the same prefix (flattened to
    independent batch rows). Returns raw scores [K] (or [B,K] when a batch
    of prefixes is supplied) plus log-softmax over the legal mask.
    """
    if not candidates:
        raise ScoringError("no candidates supplied")
    if any(len(candidate) == 0 for candidate in candidates):
        raise ScoringError("candidate rows must be nonempty")
    prefix_list = list(prefix_tokens)
    max_len = max(len(prefix_list) + len(candidate) for candidate in candidates)
    if max_len > config.model.max_seq:
        raise ScoringError("prefix + candidate exceeds the configured context limit")
    rows = [prefix_list + list(candidate) for candidate in candidates]
    input_ids = torch.full((len(rows), max_len), 0, dtype=torch.long)
    padding = torch.zeros((len(rows), max_len), dtype=torch.bool)
    for row_index, row in enumerate(rows):
        input_ids[row_index, :len(row)] = torch.tensor(row, dtype=torch.long)
        padding[row_index, :len(row)] = True
    hidden = model.decoder.forward_hidden(input_ids, padding)
    spans = torch.tensor([[len(row) - 1] for row in rows], dtype=torch.long)
    gathered = hidden.gather(
        1, spans.unsqueeze(-1).expand(-1, -1, hidden.shape[-1])).squeeze(1)
    scores = model.action_head(gathered).squeeze(-1)  # [K] live gradient
    if legal_mask is None:
        legal_mask = torch.ones_like(scores, dtype=torch.bool)
    log_probs = torch.log_softmax(scores.double().masked_fill(~legal_mask, -1e30), dim=-1)
    return {"scores": scores, "log_probs": log_probs.to(scores.dtype),
            "legal_mask": legal_mask}


def value_estimate_trainable(model: IntegratedModel, config: BuildConfig,
                             prefix_tokens: Sequence[int]) -> torch.Tensor:
    """Value head at the action-free prefix with a live gradient path."""
    prefix = list(prefix_tokens)
    if not prefix:
        raise ScoringError("value prefix must be nonempty")
    tokens = torch.tensor([prefix], dtype=torch.long)
    return model(tokens, return_value=True).value[0]


def world_transition_token_loss(model: IntegratedModel, config: BuildConfig,
                                prefix_tokens: Sequence[int],
                                action: Mapping[str, Any],
                                target_feedback: Mapping[str, Any], *,
                                goal: Mapping[str, Any] | None = None) -> torch.Tensor:
    """Next-public-outcome token NLL over the DECLARED target span (A4).

    The span is exactly the feedback event plus required EOS; the conditioned
    prefix (history + goal + action) is masked out of the loss. Uses the same
    renderer conventions as preparation (single public renderer upstream).
    """
    from bramastra_lab.research.experience.codec import SPECIAL_BOUNDARY, SPECIAL_EOS, \
        encode_event

    condition = [SPECIAL_BOUNDARY]
    if goal is not None:
        condition += encode_event("goal", goal)
    condition += encode_event("action", action)
    condition += list(prefix_tokens)
    target_span = encode_event("feedback", target_feedback) + [SPECIAL_EOS]
    sequence = condition + target_span
    if len(sequence) - 1 > config.model.max_seq:
        raise ScoringError("sequence exceeds the context limit")
    inputs = torch.tensor([sequence[:-1]], dtype=torch.long)
    targets = torch.tensor(sequence[1:], dtype=torch.long)
    logits = model(inputs).logits[0]
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    # Mask the conditioned prefix: only target-span positions contribute.
    token_lp = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    span_lp = token_lp[len(condition) - 1:]
    return -span_lp.sum()


def deduplicate_support(support: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Deduplicate identical parsed public outcomes before normalization."""
    seen: dict[str, Mapping[str, Any]] = {}
    for option in support:
        key = json_key({"feedback": option.get("feedback"),
                        "terminated": option.get("terminated", False)})
        if key not in seen:
            seen[key] = option
    return list(seen.values())
