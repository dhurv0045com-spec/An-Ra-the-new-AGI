"""Unified objective router (M05): A1 objective sums/counts over one
optimizer boundary, the single-window on-policy loss contract and
feature-off gradient routing.

This module composes experience.supervision (window/denominators) with the
existing trainer: it never creates a second training loop.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from bramastra_lab.research.experience.supervision import (
    SupervisionWindow,
    WindowLoss,
    combine_window_losses,
)


class RouterError(ValueError):
    """An objective routing request violated its contract."""


def huber_value_loss(predicted: torch.Tensor, target: torch.Tensor,
                     *, delta: float = 1.0) -> torch.Tensor:
    """Huber loss for value regression (A3, fixture delta=1)."""
    if delta <= 0:
        raise RouterError("Huber delta must be positive")
    return torch.nn.functional.smooth_l1_loss(
        predicted.float(), target.float(), beta=delta, reduction="sum")


def teacher_action_cross_entropy(logits: torch.Tensor, target_distribution: torch.Tensor,
                                 legal_mask: torch.Tensor) -> torch.Tensor:
    """Cross-entropy against a teacher DISTRIBUTION over legal actions (A3).

    Illegal slots are excluded; the teacher distribution must sum to one over
    the legal set.
    """
    if logits.shape != target_distribution.shape:
        raise RouterError("logits and teacher distribution must align")
    masked = torch.where(legal_mask, target_distribution.double(),
                         torch.zeros((), dtype=torch.float64).expand_as(target_distribution).double())
    total = masked.sum(dim=-1)
    if not torch.allclose(total, torch.ones_like(total), atol=1e-6):
        raise RouterError("teacher distribution must sum to one over legal actions")
    log_probs = torch.log_softmax(logits.double().masked_fill(~legal_mask, -1e30), dim=-1)
    return -(masked * log_probs).sum(dim=-1)


def on_policy_loss(chosen_log_probability: torch.Tensor, advantage: torch.Tensor,
                   *, legal_entropy: torch.Tensor | None = None,
                   entropy_weight: float = 0.0) -> torch.Tensor:
    """Single-window on-policy loss (A3): -stopgrad(advantage) * log pi,
    with optional legal-action entropy regularization. Targets detach."""
    if advantage.requires_grad:
        raise RouterError("advantage must be detached from the prediction graph")
    pg = -(advantage.detach() * chosen_log_probability).sum()
    total = pg
    if legal_entropy is not None and entropy_weight:
        total = total - entropy_weight * legal_entropy.sum()
    return total


def route_window(window: SupervisionWindow,
                 sums: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, Any]]:
    """Route one window's per-term unnormalized sums through the A1 weights.

    Wraps combine_window_losses with WindowLoss records; denominators come
    from the window's eligibility counts computed before backward.
    """
    losses = {}
    for term, total in sums.items():
        losses[term] = WindowLoss(term=term, total=total,
                                  denominator=window.denominator(term))
    return combine_window_losses(window, losses)


@dataclass
class FeatureSwitches:
    """A switch controls execution and objective routing, not labels."""

    token: bool = True
    world: bool = False
    action: bool = False
    value: bool = False
    pair: bool = False
    pg: bool = False

    def enabled_terms(self) -> frozenset[str]:
        terms = set()
        for name in ("token", "world", "action", "value", "pair", "pg"):
            if getattr(self, name):
                terms.add(name)
        return frozenset(terms)

    def build_window(self, weights: Mapping[str, float]) -> SupervisionWindow:
        return SupervisionWindow(weights=dict(weights),
                                 enabled_terms=self.enabled_terms())
