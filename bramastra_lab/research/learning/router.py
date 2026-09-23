"""Unified objective router (M05): A1 objective sums/counts over one
optimizer boundary, the single-window on-policy loss contract and
feature-off gradient routing.

This module composes experience.supervision (window/denominators) with the
existing trainer: it never creates a second training loop.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import torch

from bramastra_lab.research.experience.supervision import (
    DEFAULT_WEIGHTS,
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
                 sums: Mapping[str, torch.Tensor], *,
                 global_denominators: Mapping[str, Any] | None = None,
                 replica_gradient_scale: float = 1.0
                 ) -> tuple[torch.Tensor, dict[str, Any]]:
    """Route one window's per-term unnormalized sums through the A1 weights.

    Wraps combine_window_losses with WindowLoss records; denominators come
    from the window's eligibility counts computed before backward. For
    replicated training, ``global_denominators`` contains all-reduced counts
    and ``replica_gradient_scale`` compensates for the replica-mean gradient
    reduction, making the result equal to one global sum / global count.
    """
    if (not isinstance(replica_gradient_scale, (int, float))
            or isinstance(replica_gradient_scale, bool)
            or not math.isfinite(float(replica_gradient_scale))
            or replica_gradient_scale <= 0):
        raise RouterError("replica_gradient_scale must be finite and positive")
    if global_denominators is not None:
        missing = set(window.enabled_terms) - set(global_denominators)
        if missing:
            raise RouterError(
                f"global denominators missing enabled terms: {sorted(missing)}")
    losses = {}
    for term, total in sums.items():
        losses[term] = WindowLoss(term=term, total=total,
                                  denominator=window.denominator(term))
    if global_denominators is None and replica_gradient_scale == 1.0:
        return combine_window_losses(window, losses)

    parts: list[torch.Tensor] = []
    report: dict[str, Any] = {"omitted": {}, "applied": {}}
    for term in sorted(window.enabled_terms):
        loss = losses.get(term)
        local_denominator = window.denominator(term)
        denominator = (global_denominators[term]
                       if global_denominators is not None
                       else local_denominator)
        if loss is None:
            if local_denominator > 0:
                report["omitted"][term] = "loss_not_supplied"
            else:
                report["omitted"][term] = "zero_local_eligible_data"
            continue
        weight = float(window.weights.get(term, DEFAULT_WEIGHTS.get(term, 0.0)))
        if isinstance(denominator, (int, float)):
            if isinstance(denominator, bool) or not math.isfinite(float(denominator)):
                raise RouterError(f"invalid denominator for {term!r}")
            if denominator <= 0:
                report["omitted"][term] = "zero_global_eligible_data"
                continue
            normalized = loss.total / float(denominator)
        elif torch.is_tensor(denominator):
            if denominator.numel() != 1:
                raise RouterError(f"denominator for {term!r} must be scalar")
            # Counts are nonnegative all-reduces. clamp_min keeps a globally
            # empty term a zero contribution without forcing an XLA .item()
            # synchronization in the hot path.
            normalized = loss.total / denominator.to(
                device=loss.total.device, dtype=loss.total.dtype).clamp_min(1.0)
        else:
            raise RouterError(f"unsupported denominator type for {term!r}")
        parts.append(float(replica_gradient_scale) * weight * normalized)
        if global_denominators is not None:
            report["applied"][term] = {
                "weight": weight,
                "denominator": "global_replica_sum",
                "local_denominator": local_denominator,
                "replicas": float(replica_gradient_scale),
            }
        else:
            report["applied"][term] = {
                "weight": weight, "denominator": int(denominator),
                "sum": float(loss.total.detach().item()),
            }
    if not parts:
        return torch.zeros(()), report
    total = parts[0]
    for part in parts[1:]:
        total = total + part
    return total, report


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
