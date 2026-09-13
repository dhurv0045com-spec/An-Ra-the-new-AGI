"""Typed supervision targets and objective denominators (M01/M05).

The router's denominators are different units and are never merged:
tokens, world-transition tokens, action decisions, value targets,
counterfactual groups and on-policy decisions. A `SupervisionWindow`
collects per-term eligibility counts and payload tensors/metadata for one
optimizer window, computed from batch metadata BEFORE backward.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from bramastra_lab.research.experience.trajectory import ExperienceError

OBJECTIVE_TERMS = ("token", "world", "action", "value", "pair", "pg")
DEFAULT_WEIGHTS = {"token": 1.0, "world": 1.0, "action": 1.0, "value": 0.5,
                   "pair": 0.0, "pg": 0.0, "ent": 0.0}


@dataclass
class SupervisionWindow:
    """Eligibility counts and payloads for one optimizer window."""

    weights: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    counts: dict[str, int] = field(default_factory=lambda: {term: 0 for term in OBJECTIVE_TERMS})
    payloads: dict[str, list[Any]] = field(
        default_factory=lambda: {term: [] for term in OBJECTIVE_TERMS})
    enabled_terms: frozenset[str] = frozenset(OBJECTIVE_TERMS)

    def add(self, term: str, count: int, payload: Any = None) -> None:
        if term not in OBJECTIVE_TERMS:
            raise ExperienceError(f"unknown objective term {term!r}")
        if term not in self.enabled_terms:
            raise ExperienceError(
                f"objective term {term!r} is disabled by feature switch and cannot "
                "receive supervision")
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ExperienceError("eligibility count must be a nonnegative integer")
        self.counts[term] += count
        if payload is not None:
            self.payloads[term].append(payload)

    def denominator(self, term: str) -> int:
        if term not in OBJECTIVE_TERMS:
            raise ExperienceError(f"unknown objective term {term!r}")
        return self.counts[term]

    def missing_required_terms(self) -> list[str]:
        """An enabled term with zero eligible data across the whole stage is a
        configuration/data error, not a successful zero loss."""
        return [term for term in sorted(self.enabled_terms)
                if self.counts[term] == 0]

    def summary(self) -> dict[str, Any]:
        return {"counts": dict(self.counts),
                "weights": {key: float(value) for key, value in self.weights.items()},
                "missing_enabled": self.missing_required_terms()}


@dataclass(frozen=True)
class WindowLoss:
    """One term's unnormalized sum plus its window denominator."""

    term: str
    total: Any          # torch scalar tensor (sum over eligible units)
    denominator: int

    def normalized(self, reduce_fn: Callable[[Any], float] | None = None):
        if self.denominator <= 0:
            raise ExperienceError(
                f"term {term!r} normalized with zero denominator".replace("term", "term "))
        return self.total / self.denominator


def combine_window_losses(window: SupervisionWindow,
                          losses: Mapping[str, WindowLoss]) -> tuple[Any, dict[str, Any]]:
    """Combine per-term normalized losses into one scalar (A1).

    Each term divides by its OWN window denominator before weighting; terms
    with zero eligible data are omitted with a recorded reason rather than
    counted as zero loss. Caller supplies per-term weight lookup.
    """
    import torch

    parts: list[Any] = []
    report: dict[str, Any] = {"omitted": {}, "applied": {}}
    supplied = set(losses)
    for term in sorted(window.enabled_terms):
        loss = losses.get(term)
        denominator = window.denominator(term)
        if loss is not None:
            if term not in OBJECTIVE_TERMS:
                raise ExperienceError(f"unknown loss term {term!r}")
            if denominator == 0:
                report["omitted"][term] = "zero_eligible_data"
                continue
            weight = float(window.weights.get(term, DEFAULT_WEIGHTS.get(term, 0.0)))
            parts.append(weight * loss.normalized())
            report["applied"][term] = {"weight": weight, "denominator": denominator,
                                       "sum": float(loss.total.detach().item())}
        elif denominator == 0:
            report["omitted"][term] = "zero_eligible_data"
        else:
            report["omitted"][term] = "loss_not_supplied"
    if not parts:
        return torch.zeros(()), report
    total = parts[0]
    for part in parts[1:]:
        total = total + part
    return total, report
