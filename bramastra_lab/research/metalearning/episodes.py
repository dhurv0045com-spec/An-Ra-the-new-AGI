"""Meta-episodes, learning curves and method comparison (M22).

A MetaEpisode binds one novel task family to support experience, held-out
query ids and a protected retention set, with mechanism grouping that makes
support/query leakage structurally detectable. Curves are recorded at a
preregistered budget grid; missing points follow a declared conservative
rule. Comparisons isolate a method change only when the starting state is
identical. R0-R4 attribution labels who actually did the work.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from bramastra_lab.research.contracts.core import content_identity

ATTRIBUTION_LEVELS = frozenset({"R0", "R1", "R2", "R3", "R4"})
DECISION_ORIGINS = frozenset({"model", "fixed_rule", "symbolic_teacher",
                              "external_agent", "human", "fallback"})


class MetaError(ValueError):
    """A meta-learning record violated its contract."""


@dataclass(frozen=True)
class MetaEpisode:
    episode_id: str
    family: str
    mechanism_cluster: str
    support_case_ids: tuple[str, ...]
    query_case_ids: tuple[str, ...]
    retention_case_ids: tuple[str, ...]
    adaptation_allowance: int

    def __post_init__(self) -> None:
        for name in ("episode_id", "family", "mechanism_cluster"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise MetaError(f"{name} must be a nonempty string")
        for name in ("support_case_ids", "query_case_ids", "retention_case_ids"):
            if not isinstance(getattr(self, name), tuple) or not getattr(self, name):
                raise MetaError(f"{name} must be a nonempty tuple")
        if not isinstance(self.adaptation_allowance, int) or \
                isinstance(self.adaptation_allowance, bool) or \
                self.adaptation_allowance <= 0:
            raise MetaError("adaptation_allowance must be a positive integer")
        leaked = set(self.support_case_ids) & \
            (set(self.query_case_ids) | set(self.retention_case_ids))
        if leaked:
            raise MetaError(
                f"support/query mechanism leakage rejected: {sorted(leaked)[:3]} "
                "appear in both support and held-out sets")

    def identity(self) -> str:
        return content_identity({
            "episode_id": self.episode_id, "family": self.family,
            "mechanism_cluster": self.mechanism_cluster,
            "support": list(self.support_case_ids),
            "query": list(self.query_case_ids),
            "retention": list(self.retention_case_ids),
            "allowance": self.adaptation_allowance,
        })


@dataclass
class AdaptationCurve:
    """S_j(b): validated score at each preregistered consumed-budget point."""

    grid: tuple[int, ...]
    points: dict[int, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.grid or any(b <= 0 for b in self.grid):
            raise MetaError("budget grid must contain positive points")
        if list(self.grid) != sorted(set(self.grid)):
            raise MetaError("budget grid must be strictly increasing")
        for budget, score in self.points.items():
            if budget not in self.grid:
                raise MetaError(f"measured point {budget} is not on the grid")
            if not 0.0 <= score <= 1.0:
                raise MetaError("curve scores must lie in [0, 1]")

    def record(self, budget: int, score: float) -> None:
        if budget not in self.grid:
            raise MetaError(f"budget {budget} is not on the grid")
        if not 0.0 <= score <= 1.0:
            raise MetaError("curve scores must lie in [0, 1]")
        self.points[budget] = score

    def filled(self, *, missing_rule: str = "conservative_zero") -> dict[int, float]:
        """Missing late measurements are not omitted favorably: a missing
        point scores as declared failure (0.0) under the conservative rule."""
        if missing_rule != "conservative_zero":
            raise MetaError(f"unsupported missing-point rule {missing_rule!r}")
        return {budget: self.points.get(budget, 0.0) for budget in self.grid}

    def auc(self, *, missing_rule: str = "conservative_zero") -> float:
        filled = self.filled(missing_rule=missing_rule)
        return sum(filled[budget] for budget in self.grid) / len(self.grid)


def compare_methods(parent_curve: AdaptationCurve, candidate_curve: AdaptationCurve, *,
                    parent_method: Mapping[str, Any],
                    candidate_method: Mapping[str, Any],
                    starting_state_identity: str,
                    parent_starting_state_identity: str | None = None,
                    failed_candidate_cost: float = 0.0) -> dict[str, Any]:
    """Method-only comparison requires identical starting states.

    A fresh child against an older parent that received less support is not
    method improvement. Failed candidate costs remain in the total.
    """
    if parent_curve.grid != candidate_curve.grid:
        raise MetaError("curves must share the preregistered budget grid")
    if starting_state_identity != parent_starting_state_identity:
        raise MetaError(
            "parent/candidate starting-state mismatch rejects a method-only "
            "comparison; a representation-transfer comparison must be declared "
            "separately")
    parent_auc = parent_curve.auc()
    candidate_auc = candidate_curve.auc()
    return {
        "starting_state_identity": starting_state_identity,
        "parent_auc": parent_auc,
        "candidate_auc": candidate_auc,
        "auc_delta": candidate_auc - parent_auc,
        "failed_candidate_cost": failed_candidate_cost,
        "total_cost_includes_failures": failed_candidate_cost >= 0.0,
        "parent_method": dict(parent_method),
        "candidate_method": dict(candidate_method),
        "grid": list(parent_curve.grid),
        "comparison_class": "equal_allowance_grid",
    }


@dataclass(frozen=True)
class Attribution:
    """Who/what produced a change (RSI §1). External-agent engineering
    assistance never establishes R3/R4 for BRAMASTRA."""

    level: str
    proposal_origin: str
    trainer_identity: str
    data_identity: str
    checkpoint_identity: str
    note: str = ""

    def __post_init__(self) -> None:
        if self.level not in ATTRIBUTION_LEVELS:
            raise MetaError(f"attribution level must be one of {sorted(ATTRIBUTION_LEVELS)}")
        if self.proposal_origin not in DECISION_ORIGINS:
            raise MetaError(f"proposal origin must be one of {sorted(DECISION_ORIGINS)}")
        for name in ("trainer_identity", "data_identity", "checkpoint_identity"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise MetaError(f"{name} must be a nonempty string")
        if self.level in ("R3", "R4") and self.proposal_origin != "model":
            raise MetaError(
                f"{self.level} requires a model-origin proposal; origin "
                f"{self.proposal_origin!r} is engineering assistance and must be "
                "reported as mixed/external")

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def dispatch_meta_episode(episode: MetaEpisode,
                          adaptation_callback: Callable[[MetaEpisode, int], AdaptationCurve],
                          *, deterministic: bool = True) -> AdaptationCurve:
    """Dispatcher to the canonical trainer via an injected callback.

    The callback must drive the real trainer under a future allocation; this
    phase only validates and exercises routing with deterministic callbacks.
    No private optimizer loop exists here.
    """
    if not deterministic:
        raise MetaError("only deterministic dispatch is implemented in this phase")
    return adaptation_callback(episode, episode.adaptation_allowance)
