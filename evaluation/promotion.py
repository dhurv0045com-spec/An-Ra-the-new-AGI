"""Separate model-capability and execution-integration promotion gates."""

from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import mean, pstdev
from typing import Iterable


@dataclass(frozen=True)
class PromotionDecision:
    allowed: bool
    gates: dict[str, bool]
    deltas: dict[str, float]
    reasons: tuple[str, ...]


class CapabilityPromotionGate:
    def __init__(
        self,
        *,
        protected_dimensions: tuple[str, ...] = ("identity",),
        confidence_z: float = 1.96,
    ) -> None:
        self.protected_dimensions = protected_dimensions
        self.confidence_z = float(confidence_z)

    @staticmethod
    def _seed_scores(reports: Iterable[dict[str, object]]) -> list[float]:
        return [float(report.get("overall", report.get("overall_score", 0.0))) for report in reports]

    def compare(
        self,
        baseline_reports: Iterable[dict[str, object]],
        candidate_reports: Iterable[dict[str, object]],
        *,
        owner_baseline: float,
        owner_candidate: float,
    ) -> PromotionDecision:
        baselines = list(baseline_reports)
        candidates = list(candidate_reports)
        if len(baselines) < 3 or len(candidates) < 3:
            raise ValueError("Capability promotion requires at least three seeded reports per model.")
        base_scores = self._seed_scores(baselines)
        cand_scores = self._seed_scores(candidates)
        base_mean = mean(base_scores)
        cand_mean = mean(cand_scores)
        standard_error = math.sqrt(
            pstdev(base_scores) ** 2 / len(base_scores)
            + pstdev(cand_scores) ** 2 / len(cand_scores)
        )
        lower_delta = cand_mean - base_mean - self.confidence_z * standard_error

        base_dims = baselines[0].get("dimensions", {})
        cand_dims = candidates[0].get("dimensions", {})
        dimensions_ok = all(
            float(cand_dims.get(name, 0.0)) >= float(base_dims.get(name, 0.0))
            for name in self.protected_dimensions
        )
        gates = {
            "three_seed_reproducibility": True,
            "aggregate_improvement": cand_mean > base_mean,
            "confidence_calibrated_improvement": lower_delta >= 0.0,
            "protected_dimensions_no_regression": dimensions_ok,
            "owner_suite_no_regression": float(owner_candidate) >= float(owner_baseline),
        }
        reasons = tuple(name for name, passed in gates.items() if not passed)
        return PromotionDecision(
            allowed=all(gates.values()),
            gates=gates,
            deltas={
                "overall_mean": cand_mean - base_mean,
                "confidence_lower_bound": lower_delta,
                "owner_suite": float(owner_candidate) - float(owner_baseline),
            },
            reasons=reasons,
        )


class DeploymentPromotionGate:
    REQUIRED = (
        "tool_schema",
        "rollback",
        "timeouts",
        "authorization",
        "robotics_boundary",
    )

    def evaluate(self, checks: dict[str, bool]) -> PromotionDecision:
        gates = {name: bool(checks.get(name, False)) for name in self.REQUIRED}
        return PromotionDecision(
            allowed=all(gates.values()),
            gates=gates,
            deltas={},
            reasons=tuple(name for name, passed in gates.items() if not passed),
        )
