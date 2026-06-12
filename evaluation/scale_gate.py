"""Scale-up gate based on measured capability and compute efficiency."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScaleGateResult:
    allowed: bool
    scaling_efficiency: float
    gates: dict[str, bool]


def evaluate_scale_up(
    *,
    smaller_score: float,
    larger_score: float,
    smaller_compute: float,
    larger_compute: float,
    identity_similarity: float,
    max_compute_budget: float,
) -> ScaleGateResult:
    if min(smaller_compute, larger_compute) <= 0:
        raise ValueError("Compute measurements must be positive.")
    capability_gain = float(larger_score) - float(smaller_score)
    compute_ratio = float(larger_compute) / float(smaller_compute)
    efficiency = capability_gain / compute_ratio
    gates = {
        "larger_is_better": capability_gain > 0.0,
        "positive_scaling_efficiency": efficiency > 0.0,
        "identity_transfer_stable": float(identity_similarity) >= 0.90,
        "within_compute_budget": float(larger_compute) <= float(max_compute_budget),
    }
    return ScaleGateResult(all(gates.values()), efficiency, gates)
