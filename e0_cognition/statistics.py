"""Preregistered, dependency-free statistical helpers for E0."""

from __future__ import annotations

import math

from .contracts import EvaluationSuite


def uniform_candidate_chance(suite: EvaluationSuite) -> float:
    return sum(1.0 / len(case.candidates) for case in suite.cases) / len(suite.cases)


def wilson_interval(correct: int, total: int, *, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0 or not 0 <= correct <= total:
        raise ValueError("require 0 <= correct <= positive total")
    p = correct / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return center - radius, center + radius


def approximate_two_proportion_n_per_arm(
    p0: float, p1: float, *, alpha_z: float = 1.959963984540054, power_z: float = 0.8416212335729143
) -> int:
    """Conservative normal-approximation planning number, stated as approximate."""

    if not 0 < p0 < p1 < 1:
        raise ValueError("require 0 < p0 < p1 < 1")
    pooled = (p0 + p1) / 2
    numerator = (
        alpha_z * math.sqrt(2 * pooled * (1 - pooled))
        + power_z * math.sqrt(p0 * (1 - p0) + p1 * (1 - p1))
    ) ** 2
    return math.ceil(numerator / ((p1 - p0) ** 2))
