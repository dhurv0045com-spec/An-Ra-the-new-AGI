"""Pure metric computers for V5 evaluation dimensions.

Representation, addressing, transformation, selection, realization, and
substrate metrics derive from recorded outcomes only. Confidence uses
two-sided Wilson 95% intervals; promotion consumes lower bounds, never point
estimates. Conditional realization conditions on correct unassisted
selection with a 100-case eligibility floor.
"""

from __future__ import annotations

import math


Z95 = 1.96
MIN_CONDITIONAL_ELIGIBLE = 100


def wilson_lcb(successes: int, trials: int, *, z: float = Z95) -> float:
    """Return the two-sided Wilson lower confidence bound."""

    if trials <= 0 or successes < 0 or successes > trials:
        raise ValueError("binomial counts are invalid")
    if z <= 0:
        raise ValueError("z-score must be positive")
    p = successes / trials
    denominator = 1 + z * z / trials
    center = p + z * z / (2 * trials)
    margin = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials))
    return max(0.0, (center - margin) / denominator)


def accuracy(correct: int, total: int) -> float:
    if total <= 0 or correct < 0 or correct > total:
        raise ValueError("accuracy counts are invalid")
    return correct / total


def sensitivity_flip_rate(flipped_correctly: int, pairs: int) -> float:
    """Share of counterfactual pairs where the answer flipped as mechanics demand."""

    return accuracy(flipped_correctly, pairs)


def invariance_stability(stable_and_correct: int, pairs: int) -> float:
    """Share of invariance pairs stable and correct on both variants."""

    return accuracy(stable_and_correct, pairs)


def conditional_realization(
    realized: int, eligible: int
) -> float:
    """Realization rate conditioned on correct unassisted selection."""

    if eligible < MIN_CONDITIONAL_ELIGIBLE:
        raise ValueError(
            f"conditional realization needs at least {MIN_CONDITIONAL_ELIGIBLE} eligible cases"
        )
    return accuracy(realized, eligible)


def loss_regression(baseline_loss: float, candidate_loss: float) -> float:
    """Relative loss change; positive means regression."""

    if not math.isfinite(baseline_loss) or not math.isfinite(candidate_loss):
        raise ValueError("losses must be finite")
    if baseline_loss <= 0:
        raise ValueError("baseline loss must be positive")
    return (candidate_loss - baseline_loss) / baseline_loss


__all__ = [
    "MIN_CONDITIONAL_ELIGIBLE",
    "Z95",
    "accuracy",
    "conditional_realization",
    "invariance_stability",
    "loss_regression",
    "sensitivity_flip_rate",
    "wilson_lcb",
]
