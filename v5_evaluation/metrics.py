"""Pure metric computers for V5 evaluation dimensions.

Representation, addressing, transformation, selection, realization, and
substrate metrics derive from recorded outcomes only. Confidence uses
two-sided Wilson 95% intervals; promotion consumes lower bounds, never point
estimates. Conditional realization conditions on correct unassisted
selection with a 100-case eligibility floor.

METRIC_REGISTRY maps the only producible metric names to implementations
over enriched task records. Unknown names fail before any run.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Mapping


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


def _require_records(records: list[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    if not records:
        raise ValueError("metrics need at least one task record")
    return records


def _with_scores(
    records: list[Mapping[str, Any]],
) -> list[tuple[list[float], int]]:
    """Resolve (scores, gold_index) pairs; records without both are skipped."""

    resolved = []
    for record in records:
        scores = record.get("candidate_scores")
        candidates = record.get("candidates")
        gold = record.get("gold")
        if not scores or not candidates or gold is None:
            continue
        try:
            gold_index = list(candidates).index(gold)
        except ValueError:
            continue
        resolved.append(([float(value) for value in scores], gold_index))
    if not resolved:
        raise ValueError("no records carry scorable candidate sets")
    return resolved


def exact_accuracy(records: list[Mapping[str, Any]]) -> float:
    """Share of records with correct outcomes."""

    records = _require_records(records)
    return sum(1 for record in records if record.get("correct")) / len(records)


def candidate_rank1(records: list[Mapping[str, Any]]) -> float:
    """Share of scorable records where argmax selects gold."""

    resolved = _with_scores(_require_records(records))
    hits = 0
    for scores, gold_index in resolved:
        best = max(range(len(scores)), key=lambda i: scores[i])
        hits += int(best == gold_index)
    return hits / len(resolved)


def candidate_margin(records: list[Mapping[str, Any]]) -> float:
    """Mean top1-minus-top2 score gap over scorable records."""

    resolved = _with_scores(_require_records(records))
    gaps = []
    for scores, _gold_index in resolved:
        ordered = sorted(scores, reverse=True)
        gaps.append(ordered[0] - (ordered[1] if len(ordered) > 1 else 0.0))
    return sum(gaps) / len(gaps)


def gold_suffix_nll(records: list[Mapping[str, Any]]) -> float:
    """Mean negative gold-suffix log-probability over scorable records."""

    resolved = _with_scores(_require_records(records))
    return sum(-scores[gold_index] for scores, gold_index in resolved) / len(resolved)


def balanced_accuracy(records: list[Mapping[str, Any]]) -> float:
    """Mean of per-family accuracies; families, not cases, are the units."""

    records = _require_records(records)
    families: dict[str, list[bool]] = {}
    for record in records:
        families.setdefault(str(record.get("family", "")), []).append(bool(record.get("correct")))
    if not families:
        raise ValueError("no family labels to balance over")
    return sum(sum(values) / len(values) for values in families.values()) / len(families)


def conditional_realization_rate(records: list[Mapping[str, Any]]) -> float:
    """Realization share among correctly selected records (registry form)."""

    records = _require_records(records)
    eligible = [record for record in records if record.get("selection_correct")]
    if len(eligible) < MIN_CONDITIONAL_ELIGIBLE:
        raise ValueError(
            f"conditional realization needs at least {MIN_CONDITIONAL_ELIGIBLE} eligible cases"
        )
    return sum(1 for record in eligible if record.get("realized")) / len(eligible)


METRIC_REGISTRY: dict[str, Callable[[list[Mapping[str, Any]]], float]] = {
    "EXACT_ACCURACY": exact_accuracy,
    "CANDIDATE_RANK1": candidate_rank1,
    "CANDIDATE_MARGIN": candidate_margin,
    "GOLD_SUFFIX_NLL": gold_suffix_nll,
    "BALANCED_ACCURACY": balanced_accuracy,
    "CONDITIONAL_REALIZATION": conditional_realization_rate,
}


__all__ = [
    "METRIC_REGISTRY",
    "MIN_CONDITIONAL_ELIGIBLE",
    "Z95",
    "accuracy",
    "balanced_accuracy",
    "candidate_margin",
    "candidate_rank1",
    "conditional_realization",
    "conditional_realization_rate",
    "exact_accuracy",
    "gold_suffix_nll",
    "invariance_stability",
    "loss_regression",
    "sensitivity_flip_rate",
    "wilson_lcb",
]
