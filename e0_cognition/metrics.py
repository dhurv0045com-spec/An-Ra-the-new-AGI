"""Representation, selection and realization measurements for E0."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True, slots=True)
class SelectionMeasurement:
    rank: int
    margin: float
    reciprocal_rank: float


@dataclass(frozen=True, slots=True)
class RealizationMeasurement:
    raw_exact: bool
    constrained_exact: bool
    conditional_realization: float


def measure_selection(scores: Mapping[str, float], answer: str) -> SelectionMeasurement:
    if answer not in scores:
        raise ValueError("answer is absent from candidate scores")
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    rank = next(i for i, (candidate, _) in enumerate(ordered, 1) if candidate == answer)
    competitors = [score for candidate, score in scores.items() if candidate != answer]
    margin = scores[answer] - max(competitors) if competitors else float("inf")
    return SelectionMeasurement(rank, margin, 1.0 / rank)


def query_conditioning_lift(
    conditioned_scores: Mapping[str, float], unconditioned_scores: Mapping[str, float], answer: str
) -> float:
    if answer not in conditioned_scores or answer not in unconditioned_scores:
        raise ValueError("answer is absent from one score mapping")
    return conditioned_scores[answer] - unconditioned_scores[answer]


def measure_realization(raw_output: str, constrained_output: str, answer: str) -> RealizationMeasurement:
    raw = raw_output.strip() == answer
    constrained = constrained_output.strip() == answer
    return RealizationMeasurement(raw, constrained, float(raw) if constrained else 0.0)


def exact_accuracy(predictions: Sequence[str], answers: Sequence[str]) -> float:
    if len(predictions) != len(answers) or not answers:
        raise ValueError("predictions and nonempty answers must have equal length")
    return sum(p.strip() == a for p, a in zip(predictions, answers)) / len(answers)
