"""Representation, selection and realization measurements for E0."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from .contracts import (
    CausalCase,
    EvaluationSuite,
    INVARIANCE_PAIR_KINDS,
    SENSITIVITY_PAIR_KINDS,
)


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


@dataclass(frozen=True, slots=True)
class AssistanceMeasurement:
    raw_exact: bool
    assisted_exact: bool
    intervention_dependence: bool
    assistance_harm: bool


@dataclass(frozen=True, slots=True)
class PairMeasurement:
    sensitivity_total: int
    sensitivity_both_correct: int
    sensitivity_correct_flip: int
    invariance_total: int
    invariance_both_correct: int
    invariance_stable: int


def selection_eligible(case: CausalCase) -> bool:
    """Candidate-selection metrics exclude one-candidate realization controls."""

    return len(case.candidates) > 1 and case.family != "exact_contextual_copy"


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


def measure_assistance(raw_output: str, assisted_output: str, answer: str) -> AssistanceMeasurement:
    raw = raw_output.strip() == answer
    assisted = assisted_output.strip() == answer
    return AssistanceMeasurement(raw, assisted, assisted and not raw, raw and not assisted)


def measure_pair_behavior(
    suite: EvaluationSuite, predictions: Mapping[str, str]
) -> PairMeasurement:
    sensitivity_total = sensitivity_both = sensitivity_flip = 0
    invariance_total = invariance_both = invariance_stable = 0
    for pair in suite.pairs:
        if pair.base.case_id not in predictions or pair.changed.case_id not in predictions:
            raise ValueError(f"missing prediction for pair {pair.pair_id}")
        base_prediction, changed_prediction = predictions[pair.base.case_id], predictions[pair.changed.case_id]
        if pair.kind in SENSITIVITY_PAIR_KINDS:
            sensitivity_total += 1
            both_correct = base_prediction == pair.base.answer and changed_prediction == pair.changed.answer
            sensitivity_both += int(both_correct)
            sensitivity_flip += int(both_correct and base_prediction != changed_prediction)
        elif pair.kind in INVARIANCE_PAIR_KINDS:
            invariance_total += 1
            both_correct = base_prediction == pair.base.answer and changed_prediction == pair.changed.answer
            invariance_both += int(both_correct)
            invariance_stable += int(both_correct and base_prediction == changed_prediction)
    return PairMeasurement(
        sensitivity_total, sensitivity_both, sensitivity_flip,
        invariance_total, invariance_both, invariance_stable,
    )


def accuracy_by_difficulty(
    suite: EvaluationSuite, predictions: Mapping[str, str]
) -> dict[str, dict[int, float]]:
    buckets: dict[str, dict[int, list[int]]] = {}
    for case in suite.cases:
        if case.case_id not in predictions:
            raise ValueError(f"missing prediction for {case.case_id}")
        hit = int(predictions[case.case_id] == case.answer)
        for axis, value in case.difficulty:
            buckets.setdefault(axis, {}).setdefault(value, []).append(hit)
    return {
        axis: {value: sum(hits) / len(hits) for value, hits in sorted(levels.items())}
        for axis, levels in sorted(buckets.items())
    }


def selection_accuracy(
    suite: EvaluationSuite, predictions: Mapping[str, str]
) -> float:
    eligible = [case for case in suite.cases if selection_eligible(case)]
    if not eligible:
        raise ValueError("suite has no selection-eligible cases")
    return sum(predictions.get(case.case_id) == case.answer for case in eligible) / len(eligible)


def exact_accuracy(predictions: Sequence[str], answers: Sequence[str]) -> float:
    if len(predictions) != len(answers) or not answers:
        raise ValueError("predictions and nonempty answers must have equal length")
    return sum(p.strip() == a for p, a in zip(predictions, answers)) / len(answers)
