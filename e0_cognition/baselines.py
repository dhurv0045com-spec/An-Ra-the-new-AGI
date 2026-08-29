"""Transparent non-neural baselines used to expose leaky E0 cases."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable

from .contracts import CausalCase, EvaluationSuite


Baseline = Callable[[CausalCase], str]


def deterministic_random(case: CausalCase) -> str:
    index = int(hashlib.sha256(case.case_id.encode()).hexdigest()[:8], 16) % len(case.candidates)
    return case.candidates[index]


def first_candidate(case: CausalCase) -> str:
    return case.candidates[0]


def last_candidate(case: CausalCase) -> str:
    return case.candidates[-1]


def lexical_overlap(case: CausalCase) -> str:
    query_tokens = set(re.findall(r"[A-Za-z0-9_-]+", case.query.lower()))
    scored: list[tuple[int, int, str]] = []
    for index, fact in enumerate(case.facts):
        overlap = len(query_tokens & set(re.findall(r"[A-Za-z0-9_-]+", fact.lower())))
        scored.append((overlap, -index, fact))
    fact = max(scored)[2]
    hits = [candidate for candidate in case.candidates if candidate in fact]
    return hits[-1] if hits else case.candidates[0]


def latest_fact(case: CausalCase) -> str:
    for fact in reversed(case.facts):
        hits = [candidate for candidate in case.candidates if candidate in fact]
        if hits:
            return hits[-1]
    return case.candidates[0]


def nearest_position(case: CausalCase) -> str:
    """Choose the candidate whose last context mention is nearest the query."""

    context = case.context()
    ranked = [(context.rfind(candidate), candidate) for candidate in case.candidates]
    present = [item for item in ranked if item[0] >= 0]
    return max(present)[1] if present else case.candidates[0]


def bag_of_words(case: CausalCase) -> str:
    query = set(re.findall(r"[A-Za-z0-9]+", case.query.lower()))
    best: tuple[float, str] = (-1.0, case.candidates[0])
    for candidate in case.candidates:
        containing = " ".join(fact for fact in case.facts if candidate in fact)
        tokens = set(re.findall(r"[A-Za-z0-9]+", containing.lower()))
        union = query | tokens
        score = len(query & tokens) / len(union) if union else 0.0
        best = max(best, (score, candidate))
    return best[1]


def broken_state_tracker(case: CausalCase) -> str:
    """Deliberately uses the oldest update, exposing state-overwrite scoring."""

    if "state" in case.family:
        for candidate in case.candidates:
            if candidate in case.facts[0]:
                return candidate
    return first_candidate(case)


def direct_retrieval_control(case: CausalCase) -> str:
    """An oracle only for non-compositional retrieval families."""

    if case.family in {
        "exact_contextual_copy",
        "nonce_identifier_retrieval",
        "entity_value_binding",
        "matched_direct_retrieval",
        "natural_binding_analogue",
    }:
        return case.answer
    return first_candidate(case)


def full_truth_oracle(case: CausalCase) -> str:
    return case.answer


BASELINES: dict[str, Baseline] = {
    "deterministic_random": deterministic_random,
    "first_candidate": first_candidate,
    "last_candidate": last_candidate,
    "lexical_overlap": lexical_overlap,
    "latest_fact": latest_fact,
    "nearest_position": nearest_position,
    "bag_of_words": bag_of_words,
    "broken_state_tracker": broken_state_tracker,
    "direct_retrieval_control": direct_retrieval_control,
    "full_truth_oracle": full_truth_oracle,
}


def evaluate_baseline(suite: EvaluationSuite, baseline: Baseline) -> dict[str, object]:
    correct = 0
    by_family: dict[str, list[int]] = {}
    for case in suite.cases:
        hit = int(baseline(case) == case.answer)
        correct += hit
        by_family.setdefault(case.family, []).append(hit)
    return {
        "accuracy": correct / len(suite.cases),
        "correct": correct,
        "total": len(suite.cases),
        "by_family": {
            family: sum(values) / len(values) for family, values in sorted(by_family.items())
        },
    }


def evaluate_all_baselines(suite: EvaluationSuite) -> dict[str, dict[str, object]]:
    return {name: evaluate_baseline(suite, fn) for name, fn in BASELINES.items()}
