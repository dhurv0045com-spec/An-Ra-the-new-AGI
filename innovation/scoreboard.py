from __future__ import annotations

import json
from pathlib import Path

from innovation.schema import Hypothesis, InnovationScore


WEIGHTS = {
    "repo_leverage": 20,
    "verifier_strength": 20,
    "learning_value": 15,
    "novelty_inside_repo": 15,
    "implementation_smallness": 10,
    "safety_and_owner_control": 10,
    "failure_replay_value": 10,
}


def _contains(text: str, words: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(word in lowered for word in words)


def _scale(value: float, max_points: float) -> float:
    return max(0.0, min(max_points, float(value)))


def _decision(total: float) -> str:
    if total >= 80:
        return "implement"
    if total >= 60:
        return "experiment_first"
    return "research_only"


def score_hypothesis(hyp: Hypothesis) -> InnovationScore:
    """Score a falsifiable improvement hypothesis using the AIE formula."""
    text = " ".join(
        [
            hyp.description,
            hyp.falsifier,
            hyp.smallest_experiment,
            hyp.verifier_path,
            json.dumps(hyp.predicted_delta, sort_keys=True),
            " ".join(hyp.constraints),
        ]
    ).lower()

    repo_leverage = 12.0
    if _contains(text, ("training", "memory", "hal", "rlvr", "ouroboros", "verifier", "tokenizer", "system graph")):
        repo_leverage = 18.0
    if _contains(text, ("checkpoint", ".pt", "history")):
        repo_leverage -= 4.0

    verifier_strength = 8.0
    if _contains(text, ("pytest", "verifier", "rdkit", "qiskit", "verilator", "constraint", "lint", "unit test")):
        verifier_strength = 18.0
    if _contains(text, ("manual", "subjective")):
        verifier_strength -= 5.0

    learning_value = 8.0
    if _contains(text, ("failure", "replay", "falsifier", "dataset", "training", "reward")):
        learning_value = 14.0

    novelty_inside_repo = 7.0
    if _contains(text, ("new package", "cross-domain", "innovation", "hal", "dfc", "frontier")):
        novelty_inside_repo = 13.0
    if _contains(text, ("duplicate", "wrapper only")):
        novelty_inside_repo -= 3.0

    implementation_smallness = 8.0
    if _contains(text, ("one-line", "constructor", "smallest", "single file", "narrow")):
        implementation_smallness = 10.0
    if _contains(text, ("rewrite", "large migration", "checkpoint format")):
        implementation_smallness = 3.0

    safety_and_owner_control = 8.0
    if _contains(text, ("no mandatory dependency", "owner", "human", "refuse", "threshold", "safe", "constraint")):
        safety_and_owner_control = 10.0
    if _contains(text, ("autonomous push", "destructive", "external authority")):
        safety_and_owner_control = 4.0

    failure_replay_value = 4.0
    if _contains(text, ("failure_replay", "replay", "falsified", "error", "failed")):
        failure_replay_value = 10.0

    values = {
        "repo_leverage": _scale(repo_leverage, WEIGHTS["repo_leverage"]),
        "verifier_strength": _scale(verifier_strength, WEIGHTS["verifier_strength"]),
        "learning_value": _scale(learning_value, WEIGHTS["learning_value"]),
        "novelty_inside_repo": _scale(novelty_inside_repo, WEIGHTS["novelty_inside_repo"]),
        "implementation_smallness": _scale(implementation_smallness, WEIGHTS["implementation_smallness"]),
        "safety_and_owner_control": _scale(safety_and_owner_control, WEIGHTS["safety_and_owner_control"]),
        "failure_replay_value": _scale(failure_replay_value, WEIGHTS["failure_replay"]),
    }
    total = round(sum(values.values()), 3)
    return InnovationScore(hyp_id=hyp.hyp_id, total=total, decision=_decision(total), **values)


def write_report(scores: list[InnovationScore], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "scores": [score.to_dict() for score in scores],
        "weights": WEIGHTS,
        "count": len(scores),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
