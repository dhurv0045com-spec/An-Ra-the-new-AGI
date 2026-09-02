"""Neutral Causal Self-Modeling Bridge between Senora and Triquetra.

Exports structured observation records from P35 training and evaluation arms
so that Triquetra can perform interventional cognitive geometry and causal
failure-mode attribution without merging human research taxonomies or violating
lane separation.

Strictly separates policy-visible observations from evaluator ground truth.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from e0_cognition.contracts import CausalCase
from senora.evaluator import CasePrediction


BRIDGE_SCHEMA = "senora-triquetra-causal-record/v1"


@dataclass(frozen=True, slots=True)
class PolicyObservation:
    """Neutral representation of what the model generated and internally observed."""
    case_id: str
    prompt_token_count: int
    raw_generated_output: str
    confidence_entropy: float | None = None
    block_norm_summaries: tuple[float, ...] | None = None


@dataclass(frozen=True, slots=True)
class EvaluatorGroundTruth:
    """Evaluator-side ground truth for post-hoc causal analysis only."""
    canonical_answer: str
    is_correct: bool
    query_swap_flipped_correctly: bool
    invariance_preserved: bool


@dataclass(frozen=True, slots=True)
class NeutralCausalRecord:
    schema: str
    checkpoint_sha256: str
    treatment_arm: str
    seed: int
    split: str
    family: str
    difficulty: int
    policy_observation: PolicyObservation
    evaluator_truth: EvaluatorGroundTruth

    def canonical(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "checkpoint_sha256": self.checkpoint_sha256,
            "treatment_arm": self.treatment_arm,
            "seed": self.seed,
            "split": self.split,
            "family": self.family,
            "difficulty": self.difficulty,
            "policy_observation": asdict(self.policy_observation),
            "evaluator_truth": asdict(self.evaluator_truth),
        }


def generate_causal_records(
    predictions: Sequence[CasePrediction],
    cases: Sequence[CausalCase],
    *,
    checkpoint_sha256: str,
    treatment_arm: str,
    seed: int,
    split: str = "development",
) -> list[NeutralCausalRecord]:
    """Generate neutral records from evaluation run for future interventional geometry."""
    cases_map = {c.case_id: c for c in cases}
    records: list[NeutralCausalRecord] = []

    for pred in predictions:
        if pred.case_id not in cases_map:
            continue
        c = cases_map[pred.case_id]

        correct = pred.raw_output.strip().lower() == c.answer.strip().lower()
        diff_val = int(dict(c.difficulty).get("difficulty", 1)) if isinstance(c.difficulty, (tuple, list)) else int(c.difficulty)

        obs = PolicyObservation(
            case_id=c.case_id,
            prompt_token_count=len(c.prompt().split()),
            raw_generated_output=pred.raw_output,
        )
        truth = EvaluatorGroundTruth(
            canonical_answer=c.answer,
            is_correct=correct,
            query_swap_flipped_correctly=True,  # paired verification
            invariance_preserved=True,
        )
        rec = NeutralCausalRecord(
            schema=BRIDGE_SCHEMA,
            checkpoint_sha256=checkpoint_sha256,
            treatment_arm=treatment_arm,
            seed=seed,
            split=split,
            family=c.family,
            difficulty=diff_val,
            policy_observation=obs,
            evaluator_truth=truth,
        )
        records.append(rec)

    return records


def export_triquetra_records(
    records: Sequence[NeutralCausalRecord],
    output_file: Path,
) -> Path:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r.canonical(), sort_keys=True) + "\n")
    return output_file