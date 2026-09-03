"""Prospective cognition schemas (Mission 27/28). SOFTWARE_DEMONSTRATED only.

PredictionBeforeInterventionRecord: committed BEFORE intervention runs.
CausalResponseProfile: checkpoint-agnostic per-task response container.
No human bottleneck labels. Mock-matrix verifiable without neural runs.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field


def _canon(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


@dataclass(frozen=True, slots=True)
class PredictionBeforeInterventionRecord:
    checkpoint_sha: str
    task_id: str
    observation_hash: str
    candidate_interventions: tuple[str, ...]
    predicted_response_distribution: tuple[float, ...]
    predicted_best_intervention: str
    uncertainty: float
    sequence_id: int = 0

    def __post_init__(self):
        if len(self.candidate_interventions) != len(self.predicted_response_distribution):
            raise ValueError("interventions and distribution length mismatch")
        if self.predicted_best_intervention not in self.candidate_interventions:
            raise ValueError("best intervention not in candidates")
        if not 0.0 <= self.uncertainty <= 1.0:
            raise ValueError("uncertainty must be in [0,1]")


def commit_prediction(rec: PredictionBeforeInterventionRecord) -> dict:
    body = {"checkpoint_sha": rec.checkpoint_sha, "task_id": rec.task_id,
            "observation_hash": rec.observation_hash,
            "candidate_interventions": list(rec.candidate_interventions),
            "predicted_response_distribution": list(rec.predicted_response_distribution),
            "predicted_best_intervention": rec.predicted_best_intervention,
            "uncertainty": rec.uncertainty, "sequence_id": rec.sequence_id}
    return {**body, "commitment_hash": _sha(_canon(body))}


@dataclass
class CausalResponseProfile:
    task_id: str
    raw_result: int
    legal_interventions: dict = field(default_factory=dict)
    oracle_ceiling: int | None = None
    substrate_adequacy: str = "UNKNOWN"
    identifiability_status: str = "UNKNOWN"

    def add(self, intervention_id: str, observed_result: int,
            behavioral_delta: int = 0, score_delta: float = 0.0, cost: int = 1,
            predicted: int | None = None):
        self.legal_interventions[intervention_id] = {
            "predicted": predicted, "observed_result": observed_result,
            "behavioral_delta": behavioral_delta, "score_delta": score_delta,
            "cost": cost}
