from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class CapabilityGap:
    gap_id: str
    description: str
    detected_in: str
    severity: str
    evidence: list[str]
    detected_at: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Hypothesis:
    hyp_id: str
    gap_id: str
    description: str
    falsifier: str
    predicted_delta: dict
    constraints: list[str]
    smallest_experiment: str
    verifier_path: str
    created_at: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ExperimentResult:
    hyp_id: str
    verifier_output: dict
    actual_delta: dict
    error: dict
    passed: bool
    failure_reason: str
    replay_value: float
    completed_at: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class InnovationScore:
    hyp_id: str
    repo_leverage: float
    verifier_strength: float
    learning_value: float
    novelty_inside_repo: float
    implementation_smallness: float
    safety_and_owner_control: float
    failure_replay_value: float
    total: float
    decision: str

    def to_dict(self) -> dict:
        return asdict(self)
