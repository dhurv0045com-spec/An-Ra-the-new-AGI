"""Preregistered, hash-bound experiment contracts with matched arms.

An experiment exists as a machine-readable object -- never as a shell command
plus README.  Freezing is one-way: the spec's canonical SHA-256 becomes its
identity, the chronology binds each stage by payload hashes, and the
matched-arms comparator mechanically fails any control/treatment pair whose
identities differ beyond the declared treatment fields.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Mapping


SPEC_SCHEMA = "anra-v5-experiment-spec/v1"
RECEIPT_SCHEMA = "anra-v5-preregistration-receipt/v1"
INTERVENTION_SCHEMA = "anra-v5-training-intervention-record/v1"
CHRONOLOGY_SCHEMA = "anra-v5-experiment-chronology/v1"

CHRONOLOGY_STAGES = (
    "QUESTION",
    "PREREGISTRATION",
    "CODE_FREEZE",
    "EXECUTION",
    "RECEIPT",
    "ANALYSIS",
    "CLAIM",
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def _sha_of(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _assert_sha256(name: str, value: str | None) -> None:
    if value is None:
        return
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} must be a lowercase SHA-256")


@dataclass(frozen=True, slots=True)
class TrainingInterventionRecord:
    """Structured record of one planned training change."""

    hypothesis: str
    mechanism_target: str
    treatment_definition: str
    control_definition: str
    expected_behavioral_effect: str
    expected_failure_profile_effect: str
    risks: tuple[str, ...]

    def __post_init__(self) -> None:
        self.assert_valid()

    def assert_valid(self) -> None:
        fields = asdict(self)
        for name, value in fields.items():
            if name == "risks":
                continue
            if not value:
                raise ValueError(f"intervention record requires {name}")
        if not self.risks:
            raise ValueError("intervention record must name its risks")

    def sha256(self) -> str:
        self.assert_valid()
        return _sha_of(asdict(self))


@dataclass(frozen=True, slots=True)
class ExperimentSpec:
    experiment_id: str
    hypothesis: str
    intervention: TrainingInterventionRecord
    parent_checkpoint_sha256s: tuple[str, ...]
    model_spec_sha256: str
    tokenizer_artifact_sha256: str
    training_spec_sha256: str
    data_manifest_sha256: str
    optimizer_spec_sha256: str
    schedule_spec_sha256: str
    token_budget: int
    seeds: tuple[int, ...]
    evaluation_protocol_sha256: str
    promotion_rule: str
    stop_rule: str
    treatment_fields: tuple[str, ...] = ("treatment",)

    def assert_valid(self) -> None:
        if not self.experiment_id or not self.hypothesis:
            raise ValueError("experiment identity and hypothesis are required")
        self.intervention.assert_valid()
        for parent in self.parent_checkpoint_sha256s:
            _assert_sha256("parent checkpoint", parent)
        for name in (
            "model_spec_sha256",
            "tokenizer_artifact_sha256",
            "training_spec_sha256",
            "data_manifest_sha256",
            "optimizer_spec_sha256",
            "schedule_spec_sha256",
            "evaluation_protocol_sha256",
        ):
            _assert_sha256(name, getattr(self, name))
        if self.token_budget <= 0 or not self.seeds:
            raise ValueError("budget and seed policy are required")
        if not self.promotion_rule or not self.stop_rule:
            raise ValueError("promotion and stop rules must be preregistered")

    def canonical(self) -> dict[str, object]:
        self.assert_valid()
        data = asdict(self)
        data["intervention"] = asdict(self.intervention)
        return data

    def sha256(self) -> str:
        return _sha_of(self.canonical())


@dataclass(frozen=True, slots=True)
class PreregistrationReceipt:
    receipt_schema: str
    spec: dict[str, object]
    spec_sha256: str

    @classmethod
    def freeze(cls, spec: ExperimentSpec) -> "PreregistrationReceipt":
        spec.assert_valid()
        return cls(
            receipt_schema=RECEIPT_SCHEMA,
            spec=spec.canonical(),
            spec_sha256=spec.sha256(),
        )

    def verify(self) -> bool:
        return _sha_of(self.spec) == self.spec_sha256


@dataclass(frozen=True, slots=True)
class ExperimentChronology:
    """Hash-bound stage ordering: QUESTION -> ... -> CLAIM."""

    chronology_schema: str
    experiment_id: str
    events: tuple[dict[str, str], ...] = field(default=())

    @classmethod
    def begin(cls, spec: ExperimentSpec, *, question_payload_sha256: str) -> "ExperimentChronology":
        spec.assert_valid()
        _assert_sha256("question payload", question_payload_sha256)
        return cls(
            chronology_schema=CHRONOLOGY_SCHEMA,
            experiment_id=spec.experiment_id,
            events=({"stage": "QUESTION", "payload_sha256": question_payload_sha256},),
        )

    def record(self, *, stage: str, payload_sha256: str) -> "ExperimentChronology":
        if stage not in CHRONOLOGY_STAGES:
            raise ValueError(f"unknown chronology stage: {stage}")
        if not self.events:
            raise ValueError("chronology must begin with a QUESTION")
        expected_index = CHRONOLOGY_STAGES.index(stage)
        last_stage = self.events[-1]["stage"]
        last_index = CHRONOLOGY_STAGES.index(last_stage)
        if stage == last_stage:
            raise ValueError(f"stage {stage} already recorded; stages never repeat")
        if expected_index != last_index + 1:
            raise ValueError(
                f"chronology violation: {last_stage} must be followed by "
                f"{CHRONOLOGY_STAGES[last_index + 1]}, not {stage}"
            )
        _assert_sha256("payload", payload_sha256)
        return ExperimentChronology(
            chronology_schema=self.chronology_schema,
            experiment_id=self.experiment_id,
            events=self.events + ({"stage": stage, "payload_sha256": payload_sha256},),
        )


def assert_matched_arms(
    control: ExperimentSpec,
    treatment: ExperimentSpec,
    *,
    allowed_differences: tuple[str, ...],
) -> dict[str, object]:
    """Mechanically verify a control/treatment pair differs only where declared.

    Every identity field -- architecture, tokenizer, base corpus, optimizer,
    schedule, budget, seed policy -- must agree unless the field is explicitly
    declared as part of the treatment.  A supposedly matched experiment that
    differs unexpectedly fails closed.
    """

    control.canonical()
    treatment.canonical()
    differences: list[str] = []
    left = control.canonical()
    right = treatment.canonical()
    for name in left:
        if left[name] != right[name]:
            differences.append(name)
    declared = set(allowed_differences) | set(control.treatment_fields)
    unexpected = sorted(set(differences) - declared)
    if unexpected:
        raise ValueError(
            f"control/treatment mismatch in undeclared fields: {unexpected}; "
            "matched experiments must not drift silently"
        )
    return {
        "matched": True,
        "declared_differences": sorted(set(differences) & declared),
        "compared_fields": sorted(left),
    }


__all__ = [
    "CHRONOLOGY_SCHEMA",
    "CHRONOLOGY_STAGES",
    "ExperimentChronology",
    "ExperimentSpec",
    "INTERVENTION_SCHEMA",
    "PreregistrationReceipt",
    "RECEIPT_SCHEMA",
    "SPEC_SCHEMA",
    "TrainingInterventionRecord",
    "assert_matched_arms",
]
