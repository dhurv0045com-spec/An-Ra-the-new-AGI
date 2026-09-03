"""General evaluation engine: protocol -> task-level evidence -> receipt.

One engine instead of one script per benchmark.  An ``EvaluationProtocol``
binds the generator identity, split, seed, case count, decoding mode, scoring
mode, metrics, and statistical rule.  Executing a protocol through the gold
firewall produces immutable ``TaskLevelEvidence`` records; the
``EvaluationReceipt`` carries only protocol identity, adapter identity,
subject identity, and evidence hashes -- aggregates are derived from task
evidence, never asserted.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Mapping

from v5_registry.subject import CoreSubjectManifest
from v5_evaluation.firewall import (
    CommittedOutput,
    EvaluatorTruth,
    VisibleTask,
    build_evaluator_truth,
    build_visible_tasks,
    score_committed,
)


PROTOCOL_SCHEMA = "anra-v5-evaluation-protocol/v1"
EVIDENCE_SCHEMA = "anra-v5-task-evidence/v1"
RECEIPT_SCHEMA = "anra-v5-evaluation-receipt/v1"

EVALUATION_MODES = (
    "RAW_FREE_GENERATION",
    "RAW_CANDIDATE_SCORING",
    "CONSTRAINED_SELECTION",
    "CONNECTOR_ASSISTED",
    "ORACLE_ASSISTED",
)

RAW_MODES = frozenset({"RAW_FREE_GENERATION", "RAW_CANDIDATE_SCORING", "CONSTRAINED_SELECTION"})

ALLOWED_SPLITS = frozenset({"training", "development", "sealed", "fresh", "software_eval"})


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def _sha_of(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


@dataclass(frozen=True, slots=True)
class EvaluationProtocol:
    protocol_id: str
    generator_id: str
    generator_sha256: str
    split: str
    seed: int
    n_cases: int
    decoding_mode: str
    candidate_scoring_mode: str
    metrics: tuple[str, ...]
    statistical_rule: str

    def assert_valid(self) -> None:
        if not self.protocol_id or not self.generator_id:
            raise ValueError("protocol and generator identities are required")
        if len(self.generator_sha256) != 64 or any(
            c not in "0123456789abcdef" for c in self.generator_sha256
        ):
            raise ValueError("generator must be bound by SHA-256")
        if self.split not in ALLOWED_SPLITS:
            raise ValueError(f"protocol split is not allowed: {self.split}")
        if self.seed < 0 or self.n_cases <= 0:
            raise ValueError("protocol seed and case count are invalid")
        if self.decoding_mode not in EVALUATION_MODES:
            raise ValueError(f"unknown decoding mode: {self.decoding_mode}")
        if not self.metrics:
            raise ValueError("a protocol must declare its metrics")
        if not self.statistical_rule:
            raise ValueError("a protocol must declare its statistical rule")

    def sha256(self) -> str:
        self.assert_valid()
        return _sha_of(asdict(self))


@dataclass(frozen=True, slots=True)
class TaskLevelEvidence:
    """One immutable scored evaluation unit with full provenance."""

    evidence_schema: str
    task_id: str
    cluster_id: str
    surface_variant_id: str
    family: str
    difficulty: str
    split: str
    evaluation_mode: str
    subject_manifest_sha256: str
    checkpoint_sha256: str
    adapter_sha256: str
    protocol_sha256: str
    visible_prompt: str
    raw_output: str
    candidate_scores: tuple[float, ...] | None
    gold_reference: str
    correct: bool
    latency_seconds: float
    output_tokens: int
    prompt_tokens: int

    def sha256(self) -> str:
        data = asdict(self)
        data["candidate_scores"] = list(self.candidate_scores) if self.candidate_scores is not None else None
        return _sha_of(data)


@dataclass(frozen=True, slots=True)
class EvaluationReceipt:
    receipt_schema: str
    protocol_sha256: str
    adapter_sha256: str
    subject_manifest_sha256: str
    evaluation_mode: str
    n_tasks: int
    task_evidence_sha256: tuple[str, ...]
    aggregate_correct_rate: float
    derived_from_task_evidence: bool

    def sha256(self) -> str:
        return _sha_of(asdict(self))


def run_evaluation(
    *,
    protocol: EvaluationProtocol,
    subject: CoreSubjectManifest,
    adapter: Any,
    tasks: list[Mapping[str, Any]],
    clock: Callable[[], float] | None = None,
) -> tuple[EvaluationReceipt, list[TaskLevelEvidence]]:
    """Execute a protocol over truth-carrying task records behind the firewall.

    The records are projected into ``VisibleTask`` (structurally truth-free)
    before the adapter sees them; ``EvaluatorTruth`` is joined only after the
    model output is committed.  Aggregates are derived from task evidence.
    """

    protocol.assert_valid()
    import time

    clock = clock or time.perf_counter
    visible_tasks: list[VisibleTask] = build_visible_tasks(tasks)
    truths: list[EvaluatorTruth] = build_evaluator_truth(tasks)
    evidence: list[TaskLevelEvidence] = []
    for task, visible, truth in zip(tasks, visible_tasks, truths):
        candidates = tuple(str(candidate) for candidate in task.get("candidates", ()))
        started = clock()
        if protocol.decoding_mode == "RAW_FREE_GENERATION":
            raw_output = adapter.generate_free(visible.prompt)
            scores = None
        elif protocol.decoding_mode == "RAW_CANDIDATE_SCORING":
            scores = tuple(adapter.score_candidates("", visible.prompt, list(candidates)))
            raw_output = ""
        elif protocol.decoding_mode == "CONSTRAINED_SELECTION":
            raw_output = adapter.generate_constrained(visible.prompt, list(candidates))
            scores = tuple(adapter.score_candidates("", visible.prompt, list(candidates)))
        else:
            raise ValueError(
                f"mode {protocol.decoding_mode} requires an assisted adapter contract; "
                "raw modes never collapse into assisted ones"
            )
        latency = clock() - started
        committed = CommittedOutput(
            task_id=visible.task_id,
            output=raw_output,
            candidate_scores=scores,
        )
        scored = score_committed(committed, visible, truth)
        adapter_sha = (
            adapter.identity.sha256() if hasattr(adapter, "identity") else "0" * 64
        )
        token_count = getattr(adapter, "token_count", None)
        record = TaskLevelEvidence(
            evidence_schema=EVIDENCE_SCHEMA,
            task_id=visible.task_id,
            cluster_id=visible.cluster_id,
            surface_variant_id=str(task.get("surface_variant_id", visible.task_id)),
            family=visible.family,
            difficulty=visible.difficulty,
            split=protocol.split,
            evaluation_mode=protocol.decoding_mode,
            subject_manifest_sha256=subject.sha256(),
            checkpoint_sha256=subject.checkpoint_sha256,
            adapter_sha256=adapter_sha,
            protocol_sha256=protocol.sha256(),
            visible_prompt=visible.prompt,
            raw_output=scored.raw_output,
            candidate_scores=scored.candidate_scores,
            gold_reference=scored.gold,
            correct=scored.correct,
            latency_seconds=latency,
            output_tokens=int(token_count(scored.raw_output)) if token_count else 0,
            prompt_tokens=int(token_count(visible.prompt)) if token_count else 0,
        )
        evidence.append(record)
    aggregate = sum(1 for record in evidence if record.correct) / len(evidence)
    receipt = EvaluationReceipt(
        receipt_schema=RECEIPT_SCHEMA,
        protocol_sha256=protocol.sha256(),
        adapter_sha256=evidence[0].adapter_sha256 if evidence else "0" * 64,
        subject_manifest_sha256=subject.sha256(),
        evaluation_mode=protocol.decoding_mode,
        n_tasks=len(evidence),
        task_evidence_sha256=tuple(record.sha256() for record in evidence),
        aggregate_correct_rate=aggregate,
        derived_from_task_evidence=True,
    )
    return receipt, evidence


__all__ = [
    "ALLOWED_SPLITS",
    "EVALUATION_MODES",
    "EvaluationProtocol",
    "EvaluationReceipt",
    "RAW_MODES",
    "RECEIPT_SCHEMA",
    "TaskLevelEvidence",
    "run_evaluation",
]
