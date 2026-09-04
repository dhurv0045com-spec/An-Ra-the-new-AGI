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
from v5_evaluation.fixture import TaskFixtureBatch
from v5_evaluation.metrics import METRIC_REGISTRY
from v5_evaluation.stats import STATISTICAL_RULES


PROTOCOL_SCHEMA = "anra-v5-evaluation-protocol/v1"
EVIDENCE_SCHEMA = "anra-v5-task-evidence/v1"
RECEIPT_SCHEMA = "anra-v5-evaluation-receipt/v2"

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
    scoring_contract_sha256: str
    metric_values: tuple[tuple[str, float], ...]
    statistical_rule: str
    statistical_analyses: tuple[tuple[str, str], ...]
    evidence_artifact_sha256: str

    def sha256(self) -> str:
        data = asdict(self)
        data["metric_values"] = [list(item) for item in self.metric_values]
        data["statistical_analyses"] = [list(item) for item in self.statistical_analyses]
        return _sha_of(data)


def _adapter_scoring_contract(adapter: Any) -> str:
    """Resolve the adapter's advertised scoring-contract identity."""

    contract = getattr(adapter, "scoring_contract_sha256", None)
    if callable(contract):
        contract = contract()
    if not isinstance(contract, str) or not contract:
        raise ValueError(
            "evaluation adapter must advertise a scoring-contract identity"
        )
    return contract


def write_evidence_artifact(
    evidence: list[TaskLevelEvidence], path: Path
) -> str:
    """Persist canonical task-evidence JSONL; return the artifact SHA-256."""

    if not evidence:
        raise ValueError("cannot persist an empty evidence artifact")
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with path.open("wb") as handle:
        for record in evidence:
            data = asdict(record)
            if data["candidate_scores"] is not None:
                data["candidate_scores"] = list(data["candidate_scores"])
            line = _canonical_json(data) + b"\n"
            digest.update(line)
            handle.write(line)
    return digest.hexdigest()


def verify_evidence_artifact(path: Path, expected_sha256: str) -> list[dict[str, object]]:
    """Re-hash a persisted artifact and return its records; tampering fails."""

    digest = hashlib.sha256()
    records: list[dict[str, object]] = []
    with path.open("rb") as handle:
        for line in handle:
            digest.update(line)
            records.append(json.loads(line.decode("utf-8")))
    if digest.hexdigest() != expected_sha256:
        raise ValueError("task-evidence artifact disagrees with the receipt binding")
    if not records:
        raise ValueError("task-evidence artifact holds no records")
    return records


def run_evaluation(
    *,
    protocol: EvaluationProtocol,
    subject: CoreSubjectManifest,
    adapter: Any,
    fixture: TaskFixtureBatch,
    evidence_path: Path,
    clock: Callable[[], float] | None = None,
    paired_outcomes: list[tuple[bool, bool]] | None = None,
) -> tuple[EvaluationReceipt, list[TaskLevelEvidence]]:
    """Execute a protocol over a frozen fixture behind the firewall.

    Every binding is enforced: fixture case count equals ``n_cases`` (exact,
    never truncated or sampled), fixture split/seed/generator equal the
    protocol's, the adapter's scoring contract equals the protocol's, and
    metrics/statistical rules must exist in their registries before the run.
    Task evidence persists to ``evidence_path``; the receipt binds its hash.
    """

    protocol.assert_valid()
    fixture.assert_valid()
    if len(fixture.cases) != protocol.n_cases:
        raise ValueError(
            f"fixture holds {len(fixture.cases)} cases but protocol demands "
            f"exactly {protocol.n_cases}: no truncation, no sampling"
        )
    if fixture.split != protocol.split:
        raise ValueError("fixture split disagrees with protocol split")
    if fixture.seed != protocol.seed:
        raise ValueError("fixture seed disagrees with protocol seed")
    if fixture.generator_sha256 != protocol.generator_sha256:
        raise ValueError("fixture generator disagrees with protocol generator")
    contract = _adapter_scoring_contract(adapter)
    if protocol.candidate_scoring_mode != contract:
        raise ValueError("adapter scoring contract disagrees with the protocol")
    unknown_metrics = [name for name in protocol.metrics if name not in METRIC_REGISTRY]
    if unknown_metrics:
        raise ValueError(f"unknown production metrics: {unknown_metrics}")
    if protocol.statistical_rule not in STATISTICAL_RULES:
        raise ValueError(f"unknown statistical rule: {protocol.statistical_rule}")
    import time

    tasks = [dict(case) for case in fixture.cases]
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
    enriched = _enrich_for_metrics(evidence, tasks)
    metric_values = tuple(
        (name, float(METRIC_REGISTRY[name](enriched))) for name in protocol.metrics
    )
    rule = STATISTICAL_RULES[protocol.statistical_rule]
    if protocol.statistical_rule == "EXACT_MCNEMAR":
        if paired_outcomes is None:
            raise ValueError("McNemar analysis needs explicitly paired outcomes")
        analyses = (("EXACT_MCNEMAR", _sha_of(rule(paired_outcomes, seed=protocol.seed))),)
    elif protocol.statistical_rule == "CLUSTER_BOOTSTRAP_DELTA":
        analyses = (
            ("CLUSTER_BOOTSTRAP_DELTA", _sha_of(rule(enriched, seed=protocol.seed))),
        )
    else:
        analyses = (("WILSON_BINOMIAL", _sha_of(rule(enriched, seed=protocol.seed))),)
    artifact_sha = write_evidence_artifact(evidence, evidence_path)
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
        scoring_contract_sha256=contract,
        metric_values=metric_values,
        statistical_rule=protocol.statistical_rule,
        statistical_analyses=analyses,
        evidence_artifact_sha256=artifact_sha,
    )
    return receipt, evidence


def _enrich_for_metrics(
    evidence: list[TaskLevelEvidence], tasks: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Join evidence with fixture candidates/gold for metric computation."""

    by_task = {str(task["task_id"]): task for task in tasks}
    enriched: list[dict[str, Any]] = []
    for record in evidence:
        task = by_task[record.task_id]
        candidates = [str(candidate) for candidate in task.get("candidates", ())]
        try:
            gold_index: int | None = candidates.index(record.gold_reference)
        except ValueError:
            gold_index = None
        selection_correct: bool | None = None
        if record.candidate_scores is not None and gold_index is not None:
            best = max(
                range(len(record.candidate_scores)),
                key=lambda i: float(record.candidate_scores[i]),  # type: ignore[index]
            )
            selection_correct = best == gold_index
        elif record.candidate_scores is None:
            selection_correct = record.correct
        enriched.append(
            {
                "task_id": record.task_id,
                "cluster_id": record.cluster_id,
                "family": record.family,
                "correct": record.correct,
                "candidate_scores": list(record.candidate_scores)
                if record.candidate_scores is not None
                else None,
                "candidates": candidates,
                "gold": record.gold_reference,
                "gold_index": gold_index,
                "selection_correct": selection_correct,
                "realized": record.raw_output == record.gold_reference,
            }
        )
    return enriched


__all__ = [
    "ALLOWED_SPLITS",
    "EVALUATION_MODES",
    "EvaluationProtocol",
    "EvaluationReceipt",
    "RAW_MODES",
    "RECEIPT_SCHEMA",
    "TaskLevelEvidence",
    "run_evaluation",
    "verify_evidence_artifact",
    "write_evidence_artifact",
]
