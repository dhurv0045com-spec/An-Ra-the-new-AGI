"""Schemas for future data, tokenizer, checkpoint, evaluation, and promotion receipts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


SHA256_LENGTH = 64


def _assert_sha256(name: str, value: str) -> None:
    if len(value) != SHA256_LENGTH or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} must be lowercase SHA-256")


@dataclass(frozen=True, slots=True)
class ArtifactIdentity:
    artifact_id: str
    sha256: str
    byte_size: int

    def assert_valid(self) -> None:
        if not self.artifact_id or self.byte_size <= 0:
            raise ValueError("artifact identity requires id and positive size")
        _assert_sha256("artifact sha256", self.sha256)


@dataclass(frozen=True, slots=True)
class CheckpointManifest:
    schema: str
    lineage_id: str
    checkpoint_id: str
    parent_checkpoint_sha256: str | None
    source_commit: str
    model_spec_sha256: str
    tokenizer_sha256: str
    data_manifest_sha256: str
    global_update: int
    cumulative_tokens: int
    tokens_by_source: Mapping[str, int]
    curriculum_phase: str
    sampler_cursor: str
    distributed_topology: str
    precision: str
    parameter_sha256: str
    optimizer_step_max: int
    rng_state_sha256: str

    def assert_valid(self) -> None:
        if self.schema != "anra-v5-checkpoint/v1":
            raise ValueError("unsupported checkpoint schema")
        if not all(
            (
                self.lineage_id,
                self.checkpoint_id,
                self.curriculum_phase,
                self.sampler_cursor,
                self.distributed_topology,
                self.precision,
            )
        ):
            raise ValueError("checkpoint identity and runtime fields are required")
        if len(self.source_commit) != 40 or any(
            character not in "0123456789abcdef" for character in self.source_commit
        ):
            raise ValueError("source commit must be a full lowercase Git SHA-1")
        for name, value in (
            ("model spec", self.model_spec_sha256),
            ("tokenizer", self.tokenizer_sha256),
            ("data manifest", self.data_manifest_sha256),
            ("parameters", self.parameter_sha256),
            ("rng state", self.rng_state_sha256),
        ):
            _assert_sha256(name, value)
        if self.parent_checkpoint_sha256 is not None:
            _assert_sha256("parent checkpoint", self.parent_checkpoint_sha256)
        if self.global_update < 0 or self.cumulative_tokens < 0:
            raise ValueError("steps and tokens cannot be negative")
        if any(not source or tokens < 0 for source, tokens in self.tokens_by_source.items()):
            raise ValueError("checkpoint source ledger is invalid")
        if self.optimizer_step_max != self.global_update:
            raise ValueError("optimizer step must equal global update")
        if sum(self.tokens_by_source.values()) != self.cumulative_tokens:
            raise ValueError("tokens_by_source must equal cumulative_tokens")


@dataclass(frozen=True, slots=True)
class PromotionDecision:
    schema: str
    checkpoint_sha256: str
    evaluation_receipt_sha256: str
    durability_receipt_sha256: str
    gate_spec_sha256: str
    decision: str
    passed_gates: tuple[str, ...]
    failed_gates: tuple[str, ...]
    raw_core_gate_passed: bool
    fresh_replication_passed: bool
    immutable_milestone: bool
    selection_basis: str
    signed_by: str
    detached_signature_sha256: str

    def assert_valid(self) -> None:
        if self.schema != "anra-v5-promotion/v2":
            raise ValueError("unsupported promotion schema")
        for name, value in (
            ("checkpoint", self.checkpoint_sha256),
            ("evaluation receipt", self.evaluation_receipt_sha256),
            ("durability receipt", self.durability_receipt_sha256),
            ("gate spec", self.gate_spec_sha256),
            ("detached signature", self.detached_signature_sha256),
        ):
            _assert_sha256(name, value)
        if self.decision not in {"promote", "reject", "inconclusive"}:
            raise ValueError("unsupported promotion decision")
        if not self.signed_by or not self.selection_basis:
            raise ValueError("promotion requires an independent signer and selection basis")
        if len(set(self.passed_gates)) != len(self.passed_gates) or len(
            set(self.failed_gates)
        ) != len(self.failed_gates):
            raise ValueError("promotion gate names must be unique")
        if set(self.passed_gates) & set(self.failed_gates):
            raise ValueError("a gate cannot both pass and fail")
        if self.selection_basis.lower() in {
            "latest",
            "latest checkpoint",
            "final",
            "final checkpoint",
            "last checkpoint",
        }:
            raise ValueError("chronology alone cannot select a checkpoint")
        if self.decision == "promote" and (
            self.failed_gates
            or not self.passed_gates
            or not self.raw_core_gate_passed
            or not self.fresh_replication_passed
            or not self.immutable_milestone
        ):
            raise ValueError("promotion requires every native, fresh, and immutability gate")


@dataclass(frozen=True, slots=True)
class DurabilityReceipt:
    schema: str
    checkpoint_sha256: str
    artifact_sha256: str
    redownload_sha256: str
    restore_receipt_sha256: str
    byte_size: int
    immutable: bool
    storage_provider: str
    object_identity: str
    independently_verified_by: str

    def assert_valid(self) -> None:
        if self.schema != "anra-v5-durability/v1":
            raise ValueError("unsupported durability schema")
        for name, value in (
            ("checkpoint", self.checkpoint_sha256),
            ("artifact", self.artifact_sha256),
            ("redownload", self.redownload_sha256),
            ("restore receipt", self.restore_receipt_sha256),
        ):
            _assert_sha256(name, value)
        if self.artifact_sha256 != self.redownload_sha256:
            raise ValueError("redownloaded durable artifact hash does not match upload")
        if self.byte_size <= 0 or not self.immutable:
            raise ValueError("durability requires a positive immutable artifact")
        if not self.storage_provider or not self.object_identity or not self.independently_verified_by:
            raise ValueError("durability custody identities are required")


@dataclass(frozen=True, slots=True)
class EvaluationReceipt:
    schema: str
    checkpoint_sha256: str
    evaluator_commit_sha256: str
    evaluator_config_sha256: str
    statistical_protocol_sha256: str
    model_adapter_sha256: str
    tokenizer_sha256: str
    fixture_commitments: Mapping[str, str]
    scoring_mode: str
    raw_core_metrics_sha256: str
    assisted_metrics_sha256: str
    substrate_metrics_sha256: str
    sealed_fixture_consumed: bool
    completed_at: str

    def assert_valid(self) -> None:
        if self.schema != "anra-v5-evaluation/v1":
            raise ValueError("unsupported evaluation schema")
        for name, value in (
            ("checkpoint", self.checkpoint_sha256),
            ("evaluator commit", self.evaluator_commit_sha256),
            ("evaluator config", self.evaluator_config_sha256),
            ("statistical protocol", self.statistical_protocol_sha256),
            ("model adapter", self.model_adapter_sha256),
            ("tokenizer", self.tokenizer_sha256),
            ("raw core metrics", self.raw_core_metrics_sha256),
            ("assisted metrics", self.assisted_metrics_sha256),
            ("substrate metrics", self.substrate_metrics_sha256),
        ):
            _assert_sha256(name, value)
        if self.scoring_mode not in {"sum", "token_normalized", "byte_normalized"}:
            raise ValueError("evaluation scoring mode is not certified")
        if not self.completed_at or not self.fixture_commitments:
            raise ValueError("evaluation completion and fixture commitments are required")
        for tier, value in self.fixture_commitments.items():
            if tier not in {"development", "sealed", "fresh", "natural"}:
                raise ValueError("unknown evaluation fixture tier")
            _assert_sha256(f"{tier} fixture", value)
        if "sealed" in self.fixture_commitments and not self.sealed_fixture_consumed:
            raise ValueError("sealed fixture use must consume the registered fixture")
