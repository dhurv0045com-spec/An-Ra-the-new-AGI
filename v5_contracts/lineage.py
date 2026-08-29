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
        if self.optimizer_step_max != self.global_update:
            raise ValueError("optimizer step must equal global update")
        if sum(self.tokens_by_source.values()) != self.cumulative_tokens:
            raise ValueError("tokens_by_source must equal cumulative_tokens")


@dataclass(frozen=True, slots=True)
class PromotionDecision:
    schema: str
    checkpoint_sha256: str
    evaluation_receipt_sha256: str
    decision: str
    failed_gates: tuple[str, ...]
    signed_by: str

    def assert_valid(self) -> None:
        _assert_sha256("checkpoint", self.checkpoint_sha256)
        _assert_sha256("evaluation receipt", self.evaluation_receipt_sha256)
        if self.decision not in {"promote", "reject", "inconclusive"}:
            raise ValueError("unsupported promotion decision")
        if self.decision == "promote" and self.failed_gates:
            raise ValueError("a promoted checkpoint cannot have failed gates")
        if not self.signed_by:
            raise ValueError("promotion requires an independent signer identity")
