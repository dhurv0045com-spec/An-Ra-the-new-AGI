"""CoreSubjectManifest: the canonical identity of one trained Core checkpoint.

A checkpoint is a first-class research subject, not merely weights.  The
manifest binds every identity required to reason about a subject --
architecture, tokenizer, data lineage, optimizer, schedule, curriculum,
lineage, training amount, and creation receipt -- with no placeholders and no
defaults.  Field names required by the Triquetra handshake validator
(``x_factor/manifest_validator.py`` on the triquetra branch) are reproduced
verbatim so a manifest validates there unmodified.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Mapping


SUBJECT_SCHEMA = "anra-v5-core-subject-manifest/v1"

# The Triquetra handshake contract: these fields, these names, no placeholders.
TRIQUETRA_REQUIRED_FIELDS = frozenset(
    {
        "schema",
        "checkpoint_file_sha256",
        "parameter_sha256",
        "model_spec_sha256",
        "tokenizer_artifact_sha256",
        "tokenizer_identity_sha256",
        "training_spec_sha256",
        "data_manifest_sha256",
        "pack_manifest_sha256",
        "source_commit",
        "cumulative_training_tokens",
        "global_update",
        "stage",
        "seed",
    }
)
PLACEHOLDER_VALUES = frozenset({"UNFILLED", "", None, "TODO", "PENDING"})


def _assert_sha256(name: str, value: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


@dataclass(frozen=True, slots=True)
class CoreSubjectManifest:
    schema: str
    checkpoint_sha256: str
    checkpoint_file_sha256: str
    parameter_sha256: str
    model_spec_sha256: str
    tokenizer_artifact_sha256: str
    tokenizer_identity_sha256: str
    training_spec_sha256: str
    data_manifest_sha256: str
    pack_manifest_sha256: str
    optimizer_spec_sha256: str
    schedule_spec_sha256: str
    curriculum_spec_sha256: str
    source_commit: str
    parent_checkpoint_sha256: str | None
    global_update: int
    cumulative_training_tokens: int
    training_stage: str
    stage: str
    seed: int
    custody: str
    creation_receipt_sha256: str

    @classmethod
    def create(
        cls,
        *,
        checkpoint_sha256: str,
        checkpoint_file_sha256: str,
        parameter_sha256: str,
        model_spec_sha256: str,
        tokenizer_artifact_sha256: str,
        tokenizer_identity_sha256: str,
        training_spec_sha256: str,
        data_manifest_sha256: str,
        pack_manifest_sha256: str,
        optimizer_spec_sha256: str,
        schedule_spec_sha256: str,
        curriculum_spec_sha256: str,
        source_commit: str,
        parent_checkpoint_sha256: str | None,
        global_update: int,
        cumulative_training_tokens: int,
        stage: str,
        seed: int,
        custody: str,
        creation_receipt_sha256: str,
    ) -> "CoreSubjectManifest":
        manifest = cls(
            schema=SUBJECT_SCHEMA,
            checkpoint_sha256=checkpoint_sha256,
            checkpoint_file_sha256=checkpoint_file_sha256,
            parameter_sha256=parameter_sha256,
            model_spec_sha256=model_spec_sha256,
            tokenizer_artifact_sha256=tokenizer_artifact_sha256,
            tokenizer_identity_sha256=tokenizer_identity_sha256,
            training_spec_sha256=training_spec_sha256,
            data_manifest_sha256=data_manifest_sha256,
            pack_manifest_sha256=pack_manifest_sha256,
            optimizer_spec_sha256=optimizer_spec_sha256,
            schedule_spec_sha256=schedule_spec_sha256,
            curriculum_spec_sha256=curriculum_spec_sha256,
            source_commit=source_commit,
            parent_checkpoint_sha256=parent_checkpoint_sha256,
            global_update=global_update,
            cumulative_training_tokens=cumulative_training_tokens,
            training_stage=stage,
            stage=stage,
            seed=seed,
            custody=custody,
            creation_receipt_sha256=creation_receipt_sha256,
        )
        manifest.assert_valid()
        return manifest

    def assert_valid(self) -> None:
        if self.schema != SUBJECT_SCHEMA:
            raise ValueError("unsupported core-subject-manifest schema")
        if self.checkpoint_sha256 != self.checkpoint_file_sha256:
            raise ValueError("checkpoint identity fields disagree")
        for name in (
            "checkpoint_sha256",
            "checkpoint_file_sha256",
            "parameter_sha256",
            "model_spec_sha256",
            "tokenizer_artifact_sha256",
            "tokenizer_identity_sha256",
            "training_spec_sha256",
            "data_manifest_sha256",
            "pack_manifest_sha256",
            "optimizer_spec_sha256",
            "schedule_spec_sha256",
            "curriculum_spec_sha256",
            "creation_receipt_sha256",
        ):
            _assert_sha256(name, getattr(self, name))
        if len(self.source_commit) != 40 or any(
            character not in "0123456789abcdef" for character in self.source_commit
        ):
            raise ValueError("source_commit must be a full lowercase git SHA-1")
        if self.parent_checkpoint_sha256 is not None:
            _assert_sha256("parent_checkpoint_sha256", self.parent_checkpoint_sha256)
        if self.global_update <= 0 or self.cumulative_training_tokens <= 0:
            raise ValueError("a subject exists only after trained updates")
        if not self.training_stage or not self.stage:
            raise ValueError("training stage identity is required")
        if not self.custody:
            raise ValueError("checkpoint custody must be recorded honestly")
        if self.seed < 0:
            raise ValueError("seed cannot be negative")
        canonical = asdict(self)
        for field in TRIQUETRA_REQUIRED_FIELDS:
            value = canonical.get(field)
            if field == "schema":
                continue
            if value is None or str(value).strip() in PLACEHOLDER_VALUES:
                raise ValueError(f"placeholder or missing handshake field: {field}")

    def canonical(self) -> dict[str, object]:
        self.assert_valid()
        return asdict(self)

    def sha256(self) -> str:
        return hashlib.sha256(_canonical_json(self.canonical())).hexdigest()

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "CoreSubjectManifest":
        expected = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        if set(value) != expected:
            raise ValueError("core-subject-manifest fields do not match schema")
        return cls(**value)  # type: ignore[arg-type]


def triquetra_validation(manifest: Mapping[str, object]) -> dict[str, object]:
    """Mirror of the Triquetra handshake validator for local pre-flight."""

    missing = sorted(TRIQUETRA_REQUIRED_FIELDS - set(manifest.keys()))
    placeholders = sorted(
        field
        for field in TRIQUETRA_REQUIRED_FIELDS & set(manifest.keys())
        if field != "schema" and str(manifest.get(field, "")).strip() in PLACEHOLDER_VALUES
    )
    return {
        "valid": not missing and not placeholders,
        "missing_fields": missing,
        "placeholder_fields": placeholders,
        "checked_fields": sorted(TRIQUETRA_REQUIRED_FIELDS),
    }


__all__ = [
    "PLACEHOLDER_VALUES",
    "SUBJECT_SCHEMA",
    "CoreSubjectManifest",
    "TRIQUETRA_REQUIRED_FIELDS",
    "triquetra_validation",
]
