"""CoreSubjectManifest V2: explicit checkpoint custody, no ambiguous names.

V1 forced ``checkpoint_sha256 == checkpoint_file_sha256`` even though a
CheckpointStore checkpoint is a content-addressed object directory, not a
single file. V2 names exactly what each hash covers: the object identity,
the manifest bytes, the model payload bytes, the optimizer payload bytes,
the live parameter bytes, and the training-state bytes. V1 stays historical
and untouched; Triquetra adoption requirements live in the handshake V2
document, not in copied schemas.
"""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


SUBJECT_SCHEMA_V2 = "anra-v5-core-subject-manifest/v2"


def _assert_sha256(name: str, value: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def _sha256_hex(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class CoreSubjectManifestV2:
    schema: str
    checkpoint_object_sha256: str
    checkpoint_manifest_sha256: str
    model_payload_sha256: str
    optimizer_payload_sha256: str
    parameter_sha256: str
    training_state_sha256: str
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
    stage: str
    seed: int
    custody: str
    creation_receipt_sha256: str

    @classmethod
    def create(cls, **fields: Any) -> "CoreSubjectManifestV2":
        manifest = cls(schema=SUBJECT_SCHEMA_V2, **fields)  # type: ignore[arg-type]
        manifest.assert_valid()
        return manifest

    def assert_valid(self) -> None:
        if self.schema != SUBJECT_SCHEMA_V2:
            raise ValueError("unsupported core-subject-manifest schema")
        for name in (
            "checkpoint_object_sha256",
            "checkpoint_manifest_sha256",
            "model_payload_sha256",
            "optimizer_payload_sha256",
            "parameter_sha256",
            "training_state_sha256",
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
        if self.checkpoint_object_sha256 != self.checkpoint_manifest_sha256:
            raise ValueError(
                "object identity must equal the store manifest hash by construction"
            )
        if len(self.source_commit) != 40 or any(
            character not in "0123456789abcdef" for character in self.source_commit
        ):
            raise ValueError("source_commit must be a full lowercase git SHA-1")
        if self.parent_checkpoint_sha256 is not None:
            _assert_sha256("parent_checkpoint_sha256", self.parent_checkpoint_sha256)
        if self.global_update <= 0 or self.cumulative_training_tokens <= 0:
            raise ValueError("a subject exists only after trained updates")
        if not self.stage or not self.custody:
            raise ValueError("stage and custody identities are required")
        if self.seed < 0:
            raise ValueError("seed cannot be negative")

    def canonical(self) -> dict[str, object]:
        self.assert_valid()
        return asdict(self)

    def sha256(self) -> str:
        return _sha256_hex(_canonical_json(self.canonical()))

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "CoreSubjectManifestV2":
        expected = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        if set(value) != expected:
            raise ValueError("core-subject-manifest v2 fields do not match schema")
        return cls(**value)  # type: ignore[arg-type]


def verify_subject_artifacts(
    manifest: CoreSubjectManifestV2,
    *,
    checkpoint_root: Path,
    lineage_id: str,
    tokenizer_artifact_path: Path,
    model_spec: Any,
    torch_module: Any = None,
) -> dict[str, object]:
    """Hash every available artifact and compare against the manifest.

    Restores the checkpoint object through the fencing store (manifest hash
    and inventory verified), re-hashes model/optimizer/training-state
    payloads, loads weights into a fresh core to recompute the parameter
    hash, and checks tokenizer bytes plus model-spec identity. Anything that
    disagrees fails closed.
    """

    from v5_training.checkpoint import CheckpointStore

    if torch_module is None:
        import torch as torch_module
    torch = torch_module
    manifest.assert_valid()
    store = CheckpointStore(checkpoint_root, lineage_id)
    state, payloads = store.restore(manifest.checkpoint_object_sha256)
    checks: dict[str, bool] = {}
    checks["object_identity"] = (
        manifest.checkpoint_object_sha256 == manifest.checkpoint_manifest_sha256
    )
    checks["model_payload"] = (
        _sha256_hex(payloads["model.bin"]) == manifest.model_payload_sha256
    )
    checks["optimizer_payload"] = (
        _sha256_hex(payloads["optimizer.bin"]) == manifest.optimizer_payload_sha256
    )
    checks["training_state_payload"] = (
        _sha256_hex(payloads["training_state.json"]) == manifest.training_state_sha256
    )
    from v5_model.core import initialize

    fresh = initialize(model_spec, 0, torch_module=torch)
    fresh.load_state_dict(
        torch.load(
            io.BytesIO(payloads["model.bin"]),
            map_location="cpu",
            weights_only=True,
        )
    )
    digest = hashlib.sha256()
    for name, parameter in sorted(fresh.named_parameters()):
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(
            parameter.detach().to("cpu", dtype=torch.float32).contiguous().numpy().tobytes()
        )
    checks["parameter_bytes"] = digest.hexdigest() == manifest.parameter_sha256
    checks["tokenizer_artifact"] = (
        _sha256_file(tokenizer_artifact_path) == manifest.tokenizer_artifact_sha256
    )
    checks["model_spec"] = model_spec.sha256() == manifest.model_spec_sha256
    checks["counters"] = (
        state.global_update == manifest.global_update
        and state.cumulative_tokens == manifest.cumulative_training_tokens
    )
    receipt: dict[str, object] = {
        "schema": "anra-v5-subject-verification/v1",
        "subject_sha256": manifest.sha256(),
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
    }
    return receipt


__all__ = [
    "SUBJECT_SCHEMA_V2",
    "CoreSubjectManifestV2",
    "verify_subject_artifacts",
]
