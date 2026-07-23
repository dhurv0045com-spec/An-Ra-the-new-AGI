"""Immutable, resumable publication for An-Ra training checkpoints.

The trainer owns the checkpoint payload.  This module owns its durable life
after the local atomic save: content-addressed chunks, an immutable manifest,
verified replica receipts, and a canonical pointer that is never published
before every referenced byte has been verified.

Mounted Google Drive and ordinary filesystem replicas deliberately share the
same backend.  The byte-level contract does not depend on a cloud SDK, which
makes it usable from Colab, a laptop sync folder, or the cluster publisher.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import json
import os
import queue
import shutil
import threading
import time
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

import torch

DURABILITY_SCHEMA_VERSION = 1
DEFAULT_CHUNK_SIZE_BYTES = 128 * 1024 * 1024
DEFAULT_COPY_BLOCK_BYTES = 4 * 1024 * 1024
DEFAULT_MAX_COPY_STREAMS = 2
HOT_STORAGE_LIMIT_BYTES = 12 * 1024**3
DEFAULT_KEEP_FULL = 2
DEFAULT_KEEP_COMPACT = 2

FULL_RESUME = "full_resume"
FP16_INFERENCE = "fp16_inference"


class ArtifactClass(str, Enum):  # noqa: UP042 - Python 3.10 compatibility
    FULL_RESUME = FULL_RESUME
    FP16_INFERENCE = FP16_INFERENCE


class DurabilityState(str, Enum):  # noqa: UP042 - Python 3.10 compatibility
    LOCAL_SAVED = "local_saved"
    STAGED = "staged"
    CANONICAL_VERIFIED = "canonical_verified"
    PROTECTED = "protected"


STATE_ORDER = {
    DurabilityState.LOCAL_SAVED: 0,
    DurabilityState.STAGED: 1,
    DurabilityState.CANONICAL_VERIFIED: 2,
    DurabilityState.PROTECTED: 3,
}


class DurabilityError(RuntimeError):
    """Base class for durability contract failures."""


class DurabilityCorruptionError(DurabilityError):
    """A content-addressed object does not match its declared digest."""


class PublicationError(DurabilityError):
    """A snapshot could not reach the requested replica state."""


class ResumeArtifactError(DurabilityError):
    """An artifact is not eligible for exact training resume."""


def _utc_timestamp() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")  # noqa: UP017


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path, *, block_size: int = DEFAULT_COPY_BLOCK_BYTES) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(payload: object) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _json_safe(value: object) -> object:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return {"sha256": _sha256_bytes(value), "size_bytes": len(value)}
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return repr(value)


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_json(path: Path, payload: object) -> None:
    _atomic_write_bytes(path, _canonical_json(payload) + b"\n")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise DurabilityError(f"Expected a JSON object: {path}")
    return payload


def _truthy_environment(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _emit_evidence(
    kind: str,
    payload: Mapping[str, object],
    *,
    artifact_refs: list[dict[str, str]] | None = None,
) -> None:
    """Publish durability truth to the stream shared by Matrix and ThirdEye."""

    from runtime.evidence_stream import append_evidence

    append_evidence(
        source="training.checkpoint_durability",
        kind=kind,
        payload=dict(_json_safe(dict(payload))),
        run_id=os.environ.get("ANRA_CLUSTER_JOB_ID", ""),
        artifact_refs=artifact_refs,
        require_signature=_truthy_environment("ANRA_REQUIRE_SIGNED_EVIDENCE"),
    )


def _validate_identifier(value: str, *, label: str) -> str:
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")
    if not value or len(value) > 200 or any(character not in allowed for character in value):
        raise DurabilityError(f"Unsafe {label}: {value!r}")
    return value


def _validate_sha256(value: str, *, label: str) -> str:
    normalized = value.lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise DurabilityCorruptionError(f"Invalid {label} SHA-256: {value!r}")
    return normalized


def durability_required_from_environment() -> bool:
    """Return whether this process must receive a remote durability ACK."""
    return any(
        _truthy_environment(name)
        for name in (
            "ANRA_REQUIRE_DURABLE_ACK",
            "ANRA_REQUIRE_SHARED_MASTER",
            "ANRA_CLUSTER_MODE",
        )
    ) or bool(os.environ.get("ANRA_CLUSTER_JOB_ID", "").strip())


def build_checkpoint_lineage(payload: Mapping[str, object]) -> dict[str, object]:
    """Expose the complete schema-9 continuity contract without tensor data."""
    model_config = _json_safe(payload.get("model_config", {}))
    tokenizer = _json_safe(payload.get("tokenizer_contract", {}))
    dataset_hashes = _json_safe(
        payload.get("dataset_manifest_hashes", payload.get("data_manifests", {}))
    )
    training_recipe = _json_safe(payload.get("training_recipe", {}))
    architecture_sha256 = _sha256_bytes(_canonical_json(model_config))
    dataset_contract_sha256 = _sha256_bytes(_canonical_json(dataset_hashes))
    recipe_sha256 = _sha256_bytes(_canonical_json(training_recipe))
    recipe = dict(training_recipe) if isinstance(training_recipe, Mapping) else {}
    seed_contract = payload.get("seed_contract", {})
    seed_payload = dict(seed_contract) if isinstance(seed_contract, Mapping) else {}
    explicit_lineage = str(
        payload.get("lineage_id")
        or os.environ.get("ANRA_CHECKPOINT_LINEAGE_ID", "")
        or os.environ.get("ANRA_CLUSTER_CAMPAIGN_ID", "")
    ).strip()
    model_profile = str(recipe.get("model_profile", "unknown"))
    lineage_id = (
        f"{explicit_lineage}/{model_profile}"
        if explicit_lineage
        else f"local/{architecture_sha256[:16]}/seed-{seed_payload.get('seed', 'unknown')}"
    )
    return {
        "lineage_id": lineage_id,
        "checkpoint_schema_version": int(
            payload.get("checkpoint_schema_version", 0) or 0
        ),
        "source_commit": str(payload.get("source_commit", "unknown")),
        "architecture": {
            "sha256": architecture_sha256,
            "config": model_config,
        },
        "tokenizer": tokenizer,
        "data": {
            "profile": str(payload.get("data_profile", "unknown")),
            "layout": str(payload.get("training_data_layout", "unknown")),
            "manifest_hashes": dataset_hashes,
            "contract_sha256": dataset_contract_sha256,
        },
        "training": {
            "recipe": training_recipe,
            "recipe_sha256": recipe_sha256,
            "seed_contract": _json_safe(seed_contract),
            "migration_provenance": _json_safe(
                payload.get("migration_provenance")
            ),
        },
        "progress": {
            "global_step": int(
                payload.get("global_step", payload.get("step", 0)) or 0
            ),
            "tokens_seen": int(payload.get("tokens_seen", 0) or 0),
            "sessions_completed": int(payload.get("sessions_completed", 0) or 0),
            "continuation_token_counts": _json_safe(
                payload.get("continuation_token_counts", {})
            ),
            "raw_window_consumption": _json_safe(
                payload.get("raw_window_consumption", {})
            ),
            "token_window": _json_safe(payload.get("token_window", {})),
        },
        "continuity": {
            "completed_optimizer_boundary": (
                payload.get("completed_optimizer_boundary") is True
            ),
            "accum_micro_steps": int(payload.get("accum_micro_steps", 0) or 0),
            "data_sampler_state": _json_safe(payload.get("data_sampler_state", {})),
            "components": {
                name: name in payload
                for name in ("model", "optimizer", "scheduler", "scaler", "rng_states")
            },
        },
    }


def assert_resume_artifact_class(blob: object, checkpoint: Path) -> None:
    """Reject inference-only artifacts before any model tensors are applied."""
    if not isinstance(blob, Mapping):
        return
    artifact_class = str(blob.get("checkpoint_artifact_class", FULL_RESUME))
    if artifact_class != FULL_RESUME:
        raise ResumeArtifactError(
            f"Checkpoint {checkpoint} is {artifact_class!r}; exact training resume "
            f"requires {FULL_RESUME!r}."
        )


def create_fp16_inference_artifact(
    checkpoint: Path,
    output_path: Path,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    """Create a hash-bound model-only artifact that can never resume training."""

    source = Path(checkpoint)
    output = Path(output_path)
    if not source.is_file():
        raise FileNotFoundError(f"Full-resume checkpoint does not exist: {source}")
    if output.exists() and not overwrite:
        raise FileExistsError(f"Compact artifact already exists: {output}")
    blob = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(blob, Mapping):
        raise ResumeArtifactError("Compact export requires a structured full-resume checkpoint")
    assert_resume_artifact_class(blob, source)
    lineage = blob.get("checkpoint_lineage")
    if not isinstance(lineage, Mapping):
        lineage = build_checkpoint_lineage(blob)
    _validate_resume_lineage(lineage)
    raw_model = blob.get("model_state_dict", blob.get("model"))
    if not isinstance(raw_model, Mapping) or not raw_model:
        raise ResumeArtifactError("Full-resume checkpoint has no model tensors")
    if not all(
        isinstance(name, str) and isinstance(value, torch.Tensor)
        for name, value in raw_model.items()
    ):
        raise ResumeArtifactError("Full-resume checkpoint model state contains non-tensors")
    compact_model = {
        name: (
            value.detach().to(device="cpu", dtype=torch.float16)
            if value.is_floating_point()
            else value.detach().to(device="cpu")
        )
        for name, value in raw_model.items()
    }
    payload: dict[str, object] = {
        "checkpoint_schema_version": int(blob.get("checkpoint_schema_version", 0)),
        "checkpoint_artifact_class": FP16_INFERENCE,
        "training_resume_allowed": False,
        "model": compact_model,
        "model_config": _json_safe(blob.get("model_config", {})),
        "tokenizer_contract": _json_safe(blob.get("tokenizer_contract", {})),
        "checkpoint_lineage": dict(_json_safe(lineage)),
        "source_full_resume_sha256": sha256_file(source),
        "source_commit": str(blob.get("source_commit", "unknown")),
        "global_step": int(blob.get("global_step", blob.get("step", 0)) or 0),
        "tokens_seen": int(blob.get("tokens_seen", 0) or 0),
        "growth_provenance": _json_safe(blob.get("growth_provenance", {})),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    try:
        torch.save(payload, temporary)
        with temporary.open("r+b") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "path": str(output),
        "sha256": sha256_file(output),
        "size_bytes": output.stat().st_size,
        "source_full_resume_sha256": payload["source_full_resume_sha256"],
        "global_step": payload["global_step"],
        "lineage": payload["checkpoint_lineage"],
    }


def _validate_resume_lineage(lineage: Mapping[str, object]) -> None:
    if int(lineage.get("checkpoint_schema_version", 0) or 0) != 9:
        raise DurabilityError("A full_resume snapshot requires checkpoint schema 9 lineage")
    architecture = dict(lineage.get("architecture", {}))
    tokenizer = dict(lineage.get("tokenizer", {}))
    data = dict(lineage.get("data", {}))
    training = dict(lineage.get("training", {}))
    continuity = dict(lineage.get("continuity", {}))
    components = dict(continuity.get("components", {}))
    missing: list[str] = []
    if not str(lineage.get("lineage_id", "")).strip():
        missing.append("lineage_id")
    if not architecture.get("sha256"):
        missing.append("architecture.sha256")
    if not (tokenizer.get("sha256") or tokenizer.get("vocabulary_sha256")):
        missing.append("tokenizer.sha256")
    if not data.get("contract_sha256"):
        missing.append("data.contract_sha256")
    if not training.get("recipe_sha256"):
        missing.append("training.recipe_sha256")
    if continuity.get("completed_optimizer_boundary") is not True:
        missing.append("continuity.completed_optimizer_boundary")
    for component in ("model", "optimizer", "scheduler", "scaler", "rng_states"):
        if components.get(component) is not True:
            missing.append(f"continuity.components.{component}")
    if missing:
        raise DurabilityError(
            "A full_resume snapshot has incomplete schema-9 lineage: " + ", ".join(missing)
        )


@dataclass(frozen=True)
class ChunkRecord:
    index: int
    offset: int
    size_bytes: int
    sha256: str

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> ChunkRecord:
        record = cls(
            index=int(payload["index"]),
            offset=int(payload["offset"]),
            size_bytes=int(payload["size_bytes"]),
            sha256=_validate_sha256(str(payload["sha256"]), label="chunk"),
        )
        if record.index < 0 or record.offset < 0 or record.size_bytes <= 0:
            raise DurabilityCorruptionError(f"Invalid chunk coordinates: {record}")
        return record

    def to_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "offset": self.offset,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class SnapshotRef:
    outbox_root: Path
    snapshot_id: str
    manifest_path: Path
    manifest_sha256: str
    artifact_class: ArtifactClass
    global_step: int

    @property
    def snapshot_dir(self) -> Path:
        return self.outbox_root / "snapshots" / self.snapshot_id


class CheckpointOutbox:
    """Registers immutable checkpoint bytes in a local content-addressed outbox."""

    def __init__(
        self,
        root: Path,
        *,
        chunk_size_bytes: int = DEFAULT_CHUNK_SIZE_BYTES,
    ) -> None:
        if chunk_size_bytes < 1:
            raise ValueError("chunk_size_bytes must be positive")
        self.root = Path(root)
        self.chunk_size_bytes = int(chunk_size_bytes)

    def chunk_path(self, sha256: str) -> Path:
        return self.root / "chunks" / sha256[:2] / f"{sha256}.chunk"

    def snapshot_dir(self, snapshot_id: str) -> Path:
        return self.root / "snapshots" / _validate_identifier(
            snapshot_id,
            label="snapshot id",
        )

    def status_path(self, snapshot_id: str) -> Path:
        return self.snapshot_dir(snapshot_id) / "status.json"

    def register_checkpoint(
        self,
        checkpoint: Path,
        *,
        artifact_class: ArtifactClass | str = ArtifactClass.FULL_RESUME,
        lineage: Mapping[str, object] | None = None,
    ) -> SnapshotRef:
        checkpoint = Path(checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
        if checkpoint.stat().st_size <= 0:
            raise DurabilityError(f"Checkpoint is empty: {checkpoint}")
        artifact_class = ArtifactClass(artifact_class)
        safe_lineage = dict(_json_safe(dict(lineage or {})))
        if artifact_class is ArtifactClass.FULL_RESUME:
            _validate_resume_lineage(safe_lineage)
        starting_stat = checkpoint.stat()
        whole_digest = hashlib.sha256()
        chunks: list[ChunkRecord] = []
        offset = 0
        with checkpoint.open("rb") as source:
            while payload := source.read(self.chunk_size_bytes):
                digest = _sha256_bytes(payload)
                whole_digest.update(payload)
                record = ChunkRecord(
                    index=len(chunks),
                    offset=offset,
                    size_bytes=len(payload),
                    sha256=digest,
                )
                self._store_chunk(record, payload)
                chunks.append(record)
                offset += len(payload)
        ending_stat = checkpoint.stat()
        if (
            starting_stat.st_size != ending_stat.st_size
            or starting_stat.st_mtime_ns != ending_stat.st_mtime_ns
        ):
            raise DurabilityError(
                f"Checkpoint changed while it was being registered: {checkpoint}"
            )
        checkpoint_sha256 = whole_digest.hexdigest()
        global_step = int(
            dict(safe_lineage.get("progress", {})).get("global_step", 0) or 0
        )
        prefix = "step" if artifact_class is ArtifactClass.FULL_RESUME else "fp16"
        snapshot_id = f"{prefix}-{global_step:012d}-{checkpoint_sha256[:16]}"
        snapshot_dir = self.snapshot_dir(snapshot_id)
        manifest_path = snapshot_dir / "manifest.json"
        if manifest_path.exists():
            manifest = self.load_manifest(snapshot_id)
            if (
                str(manifest.get("artifact_class")) != artifact_class.value
                or dict(manifest.get("source", {})).get("sha256") != checkpoint_sha256
                or manifest.get("lineage") != safe_lineage
            ):
                raise DurabilityCorruptionError(
                    f"Snapshot ID collision or immutable manifest mismatch: {snapshot_id}"
                )
            self._verify_local_chunks(manifest)
            return self._snapshot_ref(manifest, manifest_path)

        manifest: dict[str, object] = {
            "schema_version": DURABILITY_SCHEMA_VERSION,
            "snapshot_id": snapshot_id,
            "created_at": _utc_timestamp(),
            "artifact_class": artifact_class.value,
            "resume_eligible": artifact_class is ArtifactClass.FULL_RESUME,
            "initial_state": DurabilityState.LOCAL_SAVED.value,
            "source": {
                "filename": checkpoint.name,
                "size_bytes": ending_stat.st_size,
                "sha256": checkpoint_sha256,
            },
            "chunk_size_bytes": self.chunk_size_bytes,
            "chunks": [record.to_dict() for record in chunks],
            "lineage": safe_lineage,
        }
        manifest_bytes = _canonical_json(manifest) + b"\n"
        status = {
            "schema_version": DURABILITY_SCHEMA_VERSION,
            "snapshot_id": snapshot_id,
            "state": DurabilityState.LOCAL_SAVED.value,
            "updated_at": _utc_timestamp(),
            "replicas": {},
            "errors": [],
        }
        snapshots_root = self.root / "snapshots"
        snapshots_root.mkdir(parents=True, exist_ok=True)
        staging = snapshots_root / (
            f".{snapshot_id}.{os.getpid()}.{threading.get_ident()}.{time.time_ns()}.tmp"
        )
        staging.mkdir()
        try:
            _atomic_write_bytes(staging / "manifest.json", manifest_bytes)
            _atomic_write_json(staging / "status.json", status)
            try:
                os.replace(staging, snapshot_dir)
            except OSError as exc:
                if not manifest_path.exists():
                    raise
                existing = self.load_manifest(snapshot_id)
                if existing != manifest:
                    raise DurabilityCorruptionError(
                        f"Concurrent immutable manifest mismatch: {snapshot_id}"
                    ) from exc
        finally:
            if staging.exists():
                (staging / "manifest.json").unlink(missing_ok=True)
                (staging / "status.json").unlink(missing_ok=True)
                staging.rmdir()
        ref = SnapshotRef(
            outbox_root=self.root,
            snapshot_id=snapshot_id,
            manifest_path=manifest_path,
            manifest_sha256=_sha256_bytes(manifest_bytes),
            artifact_class=artifact_class,
            global_step=global_step,
        )
        _emit_evidence(
            "checkpoint.local_saved",
            {
                "snapshot_id": snapshot_id,
                "state": DurabilityState.LOCAL_SAVED.value,
                "artifact_class": artifact_class.value,
                "global_step": global_step,
                "size_bytes": ending_stat.st_size,
            },
            artifact_refs=[
                {
                    "kind": "checkpoint",
                    "sha256": checkpoint_sha256,
                    "manifest_sha256": ref.manifest_sha256,
                }
            ],
        )
        return ref

    def _store_chunk(self, record: ChunkRecord, payload: bytes) -> None:
        target = self.chunk_path(record.sha256)
        if target.exists():
            if target.stat().st_size != record.size_bytes or sha256_file(target) != record.sha256:
                raise DurabilityCorruptionError(
                    f"Local content-addressed chunk is corrupt: {target}"
                )
            return
        _atomic_write_bytes(target, payload)
        if target.stat().st_size != record.size_bytes or sha256_file(target) != record.sha256:
            raise DurabilityCorruptionError(f"Could not verify local chunk: {target}")

    def load_manifest(self, snapshot_id: str) -> dict[str, Any]:
        manifest = _read_json(self.snapshot_dir(snapshot_id) / "manifest.json")
        if manifest.get("snapshot_id") != snapshot_id:
            raise DurabilityCorruptionError(
                f"Snapshot directory and manifest ID disagree: {snapshot_id}"
            )
        if int(manifest.get("schema_version", 0)) != DURABILITY_SCHEMA_VERSION:
            raise DurabilityError(f"Unsupported durability manifest: {snapshot_id}")
        artifact_class = ArtifactClass(str(manifest.get("artifact_class", "")))
        lineage = dict(manifest.get("lineage", {}))
        if artifact_class is ArtifactClass.FULL_RESUME:
            _validate_resume_lineage(lineage)
        source = dict(manifest.get("source", {}))
        source_size = int(source.get("size_bytes", -1))
        _validate_sha256(str(source.get("sha256", "")), label="checkpoint")
        expected_offset = 0
        for expected_index, raw in enumerate(manifest.get("chunks", [])):
            record = ChunkRecord.from_dict(dict(raw))
            if record.index != expected_index or record.offset != expected_offset:
                raise DurabilityCorruptionError(
                    f"Non-contiguous chunk table in snapshot {snapshot_id}"
                )
            expected_offset += record.size_bytes
        if source_size <= 0 or expected_offset != source_size:
            raise DurabilityCorruptionError(
                f"Chunk table size does not match snapshot {snapshot_id}"
            )
        return manifest

    def load_ref(self, snapshot_id: str) -> SnapshotRef:
        path = self.snapshot_dir(snapshot_id) / "manifest.json"
        return self._snapshot_ref(self.load_manifest(snapshot_id), path)

    def _snapshot_ref(
        self,
        manifest: Mapping[str, object],
        manifest_path: Path,
    ) -> SnapshotRef:
        manifest_bytes = manifest_path.read_bytes()
        lineage = dict(manifest.get("lineage", {}))
        progress = dict(lineage.get("progress", {}))
        return SnapshotRef(
            outbox_root=self.root,
            snapshot_id=str(manifest["snapshot_id"]),
            manifest_path=manifest_path,
            manifest_sha256=_sha256_bytes(manifest_bytes),
            artifact_class=ArtifactClass(str(manifest["artifact_class"])),
            global_step=int(progress.get("global_step", 0) or 0),
        )

    def _verify_local_chunks(self, manifest: Mapping[str, object]) -> None:
        for raw in manifest.get("chunks", []):
            record = ChunkRecord.from_dict(dict(raw))
            path = self.chunk_path(record.sha256)
            if not path.is_file() or path.stat().st_size != record.size_bytes:
                raise DurabilityCorruptionError(f"Missing local chunk: {path}")
            if sha256_file(path) != record.sha256:
                raise DurabilityCorruptionError(f"Corrupt local chunk: {path}")

    def materialize(
        self,
        snapshot_id: str,
        output_path: Path,
        *,
        for_resume: bool = False,
    ) -> Path:
        manifest = self.load_manifest(snapshot_id)
        artifact_class = ArtifactClass(str(manifest["artifact_class"]))
        if for_resume and artifact_class is not ArtifactClass.FULL_RESUME:
            raise ResumeArtifactError(
                f"Snapshot {snapshot_id} is {artifact_class.value}; it cannot resume training."
            )
        output_path = Path(output_path)
        temporary = output_path.with_suffix(output_path.suffix + ".materialize.tmp")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        try:
            with temporary.open("wb") as target:
                for raw in manifest.get("chunks", []):
                    record = ChunkRecord.from_dict(dict(raw))
                    chunk_path = self.chunk_path(record.sha256)
                    if (
                        not chunk_path.is_file()
                        or chunk_path.stat().st_size != record.size_bytes
                        or sha256_file(chunk_path) != record.sha256
                    ):
                        raise DurabilityCorruptionError(
                            f"Cannot materialize corrupt chunk: {chunk_path}"
                        )
                    with chunk_path.open("rb") as source:
                        while block := source.read(DEFAULT_COPY_BLOCK_BYTES):
                            target.write(block)
                            digest.update(block)
                target.flush()
                os.fsync(target.fileno())
            source_meta = dict(manifest["source"])
            if temporary.stat().st_size != int(source_meta["size_bytes"]):
                raise DurabilityCorruptionError("Materialized checkpoint has the wrong size")
            if digest.hexdigest() != str(source_meta["sha256"]):
                raise DurabilityCorruptionError("Materialized checkpoint has the wrong digest")
            os.replace(temporary, output_path)
        finally:
            temporary.unlink(missing_ok=True)
        return output_path

    def snapshots(self) -> list[SnapshotRef]:
        snapshots_root = self.root / "snapshots"
        if not snapshots_root.exists():
            return []
        refs: list[SnapshotRef] = []
        for manifest in sorted(snapshots_root.glob("*/manifest.json")):
            refs.append(self.load_ref(manifest.parent.name))
        return refs

    def prune(self, snapshot_ids: Iterable[str]) -> tuple[str, ...]:
        """Delete verified obsolete local snapshots and unreferenced CAS chunks.

        The caller supplies a retention decision.  This method still fails
        closed unless every target has reached a remote verified state.
        """

        targets = tuple(
            sorted({_validate_identifier(value, label="snapshot id") for value in snapshot_ids})
        )
        if not targets:
            return ()
        for snapshot_id in targets:
            status = _read_json(self.status_path(snapshot_id))
            state = DurabilityState(str(status.get("state", "")))
            if STATE_ORDER[state] < STATE_ORDER[DurabilityState.CANONICAL_VERIFIED]:
                raise PublicationError(
                    f"Refusing to prune unverified snapshot {snapshot_id}: {state.value}"
                )
        for snapshot_id in targets:
            directory = self.snapshot_dir(snapshot_id).resolve()
            snapshots_root = (self.root / "snapshots").resolve()
            if snapshots_root not in directory.parents:
                raise DurabilityError(f"Snapshot prune escaped the outbox: {directory}")
            shutil.rmtree(directory)

        referenced: set[str] = set()
        for ref in self.snapshots():
            manifest = self.load_manifest(ref.snapshot_id)
            referenced.update(
                ChunkRecord.from_dict(dict(raw)).sha256
                for raw in manifest.get("chunks", [])
            )
        chunks_root = self.root / "chunks"
        if chunks_root.exists():
            for chunk in chunks_root.rglob("*.chunk"):
                if chunk.stem not in referenced:
                    chunk.unlink()
            for directory in sorted(
                (path for path in chunks_root.rglob("*") if path.is_dir()),
                key=lambda path: len(path.parts),
                reverse=True,
            ):
                with contextlib.suppress(OSError):
                    directory.rmdir()
        _emit_evidence(
            "checkpoint.retention_pruned",
            {"snapshot_ids": list(targets), "remaining": len(self.snapshots())},
        )
        return targets


class ReplicaBackend(Protocol):
    name: str
    kind: str
    canonical: bool

    def stage_chunk(self, local_chunk: Path, record: ChunkRecord) -> None: ...

    def publish_manifest(self, ref: SnapshotRef, manifest_bytes: bytes) -> Path: ...

    def verify_snapshot(self, manifest: Mapping[str, object]) -> None: ...

    def publish_pointer(self, pointer: Mapping[str, object]) -> Path: ...

    def publish_receipt(self, snapshot_id: str, receipt: Mapping[str, object]) -> Path: ...

    def prune_snapshots(self, snapshot_ids: Iterable[str]) -> tuple[str, ...]: ...


class FilesystemReplica:
    """Resumable replica for a local disk, sync folder, or mounted Drive."""

    def __init__(
        self,
        name: str,
        root: Path,
        *,
        kind: str = "filesystem",
        canonical: bool = False,
    ) -> None:
        self.name = _validate_identifier(name.strip(), label="replica name")
        self.root = Path(root)
        self.kind = kind
        self.canonical = bool(canonical)

    def chunk_path(self, sha256: str) -> Path:
        return self.root / "chunks" / sha256[:2] / f"{sha256}.chunk"

    def partial_chunk_path(self, sha256: str) -> Path:
        return self.chunk_path(sha256).with_suffix(".chunk.part")

    def stage_chunk(self, local_chunk: Path, record: ChunkRecord) -> None:
        target = self.chunk_path(record.sha256)
        if target.exists():
            self._verify_chunk(target, record)
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = self.partial_chunk_path(record.sha256)
        resume_offset = self._verified_prefix(local_chunk, partial, record.size_bytes)
        mode = "ab" if resume_offset else "wb"
        with local_chunk.open("rb") as source, partial.open(mode) as destination:
            source.seek(resume_offset)
            while block := source.read(DEFAULT_COPY_BLOCK_BYTES):
                destination.write(block)
            destination.flush()
            os.fsync(destination.fileno())
        self._verify_chunk(partial, record)
        os.replace(partial, target)
        self._verify_chunk(target, record)

    @staticmethod
    def _verified_prefix(source: Path, partial: Path, expected_size: int) -> int:
        if not partial.exists():
            return 0
        partial_size = partial.stat().st_size
        if partial_size > expected_size:
            with partial.open("r+b") as handle:
                handle.truncate(0)
            return 0
        matched = 0
        with source.open("rb") as original, partial.open("rb") as candidate:
            while matched < partial_size:
                wanted = min(DEFAULT_COPY_BLOCK_BYTES, partial_size - matched)
                left = original.read(wanted)
                right = candidate.read(wanted)
                if left != right:
                    break
                matched += len(left)
        if matched != partial_size:
            with partial.open("r+b") as handle:
                handle.truncate(matched)
        return matched

    @staticmethod
    def _verify_chunk(path: Path, record: ChunkRecord) -> None:
        if path.stat().st_size != record.size_bytes:
            raise DurabilityCorruptionError(
                f"Replica chunk has wrong size: {path}"
            )
        if sha256_file(path) != record.sha256:
            raise DurabilityCorruptionError(
                f"Replica chunk has wrong digest: {path}"
            )

    def publish_manifest(self, ref: SnapshotRef, manifest_bytes: bytes) -> Path:
        target = self.root / "manifests" / f"{ref.snapshot_id}.json"
        _atomic_write_bytes(target, manifest_bytes)
        if _sha256_bytes(target.read_bytes()) != ref.manifest_sha256:
            raise DurabilityCorruptionError(f"Replica manifest verification failed: {target}")
        return target

    def verify_snapshot(self, manifest: Mapping[str, object]) -> None:
        for raw in manifest.get("chunks", []):
            record = ChunkRecord.from_dict(dict(raw))
            target = self.chunk_path(record.sha256)
            if not target.is_file():
                raise DurabilityCorruptionError(f"Replica is missing chunk: {target}")
            self._verify_chunk(target, record)

    def publish_pointer(self, pointer: Mapping[str, object]) -> Path:
        artifact_class = ArtifactClass(str(pointer.get("artifact_class", "")))
        target = self.root / f"canonical-{artifact_class.value.replace('_', '-')}.json"
        if target.exists():
            current = _read_json(target)
            current_step = int(current.get("global_step", 0) or 0)
            next_step = int(pointer.get("global_step", 0) or 0)
            if current_step > next_step:
                raise PublicationError(
                    f"Refusing to rewind {self.name} from step {current_step} to {next_step}"
                )
            if current_step == next_step and current.get("checkpoint_sha256") not in {
                None,
                pointer.get("checkpoint_sha256"),
            }:
                raise PublicationError(
                    f"Conflicting checkpoint at step {next_step} on replica {self.name}"
                )
        _atomic_write_json(target, dict(pointer))
        if _read_json(target).get("manifest_sha256") != pointer.get("manifest_sha256"):
            raise DurabilityCorruptionError(f"Canonical pointer verification failed: {target}")
        if artifact_class is ArtifactClass.FULL_RESUME:
            compatibility = self.root / "canonical.json"
            _atomic_write_json(compatibility, dict(pointer))
            if _read_json(compatibility).get("manifest_sha256") != pointer.get(
                "manifest_sha256"
            ):
                raise DurabilityCorruptionError(
                    f"Canonical compatibility pointer verification failed: {compatibility}"
                )
        return target

    def publish_receipt(
        self,
        snapshot_id: str,
        receipt: Mapping[str, object],
    ) -> Path:
        target = self.root / "receipts" / snapshot_id / f"{self.name}.json"
        _atomic_write_json(target, dict(receipt))
        return target

    def prune_snapshots(self, snapshot_ids: Iterable[str]) -> tuple[str, ...]:
        """Prune only named obsolete manifests, then unreferenced CAS chunks."""

        targets = tuple(
            sorted({_validate_identifier(value, label="snapshot id") for value in snapshot_ids})
        )
        manifests_root = self.root / "manifests"
        receipts_root = self.root / "receipts"
        for snapshot_id in targets:
            (manifests_root / f"{snapshot_id}.json").unlink(missing_ok=True)
            receipt_dir = (receipts_root / snapshot_id).resolve()
            if receipts_root.resolve() in receipt_dir.parents and receipt_dir.exists():
                shutil.rmtree(receipt_dir)

        referenced: set[str] = set()
        if manifests_root.exists():
            for path in manifests_root.glob("*.json"):
                manifest = _read_json(path)
                referenced.update(
                    ChunkRecord.from_dict(dict(raw)).sha256
                    for raw in manifest.get("chunks", [])
                )
        chunks_root = self.root / "chunks"
        if chunks_root.exists():
            for chunk in chunks_root.rglob("*.chunk"):
                if chunk.stem not in referenced:
                    chunk.unlink()
            for directory in sorted(
                (path for path in chunks_root.rglob("*") if path.is_dir()),
                key=lambda path: len(path.parts),
                reverse=True,
            ):
                with contextlib.suppress(OSError):
                    directory.rmdir()
        return targets


MountedDriveReplica = FilesystemReplica


@dataclass(frozen=True)
class PublicationResult:
    snapshot_id: str
    state: DurabilityState
    verified_replicas: tuple[str, ...]
    errors: tuple[str, ...]


_STATUS_LOCK = threading.RLock()


class SnapshotPublisher:
    """Publishes snapshots in a background thread while training continues."""

    def __init__(
        self,
        outbox: CheckpointOutbox,
        replicas: Sequence[ReplicaBackend],
        *,
        min_protected_replicas: int = 1,
        max_copy_streams: int = DEFAULT_MAX_COPY_STREAMS,
    ) -> None:
        if not replicas:
            raise ValueError("at least one replica is required")
        if not any(replica.canonical for replica in replicas):
            raise ValueError("one replica must be marked canonical")
        if min_protected_replicas < 1 or min_protected_replicas > len(replicas):
            raise ValueError("min_protected_replicas is outside the replica count")
        self.outbox = outbox
        self.replicas = tuple(replicas)
        self.min_protected_replicas = int(min_protected_replicas)
        self.max_copy_streams = max(1, int(max_copy_streams))
        self._queue: queue.Queue[SnapshotRef | None] = queue.Queue()
        self._results: dict[str, PublicationResult] = {}
        self._submitted: set[str] = set()
        self._condition = threading.Condition()
        self._closed = False
        self._worker = threading.Thread(
            target=self._run,
            name="anra-checkpoint-publisher",
            daemon=True,
        )
        self._worker.start()

    def submit(self, ref: SnapshotRef) -> None:
        with self._condition:
            if self._closed:
                raise RuntimeError("checkpoint publisher is closed")
            if ref.snapshot_id in self._submitted:
                return
            self._results.pop(ref.snapshot_id, None)
            self._submitted.add(ref.snapshot_id)
            self._queue.put(ref)

    def prune_snapshots(self, snapshot_ids: Iterable[str]) -> tuple[str, ...]:
        targets = tuple(sorted(set(snapshot_ids)))
        if not targets:
            return ()
        for replica in self.replicas:
            replica.prune_snapshots(targets)
        return targets

    def _run(self) -> None:
        while True:
            ref = self._queue.get()
            try:
                if ref is None:
                    return
                try:
                    result = self.publish_snapshot(ref)
                except Exception as exc:
                    message = f"publisher: {type(exc).__name__}: {exc}"
                    self._record_error(ref.snapshot_id, message)
                    result = PublicationResult(
                        snapshot_id=ref.snapshot_id,
                        state=DurabilityState.LOCAL_SAVED,
                        verified_replicas=(),
                        errors=(message,),
                    )
                with self._condition:
                    self._results[ref.snapshot_id] = result
                    self._submitted.discard(ref.snapshot_id)
                    self._condition.notify_all()
            finally:
                self._queue.task_done()

    def publish_snapshot(self, ref: SnapshotRef) -> PublicationResult:
        manifest = self.outbox.load_manifest(ref.snapshot_id)
        manifest_bytes = ref.manifest_path.read_bytes()
        chunks = [ChunkRecord.from_dict(dict(raw)) for raw in manifest.get("chunks", [])]
        verified: list[str] = []
        errors: list[str] = []
        canonical_verified = False

        ordered = sorted(self.replicas, key=lambda replica: not replica.canonical)
        for replica in ordered:
            try:
                with ThreadPoolExecutor(max_workers=self.max_copy_streams) as executor:
                    futures = [
                        executor.submit(
                            replica.stage_chunk,
                            self.outbox.chunk_path(record.sha256),
                            record,
                        )
                        for record in chunks
                    ]
                    for future in futures:
                        future.result()
                if replica.canonical:
                    self._advance_status(
                        ref.snapshot_id,
                        DurabilityState.STAGED,
                        replica=replica,
                    )
                replica.publish_manifest(ref, manifest_bytes)
                replica.verify_snapshot(manifest)
                receipt = self._build_receipt(ref, replica, manifest)
                replica.publish_receipt(ref.snapshot_id, receipt)
                self._write_local_receipt(ref.snapshot_id, replica.name, receipt)
                verified.append(replica.name)
                if replica.canonical:
                    pointer = self._build_pointer(ref, manifest)
                    replica.publish_pointer(pointer)
                    canonical_verified = True
                    self._advance_status(
                        ref.snapshot_id,
                        DurabilityState.CANONICAL_VERIFIED,
                        replica=replica,
                    )
            except Exception as exc:
                message = f"{replica.name}: {type(exc).__name__}: {exc}"
                errors.append(message)
                self._record_error(ref.snapshot_id, message)

        state = DurabilityState.LOCAL_SAVED
        if canonical_verified:
            state = DurabilityState.CANONICAL_VERIFIED
        if canonical_verified and len(verified) >= self.min_protected_replicas:
            state = DurabilityState.PROTECTED
            self._advance_status(ref.snapshot_id, state)
        return PublicationResult(
            snapshot_id=ref.snapshot_id,
            state=state,
            verified_replicas=tuple(verified),
            errors=tuple(errors),
        )

    @staticmethod
    def _build_pointer(
        ref: SnapshotRef,
        manifest: Mapping[str, object],
    ) -> dict[str, object]:
        source = dict(manifest["source"])
        lineage = dict(manifest.get("lineage", {}))
        return {
            "schema_version": DURABILITY_SCHEMA_VERSION,
            "snapshot_id": ref.snapshot_id,
            "manifest_sha256": ref.manifest_sha256,
            "artifact_class": ref.artifact_class.value,
            "checkpoint_sha256": str(source["sha256"]),
            "size_bytes": int(source["size_bytes"]),
            "global_step": ref.global_step,
            "architecture_sha256": dict(lineage.get("architecture", {})).get("sha256"),
            "published_at": _utc_timestamp(),
        }

    @staticmethod
    def _build_receipt(
        ref: SnapshotRef,
        replica: ReplicaBackend,
        manifest: Mapping[str, object],
    ) -> dict[str, object]:
        source = dict(manifest["source"])
        return {
            "schema_version": DURABILITY_SCHEMA_VERSION,
            "snapshot_id": ref.snapshot_id,
            "replica": replica.name,
            "replica_kind": replica.kind,
            "canonical": replica.canonical,
            "manifest_sha256": ref.manifest_sha256,
            "checkpoint_sha256": source["sha256"],
            "size_bytes": source["size_bytes"],
            "verified_at": _utc_timestamp(),
        }

    def _write_local_receipt(
        self,
        snapshot_id: str,
        replica_name: str,
        receipt: Mapping[str, object],
    ) -> None:
        target = (
            self.outbox.snapshot_dir(snapshot_id)
            / "receipts"
            / f"{replica_name}.json"
        )
        _atomic_write_json(target, dict(receipt))

    def _advance_status(
        self,
        snapshot_id: str,
        next_state: DurabilityState,
        *,
        replica: ReplicaBackend | None = None,
    ) -> None:
        with _STATUS_LOCK:
            path = self.outbox.status_path(snapshot_id)
            status = _read_json(path)
            current = DurabilityState(str(status["state"]))
            advanced = STATE_ORDER[next_state] > STATE_ORDER[current]
            if advanced:
                status["state"] = next_state.value
            if replica is not None:
                replicas = dict(status.get("replicas", {}))
                replicas[replica.name] = {
                    "kind": replica.kind,
                    "canonical": replica.canonical,
                    "state": next_state.value,
                    "updated_at": _utc_timestamp(),
                }
                status["replicas"] = replicas
            status["updated_at"] = _utc_timestamp()
            _atomic_write_json(path, status)
        if advanced:
            _emit_evidence(
                f"checkpoint.{next_state.value}",
                {
                    "snapshot_id": snapshot_id,
                    "previous_state": current.value,
                    "state": next_state.value,
                    "replica": replica.name if replica is not None else None,
                },
            )

    def _record_error(self, snapshot_id: str, message: str) -> None:
        with _STATUS_LOCK:
            path = self.outbox.status_path(snapshot_id)
            status = _read_json(path)
            errors = list(status.get("errors", []))
            errors.append({"at": _utc_timestamp(), "message": message})
            status["errors"] = errors[-50:]
            status["updated_at"] = _utc_timestamp()
            _atomic_write_json(path, status)
        _emit_evidence(
            "checkpoint.publication_failed",
            {"snapshot_id": snapshot_id, "message": message},
        )

    def wait_for(
        self,
        ref: SnapshotRef,
        target: DurabilityState = DurabilityState.CANONICAL_VERIFIED,
        *,
        timeout_seconds: float = 1800.0,
    ) -> PublicationResult:
        deadline = time.monotonic() + timeout_seconds
        with self._condition:
            while True:
                status = _read_json(self.outbox.status_path(ref.snapshot_id))
                state = DurabilityState(str(status["state"]))
                if STATE_ORDER[state] >= STATE_ORDER[target]:
                    result = self._results.get(ref.snapshot_id)
                    return result or PublicationResult(
                        snapshot_id=ref.snapshot_id,
                        state=state,
                        verified_replicas=tuple(dict(status.get("replicas", {}))),
                        errors=(),
                    )
                result = self._results.get(ref.snapshot_id)
                if result is not None:
                    raise PublicationError(
                        f"Snapshot {ref.snapshot_id} reached {result.state.value}, not "
                        f"{target.value}: {'; '.join(result.errors) or 'no verified primary'}"
                    )
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"Timed out waiting for {ref.snapshot_id} to reach {target.value}"
                    )
                self._condition.wait(timeout=min(0.25, remaining))

    def drain(self, *, timeout_seconds: float | None = None) -> None:
        deadline = None if timeout_seconds is None else time.monotonic() + timeout_seconds
        while self._queue.unfinished_tasks:
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("Timed out draining checkpoint publisher")
            time.sleep(0.05)

    def close(self, *, wait: bool = True, timeout_seconds: float | None = None) -> None:
        if wait:
            self.drain(timeout_seconds=timeout_seconds)
        with self._condition:
            if self._closed:
                return
            self._closed = True
            self._queue.put(None)
        if wait:
            self._worker.join(timeout=timeout_seconds)


def _parse_replica_environment() -> list[FilesystemReplica]:
    raw = os.environ.get("ANRA_DURABILITY_REPLICAS", "").strip()
    if not raw:
        shared_root = os.environ.get("ANRA_SHARED_CHECKPOINT_DIR", "").strip()
        if shared_root:
            return [
                FilesystemReplica(
                    "shared-drive",
                    Path(shared_root) / "durability-v1",
                    kind="mounted_drive",
                    canonical=True,
                )
            ]
        return []
    replicas: list[FilesystemReplica] = []
    if raw.startswith("["):
        entries = json.loads(raw)
        if not isinstance(entries, list):
            raise ValueError("ANRA_DURABILITY_REPLICAS must contain a JSON list")
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                raise ValueError("Each durability replica must be a JSON object")
            replicas.append(
                FilesystemReplica(
                    str(entry.get("name", f"replica-{index}")),
                    Path(str(entry["path"])),
                    kind=str(entry.get("kind", "filesystem")),
                    canonical=bool(entry.get("canonical", index == 0)),
                )
            )
    else:
        for index, entry in enumerate(item for item in raw.split(";") if item.strip()):
            if "=" in entry:
                name, path = entry.split("=", 1)
            else:
                name, path = f"replica-{index}", entry
            replicas.append(
                FilesystemReplica(
                    name.strip(),
                    Path(path.strip()),
                    canonical=index == 0,
                )
            )
    canonical_count = sum(replica.canonical for replica in replicas)
    if canonical_count != 1:
        raise ValueError("Exactly one durability replica must be canonical")
    return replicas


class CheckpointDurabilitySession:
    """Training-facing facade for registration and background publication."""

    def __init__(
        self,
        outbox: CheckpointOutbox,
        publisher: SnapshotPublisher | None,
        *,
        enabled: bool,
        required: bool,
        scratch_run: bool,
        ack_timeout_seconds: float,
        hot_storage_limit_bytes: int = HOT_STORAGE_LIMIT_BYTES,
    ) -> None:
        self.outbox = outbox
        self.publisher = publisher
        self.enabled = enabled
        self.required = required
        self.scratch_run = scratch_run
        self.ack_timeout_seconds = ack_timeout_seconds
        self.hot_storage_limit_bytes = int(hot_storage_limit_bytes)
        if self.hot_storage_limit_bytes <= 0:
            raise ValueError("hot_storage_limit_bytes must be positive")
        self.initial_acknowledged = False
        self.latest: SnapshotRef | None = None

    @classmethod
    def from_environment(
        cls,
        default_outbox: Path,
        *,
        scratch_run: bool,
    ) -> CheckpointDurabilitySession:
        required = durability_required_from_environment()
        enabled = required or _truthy_environment("ANRA_ENABLE_DURABILITY_OUTBOX")
        if _truthy_environment("ANRA_REQUIRE_SIGNED_EVIDENCE") and not os.environ.get(
            "ANRA_EVIDENCE_SIGNING_KEY", ""
        ):
            raise PublicationError(
                "ANRA_REQUIRE_SIGNED_EVIDENCE=1 requires ANRA_EVIDENCE_SIGNING_KEY"
            )
        root = Path(os.environ.get("ANRA_DURABILITY_OUTBOX", str(default_outbox)))
        outbox = CheckpointOutbox(root)
        replicas = _parse_replica_environment() if enabled else []
        if required and not replicas:
            raise PublicationError(
                "Durable remote ACK is required, but ANRA_DURABILITY_REPLICAS (or "
                "ANRA_SHARED_CHECKPOINT_DIR) does not define a replica."
            )
        min_protected = int(
            os.environ.get(
                "ANRA_DURABILITY_MIN_PROTECTED_REPLICAS",
                str(min(2, len(replicas))) if replicas else "1",
            )
        )
        publisher = (
            SnapshotPublisher(
                outbox,
                replicas,
                min_protected_replicas=min_protected,
                max_copy_streams=int(
                    os.environ.get(
                        "ANRA_DURABILITY_COPY_STREAMS",
                        str(DEFAULT_MAX_COPY_STREAMS),
                    )
                ),
            )
            if replicas
            else None
        )
        return cls(
            outbox,
            publisher,
            enabled=enabled,
            required=required,
            scratch_run=scratch_run,
            ack_timeout_seconds=float(
                os.environ.get("ANRA_DURABILITY_ACK_TIMEOUT_SECONDS", "1800")
            ),
            hot_storage_limit_bytes=int(
                os.environ.get(
                    "ANRA_DURABILITY_HOT_LIMIT_BYTES",
                    str(HOT_STORAGE_LIMIT_BYTES),
                )
            ),
        )

    @property
    def requires_initial_boundary(self) -> bool:
        return self.required and self.scratch_run and not self.initial_acknowledged

    def publish_checkpoint(
        self,
        checkpoint: Path,
        payload: Mapping[str, object],
        *,
        final: bool = False,
    ) -> SnapshotRef | None:
        if not self.enabled:
            return None
        lineage = payload.get("checkpoint_lineage")
        if not isinstance(lineage, Mapping):
            lineage = build_checkpoint_lineage(payload)
        lineage_id = str(lineage.get("lineage_id", "")).strip()
        if not lineage_id:
            raise DurabilityError("A durable checkpoint requires a stable lineage_id")
        # Exactly one upload may be in flight.  If Drive is slower than the
        # trainer, the next checkpoint boundary applies backpressure rather
        # than filling ephemeral disk with an unbounded queue.
        if self.publisher is not None and self.latest is not None:
            self.publisher.wait_for(
                self.latest,
                DurabilityState.PROTECTED,
                timeout_seconds=self.ack_timeout_seconds,
            )
        plan = plan_hot_retention(
            self.outbox,
            in_flight_bytes=Path(checkpoint).stat().st_size,
            hot_limit_bytes=self.hot_storage_limit_bytes,
            lineage_id=lineage_id,
        )
        if not plan.fits:
            raise PublicationError(
                "Checkpoint hot-storage contract cannot fit two resume states, "
                "two compact states and one in-flight artifact: "
                f"deficit={plan.deficit_bytes} bytes"
            )
        if plan.delete_snapshot_ids:
            if self.publisher is not None:
                self.publisher.prune_snapshots(plan.delete_snapshot_ids)
            self.outbox.prune(plan.delete_snapshot_ids)
        ref = self.outbox.register_checkpoint(
            checkpoint,
            artifact_class=ArtifactClass.FULL_RESUME,
            lineage=lineage,
        )
        self.latest = ref
        if self.publisher is not None:
            self.publisher.submit(ref)
        wait_for_ack = self.required and (self.requires_initial_boundary or final)
        if wait_for_ack:
            if self.publisher is None:
                raise PublicationError("A required durability ACK has no publisher")
            target = (
                DurabilityState.PROTECTED
                if final
                else DurabilityState.CANONICAL_VERIFIED
            )
            self.publisher.wait_for(
                ref,
                target,
                timeout_seconds=self.ack_timeout_seconds,
            )
            self.initial_acknowledged = True
        if final and self.publisher is not None:
            self.publisher.wait_for(
                ref,
                DurabilityState.PROTECTED,
                timeout_seconds=self.ack_timeout_seconds,
            )
            final_plan = plan_hot_retention(
                self.outbox,
                hot_limit_bytes=self.hot_storage_limit_bytes,
                lineage_id=lineage_id,
            )
            if final_plan.delete_snapshot_ids:
                self.publisher.prune_snapshots(final_plan.delete_snapshot_ids)
                self.outbox.prune(final_plan.delete_snapshot_ids)
        return ref

    def close(self) -> None:
        if self.publisher is None:
            return
        if self.required and self.latest is not None:
            self.publisher.wait_for(
                self.latest,
                DurabilityState.PROTECTED,
                timeout_seconds=self.ack_timeout_seconds,
            )
        self.publisher.close(wait=True, timeout_seconds=self.ack_timeout_seconds)


@dataclass(frozen=True)
class RetentionPlan:
    keep_snapshot_ids: tuple[str, ...]
    delete_snapshot_ids: tuple[str, ...]
    retained_logical_bytes: int
    in_flight_bytes: int
    hot_limit_bytes: int
    fits: bool
    deficit_bytes: int

    def to_dict(self) -> dict[str, object]:
        return {
            "keep_snapshot_ids": list(self.keep_snapshot_ids),
            "delete_snapshot_ids": list(self.delete_snapshot_ids),
            "retained_logical_bytes": self.retained_logical_bytes,
            "in_flight_bytes": self.in_flight_bytes,
            "hot_limit_bytes": self.hot_limit_bytes,
            "fits": self.fits,
            "deficit_bytes": self.deficit_bytes,
        }


def snapshot_lineage_id(outbox: CheckpointOutbox, ref: SnapshotRef) -> str:
    lineage = dict(outbox.load_manifest(ref.snapshot_id).get("lineage", {}))
    explicit = str(lineage.get("lineage_id", "")).strip()
    if explicit:
        return explicit
    # Schema-v1 outboxes created before lineage_id was mandatory remain
    # groupable for an explicit migration/retention pass.
    architecture = dict(lineage.get("architecture", {}))
    training = dict(lineage.get("training", {}))
    seed = dict(training.get("seed_contract", {})).get("seed", "unknown")
    return f"legacy/{architecture.get('sha256', 'unknown')}/seed-{seed}"


def plan_hot_retention(
    outbox: CheckpointOutbox,
    *,
    keep_full: int = DEFAULT_KEEP_FULL,
    keep_compact: int = DEFAULT_KEEP_COMPACT,
    in_flight_bytes: int = 0,
    hot_limit_bytes: int = HOT_STORAGE_LIMIT_BYTES,
    lineage_id: str | None = None,
) -> RetentionPlan:
    all_refs = outbox.snapshots()
    refs = (
        [ref for ref in all_refs if snapshot_lineage_id(outbox, ref) == lineage_id]
        if lineage_id is not None
        else all_refs
    )
    by_class: dict[ArtifactClass, list[SnapshotRef]] = {
        ArtifactClass.FULL_RESUME: [],
        ArtifactClass.FP16_INFERENCE: [],
    }
    for ref in refs:
        by_class[ref.artifact_class].append(ref)
    for group in by_class.values():
        group.sort(key=lambda ref: (ref.global_step, ref.snapshot_id), reverse=True)
    kept = (
        by_class[ArtifactClass.FULL_RESUME][: max(0, keep_full)]
        + by_class[ArtifactClass.FP16_INFERENCE][: max(0, keep_compact)]
    )
    keep_ids = {ref.snapshot_id for ref in kept}
    retained = 0
    for ref in kept:
        manifest = outbox.load_manifest(ref.snapshot_id)
        retained += int(dict(manifest["source"])["size_bytes"])
    required = retained + max(0, int(in_flight_bytes))
    return RetentionPlan(
        keep_snapshot_ids=tuple(sorted(keep_ids)),
        delete_snapshot_ids=tuple(
            sorted(ref.snapshot_id for ref in refs if ref.snapshot_id not in keep_ids)
        ),
        retained_logical_bytes=retained,
        in_flight_bytes=max(0, int(in_flight_bytes)),
        hot_limit_bytes=int(hot_limit_bytes),
        fits=required <= hot_limit_bytes,
        deficit_bytes=max(0, required - hot_limit_bytes),
    )


def _replicas_from_cli(
    values: Iterable[str],
    drive_values: Iterable[str],
) -> list[FilesystemReplica]:
    replicas: list[FilesystemReplica] = []
    entries = [(value, "filesystem") for value in values]
    entries += [(value, "mounted_drive") for value in drive_values]
    for index, (entry, kind) in enumerate(entries):
        if "=" not in entry:
            raise ValueError("Replica must be NAME=PATH")
        name, path = entry.split("=", 1)
        replicas.append(
            FilesystemReplica(
                name,
                Path(path),
                kind=kind,
                canonical=index == 0,
            )
        )
    return replicas


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    register = subparsers.add_parser("register", help="register an immutable checkpoint")
    register.add_argument("--outbox", required=True, type=Path)
    register.add_argument("--checkpoint", required=True, type=Path)
    register.add_argument(
        "--artifact-class",
        choices=[item.value for item in ArtifactClass],
        default=ArtifactClass.FULL_RESUME.value,
    )
    register.add_argument("--lineage-json", type=Path)

    publish = subparsers.add_parser("publish", help="publish one or all snapshots")
    publish.add_argument("--outbox", required=True, type=Path)
    publish.add_argument("--snapshot-id")
    publish.add_argument("--replica", action="append", default=[], metavar="NAME=PATH")
    publish.add_argument(
        "--drive-replica",
        action="append",
        default=[],
        metavar="NAME=MOUNTED_PATH",
    )
    publish.add_argument("--min-protected-replicas", type=int, default=1)

    materialize = subparsers.add_parser("materialize", help="reassemble a snapshot")
    materialize.add_argument("--outbox", required=True, type=Path)
    materialize.add_argument("--snapshot-id", required=True)
    materialize.add_argument("--output", required=True, type=Path)
    materialize.add_argument("--for-resume", action="store_true")

    compact = subparsers.add_parser(
        "compact",
        help="create and register an fp16 model-only artifact from full resume",
    )
    compact.add_argument("--outbox", required=True, type=Path)
    compact.add_argument("--checkpoint", required=True, type=Path)
    compact.add_argument("--output", required=True, type=Path)
    compact.add_argument("--overwrite", action="store_true")

    retention = subparsers.add_parser("retention", help="print the non-destructive hot plan")
    retention.add_argument("--outbox", required=True, type=Path)
    retention.add_argument("--in-flight-bytes", type=int, default=0)

    args = parser.parse_args(argv)
    outbox = CheckpointOutbox(args.outbox)
    if args.command == "compact":
        report = create_fp16_inference_artifact(
            args.checkpoint,
            args.output,
            overwrite=args.overwrite,
        )
        ref = outbox.register_checkpoint(
            args.output,
            artifact_class=ArtifactClass.FP16_INFERENCE,
            lineage=dict(report["lineage"]),
        )
        print(
            json.dumps(
                {
                    **{key: value for key, value in report.items() if key != "lineage"},
                    "snapshot_id": ref.snapshot_id,
                },
                indent=2,
            )
        )
        return 0
    if args.command == "register":
        if args.lineage_json:
            lineage = _read_json(args.lineage_json)
        elif args.artifact_class == ArtifactClass.FULL_RESUME.value:
            from runtime.safe_load import safe_torch_load

            blob = safe_torch_load(args.checkpoint, map_location="cpu")
            if not isinstance(blob, Mapping):
                raise DurabilityError("Full-resume checkpoint must contain a mapping payload")
            lineage = build_checkpoint_lineage(blob)
        else:
            lineage = {}
        ref = outbox.register_checkpoint(
            args.checkpoint,
            artifact_class=args.artifact_class,
            lineage=lineage,
        )
        print(json.dumps({"snapshot_id": ref.snapshot_id}, indent=2))
        return 0
    if args.command == "materialize":
        result = outbox.materialize(
            args.snapshot_id,
            args.output,
            for_resume=args.for_resume,
        )
        print(result)
        return 0
    if args.command == "retention":
        print(
            json.dumps(
                plan_hot_retention(
                    outbox,
                    in_flight_bytes=args.in_flight_bytes,
                ).to_dict(),
                indent=2,
            )
        )
        return 0
    replicas = _replicas_from_cli(args.replica, args.drive_replica)
    publisher = SnapshotPublisher(
        outbox,
        replicas,
        min_protected_replicas=args.min_protected_replicas,
    )
    refs = [outbox.load_ref(args.snapshot_id)] if args.snapshot_id else outbox.snapshots()
    try:
        for ref in refs:
            publisher.submit(ref)
        for ref in refs:
            result = publisher.wait_for(ref, DurabilityState.CANONICAL_VERIFIED)
            print(json.dumps({"snapshot_id": ref.snapshot_id, "state": result.state.value}))
    finally:
        publisher.close(wait=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
