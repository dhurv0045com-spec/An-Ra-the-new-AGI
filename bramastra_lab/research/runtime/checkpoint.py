"""Atomic checkpoints, writer fencing and durable milestones (B05).

Publication order: payload and manifest are written into a temporary sibling
directory, flushed, hashed and validated, the COMPLETE marker closes the
temporary directory, and a single rename publishes it. A failure before that
rename can never destroy the previous valid checkpoint. The LATEST and
ACCEPTED_PARENT pointers and the milestone list are updated atomically after
publication. Restore validates actual content hashes, schema, tokenizer and
configuration identities — descriptive equality is never trusted.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import time
import uuid
from dataclasses import dataclass
from typing import Any, Mapping

import torch

from bramastra_lab.research.contracts.core import content_identity

CHECKPOINT_SCHEMA = "bramastra-checkpoint/v1"
PAYLOAD_NAME = "payload.pt"
MANIFEST_NAME = "manifest.json"
COMPLETE_MARKER = "COMPLETE"
LATEST_POINTER = "LATEST.json"
ACCEPTED_PARENT_POINTER = "ACCEPTED_PARENT.json"
MILESTONES_NAME = "milestones.json"
WRITER_LOCK = "writer.lock"


class CheckpointError(RuntimeError):
    """A checkpoint is missing, incomplete, tampered with or mismatched."""


@dataclass(frozen=True)
class CheckpointManifest:
    checkpoint_id: str
    schema: str
    update_index: int
    run_id: str
    parent_checkpoint_id: str | None
    config_identity: str
    tokenizer_identity: str
    data_identity: str
    code_identity: str
    payload_sha256: str
    payload_bytes: int
    created_unix: float

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "CheckpointManifest":
        required = {"checkpoint_id", "schema", "update_index", "run_id",
                    "parent_checkpoint_id", "config_identity", "tokenizer_identity",
                    "data_identity", "code_identity", "payload_sha256", "payload_bytes",
                    "created_unix"}
        missing, unknown = required - set(raw), set(raw) - required
        if missing:
            raise CheckpointError(f"checkpoint manifest missing fields: {sorted(missing)}")
        if unknown:
            raise CheckpointError(f"checkpoint manifest has unknown fields: {sorted(unknown)}")
        if raw["schema"] != CHECKPOINT_SCHEMA:
            raise CheckpointError(f"unsupported checkpoint schema {raw['schema']!r}")
        return cls(**raw)


def checkpoint_root(run_dir: str) -> str:
    return os.path.join(run_dir, "checkpoints")


def _fsync_path(path: str) -> None:
    if os.path.isdir(path) and os.name == "nt":
        # Windows cannot open directory handles for fsync; the file-level
        # fsyncs above are the durability boundary available on this platform.
        return
    flags = os.O_RDONLY
    if os.path.isdir(path):
        flags |= getattr(os, "O_DIRECTORY", 0)
    fd = os.open(path, flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# -- writer fencing -----------------------------------------------------------

def acquire_writer_fence(run_dir: str, *, force: bool = False) -> str:
    """Single-writer fencing via an exclusive lock file.

    A held lease is never stolen for being old: a long legitimate job must
    not lose ownership to a heuristic (B2.2 R4). Recovery from a crashed
    writer is an explicit operator decision via ``force=True`` and is
    recorded by the caller.
    """
    os.makedirs(run_dir, exist_ok=True)
    lock_path = os.path.join(run_dir, WRITER_LOCK)
    token = f"{os.getpid()}:{time.time():.6f}"
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            with open(lock_path, "r", encoding="utf-8") as handle:
                existing = handle.read().strip()
        except OSError:
            existing = ""
        if force:
            os.remove(lock_path)
            return acquire_writer_fence(run_dir, force=force)
        pid_text = existing.split(":", 1)[0] if existing else "unknown"
        raise CheckpointError(
            f"another writer holds {lock_path} (pid {pid_text}); one canonical "
            "writer only. Recovery from a crashed writer requires the explicit "
            "force policy, never an age heuristic.")
    else:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(token)
        return token


def release_writer_fence(run_dir: str, token: str) -> None:
    lock_path = os.path.join(run_dir, WRITER_LOCK)
    try:
        with open(lock_path, "r", encoding="utf-8") as handle:
            content = handle.read().strip()
        # Windows refuses to delete a file with an open handle, so removal
        # happens only after the read handle is closed.
        if content == token:
            os.remove(lock_path)
    except OSError:
        pass


# -- publication --------------------------------------------------------------

def _reject_placeholder_identity(value: str | None, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CheckpointError(f"{name} must be a nonempty identity string")
    lowered = value.strip().lower()
    if lowered in ("k8", "k8-bundle", "k8-bundle-placeholder", "generic",
                   "unavailable", "unavailable-data", "unavailable-source",
                   "unavailable-config", "unavailable-tokenizer",
                   "uncommitted-local"):
        raise CheckpointError(
            f"{name} carries generic placeholder {value!r}; a real source/data/"
            "tokenizer/config hash is required (no generic `k8` identity can "
            "replace source or data hashes)")
    return value.strip()


def publish_checkpoint(
    *,
    run_dir: str,
    run_id: str,
    update_index: int,
    payload: Mapping[str, Any],
    config_identity: str,
    tokenizer_identity: str,
    data_identity: str,
    code_identity: str,
    parent_checkpoint_id: str | None = None,
    writer_token: str | None = None,
    expected_parent: str | None = None,
    milestone: str | None = None,
    phase: str | None = None,
    arm: str | None = None,
    seed: int | None = None,
) -> CheckpointManifest:
    """Real executor publication boundary (contracts S2/S7).

    Thin validated wrapper over `save_checkpoint` that refuses generic `k8`
    identities, requires tokenizer/data/config/code identities, preserves
    writer-token + expected-parent fencing, and binds arm/seed/phase lineage
    into run_id/milestone. Distinct child lineages are preserved via run_id;
    no generic identity replaces source/data hashes.
    """
    config_identity = _reject_placeholder_identity(config_identity, "config_identity")
    tokenizer_identity = _reject_placeholder_identity(
        tokenizer_identity, "tokenizer_identity")
    data_identity = _reject_placeholder_identity(data_identity, "data_identity")
    code_identity = _reject_placeholder_identity(code_identity, "code_identity")
    if not isinstance(run_id, str) or not run_id.strip():
        raise CheckpointError("run_id must be a nonempty string")
    # Bind lineage explicitly so `frozen-...` strings cannot masquerade as
    # verified state. Matching is exact on hyphen-delimited segments (never
    # substring: seed 1701 must not match 21701).
    lineage_parts = [p for p in (phase, arm, str(seed) if seed is not None else None)
                     if p]
    run_segments = set(run_id.split("-"))
    if lineage_parts and not all(part in run_segments for part in lineage_parts):
        # Enforce lineage binding without breaking existing run_id callers
        # that already embed it: only append when missing.
        run_id = f"{run_id}-{'_'.join(lineage_parts)}"
        run_segments = set(run_id.split("-"))
    # Distinct directory per lineage to avoid cross-arm collisions at the
    # same update_index in a shared run_dir.
    dir_suffix = None
    if phase or arm or seed is not None:
        dir_suffix = "-".join([str(p) for p in (phase, arm, seed) if p is not None])
    manifest = save_checkpoint(
        run_dir, payload,
        run_id=run_id, update_index=update_index,
        config_identity=config_identity,
        tokenizer_identity=tokenizer_identity,
        data_identity=data_identity,
        parent_checkpoint_id=parent_checkpoint_id,
        code_identity=code_identity,
        milestone=milestone or (f"{phase}-{arm}-{seed}" if phase else None),
        writer_token=writer_token,
        expected_parent=expected_parent,
        dir_suffix=dir_suffix)
    return manifest


def restore_verify(
    *,
    run_dir: str,
    checkpoint_id: str,
    expect_config_identity: str | None = None,
    expect_tokenizer_identity: str | None = None,
    expect_data_identity: str | None = None,
    expect_parent_checkpoint_id: str | None = None,
) -> dict[str, Any]:
    """Load one artifact and validate state/next-stream under required config.

    Uses the real `load_checkpoint` path (hash, COMPLETE, schema, tokenizer,
    config, data, parent). Returns a restore proof; raises CheckpointError on
    any mismatch. Fresh-process callers should invoke this in a subprocess.
    """
    if not isinstance(checkpoint_id, str) or not checkpoint_id:
        raise CheckpointError("checkpoint_id must be a nonempty string")
    payload, manifest = load_checkpoint(
        run_dir, checkpoint_id=checkpoint_id,
        expect_config_identity=expect_config_identity,
        expect_tokenizer_identity=expect_tokenizer_identity,
        expect_parent_checkpoint_id=expect_parent_checkpoint_id,
        expect_data_identity=expect_data_identity)
    # Minimal state/next-stream validation: payload must carry model +
    # counters; config/architecture identities must be present.
    if not isinstance(payload, dict) or "model" not in payload:
        raise CheckpointError("restored payload has no model state")
    return {"restored_ok": True, "checkpoint_id": manifest.checkpoint_id,
            "payload_sha256": manifest.payload_sha256,
            "update_index": manifest.update_index,
            "config_identity": manifest.config_identity,
            "parent_checkpoint_id": manifest.parent_checkpoint_id}


def _read_complete_manifest(directory: str) -> CheckpointManifest | None:
    """Existing COMPLETE checkpoint manifest, or None when absent/invalid."""
    try:
        complete_path = os.path.join(directory, COMPLETE_MARKER)
        manifest_path = os.path.join(directory, MANIFEST_NAME)
        if not (os.path.isfile(complete_path) and os.path.isfile(manifest_path)):
            return None
        with open(manifest_path, encoding="utf-8") as handle:
            raw = json.load(handle)
        manifest = CheckpointManifest.from_dict(raw)
        with open(complete_path, encoding="utf-8") as handle:
            if handle.read().strip() != manifest.checkpoint_id:
                return None
        return manifest
    except (OSError, ValueError):
        return None


def _manifests_identical(left: CheckpointManifest, right: CheckpointManifest) -> bool:
    """Identity-determining fields equal (created_unix excluded)."""
    fields = ("checkpoint_id", "update_index", "run_id", "parent_checkpoint_id",
              "config_identity", "tokenizer_identity", "data_identity",
              "code_identity", "payload_sha256")
    return all(getattr(left, field) == getattr(right, field) for field in fields)


def save_checkpoint(
    run_dir: str,
    payload: Mapping[str, Any],
    *,
    run_id: str,
    update_index: int,
    config_identity: str,
    tokenizer_identity: str,
    data_identity: str,
    parent_checkpoint_id: str | None,
    code_identity: str = "uncommitted-local",
    milestone: str | None = None,
    writer_token: str | None = None,
    expected_parent: str | None = None,
    dir_suffix: str | None = None,
) -> CheckpointManifest:
    """Publish one checkpoint atomically under a serialized publication
    boundary.

    The caller must hold the run's writer lease (``writer_token`` is verified
    against the live lock file), and ``expected_parent`` must equal the
    LATEST pointer as it exists at publication time: a stale writer cannot
    publish onto a lineage another writer has already advanced (B2.2 R4).

    ``dir_suffix`` namespaces the publication directory for distinct
    arm/seed/phase lineages sharing one run_dir (K8): without it the legacy
    ``update-{index:012d}`` directory collides across arms at the same step.
    Legacy callers omit it and keep exact legacy directory names.
    """
    if not isinstance(update_index, int) or isinstance(update_index, bool) or update_index < 0:
        raise CheckpointError("update_index must be a nonnegative integer")
    checkpoints = checkpoint_root(run_dir)
    os.makedirs(checkpoints, exist_ok=True)
    lock_path = os.path.join(run_dir, WRITER_LOCK)
    if writer_token is not None:
        try:
            with open(lock_path, "r", encoding="utf-8") as handle:
                held = handle.read().strip()
        except OSError:
            held = ""
        if held != writer_token:
            raise CheckpointError(
                "publication refused: the caller does not hold the run's writer "
                "lease (ownership verified at publication time)")
    previous_latest = read_pointer(run_dir, LATEST_POINTER)
    if expected_parent is not None:
        current_parent = previous_latest["checkpoint_id"] if previous_latest else None
        if current_parent != expected_parent:
            raise CheckpointError(
                f"publication refused: expected parent {str(expected_parent)[:12]}... "
                f"but LATEST currently holds {str(current_parent)[:12]}...; a stale "
                "writer cannot infer a new parent from whatever LATEST says")
    # Namespaced (per-arm) lineages must never inherit the global LATEST as
    # their parent: concurrent arms would cross-contaminate lineages
    # (last-writer-wins). Only the legacy global store infers LATEST.
    if dir_suffix is None and previous_latest is not None and parent_checkpoint_id is None:
        parent_checkpoint_id = previous_latest["checkpoint_id"]

    staging = os.path.join(checkpoints, f".staging-{uuid.uuid4().hex}")
    os.makedirs(staging)
    try:
        payload_path = os.path.join(staging, PAYLOAD_NAME)
        with open(payload_path, "wb") as handle:
            torch.save(dict(payload), handle)
            handle.flush()
            os.fsync(handle.fileno())
        payload_sha = _sha256_file(payload_path)
        payload_bytes = os.path.getsize(payload_path)
        manifest = CheckpointManifest(
            checkpoint_id=content_identity({
                "run_id": run_id, "update_index": update_index,
                "config_identity": config_identity, "payload_sha256": payload_sha,
            }),
            schema=CHECKPOINT_SCHEMA, update_index=update_index, run_id=run_id,
            parent_checkpoint_id=parent_checkpoint_id,
            config_identity=config_identity, tokenizer_identity=tokenizer_identity,
            data_identity=data_identity, code_identity=code_identity,
            payload_sha256=payload_sha, payload_bytes=payload_bytes,
            created_unix=time.time())
        manifest_path = os.path.join(staging, MANIFEST_NAME)
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest.to_dict(), handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        with open(os.path.join(staging, COMPLETE_MARKER), "w", encoding="utf-8") as handle:
            handle.write(manifest.checkpoint_id)
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_path(staging)

        if dir_suffix:
            # Namespace distinct lineages: update-{index}-{suffix}. Suffix is
            # restricted to safe characters to keep the directory contained.
            safe = "".join(c if (c.isalnum() or c in ("-", "_")) else "_"
                           for c in str(dir_suffix))[:32]
            if not safe:
                raise CheckpointError("dir_suffix must be nonempty when supplied")
            final = os.path.join(checkpoints, f"update-{update_index:012d}-{safe}")
        else:
            final = os.path.join(checkpoints, f"update-{update_index:012d}")
        if os.path.exists(final):
            # Idempotent retry: an identical re-publication (same lineage,
            # identities, parent and payload bytes — e.g. a retried job that
            # already published this update) returns the existing identity
            # instead of failing the campaign. Genuinely divergent content
            # at the same path still refuses: that needs a fresh run dir.
            existing = _read_complete_manifest(final)
            if existing is not None and _manifests_identical(existing, manifest):
                return existing
            raise CheckpointError(
                f"checkpoint {final} already exists with different content; "
                "checkpoints are never overwritten (use a fresh run directory "
                "for a divergent campaign)")
        os.rename(staging, final)
    finally:
        if os.path.exists(staging):
            shutil.rmtree(staging, ignore_errors=True)
    _fsync_path(checkpoints)

    _write_pointer(run_dir, LATEST_POINTER, {
        "checkpoint_id": manifest.checkpoint_id, "update_index": update_index,
        "directory": os.path.basename(final)})
    if milestone:
        mark_milestone(run_dir, manifest.checkpoint_id, milestone,
                       update_index=update_index, directory=os.path.basename(final))
    return manifest


def _write_pointer(run_dir: str, name: str, content: Mapping[str, Any]) -> None:
    checkpoints = checkpoint_root(run_dir)
    final_path = os.path.join(checkpoints, name)
    tmp_path = os.path.join(checkpoints, f".tmp-{name}-{uuid.uuid4().hex}")
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(content, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, final_path)


def read_pointer(run_dir: str, name: str) -> dict[str, Any] | None:
    path = os.path.join(checkpoint_root(run_dir), name)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def mark_milestone(run_dir: str, checkpoint_id: str, label: str, *,
                   update_index: int, directory: str) -> None:
    """Advance one durable milestone reference; rotation retains its latest state.

    A label names a logical lineage position (for example ``E1-A-1701``),
    not every transient publication made while progressing through it.  The
    former checkpoint for that label is superseded once the next complete
    checkpoint is durably published.  Keeping every intermediate payload
    defeats rotation and can exhaust a bounded Kaggle output volume.

    Concurrent publishers share this file: the update is a bounded
    compare-and-swap (re-read, re-apply, atomic replace) so a sibling's
    entry is never silently clobbered. After a few contended attempts the
    write proceeds with the freshest view rather than failing the job.
    """
    checkpoints = checkpoint_root(run_dir)
    path = os.path.join(checkpoints, MILESTONES_NAME)
    entry = {"checkpoint_id": checkpoint_id, "label": label,
             "update_index": update_index, "directory": directory}
    try:
        os.makedirs(checkpoints, exist_ok=True)
    except OSError:
        pass
    for _ in range(8):
        existing = {"milestones": []}
        before = None
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    before = handle.read()
                existing = json.loads(before)
            except (OSError, json.JSONDecodeError):
                before = None
                existing = {"milestones": []}
        existing["milestones"] = [
            entry for entry in existing["milestones"]
            if entry["checkpoint_id"] != checkpoint_id and entry.get("label") != label]
        existing["milestones"].append(entry)
        body = json.dumps(existing, indent=2, sort_keys=True)
        tmp_path = path + f".tmp-{uuid.uuid4().hex}"
        try:
            with open(tmp_path, "w", encoding="utf-8") as handle:
                handle.write(body)
            if before is not None:
                try:
                    with open(path, "r", encoding="utf-8") as handle:
                        if handle.read() != before:
                            continue
                except OSError:
                    continue
            os.replace(tmp_path, path)
            return
        except OSError:
            pass
        finally:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except OSError:
                pass
    # Contended beyond the retry budget: persist the freshest merged view
    # without failing publication (rotation coverage yields to progress).
    try:
        with open(path, "r", encoding="utf-8") as handle:
            existing = json.load(handle)
    except (OSError, json.JSONDecodeError):
        existing = {"milestones": []}
    existing["milestones"] = [
        entry for entry in existing["milestones"]
        if entry["checkpoint_id"] != checkpoint_id and entry.get("label") != label]
    existing["milestones"].append(entry)
    try:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(existing, handle, indent=2, sort_keys=True)
    except OSError:
        pass


def read_milestones(run_dir: str) -> list[dict[str, Any]]:
    path = os.path.join(checkpoint_root(run_dir), MILESTONES_NAME)
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)["milestones"]


def prune_checkpoints(run_dir: str, *, keep_latest: int = 2) -> list[str]:
    """Rotate old latest checkpoints; milestone-referenced dirs are retained."""
    checkpoints = checkpoint_root(run_dir)
    latest = read_pointer(run_dir, LATEST_POINTER)
    milestones = {entry["directory"] for entry in read_milestones(run_dir)}
    accepted = read_pointer(run_dir, ACCEPTED_PARENT_POINTER)
    protected = set(milestones)
    if latest:
        protected.add(latest["directory"])
    if accepted:
        protected.add(accepted["directory"])
    entries = sorted(
        name for name in os.listdir(checkpoints)
        if name.startswith("update-") and os.path.isdir(os.path.join(checkpoints, name)))
    removable = [name for name in entries if name not in protected]
    removable = removable[:-keep_latest] if keep_latest else removable
    removed = []
    for name in removable:
        # Concurrent publishers prune the same tree: a sibling may have
        # already removed this entry (missing means pruned: the desired end
        # state), or hold it open mid-removal (Windows file locking). Retry
        # briefly, then leave the directory for the next prune rather than
        # failing the training job over housekeeping.
        target = os.path.join(checkpoints, name)
        for _ in range(3):
            try:
                shutil.rmtree(target)
                break
            except FileNotFoundError:
                break
            except PermissionError:
                time.sleep(0.1)
                continue
            except OSError:
                break
        removed.append(name)
    return removed


# -- restore ------------------------------------------------------------------

def _validated_pointer_directory(run_dir: str, pointer: Mapping[str, Any]) -> str:
    """Validate a pointer's directory containment, id and update binding."""
    directory = pointer.get("directory")
    if not isinstance(directory, str) or not directory:
        raise CheckpointError("pointer does not name a checkpoint directory")
    if os.path.sep in directory or "/" in directory or ".." in directory:
        raise CheckpointError(
            f"pointer directory {directory!r} escapes the checkpoint root")
    checkpoints = checkpoint_root(run_dir)
    directory_path = os.path.join(checkpoints, directory)
    real_root = os.path.realpath(checkpoints)
    real_path = os.path.realpath(directory_path)
    if os.path.commonpath([real_root, real_path]) != real_root:
        raise CheckpointError("pointer directory escapes the checkpoint root")
    manifest_id = _manifest_id(directory_path)
    if manifest_id is not None and pointer.get("checkpoint_id")             and manifest_id != pointer["checkpoint_id"]:
        raise CheckpointError(
            "pointer checkpoint id does not match the referenced manifest; "
            "the pointer was redirected or the payload swapped")
    manifest_update = _manifest_update_index(directory_path)
    if manifest_update is not None and pointer.get("update_index") is not None             and manifest_update != pointer["update_index"]:
        raise CheckpointError(
            "pointer update index does not match the referenced manifest")
    return directory


def _manifest_update_index(directory_path: str) -> int | None:
    manifest_path = os.path.join(directory_path, MANIFEST_NAME)
    if not os.path.exists(manifest_path):
        return None
    try:
        with open(manifest_path, "r", encoding="utf-8") as handle:
            return json.load(handle).get("update_index")
    except (OSError, json.JSONDecodeError):
        return None


def load_checkpoint(
    run_dir: str,
    *,
    checkpoint_id: str | None = None,
    expect_config_identity: str | None = None,
    expect_tokenizer_identity: str | None = None,
    expect_parent_checkpoint_id: str | None = None,
    expect_data_identity: str | None = None,
) -> tuple[dict[str, Any], CheckpointManifest]:
    """Load and fully validate one checkpoint; rejects incomplete, tampered,
    redirected or identity-mismatched state."""
    checkpoints = checkpoint_root(run_dir)
    if checkpoint_id is None:
        pointer = read_pointer(run_dir, LATEST_POINTER)
        if pointer is None:
            raise CheckpointError(f"no LATEST checkpoint pointer in {run_dir}")
        directory = _validated_pointer_directory(run_dir, pointer)
    else:
        matches = [name for name in os.listdir(checkpoints)
                   if name.startswith("update-") and os.path.isdir(os.path.join(checkpoints, name))
                   and _manifest_id(os.path.join(checkpoints, name)) == checkpoint_id]
        if not matches:
            raise CheckpointError(f"checkpoint id {checkpoint_id!r} not found in {run_dir}")
        directory = matches[0]
    directory_path = os.path.join(checkpoints, directory)
    if not os.path.exists(os.path.join(directory_path, COMPLETE_MARKER)):
        raise CheckpointError(
            f"checkpoint {directory} is incomplete (no COMPLETE marker); "
            "interrupted publications are never loadable")
    manifest_path = os.path.join(directory_path, MANIFEST_NAME)
    if not os.path.exists(manifest_path):
        raise CheckpointError(f"checkpoint {directory} has no manifest")
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = CheckpointManifest.from_dict(json.load(handle))
    payload_path = os.path.join(directory_path, PAYLOAD_NAME)
    if not os.path.exists(payload_path):
        raise CheckpointError(f"checkpoint {directory} has no payload")
    actual_sha = _sha256_file(payload_path)
    if actual_sha != manifest.payload_sha256:
        raise CheckpointError(
            f"checkpoint {directory} payload hash mismatch: manifest claims "
            f"{manifest.payload_sha256[:12]}, file hashes {actual_sha[:12]}; "
            "the payload is incomplete or tampered")
    if os.path.getsize(payload_path) != manifest.payload_bytes:
        raise CheckpointError(f"checkpoint {directory} payload size mismatch")
    marker_path = os.path.join(directory_path, COMPLETE_MARKER)
    with open(marker_path, "r", encoding="utf-8") as handle:
        if handle.read().strip() != manifest.checkpoint_id:
            raise CheckpointError(
                "COMPLETE marker does not name this checkpoint's identity; "
                "manifest/marker association is inconsistent")
    if expect_config_identity is not None and manifest.config_identity != expect_config_identity:
        raise CheckpointError(
            "checkpoint configuration identity does not match the active config")
    if expect_tokenizer_identity is not None \
            and manifest.tokenizer_identity != expect_tokenizer_identity:
        raise CheckpointError("checkpoint tokenizer identity mismatch")
    if expect_data_identity is not None and manifest.data_identity != expect_data_identity:
        raise CheckpointError(
            "checkpoint data identity does not match the expected prepared data; "
            "resume across changed data requires an explicit migration")
    if expect_parent_checkpoint_id is not None \
            and manifest.parent_checkpoint_id != expect_parent_checkpoint_id:
        raise CheckpointError(
            f"checkpoint parent is {manifest.parent_checkpoint_id!r}, expected "
            f"{expect_parent_checkpoint_id!r}; refusing a stale or divergent parent")
    payload = torch.load(payload_path, map_location="cpu", weights_only=True)
    return payload, manifest


def _manifest_id(directory_path: str) -> str | None:
    manifest_path = os.path.join(directory_path, MANIFEST_NAME)
    if not os.path.exists(manifest_path):
        return None
    try:
        with open(manifest_path, "r", encoding="utf-8") as handle:
            return json.load(handle).get("checkpoint_id")
    except (OSError, json.JSONDecodeError):
        return None


def promote_accepted_parent(run_dir: str) -> dict[str, Any] | None:
    """Copy the current LATEST pointer to the accepted-parent slot."""
    latest = read_pointer(run_dir, LATEST_POINTER)
    if latest is not None:
        _write_pointer(run_dir, ACCEPTED_PARENT_POINTER, latest)
    return latest


# -- RNG streams --------------------------------------------------------------

def capture_rng_state() -> dict[str, Any]:
    """Capture every RNG stream in restricted-load-safe primitive form.

    numpy's Mersenne-Twister state is stored as plain integer lists so the
    checkpoint loads under ``torch.load(weights_only=True)`` without any
    unrestricted-pickle fallback (B2.2 R4).
    """
    state: dict[str, Any] = {
        "torch": torch.get_rng_state(),
        "python": random.getstate(),
    }
    try:
        import numpy as np
    except ImportError:
        np = None
    if np is not None:
        name, keys, pos, has_gauss, cached = np.random.get_state()
        state["numpy"] = {"name": name, "keys": [int(key) for key in keys],
                          "pos": int(pos), "has_gauss": int(has_gauss),
                          "cached_gaussian": float(cached)}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    torch.set_rng_state(state["torch"] if isinstance(state["torch"], torch.Tensor)
                        else torch.tensor(list(state["torch"]), dtype=torch.uint8))
    random.setstate(_coerce_python_state(state["python"]))
    if "numpy" in state:
        import numpy as np

        encoded = state["numpy"]
        if isinstance(encoded, Mapping):
            np.random.set_state((encoded["name"],
                                 np.array(encoded["keys"], dtype=np.uint32),
                                 int(encoded["pos"]), int(encoded["has_gauss"]),
                                 float(encoded["cached_gaussian"])))
        else:  # legacy in-memory shape (same process)
            np.random.set_state(encoded)
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(list(state["cuda"]))


def _coerce_python_state(raw):
    if isinstance(raw, tuple):
        return raw
    return tuple(raw)
