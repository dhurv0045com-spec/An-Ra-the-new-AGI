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

def acquire_writer_fence(run_dir: str, *, stale_after_seconds: float = 6 * 3600) -> str:
    """Single-writer fencing via an exclusive lock file.

    A leftover lock is broken when it names this process or is older than the
    declared staleness bound; a live foreign writer raises instead.
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
        pid_text = existing.split(":", 1)[0]
        try:
            age = time.time() - os.path.getmtime(lock_path)
        except OSError:
            age = 0.0
        if age > stale_after_seconds:
            os.remove(lock_path)
            return acquire_writer_fence(run_dir, stale_after_seconds=stale_after_seconds)
        raise CheckpointError(
            f"another writer holds {lock_path} (pid {pid_text}); one canonical writer only")
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
) -> CheckpointManifest:
    """Publish one checkpoint atomically and advance the LATEST pointer."""
    if not isinstance(update_index, int) or isinstance(update_index, bool) or update_index < 0:
        raise CheckpointError("update_index must be a nonnegative integer")
    checkpoints = checkpoint_root(run_dir)
    os.makedirs(checkpoints, exist_ok=True)
    previous_latest = read_pointer(run_dir, LATEST_POINTER)
    if previous_latest is not None and parent_checkpoint_id is None:
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

        final = os.path.join(checkpoints, f"update-{update_index:012d}")
        if os.path.exists(final):
            raise CheckpointError(
                f"checkpoint {final} already exists; checkpoints are never overwritten")
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
    """Record a durable milestone reference; rotation must retain it."""
    checkpoints = checkpoint_root(run_dir)
    path = os.path.join(checkpoints, MILESTONES_NAME)
    existing = {"milestones": []}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as handle:
            existing = json.load(handle)
    existing["milestones"] = [
        entry for entry in existing["milestones"] if entry["checkpoint_id"] != checkpoint_id]
    existing["milestones"].append({"checkpoint_id": checkpoint_id, "label": label,
                                   "update_index": update_index, "directory": directory})
    tmp_path = path + f".tmp-{uuid.uuid4().hex}"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(existing, handle, indent=2, sort_keys=True)
    os.replace(tmp_path, path)


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
        shutil.rmtree(os.path.join(checkpoints, name))
        removed.append(name)
    return removed


# -- restore ------------------------------------------------------------------

def load_checkpoint(
    run_dir: str,
    *,
    checkpoint_id: str | None = None,
    expect_config_identity: str | None = None,
    expect_tokenizer_identity: str | None = None,
    expect_parent_checkpoint_id: str | None = None,
) -> tuple[dict[str, Any], CheckpointManifest]:
    """Load and fully validate one checkpoint; rejects incomplete or tampered state."""
    checkpoints = checkpoint_root(run_dir)
    if checkpoint_id is None:
        pointer = read_pointer(run_dir, LATEST_POINTER)
        if pointer is None:
            raise CheckpointError(f"no LATEST checkpoint pointer in {run_dir}")
        directory = pointer["directory"]
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
    if expect_config_identity is not None and manifest.config_identity != expect_config_identity:
        raise CheckpointError(
            "checkpoint configuration identity does not match the active config")
    if expect_tokenizer_identity is not None \
            and manifest.tokenizer_identity != expect_tokenizer_identity:
        raise CheckpointError("checkpoint tokenizer identity mismatch")
    if expect_parent_checkpoint_id is not None \
            and manifest.parent_checkpoint_id != expect_parent_checkpoint_id:
        raise CheckpointError(
            f"checkpoint parent is {manifest.parent_checkpoint_id!r}, expected "
            f"{expect_parent_checkpoint_id!r}; refusing a stale or divergent parent")
    try:
        payload = torch.load(payload_path, map_location="cpu", weights_only=True)
    except Exception:
        payload = torch.load(payload_path, map_location="cpu", weights_only=False)
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
    state: dict[str, Any] = {
        "torch": torch.get_rng_state(),
        "python": random.getstate(),
    }
    try:
        import numpy as np
    except ImportError:
        np = None
    if np is not None:
        state["numpy"] = np.random.get_state()
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    torch.set_rng_state(state["torch"] if isinstance(state["torch"], torch.Tensor)
                        else torch.tensor(list(state["torch"]), dtype=torch.uint8))
    random.setstate(_coerce_python_state(state["python"]))
    if "numpy" in state:
        import numpy as np

        np.random.set_state(state["numpy"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(list(state["cuda"]))


def _coerce_python_state(raw):
    if isinstance(raw, tuple):
        return raw
    return tuple(raw)
