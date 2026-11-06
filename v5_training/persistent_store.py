"""Durable mounted-filesystem mirror for checkpoint generations.

The local CheckpointStore claims only local fsync + atomic publish — honest
on ephemeral runtimes. This adapter adds persistence across runtimes on a
mounted durable filesystem (e.g. Google Drive) WITHOUT assuming local POSIX
atomic semantics on the mount:

1. publish + verify locally (CheckpointStore does this),
2. copy the checkpoint object + manifest to the mirror via a staging name,
3. verify every byte/hash FROM THE DESTINATION,
4. advance the mirror pointer only after the verified copy,
5. re-open and re-verify through the pointer.

Only then is the generation PERSISTENT_DURABLE. Resume on a fresh runtime
materializes the mirrored generation into a local store (which re-verifies
everything through its own restore path) and continues. Every failure —
interrupted copy, missing/corrupt file, stale pointer, wrong lineage,
pointer-before-object — fails closed.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

from .checkpoint import CheckpointStore, _validate_checkpoint_sha, _validate_lineage_id


MIRROR_SCHEMA = "anra-v5-persistent-mirror/v1"
POINTER_FILENAME = "MIRROR_HEAD"
DURABLE_STATUS = "PERSISTENT_DURABLE"


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_mirror_object(directory: Path, checkpoint_sha256: str) -> None:
    """Stream-check the object that a mirror or local pointer will name."""
    try:
        manifest_bytes = (directory / "manifest.json").read_bytes()
        if hashlib.sha256(manifest_bytes).hexdigest() != checkpoint_sha256:
            raise ValueError("mirrored manifest hash mismatch at destination")
        items = json.loads(manifest_bytes)["components"]
        names = {"manifest.json"}
        for item in items:
            name = item["name"]
            _validate_lineage_id(name)
            if name in names:
                raise ValueError("mirrored manifest has duplicate component names")
            names.add(name)
            path = directory / name
            if (not path.is_file() or path.stat().st_size != item["byte_size"]
                    or _sha256_file(path) != item["sha256"]):
                raise ValueError(f"mirrored component corrupt: {name}")
        if {path.name for path in directory.iterdir()} != names:
            raise ValueError("mirrored object has an unexpected component inventory")
    except (OSError, TypeError, KeyError, ValueError) as exc:
        raise ValueError(f"mirrored object fails destination verification: {exc}") from exc


def _copy_verified_object(source: Path, destination: Path, identity: str) -> None:
    if destination.is_dir():
        _verify_mirror_object(destination, identity)
        return
    try:
        with tempfile.TemporaryDirectory(prefix=".checkpoint-copy-", dir=destination.parent) as temp:
            staging = Path(temp)
            for child in sorted(source.iterdir()):
                if child.is_file():
                    shutil.copyfile(child, staging / child.name)
            _verify_mirror_object(staging, identity)
            try:
                os.replace(staging, destination)
            except OSError:
                if not destination.is_dir():
                    raise
                _verify_mirror_object(destination, identity)
    except OSError as exc:
        raise ValueError(f"mirror copy interrupted: {exc}") from exc


def _replace_pointer(path: Path, payload: bytes) -> None:
    with tempfile.NamedTemporaryFile(prefix=f".{path.name}-", dir=path.parent, delete=False) as file:
        temporary = Path(file.name)
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def mirror_checkpoint(store: CheckpointStore, mirror_root: str | Path, *,
                      checkpoint_sha256: str) -> dict[str, object]:
    """Copy one committed generation to the durable mirror and fence the pointer."""

    _validate_checkpoint_sha(checkpoint_sha256)
    mirror = Path(mirror_root).resolve() / store.lineage_id
    mirror.mkdir(parents=True, exist_ok=True)
    source = store.objects / checkpoint_sha256
    if not source.is_dir():
        raise ValueError("mirror source generation is not committed locally")
    try:
        store.restore(checkpoint_sha256)
    except ValueError as exc:
        raise ValueError(f"mirror source fails local verification: {exc}") from exc
    _copy_verified_object(source, mirror / checkpoint_sha256, checkpoint_sha256)
    _replace_pointer(
        mirror / POINTER_FILENAME,
        _canonical_json({"schema": MIRROR_SCHEMA, "lineage_id": store.lineage_id,
                         "head_sha256": checkpoint_sha256}),
    )
    reread = read_mirror_pointer(mirror_root, lineage_id=store.lineage_id)
    if reread != checkpoint_sha256:
        raise ValueError("mirror pointer reread disagrees after advance")
    return {"schema": MIRROR_SCHEMA, "lineage_id": store.lineage_id,
            "checkpoint_sha256": checkpoint_sha256,
            "status": DURABLE_STATUS}


def read_mirror_pointer(mirror_root: str | Path, *, lineage_id: str) -> str | None:
    """Read and validate the mirror head pointer (None when never mirrored)."""

    _validate_lineage_id(lineage_id)
    pointer = Path(mirror_root).resolve() / lineage_id / POINTER_FILENAME
    if not pointer.exists():
        return None
    try:
        document = json.loads(pointer.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ValueError("mirror pointer is corrupt") from exc
    if (not isinstance(document, dict) or document.get("schema") != MIRROR_SCHEMA
            or document.get("lineage_id") != lineage_id):
        raise ValueError("mirror pointer lineage mismatch")
    head = document.get("head_sha256")
    if not isinstance(head, str) or len(head) != 64 or any(
            c not in "0123456789abcdef" for c in head):
        raise ValueError("mirror pointer head is not a SHA-256")
    return head


def materialize_local(mirror_root: str | Path, local_store: CheckpointStore, *,
                      checkpoint_sha256: str | None = None) -> str:
    """Copy a mirrored generation into a local store and verify it there.

    Returns the materialized SHA. The local store's own restore path
    re-verifies manifest, components, and state, so a corrupt mirror cannot
    become a trusted local head.
    """

    mirror = Path(mirror_root).resolve() / local_store.lineage_id
    head = read_mirror_pointer(mirror_root, lineage_id=local_store.lineage_id)
    wanted = head if checkpoint_sha256 is None else checkpoint_sha256
    if wanted is None:
        raise ValueError("mirror holds no head for this lineage")
    _validate_checkpoint_sha(wanted)
    source = mirror / wanted
    if not source.is_dir():
        raise ValueError("mirrored generation object is missing")
    _copy_verified_object(source, local_store.objects / wanted, wanted)
    local_store.restore(wanted)
    if checkpoint_sha256 is None and read_mirror_pointer(
        mirror_root, lineage_id=local_store.lineage_id
    ) != wanted:
        raise ValueError("mirror pointer moved during materialization")
    current = local_store.latest_sha256()
    if current is None:
        _replace_pointer(local_store.latest, f"{wanted}\n".encode("ascii"))
    elif current != wanted and checkpoint_sha256 is None:
        raise ValueError("local head disagrees with the mirrored head")
    return wanted


__all__ = ["DURABLE_STATUS", "MIRROR_SCHEMA", "POINTER_FILENAME",
           "materialize_local", "mirror_checkpoint", "read_mirror_pointer"]
