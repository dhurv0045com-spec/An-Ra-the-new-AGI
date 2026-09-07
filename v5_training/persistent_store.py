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
from pathlib import Path

from .checkpoint import CheckpointStore


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


def mirror_checkpoint(store: CheckpointStore, mirror_root: str | Path, *,
                      checkpoint_sha256: str) -> dict[str, object]:
    """Copy one committed generation to the durable mirror and fence the pointer."""

    mirror = Path(mirror_root).resolve() / store.lineage_id
    mirror.mkdir(parents=True, exist_ok=True)
    source = store.objects / checkpoint_sha256
    if not source.is_dir():
        raise ValueError("mirror source generation is not committed locally")
    try:
        expected_state, _ = store.restore(checkpoint_sha256)
    except ValueError as exc:
        raise ValueError(f"mirror source fails local verification: {exc}") from exc
    _ = expected_state
    staging = mirror / f".staging-{checkpoint_sha256}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        for child in sorted(source.iterdir()):
            if child.is_file():
                shutil.copyfile(child, staging / child.name)
        manifest_bytes = (staging / "manifest.json").read_bytes()
    except OSError as exc:
        shutil.rmtree(staging, ignore_errors=True)
        raise ValueError(f"mirror copy interrupted: {exc}") from exc
    if hashlib.sha256(manifest_bytes).hexdigest() != checkpoint_sha256:
        shutil.rmtree(staging, ignore_errors=True)
        raise ValueError("mirrored manifest hash mismatch at destination")
    try:
        manifest = json.loads(manifest_bytes)
        for item in manifest["components"]:
            staged = staging / item["name"]
            if (not staged.is_file() or len(staged.read_bytes()) != item["byte_size"]
                    or _sha256_file(staged) != item["sha256"]):
                raise ValueError(f"mirrored component corrupt: {item['name']}")
    except (ValueError, KeyError) as exc:
        shutil.rmtree(staging, ignore_errors=True)
        raise ValueError(f"mirrored object fails destination verification: {exc}") from exc
    destination = mirror / checkpoint_sha256
    if destination.exists():
        shutil.rmtree(staging, ignore_errors=True)
    else:
        os.replace(staging, destination)
    pointer_tmp = mirror / f".{POINTER_FILENAME}.tmp"
    pointer_tmp.write_text(
        _canonical_json({"schema": MIRROR_SCHEMA, "lineage_id": store.lineage_id,
                         "head_sha256": checkpoint_sha256}).decode("utf-8"),
        encoding="utf-8")
    os.replace(pointer_tmp, mirror / POINTER_FILENAME)
    reread = read_mirror_pointer(mirror_root, lineage_id=store.lineage_id)
    if reread != checkpoint_sha256:
        raise ValueError("mirror pointer reread disagrees after advance")
    return {"schema": MIRROR_SCHEMA, "lineage_id": store.lineage_id,
            "checkpoint_sha256": checkpoint_sha256,
            "status": DURABLE_STATUS}


def read_mirror_pointer(mirror_root: str | Path, *, lineage_id: str) -> str | None:
    """Read and validate the mirror head pointer (None when never mirrored)."""

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
    wanted = checkpoint_sha256 or head
    if wanted is None:
        raise ValueError("mirror holds no head for this lineage")
    if head is not None and wanted != head and checkpoint_sha256 is None:
        raise ValueError("mirror pointer moved during materialization")
    source = mirror / wanted
    if not source.is_dir():
        raise ValueError("mirrored generation object is missing")
    destination = local_store.objects / wanted
    if not destination.exists():
        staging = local_store.lineage_root / f".mirror-{wanted}"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        for child in sorted(source.iterdir()):
            if child.is_file():
                shutil.copyfile(child, staging / child.name)
        os.replace(staging, destination)
    state, _ = local_store.restore(wanted)
    current = local_store.latest_sha256()
    if current is None:
        pointer_tmp = local_store.lineage_root / f".LATEST-mirror-{wanted[:8]}"
        pointer_tmp.write_text(f"{wanted}\n", encoding="ascii")
        os.replace(pointer_tmp, local_store.latest)
    elif current != wanted and checkpoint_sha256 is None:
        raise ValueError("local head disagrees with the mirrored head")
    _ = state
    return wanted


__all__ = ["DURABLE_STATUS", "MIRROR_SCHEMA", "POINTER_FILENAME",
           "materialize_local", "mirror_checkpoint", "read_mirror_pointer"]
