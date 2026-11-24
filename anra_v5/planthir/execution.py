"""Durable, single-writer execution metadata and update journals."""
from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import tempfile
import time
import uuid
from typing import Any, Iterator

from v5_training.checkpoint_lock import checkpoint_writer


def sanitize_for_json(obj: Any) -> Any:
    """Recursively convert Tensors, NumPy scalars/arrays, Paths, NaN/Inf floats, and sets to strict JSON values."""

    if obj is None or isinstance(obj, bool):
        return obj
    if isinstance(obj, int):
        return int(obj)
    if isinstance(obj, str):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (bytes, bytearray)):
        return bytes(obj).decode("utf-8", errors="replace")
    if isinstance(obj, Mapping):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [sanitize_for_json(v) for v in obj]
    if hasattr(obj, "detach") and hasattr(obj, "cpu") and hasattr(obj, "numel"):
        t = obj.detach().cpu()
        n = int(t.numel())
        if t.ndim == 0 or n == 1:
            return sanitize_for_json(t.reshape(()).item()) if n == 1 else []
        if n <= 4096:
            return sanitize_for_json(t.tolist())
        return {
            "__tensor__": True,
            "shape": [int(d) for d in t.shape],
            "dtype": str(t.dtype),
            "numel": n,
        }
    if hasattr(obj, "tolist") and hasattr(obj, "shape") and hasattr(obj, "size"):
        n = int(obj.size)
        if getattr(obj, "ndim", None) == 0:
            return sanitize_for_json(obj.item())
        if n <= 4096:
            return sanitize_for_json(obj.tolist())
        return {
            "__ndarray__": True,
            "shape": [int(d) for d in obj.shape],
            "dtype": str(obj.dtype),
            "size": n,
        }
    if hasattr(obj, "item") and callable(getattr(obj, "item")):
        scalar = obj.item()
        if scalar is not obj:
            return sanitize_for_json(scalar)
    return str(obj)


def dump_json_safe(obj: Any, *, indent: int | None = 2) -> str:
    """Serialize any Python/PyTorch/NumPy telemetry object to deterministic strict JSON text."""

    return json.dumps(sanitize_for_json(obj), indent=indent, sort_keys=True, allow_nan=False) + "\n"


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _sync_dir(path: Path) -> None:
    if os.name != "nt":
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def _atomic_replace(src: str | Path, dst: str | Path) -> None:
    """Replace `dst` with `src`, retrying brief Windows AV/indexer file locks."""
    if os.name != "nt":
        os.replace(src, dst)
        return
    last_exc: PermissionError | None = None
    for attempt in range(6):
        try:
            os.replace(src, dst)
            return
        except PermissionError as exc:
            last_exc = exc
            time.sleep(0.02 * (attempt + 1))
    assert last_exc is not None
    raise last_exc


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        _atomic_replace(name, path)
        _sync_dir(path.parent)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def run_session(work: str | Path, identity: dict[str, Any], resume: bool = False) -> Iterator[dict[str, Any]]:
    """Bind a work directory to an invocation and record its durable lifecycle.

    The lock is process-held. An invalid invocation never changes the identity or
    status of the previous run. The mutable yielded state is included in status.
    """
    if not isinstance(identity, dict):
        raise TypeError("identity must be a dictionary")
    canonical = _json_bytes(identity)
    digest = sha256(canonical).hexdigest()
    root = Path(work)
    root.mkdir(parents=True, exist_ok=True)
    identity_path = root / "RUN_IDENTITY.json"
    status_path = root / "RUN_STATUS.json"
    with checkpoint_writer(root, filename=".planthir-run-writer"):
        if identity_path.exists():
            try:
                stored = json.loads(identity_path.read_text(encoding="utf-8"))
            except (ValueError, OSError) as exc:
                raise ValueError("existing run identity is unreadable") from exc
            if not isinstance(stored, dict) or stored.get("sha256") != digest or _json_bytes(stored.get("identity")) != canonical:
                raise ValueError("run identity mismatch")
            if not resume:
                raise ValueError("work directory already belongs to a run; resume required")
        else:
            if resume:
                raise ValueError("cannot resume without a run identity")
            if any(p.name not in {".planthir-run-writer", ".planthir_worker.lock"} for p in root.iterdir()):
                raise ValueError("nonempty work directory has no run identity")
            _atomic_write(identity_path, _json_bytes({"sha256": digest, "identity": identity}) + b"\n")
        if resume and status_path.exists():
            # Preserve the previous attempt before publishing a new RUNNING status.
            prior_status = json.loads(status_path.read_text(encoding="utf-8"))
            _append_jsonl(root / "ATTEMPTS.jsonl", prior_status)
        state: dict[str, Any] = {"phase": "starting"}

        def write_status(status: str, exc: BaseException | None = None) -> None:
            latest = sorted(p.relative_to(root).as_posix() for p in root.rglob("LATEST") if p.is_file())
            record: dict[str, Any] = {
                "status": status, "identity_sha256": digest, "updated_at": _timestamp(),
                "phase": state.get("phase"), "state": dict(state), "checkpoint_latest_paths": latest,
            }
            if exc is not None:
                record["exception"] = {"type": type(exc).__name__, "message": str(exc)}
            _atomic_write(status_path, _json_bytes(record) + b"\n")

        write_status("RUNNING")
        try:
            yield state
        except BaseException as exc:
            write_status("INTERRUPTED" if isinstance(exc, (KeyboardInterrupt, SystemExit, TimeoutError)) else "FAILED", exc)
            raise
        else:
            write_status("COMPLETE")


def read_update_log(path: str | Path, restored_update: int) -> list[dict[str, Any]]:
    """Validate a contiguous journal and quarantine updates beyond the checkpoint."""
    if type(restored_update) is not int or restored_update < 0:
        raise ValueError("restored_update must be a nonnegative integer")
    file = Path(path)
    if not file.exists():
        if restored_update:
            raise ValueError("journal missing committed updates")
        return []
    raw = file.read_bytes()
    lines = raw.splitlines(keepends=True)
    records: list[dict[str, Any]] = []
    committed_end = 0
    offset = 0
    for index, line in enumerate(lines, 1):
        complete = line.endswith(b"\n")
        try:
            record = json.loads(line)
            valid = isinstance(record, dict) and type(record.get("update")) is int and record["update"] == index
        except (json.JSONDecodeError, UnicodeError):
            valid = False
            record = None
        if not complete or not valid:
            # Only an incomplete final append beyond the durable checkpoint can
            # be discarded. Complete malformed records are corruption.
            if index > restored_update and index == len(lines) and not complete:
                break
            raise ValueError(f"journal update sequence invalid at line {index}")
        records.append(record)
        offset += len(line)
        if index == restored_update:
            committed_end = offset
    if len(records) < restored_update:
        raise ValueError("journal missing committed updates")
    if len(raw) > committed_end:
        orphan = file.with_name(f"{file.name}.orphan-{uuid.uuid4().hex}")
        _atomic_write(orphan, raw[committed_end:])
        _atomic_write(file, raw[:committed_end])

    return records[:restored_update]


def append_update_log(path: str | Path, record: dict[str, Any]) -> None:
    """Append and sync one update; caller holds the enclosing run lock."""
    if not isinstance(record, dict) or type(record.get("update")) is not int or record["update"] < 1:
        raise ValueError("update record requires a positive integer update")
    _append_jsonl(Path(path), record)


def _append_jsonl(file: Path, record: Any) -> None:
    data = _json_bytes(record) + b"\n"
    file.parent.mkdir(parents=True, exist_ok=True)
    created = not file.exists()
    with file.open("ab") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    if created:
        _sync_dir(file.parent)


def trigger_fault_point(stage: str) -> None:
    """Test-only deterministic crash boundary for subprocess interruption recovery tests.

    Activated only when PLANTHIR_FAULT_INJECTION_POINT matches ``stage`` (or prefix
    before ':') and optional marker file PLANTHIR_FAULT_ONCE_FILE does not yet exist.
    """
    target = os.environ.get("PLANTHIR_FAULT_INJECTION_POINT", "").strip()
    if not target:
        return
    if stage != target and not stage.startswith(target + ":"):
        return
    once_path_raw = os.environ.get("PLANTHIR_FAULT_ONCE_FILE", "").strip()
    if once_path_raw:
        once_path = Path(once_path_raw)
        if once_path.exists():
            return
        once_path.parent.mkdir(parents=True, exist_ok=True)
        once_path.write_text(stage + "\n", encoding="utf-8")
    exit_code = int(os.environ.get("PLANTHIR_FAULT_EXIT_CODE", "99"))
    os._exit(exit_code)
