"""Durable, replayable runtime experience ledger.

The ledger is deliberately independent from training code.  Serving only
appends schema-versioned events; promotion into training data happens through
the explicit, hash-manifested compaction boundary below.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
SEAL_MANIFEST_VERSION = 1
DEFAULT_MAX_SHARD_BYTES = 64 * 1024 * 1024
_PII_PATTERNS = (
    re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE),
    re.compile(r"\b(?:\+?\d[\s().-]?){10,15}\b"),
    re.compile(r"\b(?:\d[ -]*?){13,19}\b"),
)


def _json_default(value: object) -> str:
    return f"<{type(value).__name__}:{value!s}>"


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        default=_json_default,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def content_hash(value: object) -> str:
    """Return a stable SHA-256 for arbitrary JSON-like input."""
    return hashlib.sha256(_canonical_json(value)).hexdigest()


@dataclass(frozen=True)
class ExperienceEvent:
    trace_id: str
    kind: str
    inputs_hash: str
    output: Any = None
    verifier_verdicts: tuple[dict[str, Any], ...] = ()
    gate_record: dict[str, Any] = field(default_factory=dict)
    tokens: dict[str, int] = field(default_factory=dict)
    latency: dict[str, float] = field(default_factory=dict)
    source: str = "runtime"
    metadata: dict[str, Any] = field(default_factory=dict)
    ts: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    schema_version: int = SCHEMA_VERSION

    def envelope(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["verifier_verdicts"] = list(self.verifier_verdicts)
        payload["event_hash"] = content_hash(payload)
        return payload


def verify_envelope(envelope: Mapping[str, Any]) -> bool:
    """Validate schema and the event's tamper-evident content hash."""
    if int(envelope.get("schema_version", -1)) != SCHEMA_VERSION:
        return False
    expected = str(envelope.get("event_hash", ""))
    unsigned = {key: value for key, value in envelope.items() if key != "event_hash"}
    return bool(expected) and expected == content_hash(unsigned)


class ExperienceLedger:
    """Thread-safe JSONL writer with fail-open serving semantics."""

    def __init__(
        self,
        root: str | Path,
        *,
        strict: bool = False,
        max_shard_bytes: int = DEFAULT_MAX_SHARD_BYTES,
        retention_days: int | None = None,
    ) -> None:
        self.root = Path(root)
        self.strict = strict
        self.max_shard_bytes = max(1, int(max_shard_bytes))
        self.retention_days = retention_days
        self._lock = threading.Lock()
        self._active_shard: Path | None = None
        self._active_day = ""
        self.write_failures = 0

    @property
    def active_shard(self) -> Path:
        return self._select_active_shard()

    @property
    def seal_manifest_path(self) -> Path:
        return self.root / "manifests" / f"sealed-shards-v{SEAL_MANIFEST_VERSION}.json"

    def _select_active_shard(self) -> Path:
        day = datetime.now(UTC).strftime("%Y%m%d")
        if (
            self._active_shard is not None
            and self._active_day == day
            and (
                not self._active_shard.exists()
                or self._active_shard.stat().st_size < self.max_shard_bytes
            )
        ):
            return self._active_shard

        shard_dir = self.root / "shards"
        shard_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"experience-v{SCHEMA_VERSION}-{day}-{os.getpid()}-"
        candidates = sorted(shard_dir.glob(f"{prefix}*.jsonl"))
        if candidates and candidates[-1].stat().st_size < self.max_shard_bytes:
            selected = candidates[-1]
        else:
            selected = shard_dir / f"{prefix}{len(candidates):04d}.jsonl"
        self._active_day = day
        self._active_shard = selected
        return selected

    def append(self, event: ExperienceEvent) -> bool:
        """Append one complete line; return False on a fail-open write failure."""
        try:
            envelope = event.envelope()
            encoded = _canonical_json(envelope) + b"\n"
            with self._lock:
                path = self._select_active_shard()
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("ab", buffering=0) as stream:
                    stream.write(encoded)
            return True
        except Exception:
            self.write_failures += 1
            if self.strict:
                raise
            return False

    def record(
        self,
        *,
        kind: str,
        inputs: object,
        output: Any = None,
        trace_id: str | None = None,
        verifier_verdicts: Iterable[Mapping[str, Any]] = (),
        gate_record: Mapping[str, Any] | None = None,
        tokens: Mapping[str, int] | None = None,
        latency: Mapping[str, float] | None = None,
        source: str = "runtime",
        metadata: Mapping[str, Any] | None = None,
    ) -> tuple[str, bool]:
        trace_id = trace_id or str(uuid.uuid4())
        event = ExperienceEvent(
            trace_id=trace_id,
            kind=kind,
            inputs_hash=content_hash(inputs),
            output=output,
            verifier_verdicts=tuple(dict(item) for item in verifier_verdicts),
            gate_record=dict(gate_record or {}),
            tokens={str(k): int(v) for k, v in (tokens or {}).items()},
            latency={str(k): float(v) for k, v in (latency or {}).items()},
            source=source,
            metadata=dict(metadata or {}),
        )
        return trace_id, self.append(event)

    def iter_events(self, *, validate: bool = True) -> Iterable[dict[str, Any]]:
        for path in sorted((self.root / "shards").glob("experience-v*.jsonl")):
            with path.open("r", encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, 1):
                    if not line.strip():
                        continue
                    envelope = json.loads(line)
                    if validate and not verify_envelope(envelope):
                        raise ValueError(f"invalid experience event: {path}:{line_number}")
                    yield envelope

    def replay(self, trace_id: str) -> list[dict[str, Any]]:
        """Replay a trace as its validated, ordered recorded outcomes."""
        return [event for event in self.iter_events() if event["trace_id"] == trace_id]

    def seal_shards(self, *, include_active: bool = False) -> dict[str, Any]:
        """Seal immutable shard hashes into a tamper-evident manifest."""
        with self._lock:
            active = self._select_active_shard()
            sealed: list[dict[str, Any]] = []
            for path in sorted((self.root / "shards").glob("experience-v*.jsonl")):
                if not include_active and path == active:
                    continue
                if not path.exists() or path.stat().st_size == 0:
                    continue
                sealed.append(_sealed_shard_entry(self.root, path))

            manifest = {
                "schema_version": SEAL_MANIFEST_VERSION,
                "ledger_schema_version": SCHEMA_VERSION,
                "created_at": datetime.now(UTC).isoformat(),
                "retention_days": self.retention_days,
                "shards": sealed,
            }
            manifest["manifest_hash"] = content_hash(manifest)
            self.seal_manifest_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.seal_manifest_path.with_suffix(".json.tmp")
            temporary.write_bytes(_canonical_json(manifest) + b"\n")
            os.replace(temporary, self.seal_manifest_path)
            return manifest

    def verify_sealed_manifest(
        self,
        manifest_path: str | Path | None = None,
        *,
        validate_events: bool = True,
    ) -> dict[str, Any]:
        """Verify sealed manifest hash, shard hashes, sizes, and event envelopes."""
        path = Path(manifest_path) if manifest_path is not None else self.seal_manifest_path
        manifest = json.loads(path.read_text(encoding="utf-8"))
        expected = str(manifest.get("manifest_hash", ""))
        unsigned = {key: value for key, value in manifest.items() if key != "manifest_hash"}
        if not expected or expected != content_hash(unsigned):
            raise ValueError(f"invalid sealed manifest hash: {path}")
        verified = 0
        for shard in manifest.get("shards", []):
            rel = str(shard["path"])
            shard_path = self.root / rel
            if not shard_path.exists():
                raise ValueError(f"sealed shard missing: {rel}")
            data = shard_path.read_bytes()
            if int(shard.get("bytes", -1)) != len(data):
                raise ValueError(f"sealed shard size mismatch: {rel}")
            if str(shard.get("sha256", "")) != hashlib.sha256(data).hexdigest():
                raise ValueError(f"sealed shard hash mismatch: {rel}")
            if validate_events:
                record_count = 0
                for line_number, line in enumerate(
                    shard_path.read_text(encoding="utf-8").splitlines(),
                    1,
                ):
                    if not line.strip():
                        continue
                    envelope = json.loads(line)
                    if not verify_envelope(envelope):
                        raise ValueError(
                            f"invalid sealed event: {rel}:{line_number}"
                        )
                    record_count += 1
                if int(shard.get("records", -1)) != record_count:
                    raise ValueError(f"sealed shard record-count mismatch: {rel}")
            verified += 1
        return {"verified": True, "shards": verified, "manifest_path": str(path)}

    def apply_retention(self, *, retention_days: int | None = None) -> list[Path]:
        """Delete only verified sealed shards older than the retention window."""
        days = self.retention_days if retention_days is None else retention_days
        if days is None:
            return []
        self.verify_sealed_manifest()
        manifest = json.loads(self.seal_manifest_path.read_text(encoding="utf-8"))
        cutoff = time.time() - max(0, int(days)) * 86_400
        removed: list[Path] = []
        for shard in manifest.get("shards", []):
            rel = str(shard["path"])
            shard_path = self.root / rel
            if not shard_path.exists():
                continue
            last_ts = str(shard.get("last_ts") or "")
            try:
                shard_time = datetime.fromisoformat(last_ts).timestamp()
            except ValueError:
                shard_time = shard_path.stat().st_mtime
            if shard_time < cutoff:
                shard_path.unlink()
                removed.append(shard_path)
        return removed


def _sealed_shard_entry(root: Path, path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    events: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        envelope = json.loads(line)
        if not verify_envelope(envelope):
            rel = path.relative_to(root).as_posix()
            raise ValueError(f"invalid event while sealing shard: {rel}:{line_number}")
        events.append(envelope)
    rel_path = path.relative_to(root).as_posix()
    return {
        "path": rel_path,
        "bytes": len(data),
        "records": len(events),
        "sha256": hashlib.sha256(data).hexdigest(),
        "first_event_id": events[0]["event_id"] if events else None,
        "last_event_id": events[-1]["event_id"] if events else None,
        "first_ts": events[0]["ts"] if events else None,
        "last_ts": events[-1]["ts"] if events else None,
    }


_DEFAULT_LEDGER: ExperienceLedger | None = None
_DEFAULT_LOCK = threading.Lock()


def get_default_ledger() -> ExperienceLedger:
    global _DEFAULT_LEDGER
    with _DEFAULT_LOCK:
        if _DEFAULT_LEDGER is None:
            configured = os.environ.get("ANRA_EXPERIENCE_LEDGER_DIR", "").strip()
            root = (
                Path(configured)
                if configured
                else Path(__file__).resolve().parents[1] / "state" / "experience_ledger"
            )
            _DEFAULT_LEDGER = ExperienceLedger(root)
        return _DEFAULT_LEDGER


def record_experience(**kwargs: Any) -> tuple[str, bool]:
    """Best-effort process-wide recording helper used by runtime chokepoints."""
    return get_default_ledger().record(**kwargs)


def _qualifies_for_training(event: Mapping[str, Any], threshold: float) -> bool:
    verdicts = event.get("verifier_verdicts") or []
    gate = event.get("gate_record") or {}
    if not verdicts or gate.get("allowed") is not True or event.get("output") in (None, ""):
        return False
    for verdict in verdicts:
        if verdict.get("passed") is False or float(verdict.get("score", 0.0)) < threshold:
            return False
    text = json.dumps(event.get("output"), ensure_ascii=False)
    return not any(pattern.search(text) for pattern in _PII_PATTERNS)


def compact_for_training(
    ledger: ExperienceLedger,
    output_dir: str | Path,
    *,
    verifier_threshold: float = 0.8,
) -> dict[str, Any]:
    """Promote qualified events across a manifest-hashed train/serve firewall."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    records: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    rejected = 0
    for event in ledger.iter_events():
        if not _qualifies_for_training(event, verifier_threshold):
            rejected += 1
            continue
        # Assign the split by INPUT identity, not event identity: every event
        # sharing a prompt must land on the same side, or the same input leaks
        # across the train/validation firewall (event_hash carries the per-event
        # timestamp+uuid and would scatter identical prompts on both sides).
        input_hash = str(event["inputs_hash"])
        split = "validation" if int(input_hash[:8], 16) % 10 == 0 else "train"
        records[split].append(
            {
                "trace_id": event["trace_id"],
                "source_hash": str(event["event_hash"]),
                "input_hash": input_hash,
                "output": event["output"],
                "verifier_verdicts": event["verifier_verdicts"],
            }
        )

    manifest_files: dict[str, dict[str, Any]] = {}
    for split, rows in records.items():
        path = output / f"experience-{split}-v{SCHEMA_VERSION}.jsonl"
        temporary = path.with_suffix(path.suffix + ".tmp")
        data = b"".join(_canonical_json(row) + b"\n" for row in rows)
        temporary.write_bytes(data)
        os.replace(temporary, path)
        manifest_files[split] = {
            "path": path.name,
            "records": len(rows),
            "sha256": hashlib.sha256(data).hexdigest(),
        }

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "verifier_threshold": verifier_threshold,
        "files": manifest_files,
        "promoted": sum(len(rows) for rows in records.values()),
        "rejected": rejected,
    }
    manifest["manifest_hash"] = content_hash(manifest)
    manifest_path = output / f"experience-manifest-v{SCHEMA_VERSION}.json"
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_bytes(_canonical_json(manifest) + b"\n")
    os.replace(temporary, manifest_path)
    return manifest
