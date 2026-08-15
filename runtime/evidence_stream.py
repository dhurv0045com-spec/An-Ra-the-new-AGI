"""One append-only evidence stream for An-Ra operators and evaluators.

The stream is intentionally small and dependency-free.  Matrix reads it for
operational truth and ThirdEye attaches the same verified snapshot to its
evaluation reports.  Events are content-hashed and hash-chained; canonical
training/promotion callers can additionally require an HMAC signature.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import threading
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

EVIDENCE_SCHEMA = "anra-evidence-event/v1"
DEFAULT_EVIDENCE_PATH = Path(__file__).resolve().parents[1] / "state" / "evidence" / "events.jsonl"
_LOCK = threading.RLock()


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _signing_key(explicit: str | None = None) -> str:
    return explicit if explicit is not None else os.environ.get("ANRA_EVIDENCE_SIGNING_KEY", "")


def _last_event_hash(path: Path) -> str:
    if not path.is_file():
        return ""
    last = ""
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            if line.strip():
                last = line
    if not last:
        return ""
    try:
        payload = json.loads(last)
    except json.JSONDecodeError:
        return ""
    return str(payload.get("event_sha256", "")) if isinstance(payload, dict) else ""


def append_evidence(
    *,
    source: str,
    kind: str,
    payload: dict[str, Any],
    run_id: str = "",
    artifact_refs: list[dict[str, str]] | None = None,
    path: str | Path = DEFAULT_EVIDENCE_PATH,
    signing_key: str | None = None,
    require_signature: bool = False,
) -> dict[str, Any]:
    """Append one immutable event after hashing and optionally signing it."""

    if not source.strip() or not kind.strip():
        raise ValueError("Evidence events require non-empty source and kind")
    key = _signing_key(signing_key)
    if require_signature and not key:
        raise PermissionError("ANRA_EVIDENCE_SIGNING_KEY is required for canonical evidence")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK:
        body: dict[str, Any] = {
            "schema": EVIDENCE_SCHEMA,
            "event_id": str(uuid.uuid4()),
            "occurred_at": time.time(),
            "run_id": str(run_id),
            "source": source.strip(),
            "kind": kind.strip(),
            "payload": dict(payload),
            "artifact_refs": list(artifact_refs or []),
            "previous_event_sha256": _last_event_hash(target),
        }
        event_hash = _sha256(body)
        event = {
            **body,
            "event_sha256": event_hash,
            "signature": (
                {
                    "algorithm": "hmac-sha256",
                    "key_id": os.environ.get("ANRA_EVIDENCE_KEY_ID", "owner"),
                    "value": hmac.new(
                        key.encode("utf-8"), event_hash.encode("ascii"), hashlib.sha256
                    ).hexdigest(),
                }
                if key
                else None
            ),
        }
        line = json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n"
        with target.open("a", encoding="utf-8") as stream:
            stream.write(line)
            stream.flush()
            os.fsync(stream.fileno())
    return event


def read_evidence(
    path: str | Path = DEFAULT_EVIDENCE_PATH,
    *,
    limit: int | None = None,
    signing_key: str | None = None,
) -> dict[str, Any]:
    """Read and verify the hash chain; never upgrade unsigned events to signed."""

    target = Path(path)
    if not target.is_file():
        return {
            "schema": "anra-evidence-stream/v1",
            "events": [],
            "integrity": "empty",
            "signed_events": 0,
            "unsigned_events": 0,
            "invalid_events": 0,
        }
    key = _signing_key(signing_key)
    events: list[dict[str, Any]] = []
    invalid = 0
    signed = 0
    unsigned = 0
    expected_previous = ""
    for raw_line in target.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            event = json.loads(raw_line)
        except json.JSONDecodeError:
            invalid += 1
            continue
        if not isinstance(event, dict) or event.get("schema") != EVIDENCE_SCHEMA:
            invalid += 1
            continue
        event_hash = str(event.get("event_sha256", ""))
        body = {
            name: value
            for name, value in event.items()
            if name not in {"event_sha256", "signature"}
        }
        valid_hash = hmac.compare_digest(event_hash, _sha256(body))
        valid_chain = hmac.compare_digest(
            str(event.get("previous_event_sha256", "")), expected_previous
        )
        signature = event.get("signature")
        signature_state = "unsigned"
        valid_signature = True
        if isinstance(signature, dict):
            signed += 1
            if key:
                expected_signature = hmac.new(
                    key.encode("utf-8"), event_hash.encode("ascii"), hashlib.sha256
                ).hexdigest()
                valid_signature = hmac.compare_digest(
                    str(signature.get("value", "")), expected_signature
                )
                signature_state = "verified" if valid_signature else "invalid"
            else:
                signature_state = "unverified_key_unavailable"
        else:
            unsigned += 1
        valid = valid_hash and valid_chain and valid_signature
        if not valid:
            invalid += 1
        event["verification"] = {
            "valid": valid,
            "hash": valid_hash,
            "chain": valid_chain,
            "signature": signature_state,
        }
        events.append(event)
        expected_previous = event_hash
    if limit is None:
        visible = events
    else:
        requested = max(0, int(limit))
        visible = events[-requested:] if requested else []
    return {
        "schema": "anra-evidence-stream/v1",
        "events": visible,
        "integrity": "valid" if events and invalid == 0 else "invalid" if invalid else "empty",
        "signed_events": signed,
        "unsigned_events": unsigned,
        "invalid_events": invalid,
        "total_events": len(events),
    }


def evidence_snapshot(path: str | Path = DEFAULT_EVIDENCE_PATH) -> dict[str, Any]:
    report = read_evidence(path)
    events = report.pop("events")
    sources = Counter(str(event.get("source", "unknown")) for event in events)
    kinds = Counter(str(event.get("kind", "unknown")) for event in events)
    return {
        **report,
        "path": str(Path(path)),
        "sources": dict(sorted(sources.items())),
        "kinds": dict(sorted(kinds.items())),
        "latest_event_sha256": (
            str(events[-1].get("event_sha256", "")) if events else ""
        ),
    }
