from __future__ import annotations

import json
from pathlib import Path

import pytest

from runtime.evidence_stream import append_evidence, evidence_snapshot, read_evidence


def test_evidence_stream_hash_chain_and_signature(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    first = append_evidence(
        source="trainer",
        kind="checkpoint.saved",
        payload={"step": 100},
        run_id="run-1",
        path=path,
        signing_key="secret",
        require_signature=True,
    )
    second = append_evidence(
        source="trainer",
        kind="checkpoint.protected",
        payload={"step": 100},
        run_id="run-1",
        path=path,
        signing_key="secret",
        require_signature=True,
    )

    assert second["previous_event_sha256"] == first["event_sha256"]
    report = read_evidence(path, signing_key="secret")
    assert report["integrity"] == "valid"
    assert report["signed_events"] == 2
    assert all(event["verification"]["signature"] == "verified" for event in report["events"])


def test_evidence_stream_detects_payload_tampering(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    append_evidence(
        source="trainer",
        kind="metric",
        payload={"loss": 2.0},
        path=path,
    )
    event = json.loads(path.read_text(encoding="utf-8"))
    event["payload"]["loss"] = 0.1
    path.write_text(json.dumps(event) + "\n", encoding="utf-8")

    report = read_evidence(path)
    assert report["integrity"] == "invalid"
    assert report["invalid_events"] == 1
    assert report["events"][0]["verification"]["hash"] is False


def test_canonical_evidence_requires_a_signing_key(tmp_path: Path) -> None:
    with pytest.raises(PermissionError, match="ANRA_EVIDENCE_SIGNING_KEY"):
        append_evidence(
            source="promotion",
            kind="checkpoint.promoted",
            payload={},
            path=tmp_path / "events.jsonl",
            signing_key="",
            require_signature=True,
        )


def test_snapshot_is_the_consumer_safe_aggregate(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    append_evidence(source="matrix", kind="health", payload={"ok": True}, path=path)
    snapshot = evidence_snapshot(path)
    assert snapshot["total_events"] == 1
    assert snapshot["sources"] == {"matrix": 1}
    assert "events" not in snapshot

