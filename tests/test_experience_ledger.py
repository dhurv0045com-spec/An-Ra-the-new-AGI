from __future__ import annotations

import json
from pathlib import Path
from datetime import UTC, datetime, timedelta

import pytest

import app
import runtime.experience_ledger as experience_ledger
from runtime.experience_ledger import (
    ExperienceEvent,
    ExperienceLedger,
    compact_for_training,
    content_hash,
    verify_envelope,
)
from training.verifier import VerifierHierarchy


def test_append_and_replay_preserve_verdict(tmp_path: Path) -> None:
    ledger = ExperienceLedger(tmp_path, strict=True)
    trace_id, written = ledger.record(
        trace_id="trace-1",
        kind="verifier",
        inputs={"claim": "2+2"},
        output="4",
        verifier_verdicts=[{"name": "math", "score": 1.0, "passed": True}],
        gate_record={"allowed": True, "gate": "sovereignty"},
        tokens={"output": 1},
        latency={"total_ms": 0.5},
    )
    assert written is True
    replay = ledger.replay(trace_id)
    assert len(replay) == 1
    assert replay[0]["verifier_verdicts"][0]["score"] == 1.0
    assert verify_envelope(replay[0])


def test_tampering_is_detected(tmp_path: Path) -> None:
    event = ExperienceEvent("t", "chat", content_hash("input"), output="original")
    envelope = event.envelope()
    envelope["output"] = "tampered"
    assert verify_envelope(envelope) is False

    ledger = ExperienceLedger(tmp_path, strict=True)
    path = ledger.active_shard
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(envelope) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid experience event"):
        list(ledger.iter_events())


def test_write_failure_is_fail_open(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ledger = ExperienceLedger(tmp_path)

    def explode(*_args: object, **_kwargs: object) -> None:
        raise OSError("disk unavailable")

    monkeypatch.setattr(Path, "open", explode)
    _, written = ledger.record(kind="chat", inputs="hello", output="world")
    assert written is False
    assert ledger.write_failures == 1


def test_compactor_promotes_only_verified_gated_non_pii(tmp_path: Path) -> None:
    ledger = ExperienceLedger(tmp_path / "ledger", strict=True)
    for trace_id, output, score, allowed in (
        ("good", "A useful verified answer", 0.95, True),
        ("weak", "An unverified answer", 0.2, True),
        ("denied", "A denied answer", 1.0, False),
        ("pii", "Contact person@example.com", 1.0, True),
    ):
        ledger.record(
            trace_id=trace_id,
            kind="chat",
            inputs={"trace": trace_id},
            output=output,
            verifier_verdicts=[{"name": "quality", "score": score, "passed": score >= 0.8}],
            gate_record={"allowed": allowed},
        )

    manifest = compact_for_training(ledger, tmp_path / "training")
    assert manifest["promoted"] == 1
    assert manifest["rejected"] == 3
    assert len(manifest["manifest_hash"]) == 64
    total = 0
    for details in manifest["files"].values():
        path = tmp_path / "training" / details["path"]
        assert content_hash([]) != details["sha256"]
        total += details["records"]
    assert total == 1


def test_compaction_firewall_never_splits_one_input_across_train_and_validation(
    tmp_path: Path,
) -> None:
    # The same qualifying prompt, recorded many times, must never appear on both
    # sides of the train/validation firewall.
    ledger = ExperienceLedger(tmp_path, strict=True)
    for index in range(40):
        ledger.record(
            kind="verifier",
            inputs={"prompt": "shared identical prompt"},
            output=f"answer-{index}",
            verifier_verdicts=[{"name": "math", "score": 1.0, "passed": True}],
            gate_record={"allowed": True},
        )
    manifest = compact_for_training(ledger, tmp_path / "training")
    assert manifest["promoted"] == 40
    input_hash = content_hash({"prompt": "shared identical prompt"})
    splits_seen = set()
    for split, details in manifest["files"].items():
        rows = [
            json.loads(line)
            for line in (tmp_path / "training" / details["path"])
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        if any(row["input_hash"] == input_hash for row in rows):
            splits_seen.add(split)
    assert splits_seen == {"validation" if int(input_hash[:8], 16) % 10 == 0 else "train"}


def test_input_payload_is_stored_only_as_hash(tmp_path: Path) -> None:
    ledger = ExperienceLedger(tmp_path, strict=True)
    secret = "private prompt text"
    ledger.record(kind="chat", inputs={"prompt": secret}, output="ok")
    raw = ledger.active_shard.read_text(encoding="utf-8")
    assert secret not in raw
    assert content_hash({"prompt": secret}) in raw


def test_shard_rotation_sealing_and_manifest_verification(tmp_path: Path) -> None:
    ledger = ExperienceLedger(tmp_path, strict=True, max_shard_bytes=1)
    for index in range(3):
        ledger.record(kind="chat", inputs={"i": index}, output=f"ok-{index}")

    shards = sorted((tmp_path / "shards").glob("experience-v*.jsonl"))
    assert len(shards) >= 2
    manifest = ledger.seal_shards(include_active=True)
    assert manifest["shards"]
    verification = ledger.verify_sealed_manifest()
    assert verification["verified"] is True
    assert verification["shards"] == len(manifest["shards"])


def test_sealed_manifest_detects_shard_tampering(tmp_path: Path) -> None:
    ledger = ExperienceLedger(tmp_path, strict=True)
    ledger.record(kind="chat", inputs="hello", output="world")
    manifest = ledger.seal_shards(include_active=True)
    shard = tmp_path / manifest["shards"][0]["path"]
    with shard.open("ab") as stream:
        stream.write(b"\n")
    with pytest.raises(ValueError, match="sealed shard size mismatch"):
        ledger.verify_sealed_manifest()


def test_retention_prunes_only_verified_old_sealed_shards(tmp_path: Path) -> None:
    old_ts = (datetime.now(UTC) - timedelta(days=10)).isoformat()
    ledger = ExperienceLedger(tmp_path, strict=True, retention_days=1)
    event = ExperienceEvent(
        trace_id="old",
        kind="chat",
        inputs_hash=content_hash("old"),
        output="old",
        ts=old_ts,
    )
    assert ledger.append(event) is True
    manifest = ledger.seal_shards(include_active=True)
    shard = tmp_path / manifest["shards"][0]["path"]
    assert shard.exists()
    removed = ledger.apply_retention()
    assert shard in removed
    assert not shard.exists()


def test_live_trace_chokepoint_records_chat(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ledger = ExperienceLedger(tmp_path, strict=True)
    monkeypatch.setattr(experience_ledger, "_DEFAULT_LEDGER", ledger)
    trace_id = app._record_trace(
        {
            "request_id": "request-1",
            "session_id": "session-1",
            "formatted_prompt": "hello",
            "generation": {
                "output": "hello back",
                "quality_state": "accepted",
                "tokens_generated": 2,
                "time_ms": 3.0,
            },
        }
    )
    replay = ledger.replay(trace_id)
    assert replay[0]["kind"] == "chat"
    assert replay[0]["output"] == "hello back"
    assert replay[0]["gate_record"]["allowed"] is True


def test_verifier_chokepoint_records_outcome(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ledger = ExperienceLedger(tmp_path / "ledger", strict=True)
    monkeypatch.setattr(experience_ledger, "_DEFAULT_LEDGER", ledger)
    verifier = VerifierHierarchy(tmp_path / "workspace")
    result = verifier.score("math", expression="2 + 2", expected="4")
    assert result.score == 1.0
    events = list(ledger.iter_events())
    assert events[0]["kind"] == "verifier"
    assert events[0]["verifier_verdicts"][0]["passed"] is True
