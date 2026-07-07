from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

import runtime.experience_ledger as experience_ledger
from memory.memory_router import MemoryRouter
from runtime.experience_ledger import ExperienceLedger


def test_memory_lifecycle_events_share_trace_and_omit_raw_content(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger = ExperienceLedger(tmp_path / "ledger", strict=True)
    monkeypatch.setattr(experience_ledger, "_DEFAULT_LEDGER", ledger)
    router = MemoryRouter(dim=16, faiss_index_path=tmp_path / "episodic.faiss")
    trace_id = "trace-memory-1"
    original = "memory lifecycle secret alpha"
    replacement = "memory lifecycle public beta"

    stored = router.write(
        original,
        metadata={"type": "fact", "salience": 1.0},
        trace_id=trace_id,
    )
    recalled = router.read(original, n=3, tier="hybrid", trace_id=trace_id)
    edited = router.edit(
        stored.record_id,
        replacement,
        metadata={"type": "fact", "salience": 1.0},
        trace_id=trace_id,
    )

    assert recalled
    assert edited is not None
    assert edited.record_id != stored.record_id
    assert all(
        row["record_id"] != stored.record_id
        for row in router.read(original, n=5, tier="hybrid", trace_id=trace_id)
    )
    assert router.forget(edited.record_id, trace_id=trace_id) is True

    replay = ledger.replay(trace_id)
    kinds = [event["kind"] for event in replay]
    assert {"memory_write", "memory_recall", "memory_forget", "memory_edit"} <= set(kinds)
    assert all(event["trace_id"] == trace_id for event in replay)

    raw = ledger.active_shard.read_text(encoding="utf-8")
    assert original not in raw
    assert replacement not in raw
    assert hashlib.sha256(original.encode()).hexdigest() in raw
    assert hashlib.sha256(replacement.encode()).hexdigest() in raw
