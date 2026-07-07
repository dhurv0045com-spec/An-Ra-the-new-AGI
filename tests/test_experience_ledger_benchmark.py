from __future__ import annotations

from pathlib import Path

from scripts.benchmark_experience_ledger import (
    run_crash_flush_stress,
    run_write_benchmark,
)


def test_experience_ledger_write_benchmark_validates_shards(tmp_path: Path) -> None:
    result = run_write_benchmark(tmp_path / "bench", events=50, max_shard_bytes=512)
    assert result["validated_events"] == 50
    assert result["write_failures"] == 0
    assert result["seal_verified"] is True
    assert result["p50_ms"] < 10.0


def test_experience_ledger_crash_flush_stress_keeps_valid_jsonl(tmp_path: Path) -> None:
    result = run_crash_flush_stress(
        tmp_path / "stress",
        workers=2,
        events_per_worker=20,
        terminate_one=False,
    )
    assert result["validated_events"] == 40
    assert result["seal_verified"] is True
