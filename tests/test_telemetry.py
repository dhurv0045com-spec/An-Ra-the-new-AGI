import json

import pytest

from engine import telemetry
from engine.telemetry import TelemetryBus, TelemetryRecord, trace


def _use_bus(tmp_path, monkeypatch):
    bus = TelemetryBus(tmp_path / "telemetry.jsonl")
    monkeypatch.setattr(telemetry, "_bus", bus)
    return bus


def test_trace_decorator_writes_jsonl(tmp_path, monkeypatch):
    bus = _use_bus(tmp_path, monkeypatch)

    @trace("demo", "call")
    def fn():
        return {"output": "hello", "tokens_used": 3}

    assert fn()["output"] == "hello"
    rows = bus.recent(1)
    assert rows[0]["module"] == "demo"
    assert rows[0]["operation"] == "call"
    assert rows[0]["success"] is True
    assert rows[0]["tokens_used"] == 3


def test_trace_captures_failure(tmp_path, monkeypatch):
    bus = _use_bus(tmp_path, monkeypatch)

    @trace("demo", "fail")
    def fn():
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        fn()
    row = bus.recent(1)[0]
    assert row["success"] is False
    assert "RuntimeError: boom" == row["error"]


def test_telemetry_bus_summary_by_module(tmp_path):
    bus = TelemetryBus(tmp_path / "telemetry.jsonl")
    bus.record(TelemetryRecord("a", "x", 1.0, 1.1, 10.0, True))
    bus.record(TelemetryRecord("a", "x", 1.0, 1.2, 30.0, False, error="bad"))
    bus.record(TelemetryRecord("b", "x", 1.0, 1.1, 5.0, True))

    summary = bus.summary_by_module()
    assert summary["a"]["calls_total"] == 2
    assert summary["a"]["success_rate"] == 0.5
    assert summary["a"]["avg_latency_ms"] == 20.0
    assert summary["a"]["error_count"] == 1
    assert summary["b"]["calls_total"] == 1


def test_telemetry_recent_returns_correct_count(tmp_path):
    bus = TelemetryBus(tmp_path / "telemetry.jsonl")
    for i in range(5):
        bus.record(TelemetryRecord("m", str(i), 1.0, 1.1, 1.0, True))

    rows = bus.recent(3)
    assert len(rows) == 3
    assert [json.loads(json.dumps(row))["operation"] for row in rows] == ["2", "3", "4"]
