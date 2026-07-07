from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from training import forecast_ledger


def _register(path: Path, cell_id: str = "p150-muon") -> dict[str, object]:
    return forecast_ledger.register_forecast(
        cell_id=cell_id,
        metric="token_efficiency_x",
        predicted_low=1.3,
        predicted_high=1.6,
        gate="adopt if >=1.2x vs baseline",
        seeds=[1301, 2602, 3903],
        rationale="honest literature range",
        path=path,
    )


def test_register_appends_hash_chained_entries(tmp_path: Path) -> None:
    ledger = tmp_path / "forecasts.jsonl"
    first = _register(ledger)
    second = _register(ledger, cell_id="p150-moe")

    assert first["prev_hash"] == forecast_ledger.GENESIS_HASH
    assert second["prev_hash"] == first["entry_hash"]
    chain = forecast_ledger.verify_ledger(ledger)
    assert chain["valid"] is True
    assert chain["entries"] == 2


def test_tampered_entry_breaks_the_chain(tmp_path: Path) -> None:
    ledger = tmp_path / "forecasts.jsonl"
    _register(ledger)
    _register(ledger, cell_id="p150-moe")

    lines = ledger.read_text(encoding="utf-8").splitlines()
    entry = json.loads(lines[0])
    entry["predicted_high"] = 9.9
    lines[0] = json.dumps(entry, sort_keys=True, separators=(",", ":"))
    ledger.write_text("\n".join(lines) + "\n", encoding="utf-8")

    chain = forecast_ledger.verify_ledger(ledger)
    assert chain["valid"] is False
    assert chain["broken_at"] == 0


def test_register_rejects_inverted_range_and_empty_fields(tmp_path: Path) -> None:
    ledger = tmp_path / "forecasts.jsonl"
    with pytest.raises(ValueError, match="predicted_low"):
        forecast_ledger.register_forecast(
            cell_id="x",
            metric="m",
            predicted_low=2.0,
            predicted_high=1.0,
            gate="g",
            seeds=[1],
            path=ledger,
        )
    with pytest.raises(ValueError, match="cell_id"):
        forecast_ledger.register_forecast(
            cell_id="",
            metric="m",
            predicted_low=1.0,
            predicted_high=2.0,
            gate="g",
            seeds=[1],
            path=ledger,
        )


def test_pre_launch_audit_passes_when_forecast_predates_manifest(tmp_path: Path) -> None:
    ledger = tmp_path / "forecasts.jsonl"
    forecast = _register(ledger)
    manifest = {
        "forecast_id": forecast["forecast_id"],
        "pilot_cell_id": "p150-muon",
        "created_at": time.time() + 1.0,
    }
    audit = forecast_ledger.audit_pre_launch(manifest, path=ledger)
    assert audit["passed"] is True
    assert audit["lead_seconds"] > 0


def test_pre_launch_audit_rejects_post_hoc_forecast(tmp_path: Path) -> None:
    ledger = tmp_path / "forecasts.jsonl"
    forecast = _register(ledger)
    manifest = {
        "forecast_id": forecast["forecast_id"],
        "pilot_cell_id": "p150-muon",
        "created_at": float(forecast["registered_at"]) - 60.0,
    }
    with pytest.raises(forecast_ledger.ForecastAuditError, match="Post-hoc"):
        forecast_ledger.audit_pre_launch(manifest, path=ledger)


def test_pre_launch_audit_rejects_missing_or_mismatched_forecast(tmp_path: Path) -> None:
    ledger = tmp_path / "forecasts.jsonl"
    forecast = _register(ledger)
    with pytest.raises(forecast_ledger.ForecastAuditError, match="no forecast_id"):
        forecast_ledger.audit_pre_launch({"created_at": time.time()}, path=ledger)
    with pytest.raises(forecast_ledger.ForecastAuditError, match="No registered forecast"):
        forecast_ledger.audit_pre_launch(
            {"forecast_id": "nope", "created_at": time.time()}, path=ledger
        )
    with pytest.raises(forecast_ledger.ForecastAuditError, match="belongs to cell"):
        forecast_ledger.audit_pre_launch(
            {
                "forecast_id": forecast["forecast_id"],
                "pilot_cell_id": "some-other-cell",
                "created_at": time.time() + 1.0,
            },
            path=ledger,
        )


def test_outcomes_feed_the_calibration_report(tmp_path: Path) -> None:
    ledger = tmp_path / "forecasts.jsonl"
    hit = _register(ledger)
    miss = _register(ledger, cell_id="p150-moe")
    forecast_ledger.record_outcome(
        forecast_id=str(hit["forecast_id"]),
        realized_value=1.45,
        verdict="adopted",
        path=ledger,
    )
    forecast_ledger.record_outcome(
        forecast_id=str(miss["forecast_id"]),
        realized_value=1.05,
        verdict="rejected",
        path=ledger,
    )

    report = forecast_ledger.calibration_report(ledger)
    assert report["forecasts_registered"] == 2
    assert report["outcomes_recorded"] == 2
    assert report["within_range"] == 1
    assert report["hit_rate"] == 0.5
    assert forecast_ledger.verify_ledger(ledger)["valid"] is True

    with pytest.raises(forecast_ledger.ForecastLedgerError):
        forecast_ledger.record_outcome(
            forecast_id="unknown",
            realized_value=1.0,
            verdict="adopted",
            path=ledger,
        )
