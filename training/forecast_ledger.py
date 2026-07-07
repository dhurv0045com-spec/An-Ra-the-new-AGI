"""Pre-registered forecast ledger (MASTER_UPGRADE Part 1 / Layer 11).

Every pilot cell's predicted outcome is written down *before* its launch
manifest exists. Entries are append-only JSONL with a per-entry content hash
chained through ``prev_hash``, so late insertion, deletion, or reordering of
a "prediction" is detectable. A forecast registered after its manifest's
``created_at`` is a Gate-5 violation and voids the launch.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from collections.abc import Mapping
from pathlib import Path

from anra.anra_paths import OUTPUT_V2_DIR

SCHEMA_VERSION = 1
GENESIS_HASH = "genesis"
FORECAST_LEDGER = OUTPUT_V2_DIR / "forecast_ledger.jsonl"


class ForecastLedgerError(RuntimeError):
    """Raised when the ledger is malformed or its hash chain does not verify."""


class ForecastAuditError(PermissionError):
    """Raised when a launch fails the pre-registration timestamp audit."""


def _canonical(payload: Mapping[str, object]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _entry_hash(prev_hash: str, unsigned_entry: Mapping[str, object]) -> str:
    material = prev_hash.encode("utf-8") + _canonical(unsigned_entry)
    return hashlib.sha256(material).hexdigest()


def read_ledger(path: str | Path = FORECAST_LEDGER) -> list[dict[str, object]]:
    target = Path(path)
    if not target.exists():
        return []
    entries: list[dict[str, object]] = []
    for line_number, line in enumerate(
        target.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        entry = json.loads(line)
        if not isinstance(entry, dict):
            raise ForecastLedgerError(f"Ledger line {line_number} is not a JSON object")
        entries.append(entry)
    return entries


def verify_ledger(path: str | Path = FORECAST_LEDGER) -> dict[str, object]:
    """Recompute the full hash chain; any edit or reorder breaks it."""
    entries = read_ledger(path)
    prev_hash = GENESIS_HASH
    for index, entry in enumerate(entries):
        recorded = str(entry.get("entry_hash", ""))
        unsigned = {key: value for key, value in entry.items() if key != "entry_hash"}
        if str(entry.get("prev_hash", "")) != prev_hash:
            return {"valid": False, "entries": len(entries), "broken_at": index}
        if _entry_hash(prev_hash, unsigned) != recorded:
            return {"valid": False, "entries": len(entries), "broken_at": index}
        prev_hash = recorded
    return {"valid": True, "entries": len(entries), "head_hash": prev_hash}


def _append_entry(
    entry: dict[str, object], path: str | Path
) -> dict[str, object]:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    entries = read_ledger(target)
    prev_hash = str(entries[-1]["entry_hash"]) if entries else GENESIS_HASH
    entry = {**entry, "prev_hash": prev_hash}
    entry["entry_hash"] = _entry_hash(prev_hash, entry)
    with target.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(entry, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
    return entry


def register_forecast(
    *,
    cell_id: str,
    metric: str,
    predicted_low: float,
    predicted_high: float,
    gate: str,
    seeds: list[int],
    rationale: str = "",
    registered_by: str = "stream-a",
    path: str | Path = FORECAST_LEDGER,
) -> dict[str, object]:
    """Record a falsifiable prediction before the launch manifest is built."""
    if predicted_low > predicted_high:
        raise ValueError("predicted_low must not exceed predicted_high")
    if not cell_id or not metric or not gate:
        raise ValueError("Forecasts require a cell_id, metric, and gate statement")
    entry = {
        "schema_version": SCHEMA_VERSION,
        "kind": "forecast",
        "forecast_id": str(uuid.uuid4()),
        "cell_id": cell_id,
        "metric": metric,
        "predicted_low": float(predicted_low),
        "predicted_high": float(predicted_high),
        "gate": gate,
        "seeds": [int(seed) for seed in seeds],
        "rationale": rationale,
        "registered_by": registered_by,
        "registered_at": time.time(),
    }
    return _append_entry(entry, path)


def record_outcome(
    *,
    forecast_id: str,
    realized_value: float,
    verdict: str,
    evidence_path: str = "",
    path: str | Path = FORECAST_LEDGER,
) -> dict[str, object]:
    """Append the realized result for a registered forecast (calibration data)."""
    forecasts = {
        str(entry.get("forecast_id")): entry
        for entry in read_ledger(path)
        if entry.get("kind") == "forecast"
    }
    if forecast_id not in forecasts:
        raise ForecastLedgerError(f"Outcome references unknown forecast_id {forecast_id}")
    forecast = forecasts[forecast_id]
    in_range = (
        float(forecast["predicted_low"])
        <= float(realized_value)
        <= float(forecast["predicted_high"])
    )
    entry = {
        "schema_version": SCHEMA_VERSION,
        "kind": "outcome",
        "forecast_id": forecast_id,
        "cell_id": forecast["cell_id"],
        "metric": forecast["metric"],
        "realized_value": float(realized_value),
        "within_predicted_range": in_range,
        "verdict": verdict,
        "evidence_path": evidence_path,
        "recorded_at": time.time(),
    }
    return _append_entry(entry, path)


def forecasts_for_cell(
    cell_id: str, path: str | Path = FORECAST_LEDGER
) -> list[dict[str, object]]:
    return [
        entry
        for entry in read_ledger(path)
        if entry.get("kind") == "forecast" and entry.get("cell_id") == cell_id
    ]


def audit_pre_launch(
    manifest: Mapping[str, object], *, path: str | Path = FORECAST_LEDGER
) -> dict[str, object]:
    """Gate-5 timestamp audit: the forecast must predate the launch manifest.

    Raises :class:`ForecastAuditError` when the manifest carries no forecast,
    references an unknown one, or the forecast was registered after the
    manifest was created (a post-hoc prediction — the result is void).
    """
    chain = verify_ledger(path)
    if not chain["valid"]:
        raise ForecastAuditError(f"Forecast ledger hash chain is broken: {chain}")
    forecast_id = str(manifest.get("forecast_id", ""))
    cell_id = str(manifest.get("pilot_cell_id", ""))
    if not forecast_id:
        raise ForecastAuditError("Launch manifest carries no forecast_id")
    matches = [
        entry
        for entry in read_ledger(path)
        if entry.get("kind") == "forecast" and entry.get("forecast_id") == forecast_id
    ]
    if not matches:
        raise ForecastAuditError(f"No registered forecast with id {forecast_id}")
    forecast = matches[0]
    if cell_id and str(forecast.get("cell_id")) != cell_id:
        raise ForecastAuditError(
            f"Forecast {forecast_id} belongs to cell {forecast.get('cell_id')}, "
            f"manifest claims {cell_id}"
        )
    registered_at = float(forecast["registered_at"])
    created_at = float(manifest.get("created_at", 0.0))
    if not created_at:
        raise ForecastAuditError("Launch manifest has no created_at timestamp")
    if registered_at > created_at:
        raise ForecastAuditError(
            "Post-hoc forecast: registered_at "
            f"{registered_at} > manifest created_at {created_at} (Gate-5 violation)"
        )
    return {
        "audit": "pre_launch_forecast",
        "passed": True,
        "forecast_id": forecast_id,
        "cell_id": str(forecast.get("cell_id")),
        "registered_at": registered_at,
        "manifest_created_at": created_at,
        "lead_seconds": round(created_at - registered_at, 6),
        "chain_head": chain["head_hash"],
    }


def calibration_report(path: str | Path = FORECAST_LEDGER) -> dict[str, object]:
    """Predicted-vs-realized summary: the program practices DFC on itself."""
    entries = read_ledger(path)
    forecasts = [entry for entry in entries if entry.get("kind") == "forecast"]
    outcomes = [entry for entry in entries if entry.get("kind") == "outcome"]
    resolved = len(outcomes)
    hits = sum(1 for entry in outcomes if entry.get("within_predicted_range"))
    return {
        "forecasts_registered": len(forecasts),
        "outcomes_recorded": resolved,
        "within_range": hits,
        "hit_rate": round(hits / resolved, 6) if resolved else None,
        "unresolved": len(forecasts) - len({str(o["forecast_id"]) for o in outcomes}),
    }
