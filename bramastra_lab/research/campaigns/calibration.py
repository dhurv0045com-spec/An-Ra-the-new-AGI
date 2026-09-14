"""E0 calibration and frozen protocol (O03).

E0 measures pilot throughput on the actual K8 configuration and heaviest
active arms; this adapter turns those measurements into frozen targets
*before* learning outcomes can influence them. The frozen artifact binds the
selected settings to source/data/device identities; the runner consumes it
instead of hardcoded minima.

Selection rules (campaign.json / experiment.md):
- update target: N = min(4000, floor(0.75 * slot_seconds /
  worst_measured_update_seconds)), with informative minima E1=200, E3/E4=80.
  A profile that cannot meet the minima stops before E1 (refusal, not a
  smaller silent target).
- confirmation inventory: 128, 64 or 32 clusters per family, largest
  predicted to fit the E2 allowance with 20% reserve; below 32 is
  EVALUATION_BUDGET_INSUFFICIENT.
"""
from __future__ import annotations

import json
import math
import os
import time
from typing import Any, Mapping

CALIBRATION_SCHEMA = "bramastra-k8-calibration/v1"
PROTOCOL_SCHEMA = "bramastra-k8-frozen-protocol/v1"

MINIMUM_UPDATES = {"E1": 200, "E3": 80, "E4": 80}
MAX_UPDATES = 4000
UPDATE_BUDGET_FRACTION = 0.75
CONFIRMATION_CANDIDATES = (128, 64, 32)
MIN_CONFIRMATION_CLUSTERS = 32
EVAL_RESERVE_FRACTION = 0.20

# Phase wall allowances in seconds (campaign plan): E1 120min, E3/E4 60min
# per arm-pair slot structure uses 60min E1 slots and 30min E3/E4 slots.
SLOT_SECONDS = {"E1": 60 * 60.0, "E3": 30 * 60.0, "E4": 30 * 60.0}
E2_ALLOWANCE_SECONDS = 45 * 60.0


class CalibrationError(ValueError):
    """Calibration input is missing, inconsistent or insufficient."""


def select_update_target(*, phase: str, worst_update_seconds: float,
                         slot_seconds: float | None = None) -> int:
    """Apply N = min(4000, floor(0.75*slot/worst)) with informative minima."""
    if phase not in MINIMUM_UPDATES:
        raise CalibrationError(f"no calibrated target rule for phase {phase!r}")
    if not isinstance(worst_update_seconds, (int, float)) \
            or not math.isfinite(worst_update_seconds) \
            or worst_update_seconds <= 0:
        raise CalibrationError("worst_update_seconds must be positive finite")
    slot = slot_seconds if slot_seconds is not None else SLOT_SECONDS[phase]
    if not isinstance(slot, (int, float)) or slot <= 0:
        raise CalibrationError("slot_seconds must be positive")
    candidate = min(MAX_UPDATES,
                    math.floor(UPDATE_BUDGET_FRACTION * slot / worst_update_seconds))
    minimum = MINIMUM_UPDATES[phase]
    if candidate < minimum:
        raise CalibrationError(
            f"phase {phase}: calibrated target {candidate} below informative "
            f"minimum {minimum} (worst update {worst_update_seconds:.2f}s); "
            "stop before E1 with a measured profile report")
    return int(candidate)


def select_confirmation_inventory(*, eval_cases_per_second: float,
                                  families: int = 3) -> int:
    """Largest 128/64/32-per-family inventory fitting E2 with 20% reserve."""
    if not isinstance(eval_cases_per_second, (int, float)) \
            or not math.isfinite(eval_cases_per_second) \
            or eval_cases_per_second <= 0:
        raise CalibrationError("eval_cases_per_second must be positive finite")
    usable = E2_ALLOWANCE_SECONDS * (1.0 - EVAL_RESERVE_FRACTION)
    for clusters in CONFIRMATION_CANDIDATES:
        # E2 evaluates every confirmation case under several modes; the
        # fitted quantity is total evaluated cases across families.
        total_cases = clusters * families
        if total_cases / eval_cases_per_second <= usable:
            return clusters
    raise CalibrationError(
        f"EVALUATION_BUDGET_INSUFFICIENT: even {MIN_CONFIRMATION_CLUSTERS} "
        "clusters per family do not fit the E2 allowance with 20% reserve")


def freeze_protocol(run_dir: str, *, selection: Mapping[str, Any],
                    identities: Mapping[str, Any]) -> dict[str, Any]:
    """Persist the frozen protocol artifact (once; never overwritten)."""
    for name in ("source_hash", "data_hash", "device_ids"):
        if not identities.get(name):
            raise CalibrationError(f"frozen protocol requires {name}")
    if not selection.get("update_targets") or not selection.get(
            "confirmation_clusters_per_family"):
        raise CalibrationError("selection carries no targets/inventory")
    path = os.path.join(run_dir, "frozen_protocol.json")
    if os.path.exists(path):
        raise CalibrationError(
            f"frozen protocol already exists at {path}; refusing overwrite "
            "(resume consumes the frozen artifact)")
    protocol = {"schema": PROTOCOL_SCHEMA,
                "created_unix": time.time(),
                "selection": dict(selection),
                "identities": {key: identities[key]
                               for key in ("source_hash", "data_hash",
                                           "device_ids")},
                "calibration_source": "e0-measured"}
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(protocol, handle, indent=2, sort_keys=True)
    return protocol


def load_frozen_protocol(run_dir: str) -> dict[str, Any] | None:
    """Load the frozen artifact, or None when E0 has not frozen one yet."""
    path = os.path.join(run_dir, "frozen_protocol.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as handle:
        protocol = json.load(handle)
    if protocol.get("schema") != PROTOCOL_SCHEMA:
        raise CalibrationError(
            f"unsupported frozen protocol schema {protocol.get('schema')!r}")
    return protocol


def check_hardware_match(protocol: Mapping[str, Any], *,
                         device_ids: list[str]) -> None:
    """Refuse a frozen protocol on changed hardware (recalibrate, don't run)."""
    recorded = (protocol.get("identities") or {}).get("device_ids")
    if sorted(recorded or []) != sorted(device_ids):
        raise CalibrationError(
            f"frozen protocol devices {recorded} != live {device_ids}; "
            "interrupted-E0/changed-hardware requires recalibration")
