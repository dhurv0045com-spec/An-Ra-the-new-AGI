from __future__ import annotations

from dataclasses import asdict
import json
import time
from pathlib import Path
from typing import Any

from anra_paths import HAL_AUDIT_LOG, HAL_STATE_FILE


def _state_payload(hal: Any, *, source: str = "runtime") -> dict[str, Any]:
    state = getattr(hal, "state", hal)
    hormones = state.hormones() if hasattr(state, "hormones") else {}
    counters = {
        "rolling_verifier_mean": float(getattr(state, "rolling_verifier_mean", 0.0)),
        "consecutive_failures": int(getattr(state, "consecutive_failures", 0)),
        "consecutive_high_reward_outputs": int(getattr(state, "consecutive_high_reward_outputs", 0)),
        "cooperative_session_turn_count": int(getattr(state, "cooperative_session_turn_count", 0)),
    }
    return {
        "source": source,
        "updated_at": time.time(),
        "hormones": {k: float(v) for k, v in hormones.items()},
        "counters": counters,
        "esv": hal.hal_to_esv() if hasattr(hal, "hal_to_esv") else {},
        "raw_state": asdict(state) if hasattr(state, "__dataclass_fields__") else {},
    }


def publish_hal_state(hal: Any, *, source: str = "runtime", path: str | Path = HAL_STATE_FILE) -> dict[str, Any]:
    payload = _state_payload(hal, source=source)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    audit_hal_anomalies(payload, source=source)
    return payload


def read_hal_state(path: str | Path = HAL_STATE_FILE) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {
            "source": "uninitialized",
            "updated_at": None,
            "hormones": {},
            "counters": {},
            "esv": {},
            "anomalies": [],
        }
    payload = json.loads(p.read_text(encoding="utf-8"))
    payload["anomalies"] = detect_hal_anomalies(payload)
    return payload


def detect_hal_anomalies(payload: dict[str, Any]) -> list[dict[str, Any]]:
    hormones = payload.get("hormones", {}) if isinstance(payload, dict) else {}
    anomalies: list[dict[str, Any]] = []
    cortisol = float(hormones.get("cortisol", 0.0) or 0.0)
    adrenaline = float(hormones.get("adrenaline", 0.0) or 0.0)
    if cortisol > 0.8:
        anomalies.append({"type": "sustained_cortisol", "severity": "warning", "value": cortisol})
    if adrenaline > 0.75:
        anomalies.append({"type": "adrenaline_spike", "severity": "warning", "value": adrenaline})
    return anomalies


def audit_hal_anomalies(
    payload_or_hal: Any,
    *,
    source: str = "runtime",
    path: str | Path = HAL_AUDIT_LOG,
) -> list[dict[str, Any]]:
    payload = payload_or_hal if isinstance(payload_or_hal, dict) else _state_payload(payload_or_hal, source=source)
    events = []
    for anomaly in detect_hal_anomalies(payload):
        event = {
            "ts": time.time(),
            "event_type": "HAL_ANOMALY",
            "component": "hal",
            "source": source,
            "action": anomaly["type"],
            "details": {"anomaly": anomaly, "state": payload},
        }
        events.append(event)
        try:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event, sort_keys=True) + "\n")
        except Exception:
            pass
        try:
            from sovereignty.logger import audit_event

            audit_event("hal", "HAL_ANOMALY", anomaly["type"], details=event["details"])
        except Exception:
            pass
    return events

