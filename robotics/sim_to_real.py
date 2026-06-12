"""Explicit simulation-to-real promotion ladder."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SimToRealDecision:
    next_mode: str
    allowed: bool
    reason: str


def decide_next_mode(
    *,
    current_mode: str,
    randomized_sim_success: float,
    shadow_anomaly_rate: float = 1.0,
    supervised_success: float = 0.0,
) -> SimToRealDecision:
    if current_mode == "simulation":
        passed = randomized_sim_success >= 0.80
        return SimToRealDecision("shadow", passed, "simulation gate" if passed else "need >=80% randomized simulation success")
    if current_mode == "shadow":
        passed = shadow_anomaly_rate <= 0.05
        return SimToRealDecision("supervised", passed, "shadow gate" if passed else "shadow anomaly rate too high")
    if current_mode == "supervised":
        passed = supervised_success >= 0.90
        return SimToRealDecision("promoted", passed, "supervised gate" if passed else "need >=90% supervised success")
    return SimToRealDecision(current_mode, False, "unknown or terminal mode")
