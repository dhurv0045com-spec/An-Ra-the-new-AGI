"""ARK-021 core — probe scheduling and preserved/reconstructed classification (pure)."""
from __future__ import annotations
from typing import Any, Mapping

PROBE_EVERY = 200
PRESERVED_THRESHOLD = 0.85
RECONSTRUCTED_THRESHOLD = 0.30
RECOVERY_LATENCY_LIMIT = 100


def probe_steps(horizon: int, every: int = PROBE_EVERY) -> list[int]:
    return list(range(every, horizon + 1, every))


def classify(probe_robust_min: list[float], recovery_latency_updates: float | None) -> str:
    """PRESERVED / RECONSTRUCTED / MIXED from probe-time A accuracy (replay withheld)."""
    if not probe_robust_min:
        return "INSUFFICIENT_PROBES"
    mean_acc = sum(probe_robust_min) / len(probe_robust_min)
    if mean_acc >= PRESERVED_THRESHOLD:
        return "PRESERVED"
    if mean_acc < RECONSTRUCTED_THRESHOLD:
        if recovery_latency_updates is not None and recovery_latency_updates <= RECOVERY_LATENCY_LIMIT:
            return "RECONSTRUCTED"
        return "RECONSTRUCTED_SLOW_OR_UNSPECIFIED"
    return "MIXED"
