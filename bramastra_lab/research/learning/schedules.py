"""Learning-rate schedules with explicit counters (B04).

Schedules are pure functions of the optimizer-update counter. The plasticity
controller (B06) may multiply the base schedule at optimizer boundaries; it
never edits tensors directly.
"""
from __future__ import annotations

import math
from typing import Callable

SCHEDULES = frozenset({"fixed", "warmup_linear"})


def make_schedule(name: str, *, base_lr: float, warmup_updates: int = 0) -> Callable[[int], float]:
    """Return lr(update_index) for update_index in [0, ...)."""
    if name not in SCHEDULES:
        raise ValueError(f"unknown schedule {name!r}; expected one of {sorted(SCHEDULES)}")
    if not math.isfinite(base_lr) or base_lr <= 0:
        raise ValueError("base_lr must be finite and positive")
    if not isinstance(warmup_updates, int) or isinstance(warmup_updates, bool) or warmup_updates < 0:
        raise ValueError("warmup_updates must be a nonnegative integer")
    if name == "fixed":
        return lambda update_index: base_lr
    if warmup_updates == 0:
        return lambda update_index: base_lr

    def warmup_linear(update_index: int) -> float:
        if update_index < warmup_updates:
            return base_lr * (update_index + 1) / warmup_updates
        return base_lr

    return warmup_linear
