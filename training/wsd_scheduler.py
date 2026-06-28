"""Warmup-Stable-Decay scheduler with explicit phase reporting."""

from __future__ import annotations

from dataclasses import dataclass

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


@dataclass(frozen=True)
class WSDPhase:
    name: str
    step: int
    annealing_started: bool


def wsd_multiplier(
    step: int,
    *,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1,
    decay_fraction: float = 0.1,
) -> float:
    if total_steps <= warmup_steps:
        raise ValueError("total_steps must be greater than warmup_steps")
    decay_start = max(warmup_steps, int(total_steps * (1.0 - decay_fraction)))
    if step < warmup_steps:
        return float(step) / max(1, warmup_steps)
    if step < decay_start:
        return 1.0
    progress = min(1.0, (step - decay_start) / max(1, total_steps - decay_start))
    return max(min_lr_ratio, 1.0 - progress * (1.0 - min_lr_ratio))


def phase_for_step(
    step: int, *, warmup_steps: int, total_steps: int, decay_fraction: float = 0.1
) -> WSDPhase:
    decay_start = max(warmup_steps, int(total_steps * (1.0 - decay_fraction)))
    if step < warmup_steps:
        return WSDPhase("warmup", step, False)
    if step < decay_start:
        return WSDPhase("stable", step, False)
    return WSDPhase("decay", step, True)


def get_wsd_schedule(
    optimizer: Optimizer,
    *,
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float = 0.1,
    decay_fraction: float = 0.1,
    last_epoch: int = -1,
) -> LambdaLR:
    return LambdaLR(
        optimizer,
        lr_lambda=lambda step: wsd_multiplier(
            step,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr_ratio=min_lr_ratio,
            decay_fraction=decay_fraction,
        ),
        last_epoch=last_epoch,
    )
