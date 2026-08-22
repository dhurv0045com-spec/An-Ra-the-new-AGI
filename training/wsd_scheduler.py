"""Warmup-Stable-Decay LR schedule with explicit phase reporting.

Adopted from the iterate500 lineage (training/wsd_scheduler.py) where it was
battle-tested. The canonical recipe in PROGRESS.md specifies warmup + decay;
the TPU trainer never implemented it. This module restores that contract.

The schedule is *pack-aware*: ``total_steps`` is derived from the declared
token budget of the current data pack so the decay phase lands at the end of
exactly one pass over unique data - not mid-repeat.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


@dataclass(frozen=True)
class SchedulePhase:
    name: str
    step: int
    decay_started: bool


def wsd_multiplier(
    step: int,
    *,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1,
    decay_fraction: float = 0.1,
) -> float:
    """LR multiplier for step ``step`` in [0, total_steps)."""
    if total_steps <= 0 or warmup_steps < 0 or total_steps <= warmup_steps:
        raise ValueError("total_steps must be positive and greater than warmup_steps")
    if not 0.0 <= min_lr_ratio <= 1.0:
        raise ValueError("min_lr_ratio must be in [0, 1]")
    if not 0.0 < decay_fraction <= 1.0:
        raise ValueError("decay_fraction must be in (0, 1]")
    decay_start = max(warmup_steps, int(total_steps * (1.0 - decay_fraction)))
    if step < warmup_steps:
        return float(step) / max(1, warmup_steps)
    if step < decay_start:
        return 1.0
    progress = min(1.0, (step - decay_start) / max(1, total_steps - 1 - decay_start))
    return max(min_lr_ratio, 1.0 - progress * (1.0 - min_lr_ratio))


def phase_for_step(
    step: int, *, warmup_steps: int, total_steps: int, decay_fraction: float = 0.1
) -> SchedulePhase:
    decay_start = max(warmup_steps, int(total_steps * (1.0 - decay_fraction)))
    if step < warmup_steps:
        return SchedulePhase("warmup", step, False)
    if step < decay_start:
        return SchedulePhase("stable", step, False)
    return SchedulePhase("decay", step, True)


def steps_for_tokens(tokens: int, *, tokens_per_step: int) -> int:
    """Optimizer steps needed to consume ``tokens`` at ``tokens_per_step``."""
    if tokens_per_step <= 0:
        raise ValueError("tokens_per_step must be positive")
    return max(1, tokens // tokens_per_step)


@dataclass(frozen=True, slots=True)
class PackWsdSchedule:
    """Serializable WSD schedule whose position is the checkpointed pack step."""

    base_lr: float
    total_steps: int
    warmup_steps: int = 0
    min_lr_ratio: float = 0.1
    decay_fraction: float = 0.1

    def __post_init__(self) -> None:
        if self.base_lr <= 0:
            raise ValueError("base_lr must be positive")
        # Reuse the canonical validator.
        wsd_multiplier(
            0,
            warmup_steps=self.warmup_steps,
            total_steps=self.total_steps,
            min_lr_ratio=self.min_lr_ratio,
            decay_fraction=self.decay_fraction,
        )

    def lr_at(self, pack_step: int) -> float:
        return self.base_lr * wsd_multiplier(
            pack_step,
            warmup_steps=self.warmup_steps,
            total_steps=self.total_steps,
            min_lr_ratio=self.min_lr_ratio,
            decay_fraction=self.decay_fraction,
        )

    def to_dict(self) -> dict[str, object]:
        return {"name": "wsd_pack_v1", **asdict(self)}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PackWsdSchedule":
        if payload.get("name") != "wsd_pack_v1":
            raise ValueError(f"unsupported pack schedule: {payload.get('name')!r}")
        return cls(
            base_lr=float(payload["base_lr"]),
            total_steps=int(payload["total_steps"]),
            warmup_steps=int(payload["warmup_steps"]),
            min_lr_ratio=float(payload["min_lr_ratio"]),
            decay_fraction=float(payload["decay_fraction"]),
        )


def build_wsd_schedule(
    optimizer: Optimizer,
    *,
    total_steps: int,
    warmup_steps: int | None = None,
    warmup_fraction: float = 0.02,
    min_lr_ratio: float = 0.1,
    decay_fraction: float = 0.1,
    last_epoch: int = -1,
) -> LambdaLR:
    """Canonical recipe: 2% warmup, stable, then linear decay to min ratio."""
    resolved_warmup = warmup_steps or max(1, int(total_steps * warmup_fraction))
    return LambdaLR(
        optimizer,
        lr_lambda=lambda step: wsd_multiplier(
            step,
            warmup_steps=resolved_warmup,
            total_steps=total_steps,
            min_lr_ratio=min_lr_ratio,
            decay_fraction=decay_fraction,
        ),
        last_epoch=last_epoch,
    )
