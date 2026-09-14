"""Typed job input and phase result for K8 executors (D4)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class JobInput:
    """One typed phase job (D4)."""

    phase: str
    slot: int | None
    arm: str | None
    seed: int | None
    parent: str | None
    physical_device: str
    local_device: str
    data_dir: str
    run_dir: str
    precision: str
    deadline: float

    def validate(self) -> None:
        if self.phase not in ("E1", "E2", "E3", "E4", "E5", "E6", "E0"):
            raise ValueError(f"unknown phase {self.phase!r}")
        if not self.data_dir or not self.run_dir:
            raise ValueError("data_dir and run_dir are required")
        if not self.physical_device or not self.local_device:
            raise ValueError("physical and local devices are required")


@dataclass
class PhaseResult:
    """Typed executor result with actual counts (D4)."""

    status: str  # completed | failed
    committed_updates: int = 0
    attempted_updates: int = 0
    supervised_exposure: int = 0
    device_seconds: float = 0.0
    checkpoint_identity: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out = {"status": self.status,
               "committed_updates": self.committed_updates,
               "attempted_updates": self.attempted_updates,
               "supervised_exposure": self.supervised_exposure,
               "device_seconds": self.device_seconds,
               "checkpoint_identity": self.checkpoint_identity}
        out.update(self.extra)
        if self.error:
            out["error"] = self.error
        return out
