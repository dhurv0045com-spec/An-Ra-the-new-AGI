"""Hardware-agnostic skill and observation schemas."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


@dataclass(frozen=True)
class SensorObservation:
    timestamp: float
    frame_id: str
    values: dict[str, Any]
    covariance: tuple[float, ...] = ()
    confidence: float = 1.0
    uncertainty: dict[str, float] = field(default_factory=dict)
    natural_language_summary: str = ""


@dataclass(frozen=True)
class SkillGoal:
    skill_name: str
    parameters: dict[str, Any]
    preconditions: tuple[str, ...]
    postconditions: tuple[str, ...]
    timeout_seconds: float
    approval_required: bool = False
    source_mission: str = ""


@dataclass(frozen=True)
class SkillResult:
    success: bool
    observations: tuple[SensorObservation, ...] = ()
    artifacts: tuple[str, ...] = ()
    error: str = ""


class WorkflowState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Workflow:
    workflow_id: str
    goal: str
    skills: list[SkillGoal]
    state: WorkflowState = WorkflowState.PENDING
    current_index: int = 0
    trace: list[dict[str, Any]] = field(default_factory=list)
