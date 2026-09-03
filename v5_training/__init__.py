"""Executable, framework-neutral V5 training-state and checkpoint contracts."""

from .checkpoint import CheckpointStore, InjectedCrash
from .distributed import DistributedCheckpoint, RankCheckpoint
from .optimizer import (
    build_adamw_optimizer,
    build_optimizer,
    group_receipt,
    optimizer_group_receipt,
    validate_parameter_ownership,
)
from .runner import RunController, RunStatus, RunnerState
from .schedule import lr_at, schedule_receipt
from .state import CursorState, IdentityBindings, TrainingState, next_update_tokens
from .step import certify_update
from .trainer import BackendReport, train

__all__ = [
    "CheckpointStore",
    "CursorState",
    "IdentityBindings",
    "InjectedCrash",
    "DistributedCheckpoint",
    "RankCheckpoint",
    "build_adamw_optimizer",
    "build_optimizer",
    "group_receipt",
    "optimizer_group_receipt",
    "validate_parameter_ownership",
    "BackendReport",
    "RunController",
    "RunStatus",
    "RunnerState",
    "TrainingState",
    "certify_update",
    "lr_at",
    "next_update_tokens",
    "schedule_receipt",
    "train",
]
