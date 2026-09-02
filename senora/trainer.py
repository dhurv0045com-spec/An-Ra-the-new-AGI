"""Production-grade P35 trainer interface, token-based schedule, and fail-safe state transitions."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from v5_contracts.model_spec import ModelSpec
from v5_training.checkpoint import REQUIRED_COMPONENTS, CheckpointStore
from v5_training.runner import RunController, RunStatus
from v5_training.state import CursorState, IdentityBindings, TrainingState


class LocalScientificComputeConstraintError(RuntimeError):
    """Raised when an attempt is made to execute scientific training on unauthorized local hardware."""
    pass


@dataclass(frozen=True, slots=True)
class WSDSchedule:
    """Warmup-Stable-Decay schedule indexed strictly by cumulative tokens."""
    warmup_tokens: int
    stable_tokens: int
    decay_tokens: int
    peak_lr: float
    min_lr: float

    @classmethod
    def from_budget(
        cls,
        *,
        token_budget: int,
        peak_lr: float = 3e-4,
        warmup_fraction: float = 0.03,
        decay_fraction: float = 0.20,
        min_lr_ratio: float = 0.10,
    ) -> "WSDSchedule":
        warmup = int(round(token_budget * warmup_fraction))
        decay = int(round(token_budget * decay_fraction))
        stable = token_budget - warmup - decay
        if stable < 0:
            raise ValueError("warmup + decay fractions cannot exceed 1.0")
        return cls(
            warmup_tokens=warmup,
            stable_tokens=stable,
            decay_tokens=decay,
            peak_lr=peak_lr,
            min_lr=peak_lr * min_lr_ratio,
        )

    def get_lr(self, cumulative_tokens: int) -> float:
        if cumulative_tokens < self.warmup_tokens:
            if self.warmup_tokens == 0:
                return self.peak_lr
            return self.peak_lr * (cumulative_tokens / self.warmup_tokens)
        if cumulative_tokens < self.warmup_tokens + self.stable_tokens:
            return self.peak_lr
        decay_progress = (cumulative_tokens - self.warmup_tokens - self.stable_tokens) / max(1, self.decay_tokens)
        decay_progress = min(1.0, max(0.0, decay_progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
        return self.min_lr + (self.peak_lr - self.min_lr) * cosine


@dataclass(frozen=True, slots=True)
class P35TrainerConfig:
    """Configuration for P35 training execution."""
    model_spec: ModelSpec
    token_budget: int
    tokens_per_update: int
    learning_rate: float
    weight_decay: float
    gradient_clip_norm: float
    query_swap_lambda: float
    remote_authorized: bool = False

    def assert_valid(self) -> None:
        self.model_spec.assert_valid()
        if self.token_budget <= 0 or self.tokens_per_update <= 0:
            raise ValueError("token counts must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning rate and weight decay must be non-negative")
        if self.gradient_clip_norm <= 0:
            raise ValueError("gradient clip norm must be positive")
        if self.query_swap_lambda < 0.0:
            raise ValueError("query_swap_lambda cannot be negative")


class P35Trainer:
    """Fail-closed P35 training coordinator.

    Enforces that empirical model training is blocked on local hardware unless
    explicitly authorized for remote target cluster execution.
    """

    def __init__(
        self,
        config: P35TrainerConfig,
        *,
        identity_bindings: IdentityBindings,
        checkpoint_directory: Path,
    ) -> None:
        config.assert_valid()
        identity_bindings.assert_valid()
        self.config = config
        self.identity_bindings = identity_bindings
        self.checkpoint_directory = checkpoint_directory
        self.schedule = WSDSchedule.from_budget(
            token_budget=config.token_budget,
            peak_lr=config.learning_rate,
        )
        self.controller = RunController(
            target_update=config.token_budget // config.tokens_per_update
        )

    def verify_remote_execution_guard(self) -> None:
        """Enforce the Hard Compute Constraint locally."""
        if not self.config.remote_authorized:
            raise LocalScientificComputeConstraintError(
                "HARD COMPUTE CONSTRAINT ACTIVE: Local P35 scientific training is NOT authorized. "
                "Senora requires remote target compute or an authorized dry-run flag."
            )

    def initialize_training_state(
        self,
        *,
        initial_cursor: CursorState,
        rng_state_sha256: str,
        lineage_id: str = "p35-senora-run",
    ) -> TrainingState:
        """Create the initial immutable training state."""
        return TrainingState.initial(
            lineage_id=lineage_id,
            token_budget=self.config.token_budget,
            tokens_per_update=self.config.tokens_per_update,
            cursor=initial_cursor,
            rng_state_sha256=rng_state_sha256,
            curriculum_phase="phase-1-wsd",
            identities=self.identity_bindings,
        )

    def advance_step(
        self,
        current_state: TrainingState,
        *,
        tokens_by_source: Mapping[str, int],
        new_cursor: CursorState,
        new_rng_state_sha256: str,
        loss_value: float,
        gradient_norm: float,
        parent_checkpoint_sha256: str | None = None,
    ) -> TrainingState:
        """Advance one training step with mathematical and finite-gradient sanity checks."""
        if not math.isfinite(loss_value):
            self.controller.fail(code="NON_FINITE_LOSS")
            raise ValueError(f"Aborting run: non-finite loss {loss_value}")
        if not math.isfinite(gradient_norm) or gradient_norm > 100.0:
            self.controller.fail(code="GRADIENT_EXPLOSION")
            raise ValueError(f"Aborting run: gradient explosion with norm {gradient_norm}")

        # Advance runner lifecycle
        if self.controller.state.status is RunStatus.CREATED:
            self.controller.start()
        self.controller.complete_update()

        # Advance training state
        return current_state.advance(
            tokens_by_source=tokens_by_source,
            cursor=new_cursor,
            rng_state_sha256=new_rng_state_sha256,
            parent_checkpoint_sha256=parent_checkpoint_sha256,
        )

    def save_checkpoint(
        self,
        state: TrainingState,
        *,
        payloads: Mapping[str, bytes],
        expected_parent_sha256: str | None = None,
    ) -> str:
        """Atomically persist checkpoint and commit to RunController."""
        self.controller.begin_checkpoint()
        store = CheckpointStore(root=self.checkpoint_directory, lineage_id=state.lineage_id)
        manifest_sha = store.publish(
            state=state,
            payloads=payloads,
            expected_parent_sha256=expected_parent_sha256,
        )
        self.controller.commit_checkpoint(checkpoint_sha256=manifest_sha)
        return manifest_sha