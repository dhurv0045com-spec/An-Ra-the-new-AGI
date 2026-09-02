"""Rigorous experiment state machine preventing scientific gate skipping.

Enforces the non-negotiable progression:
DRAFT
  -> PREREGISTERED
  -> IDENTITIES_BOUND
  -> REMOTE_CANARY_REQUIRED
  -> REMOTE_CANARY_PASS
  -> DEVELOPMENT_RUN
  -> DEVELOPMENT_COMPLETE
  -> RECIPE_FROZEN
  -> FRESH_COMMITMENT
  -> FRESH_RUN
  -> REPLICATED / FAILED
  -> M102_ELIGIBLE / STOP

Mechanically guarantees that FRESH / SEALED evaluation suites cannot be used
for iteration, hyperparameter tuning, or arm selection.
"""

from __future__ import annotations

from enum import Enum
from typing import Any


class IllegalPhaseTransitionError(RuntimeError):
    """Raised when an illegal phase transition is attempted."""


class FreshLeakageViolationError(RuntimeError):
    """Raised when FRESH or SEALED suites are accessed during model selection or tuning."""


class ExperimentPhase(str, Enum):
    DRAFT = "DRAFT"
    PREREGISTERED = "PREREGISTERED"
    IDENTITIES_BOUND = "IDENTITIES_BOUND"
    REMOTE_CANARY_REQUIRED = "REMOTE_CANARY_REQUIRED"
    REMOTE_CANARY_PASS = "REMOTE_CANARY_PASS"
    DEVELOPMENT_RUN = "DEVELOPMENT_RUN"
    DEVELOPMENT_COMPLETE = "DEVELOPMENT_COMPLETE"
    RECIPE_FROZEN = "RECIPE_FROZEN"
    FRESH_COMMITMENT = "FRESH_COMMITMENT"
    FRESH_RUN = "FRESH_RUN"
    REPLICATED = "REPLICATED"
    FAILED = "FAILED"
    M102_ELIGIBLE = "M102_ELIGIBLE"
    STOP = "STOP"


VALID_TRANSITIONS: dict[ExperimentPhase, set[ExperimentPhase]] = {
    ExperimentPhase.DRAFT: {ExperimentPhase.PREREGISTERED},
    ExperimentPhase.PREREGISTERED: {ExperimentPhase.IDENTITIES_BOUND},
    ExperimentPhase.IDENTITIES_BOUND: {ExperimentPhase.REMOTE_CANARY_REQUIRED},
    ExperimentPhase.REMOTE_CANARY_REQUIRED: {ExperimentPhase.REMOTE_CANARY_PASS, ExperimentPhase.STOP},
    ExperimentPhase.REMOTE_CANARY_PASS: {ExperimentPhase.DEVELOPMENT_RUN},
    ExperimentPhase.DEVELOPMENT_RUN: {ExperimentPhase.DEVELOPMENT_COMPLETE, ExperimentPhase.STOP},
    ExperimentPhase.DEVELOPMENT_COMPLETE: {ExperimentPhase.RECIPE_FROZEN, ExperimentPhase.STOP},
    ExperimentPhase.RECIPE_FROZEN: {ExperimentPhase.FRESH_COMMITMENT, ExperimentPhase.STOP},
    ExperimentPhase.FRESH_COMMITMENT: {ExperimentPhase.FRESH_RUN},
    ExperimentPhase.FRESH_RUN: {ExperimentPhase.REPLICATED, ExperimentPhase.FAILED, ExperimentPhase.STOP},
    ExperimentPhase.REPLICATED: {ExperimentPhase.M102_ELIGIBLE, ExperimentPhase.STOP},
    ExperimentPhase.FAILED: {ExperimentPhase.STOP},
    ExperimentPhase.M102_ELIGIBLE: set(),
    ExperimentPhase.STOP: set(),
}


class ExperimentLifecycle:
    """Manages experiment phase progression and enforces suite separation."""

    def __init__(self, initial_phase: ExperimentPhase = ExperimentPhase.DRAFT) -> None:
        self._current_phase = initial_phase
        self._history: list[ExperimentPhase] = [initial_phase]
        self._frozen_recipe: dict[str, Any] | None = None

    @property
    def current_phase(self) -> ExperimentPhase:
        return self._current_phase

    @property
    def history(self) -> list[ExperimentPhase]:
        return list(self._history)

    def transition_to(self, next_phase: ExperimentPhase) -> None:
        valid_targets = VALID_TRANSITIONS.get(self._current_phase, set())
        if next_phase not in valid_targets:
            raise IllegalPhaseTransitionError(
                f"Illegal phase transition from {self._current_phase.value} to {next_phase.value}. "
                f"Allowed transitions: {[p.value for p in valid_targets]}"
            )
        self._current_phase = next_phase
        self._history.append(next_phase)

    def freeze_winning_recipe(self, recipe_details: dict[str, Any]) -> None:
        if self._current_phase != ExperimentPhase.DEVELOPMENT_COMPLETE:
            raise IllegalPhaseTransitionError(
                f"Cannot freeze recipe while in phase {self._current_phase.value}; must be in DEVELOPMENT_COMPLETE"
            )
        self._frozen_recipe = dict(recipe_details)
        self.transition_to(ExperimentPhase.RECIPE_FROZEN)

    def verify_suite_access(self, split_name: str) -> None:
        """Mechanically block access to FRESH/SEALED suites before recipe is frozen."""
        split_lower = split_name.lower()
        if split_lower in ("fresh", "sealed"):
            if self._current_phase not in (
                ExperimentPhase.FRESH_COMMITMENT,
                ExperimentPhase.FRESH_RUN,
                ExperimentPhase.REPLICATED,
                ExperimentPhase.FAILED,
                ExperimentPhase.M102_ELIGIBLE,
            ):
                raise FreshLeakageViolationError(
                    f"CRITICAL: Attempted to access prospective evaluation split '{split_name}' while in "
                    f"phase {self._current_phase.value}. Prospective splits can ONLY be evaluated after "
                    f"winning recipe is frozen and committed (FRESH_COMMITMENT or FRESH_RUN)."
                )