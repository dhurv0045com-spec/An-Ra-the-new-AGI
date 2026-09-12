"""Shared public reset/step protocol for finite research environments."""
from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Protocol

from ..contracts import Action, PublicObservation, TaskSpec, content_identity


class EnvironmentError(ValueError):
    """An action violates the public episode protocol."""


@dataclass(frozen=True)
class StepResult:
    observation: PublicObservation
    reward: float
    termination: bool
    truncation: bool
    cost: float


class PublicEnvironment(Protocol):
    task_spec: TaskSpec

    def reset(self, seed: int) -> PublicObservation: ...
    def step(self, action: Action) -> StepResult: ...


class FiniteEnvironment:
    """Protocol enforcement; subclasses only implement legal domain actions."""
    family = "finite"

    def __init__(self, task_spec: TaskSpec):
        self.task_spec = task_spec
        self._episode_id = ""
        self._step = 0
        self._budget = 0
        self._done = True
        self._feedback = None
        self._rng = random.Random(0)

    def reset(self, seed: int) -> PublicObservation:
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise EnvironmentError("seed must be an integer")
        self._rng = random.Random(seed)
        self._episode_id = content_identity({"task": self.task_spec.semantic_id, "seed": seed})
        self._step = 0
        budget = self.task_spec.resource_limits["inquiries"]
        if not isinstance(budget, int) or isinstance(budget, bool) or budget <= 0:
            raise EnvironmentError("task inquiry budget must be a positive integer")
        self._budget = budget
        self._done = False
        self._feedback = None
        self._reset_state()
        return self._observation()

    def step(self, action: Action) -> StepResult:
        if self._done:
            raise EnvironmentError("episode is already complete")
        if action.episode_id != self._episode_id or action.step_id != self._step:
            raise EnvironmentError("action episode or step does not match current observation")
        if self._budget <= 0:
            self._done = True
            raise EnvironmentError("action budget exhausted")
        reward, termination, feedback = self._apply(action)
        self._budget -= 1
        self._step += 1
        truncation = self._budget == 0 and not termination
        self._done = termination or truncation
        self._feedback = feedback
        return StepResult(self._observation(), float(reward), termination, truncation, 1.0)

    def _observation(self) -> PublicObservation:
        return PublicObservation(
            self.task_spec.semantic_id, self._episode_id, self._step,
            self._observable_values(), self._legal_schema(), self._feedback, self._budget,
        )

    def _reset_state(self) -> None: raise NotImplementedError
    def _apply(self, action: Action): raise NotImplementedError
    def _observable_values(self): raise NotImplementedError
    def _legal_schema(self): raise NotImplementedError


def require_arguments(action: Action, kind: str, fields: dict[str, type]) -> dict:
    if action.action_kind != kind:
        raise EnvironmentError(f"expected action kind {kind}")
    args = dict(action.arguments)
    if set(args) != set(fields):
        raise EnvironmentError(f"{kind} arguments must be exactly {sorted(fields)}")
    for key, typ in fields.items():
        if not isinstance(args[key], typ) or (typ is int and isinstance(args[key], bool)):
            raise EnvironmentError(f"{key} must have type {typ.__name__}")
    return args
