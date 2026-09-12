"""Public environments (B08): switch, inventory and program laboratory.

All three share one interface, one public-information rule and one declared
charging rule. Hidden mechanism objects live only in private environment
attributes and can never enter the public codec path. Independent oracle
checkers live in ``oracles.py`` as named diagnostic controls — they are never
part of the learner's input path.
"""
from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Any, Mapping

from bramastra_lab.research.contracts.core import Action, PublicObservation

# Declared charging rule for every B2 environment: an invalid action consumes
# one budget unit and reports a failure without changing hidden state.
CHARGING_RULE = "invalid_action_consumes_one_budget_unit_and_fails"
INQUIRY_COST = 1.0
SUBMIT_COST = 1.0  # counted separately from inquiries in episode summaries


class EnvironmentError(RuntimeError):
    """An environment was driven against its declared interface."""


class BaseEnvironment(ABC):
    """One deterministic public environment with a finite query space."""

    name: str = "abstract"
    charging_rule: str = CHARGING_RULE

    def __init__(self, *, budget: int, seed: int = 0) -> None:
        if budget <= 0:
            raise EnvironmentError("budget must be positive")
        self.budget = budget
        self.seed = seed
        self._episode_id: str | None = None
        self._step_id = 0
        self._remaining = budget
        self._terminated = False
        self._inquiries = 0
        self._submissions = 0
        self._invalid_actions = 0
        self._inquiry_cost_total = 0.0
        self._submission_cost_total = 0.0
        self._rng = random.Random(seed)
        self._reset_hidden()

    @abstractmethod
    def _reset_hidden(self) -> None:
        """Sample the hidden mechanism; private state only."""

    @abstractmethod
    def _public_values(self) -> Mapping[str, Any]:
        """Everything the learner may see. No hidden mechanism objects."""

    @abstractmethod
    def _legal_actions(self) -> list[Mapping[str, Any]]:
        """Explicitly enumerated legal candidate actions for the current step."""

    @abstractmethod
    def _apply_action(self, action: Mapping[str, Any]) -> tuple[Mapping[str, Any], bool]:
        """Return (feedback, success). Hidden state changes belong here."""

    @abstractmethod
    def _is_submission(self, action: Mapping[str, Any]) -> bool:
        """True when the action is the final submission."""

    # -- public interface -----------------------------------------------------

    @property
    def episode_id(self) -> str:
        return self._episode_id or "unstarted"

    @property
    def remaining_budget(self) -> int:
        return self._remaining

    @property
    def inquiry_cost_total(self) -> float:
        return self._inquiry_cost_total

    @property
    def submission_cost_total(self) -> float:
        return self._submission_cost_total

    def reset(self, *, episode_id: str) -> PublicObservation:
        self._episode_id = episode_id
        self._step_id = 0
        self._remaining = self.budget
        self._terminated = False
        self._inquiries = 0
        self._submissions = 0
        self._invalid_actions = 0
        self._inquiry_cost_total = 0.0
        self._submission_cost_total = 0.0
        self._reset_hidden()
        return self._observation(feedback={"kind": "reset"})

    def _observation(self, *, feedback: Mapping[str, Any]) -> PublicObservation:
        values = dict(self._public_values())
        values["environment_name"] = self.name
        values["episode_key"] = self.episode_id
        return PublicObservation(
            task_id=self.name, episode_id=self.episode_id, step_id=self._step_id,
            observable_values=values,
            legal_action_schema={"type": "array", "items": {"type": "object"}},
            feedback=dict(feedback), remaining_budget=self._remaining)

    def legal_actions(self) -> list[Mapping[str, Any]]:
        if self._terminated:
            raise EnvironmentError("episode already terminated")
        return self._legal_actions()

    def step(self, action: Mapping[str, Any]) -> tuple[PublicObservation, float, bool, bool]:
        """Take one action; returns (observation, cost, termination, truncation).

        Invalid actions are charged under the declared rule: one budget unit,
        no state change, explicit failure feedback.
        """
        if self._terminated:
            raise EnvironmentError("episode already terminated")
        if not isinstance(action, Mapping):
            raise EnvironmentError("action must be a mapping")
        legal = self._legal_actions()
        is_valid = any(_matches_candidate(action, candidate) for candidate in legal)
        if not is_valid:
            feedback = {"kind": "invalid_action", "reason": "not in legal candidate set"}
            cost = 1.0
            self._invalid_actions += 1
            self._inquiry_cost_total += cost
        else:
            submission = self._is_submission(action)
            feedback, success = self._apply_action(action)
            cost = SUBMIT_COST if submission else INQUIRY_COST
            if submission:
                self._submissions += 1
                self._submission_cost_total += cost
                self._terminated = True
                feedback = {**feedback, "submitted_answer": feedback.get("submitted_answer"),
                            "success": success}
            else:
                self._inquiries += 1
                self._inquiry_cost_total += cost
        self._step_id += 1
        self._remaining = max(0, self._remaining - int(cost))
        truncation = self._remaining == 0 and not self._terminated
        termination = self._terminated
        if truncation:
            self._terminated = True
        return self._observation(feedback=feedback), cost, termination, truncation

    def action_record(self, action: Mapping[str, Any], policy_identity: str) -> Action:
        return Action(episode_id=self.episode_id, step_id=self._step_id,
                      action_kind=str(action.get("kind", "unknown")),
                      arguments={key: value for key, value in action.items()
                                 if key != "kind"},
                      policy_identity=policy_identity)


def _matches_candidate(action: Mapping[str, Any], candidate: Mapping[str, Any]) -> bool:
    """A legal candidate is a prefix spec: every candidate key must match the
    action; the action may carry additional payload (e.g. a submitted value)."""
    if set(candidate) - set(action):
        return False
    return all(action[key] == candidate[key] for key in candidate)


def episode_summary(environment: BaseEnvironment, success: bool) -> dict[str, Any]:
    """Cost accounting with submission cost separated from inquiries.

    The budget unit is an *action*: every action, including the final
    submission and any invalid action, consumes one unit. Inquiries,
    submissions and invalid actions are reported separately so budget
    comparisons across environments are never silently mismatched (W02
    review correction).
    """
    return {
        "environment": environment.name,
        "charging_rule": environment.charging_rule,
        "budget_unit": "action",
        "inquiries": environment._inquiries,
        "submissions": environment._submissions,
        "invalid_actions": environment._invalid_actions,
        "total_actions": environment._inquiries + environment._submissions
                         + environment._invalid_actions,
        "inquiry_cost_total": environment.inquiry_cost_total,
        "submission_cost_total": environment.submission_cost_total,
        "budget_start": environment.budget,
        "budget_remaining": environment.remaining_budget,
        "success": success,
    }
