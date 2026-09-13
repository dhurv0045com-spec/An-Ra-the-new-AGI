"""Canonical experience records (M01): public payloads, provenance, observed
transitions, predictions and teacher labels are distinct typed values.

Design rules (master DATA_AND_TRAINING §1–2):
- Public-view allowlist: episode ids, seeds, splits, source paths, run UUIDs
  and evaluator verdicts never reach token content or the model-visible view.
- Predictions are never observations: serializing a prediction into the
  observed ledger is rejected at the type level.
- Terminal and truncation are distinct; cost and reward are distinct.
- No later event can enter an earlier prefix (monotone step ids).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from bramastra_lab.research.contracts.core import (
    ContractError,
    content_identity,
)

SCHEMA = "bramastra-experience/v1"
RECORD_VERSIONS = {"PublicStep/v1", "ObservedEpisode/v1", "TeacherTarget/v1",
                   "PredictedStep/v1"}

# Fields that are audit/provenance only. They ride on the record for
# accounting but must never be rendered into learned tokens.
PROVENANCE_ONLY_FIELDS = frozenset({
    "episode_id", "seed", "split", "source_path", "run_id", "task_cluster",
    "evaluator_verdict", "generator_version", "checkpoint_id",
})

# Fields admissible into the learned token view (the public evidence).
PUBLIC_VIEW_FIELDS = frozenset({
    "goal", "observation", "action", "feedback", "remaining_budget", "step_index",
    "legal_action_digest", "memory_context",
})


class ExperienceError(ContractError):
    """An experience record violates its declared schema."""


def public_view(record: Mapping[str, Any]) -> dict[str, Any]:
    """Project the model-visible view of a record via strict allowlist.

    Provenance-only keys are dropped here, not filtered later: a leaked key
    is a contract violation at the boundary, not a rendering detail.
    """
    view = {key: record[key] for key in sorted(set(record) & PUBLIC_VIEW_FIELDS)}
    unknown_provenance = set(record) - PUBLIC_VIEW_FIELDS - PROVENANCE_ONLY_FIELDS
    if unknown_provenance:
        raise ExperienceError(
            f"record carries undeclared fields: {sorted(unknown_provenance)}")
    return view


def render_public_tokens(view: Mapping[str, Any]) -> list[int]:
    """Deterministic public-view token sketch (roles + canonical JSON bytes).

    The real byte codec lives in experience.codec; this adapter fixes the
    field ordering and role tags so the same view always renders identically
    and provenance changes cannot alter tokens.
    """
    import json

    from bramastra_lab.research.experience.codec import encode_event

    tokens: list[int] = []
    for key in sorted(view):
        tokens.extend(encode_event(key, view[key]))
    del json
    return tokens


@dataclass(frozen=True)
class PublicStep:
    """One public step: the model-visible before/after of an interaction.

    ``cost`` is the charged resource; ``reward`` is task outcome. They are
    separate fields with separate semantics and never merged silently.
    """

    step_index: int
    goal: Mapping[str, Any]
    observation: Mapping[str, Any]
    action: Mapping[str, Any]
    feedback: Mapping[str, Any]
    remaining_budget: int
    cost: float
    reward: float | None
    terminated: bool
    truncated: bool
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.step_index, int) or isinstance(self.step_index, bool) \
                or self.step_index < 0:
            raise ExperienceError("step_index must be a nonnegative integer")
        for name in ("goal", "observation", "action", "feedback"):
            if not isinstance(getattr(self, name), Mapping):
                raise ExperienceError(f"{name} must be a mapping")
        if not isinstance(self.remaining_budget, int) or isinstance(self.remaining_budget, bool) \
                or self.remaining_budget < 0:
            raise ExperienceError("remaining_budget must be a nonnegative integer")
        if not isinstance(self.cost, (int, float)) or isinstance(self.cost, bool) \
                or self.cost < 0:
            raise ExperienceError("cost must be a nonnegative number")
        if self.reward is not None:
            if not isinstance(self.reward, (int, float)) or isinstance(self.reward, bool) \
                    or not -1e18 <= float(self.reward) <= 1e18:
                raise ExperienceError("reward must be a finite number when present")
        if not isinstance(self.terminated, bool) or not isinstance(self.truncated, bool):
            raise ExperienceError("terminated/truncated must be booleans")
        if self.terminated and self.truncated:
            raise ExperienceError("terminal end and time-limit truncation are distinct")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "PublicStep/v1",
            "step_index": self.step_index, "goal": dict(self.goal),
            "observation": dict(self.observation), "action": dict(self.action),
            "feedback": dict(self.feedback), "remaining_budget": self.remaining_budget,
            "cost": self.cost, "reward": self.reward,
            "terminated": self.terminated, "truncated": self.truncated,
            "provenance": dict(self.provenance),
        }

    def token_view(self) -> list[int]:
        return render_public_tokens(public_view(self.to_dict()))

    def identity(self) -> str:
        return content_identity(self.to_dict())


@dataclass(frozen=True)
class ObservedEpisode:
    """An append-only observed episode built only from real PublicSteps."""

    episode_id: str
    steps: tuple[PublicStep, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.episode_id, str) or not self.episode_id:
            raise ExperienceError("episode_id must be a nonempty string")
        if not self.steps:
            raise ExperienceError("an observed episode requires at least one step")
        previous = None
        ended = False
        for step in self.steps:
            if ended:
                raise ExperienceError("no event may follow termination/truncation")
            if previous is not None and step.step_index != previous.step_index + 1:
                raise ExperienceError(
                    "step indices must be contiguous and monotone; a later event "
                    "cannot enter an earlier prefix")
            if previous is not None and step.remaining_budget > previous.remaining_budget:
                raise ExperienceError("remaining budget cannot increase within an episode")
            previous = step
            ended = step.terminated or step.truncated
        if not ended:
            raise ExperienceError("episodes must end in termination or truncation")

    def identity(self) -> str:
        return content_identity({"schema": "ObservedEpisode/v1",
                                 "episode_id": self.episode_id,
                                 "steps": [step.to_dict() for step in self.steps]})

    def total_cost(self) -> float:
        return float(sum(step.cost for step in self.steps))


@dataclass(frozen=True)
class PredictedStep:
    """A predicted (imagined) step. Type-distinct from PublicStep: the
    serializer refuses to convert predictions into observations."""

    step_index: int
    predicted_feedback: Mapping[str, Any]
    probability: float
    parse_valid: bool
    predictor_identity: str

    def __post_init__(self) -> None:
        if not isinstance(self.predicted_feedback, Mapping):
            raise ExperienceError("predicted_feedback must be a mapping")
        if not 0.0 <= float(self.probability) <= 1.0:
            raise ExperienceError("probability must lie in [0, 1]")
        if not isinstance(self.parse_valid, bool):
            raise ExperienceError("parse_valid must be a boolean")
        if not isinstance(self.predictor_identity, str) or not self.predictor_identity:
            raise ExperienceError("predictor_identity must be a nonempty string")

    def identity(self) -> str:
        return content_identity({"schema": "PredictedStep/v1",
                                 "step_index": self.step_index,
                                 "predicted_feedback": dict(self.predicted_feedback),
                                 "probability": self.probability,
                                 "parse_valid": self.parse_valid,
                                 "predictor_identity": self.predictor_identity})


def observe_prediction_as_step(prediction: PredictedStep, *args: Any, **kwargs: Any) -> None:
    """Deliberately unavailable: imagined branches never become observations."""
    raise ExperienceError(
        "a PredictedStep cannot be converted into an observed step without a real "
        "environment receipt; imagined transitions are rejected at the type level")


@dataclass(frozen=True)
class TeacherTarget:
    """A training-only supervision target with declared teacher provenance.

    ``kind`` selects the objective denominator family in the router (M05):
    belief | action | value | world | pair | token. Targets with NaN/inf
    payloads reject here, before any model allocation.
    """

    kind: str
    payload: Mapping[str, Any]
    teacher_identity: str
    task_cluster: str | None = None

    VALID_KINDS = frozenset({"belief", "action", "value", "world", "pair", "token",
                             "deliberation"})

    def __post_init__(self) -> None:
        if self.kind not in self.VALID_KINDS:
            raise ExperienceError(
                f"teacher target kind must be one of {sorted(self.VALID_KINDS)}")
        if not isinstance(self.payload, Mapping):
            raise ExperienceError("payload must be a mapping")
        if not isinstance(self.teacher_identity, str) or not self.teacher_identity:
            raise ExperienceError("teacher_identity must be a nonempty string")
        self._check_finite(self.payload, "payload")

    @staticmethod
    def _check_finite(value: Any, where: str) -> None:
        import math

        if isinstance(value, float):
            if not math.isfinite(value):
                raise ExperienceError(f"{where} contains a nonfinite number")
        elif isinstance(value, Mapping):
            for key, item in value.items():
                TeacherTarget._check_finite(item, f"{where}.{key}")
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                TeacherTarget._check_finite(item, f"{where}[{index}]")

    def identity(self) -> str:
        return content_identity({"schema": "TeacherTarget/v1", "kind": self.kind,
                                 "payload": dict(self.payload),
                                 "teacher_identity": self.teacher_identity,
                                 "task_cluster": self.task_cluster})
