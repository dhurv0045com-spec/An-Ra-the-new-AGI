"""Bounded planning and collection interfaces (B10).

The planner imagines futures exclusively through the public model API and
marks every predicted step as imagined; predicted outcomes are never written
as observed facts, and real interactions are counted separately. The oracle
adapter is a named diagnostic that refuses to construct without an explicit
acknowledgement and is stamped into every result identity, so it cannot be
selected accidentally in primary inference. Collection policies expose fixed
and learned hooks; neither is trained at scale in this build.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from bramastra_lab.research.config import BuildConfig
from bramastra_lab.research.models import IntegratedModel

PlanningPolicy = Callable[[Mapping[str, Any]], Mapping[str, Any]]


class PlanningError(RuntimeError):
    """A planning component was driven against its declared contract."""


@dataclass(frozen=True)
class PlanStep:
    action: Mapping[str, Any]
    imagined: bool                       # True for every predicted step
    predicted_feedback: Mapping[str, Any]
    depth: int
    score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"action": dict(self.action), "imagined": self.imagined,
                "predicted_feedback": dict(self.predicted_feedback),
                "depth": self.depth, "score": self.score}


@dataclass(frozen=True)
class PlanResult:
    steps: tuple[PlanStep, ...]
    used_oracle: bool
    bounded_by: str                      # depth | nodes | time | complete
    imagined_steps: int
    real_interactions: int = 0           # only the caller's real env steps count
    policy_identity: str = "bounded-rollout/v1"

    @property
    def identity(self) -> str:
        from bramastra_lab.research.contracts.core import content_identity

        return content_identity({
            "steps": [step.to_dict() for step in self.steps],
            "used_oracle": self.used_oracle, "bounded_by": self.bounded_by,
            "policy_identity": self.policy_identity,
        })

    def to_dict(self) -> dict[str, Any]:
        return {"steps": [step.to_dict() for step in self.steps],
                "used_oracle": self.used_oracle, "bounded_by": self.bounded_by,
                "imagined_steps": self.imagined_steps,
                "real_interactions": self.real_interactions,
                "policy_identity": self.policy_identity, "identity": self.identity}


def no_planning_baseline(view: Mapping[str, Any]) -> Mapping[str, Any]:
    """The no-planning control: act on the first legal candidate, plan nothing."""
    actions = view.get("legal_actions")
    if not actions:
        raise PlanningError("no legal actions available to the no-planning baseline")
    return actions[0]


class BoundedRolloutPlanner:
    """Depth- and node-bounded look-ahead using only the public model API.

    Candidate actions are scored with the model's action head; imagined
    continuations are generated with the model's own decoder. The planner
    never touches a real environment and never consults hidden state.
    """

    def __init__(self, model: IntegratedModel, config: BuildConfig, *,
                 max_depth: int = 2, max_nodes: int = 8,
                 max_wall_seconds: float = 5.0) -> None:
        if max_depth < 1:
            raise PlanningError("max_depth must be at least 1")
        if max_nodes < 1:
            raise PlanningError("max_nodes must be at least 1")
        self.model = model
        self.config = config
        self.max_depth = max_depth
        self.max_nodes = max_nodes
        self.max_wall_seconds = max_wall_seconds

    def plan(self, *, sequence_tokens: list[int], candidate_spans: list[int],
             candidate_actions: Sequence[Mapping[str, Any]],
             legal_mask: Sequence[bool],
             max_new_tokens_per_level: int = 8) -> PlanResult:
        """Score root candidates and imagine one bounded continuation each.

        Only the model is consulted: candidates are scored with the action
        head, and each candidate's continuation is generated greedily by the
        same decoder, capped by ``max_depth`` levels and the node/wall bounds.
        Every returned step is an imagined prediction, never an observation.
        """
        from bramastra_lab.research.runtime.inference import (
            InferenceError,
            generate_free_form,
            score_finite_actions,
        )

        if len(candidate_actions) != len(legal_mask) or len(candidate_spans) != len(legal_mask):
            raise PlanningError("candidates, spans and mask must align")
        deadline = time.monotonic() + self.max_wall_seconds
        nodes = 0
        steps: list[PlanStep] = []
        bounded_by = "complete"
        try:
            report = score_finite_actions(self.model, self.config, sequence_tokens,
                                          list(candidate_spans), list(legal_mask))
        except InferenceError as exc:
            raise PlanningError(f"model scoring failed: {exc}") from exc
        ranked = sorted(
            (index for index, legal in enumerate(legal_mask) if legal),
            key=lambda index: report.scores[index], reverse=True)
        for index in ranked:
            if nodes >= self.max_nodes:
                bounded_by = "nodes"
                break
            if time.monotonic() > deadline:
                bounded_by = "time"
                break
            imagined_prefix = list(sequence_tokens[:candidate_spans[index] + 1])
            continuation = generate_free_form(
                self.model, self.config, imagined_prefix,
                max_new_tokens=self.max_depth * max_new_tokens_per_level)
            nodes += 1
            steps.append(PlanStep(
                action=dict(candidate_actions[index]), imagined=True,
                predicted_feedback={"root_score": report.scores[index],
                                    "imagined_continuation": continuation.answer,
                                    "stopped_on_eos": continuation.stopped_on_eos},
                depth=self.max_depth, score=report.scores[index]))
        if not steps:
            raise PlanningError("planner produced no candidate within its bounds")
        return PlanResult(steps=tuple(steps), used_oracle=False,
                          bounded_by=bounded_by, imagined_steps=len(steps))


class OracleAdapter:
    """Explicit diagnostic oracle; refused without acknowledgement.

    ``allow_oracle=True`` must be passed deliberately. Every result carries
    ``used_oracle=True`` so oracle assistance cannot be reported as learned
    competence. Primary inference has no oracle parameter at all.
    """

    policy_identity = "oracle-diagnostic/v1"

    def __init__(self, oracle: Callable[[Mapping[str, Any]], Mapping[str, Any]], *,
                 allow_oracle: bool = False) -> None:
        if not allow_oracle:
            raise PlanningError(
                "OracleAdapter requires allow_oracle=True; oracle assistance is a "
                "declared diagnostic control, never a default")
        self._oracle = oracle

    def plan(self, view: Mapping[str, Any]) -> PlanResult:
        action = dict(self._oracle(view))
        return PlanResult(steps=(PlanStep(action=action, imagined=False,
                                          predicted_feedback={"source": "oracle"},
                                          depth=0),),
                          used_oracle=True, bounded_by="complete", imagined_steps=0,
                          policy_identity=self.policy_identity)


def record_real_interaction(step: PlanStep) -> None:
    """Record a real interaction; imagined predictions are rejected.

    This is the boundary that keeps predicted outcomes from being written as
    observed facts.
    """
    if step.imagined:
        raise PlanningError(
            "refusing to record an imagined step as a real interaction; "
            "predicted outcomes are not observations")


class FixedCollectionPolicy:
    """Fixed data-collection control: round-robin over legal candidates."""

    policy_identity = "fixed-collection/v1"

    def __init__(self, *, seed: int = 0) -> None:
        import random

        self._rng = random.Random(seed)
        self.real_interactions = 0

    def select(self, view: Mapping[str, Any]) -> Mapping[str, Any]:
        actions = view.get("legal_actions")
        if not actions:
            raise PlanningError("no legal actions available to the fixed policy")
        self.real_interactions += 1
        return dict(self._rng.choice(actions))


class LearnedCollectionPolicy:
    """Learner-selected experience interface (untrained hook).

    Scores legal candidates through the model's action head with a declared
    fallback when the model fails or returns no decision. Not trained in this
    build; the interface exists so a later learned policy has the same API.
    """

    policy_identity = "learned-collection-hook/v1"

    def __init__(self, model: IntegratedModel, config: BuildConfig, *,
                 fallback: PlanningPolicy = no_planning_baseline) -> None:
        self.model = model
        self.config = config
        self.fallback = fallback
        self.real_interactions = 0
        self.fallback_uses = 0

    def select(self, view: Mapping[str, Any]) -> Mapping[str, Any]:
        """Select one action; model failure falls back to the declared policy."""
        from bramastra_lab.research.runtime.inference import InferenceError

        candidates = view.get("candidates")
        sequence_tokens = view.get("sequence_tokens")
        legal_mask = view.get("legal_mask")
        try:
            if not candidates or not sequence_tokens or not legal_mask:
                raise InferenceError("view lacks model-scoring inputs")
            from bramastra_lab.research.runtime.inference import score_finite_actions

            report = score_finite_actions(self.model, self.config, sequence_tokens,
                                          [c["span_end"] for c in candidates],
                                          legal_mask)
        except (InferenceError, PlanningError):
            self.fallback_uses += 1
            action = dict(self.fallback(view))
        else:
            action = dict(candidates[report.selected_index]["action"])
        self.real_interactions += 1
        return action
