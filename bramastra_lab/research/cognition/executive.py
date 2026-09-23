"""Learned executive and metacognitive decision interfaces (M20).

Typed cognitive operations routed through the shared model's candidate-isolated
action scorer (M02). The host validates schemas/budgets/scope and executes the
selected operation; it never silently replaces a weak learned decision with an
oracle decision while labeling the run learned — every decision records its
origin: model | fixed_rule | external_agent | human.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Callable, Mapping, Sequence

from bramastra_lab.research.cognition.workspace import (
    OPERATION_VERBS,
    CognitiveWorkspace,
)

DECISION_ORIGINS = frozenset({"model", "fixed_rule", "symbolic_teacher",
                              "external_agent", "human", "fallback"})

NO_PROGRESS_LIMIT = 3


class ExecutiveError(ValueError):
    """An executive operation violated its contract."""


@dataclass(frozen=True)
class ResourceVector:
    inference_tokens: int = 0
    model_calls: int = 0
    real_interactions: int = 0
    tool_calls: int = 0
    wall_seconds: float = 0.0

    def __post_init__(self) -> None:
        for name in ("inference_tokens", "model_calls", "real_interactions",
                     "tool_calls"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ExecutiveError(f"{name} must be a nonnegative integer")
        if _finite_number(self.wall_seconds, "wall_seconds") < 0:
            raise ExecutiveError("wall_seconds must be nonnegative")

    def add(self, other: "ResourceVector") -> "ResourceVector":
        if not isinstance(other, ResourceVector):
            raise ExecutiveError("resource deltas must be ResourceVector values")
        return ResourceVector(
            inference_tokens=self.inference_tokens + other.inference_tokens,
            model_calls=self.model_calls + other.model_calls,
            real_interactions=self.real_interactions + other.real_interactions,
            tool_calls=self.tool_calls + other.tool_calls,
            wall_seconds=self.wall_seconds + other.wall_seconds)


@dataclass(frozen=True)
class CognitiveOperation:
    """A typed operation candidate. Candidates are data; the scorer picks."""

    verb: str
    arguments: Mapping[str, Any]
    evidence_aliases: tuple[str, ...] = ()
    expected_result: str = ""
    cost: ResourceVector = field(default_factory=ResourceVector)

    def __post_init__(self) -> None:
        if self.verb not in OPERATION_VERBS:
            raise ExecutiveError(
                f"verb must be one of {sorted(OPERATION_VERBS)}, got {self.verb!r}")
        if not isinstance(self.arguments, Mapping):
            raise ExecutiveError("arguments must be a mapping")
        if not isinstance(self.cost, ResourceVector):
            raise ExecutiveError("cost must be a ResourceVector")

    def identity(self) -> str:
        from bramastra_lab.research.contracts.core import content_identity

        return content_identity({"verb": self.verb, "arguments": dict(self.arguments)})


@dataclass(frozen=True)
class ExecutiveDecision:
    selected: CognitiveOperation
    origin: str
    reason: str
    fallback_used: bool
    candidate_count: int
    scores: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.origin not in DECISION_ORIGINS:
            raise ExecutiveError(
                f"decision origin must be one of {sorted(DECISION_ORIGINS)}")
        if not isinstance(self.selected, CognitiveOperation):
            raise ExecutiveError("selected must be a CognitiveOperation")


class OperationRegistry:
    """Declares the legal candidate set for the current workspace state.

    The host builds candidates from public state; the model scores them. A
    malformed/unbounded/no-progress decision terminates under the allowance.
    """

    def __init__(self, *, verbs: Sequence[str] = ("RETRIEVE", "PREDICT", "COMPARE",
                                                  "DERIVE", "VERIFY", "ABSTAIN",
                                                  "SUBMIT")) -> None:
        if not verbs:
            raise ExecutiveError("operation registry must contain at least one verb")
        for verb in verbs:
            if not isinstance(verb, str) or verb not in OPERATION_VERBS:
                raise ExecutiveError(f"unknown verb {verb!r}")
        if len(set(verbs)) != len(verbs):
            raise ExecutiveError("operation registry verbs must be unique")
        self.verbs = tuple(verbs)

    def candidates(self, workspace: CognitiveWorkspace,
                   arguments_for: Callable[[str], Mapping[str, Any]] | None = None,
                   cost_for: Callable[[str], Any] | None = None) -> list[CognitiveOperation]:
        out = []
        for verb in self.verbs:
            arguments = arguments_for(verb) if arguments_for else {}
            cost = cost_for(verb) if cost_for else ResourceVector()
            out.append(CognitiveOperation(verb=verb, arguments=arguments,
                                          expected_result=f"{verb} result",
                                          cost=cost))
        return out


@dataclass(frozen=True)
class _Scored:
    scores: tuple[float, ...]
    selected_index: int


class Executive:
    """Selects one operation per step via the model's isolated scorer.

    ``scorer`` is a callable (candidates) -> scores aligned to candidates.
    The real implementation wraps models.decisions.score_candidates through
    the public encoding path; tests inject a frozen fake scorer. The origin
    records who actually chose: model when the scorer decided, fixed_rule or
    fallback when the host did.
    """

    def __init__(self, registry: OperationRegistry,
                 scorer: Callable[[Sequence[CognitiveOperation]], Sequence[float]],
                 *, decision_origin: str = "model",
                 fixed_verb: str | None = None,
                 max_operations: int = 32) -> None:
        if decision_origin not in DECISION_ORIGINS:
            raise ExecutiveError(f"unknown decision origin {decision_origin!r}")
        if not isinstance(registry, OperationRegistry):
            raise ExecutiveError("registry must be an OperationRegistry")
        if not isinstance(max_operations, int) or isinstance(max_operations, bool) \
                or max_operations < 1:
            raise ExecutiveError("max_operations must be a positive integer")
        if decision_origin == "model" and not callable(scorer):
            raise ExecutiveError("model decision origin requires a callable scorer")
        self.registry = registry
        self.scorer = scorer
        self.decision_origin = decision_origin
        self.fixed_verb = fixed_verb
        self.max_operations = max_operations

    def decide(self, workspace: CognitiveWorkspace) -> ExecutiveDecision:
        candidates = self.registry.candidates(workspace)
        if not candidates:
            raise ExecutiveError("no legal operation candidates")
        if len(candidates) > self.max_operations:
            raise ExecutiveError(
                f"candidate set {len(candidates)} exceeds max_operations "
                f"{self.max_operations}; unbounded decision rejected")
        if self.decision_origin == "model":
            try:
                raw_scores = tuple(self.scorer(candidates))
            except (TypeError, ValueError, OverflowError) as exc:
                raise ExecutiveError("scorer must return candidate scores") from exc
            if len(raw_scores) != len(candidates):
                raise ExecutiveError("scorer returned a misaligned score vector")
            if any(not isinstance(score, (int, float)) or isinstance(score, bool)
                   for score in raw_scores):
                raise ExecutiveError("scorer must return numeric candidate scores")
            scores = tuple(float(score) for score in raw_scores)
            if any(not math.isfinite(score) for score in scores):
                raise ExecutiveError("scorer returned a non-finite candidate score")
            best = max(range(len(candidates)), key=lambda index: scores[index])
            return ExecutiveDecision(selected=candidates[best], origin="model",
                                     reason="learned scorer selected the top candidate",
                                     fallback_used=False, candidate_count=len(candidates),
                                     scores=scores)
        if self.fixed_verb is None:
            raise ExecutiveError("fixed-rule origin requires fixed_verb")
        for candidate in candidates:
            if candidate.verb == self.fixed_verb:
                return ExecutiveDecision(
                    selected=candidate, origin=self.decision_origin,
                    reason=f"fixed rule always selects {self.fixed_verb}",
                    fallback_used=self.decision_origin == "fallback",
                    candidate_count=len(candidates), scores=())
        raise ExecutiveError(f"fixed verb {self.fixed_verb!r} is not a legal candidate")


@dataclass
class StepRecord:
    decision: ExecutiveDecision
    executed: bool
    failure: str | None = None
    resource_delta: ResourceVector = field(default_factory=ResourceVector)


class SessionRunner:
    """Runs selected operations through declared executors with cost
    aggregation exactly once, and terminates no-progress cycles."""

    def __init__(self, executive: Executive,
                 executors: Mapping[str, Callable[[CognitiveOperation, CognitiveWorkspace], ResourceVector]],
                 *, no_progress_limit: int = NO_PROGRESS_LIMIT,
                 max_steps: int = 16) -> None:
        if not isinstance(no_progress_limit, int) or isinstance(no_progress_limit, bool) \
                or no_progress_limit < 1:
            raise ExecutiveError("no_progress_limit must be a positive integer")
        if not isinstance(max_steps, int) or isinstance(max_steps, bool) or max_steps < 1:
            raise ExecutiveError("max_steps must be a positive integer")
        self.executive = executive
        self.executors = dict(executors)
        self.no_progress_limit = no_progress_limit
        self.max_steps = max_steps
        self.resources = ResourceVector()
        self.records: list[StepRecord] = []

    def run(self, workspace: CognitiveWorkspace) -> dict[str, Any]:
        # A repeated operation is a stall only when it is presented with the
        # same substantive workspace. The step counter alone is bookkeeping;
        # evidence, beliefs, goals, and commitments are real progress.
        recent_no_progress: list[tuple[str, str]] = []
        terminated_reason: str | None = None
        for _step in range(self.max_steps):
            if workspace.step_counter >= workspace.budget:
                terminated_reason = "workspace_budget_exhausted"
                break
            decision = self.executive.decide(workspace)
            verb = decision.selected.verb
            state_identity = _progress_state_identity(workspace)
            signature = (verb, state_identity)
            if verb in ("RETRIEVE", "COMPARE", "PREDICT"):
                prior_needed = self.no_progress_limit - 1
                if (self.no_progress_limit == 1
                        or (len(recent_no_progress) >= prior_needed
                            and all(item == signature for item in
                                    recent_no_progress[-prior_needed:]))):
                    terminated_reason = f"no_progress:{verb}x{self.no_progress_limit}"
                    self.records.append(StepRecord(
                        decision=decision, executed=False,
                        failure=terminated_reason))
                    break
                recent_no_progress.append(signature)
                if len(recent_no_progress) > self.no_progress_limit:
                    recent_no_progress.pop(0)
            else:
                recent_no_progress.clear()
            executor = self.executors.get(verb)
            if executor is None:
                self.records.append(StepRecord(decision=decision, executed=False,
                                               failure="no_executor"))
                if verb == "SUBMIT":
                    terminated_reason = "submitted"
                    break
                continue
            delta = executor(decision.selected, workspace)
            if not isinstance(delta, ResourceVector):
                raise ExecutiveError(
                    f"executor for {verb!r} must return a ResourceVector")
            self.resources = self.resources.add(delta)
            self.records.append(StepRecord(decision=decision, executed=True,
                                           resource_delta=delta))
            workspace.step_counter += 1
            if verb == "SUBMIT":
                terminated_reason = "submitted"
                break
        else:
            terminated_reason = "max_steps_exhausted"
        return {"terminated_reason": terminated_reason,
                "steps": len(self.records),
                "resources": self.resources.__dict__,
                "records": self.records}


class DeliberationChoice:
    """Value-of-computation decision over: direct answer, more internal
    computation, retrieval, real query, tool check, abstain."""

    OPTIONS = frozenset({"direct_answer", "more_computation", "retrieve",
                         "query", "tool_check", "abstain"})

    def __init__(self, option: str, *, origin: str, predicted_improvement: float,
                 declared_cost: float, reason: str = "") -> None:
        if option not in self.OPTIONS:
            raise ExecutiveError(
                f"deliberation option must be one of {sorted(self.OPTIONS)}")
        if origin not in DECISION_ORIGINS:
            raise ExecutiveError(f"unknown origin {origin!r}")
        predicted = _finite_number(predicted_improvement, "predicted_improvement")
        cost = _finite_number(declared_cost, "declared_cost")
        if not -1e9 <= predicted <= 1e9:
            raise ExecutiveError("predicted_improvement must be finite and in [-1e9, 1e9]")
        if cost < 0:
            raise ExecutiveError("declared_cost must be nonnegative")
        self.option = option
        self.origin = origin
        self.predicted_improvement = predicted
        self.declared_cost = cost
        self.reason = reason

    def to_dict(self) -> dict[str, Any]:
        return {"option": self.option, "origin": self.origin,
                "predicted_improvement": self.predicted_improvement,
                "declared_cost": self.declared_cost, "reason": self.reason}


def decide_deliberation(*, predicted_improvement: float, declared_cost: float,
                        threshold: float, origin: str = "model") -> DeliberationChoice:
    """Frozen decision rule: deliberate only when predicted improvement
    justifies the declared cost; otherwise answer/abstain per protocol."""
    predicted = _finite_number(predicted_improvement, "predicted_improvement")
    cost = _finite_number(declared_cost, "declared_cost")
    threshold_value = _finite_number(threshold, "threshold")
    choice = "more_computation" if predicted - cost > threshold_value else "direct_answer"
    return DeliberationChoice(choice, origin=origin,
                              predicted_improvement=predicted,
                              declared_cost=cost,
                              reason="frozen threshold rule")


def filter_capability_summary(summary: Mapping[str, Any], source_pool: str) -> Mapping[str, Any]:
    """Capability summaries derived from query/confirmation outcomes are
    rejected before rendering — including when retrieved from memory (F/
    COGNITION §6). Only training/controller pools may enter the workspace."""
    if source_pool not in {"training", "controller"}:
        raise ExecutiveError(
            f"capability summary from pool {source_pool!r} cannot enter the "
            "model-visible workspace")
    return summary


def _progress_state_identity(workspace: CognitiveWorkspace) -> str:
    """Fingerprint the model-visible state, excluding only the budget countdown."""
    from bramastra_lab.research.contracts.core import content_identity

    state = workspace.rendered_view()
    state.pop("remaining_budget", None)
    return content_identity(state)


def _finite_number(value: Any, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ExecutiveError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ExecutiveError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ExecutiveError(f"{name} must be a finite number")
    return result
