"""Bounded branching planner over predicted public outcomes (M07/A5).

Q_d(h,a) = sum_o p(o|h,a) * [r(h,a,o) + gamma * C_{d-1}(h')]
C_0(h)   = V(h)

- States are public histories plus remaining allowance; the real environment's
  hidden state is never consulted. Imagined legality comes only from the
  declared public schema rule.
- Invalid/unknown outcome mass takes the protocol's conservative lower-bound
  return; it is never renormalized away.
- Node, model-call and depth budgets stop the search. Root selection changes
  when predicted feedback or downstream value changes while root policy
  scores are held fixed (no root-score dependence).
- The root-score control (B10 planner) remains available as a named control.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from bramastra_lab.research.config import BuildConfig
from bramastra_lab.research.models import IntegratedModel
from bramastra_lab.research.models.decisions import estimate_value
from bramastra_lab.research.models.world import PredictedOutcome, predict_step_finite


class PlanningError(ValueError):
    """A branching-planning call violated its contract."""


@dataclass(frozen=True)
class SupportOption:
    feedback: Mapping[str, Any]
    rendering: str
    terminated: bool
    reward: float                 # r(h, a, o) from the protocol


@dataclass
class SearchTrace:
    expanded_nodes: int = 0
    model_calls: int = 0
    stopped_by: str = "completed"
    imagined_steps: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class BranchingDecision:
    root_action: Mapping[str, Any]
    root_action_index: int
    q_values: tuple[float, ...]
    scores: tuple[float, ...]
    trace: SearchTrace

    @property
    def selection_changed_vs_policy(self) -> bool:
        """True when the Q-argmax differs from the policy-score argmax."""
        if not self.scores:
            return False
        return max(range(len(self.q_values)), key=lambda i: self.q_values[i]) != \
            max(range(len(self.scores)), key=lambda i: self.scores[i])


class BranchingPlanner:
    def __init__(self, model: IntegratedModel, config: BuildConfig, *,
                 max_depth: int, root_candidates: int = 4,
                 max_nodes: int = 64, max_model_calls: int = 32) -> None:
        if max_depth < 1:
            raise PlanningError("max_depth must be at least 1")
        self.model = model
        self.config = config
        self.max_depth = max_depth
        self.root_candidates = root_candidates
        self.max_nodes = max_nodes
        self.max_model_calls = max_model_calls

    def plan(self, *, prefix_tokens: Sequence[int], root_actions: Sequence[Mapping[str, Any]],
             outcome_support: Callable[[Mapping[str, Any]], Sequence[SupportOption]],
             reward_bound: tuple[float, float],
             policy_scores: Sequence[float] | None = None) -> BranchingDecision:
        """Expand root actions, then outcome branches to max_depth.

        ``outcome_support(action)`` returns the declared public support for
        that action (from the public schema). ``reward_bound`` is the
        declared reward range used for the conservative lower bound on
        unknown/invalid mass.
        """
        if not root_actions:
            raise PlanningError("no root actions supplied")
        if len(reward_bound) != 2 or reward_bound[0] > reward_bound[1]:
            raise PlanningError("reward bound must be (low, high) with low <= high")
        trace = SearchTrace()
        prefix = list(prefix_tokens)
        q_values: list[float] = []
        candidates = root_actions[: self.root_candidates]
        for action in candidates:
            q = self._value_branch(prefix, action, outcome_support, depth=self.max_depth,
                                   trace=trace, reward_bound=reward_bound)
            q_values.append(q)
            if trace.expanded_nodes >= self.max_nodes or \
                    trace.model_calls >= self.max_model_calls:
                trace.stopped_by = "node_or_model_budget"
                break
        if not q_values:
            raise PlanningError("search produced no evaluated root within its budgets")
        best = max(range(len(q_values)), key=lambda index: q_values[index])
        scores = tuple(float(score) for score in policy_scores) if policy_scores is not None \
            else ()
        return BranchingDecision(root_action=candidates[best],
                                 root_action_index=best,
                                 q_values=tuple(q_values), scores=scores,
                                 trace=trace)

    def _value_branch(self, prefix: list[int], action: Mapping[str, Any],
                      outcome_support: Callable[[Mapping[str, Any]], Sequence[SupportOption]],
                      *, depth: int, trace: SearchTrace,
                      reward_bound: tuple[float, float]) -> float:
        if trace.expanded_nodes >= self.max_nodes or \
                trace.model_calls >= self.max_model_calls:
            trace.stopped_by = "node_or_model_budget"
            return self._conservative_lower_bound(reward_bound, depth)
        if depth <= 0:
            return estimate_value(self.model, self.config, prefix)
        outcome: PredictedOutcome = predict_step_finite(
            self.model, self.config, prefix, action=action,
            support=[{"feedback": option.feedback, "rendering": option.rendering,
                      "terminated": option.terminated} for option in
                     outcome_support(action)])
        trace.model_calls += 1
        trace.expanded_nodes += 1
        expected = 0.0
        for option in outcome.options:
            imagined_step = {"action": dict(action), "predicted_feedback": dict(option.feedback),
                             "probability": option.probability, "imagined": True}
            trace.imagined_steps.append(imagined_step)
            protocol_reward = self._reward_for(outcome_support, action, option)
            if option.terminated or depth == 1:
                continuation = 0.0
            else:
                next_prefix = prefix + [1]
                continuation = self._value_branch(
                    next_prefix, action, outcome_support, depth=depth - 1,
                    trace=trace, reward_bound=reward_bound)
            expected += option.probability * (protocol_reward + continuation)
        # Unknown/invalid mass is conservative: (1 - parsed mass) * lower bound.
        parsed_mass = min(1.0, sum(option.probability for option in outcome.options))
        if parsed_mass < 1.0:
            expected += (1.0 - parsed_mass) * self._conservative_lower_bound(
                reward_bound, depth)
        return expected

    @staticmethod
    def _reward_for(outcome_support: Callable[[Mapping[str, Any]], Sequence[SupportOption]],
                    action: Mapping[str, Any], option) -> float:
        for support_option in outcome_support(action):
            if support_option.feedback == option.feedback:
                return float(support_option.reward)
        raise PlanningError("predicted option missing from declared support")

    @staticmethod
    def _conservative_lower_bound(reward_bound: tuple[float, float], depth: int) -> float:
        low, _high = reward_bound
        return low * depth
