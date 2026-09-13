"""Episode compilation into learning targets (M06) and the declared teacher
interface: real ledger receipts become world/answer/action/return targets
with separate eligibility, preserving failures and excluding truncation
from complete-return objectives."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from bramastra_lab.research.experience.trajectory import (
    ExperienceError,
    ObservedEpisode,
    PublicStep,
    TeacherTarget,
)


class CompilationError(ExperienceError):
    """An episode could not be compiled into objective targets."""


@dataclass(frozen=True)
class CompiledEpisode:
    episode_id: str
    answer_targets: tuple[Mapping[str, Any], ...]
    world_targets: tuple[Mapping[str, Any], ...]
    action_targets: tuple[Mapping[str, Any], ...]
    return_targets: tuple[Mapping[str, Any], ...]
    qualified_steps: int
    excluded_steps: tuple[Mapping[str, Any], ...]

    def identity(self) -> str:
        from bramastra_lab.research.contracts.core import content_identity

        return content_identity({
            "episode_id": self.episode_id,
            "answer": [dict(t) for t in self.answer_targets],
            "world": [dict(t) for t in self.world_targets],
            "action": [dict(t) for t in self.action_targets],
            "returns": [dict(t) for t in self.return_targets],
            "qualified_steps": self.qualified_steps,
            "excluded": [dict(t) for t in self.excluded_steps],
        })


def compile_episode(episode: ObservedEpisode, *, reward_protocol: str = "terminal-1",
                    teacher_identity: str = "observed") -> CompiledEpisode:
    """Compile one observed episode into typed targets.

    - Truncated episodes are excluded from the complete-return objective
      (a time limit is not termination); their other targets remain.
    - Failed submissions are retained as observed steps but never become
      gold answer supervision.
    - Returns are Monte Carlo over real receipt costs/rewards with gamma=1.
    """
    steps = episode.steps
    truncated = steps[-1].truncated
    returns: list[float | None] = [None] * len(steps)
    if not truncated:
        running = 0.0
        for index in range(len(steps) - 1, -1, -1):
            reward = steps[index].reward or 0.0
            running = reward + running  # gamma = 1
            returns[index] = running
    answer_targets: list[Mapping[str, Any]] = []
    world_targets: list[Mapping[str, Any]] = []
    action_targets: list[Mapping[str, Any]] = []
    return_targets: list[Mapping[str, Any]] = []
    excluded: list[Mapping[str, Any]] = []
    qualified = 0
    for index, step in enumerate(steps):
        submitted = step.action.get("kind") == "submit"
        if submitted:
            success = bool(step.feedback.get("success", False))
            if success:
                answer_targets.append({
                    "step_index": step.step_index,
                    "answer": step.action.get("value", step.action.get("container")),
                    "teacher_identity": teacher_identity})
            else:
                excluded.append({"step_index": step.step_index,
                                 "reason": "failed_submission_not_gold"})
        if step.feedback:
            world_targets.append({
                "step_index": step.step_index,
                "next_feedback": dict(step.feedback),
                "terminated": step.terminated,
                "truncated": step.truncated,
                "teacher_identity": teacher_identity})
            qualified += 1
        if returns[index] is not None:
            return_targets.append({
                "step_index": step.step_index, "return": returns[index],
                "horizon": len(steps) - index, "reward_protocol": reward_protocol})
        else:
            excluded.append({"step_index": step.step_index,
                             "reason": "truncated_no_complete_return"})
        if not submitted:
            action_targets.append({
                "step_index": step.step_index,
                "action": dict(step.action),
                "teacher_identity": teacher_identity})
    return CompiledEpisode(episode_id=episode.episode_id,
                           answer_targets=tuple(answer_targets),
                           world_targets=tuple(world_targets),
                           action_targets=tuple(action_targets),
                           return_targets=tuple(return_targets),
                           qualified_steps=qualified, excluded_steps=tuple(excluded))


class Teacher:
    """Declared training-only teacher: identity, hidden-information access
    and search cost are recorded; it sees only public prefixes."""

    def __init__(self, identity: str, action_distribution: Callable[
            [Mapping[str, Any], Sequence[Mapping[str, Any]]], Sequence[float]],
            hidden_access: str = "none", search_cost: float = 0.0) -> None:
        self.identity = identity
        self._distribution = action_distribution
        self.hidden_access = hidden_access
        self.search_cost = search_cost

    def action_distribution(self, public_prefix: Mapping[str, Any],
                            legal_actions: Sequence[Mapping[str, Any]]) -> TeacherTarget:
        raw = list(self._distribution(public_prefix, legal_actions))
        if len(raw) != len(legal_actions):
            raise CompilationError("teacher distribution length mismatch")
        total = sum(raw)
        if total <= 0:
            # Ties distribute mass uniformly (A6 teacher rule).
            raw = [1.0 / len(legal_actions)] * len(legal_actions)
            total = 1.0
        distribution = [value / total for value in raw]
        return TeacherTarget(
            kind="action",
            payload={"prefix": dict(public_prefix),
                     "distribution": distribution,
                     "legal_actions": [dict(action) for action in legal_actions]},
            teacher_identity=self.identity)
