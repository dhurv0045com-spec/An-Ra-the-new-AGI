"""Deterministic skill-boundary workflow execution."""

from __future__ import annotations

import time
from typing import Callable

from robotics.contracts import SkillGoal, SkillResult, Workflow, WorkflowState


class WorkflowExecutor:
    def __init__(
        self,
        dispatch: Callable[[SkillGoal], SkillResult],
        check_condition: Callable[[str], bool],
        *,
        max_skills: int = 10,
    ) -> None:
        self.dispatch = dispatch
        self.check_condition = check_condition
        self.max_skills = int(max_skills)

    def execute(self, workflow: Workflow) -> Workflow:
        if len(workflow.skills) > self.max_skills:
            raise ValueError(f"Workflow exceeds {self.max_skills} skills.")
        workflow.state = WorkflowState.RUNNING
        for index in range(workflow.current_index, len(workflow.skills)):
            goal = workflow.skills[index]
            if not all(self.check_condition(condition) for condition in goal.preconditions):
                workflow.state = WorkflowState.FAILED
                workflow.trace.append(
                    {"index": index, "skill": goal.skill_name, "error": "precondition_failed"}
                )
                return workflow
            started = time.time()
            result = self.dispatch(goal)
            workflow.trace.append(
                {
                    "index": index,
                    "skill": goal.skill_name,
                    "success": result.success,
                    "error": result.error,
                    "duration_seconds": time.time() - started,
                }
            )
            if not result.success:
                workflow.state = WorkflowState.FAILED
                return workflow
            if not all(self.check_condition(condition) for condition in goal.postconditions):
                workflow.state = WorkflowState.FAILED
                workflow.trace[-1]["error"] = "postcondition_failed"
                return workflow
            workflow.current_index = index + 1
        workflow.state = WorkflowState.COMPLETED
        return workflow

    @staticmethod
    def cancel(workflow: Workflow) -> Workflow:
        workflow.state = WorkflowState.CANCELLED
        return workflow
