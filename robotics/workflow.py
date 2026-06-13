"""Deterministic skill-boundary workflow execution."""

from __future__ import annotations

import time
from typing import Callable
import json
from pathlib import Path

from robotics.contracts import SkillGoal, SkillResult, Workflow, WorkflowState
from robotics.world_model import PredictiveWorldModel, WorldModelCodec
from engine.feature_flags import is_enabled


class WorkflowExecutor:
    def __init__(
        self,
        dispatch: Callable[[SkillGoal], SkillResult],
        check_condition: Callable[[str], bool],
        *,
        max_skills: int = 10,
        world_model: PredictiveWorldModel | None = None,
        world_model_codec: WorldModelCodec | None = None,
        world_model_active: bool = False,
        authorize: Callable[[SkillGoal], bool] | None = None,
        cbf_check: Callable[[SkillGoal], bool] | None = None,
        transition_path: str | Path | None = None,
    ) -> None:
        self.dispatch = dispatch
        self.check_condition = check_condition
        self.max_skills = int(max_skills)
        self.world_model = world_model
        self.codec = world_model_codec or WorldModelCodec()
        self.world_model_active = bool(world_model_active)
        self.authorize = authorize or (lambda goal: not goal.approval_required)
        self.cbf_check = cbf_check or (lambda goal: True)
        self.transition_path = Path(transition_path) if transition_path else None

    def execute(self, workflow: Workflow, current_state: dict[str, object] | None = None) -> Workflow:
        if not is_enabled("agent_loop"):
            raise RuntimeError("Agent-loop feature is disabled at workflow execution.")
        if len(workflow.skills) > self.max_skills:
            raise ValueError(f"Workflow exceeds {self.max_skills} skills.")
        workflow.state = WorkflowState.RUNNING
        state = dict(current_state or {})
        for index in range(workflow.current_index, len(workflow.skills)):
            goal = workflow.skills[index]
            if not self.authorize(goal):
                workflow.state = WorkflowState.FAILED
                workflow.trace.append(
                    {
                        "index": index,
                        "skill": goal.skill_name,
                        "source_mission": goal.source_mission,
                        "error": "authorization_denied",
                    }
                )
                return workflow
            if not self.cbf_check(goal):
                workflow.state = WorkflowState.FAILED
                workflow.trace.append(
                    {
                        "index": index,
                        "skill": goal.skill_name,
                        "source_mission": goal.source_mission,
                        "error": "cbf_safety_block",
                    }
                )
                return workflow
            if not all(self.check_condition(condition) for condition in goal.preconditions):
                workflow.state = WorkflowState.FAILED
                workflow.trace.append(
                    {
                        "index": index,
                        "skill": goal.skill_name,
                        "source_mission": goal.source_mission,
                        "error": "precondition_failed",
                    }
                )
                return workflow
            prediction_summary = None
            if self.world_model is not None and self.world_model_active:
                device = next(self.world_model.parameters()).device
                state_vector = self.codec.encode_state(state).to(device).unsqueeze(0)
                action_vector = self.codec.encode_action(
                    {"skill": goal.skill_name, "parameters": goal.parameters}
                ).to(device).unsqueeze(0)
                with __import__("torch").no_grad():
                    prediction = self.world_model(state_vector, action_vector)
                success_probability = float(
                    __import__("torch").sigmoid(prediction["reward"]).mean().item()
                )
                uncertainty = float(prediction["epistemic_uncertainty"].mean().item())
                prediction_summary = {
                    "predicted_success": success_probability,
                    "uncertainty": uncertainty,
                }
                if success_probability < 0.3 or uncertainty > 0.7:
                    workflow.state = WorkflowState.FAILED
                    workflow.trace.append(
                        {
                            "index": index,
                            "skill": goal.skill_name,
                            "source_mission": goal.source_mission,
                            "error": "world_model_requests_replan",
                            **prediction_summary,
                            "suggest_replan": True,
                        }
                    )
                    return workflow
            started = time.time()
            result = self.dispatch(goal)
            workflow.trace.append(
                {
                    "index": index,
                    "skill": goal.skill_name,
                    "source_mission": goal.source_mission,
                    "success": result.success,
                    "error": result.error,
                    "duration_seconds": time.time() - started,
                    "artifacts": result.artifacts,
                    "prediction": prediction_summary,
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
            next_state = {
                **state,
                "last_skill": goal.skill_name,
                "success": result.success,
                "observations": [
                    observation.natural_language_summary
                    for observation in result.observations
                ],
            }
            if self.transition_path is not None:
                self.transition_path.parent.mkdir(parents=True, exist_ok=True)
                with self.transition_path.open("a", encoding="utf-8") as stream:
                    stream.write(
                        json.dumps(
                            {
                                "timestamp": time.time(),
                                "mode": "simulation_or_shadow",
                                "state": state,
                                "action": {
                                    "skill": goal.skill_name,
                                    "parameters": goal.parameters,
                                },
                                "next_state": next_state,
                                "reward": 1.0 if result.success else 0.0,
                                "terminal": index == len(workflow.skills) - 1,
                            },
                            default=str,
                        )
                        + "\n"
                    )
            state = next_state
        workflow.state = WorkflowState.COMPLETED
        return workflow

    @staticmethod
    def cancel(workflow: Workflow) -> Workflow:
        workflow.state = WorkflowState.CANCELLED
        return workflow
