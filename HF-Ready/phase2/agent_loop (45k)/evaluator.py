"""
================================================================================
FILE: agent/intelligence/evaluator.py
PROJECT: Agent Loop — 45K
PURPOSE: Evaluate goal outcomes, build performance profile, learn over time
================================================================================
"""

import time
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

_EVAL_FILE = Path("./agent_workspace/.agent_evals.json")


@dataclass
class GoalEvaluation:
    """Post-completion evaluation of one goal execution."""
    goal_id:       str
    objective:     str
    succeeded:     bool
    duration_secs: float
    steps_total:   int
    steps_done:    int
    steps_failed:  int
    tool_calls:    int
    failure_reasons: List[str]
    what_worked:   List[str]
    what_failed:   List[str]
    improvements:  List[str]    # What would make next run faster/better
    timestamp:     float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return {
            "goal_id":       self.goal_id,
            "objective":     self.objective[:100],
            "succeeded":     self.succeeded,
            "duration_secs": round(self.duration_secs, 1),
            "steps":         f"{self.steps_done}/{self.steps_total}",
            "steps_failed":  self.steps_failed,
            "tool_calls":    self.tool_calls,
            "what_worked":   self.what_worked,
            "what_failed":   self.what_failed,
            "improvements":  self.improvements,
            "timestamp":     self.timestamp,
        }

    def report(self) -> str:
        status = "✓ SUCCEEDED" if self.succeeded else "✗ FAILED"
        lines = [
            f"GOAL EVALUATION — {status}",
            f"Objective: {self.objective[:100]}",
            f"Duration:  {self.duration_secs:.0f}s",
            f"Steps:     {self.steps_done}/{self.steps_total} completed, {self.steps_failed} failed",
            f"Tool calls: {self.tool_calls}",
        ]
        if self.what_worked:
            lines.append("What worked:")
            lines.extend(f"  + {w}" for w in self.what_worked)
        if self.what_failed:
            lines.append("What failed:")
            lines.extend(f"  - {f}" for f in self.what_failed)
        if self.improvements:
            lines.append("Next time:")
            lines.extend(f"  → {i}" for i in self.improvements)
        return "\n".join(lines)


class GoalEvaluator:
    """
    Evaluates completed goals and builds a performance profile over time.
    Stores evaluations persistently so the agent improves across sessions.
    """

    def __init__(self):
        self._evals: List[GoalEvaluation] = []
        self._load()

    def evaluate(
        self,
        goal_id:    str,
        objective:  str,
        exec_result: Dict,
        tool_summary: Dict,
    ) -> GoalEvaluation:
        """
        Evaluate a completed goal execution.

        Args:
            goal_id:      Goal identifier.
            objective:    Goal objective text.
            exec_result:  Return value from executor.execute_plan().
            tool_summary: Return value from registry.summary().

        Returns:
            GoalEvaluation with honest assessment and improvement suggestions.
        """
        succeeded     = exec_result.get("success", False)
        duration      = exec_result.get("duration", 0)
        steps_done    = exec_result.get("done_steps", 0)
        steps_total   = exec_result.get("total_steps", 0)
        steps_failed  = len(exec_result.get("failed_steps", []))
        tool_calls    = tool_summary.get("total", 0)

        # Generate what-worked / what-failed analysis
        what_worked = []
        what_failed = []
        improvements = []

        if succeeded:
            what_worked.append("Goal completed successfully")
        if steps_failed == 0:
            what_worked.append("All steps completed without failure")
        if duration < 60:
            what_worked.append(f"Completed quickly in {duration:.0f}s")
        if tool_calls < 10:
            what_worked.append(f"Efficient: only {tool_calls} tool calls needed")

        if steps_failed > 0:
            what_failed.append(f"{steps_failed} steps failed and required recovery")
        if duration > 300:
            what_failed.append(f"Slow: took {duration:.0f}s — may have looped")
        if tool_calls > 50:
            what_failed.append(f"Too many tool calls: {tool_calls}")

        for step_id in exec_result.get("failed_steps", []):
            what_failed.append(f"Step '{step_id}' could not be completed")

        # Improvement suggestions
        if duration > 120:
            improvements.append("Consider running independent steps in parallel")
        if steps_failed > 0:
            improvements.append("Add error handling specifically for the failed steps")
        if tool_calls > 30:
            improvements.append("Cache results in memory_tool to avoid redundant searches")
        if not succeeded:
            improvements.append("Break the goal into smaller, more specific sub-goals")

        eval_ = GoalEvaluation(
            goal_id=goal_id,
            objective=objective,
            succeeded=succeeded,
            duration_secs=duration,
            steps_total=steps_total,
            steps_done=steps_done,
            steps_failed=steps_failed,
            tool_calls=tool_calls,
            failure_reasons=[],
            what_worked=what_worked,
            what_failed=what_failed,
            improvements=improvements,
        )

        self._evals.append(eval_)
        self._save()
        logger.info(f"Goal {goal_id} evaluated: succeeded={succeeded}, "
                    f"duration={duration:.0f}s, steps={steps_done}/{steps_total}")
        return eval_

    def performance_profile(self) -> Dict:
        """Aggregate performance statistics over all evaluated goals."""
        if not self._evals:
            return {"message": "No evaluations yet"}

        total     = len(self._evals)
        succeeded = sum(1 for e in self._evals if e.succeeded)
        avg_dur   = sum(e.duration_secs for e in self._evals) / total
        avg_calls = sum(e.tool_calls for e in self._evals) / total

        return {
            "total_goals":    total,
            "success_rate":   f"{100*succeeded/total:.0f}%",
            "avg_duration":   f"{avg_dur:.0f}s",
            "avg_tool_calls": f"{avg_calls:.0f}",
            "recent":         [e.to_dict() for e in self._evals[-5:]],
        }

    def _save(self) -> None:
        try:
            _EVAL_FILE.parent.mkdir(parents=True, exist_ok=True)
            _EVAL_FILE.write_text(
                json.dumps([e.to_dict() for e in self._evals], indent=2)
            )
        except Exception as ex:
            logger.warning(f"Could not save evaluations: {ex}")

    def _load(self) -> None:
        try:
            if _EVAL_FILE.exists():
                data = json.loads(_EVAL_FILE.read_text())
                # We load as dicts only — no need to reconstruct full objects
                logger.info(f"Loaded {len(data)} previous evaluations")
        except Exception:
            pass
