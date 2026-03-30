"""
================================================================================
FILE: agent/core/planner.py
PROJECT: Agent Loop — 45K
PURPOSE: Break goals into executable steps with dependency graphs
================================================================================

The planner takes a GoalSpec and produces an ExecutionPlan:
  - List of Steps with clear instructions
  - Dependency graph (what must complete before what)
  - Estimated effort per step
  - Tool assignments per step
  - Parallel execution opportunities

Replanning is supported: when a step fails or new info arrives,
the planner can patch the existing plan rather than rebuilding from scratch.
================================================================================
"""

import re
import time
import logging
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from enum import Enum

from goal import GoalSpec, GoalRisk

logger = logging.getLogger(__name__)


class StepStatus(Enum):
    PENDING   = "pending"
    READY     = "ready"      # All deps satisfied — can execute
    RUNNING   = "running"
    DONE      = "done"
    FAILED    = "failed"
    SKIPPED   = "skipped"


class StepPriority(Enum):
    CRITICAL = 1   # Must succeed or whole plan fails
    HIGH     = 2
    NORMAL   = 3
    LOW      = 4   # Nice-to-have


@dataclass
class Step:
    """One executable unit in a plan."""
    step_id:      str
    title:        str
    instruction:  str            # What the executor should do
    tools:        List[str]      # Which tools this step needs
    depends_on:   List[str]      # step_ids that must be DONE before this runs
    priority:     StepPriority   = StepPriority.NORMAL
    status:       StepStatus     = StepStatus.PENDING
    retry_budget: int            = 3
    retries_used: int            = 0
    timeout_secs: int            = 120
    success_criteria: str        = ""
    result:       Optional[str]  = None
    error:        Optional[str]  = None
    started_at:   Optional[float] = None
    completed_at: Optional[float] = None
    reasoning:    str            = ""    # Why this step exists (chain of thought)
    metadata:     Dict           = field(default_factory=dict)

    @property
    def duration_secs(self) -> Optional[float]:
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return None

    def is_ready(self, completed_ids: Set[str]) -> bool:
        return all(dep in completed_ids for dep in self.depends_on)

    def to_dict(self) -> Dict:
        return {
            "step_id":    self.step_id,
            "title":      self.title,
            "instruction": self.instruction,
            "tools":      self.tools,
            "depends_on": self.depends_on,
            "priority":   self.priority.name,
            "status":     self.status.value,
            "retry_budget": self.retry_budget,
            "retries_used": self.retries_used,
            "result":     self.result,
            "error":      self.error,
            "duration":   self.duration_secs,
        }


@dataclass
class ExecutionPlan:
    """Complete plan for achieving a goal."""
    plan_id:    str
    goal_id:    str
    steps:      List[Step]
    created_at: float = field(default_factory=time.time)
    version:    int   = 1   # Increments on replan
    notes:      str   = ""  # Planner's reasoning about the overall approach

    def get_step(self, step_id: str) -> Optional[Step]:
        return next((s for s in self.steps if s.step_id == step_id), None)

    def completed_ids(self) -> Set[str]:
        return {s.step_id for s in self.steps if s.status == StepStatus.DONE}

    def failed_ids(self) -> Set[str]:
        return {s.step_id for s in self.steps if s.status == StepStatus.FAILED}

    def ready_steps(self) -> List[Step]:
        """Return steps whose dependencies are all done and are pending/ready to run."""
        done = self.completed_ids()
        return [
            s for s in self.steps
            if s.status in (StepStatus.PENDING, StepStatus.READY) and s.is_ready(done)
        ]

    def is_complete(self) -> bool:
        return all(s.status in (StepStatus.DONE, StepStatus.SKIPPED) for s in self.steps)

    def has_critical_failure(self) -> bool:
        return any(
            s.status == StepStatus.FAILED and s.priority == StepPriority.CRITICAL
            for s in self.steps
        )

    def progress(self) -> Dict:
        total   = len(self.steps)
        done    = sum(1 for s in self.steps if s.status == StepStatus.DONE)
        failed  = sum(1 for s in self.steps if s.status == StepStatus.FAILED)
        running = sum(1 for s in self.steps if s.status == StepStatus.RUNNING)
        return {
            "total": total, "done": done, "failed": failed,
            "running": running, "pending": total - done - failed - running,
            "pct": round(100 * done / total, 1) if total else 0,
        }

    def summary(self) -> str:
        p = self.progress()
        lines = [
            f"Plan v{self.version} | {p['pct']}% complete",
            f"Steps: {p['done']}/{p['total']} done, {p['failed']} failed, {p['running']} running",
        ]
        for s in self.steps:
            icon = {"done": "✓", "failed": "✗", "running": "→",
                    "pending": "○", "ready": "●", "skipped": "-"}.get(s.status.value, "?")
            lines.append(f"  {icon} [{s.step_id}] {s.title}")
        return "\n".join(lines)

    def to_dict(self) -> Dict:
        return {
            "plan_id":    self.plan_id,
            "goal_id":    self.goal_id,
            "version":    self.version,
            "steps":      [s.to_dict() for s in self.steps],
            "created_at": self.created_at,
            "notes":      self.notes,
            "progress":   self.progress(),
        }


# ──────────────────────────────────────────────────────────────────────────────
# PLAN TEMPLATES
# Heuristic templates for common goal patterns
# ──────────────────────────────────────────────────────────────────────────────

def _research_plan(goal: GoalSpec) -> List[Dict]:
    """Template for research/comparison goals."""
    return [
        {"title": "Define search strategy",
         "instruction": f"Identify the 3-5 best search queries to answer: {goal.objective}. Store them with memory_tool.",
         "tools": ["memory_tool"], "depends_on": [], "priority": StepPriority.CRITICAL,
         "reasoning": "Need targeted queries before searching to avoid wasted calls"},
        {"title": "Gather primary sources",
         "instruction": f"Use web_search to gather information. Run each query identified in step 1. Summarize each result.",
         "tools": ["web_search", "summarizer", "memory_tool"], "depends_on": ["s1"],
         "priority": StepPriority.CRITICAL,
         "reasoning": "Collect raw data before synthesizing"},
        {"title": "Analyze and compare",
         "instruction": "Analyze the gathered information. Identify key differences, rankings, tradeoffs. Use calculator for any numeric comparisons.",
         "tools": ["calculator", "memory_tool", "code_executor"], "depends_on": ["s2"],
         "priority": StepPriority.CRITICAL,
         "reasoning": "Transform raw data into structured insights"},
        {"title": "Write report",
         "instruction": f"Write a comprehensive report answering: {goal.objective}. Save to file_manager as report.md.",
         "tools": ["file_manager", "memory_tool"], "depends_on": ["s3"],
         "priority": StepPriority.CRITICAL,
         "reasoning": "Deliver the final artifact"},
        {"title": "Verify completeness",
         "instruction": f"Review the report. Check each success criterion: {goal.success_criteria}. Flag any gaps.",
         "tools": ["file_manager", "memory_tool"], "depends_on": ["s4"],
         "priority": StepPriority.HIGH,
         "reasoning": "Ensure quality before marking done"},
    ]


def _coding_plan(goal: GoalSpec) -> List[Dict]:
    """Template for code generation goals."""
    return [
        {"title": "Clarify requirements",
         "instruction": f"List all requirements for: {goal.objective}. Store as memory_tool store requirements <list>",
         "tools": ["memory_tool"], "depends_on": [], "priority": StepPriority.CRITICAL,
         "reasoning": "Ambiguous requirements → bad code"},
        {"title": "Design solution",
         "instruction": "Design the approach: functions needed, data structures, edge cases. Store design in memory.",
         "tools": ["memory_tool"], "depends_on": ["s1"], "priority": StepPriority.HIGH,
         "reasoning": "Plan before implementing"},
        {"title": "Implement code",
         "instruction": f"Write Python code that {goal.objective}. Test it with code_executor.",
         "tools": ["code_executor", "memory_tool"], "depends_on": ["s2"],
         "priority": StepPriority.CRITICAL,
         "reasoning": "Core implementation"},
        {"title": "Test edge cases",
         "instruction": "Run edge case tests: empty input, invalid input, boundary values. Fix any failures.",
         "tools": ["code_executor"], "depends_on": ["s3"], "priority": StepPriority.HIGH,
         "reasoning": "Untested code is broken code"},
        {"title": "Save final code",
         "instruction": "Save the working, tested code to file_manager. Include usage instructions.",
         "tools": ["file_manager"], "depends_on": ["s4"], "priority": StepPriority.CRITICAL,
         "reasoning": "Deliver the artifact"},
    ]


def _analysis_plan(goal: GoalSpec) -> List[Dict]:
    """Template for analysis/calculation goals."""
    return [
        {"title": "Gather data",
         "instruction": f"Collect all data needed for: {goal.objective}. Use web_search and memory_tool.",
         "tools": ["web_search", "memory_tool"], "depends_on": [], "priority": StepPriority.CRITICAL,
         "reasoning": "Need data before analysis"},
        {"title": "Process and calculate",
         "instruction": "Run calculations, process data, build any needed tables. Use calculator and code_executor.",
         "tools": ["calculator", "code_executor", "memory_tool"], "depends_on": ["s1"],
         "priority": StepPriority.CRITICAL,
         "reasoning": "Core analysis"},
        {"title": "Summarize findings",
         "instruction": f"Summarize results clearly. Answer: {goal.objective}. Save to file.",
         "tools": ["summarizer", "file_manager", "memory_tool"], "depends_on": ["s2"],
         "priority": StepPriority.CRITICAL,
         "reasoning": "Deliver clear answer"},
    ]


def _generic_plan(goal: GoalSpec) -> List[Dict]:
    """Fallback template for any goal."""
    return [
        {"title": "Understand and plan",
         "instruction": f"Break down this goal into sub-tasks: {goal.objective}. What information is needed? What tools?",
         "tools": ["memory_tool", "task_manager"], "depends_on": [], "priority": StepPriority.CRITICAL,
         "reasoning": "Understand before acting"},
        {"title": "Gather necessary information",
         "instruction": "Collect any information, data, or resources needed for this goal.",
         "tools": ["web_search", "memory_tool", "file_manager"], "depends_on": ["s1"],
         "priority": StepPriority.HIGH,
         "reasoning": "Information gathering phase"},
        {"title": "Execute core task",
         "instruction": f"Execute the primary task: {goal.objective}",
         "tools": goal.resources, "depends_on": ["s2"], "priority": StepPriority.CRITICAL,
         "reasoning": "Main execution"},
        {"title": "Verify and deliver",
         "instruction": f"Verify the goal is achieved. Check: {goal.success_criteria}. Save results.",
         "tools": ["file_manager", "memory_tool"], "depends_on": ["s3"],
         "priority": StepPriority.CRITICAL,
         "reasoning": "Quality check and delivery"},
    ]


# ──────────────────────────────────────────────────────────────────────────────
# PLANNER
# ──────────────────────────────────────────────────────────────────────────────

class Planner:
    """
    Converts a GoalSpec into an ExecutionPlan.

    Strategy:
      1. Classify goal type (research / coding / analysis / generic)
      2. Select and customize a template
      3. Assign dependencies, priorities, tools, timeouts
      4. Return plan ready for the executor

    Replanning:
      When a step fails, the planner can patch the plan:
        - Insert a diagnostic step before the retry
        - Adjust the retry step's instruction based on the failure
        - Mark dependent steps as needing re-evaluation
    """

    def __init__(self, available_tools: List[str] = None):
        self.available_tools = available_tools or [
            "web_search", "code_executor", "file_manager",
            "calculator", "memory_tool", "summarizer", "task_manager",
        ]

    def plan(self, goal: GoalSpec) -> ExecutionPlan:
        """
        Create an ExecutionPlan for the given goal.

        Args:
            goal: An APPROVED GoalSpec from the goal interpreter.

        Returns:
            ExecutionPlan with steps, dependencies, and metadata.

        Raises:
            ValueError: If goal is not APPROVED.
        """
        from goal import GoalStatus
        if goal.status not in (GoalStatus.APPROVED, GoalStatus.ACTIVE):
            raise ValueError(
                f"Cannot plan goal with status {goal.status.value}. "
                "Goal must be APPROVED first."
            )

        plan_id  = "P_" + hashlib.md5(f"{goal.goal_id}{time.time()}".encode()).hexdigest()[:8]
        goal_type = self._classify(goal)

        logger.info(f"Planning goal {goal.goal_id} as type '{goal_type}'")

        # Select template
        template_fns = {
            "research": _research_plan,
            "coding":   _coding_plan,
            "analysis": _analysis_plan,
            "generic":  _generic_plan,
        }
        raw_steps = template_fns.get(goal_type, _generic_plan)(goal)

        # Build Step objects
        steps = []
        id_map = {}   # index → step_id for dependency rewriting
        for i, raw in enumerate(raw_steps):
            step_id = f"s{i+1}"
            id_map[f"s{i+1}"] = step_id

            # Filter tools to only available ones
            tools = [t for t in raw.get("tools", []) if t in self.available_tools]
            if not tools:
                tools = ["memory_tool"]

            # Rewrite dependencies
            deps = [id_map.get(d, d) for d in raw.get("depends_on", [])]

            # Timeout based on tools used
            timeout = 300 if "web_search" in tools else (
                       60  if "code_executor" in tools else 30
            )

            step = Step(
                step_id=step_id,
                title=raw["title"],
                instruction=raw["instruction"],
                tools=tools,
                depends_on=deps,
                priority=raw.get("priority", StepPriority.NORMAL),
                timeout_secs=timeout,
                success_criteria=raw.get("success_criteria", "Step completes without error"),
                reasoning=raw.get("reasoning", ""),
                retry_budget=3 if raw.get("priority") == StepPriority.CRITICAL else 2,
            )
            steps.append(step)

        # Mark initially ready steps
        done: Set[str] = set()
        for step in steps:
            if step.is_ready(done):
                step.status = StepStatus.READY

        plan = ExecutionPlan(
            plan_id=plan_id,
            goal_id=goal.goal_id,
            steps=steps,
            notes=(
                f"Goal type: {goal_type}\n"
                f"Total steps: {len(steps)}\n"
                f"Risk level: {goal.risk.value}\n"
                f"Estimated duration: {len(steps) * 2}-{len(steps) * 8} minutes"
            ),
        )

        logger.info(
            f"Plan {plan_id} created: {len(steps)} steps, "
            f"type={goal_type}, risk={goal.risk.value}"
        )
        return plan

    def replan(self, plan: ExecutionPlan, failed_step: Step, new_info: str = "") -> ExecutionPlan:
        """
        Patch an existing plan after a step failure.

        Strategy:
          - If step has retries remaining: adjust instruction and mark PENDING again
          - If no retries: insert a diagnostic step, then adjusted retry
          - Update dependent steps if the failure changes what they need

        Args:
            plan:        The current ExecutionPlan.
            failed_step: The Step that failed.
            new_info:    Any new information from the failure diagnosis.

        Returns:
            Updated plan (same object, modified in place, version incremented).
        """
        plan.version += 1
        step = plan.get_step(failed_step.step_id)
        if step is None:
            logger.error(f"replan: step {failed_step.step_id} not found in plan")
            return plan

        if step.retries_used < step.retry_budget:
            # Simple retry: adjust instruction
            step.retries_used += 1
            step.status = StepStatus.PENDING
            if new_info:
                step.instruction = (
                    f"{step.instruction}\n\n"
                    f"[RETRY {step.retries_used}/{step.retry_budget}] "
                    f"Previous attempt failed: {step.error}. "
                    f"New information: {new_info}. "
                    f"Try a different approach."
                )
            logger.info(
                f"Replanned step {step.step_id}: retry "
                f"{step.retries_used}/{step.retry_budget}"
            )
        else:
            # Out of retries: insert a fallback/diagnostic step
            diag_id = f"{step.step_id}_diag"
            if not plan.get_step(diag_id):
                diag = Step(
                    step_id=diag_id,
                    title=f"Diagnose failure of {step.title}",
                    instruction=(
                        f"The step '{step.title}' failed {step.retry_budget} times. "
                        f"Last error: {step.error}. "
                        f"Diagnose why it failed. What alternative approaches exist? "
                        f"Store findings in memory_tool. "
                        f"If no path forward: document the blocker clearly."
                    ),
                    tools=["memory_tool", "web_search"],
                    depends_on=[d for d in step.depends_on],
                    priority=StepPriority.CRITICAL,
                    retry_budget=1,
                    reasoning="Diagnose unrecoverable failure to inform human escalation",
                )
                # Insert diagnostic step before the failed step's position
                idx = plan.steps.index(step)
                plan.steps.insert(idx, diag)

                # Update downstream deps to also depend on diag
                for s in plan.steps:
                    if step.step_id in s.depends_on and s.step_id != step.step_id:
                        s.depends_on.append(diag_id)

            logger.warning(
                f"Step {step.step_id} exhausted retries — "
                f"inserted diagnostic step {diag_id}"
            )

        return plan

    def _classify(self, goal: GoalSpec) -> str:
        """Classify goal type for template selection."""
        text = goal.raw_goal.lower()
        if re.search(r'\b(code|script|program|implement|function|class|api|debug)\b', text):
            return "coding"
        if re.search(r'\b(research|compare|find|search|best|recommend|review|analyze|survey)\b', text):
            return "research"
        if re.search(r'\b(calculate|compute|math|statistics|data|analyze|chart|graph)\b', text):
            return "analysis"
        return "generic"
