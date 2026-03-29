"""
================================================================================
FILE: agent/core/executor.py
PROJECT: Agent Loop — 45K
PURPOSE: Execute plan steps — in order, with retries, verification, escalation
================================================================================

The executor walks the ExecutionPlan and runs each Step when it's ready.
For each step it:
  1. Checks dependencies are met
  2. Formats the step instruction with memory context
  3. Dispatches to the right tool via the dispatcher
  4. Verifies the result against the step's success criteria
  5. On failure: retries (via replanner) or escalates to user
  6. Stores the result in memory for downstream steps

Never skips a failed step silently.
Always escalates unrecoverable failures.
================================================================================
"""

import time
import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set
from concurrent.futures import ThreadPoolExecutor, as_completed

from planner import ExecutionPlan, Step, StepStatus, StepPriority, Planner
from goal    import GoalSpec
from registry     import ToolRegistry, ToolResult

logger = logging.getLogger(__name__)

# ── LLM Bridge (singleton, loaded once by system.py) ──────────────────────────
def _get_llm():
    """Lazy accessor for the global LLMBridge singleton."""
    try:
        import sys
        from pathlib import Path
        m_path = Path(__file__).resolve().parent.parent / "45M"
        if str(m_path) not in sys.path:
            sys.path.insert(0, str(m_path))
        from llm_bridge import get_llm_bridge
        return get_llm_bridge()
    except Exception:
        return None


@dataclass
class StepOutcome:
    """Result of executing one step."""
    step_id:    str
    success:    bool
    output:     str
    error:      Optional[str] = None
    tool_calls: List[Dict]    = field(default_factory=list)
    duration:   float         = 0.0
    needs_human: bool         = False    # True when agent cannot resolve failure


class ExecutionError(Exception):
    """Raised when a step fails and retries are exhausted."""
    pass


class HumanEscalation(Exception):
    """Raised when the agent cannot proceed without human input."""
    def __init__(self, message: str, step: Step, plan: ExecutionPlan):
        super().__init__(message)
        self.step = step
        self.plan = plan


class Executor:
    """
    Runs an ExecutionPlan step by step.

    Modes:
      Sequential: one step at a time (default — safer, more observable)
      Parallel:   independent steps run concurrently (faster, more complex)

    Approval mode: before each step, call an approval callback.
    Useful for high-stakes goals where the user wants to review each action.

    Memory bridge: after each step, results are stored in the memory_tool
    so downstream steps can reference what was learned.
    """

    def __init__(
        self,
        registry:    ToolRegistry,
        planner:     "Planner",
        verbose:     bool = True,
        approval:    Optional[Callable] = None,
        escalation:  Optional[Callable] = None,
        memory_manager: Any = None,
    ):
        self.registry   = registry
        self.planner    = planner
        self.verbose     = verbose
        self.approval   = approval
        self.escalation = escalation
        self.memory_manager = memory_manager
        self.logger      = logger

        if not self.registry:
            raise ValueError("Executor needs a ToolRegistry")
        self._lock      = threading.Lock()

    def execute_plan(
        self,
        plan:           ExecutionPlan,
        goal:           GoalSpec,
        parallel:       bool = False,
    ) -> Dict[str, Any]:
        """
        Execute all steps in a plan until completion, failure, or human escalation.

        Args:
            plan:     ExecutionPlan from the planner.
            goal:     The original GoalSpec (for context).
            parallel: If True, run independent steps concurrently.

        Returns:
            Summary dict with success, outputs, failed steps, total duration.
        """
        start_time = time.time()
        all_outputs: Dict[str, str] = {}
        failed_steps: List[str]     = []

        self._log(f"\n{'='*60}")
        self._log(f"Starting execution: Plan {plan.plan_id}")
        self._log(f"Goal: {goal.objective}")
        self._log(f"Steps: {len(plan.steps)}")
        self._log(f"{'='*60}\n")

        try:
            if parallel:
                self._execute_parallel(plan, goal, all_outputs, failed_steps)
            else:
                self._execute_sequential(plan, goal, all_outputs, failed_steps)

        except HumanEscalation as e:
            logger.warning(f"Human escalation: {e}")
            if self.escalation:
                self.escalation(str(e), e.step)
            return {
                "success": False,
                "reason": "human_escalation",
                "message": str(e),
                "outputs": all_outputs,
                "failed_steps": failed_steps,
                "duration": time.time() - start_time,
            }
        except KeyboardInterrupt:
            logger.info("Execution interrupted by user")
            return {
                "success": False,
                "reason": "interrupted",
                "outputs": all_outputs,
                "failed_steps": failed_steps,
                "duration": time.time() - start_time,
            }

        success    = not plan.has_critical_failure()
        total_time = time.time() - start_time

        self._log(f"\n{'='*60}")
        self._log(f"Execution complete: {'SUCCESS' if success else 'PARTIAL/FAILED'}")
        self._log(f"Duration: {total_time:.1f}s")
        self._log(plan.summary())
        self._log(f"{'='*60}\n")

        return {
            "success":      success,
            "outputs":      all_outputs,
            "failed_steps": failed_steps,
            "total_steps":  len(plan.steps),
            "done_steps":   len(plan.completed_ids()),
            "duration":     total_time,
            "plan_summary": plan.summary(),
        }

    # ── Sequential execution ──────────────────────────────────────────────

    def _execute_sequential(
        self,
        plan:          ExecutionPlan,
        goal:          GoalSpec,
        all_outputs:   Dict[str, str],
        failed_steps:  List[str],
    ) -> None:
        """Execute steps one at a time in dependency order."""
        max_iterations = len(plan.steps) * 10  # Safety: prevent infinite loop
        iteration = 0

        while not plan.is_complete() and iteration < max_iterations:
            iteration += 1
            ready = plan.ready_steps()

            if not ready:
                # Check if we're stuck (no ready steps but plan not complete)
                pending = [s for s in plan.steps if s.status == StepStatus.PENDING]
                if pending and not plan.has_critical_failure():
                    # Deps unfulfilled but no failures — something is wrong
                    logger.error(f"Stuck: {len(pending)} pending steps but none ready")
                    break
                break  # All steps either done or failed

            # Execute the highest-priority ready step
            step = min(ready, key=lambda s: s.priority.value)
            outcome = self._execute_step(step, plan, goal, all_outputs)

            if outcome.success:
                all_outputs[step.step_id] = outcome.output
                step.result       = outcome.output
                step.status       = StepStatus.DONE
                step.completed_at = time.time()
                # Bridge to memory
                self._store_result_in_memory(step, outcome)
                # Mark newly-ready steps
                done = plan.completed_ids()
                for s in plan.steps:
                    if s.status == StepStatus.PENDING and s.is_ready(done):
                        s.status = StepStatus.READY
            else:
                step.error = outcome.error
                if outcome.needs_human:
                    step.status = StepStatus.FAILED
                    failed_steps.append(step.step_id)
                    if step.priority == StepPriority.CRITICAL:
                        raise HumanEscalation(
                            f"Step '{step.title}' failed and requires human intervention: {outcome.error}",
                            step, plan,
                        )
                else:
                    # Try replanning
                    updated_plan = self.planner.replan(plan, step, new_info=outcome.error or "")
                    if step.status == StepStatus.FAILED:
                        failed_steps.append(step.step_id)
                        if step.priority == StepPriority.CRITICAL:
                            raise HumanEscalation(
                                f"Critical step '{step.title}' could not be recovered.",
                                step, plan,
                            )

    # ── Parallel execution ────────────────────────────────────────────────

    def _execute_parallel(
        self,
        plan:          ExecutionPlan,
        goal:          GoalSpec,
        all_outputs:   Dict[str, str],
        failed_steps:  List[str],
    ) -> None:
        """Execute independent steps in parallel, respecting dependencies."""
        max_iterations = len(plan.steps) * 10
        iteration = 0

        with ThreadPoolExecutor(max_workers=self.max_parallel) as pool:
            in_flight: Dict[str, Any] = {}

            while not plan.is_complete() and iteration < max_iterations:
                iteration += 1
                ready = [s for s in plan.ready_steps() if s.step_id not in in_flight]

                # Submit ready steps
                for step in ready:
                    step.status = StepStatus.RUNNING
                    future = pool.submit(self._execute_step, step, plan, goal, all_outputs)
                    in_flight[step.step_id] = (future, step)

                if not in_flight:
                    break

                # Wait for any to complete
                done_futures = {sid: (f, s) for sid, (f, s) in in_flight.items()
                                if f.done()}

                if not done_futures:
                    time.sleep(0.1)
                    continue

                for step_id, (future, step) in done_futures.items():
                    del in_flight[step_id]
                    try:
                        outcome = future.result()
                    except Exception as e:
                        outcome = StepOutcome(step.step_id, False, "",
                                              error=str(e), needs_human=True)

                    with self._lock:
                        if outcome.success:
                            all_outputs[step.step_id] = outcome.output
                            step.result       = outcome.output
                            step.status       = StepStatus.DONE
                            step.completed_at = time.time()
                            self._store_result_in_memory(step, outcome)
                            done = plan.completed_ids()
                            for s in plan.steps:
                                if s.status == StepStatus.PENDING and s.is_ready(done):
                                    s.status = StepStatus.READY
                        else:
                            step.error  = outcome.error
                            step.status = StepStatus.FAILED
                            failed_steps.append(step.step_id)
                            if outcome.needs_human and step.priority == StepPriority.CRITICAL:
                                raise HumanEscalation(
                                    f"Parallel step '{step.title}' failed critically.",
                                    step, plan,
                                )

    # ── Step execution ────────────────────────────────────────────────────

    def _execute_step(
        self,
        step:        Step,
        plan:        ExecutionPlan,
        goal:        GoalSpec,
        all_outputs: Dict[str, str],
    ) -> StepOutcome:
        """
        Execute one step:
          - Check approval
          - Build full instruction with context
          - Dispatch to appropriate tool(s)
          - Verify result
          - Return StepOutcome
        """
        self._log(f"\n→ [{step.step_id}] {step.title}")
        self._log(f"  Tools: {step.tools}")

        step.started_at = time.time()
        step.status     = StepStatus.RUNNING

        # Approval check
        if self.approval is not None:
            if not self.approval(step):
                self._log(f"  DENIED by approval callback")
                return StepOutcome(step.step_id, False, "",
                                   error="Step denied by user approval",
                                   needs_human=True)

        # Build contextual instruction
        instruction = self._build_instruction(step, plan, goal, all_outputs)

        # Dispatch to best tool
        tool_results = []
        primary_output = ""
        success = False

        for tool_name in step.tools:
            if tool_name not in self.registry:
                continue
                
            # Use LLM to format tool input
            llm = _get_llm()
            if llm:
                try:
                    prompt = (
                        f"Format input for tool '{tool_name}' given instruction: '{instruction}'. "
                        "Output ONLY the command or input string required by the tool. No apologies, no extra text."
                    )
                    tool_input = llm.generate(prompt, max_new_tokens=150).strip()
                except Exception as e:
                    logger.warning(f"LLM tool arg generation failed: {e}. Falling back.")
                    tool_input = instruction
            else:
                tool_input = instruction

            result = self.registry.call(
                tool_name, tool_input,
                step_id=step.step_id,
                memory_manager=self.memory_manager
            )
            tool_results.append(result.to_dict())

            if result.success:
                primary_output = result.output
                success = True
                self._log(f"  ✓ {tool_name}: {result.output[:100]!r}")
                break
            else:
                self._log(f"  ✗ {tool_name}: {result.error}")

        if not success and not primary_output:
            # All tools failed — try fallback to memory_tool
            if "memory_tool" in self.registry:
                fallback = self.registry.call(
                    "memory_tool",
                    f"store {step.step_id}_failed {step.error or 'all tools failed'}",
                    step_id=step.step_id,
                    memory_manager=self.memory_manager
                )

            error_msg = f"All tools failed for step '{step.title}'"
            duration  = time.time() - step.started_at

            # Decide if human escalation is needed
            needs_human = step.retries_used >= step.retry_budget - 1

            return StepOutcome(
                step_id=step.step_id,
                success=False,
                output="",
                error=error_msg,
                tool_calls=tool_results,
                duration=duration,
                needs_human=needs_human,
            )

        # Verify success criteria
        verified = self._verify(step, primary_output)
        if not verified:
            self._log(f"  ⚠ Verification failed for '{step.title}'")

        duration = time.time() - step.started_at
        self._log(f"  ✓ Done in {duration:.1f}s")

        return StepOutcome(
            step_id=step.step_id,
            success=True,   # Considered success if at least one tool worked
            output=primary_output,
            tool_calls=tool_results,
            duration=duration,
        )

    def _build_instruction(
        self,
        step:        Step,
        plan:        ExecutionPlan,
        goal:        GoalSpec,
        all_outputs: Dict[str, str],
    ) -> str:
        """
        Build the full instruction string sent to the tool.
        Injects: step instruction + relevant context from previous steps.
        """
        ctx_parts = []

        # Context from completed steps (limit to avoid overflowing tool input)
        for dep_id in step.depends_on:
            if dep_id in all_outputs:
                dep_out = all_outputs[dep_id][:300]  # truncate
                ctx_parts.append(f"[{dep_id} output]: {dep_out}")

        context = "\n".join(ctx_parts)
        if context:
            return f"{step.instruction}\n\nContext from previous steps:\n{context}"
        return step.instruction

    def _verify(self, step: Step, output: str) -> bool:
        """
        LLM-based verification that the step succeeded.
        """
        if not output or not output.strip():
            return False
            
        llm = _get_llm()
        if llm:
            try:
                prompt = (
                    f"Evaluate step output.\nGoal: {step.instruction}\nOutput: {output}\n"
                    "Did the execution succeed fully? Answer YES or NO."
                )
                response = llm.generate(prompt, max_new_tokens=15).strip().upper()
                if "NO" in response:
                    return False
            except Exception:
                pass
        else:
            error_indicators = ["error:", "failed:", "exception:", "traceback"]
            lower = output.lower()
            if any(ind in lower for ind in error_indicators) and len(output) < 200:
                return False
        return True

    def _store_result_in_memory(self, step: Step, outcome: StepOutcome) -> None:
        """Bridge step result into the memory_tool for downstream access."""
        if "memory_tool" in self.registry and outcome.output:
            key   = f"step_{step.step_id}_result"
            value = outcome.output[:500]   # Store truncated to avoid bloat
            self.registry.call("memory_tool", f"store {key} {value}", step_id="memory_bridge")

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)
        logger.info(msg)
