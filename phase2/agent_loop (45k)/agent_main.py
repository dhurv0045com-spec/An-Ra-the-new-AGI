"""
================================================================================
FILE: agent.py
PROJECT: Agent Loop — 45K
PURPOSE: Master entry point — the complete agent loop in one clean class
================================================================================

Usage:

    # Python API
    from agent import Agent
    agent = Agent()
    result = agent.run("Research the best GPU under $500 and write a comparison")
    print(result["output"])

    # CLI
    python agent.py --goal "Research the best GPU under $500"
    python agent.py --goal "..." --approve-each-step
    python agent.py --list-tools
    python agent.py --review-last
    python agent.py --continuous --check-in-every 30

================================================================================
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# Core modules
from registry    import ToolRegistry, get_registry
from builtin     import register_all_tools
from goal        import GoalInterpreter, GoalSpec, GoalStatus, GoalRisk
from planner     import Planner, ExecutionPlan, StepPriority
from executor    import Executor, HumanEscalation
from dispatcher  import Dispatcher
from monitor     import AgentMonitor
from reasoning   import ReasoningEngine
from evaluator   import GoalEvaluator
from coordinator import MultiAgentCoordinator

logger = logging.getLogger(__name__)


def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


class Agent:
    """
    The complete agent loop.

    Components:
      GoalInterpreter  — parse and validate natural language goals
      Planner          — break goals into step graphs
      Executor         — run steps with retry/recovery
      Dispatcher       — route natural language to tools
      AgentMonitor     — detect loops, timeouts, drift
      ReasoningEngine  — chain of thought before decisions
      GoalEvaluator    — learn from outcomes over time
      ToolRegistry     — all available tools

    State:
      - All state is in-memory within one session
      - Goals, evaluations, and memories persist to disk
      - Can resume across sessions via memory_tool and task_manager
    """

    def __init__(
        self,
        approve_each_step:   bool  = False,
        verbose:             bool  = True,
        log_level:           str   = "INFO",
        checkin_interval:    int   = 30,
        max_tool_calls:      int   = 200,
        memory_manager:      Any   = None,
    ):
        _setup_logging(log_level)
        self.approve_each_step = approve_each_step
        self.verbose           = verbose
        self.memory_manager    = memory_manager

        # ── Build tool registry ─────────────────────────────────────────
        self.registry = get_registry()
        register_all_tools(self.registry)

        # Wire approval for dangerous tools
        self.registry.set_approval_callback(self._approve_dangerous_tool)

        # ── Core components ─────────────────────────────────────────────
        self.interpreter = GoalInterpreter()
        self.planner     = Planner(available_tools=list(self.registry._tools.keys()))
        self.dispatcher  = Dispatcher(self.registry)
        self.monitor     = AgentMonitor(
            escalation_fn=self._handle_monitor_alert,
            checkin_interval=checkin_interval,
            max_tool_calls=max_tool_calls,
        )
        self.reasoning   = ReasoningEngine()
        self.evaluator   = GoalEvaluator()

        step_approval = self._approve_step if approve_each_step else None
        self.executor = Executor(
            registry=self.registry,
            planner=self.planner,
            approval=step_approval,
            escalation=self._handle_escalation,
            verbose=verbose,
            memory_manager=self.memory_manager
        )

        # ── State ───────────────────────────────────────────────────────
        self._last_plan:  Optional[ExecutionPlan] = None
        self._last_goal:  Optional[GoalSpec]      = None
        self._last_result: Optional[Dict]         = None
        self._history:    List[Dict]              = []

        logger.info(f"Agent initialized with {len(self.registry)} tools")

    # ── Primary interface ─────────────────────────────────────────────────

    def run(
        self,
        goal:        str,
        parallel:    bool  = False,
        clarify:     bool  = True,
        show_plan:   bool  = True,
    ) -> Dict[str, Any]:
        """
        Execute a natural language goal end-to-end.

        Pipeline:
          interpret → [clarify] → plan → [show plan] → execute → evaluate → report

        Args:
            goal:      Natural language goal string.
            parallel:  Run independent steps in parallel.
            clarify:   Prompt user for clarifications if goal is ambiguous.
            show_plan: Print the plan before executing (recommended).

        Returns:
            Dict with: success, output, plan_summary, evaluation, duration
        """
        run_start = time.time()
        self._print(f"\n{'='*64}")
        self._print(f"  AGENT STARTING")
        self._print(f"  Goal: {goal[:100]}")
        self._print(f"{'='*64}\n")

        # ── 1. Interpret ─────────────────────────────────────────────────
        spec = self.interpreter.interpret(goal)
        self._last_goal = spec
        self._print(spec.summary())

        if spec.status == GoalStatus.REJECTED:
            return self._fail_result(
                goal=goal,
                reason=spec.clarifications_needed[0] if spec.clarifications_needed else "Goal rejected",
                duration=time.time() - run_start,
            )

        # ── 2. Clarify (if needed and enabled) ───────────────────────────
        if spec.status == GoalStatus.FLAGGED and clarify and spec.clarifications_needed:
            self._print("\nClarification needed:")
            for q in spec.clarifications_needed:
                self._print(f"  ? {q}")

            if self._is_interactive():
                answers = {}
                for q in spec.clarifications_needed:
                    try:
                        ans = input(f"\n  Answer: ").strip()
                        answers[q] = ans
                    except (EOFError, KeyboardInterrupt):
                        answers[q] = "proceed"
                spec = self.interpreter.clarify(spec, answers)
            else:
                # Non-interactive: proceed anyway with HIGH risk flagging
                self._print("  (Non-interactive mode: proceeding with flagged goal)")
                spec.status = GoalStatus.APPROVED

        # ── 3. Chain-of-thought reasoning about approach ──────────────────
        chain = self.reasoning.reason_about_step(
            step_title=spec.objective,
            instruction=spec.raw_goal,
            available_tools=spec.resources,
        )
        self._print(f"\n{chain.formatted()}\n")

        # ── 4. Plan ───────────────────────────────────────────────────────
        spec.status = GoalStatus.ACTIVE
        try:
            plan = self.planner.plan(spec)
        except Exception as e:
            return self._fail_result(goal=goal, reason=f"Planning failed: {e}",
                                     duration=time.time() - run_start)

        self._last_plan = plan

        if show_plan:
            self._print(f"\n{'─'*64}")
            self._print("EXECUTION PLAN:")
            self._print(plan.summary())
            self._print(plan.notes)
            self._print(f"{'─'*64}\n")

            # High-risk: require explicit approval before executing
            if spec.risk == GoalRisk.HIGH and self._is_interactive():
                try:
                    confirm = input("Proceed with this plan? [y/N]: ").strip().lower()
                    if confirm not in ("y", "yes"):
                        return self._fail_result(goal=goal, reason="User declined plan",
                                                  duration=time.time() - run_start)
                except (EOFError, KeyboardInterrupt):
                    pass

        # ── 5. Execute ────────────────────────────────────────────────────
        self.monitor.start()
        try:
            exec_result = self.executor.execute_plan(plan, spec, parallel=parallel)
        except HumanEscalation as e:
            self.monitor.stop()
            return self._fail_result(
                goal=goal,
                reason=f"Human escalation required: {e}",
                plan_summary=plan.summary(),
                duration=time.time() - run_start,
            )
        finally:
            self.monitor.stop()

        # ── 6. Evaluate ───────────────────────────────────────────────────
        eval_ = self.evaluator.evaluate(
            goal_id=spec.goal_id,
            objective=spec.objective,
            exec_result=exec_result,
            tool_summary=self.registry.summary(),
        )

        # ── 7. Compose output ─────────────────────────────────────────────
        # Collect the best output from the final step(s)
        outputs = exec_result.get("outputs", {})
        final_output = "\n\n".join(
            f"[{k}]: {v}" for k, v in outputs.items() if v
        ) or "Goal executed but no text output was produced."

        self._last_result = {
            "success":      exec_result.get("success", False),
            "output":       final_output,
            "plan_summary": plan.summary(),
            "evaluation":   eval_.report(),
            "duration":     time.time() - run_start,
            "tool_stats":   self.registry.summary(),
        }

        self._history.append({
            "goal":    goal,
            "result":  self._last_result,
            "eval":    eval_.to_dict(),
        })

        self._print(f"\n{'='*64}")
        self._print("GOAL COMPLETE")
        self._print(eval_.report())
        self._print(f"{'='*64}\n")

        return self._last_result

    def run_parallel_goals(self, goals: List[str]) -> Dict:
        """
        Run multiple independent goals in parallel using sub-agents.
        Each goal gets its own agent thread.
        """
        self._print(f"\nLaunching {len(goals)} parallel sub-agents...")

        coord = MultiAgentCoordinator(
            execute_fn=lambda g: self.run(g, show_plan=False, clarify=False),
            max_parallel=min(len(goals), 3),
            timeout_secs=300,
        )
        results = coord.run_parallel(goals)
        synthesis = coord.synthesize(results, synthesis_question="Combined results:")

        return {
            "success":   any(r.success for r in results),
            "output":    synthesis,
            "results":   [{"goal": r.goal, "success": r.success, "output": r.output[:200]}
                          for r in results],
        }

    # ── Inspection / review ───────────────────────────────────────────────

    def review_last(self) -> str:
        """Return a detailed review of the last goal execution."""
        if not self._last_result:
            return "No goal has been executed yet."

        lines = [
            "LAST EXECUTION REVIEW",
            "=" * 50,
            f"Goal: {self._last_goal.objective if self._last_goal else 'unknown'}",
            f"Success: {self._last_result['success']}",
            f"Duration: {self._last_result['duration']:.1f}s",
            "",
            "Plan summary:",
            self._last_result.get("plan_summary", ""),
            "",
            "Evaluation:",
            self._last_result.get("evaluation", ""),
            "",
            "Tool statistics:",
            json.dumps(self._last_result.get("tool_stats", {}), indent=2),
            "",
            "Reasoning chains:",
        ]
        for chain in self.reasoning.get_history()[-3:]:
            lines.append(chain.formatted())

        return "\n".join(lines)

    def list_tools(self) -> str:
        """Return formatted list of all available tools."""
        tools = self.registry.list_tools()
        lines = [f"Available tools ({len(tools)}):"]
        lines.append("─" * 50)
        for t in tools:
            lines.append(f"  {t['name']} [{t['safety']}]")
            lines.append(f"    {t['description']}")
            if t.get("examples"):
                lines.append(f"    Example: {t['examples'][0]}")
            lines.append("")
        return "\n".join(lines)

    def performance_profile(self) -> Dict:
        """Return agent's performance profile across all sessions."""
        return self.evaluator.performance_profile()

    # ── Callbacks ─────────────────────────────────────────────────────────

    def _approve_step(self, step) -> bool:
        """Approval callback for step-level approval mode."""
        self._print(f"\n[APPROVAL REQUIRED]")
        self._print(f"  Step: {step.title}")
        self._print(f"  Tools: {step.tools}")
        self._print(f"  Instruction: {step.instruction[:200]}")
        try:
            ans = input("  Execute? [Y/n]: ").strip().lower()
            return ans in ("", "y", "yes")
        except (EOFError, KeyboardInterrupt):
            return True   # Default approve in non-interactive

    def _approve_dangerous_tool(self, tool_name: str, input_text: str) -> bool:
        """Approval callback for DANGEROUS tool calls."""
        self._print(f"\n⚠  DANGEROUS TOOL: {tool_name}")
        self._print(f"   Input: {input_text[:200]}")
        try:
            ans = input("   Allow? [y/N]: ").strip().lower()
            return ans in ("y", "yes")
        except (EOFError, KeyboardInterrupt):
            return False

    def _handle_escalation(self, message: str, step) -> None:
        """Called when a step cannot be recovered automatically."""
        self._print(f"\n🚨 ESCALATION: {message}")
        self._print(f"   Step: {step.title}")
        self._print("   Human intervention required to continue.")

    def _handle_monitor_alert(self, message: str) -> None:
        """Called when the monitor detects a pathological behavior."""
        self._print(f"\n⚠  MONITOR ALERT: {message}")

    # ── Helpers ───────────────────────────────────────────────────────────

    def _fail_result(self, goal: str, reason: str, plan_summary: str = "",
                      duration: float = 0.0) -> Dict:
        return {
            "success":      False,
            "output":       f"Goal not completed: {reason}",
            "plan_summary": plan_summary,
            "evaluation":   f"FAILED: {reason}",
            "duration":     duration,
            "tool_stats":   self.registry.summary(),
        }

    def _print(self, msg: str) -> None:
        if self.verbose:
            print(msg)
        logger.debug(msg)

    def _is_interactive(self) -> bool:
        return sys.stdin.isatty()


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _cli():
    parser = argparse.ArgumentParser(
        prog="agent.py",
        description="Agent Loop — give it a goal and walk away",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python agent.py --goal "Research the best GPU under $500 and write a comparison"
  python agent.py --goal "Write a Python function to merge two sorted lists and test it"
  python agent.py --goal "Calculate compound interest on $10000 at 7% for 20 years"
  python agent.py --list-tools
  python agent.py --review-last
  python agent.py --goal "..." --approve-each-step
        """,
    )

    parser.add_argument("--goal",             default=None, help="Goal for the agent to execute")
    parser.add_argument("--approve-each-step", action="store_true",
                        help="Require approval before each step")
    parser.add_argument("--parallel",         action="store_true",
                        help="Run independent steps in parallel")
    parser.add_argument("--list-tools",       action="store_true",
                        help="List all available tools and exit")
    parser.add_argument("--review-last",      action="store_true",
                        help="Show detailed review of last execution")
    parser.add_argument("--performance",      action="store_true",
                        help="Show agent performance profile")
    parser.add_argument("--continuous",       action="store_true",
                        help="Run in continuous mode — prompt for goals repeatedly")
    parser.add_argument("--check-in-every",   default=30, type=int,
                        help="Monitor check-in interval in seconds (default: 30)")
    parser.add_argument("--log-level",        default="WARNING",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--no-plan",          action="store_true",
                        help="Don't show plan before executing")
    parser.add_argument("--output-file",      default=None,
                        help="Save final output to this file")

    args = parser.parse_args()

    agent = Agent(
        approve_each_step=args.approve_each_step,
        verbose=True,
        log_level=args.log_level,
        checkin_interval=args.check_in_every,
    )

    if args.list_tools:
        print(agent.list_tools())
        return

    if args.review_last:
        print(agent.review_last())
        return

    if args.performance:
        print(json.dumps(agent.performance_profile(), indent=2))
        return

    if args.continuous:
        print("Continuous mode. Type 'quit' to exit, 'review' to see last result.\n")
        while True:
            try:
                goal = input("Goal: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.")
                break

            if not goal:
                continue
            if goal.lower() in ("quit", "exit", "q"):
                break
            if goal.lower() == "review":
                print(agent.review_last())
                continue

            result = agent.run(goal, parallel=args.parallel, show_plan=not args.no_plan)
            if args.output_file:
                Path(args.output_file).write_text(result["output"])
                print(f"\nOutput saved to: {args.output_file}")
        return

    if args.goal:
        result = agent.run(
            args.goal,
            parallel=args.parallel,
            show_plan=not args.no_plan,
        )
        if args.output_file and result.get("output"):
            Path(args.output_file).write_text(result["output"])
            print(f"\nOutput saved to: {args.output_file}")
        return

    # No arguments — show help
    parser.print_help()


if __name__ == "__main__":
    _cli()
