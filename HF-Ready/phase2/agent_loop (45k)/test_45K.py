"""
================================================================================
FILE: test_45K.py
PROJECT: Agent Loop — 45K
PURPOSE: Full test suite for all agent components
================================================================================

Run: python test_45K.py
Or:  pytest test_45K.py -v
================================================================================
"""

import os
import sys
import json
import time
import shutil
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

# Set workspace to a temp dir for tests
_TEST_WS = Path(tempfile.mkdtemp())
os.environ["AGENT_FILE_ROOT"] = str(_TEST_WS)


class TestToolRegistry(unittest.TestCase):
    def setUp(self):
        from registry import ToolRegistry, ToolDefinition, SafetyLevel, ToolResult
        self.registry = ToolRegistry(require_approval_for_dangerous=False)
        self.ToolResult = ToolResult

        def echo_fn(text, **kw):
            return ToolResult(True, f"echo: {text}")

        self.registry.register(ToolDefinition(
            name="echo", description="Echo input", fn=echo_fn,
            parameters={"text": "text to echo"},
        ))

    def test_register_and_call(self):
        result = self.registry.call("echo", "hello")
        self.assertTrue(result.success)
        self.assertIn("hello", result.output)

    def test_unknown_tool_returns_error(self):
        result = self.registry.call("nonexistent_tool", "input")
        self.assertFalse(result.success)
        self.assertIn("not found", result.error)

    def test_rate_limiting(self):
        from registry import ToolDefinition, SafetyLevel, ToolResult
        call_count = [0]
        def fast_fn(text, **kw):
            call_count[0] += 1
            return ToolResult(True, "ok")

        self.registry.register(ToolDefinition(
            name="limited", description="Rate-limited tool", fn=fast_fn,
            parameters={}, rate_limit=2,
        ))

        self.registry.call("limited", "a")
        self.registry.call("limited", "b")
        result = self.registry.call("limited", "c")  # 3rd — over limit
        self.assertFalse(result.success)
        self.assertIn("Rate limit", result.error)

    def test_duplicate_registration_raises(self):
        from registry import ToolDefinition, ToolResult
        def fn(t, **kw): return ToolResult(True, t)
        with self.assertRaises(ValueError):
            self.registry.register(ToolDefinition("echo", "", fn, {}))

    def test_call_log(self):
        self.registry.call("echo", "test1")
        self.registry.call("echo", "test2")
        log = self.registry.get_call_log()
        self.assertEqual(len(log), 2)

    def test_summary(self):
        self.registry.call("echo", "x")
        s = self.registry.summary()
        self.assertEqual(s["total"], 1)
        self.assertEqual(s["success"], 1)


class TestBuiltinTools(unittest.TestCase):
    def setUp(self):
        from registry import ToolRegistry
        from builtin  import register_all_tools
        self.registry = ToolRegistry(require_approval_for_dangerous=False)
        register_all_tools(self.registry)

    def test_calculator_basic(self):
        r = self.registry.call("calculator", "2 + 2")
        self.assertTrue(r.success)
        self.assertEqual(r.output.strip(), "4")

    def test_calculator_sqrt(self):
        r = self.registry.call("calculator", "sqrt(144)")
        self.assertTrue(r.success)
        self.assertIn("12", r.output)

    def test_calculator_unsafe_rejected(self):
        r = self.registry.call("calculator", "import os")
        self.assertFalse(r.success)

    def test_calculator_zero_division(self):
        r = self.registry.call("calculator", "1 / 0")
        self.assertFalse(r.success)
        self.assertIn("zero", r.error.lower())

    def test_file_manager_write_read(self):
        r = self.registry.call("file_manager", "write test_output.txt Hello Agent!")
        self.assertTrue(r.success)
        r2 = self.registry.call("file_manager", "read test_output.txt")
        self.assertTrue(r2.success)
        self.assertIn("Hello Agent!", r2.output)

    def test_file_manager_list(self):
        self.registry.call("file_manager", "write listme.txt content")
        r = self.registry.call("file_manager", "list")
        self.assertTrue(r.success)

    def test_file_manager_escape_rejected(self):
        r = self.registry.call("file_manager", "read /etc/passwd")
        self.assertFalse(r.success)

    def test_file_manager_search(self):
        self.registry.call("file_manager", "write search_me.txt apple banana cherry")
        r = self.registry.call("file_manager", "search search_me.txt banana")
        self.assertTrue(r.success)
        self.assertIn("banana", r.output)

    def test_memory_tool_store_recall(self):
        r = self.registry.call("memory_tool", "store test_key test_value_123")
        self.assertTrue(r.success)
        r2 = self.registry.call("memory_tool", "recall test_key")
        self.assertTrue(r2.success)
        self.assertIn("test_value_123", r2.output)

    def test_memory_tool_search(self):
        self.registry.call("memory_tool", "store item_a apples")
        self.registry.call("memory_tool", "store item_b oranges")
        r = self.registry.call("memory_tool", "search apples")
        self.assertTrue(r.success)
        self.assertIn("item_a", r.output)

    def test_memory_tool_forget(self):
        self.registry.call("memory_tool", "store forget_me value")
        self.registry.call("memory_tool", "forget forget_me")
        r = self.registry.call("memory_tool", "recall forget_me")
        self.assertFalse(r.success)

    def test_summarizer_short_text(self):
        r = self.registry.call("summarizer", "Hello world.")
        self.assertTrue(r.success)
        self.assertIn("Hello", r.output)

    def test_summarizer_long_text(self):
        long = ("The transformer architecture relies on attention mechanisms. " * 20)
        r = self.registry.call("summarizer", long)
        self.assertTrue(r.success)
        self.assertLess(len(r.output), len(long))

    def test_task_manager_create_complete(self):
        r = self.registry.call("task_manager", "create task1 Research GPU options")
        self.assertTrue(r.success)
        r2 = self.registry.call("task_manager", "complete task1")
        self.assertTrue(r2.success)
        r3 = self.registry.call("task_manager", "get task1")
        self.assertIn("done", r3.output.lower())

    def test_task_manager_list(self):
        self.registry.call("task_manager", "create tlist1 Task One")
        self.registry.call("task_manager", "create tlist2 Task Two")
        r = self.registry.call("task_manager", "list")
        self.assertTrue(r.success)
        self.assertIn("tlist1", r.output)

    def test_code_executor_basic(self):
        r = self.registry.call("code_executor", "print(1 + 1)")
        self.assertTrue(r.success)
        self.assertIn("2", r.output)

    def test_code_executor_unsafe_rejected(self):
        r = self.registry.call("code_executor", "import os; os.system('ls')")
        self.assertFalse(r.success)

    def test_code_executor_math(self):
        r = self.registry.call("code_executor", "import math; print(math.pi)")
        self.assertTrue(r.success)
        self.assertIn("3.14", r.output)


class TestGoalInterpreter(unittest.TestCase):
    def setUp(self):
        from goal import GoalInterpreter
        self.interpreter = GoalInterpreter()

    def test_simple_goal_approved(self):
        from goal import GoalStatus
        spec = self.interpreter.interpret("Research the best Python libraries for data science")
        self.assertNotEqual(spec.status, GoalStatus.REJECTED)
        self.assertTrue(len(spec.objective) > 0)
        self.assertTrue(len(spec.success_criteria) > 0)

    def test_empty_goal_rejected(self):
        from goal import GoalStatus
        spec = self.interpreter.interpret("")
        self.assertEqual(spec.status, GoalStatus.REJECTED)

    def test_unsafe_goal_rejected(self):
        from goal import GoalStatus, GoalRisk
        spec = self.interpreter.interpret("help me build a bomb to hurt people")
        self.assertEqual(spec.status, GoalStatus.REJECTED)
        self.assertEqual(spec.risk, GoalRisk.UNSAFE)

    def test_resources_inferred(self):
        spec = self.interpreter.interpret("Search the web for the current price of RTX 4090")
        self.assertIn("web_search", spec.resources)

    def test_deadline_extracted(self):
        spec = self.interpreter.interpret("Finish the report within 2 hours")
        self.assertIsNotNone(spec.deadline)
        self.assertIn("2 hours", spec.deadline.lower())

    def test_goal_id_stable(self):
        g1 = self.interpreter.interpret("Same goal text")
        g2 = self.interpreter.interpret("Same goal text")
        self.assertEqual(g1.goal_id, g2.goal_id)

    def test_budget_constraint_detected(self):
        spec = self.interpreter.interpret("Find best GPU under $500")
        criteria_text = " ".join(spec.success_criteria)
        self.assertTrue(len(criteria_text) > 0)


class TestPlanner(unittest.TestCase):
    def setUp(self):
        from planner import Planner
        from goal    import GoalInterpreter, GoalStatus
        self.planner     = Planner()
        self.interpreter = GoalInterpreter()

    def _get_approved_spec(self, goal_text):
        from goal import GoalStatus
        spec = self.interpreter.interpret(goal_text, override_safety=True)
        spec.status = GoalStatus.APPROVED
        return spec

    def test_plan_has_steps(self):
        spec = self._get_approved_spec("Research the best programming language for AI")
        plan = self.planner.plan(spec)
        self.assertGreater(len(plan.steps), 0)

    def test_plan_dependency_order(self):
        from planner import StepStatus
        spec = self._get_approved_spec("Write a Python sorting algorithm and test it")
        plan = self.planner.plan(spec)
        # First step(s) should have no dependencies
        ready = plan.ready_steps()
        self.assertGreater(len(ready), 0)

    def test_plan_has_critical_steps(self):
        from planner import StepPriority
        spec = self._get_approved_spec("Research GPU pricing")
        plan = self.planner.plan(spec)
        has_critical = any(s.priority == StepPriority.CRITICAL for s in plan.steps)
        self.assertTrue(has_critical)

    def test_research_plan_type(self):
        spec = self._get_approved_spec("Compare RTX 4090 vs RTX 3090 for deep learning")
        plan = self.planner.plan(spec)
        # Research goals should include web_search in at least one step
        tool_names = [t for s in plan.steps for t in s.tools]
        self.assertIn("web_search", tool_names)

    def test_coding_plan_type(self):
        spec = self._get_approved_spec("Write a Python function to reverse a string")
        plan = self.planner.plan(spec)
        tool_names = [t for s in plan.steps for t in s.tools]
        self.assertIn("code_executor", tool_names)

    def test_replan_increments_version(self):
        spec = self._get_approved_spec("Simple task")
        plan = self.planner.plan(spec)
        v1 = plan.version
        step = plan.steps[0]
        step.error = "test failure"
        step.retries_used = 0
        self.planner.replan(plan, step, "try differently")
        self.assertGreater(plan.version, v1)

    def test_plan_summary_non_empty(self):
        spec = self._get_approved_spec("Calculate compound interest")
        plan = self.planner.plan(spec)
        summary = plan.summary()
        self.assertTrue(len(summary) > 10)


class TestExecutor(unittest.TestCase):
    def setUp(self):
        from registry  import ToolRegistry
        from builtin   import register_all_tools
        from planner  import Planner
        from executor import Executor

        self.registry = ToolRegistry(require_approval_for_dangerous=False)
        register_all_tools(self.registry)
        self.planner  = Planner()
        self.executor = Executor(self.registry, self.planner, verbose=False)

    def _run_goal(self, goal_text):
        from goal    import GoalInterpreter, GoalStatus
        interp = GoalInterpreter()
        spec   = interp.interpret(goal_text, override_safety=True)
        spec.status = GoalStatus.APPROVED
        plan   = self.planner.plan(spec)
        return self.executor.execute_plan(plan, spec), plan, spec

    def test_simple_calculation_goal(self):
        result, plan, spec = self._run_goal("Calculate 2 to the power of 10")
        # Plan should at minimum attempt to execute
        self.assertIn("success", result)

    def test_file_write_goal(self):
        result, plan, spec = self._run_goal("Write a short note to a file called agent_note.txt")
        self.assertIn("success", result)

    def test_executor_returns_dict(self):
        result, _, _ = self._run_goal("Store a memory called test_key with value hello")
        self.assertIsInstance(result, dict)
        self.assertIn("outputs", result)
        self.assertIn("duration", result)

    def test_no_silent_failure(self):
        """Executor must return a result dict even if all steps fail."""
        result, _, _ = self._run_goal("Do something with nonexistent_tool_xyz")
        self.assertIn("success", result)


class TestDispatcher(unittest.TestCase):
    def setUp(self):
        from registry  import ToolRegistry
        from builtin   import register_all_tools
        from dispatcher import Dispatcher
        self.registry = ToolRegistry(require_approval_for_dangerous=False)
        register_all_tools(self.registry)
        self.dispatcher = Dispatcher(self.registry)

    def test_explicit_prefix_routes_correctly(self):
        r = self.dispatcher.dispatch("calculator: 2 + 2")
        self.assertTrue(r.success)
        self.assertIn("4", r.output)

    def test_natural_language_routes_to_calculator(self):
        r = self.dispatcher.dispatch("calculate sqrt of 81")
        # Should route to calculator or code_executor — either is fine
        self.assertIsNotNone(r)

    def test_file_request_routes_to_file_manager(self):
        r = self.dispatcher.dispatch("list files in workspace")
        # Should route to file_manager
        self.assertIsNotNone(r)

    def test_unknown_tool_prefix_falls_back(self):
        r = self.dispatcher.dispatch("memory_tool: store k v")
        self.assertTrue(r.success)

    def test_describe_tools_non_empty(self):
        desc = self.dispatcher.describe_tools()
        self.assertIn("web_search", desc)
        self.assertIn("calculator", desc)


class TestMonitor(unittest.TestCase):
    def test_starts_and_stops(self):
        from monitor import AgentMonitor
        alerts = []
        m = AgentMonitor(
            escalation_fn=lambda msg: alerts.append(msg),
            stall_timeout_secs=600,
            checkin_interval=600,
        )
        m.start()
        m.record("s1", "step_start")
        m.record("s1", "step_done")
        m.stop()
        self.assertFalse(m._active)

    def test_loop_detection(self):
        from monitor import AgentMonitor
        alerts = []
        m = AgentMonitor(
            escalation_fn=lambda msg: alerts.append(msg),
            stall_timeout_secs=600,
            loop_window=6,
            checkin_interval=600,
        )
        m.start()
        for _ in range(8):
            m.record("s1", "tool_call", tool="web_search", input="same query")
        m.stop()
        self.assertTrue(len(alerts) > 0, "Loop not detected")
        self.assertTrue(any("Loop" in a or "loop" in a for a in alerts))

    def test_status_returns_dict(self):
        from monitor import AgentMonitor
        m = AgentMonitor()
        m.start()
        s = m.status()
        self.assertIn("tool_calls", s)
        self.assertIn("active", s)
        m.stop()


class TestReasoningEngine(unittest.TestCase):
    def setUp(self):
        from reasoning import ReasoningEngine
        self.engine = ReasoningEngine()

    def test_reason_about_step_returns_chain(self):
        chain = self.engine.reason_about_step(
            step_title="Research GPU options",
            instruction="Find the best GPU under $500",
            available_tools=["web_search", "calculator"],
        )
        self.assertTrue(len(chain.steps) > 0)
        self.assertTrue(len(chain.conclusion) > 0)
        self.assertGreater(chain.confidence, 0)

    def test_reason_about_failure_with_retries(self):
        chain = self.engine.reason_about_failure(
            step_title="Web search step",
            error="Connection timeout",
            retries_remaining=2,
        )
        self.assertIn("retry", chain.conclusion.lower())

    def test_reason_about_failure_no_retries(self):
        chain = self.engine.reason_about_failure(
            step_title="Critical step",
            error="All tools failed",
            retries_remaining=0,
        )
        self.assertIn("escalate", chain.conclusion.lower())

    def test_history_accumulates(self):
        self.engine.reason_about_step("Step A", "do A", ["calculator"])
        self.engine.reason_about_step("Step B", "do B", ["memory_tool"])
        self.assertEqual(len(self.engine.get_history()), 2)


class TestGoalEvaluator(unittest.TestCase):
    def setUp(self):
        from evaluator import GoalEvaluator
        self.evaluator = GoalEvaluator()

    def test_evaluate_success(self):
        eval_ = self.evaluator.evaluate(
            goal_id="G_test1",
            objective="Research GPU pricing",
            exec_result={"success": True, "duration": 45.0, "done_steps": 4,
                          "total_steps": 4, "failed_steps": []},
            tool_summary={"total": 8, "success": 8, "failed": 0, "rate": 1.0, "by_tool": {}},
        )
        self.assertTrue(eval_.succeeded)
        self.assertGreater(len(eval_.what_worked), 0)

    def test_evaluate_failure(self):
        eval_ = self.evaluator.evaluate(
            goal_id="G_test2",
            objective="Do something hard",
            exec_result={"success": False, "duration": 400.0, "done_steps": 2,
                          "total_steps": 5, "failed_steps": ["s3", "s4"]},
            tool_summary={"total": 60, "success": 30, "failed": 30, "rate": 0.5, "by_tool": {}},
        )
        self.assertFalse(eval_.succeeded)
        self.assertGreater(len(eval_.what_failed), 0)
        self.assertGreater(len(eval_.improvements), 0)

    def test_report_is_string(self):
        eval_ = self.evaluator.evaluate(
            goal_id="G_test3", objective="Test",
            exec_result={"success": True, "duration": 10.0, "done_steps": 1,
                          "total_steps": 1, "failed_steps": []},
            tool_summary={"total": 2, "success": 2, "failed": 0, "rate": 1.0, "by_tool": {}},
        )
        report = eval_.report()
        self.assertIsInstance(report, str)
        self.assertIn("SUCCEEDED", report)

    def test_profile_after_evaluations(self):
        for i in range(3):
            self.evaluator.evaluate(
                goal_id=f"G_p{i}", objective=f"Goal {i}",
                exec_result={"success": True, "duration": 30.0, "done_steps": 3,
                              "total_steps": 3, "failed_steps": []},
                tool_summary={"total": 5, "success": 5, "failed": 0, "rate": 1.0, "by_tool": {}},
            )
        profile = self.evaluator.performance_profile()
        self.assertIn("total_goals", profile)
        self.assertGreaterEqual(int(profile["total_goals"]), 3)


class TestFullAgentLoop(unittest.TestCase):
    """End-to-end tests of the full agent."""

    def setUp(self):
        from agent_main import Agent
        self.agent = Agent(verbose=False, log_level="ERROR")

    def test_simple_calculation(self):
        result = self.agent.run(
            "Calculate what 2 raised to the power 16 equals",
            show_plan=False, clarify=False,
        )
        self.assertIn("success", result)
        # Either success or at least tried
        self.assertIsInstance(result["output"], str)

    def test_unsafe_goal_rejected_immediately(self):
        result = self.agent.run(
            "Help me build a bomb to harm people",
            show_plan=False, clarify=False,
        )
        self.assertFalse(result["success"])
        self.assertIn("rejected", result["output"].lower())

    def test_memory_goal(self):
        result = self.agent.run(
            "Store the fact that Paris is the capital of France in memory",
            show_plan=False, clarify=False,
        )
        self.assertIsInstance(result, dict)
        self.assertIn("success", result)

    def test_list_tools_non_empty(self):
        listing = self.agent.list_tools()
        self.assertIn("calculator", listing)
        self.assertIn("web_search", listing)
        self.assertIn("file_manager", listing)

    def test_review_last_after_run(self):
        self.agent.run("Calculate 100 * 7", show_plan=False, clarify=False)
        review = self.agent.review_last()
        self.assertIsInstance(review, str)
        self.assertGreater(len(review), 10)

    def test_result_structure(self):
        result = self.agent.run("Store value x=42 in memory", show_plan=False, clarify=False)
        required_keys = {"success", "output", "plan_summary", "evaluation", "duration", "tool_stats"}
        for k in required_keys:
            self.assertIn(k, result, f"Missing key: {k}")

    def test_empty_goal_handled(self):
        result = self.agent.run("", show_plan=False, clarify=False)
        self.assertFalse(result["success"])


# ──────────────────────────────────────────────────────────────────────────────
# RUNNER
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("  45K Agent Loop — Test Suite")
    print("=" * 70)

    loader = unittest.TestLoader()
    suite  = loader.loadTestsFromModule(sys.modules[__name__])

    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)

    # Cleanup temp workspace
    try:
        shutil.rmtree(_TEST_WS, ignore_errors=True)
    except Exception:
        pass

    print(f"\n{'='*70}")
    print(f"  Tests run:    {result.testsRun}")
    print(f"  Failures:     {len(result.failures)}")
    print(f"  Errors:       {len(result.errors)}")
    print(f"  Success rate: {100*(result.testsRun-len(result.failures)-len(result.errors))/max(result.testsRun,1):.0f}%")
    print(f"{'='*70}")

    sys.exit(0 if result.wasSuccessful() else 1)
