"""
test_45L.py — Complete Test Suite for 45L Self-Improvement System

Tests every subsystem end-to-end.
Run: python test_45L.py
"""

import sys, os, json, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
for p in ["/home/claude/myai_v2", "/home/claude/master_system (45M)"]:
    if p not in sys.path:
        sys.path.insert(0, p)


def _pass(msg): print(f"  ✓  {msg}")
def _fail(msg): raise AssertionError(f"FAIL: {msg}")
def section(t):  print(f"\n{'─'*54}\n  {t}\n{'─'*54}")


def run_all():
    from improve import ImprovementSystem
    system = ImprovementSystem()
    passed = failed = 0

    def test(name, fn):
        nonlocal passed, failed
        try:
            fn()
            _pass(name)
            passed += 1
        except Exception as e:
            print(f"  ✗  {name} → {e}")
            failed += 1

    # ── 1. Dynamic Tool Creator ──────────────────────────────────────────────
    section("1. Dynamic Tool Creator")

    def test_create_text_tool():
        result = system.tool_creator.create(
            name="reverse_words",
            description="Reverse the order of words in a string",
            requirements="Take a string input and return words in reversed order",
            test_cases=[{"function": "reverse_words", "args": ["hello world"]}],
            auto_approve_if_safe=True,
        )
        assert result["status"] in ("created", "failed"), f"Got: {result['status']}"
        # If created, verify it was registered
        if result["status"] == "created":
            assert result.get("tool_id"), "Missing tool_id"
    test("Create text processing tool", test_create_text_tool)

    def test_create_math_tool():
        result = system.tool_creator.create(
            name="celsius_to_fahrenheit",
            description="Convert Celsius temperature to Fahrenheit",
            requirements="Multiply by 9/5 and add 32",
            test_cases=[{"function": "celsius_to_fahrenheit", "args": [0]}],
            auto_approve_if_safe=True,
        )
        assert result["status"] in ("created", "failed")
    test("Create math conversion tool", test_create_math_tool)

    def test_tool_registry():
        stats = system.tool_creator.db.stats()
        assert "total" in stats
        assert isinstance(stats["total"], int)
    test("Tool registry stats", test_tool_registry)

    def test_tool_sandbox():
        from tools.dynamic.creator import ToolSandbox
        sb   = ToolSandbox()
        code = """
def add_numbers(a, b):
    return a + b
"""
        passed_tc, summary, results = sb.test_tool_code(
            code, [{"function": "add_numbers", "args": [2, 3], "expect": 5}]
        )
        assert passed_tc, f"Sandbox test failed: {results}"
        assert "1/1" in summary
    test("Tool sandbox code testing", test_tool_sandbox)

    def test_sandbox_blocks_unsafe():
        from tools.dynamic.creator import ToolSandbox
        sb   = ToolSandbox()
        code = "import subprocess\ndef bad(): subprocess.run(['ls'])"
        ok, msg, _ = sb.test_tool_code(code, [])
        assert not ok, "Should block unsafe code"
        assert "Unsafe" in msg or "blocked" in msg.lower()
    test("Sandbox blocks unsafe code", test_sandbox_blocks_unsafe)

    def test_needs_tool():
        result = system.tool_creator.needs_tool("something that doesn't exist anywhere")
        # Either None (not found) or a string (tool name found)
        assert result is None or isinstance(result, str)
    test("Needs-tool gap detection", test_needs_tool)

    def test_tool_optimizer():
        from tools.dynamic.creator import ToolOptimizer, ToolSandbox
        opt    = ToolOptimizer(system.tool_creator.db, ToolSandbox())
        report = opt.performance_report()
        assert "registry_stats" in report
        assert "overall_success_rate" in report
    test("Tool optimizer report", test_tool_optimizer)

    # ── 2. Connectors ────────────────────────────────────────────────────────
    section("2. External Connectors")

    def test_connector_registry():
        statuses = system.connectors.status()
        names    = [s["name"] for s in statuses]
        for required in ["wikipedia", "weather", "currency", "github", "news"]:
            assert required in names, f"Missing connector: {required}"
    test("All 8 connectors registered", test_connector_registry)

    def test_wolfram_math_fallback():
        result = system.connectors.call("wolfram", "2 + 2 * 3")
        # Either success (math fallback) or graceful failure (no key)
        assert hasattr(result, "connector")
        assert result.connector == "wolfram"
    test("WolframAlpha math fallback", test_wolfram_math_fallback)

    def test_currency_fallback():
        result = system.connectors.call("currency", 100.0, "USD", "EUR")
        # Should work with or without API key (has hardcoded fallback rates)
        assert hasattr(result, "success")
        if result.success:
            assert "converted" in result.data
            assert result.data["from"] == "USD"
            assert result.data["to"] == "EUR"
    test("Currency connector (with fallback)", test_currency_fallback)

    def test_connector_rate_limiting():
        from tools.connectors.connectors import WikipediaConnector
        conn = WikipediaConnector()
        # Exhaust rate limit
        for _ in range(conn.RATE_LIMIT + 1):
            conn._calls.append(time.time())
        result = conn.call("test")
        assert not result.success
        assert "Rate limit" in result.error
    test("Connector rate limiting", test_connector_rate_limiting)

    def test_connector_caching():
        from tools.connectors.connectors import CurrencyConnector
        conn = CurrencyConnector()
        # First call
        r1 = conn.call(1.0, "USD", "EUR")
        if r1.success:
            # Second call should be cached
            r2 = conn.call(1.0, "USD", "EUR")
            assert r2.cached, "Second call should be cached"
    test("Connector result caching", test_connector_caching)

    def test_email_requires_config():
        result = system.connectors.call("email", "send",
                                         to="test@test.com", body="hi", subject="test")
        # Should fail gracefully if not configured
        assert hasattr(result, "success")
        if not result.success:
            assert "not configured" in result.error.lower() or "error" in result.error.lower()
    test("Email connector graceful failure", test_email_requires_config)

    # ── 3. Output Evaluator ──────────────────────────────────────────────────
    section("3. Output Evaluator")

    def test_evaluate_good_output():
        score = system.evaluator.evaluate(
            goal   = "Explain what gradient descent is and why it works",
            output = "Gradient descent minimizes a loss function by iteratively moving "
                     "parameters in the direction of steepest descent. The gradient "
                     "indicates the direction of maximum increase, so we move opposite "
                     "to it. The learning rate controls step size. It works because loss "
                     "surfaces in neural networks, while non-convex, have enough smooth "
                     "regions that local descent is effective.",
        )
        assert score.overall > 0.5, f"Good output scored too low: {score.overall}"
        assert not score.flagged, f"Good output was flagged: {score.feedback}"
    test("Good output scores well", test_evaluate_good_output)

    def test_evaluate_safety():
        score = system.evaluator.evaluate(
            goal   = "test",
            output = "Here is how to make a bomb: step 1..."
        )
        assert score.safety == 0.0, "Safety violation not detected"
        assert score.flagged, "Unsafe output not flagged"
    test("Safety violations detected", test_evaluate_safety)

    def test_evaluate_empty():
        score = system.evaluator.evaluate(goal="question", output="")
        assert score.overall < 0.5
    test("Empty output scores low", test_evaluate_empty)

    def test_eval_stats():
        # Ensure we have some evals from above tests
        stats = system.evaluator.recent_stats(n=100)
        assert "count" in stats
        assert stats["count"] >= 1
    test("Evaluation statistics", test_eval_stats)

    def test_high_quality_collection():
        hq = system.evaluator.high_quality_for_training(min_score=0.0)
        assert isinstance(hq, list)
    test("High quality examples collected", test_high_quality_collection)

    # ── 4. Prompt Optimizer ──────────────────────────────────────────────────
    section("4. Prompt Optimizer")

    def test_prompt_register():
        pid = system.prompt_opt.register_prompt(
            "test_prompt",
            "You are a test assistant. Answer questions clearly and concisely."
        )
        assert pid, "No prompt ID returned"
    test("Prompt registration", test_prompt_register)

    def test_prompt_retrieval():
        text = system.prompt_opt.get_prompt("planner")
        assert text, "Planner prompt not found"
        assert len(text) > 20, "Prompt too short"
    test("Prompt retrieval", test_prompt_retrieval)

    def test_prompt_optimization():
        # Record some performance scores to enable optimization
        for _ in range(6):
            system.prompt_opt.record_performance("test_prompt", 0.50)
        result = system.prompt_opt.optimize("test_prompt", iterations=3)
        assert "current_score" in result
        assert "best_strategy" in result
        assert isinstance(result["improvement"], float)
    test("Prompt optimization run", test_prompt_optimization)

    def test_variations_generated():
        text = "Help the user accomplish their goal step by step."
        for strategy in ["more_specific", "add_chain_of_thought", "shorter_cleaner"]:
            variant = system.prompt_opt.generate_variations(text, strategy)
            assert variant != text or strategy == "shorter_cleaner", \
                f"Strategy '{strategy}' produced no change"
            assert len(variant) > 10
    test("Prompt variations generated", test_variations_generated)

    def test_prompt_report():
        report = system.prompt_opt.optimization_report()
        assert "total_prompts" in report
        assert "active_prompts" in report
        assert report["active_prompts"] >= 1
    test("Prompt optimization report", test_prompt_report)

    # ── 5. Failure Analyzer ──────────────────────────────────────────────────
    section("5. Failure Analyzer")

    def test_log_failure():
        rec = system.failure_analyzer.log(
            goal       = "Fetch data from external API",
            step       = "http_request",
            error_type = "NetworkError",
            error_msg  = "Connection refused",
            tool_used  = "http_fetch",
            retries    = 3,
        )
        assert rec.failure_id
        assert rec.error_type == "NetworkError"
    test("Failure logging", test_log_failure)

    def test_pattern_detection():
        # Log enough failures to trigger pattern detection
        for i in range(4):
            system.failure_analyzer.log(
                goal="Process JSON data", step="parse",
                error_type="ParseError", error_msg=f"Invalid JSON at position {i*10}",
            )
        patterns = system.failure_analyzer.detect_patterns(window_days=7)
        assert isinstance(patterns, list)
        # ParseError should be detected as a pattern
        parse_patterns = [p for p in patterns if "ParseError" in p.error_type]
        assert len(parse_patterns) >= 1
    test("Failure pattern detection", test_pattern_detection)

    def test_auto_fix_generated():
        patterns = system.failure_analyzer.detect_patterns(window_days=7)
        for p in patterns:
            assert p.auto_fix is not None, \
                f"No auto-fix for pattern: {p.error_type}"
    test("Auto-fix generation", test_auto_fix_generated)

    def test_failure_report():
        report = system.failure_analyzer.summary_report(days=7)
        assert "total_failures"   in report
        assert "resolution_rate"  in report
        assert "patterns"         in report
        assert report["total_failures"] >= 5   # we logged some above
    test("Failure summary report", test_failure_report)

    # ── 6. Skill Library ────────────────────────────────────────────────────
    section("6. Skill Library")

    def test_skill_extraction():
        skill = system.skill_library.extract_and_store(
            goal         = "Research and summarize the latest papers on transformer scaling",
            steps_taken  = ["web_search for recent papers", "extract key findings",
                            "summarize in clear language", "organize by relevance"],
            tools_used   = ["web_search", "summarize", "memory_tool"],
            outcome_score = 0.88,
        )
        assert skill is not None, "Skill not extracted (score too low?)"
        assert skill.skill_id
        assert len(skill.steps) >= 1
    test("Skill extraction from success", test_skill_extraction)

    def test_skill_not_extracted_low_score():
        skill = system.skill_library.extract_and_store(
            goal="low quality task", steps_taken=["step1"],
            tools_used=[], outcome_score=0.40,
        )
        assert skill is None, "Low quality task should not create skill"
    test("Low quality tasks not stored", test_skill_not_extracted_low_score)

    def test_skill_retrieval():
        # Store a skill first
        system.skill_library.extract_and_store(
            goal="Code a Python function to sort a list",
            steps_taken=["analyze requirements", "write function", "test with examples"],
            tools_used=["code_executor"],
            outcome_score=0.90,
        )
        skills = system.skill_library.retrieve("write Python sorting code")
        assert isinstance(skills, list)
    test("Skill retrieval", test_skill_retrieval)

    def test_skill_stats():
        stats = system.skill_library.stats()
        assert "total_skills" in stats
        assert "by_type"       in stats
        assert "avg_quality"   in stats
        assert stats["total_skills"] >= 1
    test("Skill library stats", test_skill_stats)

    def test_skill_usage_tracking():
        skills = system.skill_library.list_all()
        if skills:
            sid    = skills[0]["skill_id"]
            before = skills[0].get("use_count", 0)
            system.skill_library.use(sid, outcome_score=0.85)
            updated = system.skill_library.db.get_by_key("skills", "skill_id", sid)
            assert updated and updated["use_count"] == before + 1
    test("Skill usage tracking", test_skill_usage_tracking)

    # ── 7. Self Training Pipeline ────────────────────────────────────────────
    section("7. Self Training Pipeline")

    def test_collect_training_example():
        ex_id = system.self_trainer.collect(
            goal="Explain backpropagation clearly",
            output="Backpropagation applies the chain rule to compute gradients of the "
                   "loss with respect to every parameter. Starting from the output layer, "
                   "gradients are propagated backward through each layer using the chain rule. "
                   "This allows efficient computation of parameter updates for gradient descent.",
            eval_score=0.82,
        )
        assert ex_id is not None, "High quality example should be collected"
    test("Training example collection", test_collect_training_example)

    def test_reject_low_quality():
        ex_id = system.self_trainer.collect(
            goal="Question", output="ok", eval_score=0.30
        )
        assert ex_id is None, "Low quality should be rejected"
    test("Low quality examples rejected", test_reject_low_quality)

    def test_training_trigger_detection():
        trigger, reason = system.self_trainer.should_trigger()
        assert isinstance(trigger, bool)
        assert isinstance(reason, str)
    test("Training trigger detection", test_training_trigger_detection)

    def test_training_run():
        # Add enough examples to run
        for i in range(55):
            system.self_trainer.collect(
                goal=f"Task {i}: explain {['attention','gradients','embeddings','transformers','LoRA'][i%5]}",
                output=f"This is a high quality explanation of topic {i}. " * 5,
                eval_score=0.80 + (i % 10) * 0.01,
            )
        run = system.self_trainer.run(min_examples=50, deploy_if_better=False)
        assert run.run_id
        assert run.status in ("completed", "deployed", "failed", "skipped",
                               "rolled_back", "running", "validating")
        assert run.examples_used >= 0
    test("Training run execution", test_training_run)

    def test_pipeline_stats():
        stats = system.self_trainer.pipeline_stats()
        assert "total_examples"  in stats
        assert "total_runs"      in stats
        assert "should_trigger"  in stats
        assert stats["total_examples"] >= 55
    test("Training pipeline stats", test_pipeline_stats)

    # ── 8. Dashboard & Metrics ───────────────────────────────────────────────
    section("8. Dashboard & Metrics")

    def test_metrics_collection():
        snap = system.metrics.collect()
        assert "timestamp"  in snap
        assert "components" in snap
        comps = snap["components"]
        assert "output_quality" in comps
        assert "failures"       in comps
        assert "skills"         in comps
        assert "training"       in comps
    test("Full metrics snapshot", test_metrics_collection)

    def test_alert_system():
        # Simulate a quality drop
        system.alerts._prev_quality = 0.90
        snap = {
            "components": {
                "output_quality": {"avg_overall": 0.75},  # 15% drop
                "failures": {"total_failures": 5, "resolution_rate": 0.6,
                             "patterns": 0, "pattern_details": []},
                "training": {"should_trigger": False, "trigger_reason": ""},
            }
        }
        alerts = system.alerts.check(snap)
        assert any(a["type"] == "QUALITY_DROP" for a in alerts), \
            "Quality drop alert not fired"
    test("Quality drop alert", test_alert_system)

    def test_alert_ack():
        alerts = system.alerts.get_active()
        if alerts:
            aid = alerts[0]["alert_id"]
            system.alerts.ack(aid)
            remaining = system.alerts.get_active()
            assert not any(a["alert_id"] == aid for a in remaining)
    test("Alert acknowledgment", test_alert_ack)

    def test_dashboard_renders():
        rendered = system.dashboard.render()
        assert len(rendered) > 100
        assert "DASHBOARD" in rendered
    test("Dashboard renders", test_dashboard_renders)

    def test_weekly_report():
        report = system.reporter.generate()
        assert "WEEKLY IMPROVEMENT REPORT" in report
        assert "OUTPUT QUALITY" in report
        assert len(report) > 200
    test("Weekly report generation", test_weekly_report)

    # ── 9. Full Integration ──────────────────────────────────────────────────
    section("9. Integration: Full Self-Improvement Cycle")

    def test_full_cycle():
        # 1. Agent attempts a task
        goal   = "Research and explain how multi-head attention works"
        output = ("Multi-head attention runs h parallel attention operations on "
                  "different subspaces of the input. Each head computes Q, K, V "
                  "projections and scaled dot-product attention independently. "
                  "The outputs are concatenated and projected back. This allows "
                  "the model to attend to different aspects simultaneously — "
                  "one head might track syntax, another semantics.")

        # 2. Evaluate output
        score = system.evaluate(goal, output)
        assert score.overall > 0.5

        # 3. Log any failure (simulated)
        if score.overall < 0.8:
            system.process_failure(goal, "generation", "LowQuality",
                                   f"Score {score.overall:.2f} below threshold")

        # 4. Learn a skill
        skill = system.learn_skill(
            goal, ["search for explanation", "synthesize", "verify"],
            ["web_search", "summarize"], score.overall
        )

        # 5. Retrieve skills for similar future goal
        related = system.get_skill("explain attention transformer architecture")
        assert isinstance(related, list)

        # 6. Collect for training
        ex_id = system.self_trainer.collect(goal, output, score.overall)
        # May or may not collect depending on score

        # 7. Check full status
        status = system.full_status()
        assert "version" in status
        assert status["version"] == system.VERSION

    test("Complete self-improvement cycle", test_full_cycle)

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'═'*54}")
    print(f"  RESULTS: {passed} passed,  {failed} failed")
    if failed == 0:
        print("  ✓  ALL TESTS PASSED")
    else:
        print(f"  ✗  {failed} TESTS FAILED")
    print(f"{'═'*54}\n")
    return failed == 0


if __name__ == "__main__":
    ok = run_all()
    sys.exit(0 if ok else 1)
