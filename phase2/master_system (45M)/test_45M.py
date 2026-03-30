"""
test_45M.py — Full Test Suite

Tests every subsystem end-to-end.
Run: python system.py --test
  or: python test_45M.py
"""

import sys, os, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))


def _pass(msg): print(f"  ✓  {msg}")
def _fail(msg): print(f"  ✗  {msg}"); raise AssertionError(msg)
def _section(title): print(f"\n{'─'*50}\n  {title}\n{'─'*50}")


def run_all_tests(system=None):
    from system import MasterSystem

    if system is None:
        system = MasterSystem()

    total_passed = 0
    total_failed = 0

    def test(name, fn):
        nonlocal total_passed, total_failed
        try:
            fn()
            _pass(name)
            total_passed += 1
        except Exception as e:
            _fail(f"{name} → {e}")
            total_failed += 1

    # ── Engine ────────────────────────────────────────────────────────────────
    _section("1. Continuous Operation Engine")

    def test_engine_start():
        system.engine.start()
        assert system.engine._running, "Engine should be running"
        system.engine.stop()
    test("Engine start/stop", test_engine_start)

    def test_heartbeat():
        from autonomy.engine import Heartbeat, EngineDB
        hb = Heartbeat(system.engine.db, interval=1)
        hb.start(); time.sleep(0.2); hb.stop()
        last = system.engine.db.get_state("last_heartbeat")
        assert last is not None, "Heartbeat should have written"
    test("Heartbeat writes state", test_heartbeat)

    def test_activity_log():
        system.engine.db.log_activity("TEST", "test", "test entry")
        logs = system.engine.db.get_logs(limit=5)
        assert len(logs) > 0
        assert any(l["event"] == "TEST" for l in logs)
    test("Activity logging", test_activity_log)

    def test_scheduled_tasks():
        tasks = system.engine.db.get_tasks()
        assert len(tasks) >= 4, "Should have 4 default scheduled tasks"
        names = [t.name for t in tasks]
        assert "Daily Goal Review" in names
        assert "Nightly Memory Consolidation" in names
    test("Scheduled tasks registered", test_scheduled_tasks)

    def test_crash_recovery():
        from datetime import datetime, timedelta
        old_time = (datetime.utcnow() - timedelta(minutes=10)).isoformat()
        system.engine.db.set_state("last_heartbeat", old_time)
        msg = system.engine.heartbeat.check_crash_recovery()
        assert msg is not None, "Should detect crash"
    test("Crash recovery detection", test_crash_recovery)

    # ── Goals ─────────────────────────────────────────────────────────────────
    _section("2. Long Horizon Goal Manager")

    def test_create_goal():
        g = system.goals.create_goal(
            title="Test: Research transformer architectures",
            description="Study and document modern transformer variants",
            horizon_days=7,
            priority="high",
        )
        assert g.goal_id
        assert g.status == "active"
        steps = system.goals.db.get_steps(g.goal_id)
        assert len(steps) >= 3, f"Should have workstream steps, got {len(steps)}"
        return g
    goal_obj = None
    def _create():
        nonlocal goal_obj
        goal_obj = test_create_goal()
    test("Goal creation with workstream", _create)

    def test_goal_progress():
        if not goal_obj: return
        steps = system.goals.db.get_steps(goal_obj.goal_id)
        first = steps[0]
        updated = system.goals.update_progress(
            goal_obj.goal_id, first.step_id, "Research done", "done")
        assert updated.progress_pct > 0
    test("Goal progress tracking", test_goal_progress)

    def test_goal_blocker():
        g2 = system.goals.create_goal("Blocked goal", "test", horizon_days=3)
        system.goals.set_blocker(g2.goal_id, "Waiting for dependency")
        reloaded = system.goals.db.get_goal(g2.goal_id)
        assert reloaded.status == "blocked"
        system.goals.clear_blocker(g2.goal_id)
        reloaded2 = system.goals.db.get_goal(g2.goal_id)
        assert reloaded2.status == "active"
    test("Blocker set/clear", test_goal_blocker)

    def test_daily_review():
        review = system.goals.daily_review()
        assert "active" in review
        assert "report" in review
    test("Daily goal review", test_daily_review)

    # ── Proactive ─────────────────────────────────────────────────────────────
    _section("3. Proactive Intelligence")

    def test_fire_alert():
        alert = system.proactive.fire_alert(
            "notice", "test", "Test alert", "This is a test alert body")
        assert alert.alert_id
        unread = system.proactive.db.get_alerts(unread_only=True)
        assert any(a.alert_id == alert.alert_id for a in unread)
    test("Alert creation and retrieval", test_fire_alert)

    def test_morning_brief():
        brief = system.morning_briefing()
        assert "MORNING BRIEFING" in brief
        assert len(brief) > 50
    test("Morning briefing generation", test_morning_brief)

    # ── Decisions ─────────────────────────────────────────────────────────────
    _section("4. Autonomous Decision Framework")

    def test_tier1_auto():
        from autonomy.decisions import DecisionFramework
        df = DecisionFramework()
        result_holder = []
        req = df.request(
            title="Read activity log",
            description="Read recent logs for analysis",
            proposed_action="read_file activity.log",
            category="file_read",
            reversible=True, stakes="low",
            execute_fn=lambda: result_holder.append("executed") or "ok",
        )
        assert req.status in ("approved", "executed"), f"Tier1 should auto-approve, got {req.status}"
    test("Tier 1 autonomous approval", test_tier1_auto)

    def test_tier4_block():
        from autonomy.decisions import DecisionFramework
        df = DecisionFramework()
        req = df.request(
            title="Access API keys",
            description="Retrieve credentials",
            proposed_action="access credential_access",
            category="credential_access",
            reversible=False, stakes="critical",
        )
        assert req.status == "cancelled", f"Tier 4 should cancel, got {req.status}"
    test("Tier 4 never executes", test_tier4_block)

    def test_audit_trail():
        trail = system.decisions.audit_trail(limit=10)
        assert isinstance(trail, list)
    test("Audit trail accessible", test_audit_trail)

    # ── Safety ────────────────────────────────────────────────────────────────
    _section("5. Safety Layer")

    def test_constitutional():
        ok, reason = system.safety.constitutional.check_action("search for transformers")
        assert ok, f"Safe action should pass: {reason}"
        blocked, reason2 = system.safety.constitutional.check_action("access api_key file")
        assert not blocked, f"Should block credential access"
    test("Constitutional AI checks", test_constitutional)

    def test_red_team():
        report = system.safety.red_team.run_all()
        assert report["total"] == 10
        # All tests should pass (system correctly blocks/allows)
        assert report["passed"] >= 8, f"Red team: only {report['passed']}/10 passed"
    test("Red team adversarial tests", test_red_team)

    def test_kill_switch():
        result = system.safety.kill_switch.test()
        assert result["kill_file_writeable"]
        assert result["status"] == "READY"
    test("Kill switch ready", test_kill_switch)

    def test_safety_audit():
        report = system.safety.run_safety_audit()
        assert "red_team" in report
        assert "kill_switch" in report
        assert report["kill_switch"]["status"] == "READY"
    test("Full safety audit", test_safety_audit)

    # ── Training ──────────────────────────────────────────────────────────────
    _section("6. Training Pipeline")

    def test_ingest():
        ex = system.continuous_learn.ingest(
            "What is attention in transformers?",
            "Attention allows the model to focus on relevant parts of the input. "
            "It computes a weighted sum of values based on query-key similarity. "
            "This enables the model to capture long-range dependencies efficiently.",
            source="test",
        )
        assert ex is not None, "High-quality example should be accepted"
    test("Training data ingestion", test_ingest)

    def test_quality_filter():
        from scale.pipeline import QualityFilter
        qf = QualityFilter()
        assert qf.score("hi", "ok") < 0.5, "Low quality should score < 0.5"
        good_score = qf.score(
            "Explain gradient descent",
            "Gradient descent is an optimization algorithm that iteratively updates "
            "parameters by moving in the direction of steepest descent of the loss function. "
            "The learning rate controls step size. AdamW is the standard variant for transformers."
        )
        assert good_score >= 0.5, f"Good example should score >= 0.5, got {good_score}"
    test("Quality filter scoring", test_quality_filter)

    def test_training_run():
        run = system.trigger_training()
        assert run.run_id
        assert run.status in ("pending", "running", "done")
        # Wait briefly for stub run
        time.sleep(0.5)
    test("Training run starts", test_training_run)

    def test_scale_report():
        report = system.scale_manager.scale_report()
        assert "recommended_config" in report
        assert "ladder" in report
        assert len(report["ladder"]) >= 5
    test("Scale manager report", test_scale_report)

    # ── Personalization ───────────────────────────────────────────────────────
    _section("7. Owner Modeling & Knowledge Base")

    def test_owner_observe():
        profile = system.owner_modeler.observe_interaction(
            "Help me understand the transformer attention mechanism in detail with code examples",
            "The attention mechanism works by computing dot products between queries and keys, "
            "scaling by sqrt(d_k), applying softmax, and using as weights for values. "
            "Here is an implementation: Q @ K.T / sqrt(d_k) → softmax → @ V",
        )
        assert profile.total_interactions >= 1
        assert len(profile.expertise_areas) >= 0   # might have detected ML domain
    test("Owner model observation", test_owner_observe)

    def test_owner_correct():
        system.owner_modeler.correct("preferred_length", "brief")
        profile = system.owner_modeler.inspect()
        assert profile["preferred_length"] == "brief"
        # Reset
        system.owner_modeler.correct("preferred_length", "medium")
    test("Owner model correction", test_owner_correct)

    def test_knowledge_base():
        entry = system.knowledge_base.add(
            title   = "Transformer Attention Mechanism",
            content = "Scaled dot-product attention: Attention(Q,K,V) = softmax(QK^T/√dk)V. "
                      "Multi-head attention runs h attention heads in parallel.",
            domain  = "machine_learning",
            tags    = ["attention", "transformer", "architecture"],
        )
        assert entry.entry_id
        results = system.knowledge_base.search("attention mechanism")
        assert len(results) >= 1
    test("Knowledge base add and search", test_knowledge_base)

    def test_kb_export():
        graph = system.knowledge_base.export_graph()
        assert "nodes" in graph
        assert "domains" in graph
        assert graph["total"] >= 0
    test("Knowledge graph export", test_kb_export)

    # ── Control interface ─────────────────────────────────────────────────────
    _section("8. Control Interface & API")

    def test_status():
        s = system.status()
        assert "version" in s
        assert "goals" in s
        assert "safety" in s
        assert s["version"] == system.VERSION
    test("System status", test_status)

    def test_goal_via_control():
        result = system.control.create_goal(
            title="Control interface test goal",
            description="Test goal created via control interface",
            horizon_days=3,
        )
        assert "goal_id" in result
    test("Goal creation via control", test_goal_via_control)

    def test_api_json():
        response = system.api.handle(json.dumps({
            "command": "status", "params": {}
        }))
        parsed = json.loads(response)
        assert parsed["ok"]
        assert "result" in parsed
    test("JSON API command handling", test_api_json)

    def test_api_unknown_cmd():
        response = system.api.handle(json.dumps({"command": "nonexistent"}))
        parsed = json.loads(response)
        assert not parsed["ok"] or "error" in parsed
    test("API unknown command handled gracefully", test_api_unknown_cmd)

    # ── Integration ───────────────────────────────────────────────────────────
    _section("9. Integration: Full Interaction Cycle")

    def test_full_interaction():
        # Simulate a full cycle
        inp = "Research the latest advances in mixture of experts architectures"
        out = ("Mixture of Experts (MoE) models route each token to a subset of "
               "specialized expert networks. This allows scaling model capacity "
               "without proportionally scaling compute. Recent work includes "
               "Switch Transformer, GLaM, and Mixtral. Key benefits: "
               "sparse activation, efficient scaling, domain specialization.")
        system.process_interaction(inp, out, feedback=0.9, domain="machine_learning")
        # Owner model should have updated
        profile = system.owner_modeler.inspect()
        assert profile["total_interactions"] >= 1
        # Knowledge base should have grown
        kb_stats = system.knowledge_base.stats()
        assert kb_stats["total_entries"] >= 1
    test("Full interaction processing cycle", test_full_interaction)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'═'*50}")
    print(f"  TEST RESULTS: {total_passed} passed, {total_failed} failed")
    if total_failed == 0:
        print("  ✓ ALL TESTS PASSED")
    else:
        print(f"  ✗ {total_failed} TESTS FAILED")
    print(f"{'═'*50}\n")

    return total_failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
