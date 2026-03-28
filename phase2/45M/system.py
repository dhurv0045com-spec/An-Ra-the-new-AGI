"""
system.py — Master Entry Point for 45M Phase 3 Autonomous AI System

Ties together all subsystems:
  - Continuous operation engine
  - Long-horizon goal manager
  - Proactive intelligence
  - Autonomous decision framework
  - Distributed training + continuous learning
  - Owner modeling + adaptive behavior + knowledge base
  - Safety layer (constitutional AI, red team, anomaly, audit, kill switch)
  - Owner control interface + dashboard + API

Usage:
    python system.py --start [--mode autonomous] [--tier 2]
    python system.py --briefing
    python system.py --goal "title" --horizon 7 --priority high
    python system.py --status
    python system.py --stop [--immediate]
    python system.py --owner-model --inspect
    python system.py --safety-audit
    python system.py --dashboard
    python system.py --api         (stdio JSON API)
    python system.py --test        (run full test suite)
"""

import sys, os, argparse, json, time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from autonomy.engine      import ContinuousEngine
from autonomy.goals       import GoalManager, Priority
from autonomy.proactive   import ProactiveEngine
from autonomy.decisions   import DecisionFramework
from scale.pipeline       import ContinuousLearning, DistributedTrainer, ScaleManager
from personalization.models import OwnerModeler, AdaptiveBehavior, PersonalKnowledgeBase
from safety.safety        import SafetyLayer, AuditLogger
from control.control      import ControlInterface, Dashboard, ControlAPI


class MasterSystem:
    """
    The 45M autonomous AI system.
    Instantiate once. Everything lives here.
    """

    VERSION = "45M-Phase3-v1.0"

    def __init__(self):
        self._started = False

        # Core subsystems
        self.engine          = ContinuousEngine()
        self.goals           = GoalManager()
        self.proactive       = ProactiveEngine()
        self.decisions       = DecisionFramework(notify_cb=self._notify_cb)
        self.continuous_learn = ContinuousLearning()
        self.trainer         = DistributedTrainer()
        self.scale_manager   = ScaleManager()
        self.owner_modeler   = OwnerModeler()
        self.adaptive        = AdaptiveBehavior()
        self.knowledge_base  = PersonalKnowledgeBase()
        self.safety          = SafetyLayer(engine=self.engine)
        self.audit           = self.safety.audit

        # Control layer
        self.control         = ControlInterface(system=self)
        self.dashboard       = Dashboard(system=self)
        self.api             = ControlAPI(self.control)

        # Wire kill switch to engine
        self.safety.kill_switch._engine = self.engine
        self.safety.kill_switch.register_callback(self._on_kill)

        # Wire scheduled tasks to real implementations
        self._wire_scheduled_tasks()

    def _wire_scheduled_tasks(self):
        self.engine.register_task_handler(
            "daily_goal_review",     self.goals.daily_review)
        self.engine.register_task_handler(
            "memory_consolidation",  self._memory_consolidation)
        self.engine.register_task_handler(
            "self_training_run",     self._self_training_run)
        self.engine.register_task_handler(
            "tool_performance_check", self._tool_performance_check)

    # ── Scheduled task implementations ────────────────────────────────────────

    def _memory_consolidation(self):
        """Nightly: consolidate and cross-reference knowledge base entries."""
        stats = self.knowledge_base.stats()
        self.audit.log("MEMORY_CONSOLIDATION", "training", "system",
                       f"KB stats: {stats}")
        return f"Consolidated {stats.get('total_entries', 0)} entries"

    def _self_training_run(self):
        """Weekly: run fine-tuning on accumulated high-quality examples."""
        batch = self.continuous_learn.prepare_daily_batch(max_examples=200)
        if not batch:
            return "No new training examples — skipped"
        rec   = self.scale_manager.current_recommendation()
        run   = self.trainer.start_run(
            config={
                "d_model": rec.d_model, "n_heads": rec.n_heads,
                "n_layers": rec.n_layers, "d_ff": rec.d_ff,
                "max_steps": 200, "batch_size": rec.batch_size,
                "lr": 1e-4, "seq_len": 32,
            },
            examples=batch
        )
        self.audit.log("SELF_TRAINING_STARTED", "training", "system",
                       f"Run {run.run_id[:8]} with {len(batch)} examples")
        return f"Training run {run.run_id[:8]} started"

    def _tool_performance_check(self):
        """Hourly: check system health and log metrics."""
        stats = self.continuous_learn.run_stats()
        anomalies = self.safety.anomaly.check({
            "memory_entries_delta": stats.get("total_examples", 0),
        })
        if anomalies:
            for a in anomalies:
                self.proactive.fire_alert(
                    "warning", "anomaly", a["type"], a["description"])
        return f"Health check done. {len(anomalies)} anomalies."

    # ── Notification callback ──────────────────────────────────────────────────

    def _notify_cb(self, notification: dict):
        """Called by decision framework when owner needs to be notified."""
        msg_type = notification.get("type", "")
        message  = notification.get("message", "")
        level    = "warning" if "tier3" in msg_type else "notice"
        self.proactive.fire_alert(level, "decision", msg_type, message)
        self.audit.log("OWNER_NOTIFICATION", "decisions", "system", message)

    # ── Kill callback ──────────────────────────────────────────────────────────

    def _on_kill(self):
        self.audit.log("KILL_SWITCH_FIRED", "safety", "system",
                       "System killed", level="CRITICAL")

    # ── Interaction processing ─────────────────────────────────────────────────

    def process_interaction(self, user_input: str, output: str,
                            feedback: float = None, domain: str = "general"):
        """
        Process any owner interaction.
        Updates owner model, ingests training data, updates knowledge base.
        """
        self.owner_modeler.observe_interaction(user_input, output, feedback=feedback)
        self.continuous_learn.ingest(user_input, output,
                                     source="interaction", feedback=feedback,
                                     domain=domain)
        if len(output) > 200:
            self.knowledge_base.add_from_task(
                task_title  = user_input[:80],
                task_output = output,
                domain      = domain,
            )
        self.audit.log("INTERACTION", "inference", "system",
                       f"Input len={len(user_input)} Output len={len(output)}")

    # ── Start / Stop ───────────────────────────────────────────────────────────

    def start(self, mode: str = "autonomous", default_tier: int = 2):
        if self._started:
            return

        # Check kill file before starting
        if self.safety.kill_switch.check_kill_file():
            kill_info = self.safety.kill_switch.clear_kill_file()
            print(f"⚠ Kill file found from previous session: {kill_info}")
            print("  Cleared. Starting fresh.")

        print("  Initializing Phase 1 LLM Bridge...")
        try:
            import llm_bridge
            self.llm = llm_bridge.get_llm_bridge()
            print("  [LLM Bridge] Phase 1 Model loaded and resident in memory.")
        except Exception as e:
            print(f"  [LLM Bridge] Failed to load model: {e}")

        self.safety.start()
        self.proactive.start()
        self.engine.start()
        self._started = True

        self.audit.log("SYSTEM_START", "system", "master",
                       f"45M started. Mode={mode} DefaultTier={default_tier}")
        print(f"  45M Phase 3 system started  [mode={mode}  tier={default_tier}]")
        print(f"  Version: {self.VERSION}")
        print(f"  Dashboard: python system.py --dashboard")
        print(f"  Stop:      python system.py --stop")

    def stop(self, immediate: bool = False):
        if immediate:
            self.safety.kill_switch.activate("Owner requested immediate stop")
        else:
            self.safety.stop()
            self.proactive.stop()
            self.engine.stop()
            self._started = False

    def run_forever(self):
        """Block until shutdown."""
        self.start()
        try:
            self.engine.run_forever()
        except KeyboardInterrupt:
            self.stop()

    # ── High-level queries ─────────────────────────────────────────────────────

    def status(self) -> dict:
        engine_status = self.engine.status()
        return {
            **engine_status,
            "version":      self.VERSION,
            "started":      self._started,
            "goals":        self.goals.daily_review(),
            "pending_approvals": [
                {"request_id": p.request_id, "title": p.title, "tier": p.tier}
                for p in self.decisions.pending_approvals()
            ],
            "safety":       self.safety.audit.safety_report(),
            "training":     self.continuous_learn.run_stats(),
            "knowledge":    self.knowledge_base.stats(),
            "scale":        self.scale_manager.scale_report(),
        }

    def morning_briefing(self) -> str:
        return self.proactive.morning_briefing(
            goals_manager=self.goals,
            audit_log=self.audit,
        )

    def set_goal(self, title: str, description: str,
                 horizon_days: int = 7, priority: str = "medium") -> dict:
        goal = self.goals.create_goal(
            title=title, description=description,
            horizon_days=horizon_days, priority=priority,
        )
        self.audit.log("GOAL_CREATED", "goals", "system",
                       f"Goal: {title} ({horizon_days}d, {priority})")
        return {"goal_id": goal.goal_id, "title": goal.title,
                "target": goal.target_date,
                "workstream_steps": len(self.goals.db.get_steps(goal.goal_id))}

    def trigger_training(self):
        batch = self.continuous_learn.prepare_daily_batch()
        rec   = self.scale_manager.current_recommendation()
        run   = self.trainer.start_run(
            config={"d_model": rec.d_model, "n_heads": rec.n_heads,
                    "n_layers": rec.n_layers, "d_ff": rec.d_ff,
                    "max_steps": 100, "batch_size": rec.batch_size,
                    "lr": 1e-4, "seq_len": 32},
            examples=batch
        )
        return run


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def build_parser():
    p = argparse.ArgumentParser(
        description="45M — Autonomous Personal AI System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python system.py --start --mode autonomous --tier 2
  python system.py --briefing
  python system.py --goal "Research transformer scaling" --horizon 14 --priority high
  python system.py --status
  python system.py --stop --immediate
  python system.py --owner-model --inspect
  python system.py --safety-audit
  python system.py --dashboard
  python system.py --test
        """
    )
    p.add_argument("--start",         action="store_true")
    p.add_argument("--stop",          action="store_true")
    p.add_argument("--immediate",     action="store_true")
    p.add_argument("--briefing",      action="store_true")
    p.add_argument("--status",        action="store_true")
    p.add_argument("--dashboard",     action="store_true")
    p.add_argument("--goal",          type=str,  default=None)
    p.add_argument("--horizon",       type=int,  default=7)
    p.add_argument("--priority",      type=str,  default="medium")
    p.add_argument("--mode",          type=str,  default="autonomous")
    p.add_argument("--tier",          type=int,  default=2)
    p.add_argument("--owner-model",   action="store_true")
    p.add_argument("--inspect",       action="store_true")
    p.add_argument("--safety-audit",  action="store_true")
    p.add_argument("--api",           action="store_true")
    p.add_argument("--test",          action="store_true")
    return p


def main():
    parser  = build_parser()
    args    = parser.parse_args()
    system  = MasterSystem()

    if args.test:
        from test_45M import run_all_tests
        run_all_tests(system)
        return

    if args.start:
        system.run_forever()
        return

    if args.stop:
        system.engine.db.set_state("engine_running", False)
        if args.immediate:
            system.safety.kill_switch.activate("CLI --stop --immediate")
        else:
            print("Stop signal sent.")
        return

    if args.briefing:
        print(system.morning_briefing())
        return

    if args.status:
        s = system.status()
        print(json.dumps(s, indent=2, default=str))
        return

    if args.dashboard:
        db = Dashboard(system)
        db.watch(interval=5)
        return

    if args.goal:
        result = system.set_goal(
            title        = args.goal,
            description  = args.goal,
            horizon_days = args.horizon,
            priority     = args.priority,
        )
        print(f"Goal created: {result['goal_id']}")
        print(f"Title:        {result['title']}")
        print(f"Target:       {result['target']}")
        print(f"Steps:        {result['workstream_steps']}")
        return

    if args.owner_model and args.inspect:
        profile = system.owner_modeler.inspect()
        print(json.dumps(profile, indent=2, default=str))
        return

    if args.safety_audit:
        result = system.safety.run_safety_audit()
        print(json.dumps(result, indent=2, default=str))
        return

    if args.api:
        api = ControlAPI(system.control)
        api.serve_stdio()
        return

    # Default: show status
    print(system.dashboard.render())


if __name__ == "__main__":
    main()
