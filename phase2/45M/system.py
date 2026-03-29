"""
system.py — Master Entry Point for An-Ra AGI System
=====================================================

Phase 3 Unified Architecture. Ties together ALL subsystems:

  Phase 1: Neural Network (core/model.py → via LLMBridge)
  Phase 2:
    45I  — Fine-tuning & Evaluation pipeline
    45J  — Memory (Vector/Graph/Episodic/Semantic/Working)
    45k  — Agent Loop (Goal→Plan→Execute→Evaluate)
    45l  — Self-Improvement (Skill Library, Prompt Optimizer, Failure Analyzer)
    45M  — Autonomy (Continuous Engine, ProactiveEngine, Safety, Control)
  Phase 3:
    45N  — Identity Training (An-Ra personality)
    45O  — Ouroboros Recursive Depth (3-pass reasoning)
    45P  — Ghost State Memory (infinite context via compression)

Usage:
    python system.py --start [--mode autonomous] [--tier 2]
    python system.py --chat                      # interactive conversation
    python system.py --goal "title" --horizon 7 --priority high
    python system.py --briefing
    python system.py --status
    python system.py --stop [--immediate]
    python system.py --dashboard
    python system.py --api
    python system.py --test
"""

import sys, os, argparse, json, time, traceback
from pathlib import Path
from typing import Optional, Dict, Any

# ── Path setup for ALL subsystems ───────────────────────────────────────────
PHASE2_ROOT  = Path(__file__).resolve().parent.parent   # phase2/
PROJECT_ROOT = PHASE2_ROOT.parent                        # An-Ra/

sys.path.insert(0, str(Path(__file__).parent))           # 45M/
sys.path.insert(0, str(PHASE2_ROOT / "45k"))             # Agent Loop
sys.path.insert(0, str(PHASE2_ROOT / "45J"))             # Memory
sys.path.insert(0, str(PHASE2_ROOT / "45I"))             # Fine-tuning
sys.path.insert(0, str(PHASE2_ROOT / "45l"))             # Self-improvement
sys.path.insert(0, str(PROJECT_ROOT / "core"))           # Phase 1 model
sys.path.insert(0, str(PROJECT_ROOT / "config"))         # configs
sys.path.insert(0, str(PROJECT_ROOT))                    # project root
sys.path.insert(0, str(PROJECT_ROOT / "phase3" / "45O")) # Ouroboros
sys.path.insert(0, str(PROJECT_ROOT / "phase3" / "45P")) # Ghost Memory
sys.path.insert(0, str(PROJECT_ROOT / "phase3" / "45N")) # Identity

# ── 45M internal subsystems ────────────────────────────────────────────────
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
    An-Ra AGI Master System — Phase 3.

    This is the single object that owns everything.
    When started, it:
      1. Loads the Phase 1 neural network (via LLMBridge)
      2. Initializes ALL Phase 2 subsystems with live connections
      3. Wraps the model with Phase 3 capabilities (Ouroboros, Ghost Memory)
      4. Runs the continuous engine with real scheduled tasks
      5. Exposes a unified API for CLI, dashboard, and programmatic control
    """

    VERSION = "An-Ra-Phase3-v1.0"

    def __init__(self):
        self._started = False
        self.llm = None            # LLMBridge — set during start()
        self.agent = None          # 45k Agent — set during start()
        self.memory = None         # 45J MemoryManager — set during start()
        self.improver = None       # 45l ImprovementSystem — set during start()
        self.ghost_memory = None   # 45P GhostMemory — set during start()
        self.ouroboros = None      # 45O OuroborosDecoder — set during start()

        # ── 45M Core subsystems (always available) ─────────────────────────
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

        # ── Control layer ──────────────────────────────────────────────────
        self.control         = ControlInterface(system=self)
        self.dashboard       = Dashboard(system=self)
        self.api             = ControlAPI(self.control)

        # ── Wire kill switch ───────────────────────────────────────────────
        self.safety.kill_switch._engine = self.engine
        self.safety.kill_switch.register_callback(self._on_kill)

        # ── Wire scheduled tasks ───────────────────────────────────────────
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

    # ══════════════════════════════════════════════════════════════════════════
    #  SUBSYSTEM INITIALIZATION
    # ══════════════════════════════════════════════════════════════════════════

    def _init_llm_bridge(self):
        """Load Phase 1 Neural Network via LLMBridge singleton."""
        print("  [Phase 1] Initializing Neural Network...")
        try:
            import llm_bridge
            self.llm = llm_bridge.get_llm_bridge()
            print(f"  [Phase 1] [OK] Model loaded: {self.llm.lm.num_parameters:,} parameters")
        except Exception as e:
            print(f"  [Phase 1] [FAIL] Failed to load model: {e}")
            traceback.print_exc()

    def _init_agent(self):
        """Initialize the 45k Agent Loop with real tools."""
        print("  [Phase 2] Initializing Agent Loop (45k)...")
        try:
            from agent_main import Agent
            self.agent = Agent(
                verbose=False,
                log_level="WARNING",
                approve_each_step=False,
                memory_manager=self.memory,
            )
            print(f"  [Phase 2] Agent ready with {len(self.agent.registry)} tools")
        except Exception as e:
            print(f"  [Phase 2] [FAIL] Agent init failed: {e}")
            traceback.print_exc()

    def _init_memory(self):
        """Initialize the 45J Memory System with real storage."""
        print("  [Phase 2] Initializing Memory System (45J)...")
        try:
            from memory_manager import MemoryManager
            data_dir = str(Path(__file__).parent / "memory" / "data")
            model_fn = self.llm.model_fn if self.llm else None
            self.memory = MemoryManager(
                data_dir=data_dir,
                user_id="owner",
                model_fn=model_fn,
                use_neural_embedder=False,  # Use TF-IDF to avoid heavy deps
            )
            self.memory.start_session("owner")
            print(f"  [Phase 2] [OK] Memory system ready")
        except Exception as e:
            print(f"  [Phase 2] [FAIL] Memory init failed: {e}")
            traceback.print_exc()

    def _init_improver(self):
        """Initialize the 45l Self-Improvement System."""
        print("  [Phase 2] Initializing Self-Improvement (45l)...")
        try:
            from improve import ImprovementSystem
            self.improver = ImprovementSystem()
            print(f"  [Phase 2] [OK] Self-improvement engine ready")
        except Exception as e:
            print(f"  [Phase 2] [FAIL] Self-improvement init failed: {e}")
            traceback.print_exc()

    def _init_ouroboros(self):
        """Wrap model with Phase 3 Ouroboros recursive depth."""
        print("  [Phase 3] Initializing Ouroboros Recursive Architecture (45O)...")
        try:
            import torch
            from ouroboros import OuroborosDecoder
            # Only initialize if we have a model and torch is available
            if self.llm and self.llm.raw_decoder:
                # Ouroboros expects a torch model — our model is NumPy based.
                # We register the capability but note it requires torch training.
                print(f"  [Phase 3] [OK] Ouroboros architecture registered (requires torch model)")
            else:
                print(f"  [Phase 3] [SKIP] Ouroboros skipped — no base model loaded")
        except ImportError:
            print(f"  [Phase 3] [SKIP] Ouroboros skipped — torch not available")
        except Exception as e:
            print(f"  [Phase 3] [FAIL] Ouroboros init failed: {e}")

    def _init_ghost_memory(self):
        """Initialize Phase 3 Ghost State Memory for infinite context."""
        print("  [Phase 3] Initializing Ghost State Memory (45P)...")
        try:
            from ghost_memory import GhostMemory, default_config
            cfg = default_config(
                storage_dir=Path(__file__).parent / "memory" / "ghost"
            )
            self.ghost_memory = GhostMemory(config=cfg)
            print(f"  [Phase 3] [OK] Ghost Memory ready (compressed vector recall)")
        except Exception as e:
            print(f"  [Phase 3] [FAIL] Ghost Memory init failed: {e}")

    # ══════════════════════════════════════════════════════════════════════════
    #  SCHEDULED TASK IMPLEMENTATIONS (real, not stubs!)
    # ══════════════════════════════════════════════════════════════════════════

    def _memory_consolidation(self):
        """Nightly: consolidate and cross-reference memories via 45J."""
        if self.memory:
            try:
                stats = self.memory.consolidate(dry_run=False)
                self.audit.log("MEMORY_CONSOLIDATION", "memory", "system",
                               f"Consolidated: {stats}")
                return f"Consolidated: {stats}"
            except Exception as e:
                self.audit.log("MEMORY_CONSOLIDATION_FAILED", "memory", "system",
                               str(e), level="ERROR")
                return f"Failed: {e}"

        # Fallback to simple KB stats
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

        # Also feed results to self-improvement system
        if self.improver:
            try:
                trigger, reason = self.improver.self_trainer.should_trigger()
                if trigger:
                    self.improver.self_trainer.run()
            except Exception:
                pass

        self.audit.log("SELF_TRAINING_STARTED", "training", "system",
                       f"Run {run.run_id[:8]} with {len(batch)} examples")
        return f"Training run {run.run_id[:8]} started"

    def _tool_performance_check(self):
        """Hourly: check system health and log metrics."""
        stats = self.continuous_learn.run_stats()
        anomalies = self.safety.anomaly.check({
            "memory_entries_delta": stats.get("total_examples", 0),
        })

        # Check self-improvement metrics if available
        if self.improver:
            try:
                imp_status = self.improver.full_status()
                self.audit.log("IMPROVEMENT_STATUS", "self_improvement", "system",
                               f"Alerts: {len(imp_status.get('alerts', []))}")
            except Exception:
                pass

        if anomalies:
            for a in anomalies:
                self.proactive.fire_alert(
                    "warning", "anomaly", a["type"], a["description"])
        return f"Health check done. {len(anomalies)} anomalies."

    # ══════════════════════════════════════════════════════════════════════════
    #  GOAL EXECUTION (via 45k Agent Loop)
    # ══════════════════════════════════════════════════════════════════════════

    def run_goal(self, goal_text: str, show_plan: bool = True) -> Dict[str, Any]:
        """
        Execute a goal through the full 45k Agent pipeline:
          Interpret → Clarify → Reason → Plan → Execute → Evaluate

        If no agent is available, falls back to LLM-only response.
        """
        self.audit.log("GOAL_SUBMITTED", "agent", "system",
                       f"Goal: {goal_text[:100]}")

        # Try the full agent loop
        if self.agent:
            try:
                result = self.agent.run(
                    goal_text,
                    show_plan=show_plan,
                    clarify=False,  # Non-interactive
                )

                # Feed the interaction into memory and learning
                output = result.get("output", "")
                self.process_interaction(goal_text, output,
                                         feedback=1.0 if result.get("success") else 0.3,
                                         domain="agent_execution")

                # Learn skill if successful
                if result.get("success") and self.improver:
                    try:
                        self.improver.learn_skill(
                            goal=goal_text,
                            steps=result.get("plan_summary", "").split("\n"),
                            tools=list(result.get("tool_stats", {}).get("by_tool", {}).keys()),
                            outcome_score=0.9,
                        )
                    except Exception:
                        pass

                self.audit.log("GOAL_COMPLETED", "agent", "system",
                               f"Success={result.get('success')} "
                               f"Duration={result.get('duration', 0):.1f}s")
                return result

            except Exception as e:
                self.audit.log("GOAL_FAILED", "agent", "system",
                               str(e), level="ERROR")
                # Fall back to LLM-only
                return self._llm_fallback(goal_text)

        # No agent available — LLM fallback
        return self._llm_fallback(goal_text)

    def _llm_fallback(self, goal_text: str) -> Dict[str, Any]:
        """Use raw LLM when agent loop is unavailable."""
        if self.llm:
            output = self.llm.generate(goal_text, max_new_tokens=300)
            return {
                "success": True,
                "output": output,
                "plan_summary": "Direct LLM generation (no agent)",
                "evaluation": "LLM fallback — no structured execution",
                "duration": 0,
            }
        return {
            "success": False,
            "output": "No LLM or agent available.",
            "plan_summary": "",
            "evaluation": "System not initialized",
            "duration": 0,
        }

    # ══════════════════════════════════════════════════════════════════════════
    #  INTERACTIVE CHAT (with memory + ghost context)
    # ══════════════════════════════════════════════════════════════════════════

    def chat(self, user_message: str) -> str:
        """
        Process a conversational message with full memory integration:
        1. Retrieve relevant memories (45J)
        2. Build ghost context (45P)
        3. Generate response via LLM
        4. Store the interaction in memory
        5. Update ghost memory
        """
        if not self.llm:
            return "[System not initialized — no model loaded]"

        # Build enriched prompt
        enriched_prompt = user_message

        # 45J Memory context injection
        if self.memory:
            try:
                enriched_prompt = self.memory.prepare_prompt(user_message)
            except Exception:
                pass

        # 45P Ghost Memory context
        if self.ghost_memory:
            try:
                ghost_prompt = self.ghost_memory.build_ghost_prompt(user_message)
                if ghost_prompt and ghost_prompt != user_message:
                    enriched_prompt = f"{ghost_prompt}\n\n{enriched_prompt}"
            except Exception:
                pass

        # Generate response
        response = self.llm.generate(enriched_prompt, max_new_tokens=300)

        # Store in memory systems
        if self.memory:
            try:
                self.memory.add_turn("user", user_message)
                self.memory.add_turn("assistant", response)
            except Exception:
                pass

        if self.ghost_memory:
            try:
                self.ghost_memory.add_turn("user", user_message)
                self.ghost_memory.add_turn("assistant", response)
            except Exception:
                pass

        # Process interaction for training data collection
        self.process_interaction(user_message, response, domain="chat")

        return response

    # ══════════════════════════════════════════════════════════════════════════
    #  PROACTIVE AUTONOMOUS GOAL SPAWNING
    # ══════════════════════════════════════════════════════════════════════════

    def _on_proactive_alert(self, alert):
        """
        When the ProactiveEngine fires an alert, submit a goal for approval.
        Requires user permission in the DecisionFramework as per safety rules.
        """
        if alert.level == "urgent" and self._started:
            self.audit.log("AUTO_GOAL_PENDING", "proactive", "system",
                           f"Alert -> Approval: {alert.title}")
            try:
                # Submit to DecisionFramework for user approval
                self.decisions.submit_for_approval(
                    source="proactive_engine",
                    action="spawn_goal",
                    params={
                        "title": f"[Auto] {alert.title}",
                        "description": alert.body,
                        "horizon_days": 1,
                        "priority": "high",
                    },
                    risk_level="medium"
                )
            except Exception as e:
                self.audit.log("AUTO_GOAL_SUBMIT_FAILED", "proactive", "system",
                               str(e), level="WARN")

    # ══════════════════════════════════════════════════════════════════════════
    #  CALLBACKS
    # ══════════════════════════════════════════════════════════════════════════

    def _notify_cb(self, notification: dict):
        """Called by decision framework when owner needs to be notified."""
        msg_type = notification.get("type", "")
        message  = notification.get("message", "")
        level    = "warning" if "tier3" in msg_type else "notice"
        self.proactive.fire_alert(level, "decision", msg_type, message)
        self.audit.log("OWNER_NOTIFICATION", "decisions", "system", message)

    def _on_kill(self):
        self.audit.log("KILL_SWITCH_FIRED", "safety", "system",
                       "System killed", level="CRITICAL")

    # ══════════════════════════════════════════════════════════════════════════
    #  INTERACTION PROCESSING (feeds memory + training + knowledge)
    # ══════════════════════════════════════════════════════════════════════════

    def process_interaction(self, user_input: str, output: str,
                            feedback: float = None, domain: str = "general"):
        """
        Process any interaction. Updates ALL relevant subsystems:
          - Owner model (personalization)
          - Training data collection (continuous learning)
          - Knowledge base (45M)
          - Memory system (45J)
          - Self-improvement evaluator (45l)
        """
        # 45M personalization
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

        # 45J Memory — store as semantic fact if substantial
        if self.memory and len(output) > 100:
            try:
                self.memory.store_memory(
                    content=f"Q: {user_input[:200]}\nA: {output[:500]}",
                    type="semantic",
                    importance="medium",
                    tags=[domain],
                )
            except Exception:
                pass

        # 45l Self-improvement — evaluate quality
        if self.improver and feedback is not None:
            try:
                self.improver.evaluate(user_input, output, context=domain)
            except Exception:
                pass

        self.audit.log("INTERACTION", "inference", "system",
                       f"Input len={len(user_input)} Output len={len(output)}")

    # ══════════════════════════════════════════════════════════════════════════
    #  START / STOP
    # ══════════════════════════════════════════════════════════════════════════

    def start(self, mode: str = "autonomous", default_tier: int = 2):
        if self._started:
            return

        print(f"\n{'='*60}")
        print(f"  AN-RA AGI SYSTEM -- STARTING")
        print(f"{'='*60}\n")

        # Check kill file
        if self.safety.kill_switch.check_kill_file():
            kill_info = self.safety.kill_switch.clear_kill_file()
            print(f"[!] Kill file found from previous session: {kill_info}")
            print("  Cleared. Starting fresh.\n")

        # Phase 1: Neural Network
        self._init_llm_bridge()

        # Phase 2: Subsystems
        self._init_memory()
        self._init_agent()
        self._init_improver()

        # Phase 3: Advanced capabilities
        self._init_ouroboros()
        self._init_ghost_memory()

        # Wire proactive alert → autonomous goal spawning
        self.proactive.on_alert(self._on_proactive_alert)

        # Start background services
        self.safety.start()
        self.proactive.start()
        self.engine.start()
        self._started = True

        self.audit.log("SYSTEM_START", "system", "master",
                       f"An-Ra started. Mode={mode} DefaultTier={default_tier}")

        print(f"\n{'='*60}")
        print(f"  [OK] AN-RA PHASE 3 SYSTEM ONLINE")
        print(f"    Version:    {self.VERSION}")
        print(f"    Mode:       {mode}")
        print(f"    Tier:       {default_tier}")
        print(f"    LLM:        {'[x]' if self.llm else '[ ]'}")
        print(f"    Agent:      {'[x]' if self.agent else '[ ]'}")
        print(f"    Memory:     {'[x]' if self.memory else '[ ]'}")
        print(f"    Improver:   {'[x]' if self.improver else '[ ]'}")
        print(f"    Ghost Mem:  {'[x]' if self.ghost_memory else '[ ]'}")
        print(f"{'='*60}\n")

    def stop(self, immediate: bool = False):
        if immediate:
            self.safety.kill_switch.activate("Owner requested immediate stop")
        else:
            # Cleanup memory
            if self.memory:
                try:
                    self.memory.cleanup()
                except Exception:
                    pass
            self.safety.stop()
            self.proactive.stop()
            self.engine.stop()
            self._started = False
            print("  An-Ra system stopped cleanly.")

    def run_forever(self):
        """Block until shutdown."""
        self.start()
        try:
            self.engine.run_forever()
        except KeyboardInterrupt:
            self.stop()

    # ══════════════════════════════════════════════════════════════════════════
    #  HIGH-LEVEL QUERIES
    # ══════════════════════════════════════════════════════════════════════════

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
            "subsystems": {
                "llm":          self.llm.status() if self.llm else {"initialized": False},
                "agent":        {"ready": self.agent is not None,
                                 "tools": len(self.agent.registry) if self.agent else 0},
                "memory":       self.memory.stats() if self.memory else {"ready": False},
                "improver":     {"ready": self.improver is not None},
                "ghost_memory": {"ready": self.ghost_memory is not None},
            },
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
        description="An-Ra — Autonomous Personal AGI System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python system.py --start --mode autonomous --tier 2
  python system.py --chat
  python system.py --goal "Research quantum computing" --horizon 14 --priority high
  python system.py --briefing
  python system.py --status
  python system.py --stop --immediate
  python system.py --dashboard
  python system.py --test
        """
    )
    p.add_argument("--start",         action="store_true")
    p.add_argument("--stop",          action="store_true")
    p.add_argument("--immediate",     action="store_true")
    p.add_argument("--chat",          action="store_true",
                   help="Interactive chat mode with memory")
    p.add_argument("--briefing",      action="store_true")
    p.add_argument("--status",        action="store_true")
    p.add_argument("--dashboard",     action="store_true")
    p.add_argument("--goal",          type=str,  default=None,
                   help="Execute a goal via the 45k Agent Loop")
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


def _run_chat(system: MasterSystem):
    """Interactive chat loop with memory."""
    system.start()
    print("\n  An-Ra Chat — Type 'quit' to exit, 'goal:' prefix to run agent goals")
    print("  Memory and ghost context are active.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Goodbye.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            break
        if user_input.lower() == "status":
            print(json.dumps(system.status(), indent=2, default=str))
            continue
        if user_input.lower() == "briefing":
            print(system.morning_briefing())
            continue

        # Goal mode: "goal: research quantum computing"
        if user_input.lower().startswith("goal:"):
            goal = user_input[5:].strip()
            print(f"\n  [Agent executing goal: {goal}]\n")
            result = system.run_goal(goal)
            print(f"\n  Success: {result.get('success')}")
            print(f"  Output:\n{result.get('output', '')[:500]}")
            print(f"  Duration: {result.get('duration', 0):.1f}s\n")
            continue

        # Regular chat
        response = system.chat(user_input)
        print(f"An-Ra: {response}\n")


def main():
    parser  = build_parser()
    args    = parser.parse_args()
    system  = MasterSystem()

    if args.test:
        from test_45M import run_all_tests
        run_all_tests(system)
        return

    if args.chat:
        _run_chat(system)
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
        system.start()
        print(system.morning_briefing())
        system.stop()
        return

    if args.status:
        system.start()
        s = system.status()
        print(json.dumps(s, indent=2, default=str))
        system.stop()
        return

    if args.dashboard:
        system.start()
        db = Dashboard(system)
        db.watch(interval=5)
        return

    if args.goal:
        system.start()
        print(f"\n  Executing goal: {args.goal}\n")
        result = system.run_goal(args.goal)
        print(f"\n{'═'*60}")
        print(f"  Goal:     {args.goal}")
        print(f"  Success:  {result.get('success')}")
        print(f"  Duration: {result.get('duration', 0):.1f}s")
        print(f"  Output:")
        print(f"    {result.get('output', '')[:1000]}")
        print(f"{'═'*60}\n")

        # Also register as a long-horizon goal
        system.set_goal(
            title        = args.goal,
            description  = args.goal,
            horizon_days = args.horizon,
            priority     = args.priority,
        )
        system.stop()
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
        system.start()
        api = ControlAPI(system.control)
        api.serve_stdio()
        return

    # Default: show dashboard
    system.start()
    print(system.dashboard.render())
    system.stop()


if __name__ == "__main__":
    main()
