#!/usr/bin/env python3
"""
anra.py — An-Ra AGI Unified Entry Point
=========================================

The single command to interact with An-Ra.

Usage:
    python anra.py                         # Show system dashboard
    python anra.py --start                 # Start continuous autonomous engine
    python anra.py --chat                  # Interactive chat with memory
    python anra.py --goal "..."            # Execute a goal via Agent Loop
    python anra.py --status                # System status (all subsystems)
    python anra.py --briefing              # Morning briefing + sovereignty report
    python anra.py --test                  # Run full test suite
    python anra.py --dashboard             # Live dashboard

Phase 3 specific:
    python anra.py --phase3-status         # Detailed Phase 3 subsystem status
    python anra.py --symbolic "query"      # Direct math/logic/code query (45Q)
    python anra.py --sovereignty-report    # Latest nightly self-improvement report
    python anra.py --sovereignty-run       # Trigger improvement pipeline now
"""

import sys
import os
import json
import argparse
from pathlib import Path

# ── Resolve all project paths ─────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
PHASE2_45M   = PROJECT_ROOT / "phase2" / "45M"

# ── Add Phase 3 paths to sys.path for direct imports ─────────────────────────
for p3 in ["45N", "45O", "45P", "45Q", "45R"]:
    p = PROJECT_ROOT / "phase3" / p3
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Set working directory to 45M so all relative state/ paths work
os.chdir(str(PHASE2_45M))

# Add 45M to path so system.py imports work
sys.path.insert(0, str(PHASE2_45M))

# ── Delegate to the master system ────────────────────────────────────────────
from system import MasterSystem, build_parser, _run_chat, Dashboard, ControlAPI


def _phase3_status(system: MasterSystem):
    """Print detailed Phase 3 subsystem status."""
    print(f"\n{'='*60}")
    print("  AN-RA PHASE 3 SUBSYSTEMS")
    print(f"{'='*60}\n")

    status = system.status()
    subs = status.get("subsystems", {})

    modules = [
        ("45N — Identity Injector",    "identity"),
        ("45O — Ouroboros Reasoning",  "ouroboros"),
        ("45P — Ghost State Memory",   "ghost_memory"),
        ("45Q — Symbolic Logic Bridge","symbolic"),
        ("45R — Sovereignty Daemon",   "sovereignty"),
    ]

    for name, key in modules:
        info = subs.get(key, {})
        ready = info.get("ready", info.get("enabled", False))
        mark = "[x]" if ready else "[ ]"
        print(f"  {mark}  {name}")
        if isinstance(info, dict):
            for k, v in info.items():
                if k not in ("ready", "enabled") and not isinstance(v, dict):
                    print(f"         {k}: {v}")
        print()

    print(f"{'='*60}\n")


def _symbolic_query(query: str):
    """Run a direct 45Q symbolic query (math/logic/code)."""
    print(f"\n[ Symbolic Bridge Query ]\nQ: {query}\n")
    try:
        sys.path.insert(0, str(PROJECT_ROOT / "phase3" / "45Q"))
        from symbolic_bridge import query as sym_query
        result = sym_query(query)
        print(f"Mode:       {result.mode}")
        print(f"Verdict:    {result.verdict}")
        print(f"Confidence: {result.confidence:.0%}")
        print(f"Answer:     {result.answer_text}")
        print(f"\nSteps:")
        for step in result.steps[:5]:
            print(f"  {step}")
        if result.warnings:
            print(f"\nWarnings:")
            for w in result.warnings:
                print(f"  ⚠  {w}")
    except ImportError as e:
        print(f"[ERROR] Symbolic bridge not available: {e}")
        print("Install: pip install sympy scipy")
    except Exception as e:
        print(f"[ERROR] {e}")


def _sovereignty_report(system: MasterSystem):
    """Print the latest sovereignty nightly report."""
    if not system.sovereignty:
        print("[Sovereignty] Daemon not initialized.")
        return
    report = system.sovereignty.get_nightly_report()
    bench = system.sovereignty.get_benchmark_summary()
    print(f"\n{'='*60}")
    print("  SOVEREIGNTY NIGHTLY REPORT (45R)")
    print(f"{'='*60}\n")
    if bench:
        print(bench)
        print()
    print(report)


def _sovereignty_trigger(system: MasterSystem):
    """Trigger the sovereignty improvement pipeline right now."""
    if not system.sovereignty:
        print("[Sovereignty] Daemon not initialized.")
        return
    print("[Sovereignty] Triggering improvement pipeline...")
    ok = system.sovereignty.trigger_pipeline()
    if ok:
        print("[Sovereignty] Pipeline triggered. Results will appear in the nightly report.")
    else:
        print("[Sovereignty] Could not trigger pipeline — daemon may not be running.")


def main():
    # ── Extended parser ───────────────────────────────────────────────────────
    parser = build_parser()
    parser.add_argument("--phase3-status",   action="store_true",
                        help="Show detailed Phase 3 subsystem status")
    parser.add_argument("--symbolic",        type=str, default=None,
                        help="Direct math/logic/code query via 45Q (e.g. 'solve x^2=9')")
    parser.add_argument("--sovereignty-report", action="store_true",
                        help="Show the latest nightly self-improvement report")
    parser.add_argument("--sovereignty-run", action="store_true",
                        help="Trigger the sovereignty improvement pipeline now")

    args = parser.parse_args()

    # ── Symbolic query — no system needed ────────────────────────────────────
    if args.symbolic:
        _symbolic_query(args.symbolic)
        return

    system = MasterSystem()

    # ── Phase 3 specific commands ─────────────────────────────────────────────
    if args.phase3_status:
        system.start()
        _phase3_status(system)
        system.stop()
        return

    if args.sovereignty_report:
        system.start()
        _sovereignty_report(system)
        system.stop()
        return

    if args.sovereignty_run:
        system.start()
        _sovereignty_trigger(system)
        system.stop()
        return

    # ── Original commands (delegate to system.py logic) ──────────────────────
    if getattr(args, "test", False):
        from test_45M import run_all_tests
        run_all_tests(system)
        return

    if getattr(args, "chat", False):
        _run_chat(system)
        return

    if getattr(args, "start", False):
        system.run_forever()
        return

    if getattr(args, "stop", False):
        system.engine.db.set_state("engine_running", False)
        if getattr(args, "immediate", False):
            system.safety.kill_switch.activate("CLI --stop --immediate")
        else:
            print("Stop signal sent.")
        return

    if getattr(args, "briefing", False):
        system.start()
        print(system.morning_briefing())
        system.stop()
        return

    if getattr(args, "status", False):
        system.start()
        s = system.status()
        print(json.dumps(s, indent=2, default=str))
        system.stop()
        return

    if getattr(args, "dashboard", False):
        system.start()
        db = Dashboard(system)
        db.watch(interval=5)
        return

    if getattr(args, "goal", None):
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
        system.stop()
        return

    if getattr(args, "owner_model", False) and getattr(args, "inspect", False):
        profile = system.owner_modeler.inspect()
        print(json.dumps(profile, indent=2, default=str))
        return

    if getattr(args, "safety_audit", False):
        result = system.safety.run_safety_audit()
        print(json.dumps(result, indent=2, default=str))
        return

    if getattr(args, "api", False):
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
