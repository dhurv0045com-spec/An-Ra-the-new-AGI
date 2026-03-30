"""
control/ — Owner Control Layer

interface.py  — Owner control layer (approve/reject/inspect/configure)
dashboard.py  — Real-time view of system state
api.py        — Control API (JSON over stdio for phone/PC control)

Single module. Everything the owner needs to stay in control.
Works on phone and PC. Real-time. Always accessible.
"""

import json, uuid, time, sys, os
from datetime import datetime
from typing import Optional, Any, Dict, List
from pathlib import Path


STATE_DIR = Path("state")


class ControlInterface:
    """
    The owner's single point of control over everything.
    Approve decisions, adjust autonomy, inspect memory, trigger actions.
    """

    def __init__(self, system=None):
        self._system = system   # reference to MasterSystem

    def _sys(self):
        if not self._system:
            raise RuntimeError("Control interface not wired to system")
        return self._system

    # ── Decision control ────────────────────────────────────────────────────────

    def approve(self, request_id: str) -> dict:
        return {"approved": self._sys().decisions.approve(request_id)}

    def reject(self, request_id: str, reason: str = "") -> dict:
        return {"rejected": self._sys().decisions.reject(request_id, reason)}

    def pending_approvals(self) -> List[dict]:
        from autonomy.decisions import DecisionRequest
        pending = self._sys().decisions.pending_approvals()
        return [{"request_id": p.request_id, "title": p.title,
                 "tier": p.tier, "stakes": p.stakes,
                 "proposed": p.proposed_action} for p in pending]

    # ── Goal control ────────────────────────────────────────────────────────────

    def create_goal(self, title: str, description: str,
                    horizon_days: int = 7, priority: str = "medium") -> dict:
        goal = self._sys().goals.create_goal(
            title=title, description=description,
            horizon_days=horizon_days, priority=priority
        )
        return {"goal_id": goal.goal_id, "title": goal.title,
                "target_date": goal.target_date,
                "steps": len(self._sys().goals.db.get_steps(goal.goal_id))}

    def goal_status(self, goal_id: Optional[str] = None) -> dict:
        if goal_id:
            g = self._sys().goals.db.get_goal(goal_id)
            if not g: return {"error": "Goal not found"}
            steps = self._sys().goals.db.get_steps(goal_id)
            return {"goal": g.title, "progress": g.progress_pct,
                    "status": g.status, "steps": len(steps),
                    "done": sum(1 for s in steps if s.status == "done")}
        active = self._sys().goals.db.list_goals("active")
        return {"active_goals": [{"id": g.goal_id, "title": g.title,
                                   "progress": g.progress_pct,
                                   "priority": g.priority} for g in active]}

    def pause_goal(self, goal_id: str) -> dict:
        g = self._sys().goals.db.get_goal(goal_id)
        if g:
            g.status = "paused"
            self._sys().goals.db.save_goal(g)
            return {"paused": True}
        return {"error": "Not found"}

    # ── Autonomy control ────────────────────────────────────────────────────────

    def set_tier_policy(self, category: str, tier: int) -> dict:
        self._sys().decisions.set_tier_policy(category, tier)
        return {"set": True, "category": category, "tier": tier}

    def audit_trail(self, limit: int = 50) -> List[dict]:
        return self._sys().safety.audit.full_trail(limit=limit)

    # ── Memory/knowledge control ────────────────────────────────────────────────

    def inspect_owner_model(self) -> dict:
        return self._sys().owner_modeler.inspect()

    def correct_owner_model(self, field: str, value: Any) -> dict:
        ok = self._sys().owner_modeler.correct(field, value)
        return {"corrected": ok}

    def delete_owner_field(self, field: str) -> dict:
        ok = self._sys().owner_modeler.delete_field(field)
        return {"deleted": ok}

    def search_knowledge(self, query: str, domain: Optional[str] = None) -> List[dict]:
        return self._sys().knowledge_base.search(query, domain)

    def add_knowledge(self, title: str, content: str,
                      domain: str = "general") -> dict:
        entry = self._sys().knowledge_base.add(title, content, domain,
                                                source="owner_added")
        return {"entry_id": entry.entry_id, "title": entry.title}

    # ── System control ──────────────────────────────────────────────────────────

    def status(self) -> dict:
        return self._sys().status()

    def morning_briefing(self) -> str:
        return self._sys().morning_briefing()

    def stop(self, immediate: bool = False) -> dict:
        if immediate:
            self._sys().safety.kill_switch.activate("Owner requested immediate stop")
        else:
            self._sys().engine.stop()
        return {"stopping": True, "immediate": immediate}

    def trigger_training(self) -> dict:
        run = self._sys().trigger_training()
        return {"run_id": run.run_id, "status": run.status}

    def safety_audit(self) -> dict:
        return self._sys().safety.run_safety_audit()

    def download_backup(self, path: str = "backup.json") -> dict:
        """Export full system state as a JSON backup."""
        backup = {
            "timestamp":   datetime.utcnow().isoformat(),
            "version":     "45M_phase3",
            "owner_model": self._sys().owner_modeler.inspect(),
            "knowledge":   self._sys().knowledge_base.export_graph(),
            "goals":       [g.__dict__ if hasattr(g, "__dict__") else g
                           for g in self._sys().goals.db.list_goals()],
            "audit_trail": self._sys().safety.audit.full_trail(limit=500),
        }
        with open(path, "w") as f:
            json.dump(backup, f, indent=2, default=str)
        return {"backup_path": path, "size_bytes": os.path.getsize(path)}


class Dashboard:
    """Real-time view of system state. Formatted for terminal or API consumption."""

    def __init__(self, system=None):
        self._sys = system

    def render(self) -> str:
        """Render a full system status dashboard as text."""
        s = self._sys.status() if self._sys else {}
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

        lines = [
            "╔══════════════════════════════════════════════════════╗",
            f"║  45M AUTONOMOUS AI SYSTEM — {now}  ║",
            "╚══════════════════════════════════════════════════════╝",
            "",
            f"  System:     {'● RUNNING' if s.get('running') else '○ STOPPED'}",
            f"  Uptime:     {self._fmt_uptime(s.get('uptime_seconds', 0))}",
            f"  Heartbeat:  {s.get('last_heartbeat', 'N/A')[:19] if s.get('last_heartbeat') else 'N/A'}",
        ]

        # Resources
        res = s.get("resources", {})
        if res.get("cpu_pct") is not None:
            lines.append(f"  CPU:        {res['cpu_pct']:.0f}%  "
                         f"RAM: {res.get('mem_pct', 0):.0f}%")

        # Goals
        goals = s.get("goals", {})
        if goals:
            lines += ["", "  GOALS:"]
            for g in goals.get("active_goals", [])[:5]:
                lines.append(f"    [{g.get('priority','?').upper()[:1]}] "
                              f"{g.get('title','')[:40]}  "
                              f"{g.get('progress',0):.0f}%")

        # Pending approvals
        pending = s.get("pending_approvals", [])
        if pending:
            lines += ["", f"  ⚠ PENDING APPROVAL ({len(pending)}):"]
            for p in pending[:3]:
                lines.append(f"    [{p.get('tier','?')}] {p.get('title','')[:50]}")

        # Safety
        safety = s.get("safety", {})
        if safety:
            lines += ["", f"  SAFETY: {safety.get('risk_assessment','?')}  "
                         f"| Violations: {safety.get('violations',0)} "
                         f"| Anomalies: {safety.get('open_anomalies',0)}"]

        lines += ["", "  [s]tatus  [g]oals  [a]pprove  [r]eject  [q]uit"]
        return "\n".join(lines)

    def _fmt_uptime(self, seconds: int) -> str:
        if not seconds:
            return "N/A"
        h, r = divmod(int(seconds), 3600)
        m, s = divmod(r, 60)
        return f"{h}h {m}m {s}s"

    def watch(self, interval: int = 5):
        """
        Live-updating terminal dashboard.
        Clears and redraws every `interval` seconds.
        """
        try:
            while True:
                os.system("clear" if os.name != "nt" else "cls")
                print(self.render())
                time.sleep(interval)
        except KeyboardInterrupt:
            pass


class ControlAPI:
    """
    JSON-over-stdio control API.
    Accepts JSON commands on stdin, returns JSON responses on stdout.
    Enables remote control from phone, scripts, or other processes.
    """

    COMMANDS = {
        "status":          lambda ctrl, _: ctrl.status(),
        "approve":         lambda ctrl, p: ctrl.approve(p["request_id"]),
        "reject":          lambda ctrl, p: ctrl.reject(p["request_id"], p.get("reason","")),
        "pending":         lambda ctrl, _: ctrl.pending_approvals(),
        "create_goal":     lambda ctrl, p: ctrl.create_goal(**p),
        "goal_status":     lambda ctrl, p: ctrl.goal_status(p.get("goal_id")),
        "morning_brief":   lambda ctrl, _: {"brief": ctrl.morning_briefing()},
        "inspect_owner":   lambda ctrl, _: ctrl.inspect_owner_model(),
        "correct_owner":   lambda ctrl, p: ctrl.correct_owner_model(p["field"], p["value"]),
        "search_kb":       lambda ctrl, p: ctrl.search_knowledge(p["query"], p.get("domain")),
        "add_kb":          lambda ctrl, p: ctrl.add_knowledge(**p),
        "safety_audit":    lambda ctrl, _: ctrl.safety_audit(),
        "set_tier":        lambda ctrl, p: ctrl.set_tier_policy(p["category"], p["tier"]),
        "audit_trail":     lambda ctrl, p: ctrl.audit_trail(p.get("limit", 50)),
        "stop":            lambda ctrl, p: ctrl.stop(p.get("immediate", False)),
        "train":           lambda ctrl, _: ctrl.trigger_training(),
        "backup":          lambda ctrl, p: ctrl.download_backup(p.get("path", "backup.json")),
    }

    def __init__(self, control: ControlInterface):
        self.ctrl = control

    def handle(self, raw: str) -> str:
        try:
            req  = json.loads(raw.strip())
            cmd  = req.get("command", "")
            params = req.get("params", {})
            fn   = self.COMMANDS.get(cmd)
            if not fn:
                return json.dumps({"ok": False, "error": f"Unknown command: {cmd}",
                                   "available": list(self.COMMANDS)})
            result = fn(self.ctrl, params)
            return json.dumps({"ok": True, "result": result}, default=str)
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e), "result": None})

    def serve_stdio(self):
        """
        Listen for JSON commands on stdin, respond on stdout.
        One command per line.
        """
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            print(self.handle(line), flush=True)
