"""
dashboard/ — Improvement Dashboard

metrics.py    — Collect all metrics from all subsystems
reporter.py   — Weekly improvement reports
alerts.py     — Degradation detection and alerts
visualizer.py — Terminal dashboard renderer
"""

import json, time, math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any
from collections import defaultdict


STATE_DIR = Path("state")
STATE_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR = Path("state/reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  METRICS COLLECTOR
# ══════════════════════════════════════════════════════════════════════════════

class MetricsCollector:
    """Aggregates metrics from all subsystems into one snapshot."""

    def __init__(self, evaluator=None, prompt_opt=None, failure_analyzer=None,
                 skill_library=None, self_trainer=None, tool_creator=None,
                 connector_registry=None):
        self.evaluator        = evaluator
        self.prompt_opt       = prompt_opt
        self.failure_analyzer = failure_analyzer
        self.skill_library    = skill_library
        self.self_trainer     = self_trainer
        self.tool_creator     = tool_creator
        self.connectors       = connector_registry

    def collect(self) -> dict:
        """Collect a complete metrics snapshot."""
        ts = datetime.utcnow().isoformat()
        snapshot = {"timestamp": ts, "components": {}}

        if self.evaluator:
            try:
                snapshot["components"]["output_quality"] = self.evaluator.recent_stats(n=100)
            except Exception as e:
                snapshot["components"]["output_quality"] = {"error": str(e)}

        if self.prompt_opt:
            try:
                snapshot["components"]["prompts"] = self.prompt_opt.optimization_report()
            except Exception as e:
                snapshot["components"]["prompts"] = {"error": str(e)}

        if self.failure_analyzer:
            try:
                snapshot["components"]["failures"] = self.failure_analyzer.summary_report(days=7)
            except Exception as e:
                snapshot["components"]["failures"] = {"error": str(e)}

        if self.skill_library:
            try:
                snapshot["components"]["skills"] = self.skill_library.stats()
            except Exception as e:
                snapshot["components"]["skills"] = {"error": str(e)}

        if self.self_trainer:
            try:
                snapshot["components"]["training"] = self.self_trainer.pipeline_stats()
            except Exception as e:
                snapshot["components"]["training"] = {"error": str(e)}

        if self.tool_creator:
            try:
                snapshot["components"]["tools"]    = self.tool_creator.db.stats()
                snapshot["components"]["tool_perf"]= ToolOptReport(self.tool_creator)
            except Exception as e:
                snapshot["components"]["tools"] = {"error": str(e)}

        if self.connectors:
            try:
                snapshot["components"]["connectors"] = self.connectors.status()
            except Exception as e:
                snapshot["components"]["connectors"] = {"error": str(e)}

        # Save snapshot
        snap_file = REPORTS_DIR / f"snapshot_{ts[:10]}.json"
        try:
            existing = []
            if snap_file.exists():
                existing = json.loads(snap_file.read_text())
            existing.append(snapshot)
            snap_file.write_text(json.dumps(existing[-100:], indent=2, default=str))
        except Exception:
            pass

        return snapshot

    def trend(self, days: int = 7) -> dict:
        """Compute trends across recent snapshots."""
        snapshots = []
        for i in range(days):
            d    = (datetime.utcnow() - timedelta(days=i)).strftime("%Y-%m-%d")
            file = REPORTS_DIR / f"snapshot_{d}.json"
            if file.exists():
                try:
                    snaps = json.loads(file.read_text())
                    snapshots.extend(snaps)
                except Exception:
                    pass

        if not snapshots:
            return {"insufficient_data": True}

        # Extract quality scores
        quality_scores = [
            s["components"].get("output_quality", {}).get("avg_overall", None)
            for s in snapshots
            if "components" in s
        ]
        quality_scores = [q for q in quality_scores if q is not None]

        failure_counts = [
            s["components"].get("failures", {}).get("total_failures", None)
            for s in snapshots
            if "components" in s
        ]
        failure_counts = [f for f in failure_counts if f is not None]

        trend_data = {
            "period_days": days,
            "snapshots":   len(snapshots),
        }

        if quality_scores:
            trend_data["quality"] = {
                "first": quality_scores[-1],
                "last":  quality_scores[0],
                "delta": quality_scores[0] - quality_scores[-1],
                "trend": "improving" if quality_scores[0] > quality_scores[-1] else "declining",
            }

        if failure_counts:
            trend_data["failures"] = {
                "first": failure_counts[-1],
                "last":  failure_counts[0],
                "trend": "decreasing" if failure_counts[0] < failure_counts[-1] else "increasing",
            }

        return trend_data


def ToolOptReport(creator) -> dict:
    """Helper to get tool optimizer report safely."""
    try:
        from tools.dynamic.creator import ToolOptimizer, ToolSandbox
        opt = ToolOptimizer(creator.db, ToolSandbox())
        return opt.performance_report()
    except Exception as e:
        return {"error": str(e)}


# ══════════════════════════════════════════════════════════════════════════════
#  ALERT SYSTEM
# ══════════════════════════════════════════════════════════════════════════════

class AlertSystem:
    """
    Monitors metrics and fires alerts when anything degrades.
    """

    THRESHOLDS = {
        "quality_drop":        0.10,   # alert if quality drops >10%
        "failure_rate_high":   0.30,   # alert if failure rate >30%
        "tool_success_low":    0.65,   # alert if tool success <65%
        "training_needed":     True,   # alert when training should trigger
    }

    def __init__(self):
        self._alerts:    List[dict] = []
        self._callbacks: List      = []
        self._prev_quality: Optional[float] = None

    def on_alert(self, fn):
        self._callbacks.append(fn)

    def check(self, metrics: dict) -> List[dict]:
        """Check metrics and return any new alerts."""
        new_alerts = []
        now        = datetime.utcnow().isoformat()

        # Quality drop
        qual = metrics.get("components", {}).get("output_quality", {})
        current_q = qual.get("avg_overall")
        if current_q and self._prev_quality:
            drop = self._prev_quality - current_q
            if drop > self.THRESHOLDS["quality_drop"]:
                a = self._make_alert("QUALITY_DROP", "critical",
                    f"Output quality dropped {drop:.1%}: "
                    f"{self._prev_quality:.3f} → {current_q:.3f}")
                new_alerts.append(a)
        if current_q:
            self._prev_quality = current_q

        # High failure rate
        fail = metrics.get("components", {}).get("failures", {})
        fail_rate = 1 - fail.get("resolution_rate", 1.0) if fail else None
        if fail_rate and fail_rate > self.THRESHOLDS["failure_rate_high"]:
            a = self._make_alert("HIGH_FAILURE_RATE", "warning",
                f"Failure rate is {fail_rate:.0%} "
                f"({fail.get('total_failures', 0)} failures in 7 days)")
            new_alerts.append(a)

        # Training needed
        train = metrics.get("components", {}).get("training", {})
        if train.get("should_trigger"):
            a = self._make_alert("TRAINING_RECOMMENDED", "info",
                f"Training recommended: {train.get('trigger_reason')}")
            new_alerts.append(a)

        # Recurring patterns
        patterns = fail.get("pattern_details", [])
        critical  = [p for p in patterns if p.get("count", 0) >= 5]
        if critical:
            a = self._make_alert("RECURRING_FAILURES", "warning",
                f"{len(critical)} recurring failure patterns detected: "
                f"{', '.join(p['type'] for p in critical[:3])}")
            new_alerts.append(a)

        self._alerts.extend(new_alerts)
        for alert in new_alerts:
            for cb in self._callbacks:
                try: cb(alert)
                except Exception: pass

        return new_alerts

    def _make_alert(self, alert_type: str, severity: str, message: str) -> dict:
        return {
            "alert_id": f"{alert_type}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
            "type":     alert_type,
            "severity": severity,
            "message":  message,
            "time":     datetime.utcnow().isoformat(),
            "acked":    False,
        }

    def get_active(self) -> List[dict]:
        return [a for a in self._alerts if not a.get("acked")]

    def ack(self, alert_id: str):
        for a in self._alerts:
            if a["alert_id"] == alert_id:
                a["acked"] = True


# ══════════════════════════════════════════════════════════════════════════════
#  WEEKLY REPORTER
# ══════════════════════════════════════════════════════════════════════════════

class WeeklyReporter:
    """Generates and saves weekly improvement reports."""

    def __init__(self, metrics: MetricsCollector, alerts: AlertSystem):
        self.metrics = metrics
        self.alerts  = alerts

    def generate(self) -> str:
        """Generate a weekly report as a formatted string."""
        now      = datetime.utcnow()
        snapshot = self.metrics.collect()
        trend    = self.metrics.trend(days=7)
        active   = self.alerts.get_active()

        report = []
        report.append("╔══════════════════════════════════════════════════════╗")
        report.append(f"║  45L WEEKLY IMPROVEMENT REPORT — {now.strftime('%Y-%m-%d')}      ║")
        report.append("╚══════════════════════════════════════════════════════╝")
        report.append("")

        # Quality
        qual  = snapshot.get("components", {}).get("output_quality", {})
        q_val = qual.get("avg_overall", 0)
        trend_q = trend.get("quality", {})
        q_arrow = "↑" if trend_q.get("trend") == "improving" else "↓" if trend_q.get("trend") == "declining" else "→"
        report.append(f"  OUTPUT QUALITY:   {q_val:.3f} {q_arrow}  (flagged: {qual.get('flagged_pct', 0):.1%})")

        dim = qual.get("dimension_avgs", {})
        if dim:
            report.append(f"    accuracy={dim.get('accuracy',0):.2f}  "
                          f"relevance={dim.get('relevance',0):.2f}  "
                          f"completeness={dim.get('completeness',0):.2f}  "
                          f"clarity={dim.get('clarity',0):.2f}")

        # Failures
        fail = snapshot.get("components", {}).get("failures", {})
        if fail:
            report.append(f"\n  FAILURES (7d):    {fail.get('total_failures',0)} total  "
                          f"({fail.get('resolution_rate',0):.0%} resolved)  "
                          f"patterns: {fail.get('patterns',0)}")
            tops = fail.get("top_errors", [])[:3]
            for etype, count in tops:
                report.append(f"    {count}× {etype}")

        # Skills
        skills = snapshot.get("components", {}).get("skills", {})
        if skills:
            report.append(f"\n  SKILL LIBRARY:    {skills.get('total_skills',0)} skills  "
                          f"avg quality: {skills.get('avg_quality',0):.2f}")

        # Training
        train = snapshot.get("components", {}).get("training", {})
        if train:
            report.append(f"\n  TRAINING:         {train.get('total_examples',0)} examples  "
                          f"{train.get('total_runs',0)} runs  "
                          f"{train.get('deployed_runs',0)} deployed")
            if train.get("should_trigger"):
                report.append(f"    ⚡ Training recommended: {train.get('trigger_reason','')}")

        # Tools
        tools = snapshot.get("components", {}).get("tools", {})
        if tools:
            report.append(f"\n  DYNAMIC TOOLS:    {tools.get('approved',0)} approved  "
                          f"{tools.get('pending',0)} pending review  "
                          f"{tools.get('retired',0)} retired")

        # Prompts
        prompts = snapshot.get("components", {}).get("prompts", {})
        if prompts:
            report.append(f"\n  PROMPTS:          {prompts.get('active_prompts',0)} active  "
                          f"avg score: {prompts.get('avg_score',0):.2f}")
            if prompts.get("needs_work"):
                report.append(f"    Need optimization: {', '.join(prompts['needs_work'])}")

        # Trend
        if trend.get("quality"):
            delta = trend["quality"].get("delta", 0)
            sign  = "+" if delta >= 0 else ""
            report.append(f"\n  7-DAY TREND:      Quality {sign}{delta:.3f}  "
                          f"({trend['quality']['trend']})")

        # Alerts
        if active:
            report.append(f"\n  ACTIVE ALERTS ({len(active)}):")
            for a in active[:5]:
                sev_icon = "🔴" if a["severity"]=="critical" else "🟡" if a["severity"]=="warning" else "🔵"
                report.append(f"    {sev_icon} [{a['type']}] {a['message'][:80]}")

        report.append("")
        report.append("  ════════════════════════════════════════════════════")
        report.append(f"  Generated: {now.isoformat()}")

        report_text = "\n".join(report)

        # Save report
        report_file = REPORTS_DIR / f"weekly_{now.strftime('%Y_%m_%d')}.txt"
        report_file.write_text(report_text)

        return report_text


# ══════════════════════════════════════════════════════════════════════════════
#  TERMINAL DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

class TerminalDashboard:
    """
    Renders a live terminal dashboard showing all metrics.
    Updates every N seconds.
    """

    def __init__(self, system):
        self.system = system

    def render(self) -> str:
        s      = self.system
        snap   = s.metrics.collect()
        alerts = s.alerts.get_active()
        now    = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

        lines = [
            "╔══════════════════════════════════════════════════════════╗",
            f"║  45L SELF-IMPROVEMENT DASHBOARD  —  {now}  ║",
            "╚══════════════════════════════════════════════════════════╝",
        ]

        comps = snap.get("components", {})

        # Output quality
        q = comps.get("output_quality", {})
        qv = q.get("avg_overall", 0)
        bar = self._bar(qv, 20)
        lines.append(f"\n  📊 OUTPUT QUALITY    {bar} {qv:.3f}")
        if q.get("dimension_avgs"):
            d = q["dimension_avgs"]
            lines.append(f"     acc={d.get('accuracy',0):.2f} rel={d.get('relevance',0):.2f} "
                         f"comp={d.get('completeness',0):.2f} clar={d.get('clarity',0):.2f}")

        # Failures
        f = comps.get("failures", {})
        if f:
            lines.append(f"\n  ⚠  FAILURES (7d)     {f.get('total_failures',0)} total  "
                         f"{f.get('resolution_rate',0):.0%} resolved  "
                         f"{f.get('patterns',0)} patterns")

        # Tools
        t = comps.get("tools", {})
        if t:
            lines.append(f"\n  🔧 TOOLS             {t.get('approved',0)} approved  "
                         f"{t.get('pending',0)} pending  {t.get('retired',0)} retired")

        # Training
        tr = comps.get("training", {})
        if tr:
            lines.append(f"\n  🏋  TRAINING          {tr.get('total_examples',0)} examples  "
                         f"{tr.get('deployed_runs',0)} deployed runs")
            if tr.get("should_trigger"):
                lines.append(f"     ⚡ Ready to train: {tr.get('trigger_reason','')}")

        # Skills
        sk = comps.get("skills", {})
        if sk:
            lines.append(f"\n  🧠 SKILLS            {sk.get('total_skills',0)} stored  "
                         f"avg quality {sk.get('avg_quality',0):.2f}")

        # Prompts
        pr = comps.get("prompts", {})
        if pr:
            lines.append(f"\n  📝 PROMPTS           {pr.get('active_prompts',0)} active  "
                         f"avg score {pr.get('avg_score',0):.2f}")

        # Alerts
        if alerts:
            lines.append(f"\n  🚨 ALERTS ({len(alerts)})")
            for a in alerts[:3]:
                lines.append(f"     [{a['severity'].upper()}] {a['message'][:60]}")

        lines.append("")
        return "\n".join(lines)

    def _bar(self, value: float, width: int = 20) -> str:
        filled = int(value * width)
        return "█" * filled + "░" * (width - filled)

    def watch(self, interval: int = 10):
        """Live-updating dashboard."""
        import os
        try:
            while True:
                os.system("clear" if os.name != "nt" else "cls")
                print(self.render())
                time.sleep(interval)
        except KeyboardInterrupt:
            pass
