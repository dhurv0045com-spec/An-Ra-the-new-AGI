"""
autonomy/proactive.py — Proactive Intelligence Engine

Monitors for situations the owner wants to know about.
Surfaces relevant information unprompted.
Morning briefing. Configurable triggers.
"""

import uuid, json, threading, sqlite3
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict, field
from typing import Optional, List, Dict, Callable, Any
from pathlib import Path
from enum import Enum


STATE_DIR = Path("state")
STATE_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH   = STATE_DIR / "proactive.db"


class AlertLevel(str, Enum):
    INFO     = "info"
    NOTICE   = "notice"
    WARNING  = "warning"
    URGENT   = "urgent"


@dataclass
class ProactiveAlert:
    alert_id:   str
    timestamp:  str
    level:      str
    category:   str
    title:      str
    body:       str
    source:     str
    read:       bool = False
    acted_on:   bool = False
    dismissed:  bool = False


@dataclass
class MonitorConfig:
    monitor_id: str
    name:       str
    category:   str
    enabled:    bool = True
    frequency:  str  = "hourly"   # how often to check
    last_check: Optional[str] = None
    config:     Dict[str, Any] = field(default_factory=dict)


class ProactiveDB:
    def __init__(self, path=DB_PATH):
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._lock = threading.Lock()
        self._init()

    def _init(self):
        with self._lock:
            self._conn.executescript("""
                CREATE TABLE IF NOT EXISTS alerts (
                    alert_id TEXT PRIMARY KEY, data TEXT
                );
                CREATE TABLE IF NOT EXISTS monitors (
                    monitor_id TEXT PRIMARY KEY, data TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_alert_ts ON alerts(alert_id);
            """)
            self._conn.commit()

    def save_alert(self, a: ProactiveAlert):
        with self._lock:
            self._conn.execute("INSERT OR REPLACE INTO alerts VALUES (?,?)",
                               (a.alert_id, json.dumps(asdict(a))))
            self._conn.commit()

    def get_alerts(self, unread_only=False, limit=50) -> List[ProactiveAlert]:
        with self._lock:
            rows = self._conn.execute("SELECT data FROM alerts ORDER BY alert_id DESC LIMIT ?",
                                      (limit,)).fetchall()
        alerts = [ProactiveAlert(**json.loads(r[0])) for r in rows]
        if unread_only:
            alerts = [a for a in alerts if not a.read and not a.dismissed]
        return alerts

    def mark_read(self, alert_id: str):
        with self._lock:
            row = self._conn.execute(
                "SELECT data FROM alerts WHERE alert_id=?", (alert_id,)).fetchone()
            if row:
                d = json.loads(row[0]); d["read"] = True
                self._conn.execute("UPDATE alerts SET data=? WHERE alert_id=?",
                                   (json.dumps(d), alert_id))
                self._conn.commit()

    def save_monitor(self, m: MonitorConfig):
        with self._lock:
            self._conn.execute("INSERT OR REPLACE INTO monitors VALUES (?,?)",
                               (m.monitor_id, json.dumps(asdict(m))))
            self._conn.commit()

    def get_monitors(self) -> List[MonitorConfig]:
        with self._lock:
            rows = self._conn.execute("SELECT data FROM monitors").fetchall()
        return [MonitorConfig(**json.loads(r[0])) for r in rows]


class ProactiveEngine:
    """
    Continuously monitors configured sources.
    Fires alerts to owner when something noteworthy happens.
    """

    def __init__(self):
        self.db          = ProactiveDB()
        self._monitors:  Dict[str, Callable] = {}
        self._running    = False
        self._thread:    Optional[threading.Thread] = None
        self._stop       = threading.Event()
        self._alert_cbs: List[Callable] = []   # callbacks when alert fires

        self._setup_default_monitors()

    def _setup_default_monitors(self):
        defaults = [
            MonitorConfig(str(uuid.uuid4()), "Goal Progress Monitor",
                          "goals", config={"warn_behind_pct": 20}),
            MonitorConfig(str(uuid.uuid4()), "System Health Monitor",
                          "system", config={"disk_warn_pct": 85}),
            MonitorConfig(str(uuid.uuid4()), "Training Quality Monitor",
                          "training", config={"loss_spike_threshold": 0.5}),
        ]
        existing = {m.name for m in self.db.get_monitors()}
        for m in defaults:
            if m.name not in existing:
                self.db.save_monitor(m)
                self._monitors[m.category] = self._stub_monitor

    def _stub_monitor(self, config: dict) -> Optional[ProactiveAlert]:
        return None   # Real implementations wired in from other modules

    def register_monitor(self, category: str, fn: Callable):
        """Wire in a real monitoring function for a category."""
        self._monitors[category] = fn

    def on_alert(self, cb: Callable):
        """Register a callback that fires when a new alert is created."""
        self._alert_cbs.append(cb)

    def fire_alert(self, level: str, category: str, title: str,
                   body: str, source: str = "system") -> ProactiveAlert:
        alert = ProactiveAlert(
            alert_id  = datetime.utcnow().isoformat() + "_" + str(uuid.uuid4())[:8],
            timestamp = datetime.utcnow().isoformat(),
            level     = level,
            category  = category,
            title     = title,
            body      = body,
            source    = source,
        )
        self.db.save_alert(alert)
        for cb in self._alert_cbs:
            try: cb(alert)
            except Exception: pass
        return alert

    def _run_monitors(self):
        while not self._stop.is_set():
            for m in self.db.get_monitors():
                if not m.enabled:
                    continue
                fn = self._monitors.get(m.category)
                if fn:
                    try:
                        alert = fn(m.config)
                        if alert:
                            self.db.save_alert(alert)
                    except Exception:
                        pass
                m.last_check = datetime.utcnow().isoformat()
                self.db.save_monitor(m)
            self._stop.wait(timeout=600)   # check every 10 minutes

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._run_monitors,
                                        daemon=True, name="proactive")
        self._thread.start()

    def stop(self):
        self._stop.set()

    def morning_briefing(self, goals_manager=None, audit_log=None) -> str:
        """Compile a morning briefing from all sources."""
        now   = datetime.utcnow()
        lines = [
            f"╔══════════════════════════════════════╗",
            f"║  MORNING BRIEFING — {now.strftime('%Y-%m-%d %H:%M')} UTC  ║",
            f"╚══════════════════════════════════════╝",
            "",
        ]

        # Unread alerts
        unread = self.db.get_alerts(unread_only=True, limit=20)
        if unread:
            lines.append(f"  ALERTS ({len(unread)} unread):")
            for a in unread[:5]:
                lines.append(f"  [{a.level.upper()}] {a.title}")
                lines.append(f"    {a.body[:120]}")
            if len(unread) > 5:
                lines.append(f"  ... and {len(unread)-5} more")
        else:
            lines.append("  No new alerts overnight.")

        # Goal summary
        if goals_manager:
            lines.append("")
            lines.append(goals_manager.get_morning_brief())

        # Recent activity
        if audit_log:
            recent = audit_log.get_recent(hours=8, limit=10)
            if recent:
                lines.append("\n  OVERNIGHT ACTIVITY:")
                for entry in recent[:5]:
                    lines.append(f"  {entry['timestamp'][:19]}  {entry['event']}")

        lines.append("\n  Have a productive day.")
        return "\n".join(lines)
