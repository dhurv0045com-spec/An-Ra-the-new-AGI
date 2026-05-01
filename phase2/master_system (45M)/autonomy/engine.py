"""
autonomy/engine.py — Continuous Operation Engine

Runs as a persistent background service. Survives restarts and crashes.
Picks up exactly where it left off. Manages scheduled tasks, heartbeat
monitoring, resource throttling, graceful shutdown, and full activity logging.
"""

import os, time, json, uuid, signal, threading, sched, queue
import sqlite3
from datetime import datetime, timedelta
from typing import Callable, Optional, Dict, Any, List
from pathlib import Path
from dataclasses import dataclass, asdict, field


# ── Constants ──────────────────────────────────────────────────────────────────
STATE_DIR   = Path("state")
DB_PATH     = STATE_DIR / "engine.db"
LOG_PATH    = STATE_DIR / "activity.log"
PID_FILE    = STATE_DIR / "engine.pid"
HEARTBEAT_INTERVAL  = 30    # seconds
RESOURCE_CHECK_INTERVAL = 60


from shared_logger import get_shared_logger, setup_shared_logging, emit_event

# ── Logging ────────────────────────────────────────────────────────────────────
STATE_DIR.mkdir(parents=True, exist_ok=True)
setup_shared_logging(log_dir=STATE_DIR / "logs", level="INFO")
log = get_shared_logger("engine")


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class ScheduledTask:
    task_id:    str
    name:       str
    cron_desc:  str          # "daily", "weekly", "hourly", "nightly"
    fn_name:    str          # registered callable name
    last_run:   Optional[str] = None
    next_run:   Optional[str] = None
    enabled:    bool = True
    run_count:  int  = 0
    last_result: Optional[str] = None

@dataclass
class ActivityLog:
    log_id:    str
    timestamp: str
    event:     str
    component: str
    detail:    str
    level:     str = "INFO"   # INFO / WARN / ERROR / DECISION


# ── Database ───────────────────────────────────────────────────────────────────

class EngineDB:
    def __init__(self, db_path: Path = DB_PATH):
        self.path = db_path
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._lock = threading.Lock()
        self._init_schema()

    def _init_schema(self):
        with self._lock:
            self._conn.executescript("""
                CREATE TABLE IF NOT EXISTS scheduled_tasks (
                    task_id TEXT PRIMARY KEY,
                    name TEXT,
                    cron_desc TEXT,
                    fn_name TEXT,
                    last_run TEXT,
                    next_run TEXT,
                    enabled INTEGER DEFAULT 1,
                    run_count INTEGER DEFAULT 0,
                    last_result TEXT
                );
                CREATE TABLE IF NOT EXISTS activity_log (
                    log_id TEXT PRIMARY KEY,
                    timestamp TEXT,
                    event TEXT,
                    component TEXT,
                    detail TEXT,
                    level TEXT DEFAULT 'INFO'
                );
                CREATE TABLE IF NOT EXISTS engine_state (
                    key TEXT PRIMARY KEY,
                    value TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_log_ts ON activity_log(timestamp);
                CREATE INDEX IF NOT EXISTS idx_log_component ON activity_log(component);
            """)
            self._conn.commit()

    def log_activity(self, event: str, component: str, detail: str, level: str = "INFO"):
        entry = ActivityLog(
            log_id=str(uuid.uuid4()),
            timestamp=datetime.utcnow().isoformat(),
            event=event,
            component=component,
            detail=detail,
            level=level
        )
        with self._lock:
            self._conn.execute(
                "INSERT INTO activity_log VALUES (?,?,?,?,?,?)",
                (entry.log_id, entry.timestamp, entry.event,
                 entry.component, entry.detail, entry.level)
            )
            self._conn.commit()
        emit_event(log, entry.level, "ENGINE_ACTIVITY", entry.component, entry.event, message=entry.detail, details={"log_id": entry.log_id})
        return entry

    def get_logs(self, limit: int = 100, component: Optional[str] = None,
                 level: Optional[str] = None) -> List[dict]:
        q = "SELECT * FROM activity_log WHERE 1=1"
        params = []
        if component:
            q += " AND component=?"; params.append(component)
        if level:
            q += " AND level=?"; params.append(level)
        q += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        with self._lock:
            rows = self._conn.execute(q, params).fetchall()
        cols = ["log_id","timestamp","event","component","detail","level"]
        return [dict(zip(cols, r)) for r in rows]

    def set_state(self, key: str, value: Any):
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO engine_state VALUES (?,?)",
                (key, json.dumps(value))
            )
            self._conn.commit()

    def get_state(self, key: str, default=None) -> Any:
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM engine_state WHERE key=?", (key,)
            ).fetchone()
        if row:
            return json.loads(row[0])
        return default

    def upsert_task(self, task: ScheduledTask):
        with self._lock:
            self._conn.execute(
                """INSERT OR REPLACE INTO scheduled_tasks
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (task.task_id, task.name, task.cron_desc, task.fn_name,
                 task.last_run, task.next_run, int(task.enabled),
                 task.run_count, task.last_result)
            )
            self._conn.commit()

    def get_tasks(self) -> List[ScheduledTask]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM scheduled_tasks").fetchall()
        tasks = []
        for r in rows:
            tasks.append(ScheduledTask(
                task_id=r[0], name=r[1], cron_desc=r[2], fn_name=r[3],
                last_run=r[4], next_run=r[5], enabled=bool(r[6]),
                run_count=r[7], last_result=r[8]
            ))
        return tasks


# ── Scheduler ──────────────────────────────────────────────────────────────────

class TaskScheduler:
    """
    Lightweight cron-style scheduler.
    Supports: hourly, daily, nightly (2am), weekly (Sunday midnight).
    """

    CRON_OFFSETS = {
        "hourly":  timedelta(hours=1),
        "daily":   timedelta(days=1),
        "nightly": timedelta(days=1),   # runs at 02:00
        "weekly":  timedelta(weeks=1),
    }

    def __init__(self, db: EngineDB):
        self.db = db
        self._registry: Dict[str, Callable] = {}
        self._running  = False
        self._thread:  Optional[threading.Thread] = None
        self._stop_evt = threading.Event()

    def register(self, name: str, fn: Callable):
        self._registry[name] = fn

    def _next_run_time(self, cron_desc: str) -> str:
        now = datetime.utcnow()
        if cron_desc == "hourly":
            nxt = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        elif cron_desc == "nightly":
            nxt = now.replace(hour=2, minute=0, second=0, microsecond=0)
            if nxt <= now:
                nxt += timedelta(days=1)
        elif cron_desc == "weekly":
            days_ahead = 6 - now.weekday()
            if days_ahead == 0:
                days_ahead = 7
            nxt = (now + timedelta(days=days_ahead)).replace(
                hour=0, minute=0, second=0, microsecond=0)
        else:  # daily default
            nxt = (now + timedelta(days=1)).replace(
                hour=8, minute=0, second=0, microsecond=0)
        return nxt.isoformat()

    def add_task(self, name: str, cron_desc: str, fn_name: str) -> ScheduledTask:
        task = ScheduledTask(
            task_id=str(uuid.uuid4()),
            name=name,
            cron_desc=cron_desc,
            fn_name=fn_name,
            next_run=self._next_run_time(cron_desc)
        )
        self.db.upsert_task(task)
        return task

    def _run_loop(self):
        while not self._stop_evt.is_set():
            now = datetime.utcnow().isoformat()
            for task in self.db.get_tasks():
                if not task.enabled:
                    continue
                if task.next_run and task.next_run <= now:
                    self._execute_task(task)
            self._stop_evt.wait(timeout=60)

    def _execute_task(self, task: ScheduledTask):
        fn = self._registry.get(task.fn_name)
        if not fn:
            log.warning(f"Task {task.name}: no function registered for '{task.fn_name}'")
            return
        log.info(f"Running scheduled task: {task.name}")
        self.db.log_activity("TASK_START", "scheduler", f"Running: {task.name}")
        try:
            result = fn()
            task.last_result = str(result)[:500] if result else "ok"
            self.db.log_activity("TASK_DONE", "scheduler",
                                  f"Completed: {task.name} → {task.last_result}")
        except Exception as e:
            task.last_result = f"ERROR: {e}"
            log.error(f"Task {task.name} failed: {e}")
            self.db.log_activity("TASK_ERROR", "scheduler",
                                  f"Failed: {task.name}: {e}", level="ERROR")
        task.last_run  = datetime.utcnow().isoformat()
        task.next_run  = self._next_run_time(task.cron_desc)
        task.run_count += 1
        self.db.upsert_task(task)

    def start(self):
        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="scheduler")
        self._thread.start()

    def stop(self):
        self._stop_evt.set()
        if self._thread:
            self._thread.join(timeout=5)


# ── Heartbeat monitor ──────────────────────────────────────────────────────────

class Heartbeat:
    """
    Writes a heartbeat timestamp every N seconds.
    On startup, checks if last heartbeat was too long ago → crash detected.
    """

    def __init__(self, db: EngineDB, interval: int = HEARTBEAT_INTERVAL):
        self.db       = db
        self.interval = interval
        self._stop    = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def check_crash_recovery(self) -> Optional[str]:
        last = self.db.get_state("last_heartbeat")
        if last:
            last_dt = datetime.fromisoformat(last)
            gap = (datetime.utcnow() - last_dt).total_seconds()
            if gap > self.interval * 3:
                msg = f"Crash recovery: last heartbeat was {gap:.0f}s ago ({last})"
                log.warning(msg)
                self.db.log_activity("CRASH_RECOVERY", "heartbeat", msg, level="WARN")
                return msg
        return None

    def _beat_loop(self):
        while not self._stop.is_set():
            self.db.set_state("last_heartbeat", datetime.utcnow().isoformat())
            self._stop.wait(timeout=self.interval)

    def start(self):
        self._thread = threading.Thread(target=self._beat_loop, daemon=True, name="heartbeat")
        self._thread.start()

    def stop(self):
        self._stop.set()


# ── Resource manager ───────────────────────────────────────────────────────────

class ResourceManager:
    """
    Monitors CPU/memory load and throttles background work when system is busy.
    Falls back gracefully if psutil is unavailable.
    """

    def __init__(self, db: EngineDB, cpu_threshold: float = 80.0):
        self.db  = db
        self.cpu_threshold = cpu_threshold
        self._throttled = False
        try:
            import psutil
            self._psutil = psutil
        except ImportError:
            self._psutil = None

    def should_throttle(self) -> bool:
        if not self._psutil:
            return False
        try:
            cpu = self._psutil.cpu_percent(interval=1)
            mem = self._psutil.virtual_memory().percent
            if cpu > self.cpu_threshold or mem > 90:
                if not self._throttled:
                    self.db.log_activity(
                        "THROTTLE_ON", "resources",
                        f"CPU={cpu:.0f}% MEM={mem:.0f}% — throttling", level="WARN")
                self._throttled = True
                return True
        except Exception:
            pass
        if self._throttled:
            self.db.log_activity("THROTTLE_OFF", "resources", "Resources freed, resuming")
        self._throttled = False
        return False

    def get_stats(self) -> dict:
        if not self._psutil:
            return {"cpu_pct": None, "mem_pct": None, "psutil": "unavailable"}
        try:
            return {
                "cpu_pct": self._psutil.cpu_percent(interval=0.1),
                "mem_pct": self._psutil.virtual_memory().percent,
                "disk_pct": self._psutil.disk_usage('/').percent,
            }
        except Exception as e:
            return {"error": str(e)}


# ── Continuous Operation Engine ────────────────────────────────────────────────

class ContinuousEngine:
    """
    Master engine. Ties together scheduler, heartbeat, resources, and logging.
    Entry point for the entire autonomous system.
    """

    def __init__(self):
        self.db        = EngineDB()
        self.scheduler = TaskScheduler(self.db)
        self.heartbeat = Heartbeat(self.db)
        self.resources = ResourceManager(self.db)
        self._running  = False
        self._shutdown = threading.Event()

        # Register default scheduled tasks
        self._register_default_tasks()
        self._setup_signal_handlers()

    def _register_default_tasks(self):
        defaults = [
            ("Daily Goal Review",        "daily",   "daily_goal_review"),
            ("Nightly Memory Consolidation","nightly","memory_consolidation"),
            ("Weekly Self-Training Run", "weekly",  "self_training_run"),
            ("Hourly Tool Performance",  "hourly",  "tool_performance_check"),
        ]
        existing = {t.name for t in self.db.get_tasks()}
        for name, cron, fn in defaults:
            if name not in existing:
                self.scheduler.add_task(name, cron, fn)

        # Register stub handlers (real ones wired in by system.py)
        for _, _, fn in defaults:
            self.scheduler.register(fn, lambda n=fn: self._stub_task(n))

    def _stub_task(self, name: str) -> str:
        self.db.log_activity("STUB_TASK", "engine", f"Stub ran: {name}")
        return f"stub:{name}"

    def register_task_handler(self, fn_name: str, fn: Callable):
        """Wire in real implementations from other modules."""
        self.scheduler.register(fn_name, fn)

    def _setup_signal_handlers(self):
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        signal.signal(signal.SIGINT,  self._handle_shutdown)

    def _handle_shutdown(self, signum, frame):
        log.info(f"Signal {signum} received — graceful shutdown")
        self.stop()

    def start(self):
        self.db.log_activity("ENGINE_START", "engine", "Continuous engine starting")

        # Check crash recovery
        crash = self.heartbeat.check_crash_recovery()
        if crash:
            self.db.log_activity("RECOVERY", "engine", crash, level="WARN")

        # Write PID
        PID_FILE.write_text(str(os.getpid()))

        self.heartbeat.start()
        self.scheduler.start()
        self._running = True
        self.db.set_state("engine_started", datetime.utcnow().isoformat())
        self.db.set_state("engine_running", True)
        log.info("Continuous engine running")

    def stop(self):
        if not self._running:
            return
        log.info("Stopping engine...")
        self.db.log_activity("ENGINE_STOP", "engine", "Graceful shutdown")
        self.scheduler.stop()
        self.heartbeat.stop()
        self.db.set_state("engine_running", False)
        self.db.set_state("engine_stopped", datetime.utcnow().isoformat())
        if PID_FILE.exists():
            PID_FILE.unlink()
        self._running = False
        self._shutdown.set()
        log.info("Engine stopped cleanly")

    def run_forever(self):
        """Block until shutdown signal."""
        self.start()
        try:
            self._shutdown.wait()
        except KeyboardInterrupt:
            self.stop()

    def status(self) -> dict:
        uptime_start = self.db.get_state("engine_started")
        uptime_s = None
        if uptime_start:
            delta = datetime.utcnow() - datetime.fromisoformat(uptime_start)
            uptime_s = int(delta.total_seconds())
        return {
            "running":       self._running,
            "uptime_seconds": uptime_s,
            "last_heartbeat": self.db.get_state("last_heartbeat"),
            "resources":     self.resources.get_stats(),
            "tasks":         [asdict(t) for t in self.db.get_tasks()],
            "recent_logs":   self.db.get_logs(limit=20),
        }

    def kill(self):
        """Immediate hard stop — kill switch."""
        log.warning("KILL SWITCH ACTIVATED")
        self.db.log_activity("KILL_SWITCH", "engine",
                              "Emergency stop activated", level="ERROR")
        self.stop()
        os._exit(0)
