"""
sovereignty/scheduler.py
========================
Decides when to run the nightly improvement pipeline.

The scheduler checks on every daemon tick whether the pipeline should run.
It respects the configured time window (IMPROVEMENT_START to IMPROVEMENT_END),
prevents double-runs, and supports a catch-up window for machines that were
off during the primary window.

Relationship to other modules:
    daemon.py calls should_run() on every tick.
    improver.py is invoked when should_run() returns True.
    run_log.json is the persistent record of completed runs.
"""

import json
import threading
from datetime import datetime, date, timedelta
from typing import Optional

from sovereignty.config import Config
from sovereignty.logger import get_logger

log = get_logger(__name__)


class Scheduler:
    """
    Nightly run scheduler with catch-up support.

    Attributes:
        _lock: Protects _run_log from concurrent reads and writes.
               (Daemon thread reads/writes; API thread may read for /status.)
    """

    def __init__(self, config: Config) -> None:
        """
        Parameters:
            config: Active Config instance.
        """
        self._config = config
        self._lock = threading.Lock()  # Protects _run_log dict
        self._run_log: dict = self._load_run_log()

    # ── Public interface ──────────────────────────────────────────────────

    def should_run(self, now: Optional[datetime] = None) -> bool:
        """
        Return True if the improvement pipeline should start right now.

        Checks:
          1. Has today's run already completed? → False
          2. Is 'now' inside the primary improvement window? → True
          3. Did we miss the window (system was off)?
             Is 'now' within the catch-up window after the window end? → True
          4. Otherwise → False

        Parameters:
            now: The current datetime (injectable for testing).

        Returns:
            True if the pipeline should start, False otherwise.
        """
        now = now or datetime.now()
        today_key = now.date().isoformat()

        with self._lock:
            if self._run_log.get(today_key, {}).get("completed"):
                return False

        start_dt = self._window_start(now.date())
        end_dt = self._window_end(now.date())
        catchup_end = end_dt + timedelta(hours=self._config.CATCHUP_WINDOW_HOURS)

        # Primary window
        if start_dt <= now < end_dt:
            return True

        # Catch-up window (system was off during primary window)
        if end_dt <= now < catchup_end:
            log.info("Within catch-up window — scheduling missed run")
            return True

        return False

    def mark_started(self, now: Optional[datetime] = None) -> None:
        """
        Record that today's run has started (prevents double-starts).

        Parameters:
            now: Injectable datetime for testing.
        """
        now = now or datetime.now()
        today_key = now.date().isoformat()
        with self._lock:
            self._run_log[today_key] = {
                "started": now.isoformat(),
                "completed": False,
                "result": None,
                "duration_sec": None,
            }
            self._save_run_log()
        log.info("Nightly run started — recorded in run_log")

    def mark_completed(
        self,
        result: str,
        duration_sec: float,
        now: Optional[datetime] = None,
    ) -> None:
        """
        Record that today's run has finished.

        Parameters:
            result: One of 'success', 'partial', 'failed'.
            duration_sec: How long the pipeline took.
            now: Injectable datetime for testing.
        """
        now = now or datetime.now()
        today_key = now.date().isoformat()
        with self._lock:
            entry = self._run_log.get(today_key, {})
            entry["completed"] = True
            entry["finished"] = now.isoformat()
            entry["result"] = result
            entry["duration_sec"] = round(duration_sec, 1)
            self._run_log[today_key] = entry
            self._save_run_log()
        log.info("Nightly run completed — result: %s, duration: %.1fs", result, duration_sec)

    def last_run_info(self) -> dict:
        """
        Return info about the most recent completed run.

        Returns:
            Dict with 'date', 'result', 'finished', 'duration_sec',
            or a dict indicating 'never' if no runs have completed.
        """
        with self._lock:
            completed = {
                k: v for k, v in self._run_log.items() if v.get("completed")
            }
        if not completed:
            return {"result": "never", "finished": None, "date": None, "duration_sec": None}
        latest_key = sorted(completed.keys())[-1]
        entry = completed[latest_key]
        return {
            "date": latest_key,
            "result": entry.get("result", "unknown"),
            "finished": entry.get("finished"),
            "duration_sec": entry.get("duration_sec"),
        }

    def next_run_dt(self, now: Optional[datetime] = None) -> datetime:
        """
        Return the datetime of the next scheduled pipeline run.

        Parameters:
            now: Injectable datetime for testing.

        Returns:
            datetime of next IMPROVEMENT_START (today if not yet passed,
            otherwise tomorrow).
        """
        now = now or datetime.now()
        today_start = self._window_start(now.date())
        today_key = now.date().isoformat()

        with self._lock:
            today_done = self._run_log.get(today_key, {}).get("completed", False)

        if now < today_start and not today_done:
            return today_start
        return self._window_start(now.date() + timedelta(days=1))

    # ── Private helpers ───────────────────────────────────────────────────

    def _window_start(self, d: date) -> datetime:
        """Parse IMPROVEMENT_START into a datetime on date d."""
        h, m = map(int, self._config.IMPROVEMENT_START.split(":"))
        return datetime(d.year, d.month, d.day, h, m, 0)

    def _window_end(self, d: date) -> datetime:
        """Parse IMPROVEMENT_END into a datetime on date d."""
        h, m = map(int, self._config.IMPROVEMENT_END.split(":"))
        return datetime(d.year, d.month, d.day, h, m, 0)

    def _load_run_log(self) -> dict:
        """
        Load run_log.json from disk.

        Returns:
            Dict of {date_str: run_entry}. Empty dict if file missing or corrupt.
        """
        path = self._config.RUN_LOG_FILE
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            log.warning("Could not load run_log.json: %s", exc)
            return {}

    def _save_run_log(self) -> None:
        """
        Write _run_log to run_log.json.
        Caller must hold _lock before calling this method.
        """
        try:
            self._config.RUN_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
            self._config.RUN_LOG_FILE.write_text(
                json.dumps(self._run_log, indent=2), encoding="utf-8"
            )
        except Exception as exc:
            log.error("Could not save run_log.json: %s", exc)
