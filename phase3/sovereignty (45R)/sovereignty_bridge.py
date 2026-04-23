"""
sovereignty_bridge.py — Phase 3 | Component 45R
Sovereignty Daemon Bridge for An-Ra MasterSystem
=================================================

Integrates the 45R Sovereignty self-improvement daemon into the MasterSystem
lifecycle. The daemon audits the An-Ra codebase nightly and produces measurable
improvement metrics.

What the daemon does every night (02:00–05:00):
  Pass 1: Code Audit     — cyclomatic complexity, docstring coverage
  Pass 2: Dead Code Sweep — unused imports, unreachable code, magic numbers
  Pass 3: Benchmarks     — 10 performance benchmarks vs baseline
  Pass 4: Nightly Report — plain-English report with delta arrows

This bridge:
  - Starts the daemon in a background thread when MasterSystem.start() is called
  - Queries the daemon's REST API (localhost:45000) for status and reports
  - Exposes the nightly report to the morning briefing system
  - Gracefully handles psutil not being installed (feature disabled silently)

Usage (via MasterSystem):
    self.sovereignty = SovereigntyBridge(target_path=PROJECT_ROOT)
    self.sovereignty.start()
    report = self.sovereignty.get_nightly_report()
    status = self.sovereignty.status()
"""

import sys
import json
import time
import threading
import subprocess
from pathlib import Path
from typing import Optional


class SovereigntyBridge:
    """
    Bridge between MasterSystem and the 45R Sovereignty Daemon.

    The daemon is a separate process (not a thread) because it does
    disk I/O, AST analysis, and benchmarks — we don't want it blocking
    the main event loop.
    """

    DEFAULT_PORT    = 45000
    DEFAULT_HOST    = "127.0.0.1"
    STARTUP_WAIT_S  = 3.0   # Seconds to wait for daemon to initialize

    def __init__(
        self,
        target_path: Optional[Path] = None,
        port: int = DEFAULT_PORT,
        data_dir: Optional[Path] = None,
        enabled: bool = True,
    ):
        """
        Args:
            target_path: Path to the codebase the daemon should audit.
                         Defaults to the An-Ra project root.
            port:        Local port for the daemon REST API.
            data_dir:    Where the daemon stores its data/reports.
            enabled:     If False, all methods are no-ops (psutil missing).
        """
        self._enabled    = enabled
        self._port       = port
        self._host       = self.DEFAULT_HOST
        self._token: Optional[str] = None
        self._daemon_process: Optional[subprocess.Popen] = None
        self._started    = False
        self._available  = False   # becomes True after daemon responds to /ping

        # Paths
        if target_path is None:
            target_path = Path(__file__).resolve().parent.parent.parent  # An-Ra root
        self._target_path = target_path

        if data_dir is None:
            data_dir = Path(__file__).parent / "sovereignty_data"
        self._data_dir = data_dir
        self._data_dir.mkdir(parents=True, exist_ok=True)

        # Sovereignty package path
        self._sovereignty_pkg = Path(__file__).parent

        # Check dependencies silently
        if not self._check_deps():
            self._enabled = False

    def _check_deps(self) -> bool:
        """Check if psutil and the sovereignty package are available."""
        try:
            import psutil  # noqa: F401
        except ImportError:
            print("  [Phase 3] [SKIP] Sovereignty daemon skipped — "
                  "psutil not installed (pip install psutil)")
            return False
        if not (self._sovereignty_pkg / "service.py").is_file():
            print("  [Phase 3] [SKIP] Sovereignty daemon skipped — "
                  "sovereignty/service.py not found")
            return False
        return True

    def start(self) -> bool:
        """
        Start the sovereignty daemon as a background subprocess.

        Returns True if daemon started successfully, False otherwise.
        """
        if not self._enabled or self._started:
            return self._started

        try:
            import os
            env = os.environ.copy()
            env["SOVEREIGNTY_DATA"] = str(self._data_dir)
            env["SOVEREIGNTY_TARGET"] = str(self._target_path)

            # Start daemon process
            self._daemon_process = subprocess.Popen(
                [sys.executable, "-m", "sovereignty.service", "start"],
                cwd=str(self._sovereignty_pkg),
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            # Wait briefly and check liveness
            time.sleep(self.STARTUP_WAIT_S)
            if self._ping():
                self._available = True
                self._load_token()
                self._started = True
                print(f"  [Phase 3] [OK] Sovereignty daemon running "
                      f"(port {self._port}, auditing {self._target_path.name})")
            else:
                print(f"  [Phase 3] [WARN] Sovereignty daemon started but not responding. "
                      f"Will retry on next status check.")
                self._started = True  # Mark started so we don't try again

            return self._started

        except Exception as e:
            print(f"  [Phase 3] [FAIL] Sovereignty daemon failed to start: {e}")
            return False

    def stop(self):
        """Stop the daemon gracefully."""
        if self._daemon_process:
            try:
                self._daemon_process.terminate()
                self._daemon_process.wait(timeout=5)
            except Exception:
                pass
            self._daemon_process = None
        self._started = False
        self._available = False

    # ── REST API helpers ──────────────────────────────────────────────────────

    def _api_url(self, endpoint: str) -> str:
        return f"http://{self._host}:{self._port}/{endpoint.lstrip('/')}"

    def _ping(self) -> bool:
        """Check if daemon is alive (no auth required)."""
        try:
            import urllib.request
            with urllib.request.urlopen(self._api_url("ping"), timeout=2) as resp:
                data = json.loads(resp.read())
                return data.get("status") == "alive"
        except Exception:
            return False

    def _load_token(self):
        """Load the bearer token from the data directory."""
        token_file = self._data_dir / "token.key"
        if token_file.is_file():
            self._token = token_file.read_text().strip()

    def _api_get(self, endpoint: str, params: str = "") -> Optional[dict]:
        """Authenticated GET request to daemon API."""
        if not self._token:
            self._load_token()
        try:
            import urllib.request
            url = self._api_url(endpoint)
            if params:
                url = f"{url}?{params}"
            req = urllib.request.Request(
                url,
                headers={"Authorization": f"Bearer {self._token}"},
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                return json.loads(resp.read())
        except Exception:
            return None

    # ── Public interface ──────────────────────────────────────────────────────

    def get_daemon_status(self) -> dict:
        """Query daemon for full status (threads, uptime, last run)."""
        if not self._enabled:
            return {"available": False, "reason": "disabled"}
        if not self._available and not self._ping():
            return {"available": False, "reason": "not_responding"}
        self._available = True
        data = self._api_get("status")
        if data:
            data["available"] = True
            return data
        return {"available": False, "reason": "api_error"}

    def get_nightly_report(self, date: Optional[str] = None) -> str:
        """
        Retrieve the nightly improvement report.

        Args:
            date: YYYY-MM-DD string. If None, returns today's report.

        Returns:
            Report text, or a status message if not yet available.
        """
        if not self._enabled:
            return "Sovereignty daemon not enabled (psutil missing)."

        # Try API first
        if self._available:
            params = f"date={date}" if date else ""
            data = self._api_get("report", params)
            if data and "report" in data:
                return data["report"]

        # Fall back to reading file directly
        from datetime import date as dt
        d = date or dt.today().strftime("%Y%m%d")
        report_file = self._data_dir / f"nightly_report_{d}.txt"
        if report_file.is_file():
            return report_file.read_text()

        # Check if any report exists
        reports = sorted(self._data_dir.glob("nightly_report_*.txt"))
        if reports:
            return reports[-1].read_text()

        return ("No sovereignty report available yet. "
                "The daemon runs its first audit between 02:00–05:00. "
                "You can trigger it now with: python anra.py --sovereignty-run")

    def get_benchmark_summary(self) -> str:
        """Get a compact performance benchmark summary."""
        if not self._enabled:
            return ""
        from datetime import date as dt
        today = dt.today().strftime("%Y%m%d")
        bench_file = self._data_dir / f"benchmark_{today}.json"
        if bench_file.is_file():
            try:
                data = json.loads(bench_file.read_text())
                results = data.get("results", [])
                regressions = data.get("regressions", 0)
                improvements = data.get("improvements", 0)
                return (
                    f"Benchmarks: {len(results)} run | "
                    f"↑ {improvements} improved | "
                    f"↓ {regressions} regressed"
                )
            except Exception:
                pass
        return "No benchmark data for today."

    def trigger_pipeline(self) -> bool:
        """Manually trigger the full improvement pipeline right now."""
        if not self._enabled or not self._available:
            return False
        try:
            import urllib.request
            req = urllib.request.Request(
                self._api_url("task"),
                data=json.dumps({"task": "run_full_pipeline"}).encode(),
                headers={
                    "Authorization": f"Bearer {self._token}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status == 200
        except Exception:
            return False

    def status(self) -> dict:
        """Return bridge status for system dashboard."""
        base = {
            "enabled":   self._enabled,
            "started":   self._started,
            "available": self._available,
            "port":      self._port,
            "data_dir":  str(self._data_dir),
            "target":    str(self._target_path),
        }
        if self._available:
            daemon_status = self.get_daemon_status()
            base["daemon"] = daemon_status
        return base


def health_check() -> dict:
    try:
        bridge = SovereigntyBridge(enabled=True)
        status = bridge.status()
        return {"status": "ok" if status.get("enabled") else "degraded", **status}
    except Exception as exc:
        return {"status": "degraded", "detail": str(exc)}
