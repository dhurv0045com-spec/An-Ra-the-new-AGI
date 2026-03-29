"""
sovereignty/install.py
======================
15-step atomic install with full rollback on failure.

Runs as the current user — no Administrator rights required.
All paths are user-writable. No Windows Service registration.
No auto-boot hook (optional Task Scheduler entry is transparent and visible).

Human permission: the user runs this script explicitly.

Steps:
  1:  Check Python version ≥ 3.10, psutil installed
  2:  Check data directory is writable
  3:  Create data directory
  4:  Create log subdirectory
  5:  Generate API bearer token → token.key
  6:  Write config.json with all defaults
  7:  Write benchmark_baseline.json (zeros — first run sets real baseline)
  8:  Write run_log.json (empty)
  9:  Write audit_baseline.json (empty)
 10:  Verify package imports (sovereignty.*)
 11:  Optional: create Task Scheduler entry (Windows only, asks user)
 12:  Write startup script (sovereignty_start.bat / .sh)
 13:  Test config can be loaded back from disk
 14:  Write INSTALL_COMPLETE marker
 15:  Print success summary

Rollback: if any step fails, all created files/dirs are removed.

Relationship to other modules:
    config.py provides Config (the data directory path, defaults).
    auth.py provides generate_token().
    logger.py is NOT yet available during install (logging to stdout).
"""

import json
import pathlib
import platform
import shutil
import subprocess
import sys
from datetime import datetime

# ── Bootstrap ─────────────────────────────────────────────────────────────────
_THIS_DIR = pathlib.Path(__file__).parent
if str(_THIS_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR.parent))


def _print(msg: str) -> None:
    """Print to stdout with a timestamp prefix."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def _fail(step: int, reason: str) -> None:
    """Print failure message and exit 1."""
    _print(f"✗ Step {step} FAILED: {reason}")
    _print("Rolling back...")


class Installer:
    """
    15-step atomic installer.

    Tracks created paths in self._created so rollback can remove them.
    """

    def __init__(self) -> None:
        self._created: list = []  # Paths created during install (for rollback)
        self._config = None

    def run(self) -> int:
        """
        Execute all 15 install steps.

        Returns:
            0 on success, 1 on failure.
        """
        step = 0
        try:
            step = 1; self._step1_check_python()
            step = 2; self._step2_check_writable()
            step = 3; self._step3_create_data_dir()
            step = 4; self._step4_create_log_dir()
            step = 5; self._step5_generate_token()
            step = 6; self._step6_write_config()
            step = 7; self._step7_write_bench_baseline()
            step = 8; self._step8_write_run_log()
            step = 9; self._step9_write_audit_baseline()
            step = 10; self._step10_verify_imports()
            step = 11; self._step11_optional_task_scheduler()
            step = 12; self._step12_write_startup_script()
            step = 13; self._step13_verify_config_reload()
            step = 14; self._step14_write_marker()
            step = 15; self._step15_print_summary()
            return 0
        except Exception as exc:
            _fail(step, str(exc))
            self._rollback()
            return 1

    # ── Steps ─────────────────────────────────────────────────────────────

    def _step1_check_python(self) -> None:
        _print("Step 1: Checking Python version and dependencies...")
        major, minor = sys.version_info[:2]
        if (major, minor) < (3, 10):
            raise RuntimeError(f"Python 3.10+ required; found {major}.{minor}")
        try:
            import psutil
        except ImportError:
            raise RuntimeError("psutil not installed. Run: pip install psutil")
        _print(f"  ✓ Python {major}.{minor}, psutil available")

    def _step2_check_writable(self) -> None:
        _print("Step 2: Checking data directory is writable...")
        from sovereignty.config import Config
        self._config = Config()
        parent = self._config.DATA_DIR.parent
        if not parent.exists():
            raise RuntimeError(f"Parent directory does not exist: {parent}")
        test = parent / ".sovereignty_write_test"
        try:
            test.write_text("test", encoding="utf-8")
            test.unlink()
        except Exception as exc:
            raise RuntimeError(f"Cannot write to {parent}: {exc}")
        _print(f"  ✓ {parent} is writable")

    def _step3_create_data_dir(self) -> None:
        _print(f"Step 3: Creating data directory: {self._config.DATA_DIR}")
        self._config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self._created.append(self._config.DATA_DIR)
        _print(f"  ✓ Created {self._config.DATA_DIR}")

    def _step4_create_log_dir(self) -> None:
        _print(f"Step 4: Creating log directory: {self._config.LOG_DIR}")
        self._config.LOG_DIR.mkdir(parents=True, exist_ok=True)
        self._created.append(self._config.LOG_DIR)
        _print(f"  ✓ Created {self._config.LOG_DIR}")

    def _step5_generate_token(self) -> None:
        _print("Step 5: Generating API bearer token...")
        from sovereignty.auth import generate_token
        token = generate_token(self._config)
        self._created.append(self._config.TOKEN_FILE)
        _print(f"  ✓ Token written to {self._config.TOKEN_FILE}")

    def _step6_write_config(self) -> None:
        _print("Step 6: Writing config.json...")
        config_data = self._config.to_json()
        self._config.CONFIG_FILE.write_text(
            json.dumps(config_data, indent=2), encoding="utf-8"
        )
        self._created.append(self._config.CONFIG_FILE)
        _print(f"  ✓ Config written to {self._config.CONFIG_FILE}")

    def _step7_write_bench_baseline(self) -> None:
        _print("Step 7: Writing benchmark baseline (zeros — first run sets real baseline)...")
        baseline = {
            f"B{i:02d}": {"median": 0.0, "unit": "ms"}
            for i in range(1, 11)
        }
        self._config.BENCHMARK_BASELINE_FILE.write_text(
            json.dumps(baseline, indent=2), encoding="utf-8"
        )
        self._created.append(self._config.BENCHMARK_BASELINE_FILE)
        _print(f"  ✓ {self._config.BENCHMARK_BASELINE_FILE}")

    def _step8_write_run_log(self) -> None:
        _print("Step 8: Writing empty run_log.json...")
        self._config.RUN_LOG_FILE.write_text("{}", encoding="utf-8")
        self._created.append(self._config.RUN_LOG_FILE)
        _print(f"  ✓ {self._config.RUN_LOG_FILE}")

    def _step9_write_audit_baseline(self) -> None:
        _print("Step 9: Writing empty audit_baseline.json...")
        self._config.AUDIT_BASELINE_FILE.write_text("{}", encoding="utf-8")
        self._created.append(self._config.AUDIT_BASELINE_FILE)
        _print(f"  ✓ {self._config.AUDIT_BASELINE_FILE}")

    def _step10_verify_imports(self) -> None:
        _print("Step 10: Verifying package imports...")
        modules = [
            "sovereignty.config", "sovereignty.logger", "sovereignty.auth",
            "sovereignty.scheduler", "sovereignty.watchdog", "sovereignty.api",
            "sovereignty.auditor", "sovereignty.dead_code", "sovereignty.benchmarks",
            "sovereignty.reporter", "sovereignty.improver", "sovereignty.daemon",
            "sovereignty.resource_monitor",
        ]
        for mod in modules:
            try:
                __import__(mod)
            except Exception as exc:
                raise RuntimeError(f"Cannot import {mod}: {exc}")
        _print(f"  ✓ All {len(modules)} modules import cleanly")

    def _step11_optional_task_scheduler(self) -> None:
        _print("Step 11: Optional — Task Scheduler entry (visible, user can delete)...")
        if platform.system() != "Windows":
            _print("  → Not Windows — skipping Task Scheduler setup")
            return

        answer = input("  Create a visible Task Scheduler entry for 2 AM nightly run? [y/N] ").strip().lower()
        if answer != "y":
            _print("  → Skipped by user")
            return

        script = self._config.DATA_DIR / "sovereignty_start.bat"
        python_exe = sys.executable
        task_xml = f"""<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <Triggers>
    <CalendarTrigger>
      <StartBoundary>2000-01-01T02:00:00</StartBoundary>
      <ScheduleByDay><DaysInterval>1</DaysInterval></ScheduleByDay>
    </CalendarTrigger>
  </Triggers>
  <Actions>
    <Exec>
      <Command>{python_exe}</Command>
      <Arguments>-m sovereignty.service start</Arguments>
      <WorkingDirectory>{_THIS_DIR.parent}</WorkingDirectory>
    </Exec>
  </Actions>
</Task>"""
        xml_path = self._config.DATA_DIR / "sovereignty_task.xml"
        xml_path.write_text(task_xml, encoding="utf-16")
        self._created.append(xml_path)

        try:
            result = subprocess.run(
                ["schtasks", "/Create", "/TN", "SovereigntyDaemon", "/XML", str(xml_path), "/F"],
                capture_output=True, text=True, timeout=15,
            )
            if result.returncode == 0:
                _print("  ✓ Task Scheduler entry created (visible in Task Scheduler app)")
            else:
                _print(f"  ⚠ Task Scheduler creation failed: {result.stderr.strip()}")
        except Exception as exc:
            _print(f"  ⚠ Could not create Task Scheduler entry: {exc}")

    def _step12_write_startup_script(self) -> None:
        _print("Step 12: Writing startup script...")
        python_exe = sys.executable

        if platform.system() == "Windows":
            script_path = self._config.DATA_DIR / "sovereignty_start.bat"
            script_path.write_text(
                f'@echo off\n"{python_exe}" -m sovereignty.service start\n',
                encoding="utf-8",
            )
        else:
            script_path = self._config.DATA_DIR / "sovereignty_start.sh"
            script_path.write_text(
                f'#!/bin/bash\nexec "{python_exe}" -m sovereignty.service start "$@"\n',
                encoding="utf-8",
            )
            try:
                import stat
                script_path.chmod(script_path.stat().st_mode | stat.S_IEXEC)
            except Exception:
                pass

        self._created.append(script_path)
        _print(f"  ✓ Startup script: {script_path}")

    def _step13_verify_config_reload(self) -> None:
        _print("Step 13: Verifying config round-trip...")
        from sovereignty.config import Config
        reloaded = Config.from_json(self._config.CONFIG_FILE)
        if reloaded.API_PORT != self._config.API_PORT:
            raise RuntimeError("Config round-trip failed — API_PORT mismatch")
        _print("  ✓ Config reloads cleanly from disk")

    def _step14_write_marker(self) -> None:
        _print("Step 14: Writing INSTALL_COMPLETE marker...")
        marker = self._config.DATA_DIR / "INSTALL_COMPLETE"
        marker.write_text(
            json.dumps({"installed": datetime.now().isoformat(), "python": sys.version}),
            encoding="utf-8",
        )
        self._created.append(marker)
        _print(f"  ✓ {marker}")

    def _step15_print_summary(self) -> None:
        token = self._config.TOKEN_FILE.read_text(encoding="utf-8").strip()
        python_exe = sys.executable
        api = f"http://{self._config.API_HOST}:{self._config.API_PORT}"

        _print("")
        _print("=" * 60)
        _print("  ✓ SOVEREIGNTY INSTALLED SUCCESSFULLY")
        _print("=" * 60)
        _print(f"  Data directory : {self._config.DATA_DIR}")
        _print(f"  Token file     : {self._config.TOKEN_FILE}")
        _print(f"  API URL        : {api}")
        _print(f"  Next run       : tonight at {self._config.IMPROVEMENT_START}")
        _print("")
        _print("  HOW TO START:")
        _print(f"    {python_exe} -m sovereignty.service start")
        _print("")
        _print("  HOW TO CHECK STATUS:")
        _print(f'    curl -H "Authorization: Bearer {token}" {api}/status')
        _print("")
        _print("  HOW TO UNINSTALL:")
        _print(f"    {python_exe} -m sovereignty.uninstall")
        _print("=" * 60)

    # ── Rollback ──────────────────────────────────────────────────────────

    def _rollback(self) -> None:
        """Remove all files and directories created during the failed install."""
        removed = 0
        for path in reversed(self._created):
            try:
                p = pathlib.Path(path)
                if p.is_file():
                    p.unlink()
                    removed += 1
                elif p.is_dir():
                    shutil.rmtree(p, ignore_errors=True)
                    removed += 1
            except Exception as exc:
                _print(f"  ⚠ Rollback: could not remove {path}: {exc}")
        _print(f"Rollback complete — removed {removed} items")


def main() -> int:
    """Entry point for python -m sovereignty.install"""
    _print("Sovereignty Package — Atomic Installer")
    _print("User-level install. No admin rights required.")
    _print("")
    return Installer().run()


if __name__ == "__main__":
    sys.exit(main())
