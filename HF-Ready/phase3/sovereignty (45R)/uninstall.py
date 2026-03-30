"""
sovereignty/uninstall.py
========================
Clean removal of the Sovereignty daemon and its data.

Asks the user whether to keep logs before deleting anything.
Removes the optional Task Scheduler entry (Windows).
Never leaves orphaned files.

Human permission required: the user runs this script explicitly.

Relationship to other modules:
    config.py provides the data directory path.
    service.py is contacted via API to stop a running daemon.
"""

import json
import pathlib
import platform
import shutil
import subprocess
import sys
from datetime import datetime

_THIS_DIR = pathlib.Path(__file__).parent
if str(_THIS_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR.parent))


def _print(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def _stop_daemon(config) -> None:
    """Ask a running daemon to stop via the API, best-effort."""
    import urllib.request
    import urllib.error

    try:
        token = config.TOKEN_FILE.read_text(encoding="utf-8").strip()
    except Exception:
        return  # Daemon not running or not installed

    url = f"http://{config.API_HOST}:{config.API_PORT}/shutdown"
    body = json.dumps({"confirm": "yes"}).encode("utf-8")
    req = urllib.request.Request(
        url, data=body,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5):
            _print("  ✓ Running daemon stopped via API")
        import time
        time.sleep(3)  # Give it a moment to shut down cleanly
    except Exception:
        _print("  → No running daemon found (or already stopped)")


def _remove_task_scheduler() -> None:
    """Remove the optional Task Scheduler entry on Windows."""
    if platform.system() != "Windows":
        return
    try:
        result = subprocess.run(
            ["schtasks", "/Delete", "/TN", "SovereigntyDaemon", "/F"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            _print("  ✓ Task Scheduler entry removed")
        else:
            _print("  → No Task Scheduler entry found")
    except Exception as exc:
        _print(f"  ⚠ Could not remove Task Scheduler entry: {exc}")


def main() -> int:
    """
    Interactive uninstall.

    Returns:
        0 on success, 1 on error.
    """
    from sovereignty.config import Config
    config = Config()

    _print("Sovereignty Package — Uninstaller")
    _print(f"Data directory: {config.DATA_DIR}")
    _print("")

    if not config.DATA_DIR.exists():
        _print("Data directory not found — nothing to remove.")
        return 0

    answer = input("Proceed with uninstall? [y/N] ").strip().lower()
    if answer != "y":
        _print("Aborted.")
        return 0

    keep_logs = input("Keep log files? [y/N] ").strip().lower() == "y"

    # Step 1: Stop running daemon
    _print("Stopping running daemon (if any)...")
    _stop_daemon(config)

    # Step 2: Remove Task Scheduler entry (Windows)
    _print("Removing Task Scheduler entry (if any)...")
    _remove_task_scheduler()

    # Step 3: Remove files
    _print("Removing data directory...")
    if keep_logs and config.LOG_DIR.exists():
        # Delete everything except the logs directory
        for item in config.DATA_DIR.iterdir():
            if item.resolve() == config.LOG_DIR.resolve():
                continue
            try:
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)
            except Exception as exc:
                _print(f"  ⚠ Could not remove {item}: {exc}")
        _print(f"  ✓ Data directory cleared (logs kept at {config.LOG_DIR})")
    else:
        try:
            shutil.rmtree(config.DATA_DIR)
            _print(f"  ✓ Data directory removed: {config.DATA_DIR}")
        except Exception as exc:
            _print(f"  ✗ Could not remove data directory: {exc}")
            return 1

    _print("")
    _print("=" * 50)
    _print("  ✓ SOVEREIGNTY UNINSTALLED COMPLETELY")
    if keep_logs:
        _print(f"  Logs retained at: {config.LOG_DIR}")
    _print("=" * 50)
    return 0


if __name__ == "__main__":
    sys.exit(main())
