"""
sovereignty/service.py
======================
Entry point for the Sovereignty daemon.

This module is the human-facing start/stop interface. It is NOT a Windows
Service (no pywin32 required, no SYSTEM-level install, no auto-start).

The daemon runs as the current user in the foreground. To keep it running
in the background, the user can use:
  - pythonw sovereignty/service.py start   (Windows, no console window)
  - nohup python -m sovereignty.service start &   (POSIX)
  - The Task Scheduler / cron entry created by install.py (optional, visible)

Human permission is required: the user must run this command explicitly.
There is no hidden install, no auto-boot hook.

Usage:
    python -m sovereignty.service start
    python -m sovereignty.service stop
    python -m sovereignty.service status

Relationship to other modules:
    daemon.py provides the Daemon class that manages all threads.
    config.py provides Config loaded from config.json.
    logger.py is initialised here before anything else.
    install.py creates the data directory and config.json.
"""

import json
import pathlib
import signal
import sys
import time
from datetime import datetime

# ── Bootstrap: ensure sovereignty package is importable ───────────────────────
_THIS_DIR = pathlib.Path(__file__).parent
if str(_THIS_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR.parent))

from sovereignty.config import Config
from sovereignty.logger import get_logger, setup_logging, shutdown_logging


def _load_config() -> Config:
    """Load config from disk if available, else use defaults."""
    return Config.from_json()


def cmd_start(config: Config) -> int:
    """
    Start the Sovereignty daemon.

    Blocks until the daemon is stopped (Ctrl+C, SIGTERM, or API /shutdown).

    Parameters:
        config: Active Config instance.

    Returns:
        Exit code: 0 on clean shutdown, 1 on error.
    """
    from sovereignty.daemon import Daemon

    setup_logging(config)
    log = get_logger(__name__)

    log.info("=" * 60)
    log.info("Sovereignty daemon starting")
    log.info("User: %s", _current_user())
    log.info("Data dir: %s", config.DATA_DIR)
    log.info("API: http://%s:%d", config.API_HOST, config.API_PORT)
    log.info("=" * 60)

    daemon = Daemon(config)

    # Handle Ctrl+C and SIGTERM gracefully
    def _handle_signal(sig, frame):
        log.info("Signal %d received — stopping daemon", sig)
        daemon.stop()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    try:
        daemon.start()
        daemon.wait_for_shutdown()
    except Exception as exc:
        log.critical("Daemon crashed: %s", exc, exc_info=True)
        return 1
    finally:
        shutdown_logging()

    return 0


def cmd_stop(config: Config) -> int:
    """
    Ask a running daemon to stop via the REST API /shutdown endpoint.

    Parameters:
        config: Active Config instance.

    Returns:
        Exit code: 0 if shutdown request sent, 1 if failed.
    """
    import urllib.request
    import urllib.error

    try:
        token = config.TOKEN_FILE.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        print("Token file not found — is the daemon installed?")
        return 1

    url = f"http://{config.API_HOST}:{config.API_PORT}/shutdown"
    body = json.dumps({"confirm": "yes"}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            print("Shutdown request accepted:", resp.read().decode())
        return 0
    except urllib.error.HTTPError as exc:
        print(f"Shutdown failed: HTTP {exc.code}")
        return 1
    except Exception as exc:
        print(f"Shutdown failed: {exc}")
        return 1


def cmd_status(config: Config) -> int:
    """
    Query the running daemon's status via the REST API.

    Parameters:
        config: Active Config instance.

    Returns:
        Exit code: 0 on success, 1 on failure.
    """
    import urllib.request
    import urllib.error

    try:
        token = config.TOKEN_FILE.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        print("Token file not found — daemon not installed.")
        return 1

    url = f"http://{config.API_HOST}:{config.API_PORT}/status"
    req = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {token}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
        print(json.dumps(data, indent=2))
        return 0
    except Exception as exc:
        print(f"Status check failed: {exc}")
        return 1


def _current_user() -> str:
    """Return the current OS username."""
    try:
        import os
        return os.environ.get("USERNAME") or os.environ.get("USER") or "unknown"
    except Exception:
        return "unknown"


def main() -> int:
    """
    CLI entry point.

    Usage: python -m sovereignty.service [start|stop|status]
    """
    command = sys.argv[1] if len(sys.argv) > 1 else "start"
    config = _load_config()

    if command == "start":
        return cmd_start(config)
    elif command == "stop":
        return cmd_stop(config)
    elif command == "status":
        return cmd_status(config)
    else:
        print(f"Unknown command: {command}")
        print("Usage: python -m sovereignty.service [start|stop|status]")
        return 1


if __name__ == "__main__":
    sys.exit(main())
