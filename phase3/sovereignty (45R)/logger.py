"""
sovereignty/logger.py
=====================
Thread-safe rotating log system for the Sovereignty daemon.

All worker threads put log records onto a queue. A single dedicated logger
thread drains the queue and writes to disk — this prevents file-lock contention
and ensures ordering.

Log files rotate at midnight: the current day's file is compressed to .gz and
a new file is opened. Files older than LOG_RETENTION_DAYS are auto-deleted.

Format (every line):
    [YYYY-MM-DD HH:MM:SS.mmm] [LEVEL    ] [ThreadName     ] [module.func] message

Relationship to other modules:
    Every module calls get_logger(__name__) to obtain a named logger.
    install.py calls setup_logging() before any other module is imported.
"""

import gzip
import logging
import logging.handlers
import os
import pathlib
import queue
import shutil
import sys
import threading
import time
from datetime import datetime, timedelta
from typing import Optional

from sovereignty.config import Config
from shared_logger import emit_audit_event, get_shared_logger

# ── Module-level state ────────────────────────────────────────────────────────
_log_queue: queue.Queue = queue.Queue(maxsize=10_000)
_logger_thread: Optional[threading.Thread] = None
_stop_event: threading.Event = threading.Event()
_setup_done: bool = False
_setup_lock: threading.Lock = threading.Lock()  # Protects _setup_done flag


class _QueueHandler(logging.Handler):
    """
    Logging handler that puts records onto the shared queue.
    Worker threads use this — they never write to disk directly.
    """

    def __init__(self, log_queue: queue.Queue) -> None:
        super().__init__()
        self._queue = log_queue

    def emit(self, record: logging.LogRecord) -> None:
        """Put a formatted log record onto the queue without blocking."""
        try:
            self._queue.put_nowait(record)
        except queue.Full:
            # Queue full means the logger thread is behind — drop the record
            # rather than block a worker thread.
            pass


class _FileHandler(logging.FileHandler):
    """
    Daily rotating file handler that compresses and prunes old logs.
    Only the logger thread uses this.
    """

    def __init__(self, log_dir: pathlib.Path, retention_days: int) -> None:
        """
        Parameters:
            log_dir: Directory to write log files into.
            retention_days: Delete compressed logs older than this.
        """
        self._log_dir = log_dir
        self._retention_days = retention_days
        self._current_date = datetime.now().date()
        log_dir.mkdir(parents=True, exist_ok=True)
        path = self._path_for_date(self._current_date)
        super().__init__(str(path), mode="a", encoding="utf-8", delay=False)

    def _path_for_date(self, date) -> pathlib.Path:
        """Return the log file path for a given date."""
        return self._log_dir / f"service_{date.strftime('%Y%m%d')}.log"

    def emit(self, record: logging.LogRecord) -> None:
        """Rotate if date has changed, then write the record."""
        today = datetime.now().date()
        if today != self._current_date:
            self._rotate(today)
        super().emit(record)

    def _rotate(self, new_date) -> None:
        """
        Close the current log file, compress it to .gz, open a new file,
        and prune logs older than retention_days.
        """
        old_path = self._path_for_date(self._current_date)
        self.close()
        # Compress the old file
        gz_path = pathlib.Path(str(old_path) + ".gz")
        try:
            with open(old_path, "rb") as f_in, gzip.open(gz_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
            old_path.unlink()
        except Exception:
            pass  # Best-effort compression; original file stays if it fails

        # Open the new file
        self._current_date = new_date
        new_path = self._path_for_date(new_date)
        self.baseFilename = str(new_path)
        self.stream = self._open()

        # Prune old logs
        cutoff = datetime.now() - timedelta(days=self._retention_days)
        for gz in self._log_dir.glob("service_*.log.gz"):
            try:
                date_str = gz.stem.replace("service_", "").replace(".log", "")
                file_date = datetime.strptime(date_str, "%Y%m%d")
                if file_date < cutoff:
                    gz.unlink()
            except Exception:
                pass


def _logger_thread_main(
    log_dir: pathlib.Path,
    retention_days: int,
    stop_event: threading.Event,
) -> None:
    """
    Logger thread entry point.

    Drains _log_queue and writes records to disk. Runs until stop_event is set
    and the queue is empty.

    Parameters:
        log_dir: Directory for log files.
        retention_days: Passed to _FileHandler for pruning.
        stop_event: Signal to shut down after draining the queue.
    """
    formatter = logging.Formatter(
        fmt="[%(asctime)s.%(msecs)03d] [%(levelname)-8s] [%(threadName)-20s] [%(name)s.%(funcName)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = _FileHandler(log_dir, retention_days)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    while not stop_event.is_set() or not _log_queue.empty():
        try:
            record = _log_queue.get(timeout=0.5)
            file_handler.emit(record)
            console_handler.emit(record)
            _log_queue.task_done()
        except queue.Empty:
            continue
        except Exception:
            pass  # Never let the logger thread crash

    file_handler.close()


def setup_logging(config: Config) -> None:
    """
    Initialise the logging subsystem. Must be called once before get_logger().

    Creates the log directory, starts the logger thread, and installs the
    queue handler on the root logger.

    Parameters:
        config: Active Config instance (provides LOG_DIR, LOG_RETENTION_DAYS,
                LOG_LEVEL).

    Raises:
        RuntimeError: If called a second time (safe to catch and ignore).
    """
    global _logger_thread, _setup_done

    with _setup_lock:
        if _setup_done:
            return

        config.LOG_DIR.mkdir(parents=True, exist_ok=True)

        # Root logger: accepts all levels; handlers filter
        root = logging.getLogger()
        root.setLevel(logging.DEBUG)

        # Remove any existing handlers (e.g. from basicConfig)
        for h in root.handlers[:]:
            root.removeHandler(h)

        queue_handler = _QueueHandler(_log_queue)
        queue_handler.setLevel(getattr(logging, config.LOG_LEVEL, logging.INFO))
        root.addHandler(queue_handler)

        _stop_event.clear()
        _logger_thread = threading.Thread(
            target=_logger_thread_main,
            args=(config.LOG_DIR, config.LOG_RETENTION_DAYS, _stop_event),
            name="LoggerThread",
            daemon=True,
        )
        _logger_thread.start()
        _setup_done = True


def shutdown_logging() -> None:
    """
    Signal the logger thread to flush and stop.

    Waits up to 5 seconds for the queue to drain before returning.
    Call during service shutdown after all worker threads have stopped.
    """
    _stop_event.set()
    if _logger_thread and _logger_thread.is_alive():
        _logger_thread.join(timeout=5.0)


def get_logger(name: str) -> logging.Logger:
    """Return shared logger instance routed through this module's handlers."""
    return get_shared_logger(name)


def audit_event(name: str, event_type: str, action: str, message: str = "", details: Optional[dict] = None) -> dict:
    """Emit append-only AUDIT_LOG events using required envelope fields."""
    logger = get_shared_logger(name)
    return emit_audit_event(logger, event_type=event_type, component=name, action=action, message=message, details=details)


def get_recent_lines(log_dir: pathlib.Path, n: int = 100, level: str = "DEBUG") -> list:
    """
    Return the last N log lines from today's log file, optionally filtered by level.

    Parameters:
        log_dir: Directory containing log files.
        n: Maximum number of lines to return.
        level: Minimum level string (DEBUG/INFO/WARNING/ERROR/CRITICAL).

    Returns:
        List of log line strings, most recent last.
    """
    today = datetime.now().date()
    log_file = log_dir / f"service_{today.strftime('%Y%m%d')}.log"
    if not log_file.exists():
        return []

    level_order = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    min_idx = level_order.index(level.upper()) if level.upper() in level_order else 0

    lines = []
    try:
        with open(log_file, encoding="utf-8", errors="replace") as f:
            all_lines = f.readlines()
        for line in all_lines:
            for i, lv in enumerate(level_order):
                if f"[{lv}" in line and i >= min_idx:
                    lines.append(line.rstrip())
                    break
    except Exception:
        pass

    return lines[-n:]
