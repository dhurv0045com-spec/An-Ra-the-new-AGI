"""
sovereignty/api.py
==================
Local REST API server for the Sovereignty daemon.

Pure stdlib http.server — no Flask, FastAPI, or external dependencies.
Binds to 127.0.0.1:45000 only — never exposed externally.
All requests are authenticated via Bearer token (except GET /ping).
All requests are logged with: timestamp, method, endpoint, status, auth result.

Endpoints:
  GET  /ping          — liveness check (no auth)
  GET  /status        — full daemon status (auth required)
  GET  /log           — recent log lines (auth required)
  GET  /report        — nightly report for a date (auth required)
  POST /task          — queue a manual task (auth required)
  GET  /task/{id}     — check task status (auth required)
  POST /shutdown      — graceful shutdown (auth required)

Relationship to other modules:
    daemon.py starts the API thread and passes the shared state object.
    auth.py   validates Bearer tokens.
    logger.py logs all requests.
    watchdog.py pings /ping to verify liveness.
"""

import json
import queue
import threading
import time
import traceback
import uuid
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Callable, Dict, Optional
from urllib.parse import parse_qs, urlparse

from sovereignty.auth import extract_bearer, validate_token
from sovereignty.config import Config
from sovereignty.logger import get_logger, get_recent_lines

log = get_logger(__name__)


class SharedState:
    """
    Thread-safe container for data the API handler reads from the daemon.

    Attributes:
        _lock: Protects all mutable fields below.
                The API thread reads; the daemon/watchdog threads write.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()  # Protects all state fields
        self._start_time = time.monotonic()
        self._last_run_info: Dict = {"result": "never", "finished": None}
        self._next_run: Optional[str] = None
        self._thread_status: Dict[str, str] = {}
        self._resource: Dict[str, float] = {"cpu_pct": 0.0, "ram_mb": 0.0}
        self._task_queue: queue.Queue = queue.Queue()
        self._tasks: Dict[str, Dict] = {}  # task_id → {status, task, created}
        self._shutdown_requested = False

    def update(self, **kwargs: Any) -> None:
        """Update one or more state fields atomically."""
        with self._lock:
            for key, value in kwargs.items():
                if hasattr(self, f"_{key}"):
                    setattr(self, f"_{key}", value)

    def snapshot(self) -> Dict:
        """Return a consistent snapshot of all readable state fields."""
        with self._lock:
            return {
                "uptime_seconds": round(time.monotonic() - self._start_time),
                "last_run_info": dict(self._last_run_info),
                "next_run": self._next_run,
                "thread_status": dict(self._thread_status),
                "resource": dict(self._resource),
            }

    def enqueue_task(self, task_name: str) -> str:
        """
        Add a task to the execution queue.

        Parameters:
            task_name: One of 'run_pass_1', 'run_full_pipeline', 'gc', 'rotate_logs'.

        Returns:
            A UUID string for tracking the task.
        """
        task_id = str(uuid.uuid4())
        entry = {
            "task_id": task_id,
            "task": task_name,
            "status": "pending",
            "created": datetime.now().isoformat(),
        }
        with self._lock:
            self._tasks[task_id] = entry
        self._task_queue.put(entry)
        return task_id

    def get_task(self, task_id: str) -> Optional[Dict]:
        """Return the status dict for a task, or None if not found."""
        with self._lock:
            return dict(self._tasks[task_id]) if task_id in self._tasks else None

    def update_task(self, task_id: str, status: str) -> None:
        """Update the status field of a tracked task."""
        with self._lock:
            if task_id in self._tasks:
                self._tasks[task_id]["status"] = status

    @property
    def shutdown_requested(self) -> bool:
        """True if a /shutdown request has been processed."""
        with self._lock:
            return self._shutdown_requested

    def request_shutdown(self) -> None:
        """Set the shutdown flag (read by daemon.py)."""
        with self._lock:
            self._shutdown_requested = True


class _Handler(BaseHTTPRequestHandler):
    """
    HTTP request handler for the Sovereignty API.

    self.server.state (SharedState) and self.server.config (Config)
    are injected by _SovereigntyHTTPServer.
    """

    def log_message(self, format_str, *args) -> None:  # noqa: A002
        """Suppress default BaseHTTPRequestHandler console logging."""
        pass  # All logging goes through our own logger

    def _send_json(self, status: int, body: Any) -> None:
        """Serialise body to JSON and send with appropriate headers."""
        encoded = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def _send_text(self, status: int, body: str) -> None:
        """Send plain text response."""
        encoded = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _auth(self) -> bool:
        """
        Validate the Bearer token in the Authorization header.

        Returns:
            True if valid, False if missing or invalid.
        """
        header = self.headers.get("Authorization", "")
        token = extract_bearer(header)
        return validate_token(token, self.server.config)

    def _require_auth(self) -> bool:
        """Send 401 and return False if not authenticated."""
        if not self._auth():
            self.send_response(401)
            self.end_headers()
            return False
        return True

    def _log_request(self, status: int, auth_ok: Optional[bool], elapsed_ms: float) -> None:
        """Log the request with full context."""
        auth_str = "ok" if auth_ok else ("anon" if auth_ok is None else "fail")
        log.info(
            "API %s %s → %d  auth=%s  %.1fms",
            self.command, self.path, status, auth_str, elapsed_ms,
        )

    def _read_body(self) -> dict:
        """Read and parse the JSON request body."""
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        try:
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except Exception:
            return {}

    def do_GET(self) -> None:  # noqa: N802
        """Route GET requests to the appropriate handler."""
        t0 = time.perf_counter()
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        params = parse_qs(parsed.query)

        try:
            if path == "/ping":
                state = self.server.state.snapshot()
                self._send_json(200, {"status": "alive", "uptime_seconds": state["uptime_seconds"]})
                self._log_request(200, None, (time.perf_counter() - t0) * 1000)

            elif path == "/status":
                if not self._require_auth():
                    self._log_request(401, False, (time.perf_counter() - t0) * 1000)
                    return
                state = self.server.state.snapshot()
                last = state["last_run_info"]
                body = {
                    "service": "running",
                    "uptime_seconds": state["uptime_seconds"],
                    "last_improvement_run": last.get("finished"),
                    "last_improvement_result": last.get("result", "never"),
                    "next_scheduled_run": state["next_run"],
                    "threads": state["thread_status"],
                    "resource": state["resource"],
                }
                self._send_json(200, body)
                self._log_request(200, True, (time.perf_counter() - t0) * 1000)

            elif path == "/log":
                if not self._require_auth():
                    self._log_request(401, False, (time.perf_counter() - t0) * 1000)
                    return
                n = int(params.get("lines", ["100"])[0])
                level = params.get("level", ["DEBUG"])[0].upper()
                lines = get_recent_lines(self.server.config.LOG_DIR, n=n, level=level)
                self._send_json(200, {"lines": lines, "count": len(lines)})
                self._log_request(200, True, (time.perf_counter() - t0) * 1000)

            elif path == "/report":
                if not self._require_auth():
                    self._log_request(401, False, (time.perf_counter() - t0) * 1000)
                    return
                date_str = params.get("date", [datetime.now().strftime("%Y%m%d")])[0].replace("-", "")
                report_path = self.server.config.DATA_DIR / f"nightly_report_{date_str}.txt"
                if not report_path.exists():
                    self._send_json(404, {"error": f"No report for {date_str}"})
                    self._log_request(404, True, (time.perf_counter() - t0) * 1000)
                    return
                text = report_path.read_text(encoding="utf-8")
                self._send_text(200, text)
                self._log_request(200, True, (time.perf_counter() - t0) * 1000)

            elif path.startswith("/task/"):
                if not self._require_auth():
                    self._log_request(401, False, (time.perf_counter() - t0) * 1000)
                    return
                task_id = path[len("/task/"):]
                task = self.server.state.get_task(task_id)
                if task is None:
                    self._send_json(404, {"error": "Task not found"})
                    self._log_request(404, True, (time.perf_counter() - t0) * 1000)
                    return
                self._send_json(200, task)
                self._log_request(200, True, (time.perf_counter() - t0) * 1000)

            else:
                self._send_json(404, {"error": "Not found"})
                self._log_request(404, None, (time.perf_counter() - t0) * 1000)

        except Exception as exc:
            log.error("API handler error: %s", exc, exc_info=True)
            self._send_json(500, {"error": "Internal server error"})

    def do_POST(self) -> None:  # noqa: N802
        """Route POST requests to the appropriate handler."""
        t0 = time.perf_counter()
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")

        try:
            if path == "/task":
                if not self._require_auth():
                    self._log_request(401, False, (time.perf_counter() - t0) * 1000)
                    return
                body = self._read_body()
                valid_tasks = {"run_pass_1", "run_full_pipeline", "gc", "rotate_logs"}
                task_name = body.get("task", "")
                if task_name not in valid_tasks:
                    self._send_json(400, {"error": f"Unknown task. Valid: {sorted(valid_tasks)}"})
                    self._log_request(400, True, (time.perf_counter() - t0) * 1000)
                    return
                task_id = self.server.state.enqueue_task(task_name)
                self._send_json(200, {"queued": True, "task_id": task_id})
                self._log_request(200, True, (time.perf_counter() - t0) * 1000)

            elif path == "/shutdown":
                if not self._require_auth():
                    self._log_request(401, False, (time.perf_counter() - t0) * 1000)
                    return
                body = self._read_body()
                if body.get("confirm") != "yes":
                    self._send_json(400, {"error": "Body must include confirm=yes"})
                    self._log_request(400, True, (time.perf_counter() - t0) * 1000)
                    return
                self.server.state.request_shutdown()
                self._send_json(200, {"shutting_down": True})
                self._log_request(200, True, (time.perf_counter() - t0) * 1000)
                log.info("Shutdown requested via API")

            else:
                self._send_json(404, {"error": "Not found"})
                self._log_request(404, None, (time.perf_counter() - t0) * 1000)

        except Exception as exc:
            log.error("API POST handler error: %s", exc, exc_info=True)
            self._send_json(500, {"error": "Internal server error"})


class _SovereigntyHTTPServer(HTTPServer):
    """HTTPServer subclass that carries config and shared state."""

    def __init__(self, config: Config, state: SharedState, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.state = state


class APIServer:
    """
    Wraps the HTTP server in a thread-friendly interface.
    """

    def __init__(self, config: Config, state: SharedState) -> None:
        """
        Parameters:
            config: Active Config instance (provides API_HOST, API_PORT).
            state: Shared daemon state.
        """
        self._config = config
        self._state = state
        self._server: Optional[_SovereigntyHTTPServer] = None
        self._stop_event = threading.Event()

    def run_forever(self) -> None:
        """Entry point for the API server thread."""
        try:
            self._server = _SovereigntyHTTPServer(
                self._config,
                self._state,
                (self._config.API_HOST, self._config.API_PORT),
                _Handler,
            )
            self._server.timeout = 1.0  # Allow checking stop_event every second
            log.info("API server listening on %s:%d", self._config.API_HOST, self._config.API_PORT)
            while not self._stop_event.is_set():
                self._server.handle_request()
        except Exception as exc:
            log.error("API server error: %s", exc, exc_info=True)
        finally:
            if self._server:
                self._server.server_close()
            log.info("API server stopped")

    def stop(self) -> None:
        """Signal the API server thread to stop."""
        self._stop_event.set()
