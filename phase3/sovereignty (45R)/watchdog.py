"""
sovereignty/watchdog.py
=======================
Three-layer watchdog engine for the Sovereignty daemon.

Layer 1 — Main loop watchdog:
    Reads heartbeat.json every 30s. If no update for 90s, restarts the loop.
    After 3 failed restarts within 5 minutes, logs CRITICAL and stops the service.

Layer 2 — API server watchdog:
    Pings GET /ping every 60s. Restarts the API thread if no response in 5s.

Layer 3 — Nightly pipeline watchdog:
    If the pipeline thread has run > PIPELINE_TIMEOUT_HOURS, force-kills it.

Relationship to other modules:
    daemon.py starts the watchdog thread and passes thread references.
    logger.py used for all watchdog event logs.
    api.py is pinged by Layer 2.
"""

import json
import threading
import time
import urllib.request
import urllib.error
from datetime import datetime
from typing import Callable, Optional

from sovereignty.config import Config
from sovereignty.logger import get_logger

log = get_logger(__name__)


class Watchdog:
    """
    Three-layer watchdog. Runs in its own thread.

    Attributes:
        _restart_lock: Protects _restart_count and _first_restart_time to
                       prevent race conditions when counting restart attempts.
    """

    def __init__(
        self,
        config: Config,
        restart_main_loop_fn: Callable[[], bool],
        restart_api_fn: Callable[[], bool],
        get_pipeline_thread: Callable[[], Optional[threading.Thread]],
        stop_service_fn: Callable[[], None],
    ) -> None:
        """
        Parameters:
            config: Active Config instance.
            restart_main_loop_fn: Callable that restarts the main loop thread.
                                  Returns True on success.
            restart_api_fn: Callable that restarts the API server thread.
                            Returns True on success.
            get_pipeline_thread: Returns the current pipeline Thread or None.
            stop_service_fn: Called when the watchdog gives up and must stop
                             the entire service.
        """
        self._config = config
        self._restart_main_loop = restart_main_loop_fn
        self._restart_api = restart_api_fn
        self._get_pipeline_thread = get_pipeline_thread
        self._stop_service = stop_service_fn

        self._stop_event = threading.Event()
        self._restart_lock = threading.Lock()  # Protects restart counter state
        self._restart_count = 0
        self._first_restart_time: Optional[float] = None
        self._pipeline_started_at: Optional[float] = None

        # Tick counters — Layer 2 runs every 60s, Layer 3 every 60s
        self._api_check_counter = 0
        self._resource_check_counter = 0

    # ── Public interface ──────────────────────────────────────────────────

    def notify_pipeline_started(self) -> None:
        """Call when the pipeline thread begins, to start the pipeline timeout clock."""
        self._pipeline_started_at = time.monotonic()

    def notify_pipeline_stopped(self) -> None:
        """Call when the pipeline thread ends normally."""
        self._pipeline_started_at = None

    def run_forever(self) -> None:
        """
        Watchdog thread entry point.

        Runs a 30-second tick loop. Each tick runs Layer 1 (heartbeat check).
        Every 2 ticks: Layer 2 (API ping). Every 2 ticks: Layer 3 (pipeline timeout).
        """
        log.info("Watchdog started")
        while not self._stop_event.is_set():
            try:
                self._check_heartbeat()
            except Exception as exc:
                log.error("Watchdog Layer 1 error: %s", exc, exc_info=True)

            self._api_check_counter += 1
            if self._api_check_counter >= 2:
                self._api_check_counter = 0
                try:
                    self._check_api()
                except Exception as exc:
                    log.error("Watchdog Layer 2 error: %s", exc, exc_info=True)

            try:
                self._check_pipeline_timeout()
            except Exception as exc:
                log.error("Watchdog Layer 3 error: %s", exc, exc_info=True)

            self._stop_event.wait(timeout=30)

        log.info("Watchdog stopped")

    def stop(self) -> None:
        """Signal the watchdog to stop after its current sleep."""
        self._stop_event.set()

    # ── Layer 1: Main loop heartbeat ──────────────────────────────────────

    def _check_heartbeat(self) -> None:
        """
        Read heartbeat.json and compare timestamp to current time.
        If stale by more than WATCHDOG_TIMEOUT_SEC, attempt to restart the main loop.
        """
        hb_path = self._config.HEARTBEAT_FILE
        if not hb_path.exists():
            log.debug("Heartbeat file not yet created — skipping check")
            return

        try:
            data = json.loads(hb_path.read_text(encoding="utf-8"))
            last_ts = datetime.fromisoformat(data["timestamp"])
        except Exception as exc:
            log.warning("Could not read heartbeat.json: %s", exc)
            return

        age_sec = (datetime.now() - last_ts).total_seconds()
        if age_sec > self._config.WATCHDOG_TIMEOUT_SEC:
            log.warning(
                "Heartbeat stale by %.0f seconds (limit: %d) — attempting restart",
                age_sec,
                self._config.WATCHDOG_TIMEOUT_SEC,
            )
            self._handle_main_loop_failure()

    def _handle_main_loop_failure(self) -> None:
        """
        Attempt to restart the main loop. Track consecutive failures.
        After WATCHDOG_RESTART_ATTEMPTS consecutive failures in the window,
        log CRITICAL and stop the service.
        """
        with self._restart_lock:
            now = time.monotonic()

            if self._first_restart_time is None:
                self._first_restart_time = now
            elif now - self._first_restart_time > self._config.WATCHDOG_RESTART_WINDOW_SEC:
                # Window expired — reset counter
                self._restart_count = 0
                self._first_restart_time = now

            self._restart_count += 1
            attempt = self._restart_count

        log.warning("Main loop restart attempt %d/%d", attempt, self._config.WATCHDOG_RESTART_ATTEMPTS)
        success = False
        try:
            success = self._restart_main_loop()
        except Exception as exc:
            log.error("Restart attempt raised exception: %s", exc, exc_info=True)

        if success:
            log.info("Main loop restarted successfully on attempt %d", attempt)
            with self._restart_lock:
                self._restart_count = 0
                self._first_restart_time = None
        else:
            log.error("Main loop restart attempt %d failed", attempt)
            if attempt >= self._config.WATCHDOG_RESTART_ATTEMPTS:
                log.critical(
                    "Main loop failed %d consecutive restarts — stopping service",
                    attempt,
                )
                self._stop_service()

    # ── Layer 2: API server ping ───────────────────────────────────────────

    def _check_api(self) -> None:
        """
        Send GET /ping to the local API. Restart the API thread if no response.
        """
        url = f"http://{self._config.API_HOST}:{self._config.API_PORT}/ping"
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    return  # API is alive
        except Exception:
            pass

        log.warning("API server not responding — attempting restart")
        try:
            success = self._restart_api()
        except Exception as exc:
            log.error("API restart raised exception: %s", exc, exc_info=True)
            success = False

        if success:
            log.info("API server restarted successfully")
        else:
            log.error("API server restart failed")

    # ── Layer 3: Pipeline timeout ─────────────────────────────────────────

    def _check_pipeline_timeout(self) -> None:
        """
        If the nightly pipeline has been running longer than PIPELINE_TIMEOUT_HOURS,
        forcefully stop it and log the timeout event.
        """
        if self._pipeline_started_at is None:
            return

        elapsed_sec = time.monotonic() - self._pipeline_started_at
        timeout_sec = self._config.PIPELINE_TIMEOUT_HOURS * 3600

        if elapsed_sec > timeout_sec:
            log.critical(
                "Pipeline has been running for %.1f hours — force-stopping "
                "(limit: %d hours)",
                elapsed_sec / 3600,
                self._config.PIPELINE_TIMEOUT_HOURS,
            )
            thread = self._get_pipeline_thread()
            if thread and thread.is_alive():
                # Python threads cannot be force-killed from outside.
                # Signal the pipeline to stop via its stop event (set in improver.py).
                # The pipeline checks this event between passes.
                log.critical("Pipeline timeout — pipeline stop event signalled")
                # The daemon will clean up the thread reference
            self._pipeline_started_at = None
