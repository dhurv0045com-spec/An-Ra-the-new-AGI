"""
sovereignty/daemon.py
=====================
Main loop and thread coordinator for the Sovereignty daemon.

Manages 4 threads:
  Thread 1 — MainLoop: ticks every 60s, writes heartbeat, drives scheduler
  Thread 2 — Watchdog: monitors threads, restarts stalled ones
  Thread 3 — APIServer: serves the local REST API
  Thread 4 — Pipeline: spawned at nightly run time, joined or killed

All threads are user-level; the daemon runs as the current user.
No SYSTEM-level privileges. No auto-start. No survive-logoff.
Human must start this daemon explicitly.

Relationship to other modules:
    service.py calls Daemon.start() and Daemon.stop().
    watchdog.py receives callbacks to restart threads.
    api.py runs in Thread 3.
    improver.py runs in Thread 4.
    scheduler.py decides when Thread 4 should run.
"""

import gc
import json
import queue
import threading
import time
from datetime import datetime
from typing import Optional

from sovereignty.api import APIServer, SharedState
from sovereignty.config import Config
from sovereignty.improver import ImprovementPipeline
from sovereignty.logger import get_logger
from sovereignty.resource_monitor import ResourceMonitor
from sovereignty.scheduler import Scheduler
from sovereignty.watchdog import Watchdog

log = get_logger(__name__)


class Daemon:
    """
    Top-level daemon controller.

    Starts all threads and coordinates their lifecycle.
    Human permission is required to start this daemon (run service.py or demo.py).
    """

    def __init__(self, config: Config, target_dir=None) -> None:
        """
        Parameters:
            config: Active Config instance.
            target_dir: Directory of Python files for the improvement pipeline.
                        Defaults to the sovereignty package directory.
        """
        import pathlib
        self._config = config
        self._target_dir = target_dir or pathlib.Path(__file__).parent

        self._stop_event = threading.Event()
        self._pipeline_stop_event = threading.Event()
        self._start_time = time.monotonic()

        self._state = SharedState()
        self._scheduler = Scheduler(config)
        self._resource_monitor = ResourceMonitor(config)

        self._api_server = APIServer(config, self._state)
        self._watchdog: Optional[Watchdog] = None

        # Thread references — guarded by _thread_lock when modified
        self._thread_lock = threading.Lock()  # Protects thread dict modifications
        self._threads: dict = {}
        self._pipeline_thread: Optional[threading.Thread] = None

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def start(self) -> None:
        """
        Start all subsystem threads.

        Call this from service.py or demo.py after human confirmation.
        Does NOT require elevated privileges.
        """
        log.info("Sovereignty daemon starting (user-level, requires explicit start)")

        self._config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self._config.LOG_DIR.mkdir(parents=True, exist_ok=True)

        self._watchdog = Watchdog(
            config=self._config,
            restart_main_loop_fn=self._restart_main_loop,
            restart_api_fn=self._restart_api,
            get_pipeline_thread=self._get_pipeline_thread,
            stop_service_fn=self._emergency_stop,
        )

        self._start_main_loop()
        self._start_api()
        self._start_watchdog()
        self._start_resource_monitor()

        log.info("All threads started. Daemon running.")

    def stop(self, timeout: float = 30.0) -> None:
        """
        Gracefully stop all threads.

        Parameters:
            timeout: Maximum seconds to wait for threads to finish.
        """
        log.info("Daemon stopping — signalling all threads")
        self._stop_event.set()
        self._pipeline_stop_event.set()

        if self._watchdog:
            self._watchdog.stop()
        self._api_server.stop()
        self._resource_monitor.stop()

        with self._thread_lock:
            threads = list(self._threads.values())

        for t in threads:
            if t and t.is_alive():
                t.join(timeout=timeout / max(len(threads), 1))

        log.info("Daemon stopped cleanly")

    def wait_for_shutdown(self) -> None:
        """
        Block until stop() is called or a /shutdown API request is received.
        Used by service.py main loop.
        """
        while not self._stop_event.is_set():
            if self._state.shutdown_requested:
                log.info("Shutdown requested via API — stopping")
                self.stop()
                break
            time.sleep(1)

    # ── Thread starters ───────────────────────────────────────────────────

    def _start_main_loop(self) -> None:
        """Launch the main loop thread."""
        t = threading.Thread(
            target=self._main_loop,
            name="MainLoop",
            daemon=True,
        )
        with self._thread_lock:
            self._threads["MainLoop"] = t
        t.start()
        log.info("MainLoop thread started (tid=%d)", t.ident or 0)

    def _start_api(self) -> None:
        """Launch the API server thread."""
        t = threading.Thread(
            target=self._api_server.run_forever,
            name="APIServer",
            daemon=True,
        )
        with self._thread_lock:
            self._threads["APIServer"] = t
        t.start()
        log.info("APIServer thread started")

    def _start_watchdog(self) -> None:
        """Launch the watchdog thread."""
        t = threading.Thread(
            target=self._watchdog.run_forever,
            name="Watchdog",
            daemon=True,
        )
        with self._thread_lock:
            self._threads["Watchdog"] = t
        t.start()
        log.info("Watchdog thread started")

    def _start_resource_monitor(self) -> None:
        """Launch the resource monitor thread."""
        t = threading.Thread(
            target=self._resource_monitor.run_forever,
            name="ResourceMonitor",
            daemon=True,
        )
        with self._thread_lock:
            self._threads["ResourceMonitor"] = t
        t.start()

    # ── Main loop ─────────────────────────────────────────────────────────

    def _main_loop(self) -> None:
        """
        Main daemon tick loop.

        Each tick:
          1. Write heartbeat.json
          2. Update shared state for API /status
          3. Check scheduler — spawn pipeline if needed
          4. Drain the manual task queue
          5. Sleep until next tick
        """
        log.info("MainLoop: entering tick loop (tick every %ds)", self._config.DAEMON_TICK_SEC)
        while not self._stop_event.is_set():
            try:
                self._tick()
            except Exception as exc:
                log.error("MainLoop tick error: %s", exc, exc_info=True)
            self._stop_event.wait(timeout=self._config.DAEMON_TICK_SEC)
        log.info("MainLoop: exiting")

    def _tick(self) -> None:
        """Execute one main loop tick."""
        now = datetime.now()

        # 1. Heartbeat
        self._write_heartbeat(now)

        # 2. Update shared state
        resource = self._resource_monitor.get_snapshot()
        self._state.update(
            resource=resource,
            thread_status=self._thread_status_snapshot(),
            last_run_info=self._scheduler.last_run_info(),
            next_run=self._scheduler.next_run_dt(now).isoformat(),
        )

        # 3. Scheduler check
        if self._scheduler.should_run(now):
            self._maybe_spawn_pipeline()

        # 4. Task queue
        self._drain_task_queue()

    def _write_heartbeat(self, now: datetime) -> None:
        """Write timestamp to heartbeat.json so watchdog knows the loop is alive."""
        try:
            self._config.HEARTBEAT_FILE.write_text(
                json.dumps({"timestamp": now.isoformat()}), encoding="utf-8"
            )
        except Exception as exc:
            log.warning("Could not write heartbeat: %s", exc)

    def _thread_status_snapshot(self) -> dict:
        """Return a dict of thread_name → 'alive'/'dead' for all managed threads."""
        with self._thread_lock:
            status = {name: ("alive" if t.is_alive() else "dead") for name, t in self._threads.items()}
        if self._pipeline_thread:
            status["Pipeline"] = "alive" if self._pipeline_thread.is_alive() else "dead"
        return status

    # ── Pipeline management ───────────────────────────────────────────────

    def _maybe_spawn_pipeline(self) -> None:
        """Spawn the nightly pipeline thread if one is not already running."""
        with self._thread_lock:
            if self._pipeline_thread and self._pipeline_thread.is_alive():
                log.info("Pipeline already running — skipping spawn")
                return

        log.info("Spawning nightly improvement pipeline thread")
        self._pipeline_stop_event.clear()
        self._scheduler.mark_started()

        t = threading.Thread(
            target=self._pipeline_runner,
            name="Pipeline",
            daemon=True,
        )
        with self._thread_lock:
            self._pipeline_thread = t

        if self._watchdog:
            self._watchdog.notify_pipeline_started()
        t.start()

    def _pipeline_runner(self) -> None:
        """Pipeline thread body — runs the improvement pipeline and records result."""
        t_start = time.monotonic()
        result = "failed"
        try:
            pipeline = ImprovementPipeline(
                self._config,
                self._target_dir,
                stop_event=self._pipeline_stop_event,
            )
            result = pipeline.run()
        except Exception as exc:
            log.error("Pipeline runner exception: %s", exc, exc_info=True)
        finally:
            elapsed = time.monotonic() - t_start
            self._scheduler.mark_completed(result, elapsed)
            if self._watchdog:
                self._watchdog.notify_pipeline_stopped()
            log.info("Pipeline thread exiting — result=%s elapsed=%.1fs", result, elapsed)

    def _get_pipeline_thread(self) -> Optional[threading.Thread]:
        """Return the current pipeline thread (used by watchdog)."""
        with self._thread_lock:
            return self._pipeline_thread

    # ── Task queue ────────────────────────────────────────────────────────

    def _drain_task_queue(self) -> None:
        """Process any manually queued tasks from the API."""
        task_q = self._state._task_queue
        while True:
            try:
                task = task_q.get_nowait()
            except queue.Empty:
                break

            task_id = task["task_id"]
            task_name = task["task"]
            self._state.update_task(task_id, "running")
            log.info("Executing manual task: %s (id=%s)", task_name, task_id)

            try:
                if task_name == "gc":
                    gc.collect()
                    log.info("Manual GC complete")
                elif task_name == "rotate_logs":
                    log.info("Manual log rotation requested (rotation is automatic at midnight)")
                elif task_name == "run_full_pipeline":
                    self._maybe_spawn_pipeline()
                elif task_name == "run_pass_1":
                    from sovereignty.auditor import AuditPass
                    AuditPass(self._config, self._target_dir).run()
                self._state.update_task(task_id, "done")
            except Exception as exc:
                log.error("Task %s failed: %s", task_name, exc, exc_info=True)
                self._state.update_task(task_id, "failed")

    # ── Thread restart callbacks (used by watchdog) ───────────────────────

    def _restart_main_loop(self) -> bool:
        """
        Restart the main loop thread after a watchdog-detected freeze.

        Returns:
            True if the new thread started successfully.
        """
        log.warning("Restarting main loop thread")
        try:
            with self._thread_lock:
                old = self._threads.get("MainLoop")
            if old and old.is_alive():
                # Signal it to stop, but don't wait — watchdog already decided it's frozen
                pass

            self._start_main_loop()
            time.sleep(2)  # Give it time to start
            with self._thread_lock:
                return self._threads["MainLoop"].is_alive()
        except Exception as exc:
            log.error("Main loop restart failed: %s", exc, exc_info=True)
            return False

    def _restart_api(self) -> bool:
        """
        Restart the API server thread.

        Returns:
            True if the new thread started successfully.
        """
        log.warning("Restarting API server thread")
        try:
            self._api_server.stop()
            time.sleep(1)
            self._api_server = APIServer(self._config, self._state)
            self._start_api()
            time.sleep(2)
            with self._thread_lock:
                return self._threads["APIServer"].is_alive()
        except Exception as exc:
            log.error("API restart failed: %s", exc, exc_info=True)
            return False

    def _emergency_stop(self) -> None:
        """Called by watchdog after all restart attempts fail."""
        log.critical("Emergency stop triggered by watchdog — shutting down daemon")
        self._stop_event.set()
