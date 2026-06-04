"""
sovereignty/improver.py
=======================
Coordinator for the 4-pass nightly improvement pipeline.

Runs Passes 1–4 in sequence. Between passes, checks a stop_event so the
watchdog can signal an early termination if the pipeline exceeds its timeout.

Relationship to other modules:
    daemon.py spawns the improvement thread and calls ImprovementPipeline.run().
    watchdog.py sets the stop_event if the pipeline runs too long.
    auditor.py   — Pass 1
    dead_code.py — Pass 2
    benchmarks.py — Pass 3
    reporter.py  — Pass 4
    scheduler.py is called to mark started/completed.
"""

import pathlib
import threading
import time
from datetime import datetime
from typing import Optional

from sovereignty.config import Config
from sovereignty.logger import get_logger

log = get_logger(__name__)


class ImprovementPipeline:
    """
    Runs the 4-pass self-improvement pipeline.

    Designed to run in a dedicated thread. Checks stop_event between passes
    so that watchdog timeouts are respected.
    """

    def __init__(
        self,
        config: Config,
        target_dir: pathlib.Path,
        stop_event: Optional[threading.Event] = None,
    ) -> None:
        """
        Parameters:
            config: Active Config instance.
            target_dir: Directory of Python source files to analyse.
            stop_event: Optional threading.Event — if set, the pipeline
                        stops after the current pass completes.
        """
        self._config = config
        self._target_dir = target_dir
        self._stop_event = stop_event or threading.Event()

    def run(self, date_str: Optional[str] = None) -> str:
        """
        Execute all 4 passes. Returns a result string: 'success', 'partial', 'failed'.

        Parameters:
            date_str: Date label (YYYYMMDD). Defaults to today.

        Returns:
            'success'  — all 4 passes completed
            'partial'  — some passes completed before stop_event was set
            'failed'   — Pass 1 could not complete (nothing useful produced)
        """
        from sovereignty.auditor import AuditPass
        from sovereignty.dead_code import DeadCodePass
        from sovereignty.benchmarks import BenchmarkPass
        from sovereignty.reporter import ReportPass

        date_str = date_str or datetime.now().strftime("%Y%m%d")
        t_start = time.monotonic()
        log.info("Pipeline starting — date %s, target %s", date_str, self._target_dir)

        audit_result = dead_result = bench_result = None
        passes_done = 0

        # ── Pass 1: Code Audit ─────────────────────────────────────────────
        try:
            log.info("Pipeline: Pass 1 — Code Audit")
            audit_result = AuditPass(self._config, self._target_dir).run(date_str)
            passes_done += 1
        except Exception as exc:
            log.error("Pass 1 failed: %s", exc, exc_info=True)
            return "failed"

        if self._stop_event.is_set():
            log.warning("Pipeline: stop_event set after Pass 1 — stopping early")
            return "partial"

        # ── Pass 2: Dead Code Sweep ────────────────────────────────────────
        try:
            log.info("Pipeline: Pass 2 — Dead Code Sweep")
            dead_result = DeadCodePass(self._config, self._target_dir).run(date_str)
            passes_done += 1
        except Exception as exc:
            log.error("Pass 2 failed: %s", exc, exc_info=True)

        if self._stop_event.is_set():
            log.warning("Pipeline: stop_event set after Pass 2 — stopping early")
            return "partial"

        # ── Pass 3: Benchmarks ─────────────────────────────────────────────
        try:
            log.info("Pipeline: Pass 3 — Performance Benchmarks")
            bench_result = BenchmarkPass(self._config).run(date_str)
            passes_done += 1
        except Exception as exc:
            log.error("Pass 3 failed: %s", exc, exc_info=True)

        if self._stop_event.is_set():
            log.warning("Pipeline: stop_event set after Pass 3 — stopping early")
            return "partial"

        # ── Pass 4: Nightly Report ─────────────────────────────────────────
        try:
            log.info("Pipeline: Pass 4 — Nightly Report")
            ReportPass(self._config).run(
                audit_result=audit_result,
                dead_result=dead_result,
                bench_result=bench_result,
                date_str=date_str,
            )
            passes_done += 1
        except Exception as exc:
            log.error("Pass 4 failed: %s", exc, exc_info=True)

        elapsed = time.monotonic() - t_start
        result = "success" if passes_done == 4 else "partial"
        log.info(
            "Pipeline complete: %d/4 passes, result=%s, elapsed=%.1fs",
            passes_done, result, elapsed,
        )
        return result
