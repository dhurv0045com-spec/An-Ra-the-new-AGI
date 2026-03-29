"""
sovereignty/resource_monitor.py
================================
CPU and RAM monitoring for the Sovereignty daemon process.

Runs as a periodic check (every RESOURCE_CHECK_INTERVAL_SEC seconds).
Warns when CPU stays high or RAM exceeds the configured limit.
Triggers garbage collection on high RAM before escalating to CRITICAL.

Relationship to other modules:
    watchdog.py calls check_resources() from its monitoring loop.
    daemon.py starts the resource monitor thread.
    api.py reads the latest snapshot for GET /status.
"""

import gc
import threading
import time
from typing import Dict, Optional

try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False

from sovereignty.config import Config
from sovereignty.logger import get_logger

log = get_logger(__name__)


class ResourceMonitor:
    """
    Periodic CPU/RAM monitor for the current process.

    Attributes:
        _lock: Protects _latest_snapshot from concurrent reads/writes.
               (The monitor thread writes; API thread reads.)
        _latest_snapshot: The most recent resource readings.
        _cpu_high_since: Timestamp when CPU first exceeded the warning threshold.
    """

    def __init__(self, config: Config) -> None:
        """
        Parameters:
            config: Active Config instance.
        """
        self._config = config
        self._lock = threading.Lock()  # Protects _latest_snapshot and _cpu_high_since
        self._latest_snapshot: Dict[str, float] = {"cpu_pct": 0.0, "ram_mb": 0.0}
        self._cpu_high_since: Optional[float] = None
        self._stop_event = threading.Event()

        if not _PSUTIL_AVAILABLE:
            log.warning("psutil not available — resource monitoring disabled")

    def get_snapshot(self) -> Dict[str, float]:
        """
        Return the most recently recorded CPU and RAM readings.

        Returns:
            Dict with keys 'cpu_pct' (float) and 'ram_mb' (float).
        """
        with self._lock:
            return dict(self._latest_snapshot)

    def check_resources(self) -> Dict[str, float]:
        """
        Sample current CPU and RAM usage, log warnings if thresholds are exceeded,
        and trigger GC if RAM is high.

        Returns:
            Dict with keys 'cpu_pct' and 'ram_mb'.
        """
        if not _PSUTIL_AVAILABLE:
            return {"cpu_pct": 0.0, "ram_mb": 0.0}

        try:
            proc = psutil.Process()
            cpu_pct = proc.cpu_percent(interval=1.0)
            ram_mb = proc.memory_info().rss / (1024 * 1024)
        except Exception as exc:
            log.warning("Resource check failed: %s", exc)
            return {"cpu_pct": 0.0, "ram_mb": 0.0}

        snapshot = {"cpu_pct": round(cpu_pct, 1), "ram_mb": round(ram_mb, 1)}

        with self._lock:
            self._latest_snapshot = snapshot

        # ── CPU check ──────────────────────────────────────────────────────
        if cpu_pct > self._config.RESOURCE_CPU_WARN_PCT:
            if self._cpu_high_since is None:
                self._cpu_high_since = time.monotonic()
            elif (
                time.monotonic() - self._cpu_high_since
                > self._config.RESOURCE_CPU_SUSTAINED_SEC
            ):
                log.warning(
                    "CPU has been above %.0f%% for over %d seconds (current: %.1f%%)",
                    self._config.RESOURCE_CPU_WARN_PCT,
                    self._config.RESOURCE_CPU_SUSTAINED_SEC,
                    cpu_pct,
                )
        else:
            self._cpu_high_since = None  # Reset when CPU drops back down

        # ── RAM check ──────────────────────────────────────────────────────
        if ram_mb > self._config.RESOURCE_RAM_WARN_MB:
            log.warning(
                "RAM usage %.1f MB exceeds limit %.0f MB — triggering GC",
                ram_mb,
                self._config.RESOURCE_RAM_WARN_MB,
            )
            gc.collect()

            # Re-sample after GC
            try:
                ram_mb_after = proc.memory_info().rss / (1024 * 1024)
            except Exception:
                ram_mb_after = ram_mb

            if ram_mb_after > self._config.RESOURCE_RAM_WARN_MB:
                log.critical(
                    "RAM still %.1f MB after GC — possible memory leak",
                    ram_mb_after,
                )
            else:
                log.info("GC reduced RAM from %.1f MB to %.1f MB", ram_mb, ram_mb_after)

        return snapshot

    def run_forever(self) -> None:
        """
        Entry point for the resource monitor thread.

        Checks resources every RESOURCE_CHECK_INTERVAL_SEC seconds until
        stop() is called.
        """
        log.info("Resource monitor started (interval: %ds)", self._config.RESOURCE_CHECK_INTERVAL_SEC)
        while not self._stop_event.is_set():
            try:
                self.check_resources()
            except Exception as exc:
                log.error("Resource monitor error: %s", exc, exc_info=True)
            self._stop_event.wait(timeout=self._config.RESOURCE_CHECK_INTERVAL_SEC)
        log.info("Resource monitor stopped")

    def stop(self) -> None:
        """Signal the monitor thread to stop after its current sleep."""
        self._stop_event.set()
