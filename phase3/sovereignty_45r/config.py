"""
sovereignty/config.py
=====================
Central configuration for the entire Sovereignty Package.

All tunable parameters live here. Every other module imports from this file.
Change a value here and it takes effect everywhere — no hunting through code.

Relationship to other modules:
  All modules import Config. No module defines its own magic numbers.
"""

import os
import pathlib
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Config:
    """
    All configuration parameters for the Sovereignty daemon.

    Parameters are grouped by subsystem. Defaults match the spec.
    Override by passing keyword arguments or modifying config.json.
    """

    # ── Paths ──────────────────────────────────────────────────────────────
    # Base data directory — user-writable, visible, not hidden
    DATA_DIR: pathlib.Path = field(
        default_factory=lambda: pathlib.Path(
            os.environ.get("SOVEREIGNTY_DATA", r"C:\sovereignty_data")
        )
    )

    @property
    def LOG_DIR(self) -> pathlib.Path:
        """Directory for rotating daily log files."""
        return self.DATA_DIR / "logs"

    @property
    def TOKEN_FILE(self) -> pathlib.Path:
        """File holding the 32-byte API bearer token (hex string)."""
        return self.DATA_DIR / "token.key"

    @property
    def HEARTBEAT_FILE(self) -> pathlib.Path:
        """JSON file written every 60 s by the main loop."""
        return self.DATA_DIR / "heartbeat.json"

    @property
    def RUN_LOG_FILE(self) -> pathlib.Path:
        """JSON file recording every completed nightly run."""
        return self.DATA_DIR / "run_log.json"

    @property
    def AUDIT_BASELINE_FILE(self) -> pathlib.Path:
        """JSON baseline for Pass 1 code-audit deltas."""
        return self.DATA_DIR / "audit_baseline.json"

    @property
    def BENCHMARK_BASELINE_FILE(self) -> pathlib.Path:
        """JSON baseline for Pass 3 benchmark deltas."""
        return self.DATA_DIR / "benchmark_baseline.json"

    @property
    def CONFIG_FILE(self) -> pathlib.Path:
        """Persisted config overrides (JSON)."""
        return self.DATA_DIR / "config.json"

    # ── API ────────────────────────────────────────────────────────────────
    API_PORT: int = 45000
    """TCP port for the local REST API. Binds to 127.0.0.1 only."""

    API_HOST: str = "127.0.0.1"
    """Host to bind the REST API. Always localhost — never exposed externally."""

    # ── Watchdog ───────────────────────────────────────────────────────────
    WATCHDOG_HEARTBEAT_SEC: int = 60
    """How often the main loop writes a heartbeat timestamp."""

    WATCHDOG_TIMEOUT_SEC: int = 90
    """If no heartbeat for this many seconds, the main loop is considered frozen."""

    WATCHDOG_RESTART_ATTEMPTS: int = 3
    """Max consecutive restart attempts before the watchdog escalates to CRITICAL."""

    WATCHDOG_RESTART_WINDOW_SEC: int = 300
    """If 3 restarts all fail within this window, escalate to CRITICAL."""

    # ── Resource limits ────────────────────────────────────────────────────
    RESOURCE_CPU_WARN_PCT: float = 80.0
    """Sustained CPU usage above this % triggers a WARNING log."""

    RESOURCE_CPU_SUSTAINED_SEC: int = 600
    """How many seconds CPU must stay above threshold before WARNING fires."""

    RESOURCE_RAM_WARN_MB: float = 500.0
    """RAM usage above this MB triggers WARNING + GC attempt."""

    RESOURCE_CHECK_INTERVAL_SEC: int = 300
    """How often the resource monitor samples CPU and RAM."""

    # ── Logging ────────────────────────────────────────────────────────────
    LOG_RETENTION_DAYS: int = 30
    """Compressed logs older than this many days are auto-deleted."""

    LOG_LEVEL: str = "INFO"
    """Minimum log level written to file. DEBUG produces very verbose output."""

    # ── Improvement pipeline ───────────────────────────────────────────────
    IMPROVEMENT_START: str = "02:00"
    """Local time (HH:MM) when the nightly pipeline may begin."""

    IMPROVEMENT_END: str = "05:00"
    """Local time (HH:MM) after which the pipeline will not start a new run."""

    CATCHUP_WINDOW_HOURS: int = 6
    """If the machine was off during the window, try again within this many hours."""

    PIPELINE_TIMEOUT_HOURS: int = 3
    """Force-kill the pipeline thread if it runs longer than this."""

    # ── Benchmark thresholds ───────────────────────────────────────────────
    BENCHMARK_COUNT: int = 10
    """Number of benchmarks in the suite (B01–B10)."""

    BENCHMARK_REGRESSION_PCT: float = 5.0
    """Flag REGRESSION if a benchmark is more than this % slower than baseline."""

    BENCHMARK_IMPROVEMENT_PCT: float = 5.0
    """Flag IMPROVEMENT if a benchmark is more than this % faster than baseline."""

    BENCHMARK_RUNS: int = 3
    """How many times each benchmark runs; median is taken."""

    # ── Code quality thresholds ────────────────────────────────────────────
    COMPLEXITY_MAX_CYCLOMATIC: int = 10
    """Functions with cyclomatic complexity above this are flagged."""

    COMPLEXITY_MAX_COGNITIVE: int = 15
    """Functions with cognitive complexity above this are flagged."""

    COMPLEXITY_MAX_LINES: int = 50
    """Functions longer than this many lines are flagged for review."""

    COMPLEXITY_MAX_NESTING: int = 4
    """Code nested deeper than this many levels is flagged."""

    COMPLEXITY_MIN_COMMENT_RATIO: float = 0.10
    """Functions with comment ratio below this are flagged."""

    # ── Auth ───────────────────────────────────────────────────────────────
    TOKEN_BYTES: int = 32
    """Number of random bytes used to generate the API bearer token."""

    # ── Daemon tick ────────────────────────────────────────────────────────
    DAEMON_TICK_SEC: int = 60
    """Main loop sleeps this many seconds between ticks."""

    @classmethod
    def from_json(cls, path: Optional[pathlib.Path] = None) -> "Config":
        """
        Load a Config from a JSON file, falling back to defaults for missing keys.

        Parameters:
            path: Path to config.json. If None, uses the default DATA_DIR location.

        Returns:
            Config instance with values from JSON overlaid on defaults.

        Raises:
            Does not raise — missing or malformed JSON silently uses defaults.
        """
        import json

        instance = cls()
        target = path or instance.CONFIG_FILE
        if not target.exists():
            return instance
        try:
            data = json.loads(target.read_text(encoding="utf-8"))
            for key, value in data.items():
                if hasattr(instance, key) and not key.startswith("_"):
                    setattr(instance, key, value)
        except Exception:
            pass  # Bad JSON → use defaults; logger not yet available here
        return instance

    def to_json(self) -> dict:
        """
        Serialise config to a plain dict (for writing config.json).

        Returns:
            dict of all non-property fields and their current values.
        """
        import dataclasses
        return {
            f.name: str(getattr(self, f.name))
            if isinstance(getattr(self, f.name), pathlib.Path)
            else getattr(self, f.name)
            for f in dataclasses.fields(self)
        }
