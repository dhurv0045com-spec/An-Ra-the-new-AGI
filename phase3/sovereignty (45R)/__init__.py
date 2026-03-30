"""
sovereignty/__init__.py
=======================
Public API, version metadata, and top-level exports for the Sovereignty Package.

This package implements a transparent, user-level self-improvement daemon for
Python projects. It runs as a normal user process (never SYSTEM), requires
explicit human permission to install/uninstall, and produces measurable
improvement metrics every night.

Relationship to other modules:
  - config.py      : All tunable parameters imported by every other module
  - service.py     : Entry point — starts/stops all subsystems
  - daemon.py      : Main loop coordinator
  - scheduler.py   : Decides when to run the nightly pipeline
  - watchdog.py    : Monitors threads and restarts stalled ones
  - api.py         : Local REST API for status/control
  - auth.py        : Token generation and request authentication
  - improver.py    : Orchestrates the 4-pass improvement pipeline
  - auditor.py     : Pass 1 — AST complexity analysis
  - dead_code.py   : Pass 2 — Dead code and quality sweep
  - benchmarks.py  : Pass 3 — Performance benchmark suite
  - reporter.py    : Pass 4 — Nightly human-readable report
  - logger.py      : Thread-safe rotating log system
  - resource_monitor.py : CPU/RAM usage tracking via psutil
  - install.py     : 15-step atomic install with rollback
  - uninstall.py   : Clean removal with log retention option
  - demo.py        : Full simulation requiring zero admin rights
"""

__version__ = "1.0.0"
__author__ = "Sovereignty Package"
__description__ = "Transparent user-level Python self-improvement daemon"

# Public re-exports for convenience
from sovereignty.config import Config
from sovereignty.logger import get_logger

__all__ = [
    "__version__",
    "__author__",
    "__description__",
    "Config",
    "get_logger",
]
