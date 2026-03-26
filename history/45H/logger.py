"""
================================================================================
FILE: logger.py
PROJECT: Transformer Language Model — 45H Final Phase
PURPOSE: Structured logging + live training dashboard
================================================================================

Features:
  - Structured JSON-compatible log records
  - Live terminal dashboard (loss, LR, grad norm, tokens/sec, memory)
  - Optional Weights & Biases integration (gracefully disabled if not installed)
  - Timestamped log files saved automatically
  - Configurable log levels
  - Thread-safe (no shared mutable state)
================================================================================
"""

import os
import sys
import time
import json
import logging
import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from collections import deque


# ──────────────────────────────────────────────────────────────────────────────
# FORMATTERS
# ──────────────────────────────────────────────────────────────────────────────

class ConsoleFormatter(logging.Formatter):
    """Color-coded console formatter with level-based colorization."""

    COLORS = {
        "DEBUG":    "\033[36m",     # Cyan
        "INFO":     "\033[32m",     # Green
        "WARNING":  "\033[33m",     # Yellow
        "ERROR":    "\033[31m",     # Red
        "CRITICAL": "\033[35m",     # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color  = self.COLORS.get(record.levelname, "")
        reset  = self.RESET if color else ""
        ts     = datetime.datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
        prefix = f"{color}[{ts}] [{record.levelname[:4]}]{reset}"
        return f"{prefix} {record.getMessage()}"


class JSONFormatter(logging.Formatter):
    """Structured JSON formatter for log files — machine-parseable."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts":      datetime.datetime.utcfromtimestamp(record.created).isoformat() + "Z",
            "level":   record.levelname,
            "logger":  record.name,
            "msg":     record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload)


# ──────────────────────────────────────────────────────────────────────────────
# SETUP
# ──────────────────────────────────────────────────────────────────────────────

def setup_logging(
    level:          str  = "INFO",
    log_to_file:    bool = True,
    log_to_console: bool = True,
    log_dir:        str  = "output/logs",
    run_name:       Optional[str] = None,
) -> Path:
    """
    Configure the root logger with console + file handlers.

    Args:
        level:          Log level string: "DEBUG", "INFO", "WARNING", "ERROR"
        log_to_file:    Whether to write to a file
        log_to_console: Whether to write to stdout
        log_dir:        Directory for log files
        run_name:       Optional run identifier for the log filename

    Returns:
        Path to the log file (or None if log_to_file=False)
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(numeric_level)

    # Clear existing handlers (avoid duplicate logs on re-init)
    root.handlers.clear()

    log_file_path = None

    if log_to_console:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(ConsoleFormatter())
        handler.setLevel(numeric_level)
        root.addHandler(handler)

    if log_to_file:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"{run_name}_{ts}" if run_name else ts
        log_file_path = Path(log_dir) / f"run_{name}.log"

        file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
        file_handler.setFormatter(JSONFormatter())
        file_handler.setLevel(numeric_level)
        root.addHandler(file_handler)

        logging.getLogger(__name__).info(f"Logging to file: {log_file_path}")

    return log_file_path


# ──────────────────────────────────────────────────────────────────────────────
# TRAINING DASHBOARD
# ──────────────────────────────────────────────────────────────────────────────

class TrainingDashboard:
    """
    Live terminal training dashboard.

    Displays a single-line updating status bar while training, plus periodic
    multi-line metric reports. Uses ANSI escape codes for in-place updates.

    Usage:
        dash = TrainingDashboard(max_steps=10000, log_every=10)
        for step in range(10000):
            ... # training step
            dash.update(step, loss=1.23, lr=3e-4, grad_norm=0.5, tokens_per_sec=1200)
            if step % 500 == 0:
                dash.report(val_loss=1.18)
    """

    def __init__(
        self,
        max_steps:    int,
        log_every:    int   = 10,
        smooth_window: int  = 50,
    ):
        self.max_steps    = max_steps
        self.log_every    = log_every
        self.logger       = logging.getLogger("dashboard")

        # Smoothed metrics (exponential moving average)
        self._loss_window = deque(maxlen=smooth_window)
        self._tps_window  = deque(maxlen=smooth_window)

        self._step_start_time = time.time()
        self._run_start_time  = time.time()
        self._last_log_step   = 0
        self._best_val_loss   = float("inf")

        # W&B handle (lazily initialized)
        self._wandb = None

    def _elapsed(self) -> str:
        """Format elapsed time as HH:MM:SS."""
        secs = int(time.time() - self._run_start_time)
        h, m, s = secs // 3600, (secs % 3600) // 60, secs % 60
        return f"{h:02d}:{m:02d}:{s:02d}"

    def _eta(self, step: int) -> str:
        """Estimate time remaining based on average step time."""
        elapsed = time.time() - self._run_start_time
        if step == 0:
            return "--:--:--"
        secs_per_step = elapsed / max(step, 1)
        remaining = int(secs_per_step * (self.max_steps - step))
        h, m, s = remaining // 3600, (remaining % 3600) // 60, remaining % 60
        return f"{h:02d}:{m:02d}:{s:02d}"

    def _memory_str(self) -> str:
        """Current process memory usage (RSS) as a string."""
        try:
            import resource
            rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
            return f"{rss_mb:.0f}MB"
        except Exception:
            return "N/A"

    def update(
        self,
        step:          int,
        loss:          float,
        lr:            float,
        grad_norm:     float = 0.0,
        tokens_per_sec: float = 0.0,
        **extra_metrics,
    ) -> None:
        """
        Record metrics for this step and update the dashboard display.

        Args:
            step:           Current training step
            loss:           Training loss for this step
            lr:             Current learning rate
            grad_norm:      Gradient norm (0 if not tracked)
            tokens_per_sec: Throughput estimate
            **extra_metrics: Any additional metrics to log
        """
        self._loss_window.append(loss)
        self._tps_window.append(tokens_per_sec)

        smooth_loss = sum(self._loss_window) / len(self._loss_window)
        smooth_tps  = sum(self._tps_window)  / len(self._tps_window)

        pct = 100.0 * step / max(self.max_steps, 1)
        bar_width = 20
        filled = int(bar_width * step / max(self.max_steps, 1))
        bar = "█" * filled + "░" * (bar_width - filled)

        # Build the live status line
        status = (
            f"\r  [{bar}] {pct:5.1f}%  "
            f"step={step:>6}/{self.max_steps}  "
            f"loss={smooth_loss:.4f}  "
            f"lr={lr:.2e}  "
            f"‖g‖={grad_norm:.2f}  "
            f"tok/s={smooth_tps:>6.0f}  "
            f"mem={self._memory_str()}  "
            f"elapsed={self._elapsed()}  "
            f"ETA={self._eta(step)}"
        )
        sys.stdout.write(status)
        sys.stdout.flush()

        # Periodic structured log
        if step > 0 and step % self.log_every == 0:
            metrics = {
                "step": step,
                "loss": round(smooth_loss, 6),
                "lr":   lr,
                "grad_norm": round(grad_norm, 4),
                "tokens_per_sec": round(smooth_tps, 1),
                **extra_metrics,
            }
            self.logger.info(f"train {json.dumps(metrics)}")

            # W&B logging
            if self._wandb is not None:
                try:
                    self._wandb.log(metrics, step=step)
                except Exception as e:
                    self.logger.warning(f"W&B log failed: {e}")

    def report(self, step: int = 0, **metrics) -> None:
        """
        Print a full multi-line validation/checkpoint report.

        Clears the progress bar line first, then prints a formatted block.
        """
        sys.stdout.write("\r" + " " * 120 + "\r")  # clear progress line

        val_loss = metrics.get("val_loss")
        is_best  = val_loss is not None and val_loss < self._best_val_loss
        if is_best and val_loss is not None:
            self._best_val_loss = val_loss

        tag = "★ BEST" if is_best else "     "

        self.logger.info("─" * 60)
        self.logger.info(f"  Step {step:>6} / {self.max_steps}  |  Elapsed: {self._elapsed()}")
        for k, v in metrics.items():
            formatted = f"{v:.6f}" if isinstance(v, float) else str(v)
            self.logger.info(f"  {k:<20} = {formatted}  {tag if k=='val_loss' else ''}")
        self.logger.info("─" * 60)

        if self._wandb is not None:
            try:
                self._wandb.log({f"val/{k}": v for k, v in metrics.items()}, step=step)
            except Exception:
                pass

    def init_wandb(self, project: str, entity: Optional[str] = None,
                   run_name: Optional[str] = None, config: Dict = None) -> bool:
        """
        Lazily initialize Weights & Biases. Returns True if successful.

        Does NOT crash if wandb is not installed — degrades gracefully.
        """
        try:
            import wandb
            self._wandb = wandb
            wandb.init(
                project=project,
                entity=entity,
                name=run_name,
                config=config or {},
                resume="allow",
            )
            self.logger.info(f"W&B initialized: {project}/{run_name or 'auto'}")
            return True
        except ImportError:
            self.logger.warning(
                "wandb not installed — skipping W&B logging. "
                "Install with: pip install wandb"
            )
            return False
        except Exception as e:
            self.logger.warning(f"W&B init failed (continuing without it): {e}")
            return False

    def finish(self) -> None:
        """Finalize the dashboard — print completion message and finish W&B run."""
        sys.stdout.write("\n")
        self.logger.info(
            f"Training complete. Total time: {self._elapsed()}. "
            f"Best val loss: {self._best_val_loss:.6f}"
        )
        if self._wandb is not None:
            try:
                self._wandb.finish()
            except Exception:
                pass


# ──────────────────────────────────────────────────────────────────────────────
# SELF-TEST
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = setup_logging(level="DEBUG", log_to_file=True, log_dir=tmpdir)
        print(f"Log file: {log_path}")

        log = logging.getLogger("test")
        log.info("Logger initialized")
        log.warning("This is a warning")
        log.error("This is an error")

        print("\nDashboard simulation (2 seconds)...")
        dash = TrainingDashboard(max_steps=100, log_every=5)

        for step in range(1, 51):
            import random
            loss = 4.0 * (0.98 ** step) + random.uniform(-0.05, 0.05)
            dash.update(step, loss=loss, lr=3e-4 * (0.99**step),
                        grad_norm=random.uniform(0.1, 2.0),
                        tokens_per_sec=random.uniform(800, 1200))
            time.sleep(0.04)

        dash.report(step=50, val_loss=3.2, perplexity=24.5)
        dash.finish()

        # Verify log file was written
        with open(log_path) as f:
            lines = f.readlines()
        print(f"\nLog file has {len(lines)} lines. First entry:")
        print(" ", lines[0].strip())
        assert len(lines) > 0

    print("\n✓ Logger tests passed")
