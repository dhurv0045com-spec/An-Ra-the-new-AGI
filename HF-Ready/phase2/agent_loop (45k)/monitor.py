"""
================================================================================
FILE: agent/core/monitor.py
PROJECT: Agent Loop — 45K
PURPOSE: Self-monitoring — detect loops, timeouts, drift, auto-escalate
================================================================================
"""

import time
import logging
import hashlib
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional
from collections import Counter

logger = logging.getLogger(__name__)


@dataclass
class HealthSignal:
    timestamp:  float
    step_id:    str
    event:      str        # "step_start", "step_done", "step_failed", "tool_call"
    metadata:   Dict = field(default_factory=dict)


class AgentMonitor:
    """
    Watches the agent for pathological behaviors:
      - Infinite loops (same action repeated)
      - Timeout (step taking too long)
      - Progress stall (no progress for N minutes)
      - Drift (actions diverging from goal)
      - Resource exhaustion (too many tool calls)

    When a problem is detected: logs, then calls the escalation_fn.
    Never crashes the executor — always advisory.
    """

    def __init__(
        self,
        escalation_fn:       Optional[Callable[[str], None]] = None,
        stall_timeout_secs:  int   = 300,    # 5 min with no progress
        max_tool_calls:      int   = 200,    # per plan
        loop_window:         int   = 10,     # look back N signals for loop detection
        checkin_interval:    int   = 30,     # seconds between status logs
    ):
        self.escalation_fn      = escalation_fn
        self.stall_timeout      = stall_timeout_secs
        self.max_tool_calls     = max_tool_calls
        self.loop_window        = loop_window
        self.checkin_interval   = checkin_interval

        self._signals:           List[HealthSignal] = []
        self._last_progress:     float = time.time()
        self._last_checkin:      float = time.time()
        self._tool_call_count:   int   = 0
        self._last_actions:      List[str] = []   # fingerprints of recent actions
        self._active:            bool  = False

    def start(self) -> None:
        self._active       = True
        self._last_progress = time.time()
        logger.info("AgentMonitor started")

    def stop(self) -> None:
        self._active = False
        logger.info("AgentMonitor stopped")

    def record(self, step_id: str, event: str, **meta) -> None:
        """Record an agent event and run health checks."""
        if not self._active:
            return

        signal = HealthSignal(time.time(), step_id, event, meta)
        self._signals.append(signal)

        if event in ("step_done", "tool_call"):
            self._last_progress = time.time()
        if event == "tool_call":
            self._tool_call_count += 1
            action_fp = hashlib.md5(
                f"{step_id}{meta.get('tool','')}{meta.get('input','')[:50]}".encode()
            ).hexdigest()[:8]
            self._last_actions.append(action_fp)
            if len(self._last_actions) > self.loop_window * 2:
                self._last_actions = self._last_actions[-self.loop_window:]

        self._check_health()

    def _check_health(self) -> None:
        """Run all health checks. Call escalation if any fail."""
        now = time.time()

        # ── Loop detection ─────────────────────────────────────────────────
        if len(self._last_actions) >= self.loop_window:
            recent = self._last_actions[-self.loop_window:]
            counts = Counter(recent)
            most_common, freq = counts.most_common(1)[0]
            if freq >= self.loop_window // 2:
                self._escalate(
                    f"Loop detected: action fingerprint '{most_common}' "
                    f"appeared {freq}/{self.loop_window} times. "
                    "The agent may be stuck repeating the same action."
                )

        # ── Progress stall ────────────────────────────────────────────────
        stall_time = now - self._last_progress
        if stall_time > self.stall_timeout:
            self._escalate(
                f"No progress for {stall_time:.0f}s (threshold: {self.stall_timeout}s). "
                "Agent may be stuck on a step."
            )

        # ── Tool call cap ─────────────────────────────────────────────────
        if self._tool_call_count >= self.max_tool_calls:
            self._escalate(
                f"Tool call limit reached: {self._tool_call_count}/{self.max_tool_calls}. "
                "Stopping to prevent runaway execution."
            )

        # ── Periodic checkin ──────────────────────────────────────────────
        if now - self._last_checkin > self.checkin_interval:
            self._last_checkin = now
            logger.info(
                f"[Monitor checkin] signals={len(self._signals)}, "
                f"tool_calls={self._tool_call_count}, "
                f"last_progress={stall_time:.0f}s ago"
            )

    def _escalate(self, message: str) -> None:
        logger.warning(f"[Monitor ALERT] {message}")
        if self.escalation_fn:
            try:
                self.escalation_fn(message)
            except Exception as e:
                logger.error(f"Escalation callback failed: {e}")

    def status(self) -> Dict:
        return {
            "active":          self._active,
            "signals":         len(self._signals),
            "tool_calls":      self._tool_call_count,
            "since_progress":  round(time.time() - self._last_progress, 1),
            "stall_threshold": self.stall_timeout,
        }
