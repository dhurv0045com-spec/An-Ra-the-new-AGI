"""Authoritative cumulative learned-smoke ledger (B11).

Tracks CPU and GPU learned-smoke seconds and optimizer updates across all
helpers and retries. The limits from the build packet apply to the whole
assignment, not per run. A budget limit is a stopping boundary: the ledger
refuses to record work beyond it, and callers must check before starting.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any

LEDGER_SCHEMA = "bramastra-smoke-ledger/v1"

LIMITS = {
    "cpu_learned_smoke_seconds": 300,
    "cpu_optimizer_updates": 200,
    "gpu_sessions": 1,
    "gpu_wall_seconds": 600,
    "gpu_optimizer_updates": 200,
}

# Fresh-process resume verification must be reserved before other smoke.
RESERVED_FOR_RESUME = {"cpu_optimizer_updates": 40, "cpu_learned_smoke_seconds": 120}


class SmokeBudgetExhausted(RuntimeError):
    """The cumulative learned-smoke budget cannot cover the requested work."""


class SessionLedger:
    def __init__(self, path: str) -> None:
        self.path = path
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)
        if not os.path.exists(path):
            self._write({
                "schema": LEDGER_SCHEMA,
                "limits": dict(LIMITS),
                "reserved_for_resume": dict(RESERVED_FOR_RESUME),
                "cpu_learned_smoke_seconds": 0.0,
                "cpu_optimizer_updates": 0,
                "gpu_sessions": 0,
                "gpu_wall_seconds": 0.0,
                "gpu_optimizer_updates": 0,
                "history": [],
            })

    def _read(self) -> dict[str, Any]:
        with open(self.path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def _write(self, state: dict[str, Any]) -> None:
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(state, handle, indent=2, sort_keys=True)
        os.replace(tmp, self.path)

    def remaining(self, device: str = "cpu") -> dict[str, float]:
        state = self._read()
        if device == "cpu":
            return {"seconds": LIMITS["cpu_learned_smoke_seconds"] - state["cpu_learned_smoke_seconds"],
                    "updates": LIMITS["cpu_optimizer_updates"] - state["cpu_optimizer_updates"]}
        return {"seconds": LIMITS["gpu_wall_seconds"] - state["gpu_wall_seconds"],
                "updates": LIMITS["gpu_optimizer_updates"] - state["gpu_optimizer_updates"],
                "sessions": LIMITS["gpu_sessions"] - state["gpu_sessions"]}

    def require_learned_allowance(self, *, updates: int, seconds: float,
                                  what: str = "learned work") -> None:
        """Hard pre-execution gate for ANY learned entry point (F6).

        Subprocess probes and helper agents must call this before spending
        optimizer updates; being launched outside ``train --smoke`` does not
        bypass the shared allowance. The ledger may already be over cap, in
        which case every learned request refuses.
        """
        state = self._read()
        remaining_updates = LIMITS["cpu_optimizer_updates"] - state["cpu_optimizer_updates"]
        remaining_seconds = LIMITS["cpu_learned_smoke_seconds"] - state["cpu_learned_smoke_seconds"]
        if remaining_updates < updates or remaining_seconds < seconds:
            raise SmokeBudgetExhausted(
                f"{what} refused: requested {updates} updates / {seconds}s but the "
                f"shared ledger has {remaining_updates} updates / "
                f"{remaining_seconds:.1f}s remaining "
                f"(used {state['cpu_optimizer_updates']}/{LIMITS['cpu_optimizer_updates']}); "
                "a new owner allocation is required")

    def check_can_run(self, *, device: str, updates: int, seconds: float,
                      reserve_for_resume: bool = False) -> None:
        remaining = self.remaining(device)
        reserve = RESERVED_FOR_RESUME if reserve_for_resume and device == "cpu" \
            else {"cpu_learned_smoke_seconds": 0, "cpu_optimizer_updates": 0}
        if remaining["updates"] - reserve.get("cpu_optimizer_updates", 0) < updates:
            raise SmokeBudgetExhausted(
                f"optimizer-update budget cannot cover {updates} updates "
                f"(remaining {remaining['updates']}, resume reserve "
                f"{reserve.get('cpu_optimizer_updates', 0)})")
        if remaining["seconds"] - reserve.get("cpu_learned_smoke_seconds", 0) < seconds:
            raise SmokeBudgetExhausted(
                f"smoke-second budget cannot cover {seconds}s "
                f"(remaining {remaining['seconds']:.0f})")

    def record(self, *, device: str, updates: int, seconds: float, what: str,
               evidence: str | None = None) -> None:
        state = self._read()
        state["history"].append({
            "at_unix": time.time(), "device": device, "updates": int(updates),
            "seconds": round(float(seconds), 3), "what": what,
            "evidence": evidence,
        })
        if device == "cpu":
            state["cpu_learned_smoke_seconds"] += float(seconds)
            state["cpu_optimizer_updates"] += int(updates)
        else:
            state["gpu_wall_seconds"] += float(seconds)
            state["gpu_optimizer_updates"] += int(updates)
            state["gpu_sessions"] += 1
        self._write(state)

    def snapshot(self) -> dict[str, Any]:
        return self._read()
