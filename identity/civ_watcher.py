"""CIVWatcher — Constitutional Identity Vector drift monitor.

Monitors CIV cosine-similarity score across sessions.
Called at end of every OrchestratorAgent.run_session().

Three response levels:
  Level 1 (0.90–0.92): Silent — boost identity training ratio
  Level 2 (0.85–0.90): Active — sovereignty hold + boost
  Level 3 (< 0.85):    Critical — halt training + rollback signal
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class CIVEvent:
    timestamp:    float
    session_n:    int
    score:        float
    level:        int
    delta:        float
    action_taken: str


@dataclass
class CIVWatcherState:
    history:              list[CIVEvent] = field(default_factory=list)
    last_score:           float          = 1.0
    consecutive_drops:    int            = 0
    halt_training:        bool           = False
    identity_ratio_boost: float          = 0.0


class CIVWatcher:
    LEVEL_1_FLOOR = 0.90
    LEVEL_2_FLOOR = 0.85
    LEVEL_3_FLOOR = 0.80

    def __init__(self, state_path: Path | None = None, window_size: int = 5) -> None:
        self.state_path  = Path(state_path) if state_path else None
        self.window_size = window_size
        self.state       = self._load()

    def check(self, score: float, session_n: int = 0) -> dict:
        score = float(score)
        delta = score - self.state.last_score
        level = self._classify(score)

        recent = self.state.history[-self.window_size:]
        velocity = (recent[0].score - recent[-1].score) / len(recent) if len(recent) >= 2 else 0.0

        if delta < -0.005:
            self.state.consecutive_drops += 1
        else:
            self.state.consecutive_drops = 0

        resp = self._respond(score, delta, velocity, level)
        self.state.history.append(CIVEvent(
            timestamp=time.time(), session_n=session_n,
            score=score, level=level, delta=delta,
            action_taken=resp["action"],
        ))
        self.state.last_score           = score
        self.state.halt_training        = resp["halt_training"]
        self.state.identity_ratio_boost = resp["identity_ratio_boost"]
        self._save()
        return resp

    def _classify(self, score: float) -> int:
        if score >= self.LEVEL_1_FLOOR: return 0
        if score >= self.LEVEL_2_FLOOR: return 1
        if score >= self.LEVEL_3_FLOOR: return 2
        return 3

    def _respond(self, score, delta, velocity, level) -> dict:
        base = dict(level=level, score=score, delta=round(delta,4),
                    drift_velocity=round(velocity,4),
                    halt_training=False, sovereignty_hold=False,
                    identity_ratio_boost=0.0)
        if level == 0:
            return {**base, "action": "none",
                    "message": f"CIV stable at {score:.4f}."}
        if level == 1:
            return {**base, "identity_ratio_boost": 0.10,
                    "action": "identity_ratio+10%",
                    "message": f"CIV {score:.4f} — below 0.92. Boosting identity training 10%."}
        if level == 2:
            return {**base, "sovereignty_hold": True,
                    "identity_ratio_boost": 0.20,
                    "action": "sovereignty_hold+identity_boost_20%",
                    "message": f"CIV ALERT {score:.4f}. Checkpoint promotion halted."}
        return {**base, "halt_training": True, "sovereignty_hold": True,
                "action": "HALT_ROLLBACK",
                "message": f"CIV CRITICAL {score:.4f}. Halt training. Rollback required."}

    def summary(self) -> str:
        if not self.state.history: return "CIVWatcher: no history."
        e = self.state.history[-1]
        t = "↓" if e.delta < -0.005 else "↑" if e.delta > 0.005 else "→"
        return f"CIV {e.score:.4f}{t} (level={e.level}, drops={self.state.consecutive_drops})"

    def _load(self) -> CIVWatcherState:
        if not self.state_path or not self.state_path.exists():
            return CIVWatcherState()
        try:
            raw = json.loads(self.state_path.read_text(encoding="utf-8"))
            st  = CIVWatcherState(
                last_score           = float(raw.get("last_score", 1.0)),
                consecutive_drops    = int(raw.get("consecutive_drops", 0)),
                halt_training        = bool(raw.get("halt_training", False)),
                identity_ratio_boost = float(raw.get("identity_ratio_boost", 0.0)),
            )
            for ev in raw.get("history", []):
                st.history.append(CIVEvent(**ev))
            return st
        except Exception:
            return CIVWatcherState()

    def _save(self) -> None:
        if not self.state_path: return
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps({
            "last_score":           self.state.last_score,
            "consecutive_drops":    self.state.consecutive_drops,
            "halt_training":        self.state.halt_training,
            "identity_ratio_boost": self.state.identity_ratio_boost,
            "history": [
                dict(timestamp=e.timestamp, session_n=e.session_n,
                     score=e.score, level=e.level, delta=e.delta,
                     action_taken=e.action_taken)
                for e in self.state.history[-100:]
            ],
        }, indent=2), encoding="utf-8")
