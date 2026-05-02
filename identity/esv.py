from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json


@dataclass
class EmotionalState:
    calm: float = 0.7
    focus: float = 0.8
    curiosity: float = 0.8
    stress: float = 0.2


class EmotionalStateVector:
    def __init__(self, state: EmotionalState | None = None) -> None:
        self.state = state or EmotionalState()

    def update(self, *, success: bool, difficulty: float = 0.5) -> EmotionalState:
        d = max(0.0, min(1.0, float(difficulty)))
        if success:
            self.state.calm = min(1.0, self.state.calm + 0.05 * (1 - d))
            self.state.focus = min(1.0, self.state.focus + 0.04)
            self.state.stress = max(0.0, self.state.stress - 0.08)
        else:
            self.state.stress = min(1.0, self.state.stress + 0.12 * (1 + d))
            self.state.focus = max(0.0, self.state.focus - 0.05)
        self.state.curiosity = min(1.0, max(0.0, self.state.curiosity + (0.02 if success else -0.01)))
        return self.state

    def as_dict(self) -> dict:
        return asdict(self.state)

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.as_dict(), indent=2), encoding="utf-8")
