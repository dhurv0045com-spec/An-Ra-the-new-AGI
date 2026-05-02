from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json


@dataclass
class DynamicRegretScheduler:
    """Session-aware LR scaling using moving regret.

    Regret is interpreted as `target_reward - observed_reward`.
    Positive regret increases learning pressure, low/negative regret decays LR.
    """

    base_lr: float
    min_scale: float = 0.25
    max_scale: float = 2.0
    momentum: float = 0.9
    target_reward: float = 1.0
    session_file: Path | None = None
    ema_regret: float = 0.0
    history: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.session_file and self.session_file.exists():
            try:
                payload = json.loads(self.session_file.read_text(encoding="utf-8"))
                self.ema_regret = float(payload.get("ema_regret", self.ema_regret))
                self.history = [float(x) for x in payload.get("history", [])][-2048:]
            except Exception as exc:
                print(f"[dynamic_regret] failed loading state: {exc}")

    def update(self, reward: float) -> float:
        regret = self.target_reward - float(reward)
        self.ema_regret = self.momentum * self.ema_regret + (1.0 - self.momentum) * regret
        self.history.append(regret)
        if len(self.history) > 4096:
            self.history = self.history[-4096:]
        return self.current_lr()

    def scale(self) -> float:
        # map ema regret to bounded scale factor
        s = 1.0 + self.ema_regret
        if s < self.min_scale:
            return self.min_scale
        if s > self.max_scale:
            return self.max_scale
        return s

    def current_lr(self) -> float:
        return float(self.base_lr) * self.scale()

    def state_dict(self) -> dict:
        return {
            "base_lr": self.base_lr,
            "min_scale": self.min_scale,
            "max_scale": self.max_scale,
            "momentum": self.momentum,
            "target_reward": self.target_reward,
            "ema_regret": self.ema_regret,
            "history": self.history[-2048:],
        }

    def save(self) -> None:
        if self.session_file is None:
            return
        self.session_file.parent.mkdir(parents=True, exist_ok=True)
        self.session_file.write_text(json.dumps(self.state_dict(), indent=2), encoding="utf-8")
