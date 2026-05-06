from __future__ import annotations

from pathlib import Path
import json


class DynamicRegretScheduler:
    """Session-aware learning-rate controller using the Besbes-Gur-Zeevi rate."""

    def __init__(
        self,
        optimizer=None,
        eta_base: float | None = None,
        *,
        base_lr: float | None = None,
        min_lr: float = 1e-5,
        max_lr: float = 3e-3,
        session_file: str | Path | None = None,
        warmup_sessions: int = 5,
        min_multiplier: float = 0.3,
    ) -> None:
        if optimizer is not None and not hasattr(optimizer, "param_groups"):
            if base_lr is None and eta_base is None:
                base_lr = float(optimizer)
            optimizer = None

        self.optimizer = optimizer
        self.eta_base = float(eta_base if eta_base is not None else (base_lr if base_lr is not None else 3e-4))
        self.min_lr = float(min_lr)
        self.max_lr = float(max_lr)
        self.session_file = Path(session_file) if session_file is not None else None
        self.warmup_sessions = int(warmup_sessions)
        self.min_multiplier = float(min_multiplier)
        self.session_count = 0
        self.V_total = 0.0
        self.T_total = 0
        self.session_start_loss: float | None = None
        self._current_lr = self.eta_base

        if self.session_file is not None:
            self.load(self.session_file)
        self._apply_lr(self._current_lr)

    def _clip(self, lr: float) -> float:
        return max(self.min_lr, min(self.max_lr, float(lr)))

    def _apply_lr(self, lr: float) -> float:
        self._current_lr = self._clip(lr)
        if self.optimizer is not None:
            for group in self.optimizer.param_groups:
                group["lr"] = self._current_lr
        return self._current_lr

    def current_lr(self) -> float:
        return self._current_lr

    def session_start(self, loss: float) -> None:
        self.session_start_loss = float(loss)

    def session_end(self, loss: float, steps: int) -> float:
        steps = max(0, int(steps))
        if steps == 0:
            self.session_count += 1
            return self._apply_lr(self.eta_base if self.T_total == 0 else self._besbes_lr())

        start = self.session_start_loss if self.session_start_loss is not None else float(loss)
        variation = abs(float(start) - float(loss))
        self.V_total += variation
        self.T_total += steps
        self.session_count += 1
        return self._apply_lr(self._besbes_lr())

    def _besbes_lr(self) -> float:
        if self.T_total <= 0 or self.V_total <= 0.0:
            return self.eta_base
        if self.session_count <= self.warmup_sessions:
            return self.eta_base
        raw = self.eta_base * ((self.V_total / self.T_total) ** (1.0 / 3.0))
        floored = max(raw, self.eta_base * self.min_multiplier)
        return float(max(self.min_lr, min(self.max_lr, floored)))

    def update(self, reward: float) -> float:
        regret = max(0.0, 1.0 - float(reward))
        self.V_total += regret
        self.T_total += 1
        return self._apply_lr(self._besbes_lr())

    def state_dict(self) -> dict[str, object]:
        return {
            "eta_base": self.eta_base,
            "min_lr": self.min_lr,
            "max_lr": self.max_lr,
            "V_total": self.V_total,
            "T_total": self.T_total,
            "current_lr": self._current_lr,
            "session_count": self.session_count,
            "warmup_sessions": self.warmup_sessions,
            "min_multiplier": self.min_multiplier,
        }

    def load_state_dict(self, payload: dict) -> None:
        self.eta_base = float(payload.get("eta_base", self.eta_base))
        self.min_lr = float(payload.get("min_lr", self.min_lr))
        self.max_lr = float(payload.get("max_lr", self.max_lr))
        self.V_total = float(payload.get("V_total", self.V_total))
        self.T_total = int(payload.get("T_total", self.T_total))
        self.session_count = int(payload.get("session_count", self.session_count))
        self.warmup_sessions = int(payload.get("warmup_sessions", self.warmup_sessions))
        self.min_multiplier = float(payload.get("min_multiplier", self.min_multiplier))
        self._current_lr = self._clip(float(payload.get("current_lr", self._besbes_lr())))

    def load(self, path: str | Path | None = None) -> None:
        target = Path(path) if path is not None else self.session_file
        if target is None or not target.exists():
            return
        try:
            payload = json.loads(target.read_text(encoding="utf-8"))
            self.load_state_dict(payload)
        except Exception as exc:
            print(f"[dynamic_regret] failed loading state: {exc}")

    def save(self, path: str | Path | None = None) -> None:
        target = Path(path) if path is not None else self.session_file
        if target is None:
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.state_dict(), indent=2), encoding="utf-8")
