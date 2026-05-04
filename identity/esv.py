from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json
import math

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover - torch-free environments still use EmotionalStateVector.
    torch = None
    nn = None


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


if nn is not None:
    class ESVModule(nn.Module):
        """Residual-stream emotional state vector predictor.

        The module reads the reserved ESV channel from the residual stream and
        exposes VAD controls used by attention, memory routing, and DGSA gates.
        Predictor weights use tiny random initialization so the system starts
        near neutral while gradients can flow immediately.
        """

        def __init__(self, d_model: int = 512, d_esv: int = 64) -> None:
            super().__init__()
            self.d_model = int(d_model)
            self.d_esv = int(d_esv)
            self.predictor = nn.Sequential(
                nn.Linear(self.d_esv, 3),
                nn.Tanh(),
            )
            for m in self.predictor.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0.0, std=0.01)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            self.register_buffer("state", torch.zeros(3))

        def forward(self, h):
            if h.ndim != 3:
                raise ValueError("ESVModule expects residual stream shape [batch, seq, d_model].")
            if h.shape[-1] < self.d_esv:
                raise ValueError(f"residual stream has {h.shape[-1]} channels, expected at least {self.d_esv}.")
            esv_channel = h[:, :, -self.d_esv :]
            pooled = esv_channel.mean(dim=(0, 1))
            state = self.predictor(pooled)
            self.state.copy_(state.detach())
            return state

        @property
        def valence(self) -> float:
            return float(self.state[0].item())

        @property
        def arousal(self) -> float:
            return float(self.state[1].item())

        @property
        def dominance(self) -> float:
            return float(self.state[2].item())

        def as_dict(self) -> dict[str, float]:
            return {
                "valence": self.valence,
                "arousal": self.arousal,
                "dominance": self.dominance,
            }

        def attention_temperature(self, tau0: float = 1.0) -> float:
            return float(tau0) * math.exp(-0.5 * self.arousal)

        def attention_temperature_tensor(self, state=None, tau0: float = 1.0):
            """Return a differentiable attention temperature from arousal."""
            state = self.state if state is None else state
            return float(tau0) * torch.exp(-0.5 * state[1]).clamp(0.25, 4.0)

        def memory_write_threshold(self, base: float = 0.5) -> float:
            threshold = float(base) - 0.15 * self.valence + 0.15 * self.arousal
            return max(0.01, min(0.99, threshold))

        def dgsa_gate(self) -> tuple[float, float]:
            att = 1.0 / (1.0 + math.exp(-self.dominance))
            return 1.0 - att, att
else:
    class ESVModule:  # pragma: no cover - exercised only without torch installed.
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("ESVModule requires torch.")
