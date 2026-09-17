"""HORM-001 experimental line: bounded hormonal state for the V5 architecture.

Legible, fail-closed hormone-analog state held entirely outside the autograd
graph. Hormones live in [0, 1]; appraisal consumes ``VerifiedOutcome`` events
(never the model's own confidence) plus bounded direct deltas; serotonin
dampens stress deltas; the log is capped. The only forward-pass effect runs
through ``v5_identity.hormonal_projection`` (dopamine, cortisol, serotonin,
adrenaline channels; oxytocin/GABA/norepinephrine are state-only).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Mapping

HORMONE_NAMES = (
    "dopamine",
    "cortisol",
    "serotonin",
    "adrenaline",
    "oxytocin",
    "gaba",
    "norepinephrine",
)

# (baseline, decay_rate_per_step, lower, upper) per hormone.
HORMONE_DYNAMICS: dict[str, tuple[float, float, float, float]] = {
    #              baseline  decay   low  high
    "dopamine":     (0.10, 0.050, 0.0, 1.0),
    "cortisol":     (0.10, 0.020, 0.0, 1.0),
    "serotonin":    (0.50, 0.010, 0.0, 1.0),
    "adrenaline":   (0.05, 0.200, 0.0, 1.0),
    "oxytocin":     (0.10, 0.030, 0.0, 1.0),
    "gaba":         (0.10, 0.040, 0.0, 1.0),
    "norepinephrine": (0.05, 0.150, 0.0, 1.0),
}

# Adrenaline's decay leaves a proportional cortisol hangover (V4 prior art).
CORTISOL_HANGOVER_FROM_ADRENALINE = 0.32

HISTORY_LIMIT = 64

# Verified-outcome appraisal deltas (v1 contract, PLAN.md-binding).
OUTCOME_DELTAS = {
    "success": {"dopamine": 0.25},
    "failure": {"cortisol": 0.15},
    "surprise": {"adrenaline": 0.6},
    "coherence": {"serotonin": 0.10},
    "unresolved_stress": {"gaba": 0.10, "adrenaline": -0.1},
}

# Stress-shaped kinds whose deltas are dampened by serotonin.
STRESS_KINDS = ("failure", "surprise", "unresolved_stress")


@dataclass(frozen=True, slots=True)
class VerifiedOutcome:
    """A measured outcome from the verifier system — never a self-report."""

    kind: str
    source: str
    magnitude: float = 1.0

    def __post_init__(self) -> None:
        if self.kind not in OUTCOME_DELTAS:
            raise ValueError(
                f"unknown outcome kind {self.kind!r}; must be one of {sorted(OUTCOME_DELTAS)}")
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("verified outcomes must carry a non-empty provenance source")
        if not math.isfinite(self.magnitude) or self.magnitude < 0.0:
            raise ValueError("magnitude must be finite and >= 0")

    def deltas(self) -> dict[str, float]:
        return {name: value * self.magnitude for name, value in OUTCOME_DELTAS[self.kind].items()}


@dataclass(frozen=True, slots=True)
class HALState:
    """The seven hormone analogs plus the step counter. Pure value type."""

    dopamine: float = HORMONE_DYNAMICS["dopamine"][0]
    cortisol: float = HORMONE_DYNAMICS["cortisol"][0]
    serotonin: float = HORMONE_DYNAMICS["serotonin"][0]
    adrenaline: float = HORMONE_DYNAMICS["adrenaline"][0]
    oxytocin: float = HORMONE_DYNAMICS["oxytocin"][0]
    gaba: float = HORMONE_DYNAMICS["gaba"][0]
    norepinephrine: float = HORMONE_DYNAMICS["norepinephrine"][0]
    step: int = 0
    log: tuple[Mapping[str, float], ...] = field(default=())

    def __post_init__(self) -> None:
        for name in HORMONE_NAMES:
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"hormone {name} must be finite and within [0, 1], got {value}")
            object.__setattr__(self, name, value)
        if isinstance(self.step, bool) or not isinstance(self.step, int) or self.step < 0:
            raise ValueError("step must be a nonnegative integer")

    def as_dict(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in HORMONE_NAMES}

    def with_delta(self, name: str, delta: float) -> "HALState":
        if name not in HORMONE_DYNAMICS:
            raise KeyError(f"unknown hormone: {name}")
        delta = float(delta)
        if not math.isfinite(delta):
            raise ValueError(f"delta for {name} must be finite, got {delta}")
        low, high = HORMONE_DYNAMICS[name][2], HORMONE_DYNAMICS[name][3]
        value = getattr(self, name) + delta
        return replace(self, **{name: min(high, max(low, value))})

    def decay(self) -> "HALState":
        """One decay step toward each baseline at its own rate; returns new state."""
        levels: dict[str, float] = {}
        for name, (baseline, rate, low, high) in HORMONE_DYNAMICS.items():
            value = getattr(self, name)
            value = baseline + (value - baseline) * (1.0 - rate)
            levels[name] = min(high, max(low, value))
        # Cortisol hangover: adrenaline decay feeds cortisol (V4 prior art).
        hangover = CORTISOL_HANGOVER_FROM_ADRENALINE * (
            getattr(self, "adrenaline") - levels["adrenaline"]
        )
        if hangover > 0.0:
            low, high = HORMONE_DYNAMICS["cortisol"][2], HORMONE_DYNAMICS["cortisol"][3]
            levels["cortisol"] = min(high, levels["cortisol"] + hangover)
        return replace(
            self, **levels, step=self.step + 1,
            log=(self.log + ({**levels, "step": float(self.step + 1)},))[-HISTORY_LIMIT:],
        )

    def appraise(self, *outcomes: VerifiedOutcome, **deltas: float) -> "HALState":
        """Apply measured-outcome events and/or direct bounded hormone deltas.

        Unknown signals raise; non-finite deltas raise; stress-shaped deltas
        are dampened by current serotonin (scale ``1 - 0.5 * serotonin``).
        """
        unknown = set(deltas) - set(HORMONE_DYNAMICS)
        if unknown:
            raise KeyError(f"unknown appraisal signals: {sorted(unknown)}")
        for name, delta in deltas.items():
            if not math.isfinite(delta):
                raise ValueError(f"appraisal delta for {name} must be finite, got {delta}")
        state = self
        damp = 1.0 - 0.5 * self.serotonin
        for outcome in outcomes:
            for name, value in outcome.deltas().items():
                if outcome.kind in STRESS_KINDS:
                    value = value * damp
                state = state.with_delta(name, value)
        for name, delta in deltas.items():
            state = state.with_delta(name, delta)
        return state

    def summary(self) -> str:
        return " ".join(f"{name}={getattr(self, name):.3f}" for name in HORMONE_NAMES)

    def as_receipt(self) -> dict[str, Any]:
        return {"step": self.step, **self.as_dict()}


def neutral_state() -> HALState:
    return HALState()


__all__ = [
    "HORMONE_NAMES",
    "HORMONE_DYNAMICS",
    "CORTISOL_HANGOVER_FROM_ADRENALINE",
    "HISTORY_LIMIT",
    "OUTCOME_DELTAS",
    "VerifiedOutcome",
    "HALState",
    "neutral_state",
]
