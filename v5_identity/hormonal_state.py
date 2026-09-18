"""External hormonal state for the HORM-001 V5 sibling (heuristic, out-of-graph)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

HORMONES = (
    "dopamine",
    "cortisol",
    "serotonin",
    "adrenaline",
    "oxytocin",
    "gaba",
    "norepinephrine",
)

BASELINES = {
    "dopamine": 0.10,
    "cortisol": 0.10,
    "serotonin": 0.50,
    "adrenaline": 0.05,
    "oxytocin": 0.10,
    "gaba": 0.10,
    "norepinephrine": 0.05,
}

DECAY_RATES = {
    "dopamine": 0.05,
    "cortisol": 0.02,
    "serotonin": 0.01,
    "adrenaline": 0.40,
    "oxytocin": 0.03,
    "gaba": 0.04,
    "norepinephrine": 0.30,
}

BOUNDS = (0.0, 2.0)


@dataclass
class HormonalState:
    values: dict[str, float]

    def __post_init__(self) -> None:
        if tuple(self.values) != HORMONES:
            raise ValueError("hormonal values must carry exactly the seven hormones in order")
        for name, value in self.values.items():
            if not BOUNDS[0] <= float(value) <= BOUNDS[1]:
                raise ValueError(f"hormone {name} outside bounds")

    @classmethod
    def baseline(cls) -> "HormonalState":
        return cls(values={name: BASELINES[name] for name in HORMONES})

    def appraise(self, verified_outcome: str) -> None:
        """Apply fixed deltas for verified verifier outcomes (never model self-report)."""

        if verified_outcome not in ("success", "failure"):
            raise ValueError("appraise accepts only verified success/failure outcomes")
        delta = 0.20
        if verified_outcome == "success":
            self.values["dopamine"] = self._clip(self.values["dopamine"] + delta)
            self.values["serotonin"] = self._clip(self.values["serotonin"] + delta / 4)
        else:
            self.values["cortisol"] = self._clip(self.values["cortisol"] + delta)
            self.values["adrenaline"] = self._clip(self.values["adrenaline"] + delta)

    def decay(self) -> None:
        """Decay each hormone toward its baseline; cortisol inherits adrenaline."""

        previous_adrenaline = self.values["adrenaline"]
        for name in HORMONES:
            rate = DECAY_RATES[name]
            value = self.values[name] + (BASELINES[name] - self.values[name]) * rate
            if name == "cortisol":
                value += 0.32 * previous_adrenaline
            self.values[name] = self._clip(value)

    @staticmethod
    def _clip(value: float) -> float:
        return min(BOUNDS[1], max(BOUNDS[0], float(value)))

    def vector(self) -> tuple[float, ...]:
        return tuple(self.values[name] for name in HORMONES)

    def sha256(self) -> str:
        payload = json.dumps(self.values, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def to_dict(self) -> dict[str, object]:
        """Serialize session state for checkpointing (HORM-004 prerequisite)."""

        return {
            "schema": "anra-hormonal-state/v1",
            "values": {name: self.values[name] for name in HORMONES},
            "sha256": self.sha256(),
        }

    @classmethod
    def from_dict(cls, value: object) -> "HormonalState":
        """Restore session state; fail closed on schema, shape, or hash drift."""

        if not isinstance(value, dict):
            raise ValueError("hormonal session state must be a mapping")
        if value.get("schema") != "anra-hormonal-state/v1":
            raise ValueError("unsupported hormonal session-state schema")
        values = value.get("values")
        if not isinstance(values, dict) or set(values) != set(HORMONES):
            raise ValueError("hormonal session state must carry exactly the seven hormones")
        state = cls(values={name: float(values[name]) for name in HORMONES})
        if value.get("sha256") != state.sha256():
            raise ValueError("hormonal session-state hash mismatch; refusing resume")
        return state
