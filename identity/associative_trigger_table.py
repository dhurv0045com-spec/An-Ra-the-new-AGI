from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import time
from pathlib import Path
from typing import Any


@dataclass
class ATTPreset:
    preset_id: str
    hormone_conditions: dict[str, tuple[float | None, float | None]] = field(default_factory=dict)
    domain: str = "*"
    task_type: str = "*"
    hormone_deltas: dict[str, float] = field(default_factory=dict)
    behavior: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    evidence_count: int = 0
    updated_at: float = field(default_factory=time.time)

    def matches(self, hormones: dict[str, float], domain: str = "", task_type: str = "") -> bool:
        if self.domain not in {"*", "", domain}:
            return False
        if self.task_type not in {"*", "", task_type}:
            return False
        for name, bounds in self.hormone_conditions.items():
            value = float(hormones.get(name, 0.0))
            low, high = bounds
            if low is not None and value < low:
                return False
            if high is not None and value > high:
                return False
        return True

    def specificity(self) -> int:
        score = len(self.hormone_conditions)
        score += 1 if self.domain not in {"*", ""} else 0
        score += 1 if self.task_type not in {"*", ""} else 0
        return score


class AssociativeTriggerTable:
    """Lookup behavioral presets from hormone, domain, and task context."""

    def __init__(self, presets: list[ATTPreset] | None = None, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else None
        self.presets: list[ATTPreset] = presets or self._default_presets()
        if self.path is not None and self.path.exists():
            self.load(self.path)

    def _default_presets(self) -> list[ATTPreset]:
        return [
            ATTPreset(
                preset_id="threat_quantum_verification",
                hormone_conditions={"cortisol": (0.6, None)},
                domain="quantum",
                task_type="verification",
                behavior={
                    "require_citation_grounding": True,
                    "generation_temperature_delta": -0.15,
                    "memory_salience_delta": 0.15,
                    "ouroboros_weight_bias": [0.0, 0.05, 0.25],
                },
            ),
            ATTPreset(
                preset_id="focused_constraint_solve",
                hormone_conditions={"norepinephrine": (0.45, None)},
                task_type="constraint_solve",
                behavior={
                    "prefer_constraint_json": True,
                    "generation_temperature_delta": -0.05,
                    "ouroboros_weight_bias": [-0.05, 0.2, 0.05],
                },
            ),
            ATTPreset(
                preset_id="flow_failure_replay",
                hormone_conditions={"endorphin": (0.45, None), "dopamine": (0.4, None)},
                task_type="failure_replay",
                behavior={
                    "write_replay": True,
                    "generation_temperature_delta": 0.05,
                    "memory_salience_delta": -0.1,
                },
            ),
        ]

    def lookup(self, hormones: dict[str, float], domain: str = "", task_type: str = "") -> ATTPreset | None:
        matches = [p for p in self.presets if p.matches(hormones, domain=domain, task_type=task_type)]
        if not matches:
            return None
        matches.sort(key=lambda p: (p.specificity(), p.confidence, p.evidence_count), reverse=True)
        return matches[0]

    def learn(self, preset: ATTPreset) -> None:
        existing = [p for p in self.presets if p.preset_id != preset.preset_id]
        preset.updated_at = time.time()
        self.presets = [*existing, preset]
        if self.path is not None:
            self.save(self.path)

    def record_outcome(self, preset_id: str, reward: float) -> None:
        for preset in self.presets:
            if preset.preset_id == preset_id:
                preset.evidence_count += 1
                # AN: ATT should learn from local verifier feedback without becoming an opaque model.
                preset.confidence = max(0.0, min(1.0, 0.9 * preset.confidence + 0.1 * float(reward)))
                preset.updated_at = time.time()
                break
        if self.path is not None:
            self.save(self.path)

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for preset in self.presets:
            row = asdict(preset)
            row["hormone_conditions"] = {k: list(v) for k, v in preset.hormone_conditions.items()}
            rows.append(row)
        p.write_text(json.dumps({"presets": rows}, indent=2), encoding="utf-8")

    def load(self, path: str | Path) -> None:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        presets: list[ATTPreset] = []
        for row in data.get("presets", []):
            conditions = {
                str(k): (v[0], v[1])
                for k, v in dict(row.get("hormone_conditions", {})).items()
            }
            presets.append(
                ATTPreset(
                    preset_id=str(row["preset_id"]),
                    hormone_conditions=conditions,
                    domain=str(row.get("domain", "*")),
                    task_type=str(row.get("task_type", "*")),
                    hormone_deltas=dict(row.get("hormone_deltas", {})),
                    behavior=dict(row.get("behavior", {})),
                    confidence=float(row.get("confidence", 1.0)),
                    evidence_count=int(row.get("evidence_count", 0)),
                    updated_at=float(row.get("updated_at", time.time())),
                )
            )
        self.presets = presets

