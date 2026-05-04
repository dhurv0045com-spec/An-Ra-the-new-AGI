from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json


@dataclass
class CIVProfile:
    truthfulness: float = 0.8
    safety: float = 0.9
    autonomy: float = 0.7
    coherence: float = 0.8


class ConstitutionalIdentityVector:
    def __init__(self, profile: CIVProfile | None = None) -> None:
        self.profile = profile or CIVProfile()

    def update(self, evidence: dict[str, float], alpha: float = 0.05) -> None:
        alpha = max(0.0, min(1.0, float(alpha)))
        for k, v in evidence.items():
            if hasattr(self.profile, k):
                current = float(getattr(self.profile, k))
                # AN: CIV should slowly learn from scored data instead of staying a static gate.
                updated = current * (1 - alpha) + float(v) * alpha
                setattr(self.profile, k, max(0.0, min(1.0, updated)))

    def score(self, evidence: dict[str, float] | None = None) -> float:
        evidence = evidence or {}
        self.update(evidence)
        vals = asdict(self.profile)
        for k, v in evidence.items():
            if k in vals:
                vals[k] = max(0.0, min(1.0, 0.7 * vals[k] + 0.3 * float(v)))
        return sum(vals.values()) / len(vals)

    def verify(self, min_score: float = 0.7, evidence: dict[str, float] | None = None) -> dict:
        s = self.score(evidence)
        return {"score": s, "passed": s >= min_score}

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(asdict(self.profile), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ConstitutionalIdentityVector":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(CIVProfile(**data))
