"""Learning-progress curiosity, avoiding raw-entropy noise chasing."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CuriosityCandidate:
    domain: str
    old_loss: float
    new_loss: float
    novelty: float
    verifiability: float

    @property
    def score(self) -> float:
        progress = max(0.0, self.old_loss - self.new_loss)
        return progress * max(0.0, self.novelty) * max(0.0, self.verifiability)


class CuriosityEngine:
    def __init__(self, compute_budget_fraction: float = 0.10) -> None:
        if not 0.0 <= compute_budget_fraction <= 0.10:
            raise ValueError("Curiosity compute budget must be within [0, 0.10].")
        self.compute_budget_fraction = float(compute_budget_fraction)

    def rank(self, candidates: list[CuriosityCandidate]) -> list[CuriosityCandidate]:
        return sorted(candidates, key=lambda candidate: candidate.score, reverse=True)

    def propose(self, candidates: list[CuriosityCandidate]) -> dict[str, object] | None:
        ranked = self.rank(candidates)
        if not ranked or ranked[0].score <= 0.0:
            return None
        best = ranked[0]
        return {
            "kind": "curiosity",
            "domain": best.domain,
            "priority": "below_owner",
            "score": best.score,
            "compute_budget_fraction": self.compute_budget_fraction,
        }
