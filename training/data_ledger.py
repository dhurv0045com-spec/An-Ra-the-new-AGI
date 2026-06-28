"""Data Entropy Ledger with band-pass difficulty and provenance scoring."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DataQuality:
    difficulty_percentile: float
    novelty: float
    provenance: float
    verification: float
    identity_relevance: float
    license_score: float


class DataEntropyLedger:
    WEIGHTS = {
        "difficulty": 0.25,
        "novelty": 0.20,
        "provenance": 0.15,
        "verification": 0.20,
        "identity": 0.15,
        "license": 0.05,
    }

    def __init__(self, threshold: float = 0.65) -> None:
        self.threshold = float(threshold)
        self.accepted = 0
        self.rejected = 0

    @staticmethod
    def band_pass(percentile: float, center: float = 0.5, width: float = 0.3) -> float:
        distance = abs(float(percentile) - center)
        return max(0.0, 1.0 - distance / max(width, 1e-9))

    def score(self, quality: DataQuality) -> float:
        if quality.license_score <= 0.0:
            return 0.0
        value = (
            self.WEIGHTS["difficulty"] * self.band_pass(quality.difficulty_percentile)
            + self.WEIGHTS["novelty"] * quality.novelty
            + self.WEIGHTS["provenance"] * quality.provenance
            + self.WEIGHTS["verification"] * quality.verification
            + self.WEIGHTS["identity"] * quality.identity_relevance
            + self.WEIGHTS["license"] * quality.license_score
        )
        return max(0.0, min(1.0, value))

    def evaluate(self, quality: DataQuality) -> tuple[bool, float]:
        score = self.score(quality)
        accepted = score >= self.threshold
        if accepted:
            self.accepted += 1
        else:
            self.rejected += 1
        return accepted, score

    def report(self) -> dict[str, object]:
        total = self.accepted + self.rejected
        return {
            "threshold": self.threshold,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "acceptance_rate": self.accepted / total if total else 0.0,
            "weights": dict(self.WEIGHTS),
        }
