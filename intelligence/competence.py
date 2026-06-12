"""Calibrated domain competence estimates for planning policy selection."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time


@dataclass
class DomainCompetence:
    accuracy: float
    calibration: float
    verifier_coverage: float
    updated_at: float
    samples: int

    def score(self, *, now: float, half_life_days: float) -> float:
        age_days = max(0.0, now - self.updated_at) / 86400.0
        recency = 0.5 ** (age_days / max(half_life_days, 1e-6))
        evidence = 1.0 - math.exp(-self.samples / 20.0)
        return (
            0.45 * self.accuracy
            + 0.25 * self.calibration
            + 0.20 * self.verifier_coverage
            + 0.10 * recency
        ) * evidence


class CalibratedCompetenceModel:
    def __init__(self, half_life_days: float = 30.0) -> None:
        self.half_life_days = float(half_life_days)
        self.domains: dict[str, DomainCompetence] = {}

    def update(
        self,
        domain: str,
        *,
        correct: bool,
        confidence: float,
        verified: bool,
        timestamp: float | None = None,
    ) -> DomainCompetence:
        now = time.time() if timestamp is None else float(timestamp)
        current = self.domains.get(
            domain, DomainCompetence(0.0, 0.0, 0.0, now, 0)
        )
        n = current.samples + 1
        accuracy = (current.accuracy * current.samples + float(correct)) / n
        calibration_event = 1.0 - abs(float(confidence) - float(correct))
        calibration = (current.calibration * current.samples + calibration_event) / n
        coverage = (current.verifier_coverage * current.samples + float(verified)) / n
        updated = DomainCompetence(accuracy, calibration, coverage, now, n)
        self.domains[domain] = updated
        return updated

    def score(self, domain: str, *, now: float | None = None) -> float:
        if domain not in self.domains:
            return 0.0
        return self.domains[domain].score(
            now=time.time() if now is None else float(now),
            half_life_days=self.half_life_days,
        )

    def policy(self, domain: str) -> str:
        score = self.score(domain)
        if score >= 0.80:
            return "direct"
        if score >= 0.60:
            return "verify"
        if score >= 0.35:
            return "retrieve_and_decompose"
        return "research_or_clarify"
