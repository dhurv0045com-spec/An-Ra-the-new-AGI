"""Verifier selection and consensus based on claim type."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class VerificationOutcome:
    passed: bool
    confidence: float
    verifier: str
    detail: str


class VerifierSearch:
    def __init__(self) -> None:
        self.verifiers: dict[str, Callable[[str], VerificationOutcome]] = {}
        self.routes: dict[str, tuple[str, ...]] = {
            "math": ("symbolic",),
            "logic": ("symbolic",),
            "code": ("code",),
            "fact": ("retrieval",),
            "robotics": ("simulation", "constraint"),
            "general": ("retrieval", "symbolic"),
        }

    def register(self, name: str, verifier: Callable[[str], VerificationOutcome]) -> None:
        self.verifiers[name] = verifier

    def verify(self, claim: str, claim_type: str = "general") -> VerificationOutcome:
        route = self.routes.get(claim_type, self.routes["general"])
        outcomes = [
            self.verifiers[name](claim)
            for name in route
            if name in self.verifiers
        ]
        if not outcomes:
            return VerificationOutcome(False, 0.0, "none", "No verifier available.")
        weighted = sum(item.confidence * float(item.passed) for item in outcomes)
        confidence = weighted / max(sum(item.confidence for item in outcomes), 1e-9)
        return VerificationOutcome(
            passed=confidence >= 0.5,
            confidence=confidence,
            verifier="+".join(item.verifier for item in outcomes),
            detail=" | ".join(item.detail for item in outcomes),
        )
