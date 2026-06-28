"""Budgeted multi-role self-debate with evidence-gated synthesis."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import asdict, dataclass

ROLES = ("evidence", "devils_advocate", "causal_reasoner", "uncertainty", "synthesis")


@dataclass(frozen=True)
class DebatePosition:
    role: str
    argument: str
    supporting_evidence: tuple[str, ...]
    weaknesses: tuple[str, ...]
    confidence: float
    unresolved_questions: tuple[str, ...]


@dataclass(frozen=True)
class DebateResult:
    triggered: bool
    actionable: bool
    risk_reasons: tuple[str, ...]
    positions: tuple[DebatePosition, ...]
    recommendation: str
    unresolved_uncertainty: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class DebateRiskClassifier:
    TERMS = {
        "high_stakes": ("medical", "legal", "financial", "safety"),
        "causal": ("cause", "effect", "intervention", "what if"),
        "ethical": ("ethical", "fair", "harm"),
        "recommendation": ("recommend", "should i", "best choice"),
        "contested": ("controversial", "disputed", "debate"),
    }

    def classify(self, task: str) -> tuple[str, ...]:
        lowered = task.lower()
        return tuple(
            name for name, terms in self.TERMS.items() if any(term in lowered for term in terms)
        )


class MultiAgentSelfDebate:
    def __init__(self, *, token_budget_per_role: int = 384, timeout_seconds: float = 20.0) -> None:
        self.classifier = DebateRiskClassifier()
        self.token_budget_per_role = int(token_budget_per_role)
        self.timeout_seconds = float(timeout_seconds)

    def run(
        self,
        task: str,
        generate_position: Callable[[str, str, int, int], DebatePosition],
        *,
        verify_claims: Callable[[DebatePosition], bool],
        verify_synthesis: Callable[[tuple[DebatePosition, ...]], bool],
    ) -> DebateResult:
        reasons = self.classifier.classify(task)
        if not reasons:
            return DebateResult(False, False, (), (), "", ())
        positions: list[DebatePosition] = []
        for role in ROLES:
            seed = int(hashlib.sha256(f"{task}:{role}".encode()).hexdigest()[:8], 16)
            position = generate_position(role, task, seed, self.token_budget_per_role)
            if position.role != role:
                raise ValueError(
                    f"Debate role isolation violated: expected {role}, got {position.role}"
                )
            positions.append(position)
        frozen = tuple(positions)
        evidence_passed = all(verify_claims(position) for position in frozen[:-1])
        synthesis_passed = evidence_passed and verify_synthesis(frozen)
        unresolved = tuple(
            question for position in frozen for question in position.unresolved_questions
        )
        recommendation = frozen[-1].argument if synthesis_passed else ""
        return DebateResult(True, synthesis_passed, reasons, frozen, recommendation, unresolved)
