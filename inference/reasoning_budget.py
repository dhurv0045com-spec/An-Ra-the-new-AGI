"""Transparent adaptive-compute plans for An-Ra inference.

The policy allocates retrieval, candidate generation, verification, and
revision budgets.  It does not claim that additional passes are useful before
evaluation and never executes tools or mutates model weights.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Literal

ReasoningMode = Literal["direct", "verify", "retrieve_decompose", "search_verify"]


@dataclass(frozen=True)
class ReasoningSignals:
    domain: str
    prompt_tokens: int
    estimated_difficulty: float
    competence: float
    verifier_available: bool
    retrieval_available: bool
    irreversible_action: bool = False


@dataclass(frozen=True)
class ReasoningBudget:
    schema_version: int
    mode: ReasoningMode
    candidate_count: int
    revision_count: int
    retrieval_queries: int
    verifier_calls: int
    maximum_extra_tokens: int
    difficulty_score: float
    reasons: tuple[str, ...]
    blocked_requirements: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class AdaptiveReasoningPolicy:
    """Allocate bounded compute from explicit difficulty and competence evidence."""

    SCHEMA_VERSION = 1

    def plan(
        self,
        signals: ReasoningSignals,
        *,
        base_tokens: int = 128,
        owner_token_cap: int = 512,
    ) -> ReasoningBudget:
        if signals.prompt_tokens < 0:
            raise ValueError("prompt token count cannot be negative")
        difficulty = max(0.0, min(1.0, float(signals.estimated_difficulty)))
        competence = max(0.0, min(1.0, float(signals.competence)))
        uncertainty = 1.0 - competence
        score = min(
            1.0,
            0.55 * difficulty
            + 0.35 * uncertainty
            + 0.10 * min(1.0, signals.prompt_tokens / 1024.0),
        )
        reasons = [
            f"difficulty={difficulty:.3f}",
            f"competence={competence:.3f}",
            f"combined_score={score:.3f}",
        ]
        if signals.irreversible_action:
            score = max(score, 0.75)
            reasons.append("irreversible_action_requires_verification")

        if score < 0.25 and competence >= 0.75:
            mode: ReasoningMode = "direct"
            candidates, revisions, retrieval, verifiers, multiplier = 1, 0, 0, 0, 1.0
        elif score < 0.50:
            mode = "verify"
            candidates, revisions, retrieval, verifiers, multiplier = 1, 1, 0, 1, 1.5
        elif score < 0.75:
            mode = "retrieve_decompose"
            candidates, revisions, retrieval, verifiers, multiplier = 2, 1, 1, 2, 2.0
        else:
            mode = "search_verify"
            candidates, revisions, retrieval, verifiers, multiplier = 4, 2, 2, 4, 3.0

        blocked: list[str] = []
        if retrieval and not signals.retrieval_available:
            blocked.append("retrieval_unavailable")
            retrieval = 0
        if verifiers and not signals.verifier_available:
            blocked.append("verifier_unavailable")
            verifiers = 0
            revisions = 0
        if signals.irreversible_action and not signals.verifier_available:
            blocked.append("irreversible_action_has_no_verifier")
        maximum = min(max(1, int(owner_token_cap)), max(1, int(base_tokens * multiplier)))
        return ReasoningBudget(
            schema_version=self.SCHEMA_VERSION,
            mode=mode,
            candidate_count=candidates,
            revision_count=revisions,
            retrieval_queries=retrieval,
            verifier_calls=verifiers,
            maximum_extra_tokens=maximum,
            difficulty_score=score,
            reasons=tuple(reasons),
            blocked_requirements=tuple(dict.fromkeys(blocked)),
        )


_HARD_REASONING = re.compile(
    r"\b(prove|derive|debug|compare|counterfactual|optimi[sz]e|plan|why|caus|"
    r"theorem|constraint|multi[- ]step|verify)\b",
    re.IGNORECASE,
)
_HIGH_STAKES = re.compile(
    r"\b(medical|medicine|legal|law|financial|investment|irreversible|delete|deploy)\b",
    re.IGNORECASE,
)


def estimate_prompt_difficulty(prompt: str) -> float:
    """Cheap routing estimate; it is policy input, never capability evidence."""

    text = str(prompt).strip()
    if not text:
        return 0.0
    words = text.split()
    score = min(0.35, len(words) / 400.0)
    score += min(0.35, 0.09 * len(_HARD_REASONING.findall(text)))
    score += 0.20 if _HIGH_STAKES.search(text) else 0.0
    score += 0.10 if any(symbol in text for symbol in ("```", "->", "=", "{")) else 0.0
    return min(1.0, score)


def plan_for_prompt(
    prompt: str,
    *,
    domain: str = "general",
    competence: float = 0.0,
    verifier_available: bool,
    retrieval_available: bool,
    irreversible_action: bool = False,
    owner_token_cap: int = 512,
) -> ReasoningBudget:
    return AdaptiveReasoningPolicy().plan(
        ReasoningSignals(
            domain=str(domain),
            prompt_tokens=len(str(prompt).split()),
            estimated_difficulty=estimate_prompt_difficulty(prompt),
            competence=competence,
            verifier_available=verifier_available,
            retrieval_available=retrieval_available,
            irreversible_action=irreversible_action,
        ),
        owner_token_cap=owner_token_cap,
    )
