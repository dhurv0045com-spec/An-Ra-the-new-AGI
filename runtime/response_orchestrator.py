"""Proof-first response routing for the small local An-Ra model.

The base model remains the language generator. This controller spends a small,
explicit inference budget to reject visibly collapsed candidates and routes
mechanically verifiable arithmetic to the existing bounded tool broker. It
never upgrades a heuristic score into a factuality claim: an answer is either
tool-verified, model-selected, or an explicit abstention.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal

PROOF_FIRST_SCHEMA = "anra-proof-first-response/v1"
ResponseSource = Literal["verified_tool", "selected_model", "abstained"]

_GENERIC_COLLAPSE = (
    "hello how can i help",
    "how can i help you today",
    "as an ai language model",
    "i'm here to help",
)
_STOP_WORDS = frozenset(
    {
        "about",
        "after",
        "again",
        "also",
        "answer",
        "are",
        "could",
        "does",
        "from",
        "have",
        "how",
        "into",
        "please",
        "should",
        "that",
        "tell",
        "their",
        "there",
        "these",
        "they",
        "this",
        "what",
        "when",
        "where",
        "which",
        "who",
        "with",
        "would",
        "why",
        "you",
        "your",
    }
)
_ARITHMETIC = re.compile(
    r"(?<![\w.])(-?\d+(?:\.\d+)?)\s*"
    r"(plus|minus|times|multiplied\s+by|divided\s+by|modulo|[+\-*/%])\s*"
    r"(-?\d+(?:\.\d+)?)(?![\w.])",
    re.IGNORECASE,
)
_OPERATORS = {
    "plus": "+",
    "minus": "-",
    "times": "*",
    "multiplied by": "*",
    "divided by": "/",
    "modulo": "%",
}
_ARITHMETIC_FILLER = frozenset(
    {
        "answer",
        "arithmetic",
        "briefly",
        "calculate",
        "compute",
        "equals",
        "is",
        "me",
        "please",
        "result",
        "show",
        "tell",
        "the",
        "what",
    }
)


@dataclass(frozen=True)
class CandidateScore:
    attempt: int
    score: float
    accepted: bool
    reasons: tuple[str, ...]
    response: str
    trace: Mapping[str, object]

    def public_view(self) -> dict[str, object]:
        return {
            "attempt": self.attempt,
            "score": self.score,
            "accepted": self.accepted,
            "reasons": list(self.reasons),
            "response_preview": self.response[:160],
            "trace": dict(self.trace),
        }


@dataclass(frozen=True)
class ProofFirstResult:
    answer: str
    source: ResponseSource
    confidence_scope: str
    candidates: tuple[CandidateScore, ...]
    selected_attempt: int | None = None
    tool_receipt: Mapping[str, object] | None = None
    schema: str = PROOF_FIRST_SCHEMA

    def public_evidence(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "source": self.source,
            "confidence_scope": self.confidence_scope,
            "candidate_count": len(self.candidates),
            "selected_attempt": self.selected_attempt,
            "candidates": [candidate.public_view() for candidate in self.candidates],
            "tool_receipt": dict(self.tool_receipt or {}),
        }


def extract_arithmetic_expression(prompt: str) -> str | None:
    """Extract one conservative binary arithmetic request from natural language."""

    matches = list(_ARITHMETIC.finditer(str(prompt)))
    if len(matches) != 1:
        return None
    match = matches[0]
    remainder = f"{prompt[:match.start()]} {prompt[match.end():]}".lower()
    if re.search(r"[\d+*/%=-]", remainder):
        return None
    remaining_words = set(re.findall(r"[a-z]+", remainder))
    if remaining_words - _ARITHMETIC_FILLER:
        return None
    left, operator, right = match.groups()
    normalized_operator = _OPERATORS.get(operator.lower(), operator)
    return f"{left} {normalized_operator} {right}"


def _terms(text: str) -> set[str]:
    return {
        term
        for term in re.findall(r"[a-z0-9][a-z0-9_-]{2,}", text.lower())
        if term not in _STOP_WORDS
    }


def score_candidate(
    prompt: str,
    response: str,
    trace: Mapping[str, object],
    *,
    attempt: int,
) -> CandidateScore:
    """Score only observable response quality, never hidden factual correctness."""

    text = str(response).strip()
    normalized = " ".join(text.lower().split())
    prompt_terms = _terms(prompt)
    response_terms = _terms(text)
    overlap = len(prompt_terms & response_terms) / max(1, len(prompt_terms))
    reasons: list[str] = []
    score = 0.0

    if len(text) >= 12:
        score += 0.2
    else:
        reasons.append("too_short")
    if overlap >= 0.15 or not prompt_terms:
        score += min(0.35, 0.15 + overlap * 0.4)
    else:
        reasons.append("low_prompt_relevance")
    if not bool(trace.get("repetition_detected", False)):
        score += 0.2
    else:
        reasons.append("repetition_detected")
    if not bool(trace.get("fragment_detected", False)):
        score += 0.15
    else:
        reasons.append("fragment_detected")
    if str(trace.get("quality_state", "")).lower() == "accepted":
        score += 0.1
    if any(phrase in normalized for phrase in _GENERIC_COLLAPSE):
        score -= 0.65
        reasons.append("generic_collapse")
    if "�" in text:
        score -= 0.4
        reasons.append("invalid_character")

    score = round(max(0.0, min(1.0, score)), 4)
    hard_failures = {
        "duplicate_candidate",
        "fragment_detected",
        "generic_collapse",
        "invalid_character",
        "low_prompt_relevance",
        "repetition_detected",
        "too_short",
    }
    return CandidateScore(
        attempt=attempt,
        score=score,
        accepted=score >= 0.55 and not hard_failures.intersection(reasons),
        reasons=tuple(reasons),
        response=text,
        trace=dict(trace),
    )


def proof_first_response(
    prompt: str,
    *,
    generate: Callable[[int], tuple[str, Mapping[str, object]]],
    calculate: Callable[[str], tuple[object, Mapping[str, object]]] | None = None,
    candidate_count: int = 2,
) -> ProofFirstResult:
    """Route a prompt through exact tools or bounded candidate selection."""

    if not 1 <= int(candidate_count) <= 3:
        raise ValueError("candidate_count must be in [1, 3]")
    expression = extract_arithmetic_expression(prompt)
    if expression is not None and calculate is not None:
        try:
            value, receipt = calculate(expression)
        except (ValueError, ZeroDivisionError) as error:
            return ProofFirstResult(
                answer=f"I could not verify that calculation: {error}.",
                source="abstained",
                confidence_scope="exact tool refused the arithmetic request",
                candidates=(),
                tool_receipt={"status": "refused", "reason": str(error)},
            )
        return ProofFirstResult(
            answer=f"{expression} = {value}",
            source="verified_tool",
            confidence_scope="exact bounded local arithmetic",
            candidates=(),
            tool_receipt=dict(receipt),
        )

    candidates: list[CandidateScore] = []
    seen: set[str] = set()
    for attempt in range(int(candidate_count)):
        response, trace = generate(attempt)
        candidate = score_candidate(prompt, response, trace, attempt=attempt)
        normalized = " ".join(candidate.response.lower().split())
        if normalized in seen:
            candidate = CandidateScore(
                attempt=candidate.attempt,
                score=round(max(0.0, candidate.score - 0.35), 4),
                accepted=False,
                reasons=(*candidate.reasons, "duplicate_candidate"),
                response=candidate.response,
                trace=candidate.trace,
            )
        seen.add(normalized)
        candidates.append(candidate)
        if candidate.accepted and candidate.score >= 0.9:
            break

    accepted = [candidate for candidate in candidates if candidate.accepted]
    if not accepted:
        return ProofFirstResult(
            answer=(
                "I could not produce a sufficiently relevant, non-repetitive answer "
                "within this request's inference budget. Try adding specific context, "
                "or inspect the raw model output in Developer mode."
            ),
            source="abstained",
            confidence_scope="no candidate passed observable quality gates",
            candidates=tuple(candidates),
        )
    best = max(accepted, key=lambda candidate: (candidate.score, -candidate.attempt))
    return ProofFirstResult(
        answer=best.response,
        source="selected_model",
        confidence_scope=(
            "selected for lexical relevance and absence of observable collapse; "
            "factuality unverified"
        ),
        candidates=tuple(candidates),
        selected_attempt=best.attempt,
    )
