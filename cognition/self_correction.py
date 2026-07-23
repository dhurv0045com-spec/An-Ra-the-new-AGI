"""Budgeted retrieve-plan-generate-verify-revise orchestration.

This module is intentionally model-agnostic.  It coordinates explicitly
provided generation, retrieval and verification functions, records what was
actually checked, and abstains when a required verifier cannot establish an
answer.  Merely running the loop is not evidence that a response is correct.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class CorrectionBudget:
    candidates: int = 1
    revisions: int = 1
    retrieval_queries: int = 0
    verifier_calls: int = 1
    require_verification: bool = True

    def __post_init__(self) -> None:
        for name in ("candidates", "revisions", "retrieval_queries", "verifier_calls"):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"{name} cannot be negative")
        if self.candidates < 1:
            raise ValueError("self-correction requires at least one candidate")


@dataclass(frozen=True)
class Verification:
    verified: bool
    score: float
    verifier: str
    evidence: Mapping[str, object] = field(default_factory=dict)
    feedback: str = ""

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.score) <= 1.0:
            raise ValueError("verification score must be in [0, 1]")
        if not self.verifier.strip():
            raise ValueError("verification must identify its verifier")


@dataclass(frozen=True)
class CandidateTrace:
    candidate_id: str
    text: str
    revision: int
    verification: Verification | None


@dataclass(frozen=True)
class CorrectionResult:
    trace_id: str
    status: str
    answer: str
    plan: str
    retrieval: tuple[Mapping[str, object], ...]
    candidates: tuple[CandidateTrace, ...]
    verifier_calls: int
    started_at: float
    completed_at: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


Generator = Callable[[str, str, Sequence[Mapping[str, object]], int], str]
Verifier = Callable[[str, str, Sequence[Mapping[str, object]]], Verification]
Retriever = Callable[[str, int], Sequence[Mapping[str, object]]]
Planner = Callable[[str, Sequence[Mapping[str, object]]], str]
Reviser = Callable[[str, str, Verification, int], str]
Persister = Callable[[CorrectionResult], None]


class SelfCorrectionEngine:
    def __init__(
        self,
        *,
        generate: Generator,
        verify: Verifier | None,
        retrieve: Retriever | None = None,
        plan: Planner | None = None,
        revise: Reviser | None = None,
        persist: Persister | None = None,
    ) -> None:
        self.generate = generate
        self.verify = verify
        self.retrieve = retrieve
        self.plan = plan
        self.revise = revise
        self.persist = persist

    def run(self, prompt: str, *, budget: CorrectionBudget) -> CorrectionResult:
        started_at = time.time()
        normalized_prompt = " ".join(str(prompt).split())
        if not normalized_prompt:
            raise ValueError("self-correction prompt cannot be empty")
        if budget.require_verification and self.verify is None:
            raise RuntimeError("required verifier is unavailable")

        retrieval: tuple[Mapping[str, object], ...] = ()
        if budget.retrieval_queries and self.retrieve is not None:
            retrieved = self.retrieve(normalized_prompt, budget.retrieval_queries)
            retrieval = tuple(dict(item) for item in retrieved)
        plan = (
            self.plan(normalized_prompt, retrieval)
            if self.plan is not None
            else "answer directly from the supplied prompt and provenance-bound context"
        )

        traces: list[CandidateTrace] = []
        verifier_calls = 0
        best: CandidateTrace | None = None
        for candidate_index in range(budget.candidates):
            text = self.generate(normalized_prompt, plan, retrieval, candidate_index)
            verification: Verification | None = None
            if self.verify is not None and verifier_calls < budget.verifier_calls:
                verification = self.verify(normalized_prompt, text, retrieval)
                verifier_calls += 1
            trace = CandidateTrace(str(uuid.uuid4()), text, 0, verification)
            traces.append(trace)
            if best is None or _candidate_score(trace) > _candidate_score(best):
                best = trace
            if verification is not None and verification.verified:
                best = trace
                break

        revision = 0
        while (
            best is not None
            and (best.verification is None or not best.verification.verified)
            and revision < budget.revisions
            and verifier_calls < budget.verifier_calls
            and self.revise is not None
            and self.verify is not None
            and best.verification is not None
        ):
            revision += 1
            text = self.revise(
                normalized_prompt,
                best.text,
                best.verification,
                revision,
            )
            verification = self.verify(normalized_prompt, text, retrieval)
            verifier_calls += 1
            revised = CandidateTrace(str(uuid.uuid4()), text, revision, verification)
            traces.append(revised)
            if _candidate_score(revised) >= _candidate_score(best):
                best = revised
            if verification.verified:
                best = revised
                break

        verified = bool(best and best.verification and best.verification.verified)
        if verified:
            status = "verified"
            answer = best.text if best else ""
        elif budget.require_verification:
            status = "abstained"
            answer = "I could not verify a reliable answer within the available reasoning budget."
        else:
            status = "unverified"
            answer = best.text if best else ""
        result = CorrectionResult(
            trace_id=str(uuid.uuid4()),
            status=status,
            answer=answer,
            plan=plan,
            retrieval=retrieval,
            candidates=tuple(traces),
            verifier_calls=verifier_calls,
            started_at=started_at,
            completed_at=time.time(),
        )
        if self.persist is not None:
            self.persist(result)
        return result


def _candidate_score(candidate: CandidateTrace) -> float:
    return float(candidate.verification.score) if candidate.verification is not None else -1.0

