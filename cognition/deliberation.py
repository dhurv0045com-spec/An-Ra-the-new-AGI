"""Bounded verifier-driven deliberation for inference-time capability.

This controller changes how an existing checkpoint is used; it does not claim
to add knowledge or capability to the checkpoint weights.  Every run follows
an explicit understand -> retrieve -> plan -> candidate -> verify -> revise or
abstain -> persist sequence and reports the scope of what was actually checked.
"""

from __future__ import annotations

import re
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from typing import Literal

DELIBERATION_SCHEMA = "anra-verified-deliberation/v1"
DeliberationStatus = Literal["accepted", "abstained", "unverified", "disabled"]
TaskType = Literal["arithmetic", "code", "json", "factual", "open_ended"]


@dataclass(frozen=True)
class DeliberationBudget:
    candidates: int = 1
    revisions: int = 1
    retrieval_results: int = 3
    verifier_calls: int = 2
    max_generated_tokens: int = 160
    deadline_seconds: float = 45.0
    require_verification: bool = True

    def __post_init__(self) -> None:
        for name in ("candidates", "revisions", "retrieval_results", "verifier_calls"):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"{name} cannot be negative")
        if self.candidates < 1:
            raise ValueError("deliberation requires at least one candidate")
        if self.max_generated_tokens < 1:
            raise ValueError("max_generated_tokens must be positive")
        if not 0.1 <= float(self.deadline_seconds) <= 600.0:
            raise ValueError("deadline_seconds must be in [0.1, 600]")
        if self.require_verification and self.verifier_calls < 1:
            raise ValueError("required verification needs at least one verifier call")


@dataclass(frozen=True)
class Understanding:
    task_type: TaskType
    verification_target: str
    needs_retrieval: bool
    risk: Literal["low", "moderate", "high"]


@dataclass(frozen=True)
class GenerationArtifact:
    text: str
    token_count: int
    evidence: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class VerificationDecision:
    passed: bool
    score: float
    verifier: str
    scope: str
    feedback: str = ""
    evidence: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.score) <= 1.0:
            raise ValueError("verification score must be in [0, 1]")
        if not self.verifier.strip() or not self.scope.strip():
            raise ValueError("verification must identify verifier and scope")


@dataclass(frozen=True)
class CandidateEvidence:
    candidate_id: str
    revision: int
    text: str
    token_count: int
    verification: VerificationDecision | None
    generation_evidence: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class DeliberationResult:
    trace_id: str
    schema: str
    status: DeliberationStatus
    answer: str
    understanding: Understanding
    plan: str
    retrieval: tuple[Mapping[str, object], ...]
    candidates: tuple[CandidateEvidence, ...]
    selected_candidate_id: str | None
    stages_completed: tuple[str, ...]
    verifier_calls: int
    generated_tokens: int
    elapsed_seconds: float
    deterministic: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def public_evidence(self) -> dict[str, object]:
        """Return bounded evidence without exposing rejected draft text."""
        best = next(
            (
                item
                for item in self.candidates
                if item.candidate_id == self.selected_candidate_id
            ),
            None,
        )
        return {
            "trace_id": self.trace_id,
            "schema": self.schema,
            "status": self.status,
            "understanding": asdict(self.understanding),
            "plan": self.plan,
            "retrieval_count": len(self.retrieval),
            "candidate_count": len(self.candidates),
            "revision_count": sum(item.revision > 0 for item in self.candidates),
            "verifier_calls": self.verifier_calls,
            "generated_tokens": self.generated_tokens,
            "elapsed_seconds": round(self.elapsed_seconds, 4),
            "deterministic": self.deterministic,
            "verification": asdict(best.verification) if best and best.verification else None,
            "stages_completed": list(self.stages_completed),
        }


Retriever = Callable[[str, int], Sequence[Mapping[str, object]]]
Planner = Callable[[str, Understanding, Sequence[Mapping[str, object]]], str]
Generator = Callable[
    [str, Understanding, str, Sequence[Mapping[str, object]], int, CandidateEvidence | None],
    GenerationArtifact,
]
Verifier = Callable[
    [str, Understanding, GenerationArtifact, Sequence[Mapping[str, object]]],
    VerificationDecision,
]
Persister = Callable[[DeliberationResult], bool | None]


def understand_prompt(prompt: str) -> Understanding:
    """Deterministic, claim-limited request classification."""
    text = " ".join(str(prompt).split()).lower()
    if not text:
        raise ValueError("deliberation prompt cannot be empty")
    if re.search(r"\b\d+(?:\.\d+)?\s*(?:plus|minus|times|[+*/-])\s*\d+", text):
        return Understanding("arithmetic", "symbolic answer", False, "low")
    if any(term in text for term in ("python", "function", "code", "program")):
        return Understanding("code", "syntax and declared constraints", False, "moderate")
    if "json" in text:
        return Understanding("json", "machine-readable JSON structure", False, "low")
    if any(
        term in text
        for term in ("who ", "when ", "where ", "what is", "explain", "tell me about")
    ):
        return Understanding("factual", "retrieval-grounded response", True, "moderate")
    return Understanding("open_ended", "response integrity only", False, "moderate")


def default_plan(
    _prompt: str,
    understanding: Understanding,
    retrieval: Sequence[Mapping[str, object]],
) -> str:
    retrieval_clause = (
        "use only the supplied retrieved evidence and state uncertainty for missing facts"
        if retrieval
        else "do not invent external evidence"
    )
    return (
        f"answer the {understanding.task_type} request directly; {retrieval_clause}; "
        f"make the result checkable against {understanding.verification_target}"
    )


class VerifiedDeliberationController:
    """One bounded inference controller with a hard runtime off switch."""

    def __init__(
        self,
        *,
        generate: Generator,
        verify: Verifier | None,
        retrieve: Retriever | None = None,
        plan: Planner = default_plan,
        persist: Persister | None = None,
        enabled: bool = True,
    ) -> None:
        self.generate = generate
        self.verify = verify
        self.retrieve = retrieve
        self.plan = plan
        self.persist = persist
        self.enabled = bool(enabled)

    def run(
        self,
        prompt: str,
        *,
        budget: DeliberationBudget,
        deterministic: bool = True,
    ) -> DeliberationResult:
        started = time.monotonic()
        trace_id = str(uuid.uuid4())
        normalized = " ".join(str(prompt).split())
        understanding = understand_prompt(normalized)
        stages = ["understand"]
        if not self.enabled:
            return self._finish(
                trace_id=trace_id,
                status="disabled",
                answer="Verifier-driven deliberation is disabled by runtime policy.",
                understanding=understanding,
                plan="",
                retrieval=(),
                candidates=(),
                selected_candidate_id=None,
                stages=stages,
                verifier_calls=0,
                generated_tokens=0,
                started=started,
                deterministic=deterministic,
            )
        if budget.require_verification and self.verify is None:
            raise RuntimeError("required deliberation verifier is unavailable")

        retrieved: tuple[Mapping[str, object], ...] = ()
        if budget.retrieval_results and self.retrieve is not None:
            rows = self.retrieve(normalized, budget.retrieval_results)
            retrieved = tuple(dict(row) for row in rows[: budget.retrieval_results])
        stages.append("retrieve")
        plan = self.plan(normalized, understanding, retrieved)
        stages.append("plan")

        candidates: list[CandidateEvidence] = []
        calls = 0
        generated_tokens = 0
        best: CandidateEvidence | None = None
        ordinal = 0
        for _ in range(budget.candidates):
            if self._expired(started, budget) or generated_tokens >= budget.max_generated_tokens:
                break
            artifact = self.generate(
                normalized, understanding, plan, retrieved, ordinal, None
            )
            ordinal += 1
            token_count = max(0, int(artifact.token_count))
            remaining = budget.max_generated_tokens - generated_tokens
            if token_count > remaining:
                raise RuntimeError("generator exceeded the declared deliberation token budget")
            generated_tokens += token_count
            decision = None
            stages.append("candidate")
            if self._expired(started, budget):
                item = CandidateEvidence(
                    str(uuid.uuid4()),
                    0,
                    artifact.text,
                    artifact.token_count,
                    None,
                    artifact.evidence,
                )
                candidates.append(item)
                best = self._better(best, item)
                break
            if self.verify is not None and calls < budget.verifier_calls:
                decision = self.verify(normalized, understanding, artifact, retrieved)
                calls += 1
                stages.append("verify")
            item = CandidateEvidence(
                str(uuid.uuid4()),
                0,
                artifact.text,
                artifact.token_count,
                decision,
                artifact.evidence,
            )
            candidates.append(item)
            best = self._better(best, item)
            if decision is not None and decision.passed:
                break

        revisions = 0
        while (
            best is not None
            and (best.verification is None or not best.verification.passed)
            and revisions < budget.revisions
            and calls < budget.verifier_calls
            and generated_tokens < budget.max_generated_tokens
            and not self._expired(started, budget)
        ):
            revisions += 1
            artifact = self.generate(
                normalized, understanding, plan, retrieved, ordinal, best
            )
            ordinal += 1
            token_count = max(0, int(artifact.token_count))
            remaining = budget.max_generated_tokens - generated_tokens
            if token_count > remaining:
                raise RuntimeError("generator exceeded the declared deliberation token budget")
            generated_tokens += token_count
            if self._expired(started, budget):
                item = CandidateEvidence(
                    str(uuid.uuid4()),
                    revisions,
                    artifact.text,
                    artifact.token_count,
                    None,
                    artifact.evidence,
                )
                candidates.append(item)
                stages.append("revise")
                best = self._better(best, item)
                break
            decision = (
                self.verify(normalized, understanding, artifact, retrieved)
                if self.verify
                else None
            )
            calls += int(decision is not None)
            item = CandidateEvidence(
                str(uuid.uuid4()),
                revisions,
                artifact.text,
                artifact.token_count,
                decision,
                artifact.evidence,
            )
            candidates.append(item)
            stages.extend(("revise", "verify"))
            best = self._better(best, item)
            if decision is not None and decision.passed:
                break

        passed = bool(best and best.verification and best.verification.passed)
        status: DeliberationStatus = (
            "accepted" if passed else "abstained" if budget.require_verification else "unverified"
        )
        answer = (
            best.text
            if best is not None and (passed or not budget.require_verification)
            else "I could not verify a reliable answer within the available deliberation budget."
        )
        stages.append("abstain" if status == "abstained" else "select")
        return self._finish(
            trace_id=trace_id,
            status=status,
            answer=answer,
            understanding=understanding,
            plan=plan,
            retrieval=retrieved,
            candidates=tuple(candidates),
            selected_candidate_id=best.candidate_id if best is not None else None,
            stages=stages,
            verifier_calls=calls,
            generated_tokens=generated_tokens,
            started=started,
            deterministic=deterministic,
        )

    @staticmethod
    def _expired(started: float, budget: DeliberationBudget) -> bool:
        return time.monotonic() - started >= budget.deadline_seconds

    @staticmethod
    def _better(
        current: CandidateEvidence | None, candidate: CandidateEvidence
    ) -> CandidateEvidence:
        if current is None:
            return candidate
        old = current.verification.score if current.verification else -1.0
        new = candidate.verification.score if candidate.verification else -1.0
        return candidate if new >= old else current

    def _finish(self, **values: object) -> DeliberationResult:
        started = float(values.pop("started"))
        stages = tuple(dict.fromkeys(values.pop("stages")))
        result = DeliberationResult(
            schema=DELIBERATION_SCHEMA,
            elapsed_seconds=time.monotonic() - started,
            stages_completed=stages,
            **values,  # type: ignore[arg-type]
        )
        if self.persist is not None:
            persisted = self.persist(result)
            result = replace(
                result,
                stages_completed=(
                    *result.stages_completed,
                    "persist" if persisted is not False else "persistence_failed",
                ),
            )
        return result
