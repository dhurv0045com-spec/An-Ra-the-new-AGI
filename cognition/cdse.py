"""Deterministic cross-domain structural synthesis."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Literal

ReviewStatus = Literal["candidate", "verified", "rejected", "expert_reviewed"]


@dataclass(frozen=True)
class StructuralSignature:
    domain: str
    variables: tuple[str, ...]
    relationships: tuple[str, ...]
    constraints: tuple[str, ...]
    objective: str
    causal_structure: tuple[str, ...]

    def normalized(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                value.strip().lower()
                for group in (
                    self.variables,
                    self.relationships,
                    self.constraints,
                    self.causal_structure,
                    (self.objective,),
                )
                for value in group
            )
        )


@dataclass(frozen=True)
class SynthesisHypothesis:
    hypothesis_id: str
    source_domain: str
    target_domain: str
    structural_correspondence: tuple[str, ...]
    adapted_solution: str
    testable_prediction: str
    falsification_path: str
    confidence: float
    value_if_true: str
    verification_status: ReviewStatus = "candidate"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


SEED_STRUCTURES = (
    StructuralSignature(
        "network_diffusion",
        ("nodes", "edges"),
        ("spread",),
        ("capacity",),
        "maximize adoption",
        ("neighbor influence",),
    ),
    StructuralSignature(
        "exploration_exploitation",
        ("actions", "rewards"),
        ("sample", "update"),
        ("budget",),
        "maximize cumulative reward",
        ("uncertainty affects selection",),
    ),
    StructuralSignature(
        "cascading_failure",
        ("components", "load"),
        ("dependency",),
        ("threshold",),
        "minimize systemic loss",
        ("failure transfers load",),
    ),
    StructuralSignature(
        "principal_agent",
        ("principal", "agent"),
        ("delegation",),
        ("information asymmetry",),
        "align incentives",
        ("incentive changes behavior",),
    ),
    StructuralSignature(
        "np_hard_optimization",
        ("decision variables",),
        ("combinatorial search",),
        ("feasibility",),
        "optimize objective",
        ("choices determine cost",),
    ),
)


class CrossDomainSynthesisEngine:
    def __init__(self, learned: tuple[StructuralSignature, ...] = ()) -> None:
        self.structures = (*SEED_STRUCTURES, *learned)

    @staticmethod
    def similarity(left: StructuralSignature, right: StructuralSignature) -> float:
        a, b = set(left.normalized()), set(right.normalized())
        return len(a & b) / max(1, len(a | b))

    def synthesize(
        self, problem: StructuralSignature, *, minimum_similarity: float = 0.08
    ) -> list[SynthesisHypothesis]:
        candidates: list[SynthesisHypothesis] = []
        for source in sorted(self.structures, key=lambda item: item.domain):
            if source.domain == problem.domain:
                continue
            score = self.similarity(source, problem)
            if score < minimum_similarity:
                continue
            correspondence = tuple(sorted(set(source.relationships) & set(problem.relationships)))
            if not correspondence:
                correspondence = (f"{source.objective} -> {problem.objective}",)
            raw_id = (
                f"{source.domain}:{problem.domain}:{source.normalized()}:{problem.normalized()}"
            )
            candidates.append(
                SynthesisHypothesis(
                    hypothesis_id=hashlib.sha256(raw_id.encode("utf-8")).hexdigest()[:16],
                    source_domain=source.domain,
                    target_domain=problem.domain,
                    structural_correspondence=correspondence,
                    adapted_solution=(
                        f"Test a {source.domain} strategy against the {problem.domain} objective."
                    ),
                    testable_prediction=(
                        f"The adapted strategy improves {problem.objective} over a fixed baseline."
                    ),
                    falsification_path=(
                        "A preregistered comparison shows no improvement or "
                        "violates a target constraint."
                    ),
                    confidence=min(0.8, score),
                    value_if_true=f"A reusable intervention for {problem.domain}.",
                )
            )
        return sorted(candidates, key=lambda item: (-item.confidence, item.hypothesis_id))
