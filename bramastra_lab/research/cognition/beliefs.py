"""Belief representation and reference revision (M19).

Epistemic statuses are provenance classes: model confidence can never change
them. The reference finite-support updater is an engineered diagnostic/teacher
with declared likelihood assumptions; a zero normalizer emits MODEL_MISMATCH
and never silently resets confidence to a convenient certainty.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from bramastra_lab.research.contracts.core import content_identity

BELIEF_STATUSES = frozenset({"observed_report", "hypothesis", "derived",
                             "contradicted", "retracted", "unresolved"})
EVIDENCE_STATUSES = frozenset({"admitted", "retracted"})


class BeliefError(ValueError):
    """A belief/workspace operation violated its contract."""


@dataclass(frozen=True)
class EvidenceRecord:
    alias: str                 # episode-local, e.g. "e0"
    content: Mapping[str, Any]
    ancestry: tuple[str, ...]  # root aliases this evidence derives from
    status: str = "admitted"
    reliability: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.alias, str) or not self.alias:
            raise BeliefError("evidence alias must be a nonempty string")
        if not isinstance(self.content, Mapping):
            raise BeliefError("evidence content must be a mapping")
        if self.status not in EVIDENCE_STATUSES:
            raise BeliefError(f"evidence status must be one of {sorted(EVIDENCE_STATUSES)}")
        if not 0.0 < float(self.reliability) <= 1.0:
            raise BeliefError("reliability must lie in (0, 1]")

    def identity(self) -> str:
        return content_identity({"alias": self.alias, "content": dict(self.content),
                                 "ancestry": list(self.ancestry)})


@dataclass(frozen=True)
class Belief:
    alias: str                 # episode-local, e.g. "b1"
    proposition: Mapping[str, Any]
    status: str
    support: Mapping[str, float]          # hypothesis -> probability (may be {})
    evidence_aliases: tuple[str, ...] = ()
    conflicting_aliases: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.alias, str) or not self.alias:
            raise BeliefError("belief alias must be a nonempty string")
        if self.status not in BELIEF_STATUSES:
            raise BeliefError(f"belief status must be one of {sorted(BELIEF_STATUSES)}")
        if not isinstance(self.proposition, Mapping):
            raise BeliefError("proposition must be a mapping")
        total = sum(self.support.values())
        if self.support and (not 0.99 <= total <= 1.01):
            raise BeliefError("support distribution must sum to ~1")

    def identity(self) -> str:
        return content_identity({"alias": self.alias,
                                 "proposition": dict(self.proposition),
                                 "status": self.status,
                                 "support": {key: self.support[key]
                                             for key in sorted(self.support)},
                                 "evidence": list(self.evidence_aliases)})


def reference_finite_support_update(
    support: Mapping[str, float],
    likelihood: Mapping[str, float],
) -> tuple[dict[str, float], str]:
    """q_next(h) ∝ q_current(h) * p(observation | h, chosen_action).

    Returns (next_distribution, status). A zero normalizer means the
    support/likelihood assumptions are inconsistent with the observation:
    status is ``MODEL_MISMATCH`` and the caller retains evidence — the
    updater never silently resets confidence to a convenient alternative.
    """
    if not support:
        raise BeliefError("empty support cannot be updated")
    unknown = [key for key in likelihood if key not in support]
    if unknown:
        raise BeliefError(f"likelihood references hypotheses outside the support: {unknown[:3]}")
    unnormalized = {key: float(support[key]) * float(likelihood[key])
                    for key in support}
    normalizer = sum(unnormalized.values())
    if normalizer <= 0.0:
        return {}, "MODEL_MISMATCH"
    return ({key: value / normalizer for key, value in unnormalized.items()},
            "OK")


def unique_corroboration_roots(evidence: Sequence[EvidenceRecord]) -> set[str]:
    """Correlated copies of one original observation share a root and count
    once: duplicate retrieval is not independent corroboration. Ancestry
    aliases are resolved transitively against the record set."""
    by_alias = {record.alias: record for record in evidence}

    def roots_of(record: EvidenceRecord) -> set[str]:
        stack = list(record.ancestry or (record.alias,))
        found: set[str] = set()
        seen: set[str] = set()
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            parent = by_alias.get(current)
            if parent is None or parent is record:
                found.add(current)
            else:
                stack.extend(parent.ancestry or (parent.alias,))
        return found

    roots: set[str] = set()
    for record in evidence:
        if record.status == "admitted":
            roots |= roots_of(record)
    return roots
