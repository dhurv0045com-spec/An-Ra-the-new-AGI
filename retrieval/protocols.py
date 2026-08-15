from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class RetrievalQuery:
    text: str
    limit: int = 8
    vector: Any | None = None
    filters: Mapping[str, object] = field(default_factory=dict)
    trace_id: str | None = None
    candidate_multiplier: int = 3

    def __post_init__(self) -> None:
        if self.limit <= 0:
            raise ValueError("retrieval limit must be positive")
        if self.candidate_multiplier <= 0:
            raise ValueError("candidate_multiplier must be positive")


@dataclass(frozen=True, slots=True)
class RetrievalProvenance:
    retriever: str
    rank: int
    raw_score: float
    weight: float = 1.0


@dataclass(frozen=True, slots=True)
class RetrievalHit:
    id: str
    text: str
    score: float
    metadata: Mapping[str, object] = field(default_factory=dict)
    provenance: tuple[RetrievalProvenance, ...] = ()


@runtime_checkable
class Retriever(Protocol):
    @property
    def name(self) -> str: ...

    def search(self, query: RetrievalQuery) -> Sequence[RetrievalHit]: ...
