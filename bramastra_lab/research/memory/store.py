"""Eligible-record memory connected to the actual input path (M04).

First implementation: deterministic lexical overlap retrieval over eligible
public text with stable content-hash tie breaking, frozen index identity and
explicit scope filtering BEFORE ranking. Sealed and prohibited records can
never rank or render.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from bramastra_lab.research.contracts.core import content_identity


class MemoryError(ValueError):
    """A memory operation violated its contract."""


@dataclass(frozen=True)
class MemoryRecord:
    content: str
    identity: str
    scope: str
    episode_id: str | None = None
    kind: str = "episode_trace"

    def __post_init__(self) -> None:
        if not isinstance(self.content, str) or not self.content:
            raise MemoryError("content must be a nonempty string")
        if not isinstance(self.identity, str) or not self.identity:
            raise MemoryError("identity must be a nonempty string")
        if not isinstance(self.scope, str) or not self.scope:
            raise MemoryError("scope must be a nonempty string")

    def content_identity(self) -> str:
        return content_identity({"content": self.content, "identity": self.identity,
                                 "scope": self.scope, "kind": self.kind})


RETRIEVAL_RULE = "lexical-overlap/v1"


@dataclass(frozen=True)
class MemoryIndex:
    records: tuple[MemoryRecord, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "records", tuple(self.records))
        identities = [record.identity for record in self.records]
        if len(set(identities)) != len(identities):
            raise MemoryError("duplicate record identities in index")

    @property
    def identity(self) -> str:
        """Frozen index identity: content or permission change invalidates."""
        return content_identity({
            "rule": RETRIEVAL_RULE,
            "records": [record.content_identity() for record in self.records],
        })

    @property
    def retrieval_rule_identity(self) -> str:
        return content_identity({"rule": RETRIEVAL_RULE, "top_k": None})

    def retrieve(self, query: str, *, scope_allowlist: set[str], top_k: int = 3,
                 exclude_episodes: set[str] | None = None,
                 token_budget: int | None = None) -> MemoryContext:
        from bramastra_lab.research.memory.store import retrieve as _retrieve

        return _retrieve(self, query, scope_allowlist=scope_allowlist, top_k=top_k,
                         exclude_episodes=exclude_episodes, token_budget=token_budget)


def _overlap(query: str, content: str) -> float:
    query_terms = set(query.lower().split())
    content_terms = set(content.lower().split())
    if not query_terms:
        return 0.0
    return len(query_terms & content_terms) / len(query_terms)


@dataclass(frozen=True)
class MemoryContext:
    records: tuple[MemoryRecord, ...]
    index_identity: str
    retrieval_rule_identity: str
    token_cost: int
    omitted_record_ids: tuple[str, ...]

    def rendered(self) -> str:
        return "\n".join(f"[{record.identity}] {record.content}"
                         for record in self.records)


def retrieve(index: MemoryIndex, query: str, *, scope_allowlist: set[str],
             top_k: int = 3, exclude_episodes: set[str] | None = None,
             token_budget: int | None = None) -> MemoryContext:
    """Filter before ranking: ineligible scopes never reach the ranking at all."""
    if top_k <= 0:
        raise MemoryError("top_k must be positive")
    excluded = exclude_episodes or set()
    eligible = [record for record in index.records
                if record.scope in scope_allowlist
                and (record.episode_id is None or record.episode_id not in excluded)]
    scored = sorted(
        ((-_overlap(query, record.content), record.content_identity(), record)
         for record in eligible),
        key=lambda item: (item[0], item[1]))  # stable tie break by content hash
    chosen = [record for _score, _hash, record in scored[:top_k]]
    omitted = tuple(record.identity for record in eligible[len(chosen):])
    token_cost = sum(len(record.content.split()) for record in chosen)
    if token_budget is not None:
        kept: list[MemoryRecord] = []
        used = 0
        omitted = tuple(record.identity for record in chosen)
        for record in chosen:
            cost = len(record.content.split())
            if used + cost > token_budget:
                continue
            kept.append(record)
            used += cost
            omitted = tuple(oid for oid in omitted if oid != record.identity)
        chosen = kept
        token_cost = used
    return MemoryContext(records=tuple(chosen), index_identity=index.identity,
                         retrieval_rule_identity=index.retrieval_rule_identity,
                         token_cost=token_cost, omitted_record_ids=omitted)
