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


RETRIEVAL_RULE = "lexical-overlap/exact-public-event-budget/v2"


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
    """Filter before ranking, then fill the budget with fitting ranked items.

    An oversized high-ranked item no longer blocks a smaller eligible memory
    from filling the remaining context. `omitted_record_ids` accounts for both
    over-budget candidates and eligible items beyond the top-k result.
    """
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
    from bramastra_lab.research.experience.codec import (
        DEFAULT_MAX_EVENT_BYTES, canonical_event_bytes)
    from bramastra_lab.research.experience.public_state import (
        compact_memory_content)

    chosen: list[MemoryRecord] = []
    omitted: list[str] = []
    token_cost = 0
    for _score, _hash, record in scored:
        if len(chosen) >= top_k:
            omitted.append(record.identity)
            continue
        # Budget the exact bytes/markers the model will receive, including the
        # memory envelope and role tag, rather than content alone.
        body = canonical_event_bytes(
            compact_memory_content(record.content))
        cost = len(b"observation:") + len(body) + 1
        if (len(body) > DEFAULT_MAX_EVENT_BYTES
                or cost > DEFAULT_MAX_EVENT_BYTES):
            omitted.append(record.identity)
            continue
        if token_budget is not None and token_cost + cost > token_budget:
            omitted.append(record.identity)
            continue
        chosen.append(record)
        token_cost += cost
    return MemoryContext(records=tuple(chosen), index_identity=index.identity,
                         retrieval_rule_identity=index.retrieval_rule_identity,
                         token_cost=token_cost,
                         omitted_record_ids=tuple(omitted))
