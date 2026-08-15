from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from anra.memory.bm25 import BM25MemoryTier

from retrieval.adapters import BM25RetrieverAdapter
from retrieval.protocols import RetrievalHit, RetrievalQuery


def _terms(text: str) -> set[str]:
    return set(re.findall(r"\b\w+\b", text.lower()))


@dataclass(frozen=True, slots=True)
class DuplicateDecision:
    duplicate: bool
    exact: bool
    score: float
    matched_id: str | None = None


class CorpusDedupIndex:
    """Retrieval-backed exact/near-duplicate index for corpus curation."""

    name = "corpus_dedup"

    def __init__(self, *, near_duplicate_threshold: float = 1.0) -> None:
        if not 0.0 <= near_duplicate_threshold <= 1.0:
            raise ValueError("near_duplicate_threshold must be in [0, 1]")
        self.threshold = float(near_duplicate_threshold)
        self._hashes: dict[str, str] = {}
        self._documents: dict[str, str] = {}
        self._bm25 = BM25MemoryTier()
        self._retriever = BM25RetrieverAdapter(self._bm25)

    def search(self, query: RetrievalQuery) -> list[RetrievalHit]:
        return self._retriever.search(query)

    def check_and_add(
        self,
        text: str,
        *,
        record_id: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> DuplicateDecision:
        normalized = " ".join(text.lower().split())
        digest = hashlib.sha256(normalized.encode()).hexdigest()
        if digest in self._hashes:
            return DuplicateDecision(True, True, 1.0, self._hashes[digest])

        best_id: str | None = None
        best_score = 0.0
        query_terms = _terms(normalized)
        if self._documents and query_terms:
            for hit in self.search(RetrievalQuery(normalized, limit=8)):
                candidate_terms = _terms(hit.text)
                score = len(query_terms & candidate_terms) / max(
                    1, len(query_terms | candidate_terms)
                )
                if score > best_score:
                    best_id, best_score = hit.id, score
        if best_id is not None and best_score >= self.threshold:
            return DuplicateDecision(True, False, best_score, best_id)

        canonical_id = record_id or digest[:16]
        self._hashes[digest] = canonical_id
        self._documents[canonical_id] = normalized
        self._bm25.write(
            normalized,
            {**(metadata or {}), "canonical_id": canonical_id},
        )
        return DuplicateDecision(False, False, best_score, best_id)
