"""Dependency-free BM25 exact-match memory tier."""

from __future__ import annotations

import math
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from anra.core.protocols import HealthStatus, MemoryRecord


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


@dataclass(slots=True)
class _Doc:
    id: str
    content: str
    metadata: dict[str, Any]
    tokens: list[str]
    created_at: float = field(default_factory=time.time)


class BM25MemoryTier:
    """Keyword retrieval that complements embedding-based semantic memory."""

    K1 = 1.5
    B = 0.75

    def __init__(self, max_docs: int = 50_000) -> None:
        self._max = max_docs
        self._docs: dict[str, _Doc] = {}
        self._df: dict[str, int] = defaultdict(int)
        self._total_tokens = 0
        self._counter = 0

    def write(self, content: str, metadata: dict[str, Any] | None = None) -> str:
        tokens = _tokenize(content)
        doc_id = f"bm25_{self._counter:08d}"
        self._counter += 1
        doc = _Doc(doc_id, content, dict(metadata or {}), tokens)
        self._docs[doc_id] = doc
        for token in set(tokens):
            self._df[token] += 1
        self._total_tokens += len(tokens)
        if len(self._docs) > self._max:
            self._evict(next(iter(self._docs)))
        return doc_id

    def read(self, query: str, n: int = 5) -> list[MemoryRecord]:
        if not self._docs:
            return []
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        document_count = len(self._docs)
        average_length = self._total_tokens / document_count
        scores: dict[str, float] = {}
        for doc_id, doc in self._docs.items():
            frequencies = Counter(doc.tokens)
            score = 0.0
            document_length = len(doc.tokens)
            for token in query_tokens:
                term_frequency = frequencies.get(token, 0)
                document_frequency = self._df.get(token, 0)
                if term_frequency == 0 or document_frequency == 0:
                    continue
                inverse_frequency = math.log(
                    (document_count - document_frequency + 0.5) / (document_frequency + 0.5) + 1
                )
                denominator = term_frequency + self.K1 * (
                    1 - self.B + self.B * document_length / max(average_length, 1)
                )
                score += inverse_frequency * term_frequency * (self.K1 + 1) / denominator
            if score > 0:
                scores[doc_id] = score

        top = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:n]
        return [
            MemoryRecord(
                id=doc_id,
                text=self._docs[doc_id].content,
                metadata=self._docs[doc_id].metadata,
                score=score,
                created_at=self._docs[doc_id].created_at,
            )
            for doc_id, score in top
        ]

    def delete(self, record_id: str) -> bool:
        if record_id not in self._docs:
            return False
        self._evict(record_id)
        return True

    def health(self) -> HealthStatus:
        return HealthStatus(
            healthy=True,
            message=f"BM25: {len(self._docs)} docs",
            details={"doc_count": len(self._docs), "vocab_size": len(self._df)},
        )

    def _evict(self, doc_id: str) -> None:
        doc = self._docs.pop(doc_id, None)
        if doc is None:
            return
        for token in set(doc.tokens):
            self._df[token] = max(0, self._df[token] - 1)
            if self._df[token] == 0:
                del self._df[token]
        self._total_tokens -= len(doc.tokens)


__all__ = ["BM25MemoryTier"]
