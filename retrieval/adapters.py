from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from typing import Any

from retrieval.protocols import RetrievalHit, RetrievalProvenance, RetrievalQuery


def _metadata_matches(metadata: Mapping[str, object], filters: Mapping[str, object]) -> bool:
    return all(metadata.get(key) == value for key, value in filters.items())


class BM25RetrieverAdapter:
    name = "bm25"

    def __init__(self, tier: object) -> None:
        self.tier = tier

    def search(self, query: RetrievalQuery) -> list[RetrievalHit]:
        rows = self.tier.read(
            query.text,
            n=query.limit * query.candidate_multiplier,
        )
        hits: list[RetrievalHit] = []
        for rank, row in enumerate(rows, start=1):
            metadata = dict(getattr(row, "metadata", {}) or {})
            if not _metadata_matches(metadata, query.filters):
                continue
            record_id = str(metadata.get("canonical_id") or getattr(row, "id", ""))
            score = float(getattr(row, "score", 0.0))
            hits.append(
                RetrievalHit(
                    id=record_id,
                    text=str(getattr(row, "text", getattr(row, "content", ""))),
                    score=score,
                    metadata=metadata,
                    provenance=(RetrievalProvenance(self.name, rank, score),),
                )
            )
        return hits[: query.limit * query.candidate_multiplier]


class VectorRetrieverAdapter:
    name = "semantic"

    def __init__(self, store: object, embed: Callable[[str], Any]) -> None:
        self.store = store
        self.embed = embed

    def search(self, query: RetrievalQuery) -> list[RetrievalHit]:
        vector = query.vector if query.vector is not None else self.embed(query.text)
        rows = self.store.search(vector, k=query.limit * query.candidate_multiplier)
        hits: list[RetrievalHit] = []
        for rank, row in enumerate(rows, start=1):
            payload = dict(row.get("payload", {}) or {})
            if not _metadata_matches(payload, query.filters):
                continue
            score = float(row.get("score", 0.0))
            hits.append(
                RetrievalHit(
                    id=str(row.get("record_id", "")),
                    text=str(payload.get("content", "")),
                    score=score,
                    metadata=payload,
                    provenance=(RetrievalProvenance(self.name, rank, score),),
                )
            )
        return hits


class SkillLibraryRetrieverAdapter:
    """Expose the existing agent skill library through the shared protocol."""

    name = "skills"

    def __init__(self, library: object) -> None:
        self.library = library

    def search(self, query: RetrievalQuery) -> list[RetrievalHit]:
        skills = self.library.retrieve(query.text, top_k=query.limit * query.candidate_multiplier)
        hits: list[RetrievalHit] = []
        for rank, skill in enumerate(skills, start=1):
            data = vars(skill) if hasattr(skill, "__dict__") else {}
            metadata = dict(data)
            if not _metadata_matches(metadata, query.filters):
                continue
            skill_id = str(
                data.get("skill_id")
                or hashlib.sha256(repr(skill).encode()).hexdigest()[:16]
            )
            text = " ".join(
                part
                for part in (
                    str(data.get("name", "")),
                    str(data.get("description", "")),
                    str(data.get("example", "")),
                )
                if part
            )
            raw_score = float(data.get("avg_score", 0.0))
            hits.append(
                RetrievalHit(
                    id=skill_id,
                    text=text,
                    score=raw_score,
                    metadata=metadata,
                    provenance=(RetrievalProvenance(self.name, rank, raw_score),),
                )
            )
        return hits[: query.limit]
