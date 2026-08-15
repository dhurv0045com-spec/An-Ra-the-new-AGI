from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict

from retrieval.protocols import RetrievalHit, RetrievalProvenance, RetrievalQuery, Retriever


class HybridRetriever:
    """Deterministic weighted reciprocal-rank fusion over shared retrievers."""

    name = "hybrid"

    def __init__(
        self,
        retrievers: Sequence[Retriever],
        *,
        weights: Mapping[str, float] | None = None,
        rrf_k: int = 60,
    ) -> None:
        if not retrievers:
            raise ValueError("hybrid retrieval requires at least one retriever")
        if rrf_k < 0:
            raise ValueError("rrf_k cannot be negative")
        names = [retriever.name for retriever in retrievers]
        if len(names) != len(set(names)):
            raise ValueError("retriever names must be unique")
        self.retrievers = tuple(retrievers)
        self.weights = {name: float((weights or {}).get(name, 1.0)) for name in names}
        if any(weight < 0.0 for weight in self.weights.values()):
            raise ValueError("retriever weights cannot be negative")
        self.rrf_k = int(rrf_k)

    def search(self, query: RetrievalQuery) -> list[RetrievalHit]:
        fused: dict[str, dict[str, object]] = {}
        for retriever in self.retrievers:
            weight = self.weights[retriever.name]
            if weight == 0.0:
                continue
            rows = retriever.search(query)
            for rank, row in enumerate(rows, start=1):
                entry = fused.setdefault(
                    row.id,
                    {
                        "id": row.id,
                        "text": row.text,
                        "metadata": dict(row.metadata),
                        "score": 0.0,
                        "provenance": [],
                    },
                )
                entry["score"] = float(entry["score"]) + weight / (self.rrf_k + rank)
                provenance = entry["provenance"]
                assert isinstance(provenance, list)
                provenance.append(
                    RetrievalProvenance(
                        retriever=retriever.name,
                        rank=rank,
                        raw_score=float(row.score),
                        weight=weight,
                    )
                )
                if not entry["text"] and row.text:
                    entry["text"] = row.text
                metadata = entry["metadata"]
                assert isinstance(metadata, dict)
                for key, value in row.metadata.items():
                    metadata.setdefault(key, value)

        hits = [
            RetrievalHit(
                id=str(entry["id"]),
                text=str(entry["text"]),
                score=float(entry["score"]),
                metadata=dict(entry["metadata"]),
                provenance=tuple(entry["provenance"]),
            )
            for entry in fused.values()
        ]
        hits.sort(key=lambda hit: (-hit.score, hit.id))
        selected = hits[: query.limit]
        self._record(query, selected)
        return selected

    @staticmethod
    def _record(query: RetrievalQuery, hits: Sequence[RetrievalHit]) -> None:
        try:
            from runtime.experience_ledger import record_experience

            record_experience(
                trace_id=query.trace_id,
                kind="retrieval",
                inputs={"text": query.text, "filters": query.filters, "limit": query.limit},
                output=[
                    {
                        "id": hit.id,
                        "score": hit.score,
                        "provenance": [asdict(item) for item in hit.provenance],
                    }
                    for hit in hits
                ],
                gate_record={"allowed": True, "gate": "retrieval_policy"},
                source="retrieval.hybrid",
                metadata={"hit_count": len(hits)},
            )
        except Exception:
            pass
