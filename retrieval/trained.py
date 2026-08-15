"""M5 two-tower retriever behind the canonical S3 Retriever contract."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from retrieval.protocols import RetrievalHit, RetrievalProvenance, RetrievalQuery


class TwoTowerRetriever(nn.Module):
    name = "trained_two_tower"

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.query_head = nn.Linear(dim, dim, bias=False)
        self.document_head = nn.Linear(dim, dim, bias=False)
        self._documents: list[tuple[str, str, torch.Tensor]] = []

    def index(self, rows: Sequence[tuple[str, str, torch.Tensor]]) -> None:
        self._documents = [
            (record_id, text, vector.detach().float())
            for record_id, text, vector in rows
        ]

    def search(self, query: RetrievalQuery) -> list[RetrievalHit]:
        if query.vector is None:
            raise ValueError("trained retriever requires a query vector")
        vector = torch.as_tensor(query.vector, dtype=torch.float32).reshape(1, -1)
        query_vector = torch.nn.functional.normalize(self.query_head(vector), dim=-1)[0]
        scored = []
        for record_id, text, document in self._documents:
            document_vector = torch.nn.functional.normalize(
                self.document_head(document), dim=-1
            )
            score = float(torch.dot(query_vector, document_vector).detach())
            scored.append((score, record_id, text))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [
            RetrievalHit(
                record_id,
                text,
                score,
                provenance=(RetrievalProvenance(self.name, rank, score),),
            )
            for rank, (score, record_id, text) in enumerate(scored[: query.limit], 1)
        ]
