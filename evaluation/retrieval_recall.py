"""Deterministic hybrid/memory recall gate used by CI and pilot reports."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass

from retrieval.protocols import RetrievalQuery, Retriever


@dataclass(frozen=True)
class RecallCase:
    query: str
    expected_ids: tuple[str, ...]


def evaluate_recall(
    retriever: Retriever,
    cases: Iterable[RecallCase],
    *,
    ks: tuple[int, ...] = (5, 20, 50),
    minimums: tuple[float, ...] = (0.90, 0.80, 0.70),
) -> dict[str, object]:
    """Measure hit recall at each k; fail closed on an empty suite."""
    if len(ks) != len(minimums) or not ks or any(k <= 0 for k in ks):
        raise ValueError("ks and minimums must be matching positive sequences")
    rows = list(cases)
    if not rows:
        raise ValueError("recall suite cannot be empty")
    metrics: dict[str, dict[str, object]] = {}
    for k, minimum in zip(ks, minimums, strict=True):
        matched = 0
        for case in rows:
            actual = {hit.id for hit in retriever.search(RetrievalQuery(case.query, limit=k))}
            matched += int(bool(actual.intersection(case.expected_ids)))
        recall = matched / len(rows)
        metrics[f"recall_at_{k}"] = {
            "value": recall,
            "minimum": minimum,
            "passed": recall >= minimum,
        }
    return {
        "schema_version": 1,
        "retriever": retriever.name,
        "cases": [asdict(case) for case in rows],
        "metrics": metrics,
        "passed": all(bool(metric["passed"]) for metric in metrics.values()),
    }
