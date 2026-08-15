"""Common quality/memory/latency report for KV-cache backend experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class KVBackendResult:
    backend: str
    memory_bytes: int
    latency_ms: float
    perplexity_delta: float
    retrieval_delta: float
    identity_delta: float

    @property
    def promotion_allowed(self) -> bool:
        return (
            self.perplexity_delta <= 0.02
            and self.retrieval_delta >= -0.01
            and self.identity_delta >= -0.01
        )


def benchmark_table(results: list[KVBackendResult]) -> list[dict[str, object]]:
    return [{**asdict(result), "promotion_allowed": result.promotion_allowed} for result in results]
