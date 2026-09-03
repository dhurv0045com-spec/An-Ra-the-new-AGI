"""Deterministic split assignment, exact dedup, and contamination scanning.

Split assignment is a hash function of the source-disjoint deduplication
cluster key, computed before tokenization or quality inspection. Dedup
clusters never cross splits: a cluster receives exactly one split, and every
member follows it. The contamination scanner reports normalized n-gram
collisions against benchmark texts; any hit fails the pack closed.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass


SPLITS = ("training", "development", "sealed", "fresh")
_WORD = re.compile(r"[a-z0-9]+")


def _sha256_hex(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def assign_split(cluster_key: str, *, salt: str, boundaries: dict[str, float]) -> str:
    """Assign one dedup cluster to a split by hash of its key."""

    if not cluster_key or not salt:
        raise ValueError("cluster key and salt are required")
    if tuple(boundaries) != SPLITS or abs(sum(boundaries.values()) - 1.0) > 1e-9:
        raise ValueError("boundaries must cover exactly the four splits and sum to one")
    if any(fraction < 0 for fraction in boundaries.values()):
        raise ValueError("split boundaries cannot be negative")
    digest = _sha256_hex(f"{salt}\0{cluster_key}".encode("utf-8"))
    point = int(digest[:16], 16) / 16**16
    cumulative = 0.0
    for split in SPLITS:
        cumulative += boundaries[split]
        if point < cumulative or split == SPLITS[-1]:
            return split
    raise AssertionError("unreachable split assignment")


def exact_clusters(content_hashes: dict[str, str]) -> dict[str, list[str]]:
    """Group member IDs into exact-duplicate clusters by content hash."""

    clusters: dict[str, list[str]] = {}
    for member_id, content_hash in content_hashes.items():
        if not member_id:
            raise ValueError("member ids cannot be empty")
        if len(content_hash) != 64 or any(c not in "0123456789abcdef" for c in content_hash):
            raise ValueError("content hashes must be lowercase SHA-256")
        clusters.setdefault(content_hash, []).append(member_id)
    return {key: sorted(members) for key, members in sorted(clusters.items())}


def normalize_text(text: str) -> str:
    """Canonicalize text for contamination comparison."""

    return " ".join(_WORD.findall(text.casefold()))


@dataclass(frozen=True, slots=True)
class ContaminationHit:
    benchmark_id: str
    ngram: str
    occurrences: int


def scan_contamination(
    documents: dict[str, str],
    benchmarks: dict[str, str],
    *,
    ngram_order: int = 8,
) -> list[ContaminationHit]:
    """Report benchmark n-grams that occur verbatim in training documents."""

    if ngram_order < 4:
        raise ValueError("contamination n-grams must span at least 4 words")
    if not documents or not benchmarks:
        raise ValueError("documents and benchmarks are both required")
    benchmark_ngrams: dict[str, set[str]] = {}
    for benchmark_id, text in benchmarks.items():
        words = normalize_text(text).split()
        benchmark_ngrams[benchmark_id] = {
            " ".join(words[index:index + ngram_order])
            for index in range(len(words) - ngram_order + 1)
        }
    hits: list[ContaminationHit] = []
    for benchmark_id in sorted(benchmark_ngrams):
        wanted = benchmark_ngrams[benchmark_id]
        if not wanted:
            continue
        occurrences = 0
        matched: set[str] = set()
        for text in documents.values():
            words = normalize_text(text).split()
            for index in range(len(words) - ngram_order + 1):
                ngram = " ".join(words[index:index + ngram_order])
                if ngram in wanted:
                    occurrences += 1
                    matched.add(ngram)
        for ngram in sorted(matched):
            hits.append(ContaminationHit(benchmark_id, ngram, occurrences))
    return hits


__all__ = [
    "SPLITS",
    "ContaminationHit",
    "assign_split",
    "exact_clusters",
    "normalize_text",
    "scan_contamination",
]
