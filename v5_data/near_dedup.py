"""Exact + near-duplicate clustering with persisted cluster records (M9/M10).

Exact clusters group byte-identical normalized content. Near clusters use
MinHash signatures over word-5-grams with LSH banding: a scalable,
well-understood method, not a research project. Generated cognition items
additionally carry generator-emitted ``semantic_cluster_id`` values so the
same latent problem under different surfaces stays one cluster (M11).
Every cluster persists representative, members, and sources.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass


_WORD = re.compile(r"[a-z0-9]+")
NGRAM_ORDER = 5
NUM_HASHES = 64
BANDS = 16
ROWS_PER_BAND = NUM_HASHES // BANDS


def _shingles(text: str) -> set[str]:
    words = _WORD.findall(text.casefold())
    if len(words) < NGRAM_ORDER:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i:i + NGRAM_ORDER]) for i in range(len(words) - NGRAM_ORDER + 1)}


def _hash_versions(shingle: str) -> list[int]:
    return [
        int(hashlib.sha256(f"{seed}\0{shingle}".encode("utf-8")).hexdigest(), 16)
        for seed in range(NUM_HASHES)
    ]


def minhash_signature(text: str) -> tuple[int, ...]:
    """Compute a 64-permutation MinHash signature over word-5-grams."""

    signature = [2**256 - 1] * NUM_HASHES
    shingles = _shingles(text)
    if not shingles:
        return tuple([0] * NUM_HASHES)
    for shingle in shingles:
        for index, value in enumerate(_hash_versions(shingle)):
            if value < signature[index]:
                signature[index] = value
    return tuple(signature)


def minhash_similarity(first: tuple[int, ...], second: tuple[int, ...]) -> float:
    """Estimate Jaccard similarity from two signatures."""

    if len(first) != NUM_HASHES or len(second) != NUM_HASHES:
        raise ValueError("MinHash signatures must carry 64 permutations")
    return sum(1 for a, b in zip(first, second) if a == b) / NUM_HASHES


def lsh_candidates(signatures: dict[str, tuple[int, ...]]) -> list[set[str]]:
    """Band LSH retrieval: candidate near-duplicate groups (may overlap)."""

    buckets: dict[tuple[int, tuple[int, ...]], set[str]] = {}
    for doc_id, signature in signatures.items():
        for band in range(BANDS):
            key = (band, signature[band * ROWS_PER_BAND:(band + 1) * ROWS_PER_BAND])
            buckets.setdefault(key, set()).add(doc_id)
    return [members for members in buckets.values() if len(members) > 1]


@dataclass(frozen=True, slots=True)
class DedupCluster:
    cluster_id: str
    kind: str
    representative: str
    members: tuple[str, ...]
    sources: tuple[str, ...]
    similarity: float

    def assert_valid(self) -> None:
        if self.kind not in {"exact", "near", "semantic"}:
            raise ValueError(f"unknown cluster kind: {self.kind}")
        if not self.cluster_id or not self.representative:
            raise ValueError("cluster identity and representative are required")
        if self.representative not in self.members:
            raise ValueError("representative must be a member")
        if not 0.0 <= self.similarity <= 1.0:
            raise ValueError("cluster similarity must lie in [0, 1]")


def cluster_near_duplicates(
    texts: dict[str, str],
    *,
    threshold: float = 0.80,
    sources: dict[str, str] | None = None,
) -> list[DedupCluster]:
    """Cluster near-duplicate documents at or above Jaccard estimate threshold."""

    if not 0.0 < threshold <= 1.0:
        raise ValueError("near-duplicate threshold must lie in (0, 1]")
    signatures = {doc_id: minhash_signature(text) for doc_id, text in texts.items()}
    groups = lsh_candidates(signatures)
    merged = _union_groups(groups)
    clusters: list[DedupCluster] = []
    for members in sorted(merged, key=sorted):
        ordered = sorted(members)
        verified = [
            member for member in ordered
            if all(
                minhash_similarity(signatures[member], signatures[other]) >= threshold
                for other in ordered if other != member
            ) or len(ordered) == 1
        ]
        if len(verified) < 2:
            continue
        cluster_id = hashlib.sha256(
            "\0".join(["near", *verified]).encode("utf-8")
        ).hexdigest()
        source_ids = tuple(
            dict.fromkeys(
                (sources or {}).get(member, member) for member in verified
            )
        )
        clusters.append(
            DedupCluster(
                cluster_id=cluster_id, kind="near", representative=verified[0],
                members=tuple(verified), sources=source_ids, similarity=threshold,
            )
        )
    return clusters


def _union_groups(groups: list[set[str]]) -> list[set[str]]:
    parent: dict[str, str] = {}

    def find(member: str) -> str:
        while parent.get(member, member) != member:
            member = parent[member]
        return parent.get(member, member)

    for group in groups:
        members = sorted(group)
        for member in members[1:]:
            parent[find(member)] = find(members[0])
    merged: dict[str, set[str]] = {}
    universe = {member for group in groups for member in group}
    for member in universe:
        merged.setdefault(find(member), set()).add(member)
    return list(merged.values())


__all__ = [
    "BANDS",
    "NGRAM_ORDER",
    "NUM_HASHES",
    "DedupCluster",
    "cluster_near_duplicates",
    "lsh_candidates",
    "minhash_signature",
    "minhash_similarity",
]
