"""Contamination engine V2: layered checks with a bound receipt (M14).

Layers: exact byte hash, normalized exact match, long n-gram overlap,
near-duplicate fingerprint against benchmark texts, and generated
semantic-cluster ancestry (a train item sharing an eval cluster fails even
when surfaces differ). Train artifacts are checked BEFORE qualification;
the contamination index lives apart from the training corpus (M15).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from .near_dedup import minhash_signature, minhash_similarity
from .split import normalize_text


CONTAMINATION_SCHEMA = "anra-v5-contamination-receipt/v1"
FINGERPRINT_THRESHOLD = 0.90


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def _sha256_hex(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class ContaminationHit:
    layer: str
    train_doc_id: str
    benchmark_id: str
    detail: str


def check_exact_hashes(
    train_shas: dict[str, str], benchmark_shas: dict[str, str]
) -> list[ContaminationHit]:
    """Flag training documents byte-identical to a benchmark artifact."""

    benchmark_lookup = {digest: name for name, digest in benchmark_shas.items()}
    return [
        ContaminationHit("exact_hash", doc_id, benchmark_lookup[digest], digest[:16])
        for doc_id, digest in sorted(train_shas.items())
        if digest in benchmark_lookup
    ]


def check_normalized_exact(
    train_texts: dict[str, str], benchmark_texts: dict[str, str]
) -> list[ContaminationHit]:
    """Flag normalized-text equality after canonical normalization."""

    benchmark_lookup: dict[str, str] = {}
    for name, text in benchmark_texts.items():
        benchmark_lookup.setdefault(normalize_text(text), name)
    hits = []
    for doc_id in sorted(train_texts):
        normalized = normalize_text(train_texts[doc_id])
        if normalized and normalized in benchmark_lookup:
            hits.append(
                ContaminationHit(
                    "normalized_exact", doc_id, benchmark_lookup[normalized],
                    f"{len(normalized)} chars",
                )
            )
    return hits


def check_fingerprints(
    train_texts: dict[str, str],
    benchmark_texts: dict[str, str],
    *,
    threshold: float = FINGERPRINT_THRESHOLD,
) -> list[ContaminationHit]:
    """Flag training texts near-duplicate to a benchmark fingerprint."""

    benchmark_signatures = {
        name: minhash_signature(text) for name, text in benchmark_texts.items()
    }
    hits = []
    for doc_id in sorted(train_texts):
        signature = minhash_signature(train_texts[doc_id])
        for name in sorted(benchmark_signatures):
            similarity = minhash_similarity(signature, benchmark_signatures[name])
            if similarity >= threshold:
                hits.append(
                    ContaminationHit(
                        "fingerprint", doc_id, name, f"jaccard~{similarity:.3f}"
                    )
                )
    return hits


def check_cluster_ancestry(
    train_clusters: dict[str, str], eval_clusters: dict[str, str]
) -> list[ContaminationHit]:
    """Flag training items sharing a semantic cluster with evaluation."""

    eval_lookup = {cluster: name for name, cluster in eval_clusters.items()}
    return [
        ContaminationHit("cluster_ancestry", doc_id, eval_lookup[cluster], cluster[:16])
        for doc_id, cluster in sorted(train_clusters.items())
        if cluster in eval_lookup
    ]


def check_ngram_overlap(
    train_texts: dict[str, str],
    benchmark_texts: dict[str, str],
    *,
    ngram_order: int = 12,
) -> list[ContaminationHit]:
    """Flag per-document long n-gram overlap with benchmark texts."""

    if ngram_order < 4:
        raise ValueError("contamination n-grams must span at least 4 words")
    benchmark_ngrams: dict[str, set[str]] = {}
    for benchmark_id, text in benchmark_texts.items():
        words = normalize_text(text).split()
        benchmark_ngrams[benchmark_id] = {
            " ".join(words[index:index + ngram_order])
            for index in range(len(words) - ngram_order + 1)
        }
    hits = []
    for doc_id in sorted(train_texts):
        words = normalize_text(train_texts[doc_id]).split()
        seen = {
            " ".join(words[index:index + ngram_order])
            for index in range(len(words) - ngram_order + 1)
        }
        for benchmark_id in sorted(benchmark_ngrams):
            matched = sorted(seen & benchmark_ngrams[benchmark_id])
            for ngram in matched[:5]:
                hits.append(ContaminationHit("ngram", doc_id, benchmark_id, ngram[:80]))
    return hits


def scan_all(
    *,
    train_texts: dict[str, str],
    train_shas: dict[str, str],
    train_clusters: dict[str, str],
    benchmarks: dict[str, str],
    benchmark_shas: dict[str, str],
    eval_clusters: dict[str, str],
    ngram_order: int = 12,
) -> dict[str, object]:
    """Run every layer and bind a ContaminationReceipt."""

    hits: list[ContaminationHit] = []
    hits.extend(check_exact_hashes(train_shas, benchmark_shas))
    hits.extend(check_normalized_exact(train_texts, benchmarks))
    hits.extend(check_ngram_overlap(train_texts, benchmarks, ngram_order=ngram_order))
    hits.extend(check_fingerprints(train_texts, benchmarks))
    hits.extend(check_cluster_ancestry(train_clusters, eval_clusters))
    hits.sort(key=lambda hit: (hit.layer, hit.train_doc_id, hit.benchmark_id))
    receipt: dict[str, object] = {
        "schema": CONTAMINATION_SCHEMA,
        "ngram_order": ngram_order,
        "fingerprint_threshold": FINGERPRINT_THRESHOLD,
        "train_documents": len(train_texts),
        "benchmarks": sorted(benchmarks),
        "hits": [
            {"layer": hit.layer, "train_doc_id": hit.train_doc_id,
             "benchmark_id": hit.benchmark_id, "detail": hit.detail}
            for hit in hits
        ],
        "status": "CONTAMINATED" if hits else "CLEAN",
    }
    receipt["sha256"] = _sha256_hex(_canonical_json(receipt))
    return receipt


__all__ = [
    "CONTAMINATION_SCHEMA",
    "FINGERPRINT_THRESHOLD",
    "ContaminationHit",
    "check_cluster_ancestry",
    "check_exact_hashes",
    "check_fingerprints",
    "check_ngram_overlap",
    "check_normalized_exact",
    "scan_all",
]
