"""Executable data-manifest construction: dedup, cluster split, contamination.

Turns a folder of real text documents into a validated
``v5_contracts.data_spec.DataManifest``. The order is fixed by design:
content-hash identity, exact-duplicate clustering (only the canonical member
survives and the drops are recorded), cluster-level split assignment so
duplicates can never straddle splits, contamination scanning against held-out
benchmarks (any hit fails closed), then token accounting per family.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Callable, Mapping

from v5_contracts.data_spec import DataManifest, SourceRecord, assert_source_disjoint

from .split import assign_split, exact_clusters, normalize_text, scan_contamination


MANIFEST_SCHEMA = "anra-v5-data-manifest/v1"


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


@dataclass(frozen=True, slots=True)
class Document:
    """One raw text document with its source attribution."""

    doc_id: str
    text: str
    source_id: str
    domain: str
    family: str
    authorization_category: str
    acquired_date: str


def build_data_manifest(
    documents: list[Document],
    *,
    manifest_id: str,
    tokenizer_sha256: str,
    filter_version: str,
    dedup_version: str,
    split_salt: str,
    split_boundaries: Mapping[str, float],
    count_tokens: Callable[[str], int],
    contamination_benchmarks: Mapping[str, str] | None = None,
    ngram_order: int = 8,
) -> tuple[DataManifest, dict[str, object]]:
    """Deduplicate, split, scan, and account documents into a manifest."""

    if not documents:
        raise ValueError("data manifest requires documents")
    content_hashes = {document.doc_id: _sha256_text(document.text) for document in documents}
    clusters = exact_clusters({doc_id: digest for doc_id, digest in content_hashes.items()})
    dropped: dict[str, str] = {}
    kept: list[Document] = []
    split_of: dict[str, str] = {}
    for digest, members in clusters.items():
        canonical = members[0]
        for member in members[1:]:
            dropped[member] = canonical
        kept.append(next(document for document in documents if document.doc_id == canonical))
        split_of[canonical] = assign_split(
            f"{digest}", salt=split_salt, boundaries=dict(split_boundaries)
        )
    kept.sort(key=lambda document: document.doc_id)

    hits = []
    if contamination_benchmarks:
        hits = scan_contamination(
            {document.doc_id: document.text for document in kept},
            dict(contamination_benchmarks),
            ngram_order=ngram_order,
        )
        if hits:
            raise ValueError(
                f"abort CONTAMINATION: {len(hits)} benchmark n-gram collisions; pack fails closed"
            )
    scan_payload = _canonical_json(
        {
            "ngram_order": ngram_order,
            "benchmarks": sorted(contamination_benchmarks or {}),
            "hits": [],
        }
    )
    scan_sha256 = hashlib.sha256(scan_payload).hexdigest()

    records: list[SourceRecord] = []
    tokens_by_family: dict[str, int] = {}
    for document in kept:
        record = SourceRecord(
            source_id=document.source_id,
            authorization_category=document.authorization_category,
            acquired_date=document.acquired_date,
            raw_sha256=content_hashes[document.doc_id],
            split=split_of[document.doc_id],
            domain=document.domain,
        )
        records.append(record)
        tokens = count_tokens(document.text) + 2  # BOS/EOS segment overhead
        tokens_by_family[document.family] = tokens_by_family.get(document.family, 0) + tokens
    manifest = DataManifest(
        schema=MANIFEST_SCHEMA,
        manifest_id=manifest_id,
        tokenizer_sha256=tokenizer_sha256,
        filter_version=filter_version,
        dedup_version=dedup_version,
        contamination_scan_sha256=scan_sha256,
        sources=tuple(records),
        tokens_by_family=dict(sorted(tokens_by_family.items())),
        total_tokens=sum(tokens_by_family.values()),
    )
    manifest.assert_valid()
    audit = {
        "documents_ingested": len(documents),
        "exact_duplicate_drops": dict(sorted(dropped.items())),
        "clusters": len(clusters),
        "split_counts": {
            split: sum(1 for value in split_of.values() if value == split)
            for split in sorted(set(split_of.values()))
        },
        "contamination_hits": 0,
        "normalization": "casefold + word extraction for contamination comparison only",
        "note": "count_tokens(text)+2 approximates BOS/EOS overhead; the pack manifest carries exact counts",
    }
    return manifest, audit


def manifest_sha256(manifest: DataManifest) -> str:
    """Canonical hash of a validated manifest."""

    manifest.assert_valid()
    return hashlib.sha256(_canonical_json(_manifest_dict(manifest))).hexdigest()


def _manifest_dict(manifest: DataManifest) -> dict[str, object]:
    from dataclasses import asdict

    return asdict(manifest)


def assert_manifests_source_disjoint(*manifests: DataManifest) -> None:
    """Public re-export so pipelines fail closed on source reuse."""

    assert_source_disjoint(*manifests)


__all__ = [
    "MANIFEST_SCHEMA",
    "Document",
    "assert_manifests_source_disjoint",
    "build_data_manifest",
    "manifest_sha256",
    "normalize_text",
]
