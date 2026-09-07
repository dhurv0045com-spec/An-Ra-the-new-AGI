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
    """One raw text document with its source attribution.

    ``raw_source_sha256`` identifies the immutable source bytes the document
    was derived from; ``text`` is the processed form actually tokenized.  Both
    identities are recorded so every training example traces back to immutable
    source identity AND the exact processing that produced it.
    """

    doc_id: str
    text: str
    source_id: str
    domain: str
    family: str
    authorization_category: str
    acquired_date: str
    raw_source_sha256: str = ""


def _merge_clusters(exact: dict[str, list[str]], near: list) -> dict[str, list[str]]:
    """Union exact clusters with near-duplicate clusters (cluster-before-split).

    Members link through shared documents; the merged canonical is the
    smallest doc id for determinism. Every merged cluster still splits as one
    unit downstream.
    """

    parent: dict[str, str] = {}

    def find(item: str) -> str:
        while parent.get(item, item) != item:
            item = parent[item]
        return parent.get(item, item)

    groups: list[set[str]] = [set(members) for members in exact.values()]
    for cluster in near:
        groups.append(set(cluster.members))
    for group in groups:
        for member in group:
            parent.setdefault(member, member)
    for group in groups:
        members = sorted(group)
        root = find(members[0])
        for member in members[1:]:
            parent[find(member)] = root
    merged: dict[str, set[str]] = {}
    for member in parent:
        merged.setdefault(find(member), set()).add(member)
    return {min(members): sorted(members) for members in merged.values()}


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
    strict_provenance: bool = False,
    near_duplicate_policy: dict[str, object] | None = None,
) -> tuple[DataManifest, dict[str, object]]:
    """Deduplicate, split, scan, and account documents into a manifest.

    Exact duplicates always cluster; ``near_duplicate_policy`` (see
    ``v5_data.near_dedup.FROZEN_NEAR_DUP_POLICY``) additionally clusters
    near-duplicates BEFORE the split, keeping whole clusters in one split.
    Without it the exact-only path runs byte-identically. ``strict_provenance``
    requires real raw-source identity on every document (production); without
    it development fixtures may fall back to the processed-text hash.
    """

    if not documents:
        raise ValueError("data manifest requires documents")
    if strict_provenance:
        missing = sorted(document.doc_id for document in documents
                         if not document.raw_source_sha256)
        if missing:
            raise ValueError(
                "production provenance requires raw_source_sha256; "
                f"missing on {len(missing)} documents")
    processed_hashes = {document.doc_id: _sha256_text(document.text) for document in documents}
    content_hashes = dict(processed_hashes)
    clusters = exact_clusters({doc_id: digest for doc_id, digest in content_hashes.items()})
    near_stats: dict[str, object] = {"enabled": False}
    if near_duplicate_policy is not None:
        from .near_dedup import cluster_near_duplicates
        near_clusters = cluster_near_duplicates(
            {document.doc_id: document.text for document in documents},
            threshold=float(near_duplicate_policy.get("threshold", 0.80)))
        merged = _merge_clusters(clusters, near_clusters)
        near_stats = {"enabled": True,
                      "method": str(near_duplicate_policy.get("method", "minhash-lsh")),
                      "version": str(near_duplicate_policy.get("version", "v1")),
                      "threshold": float(near_duplicate_policy.get("threshold", 0.80)),
                      "near_clusters": len(near_clusters),
                      "merged_clusters": len(merged)}
        clusters = merged
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
    benchmark_contents = {name: _sha256_text(text)
                          for name, text in (contamination_benchmarks or {}).items()}
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
            "benchmarks": sorted(benchmark_contents),
            "benchmark_contents": dict(sorted(benchmark_contents.items())),
            "hits": [],
        }
    )
    scan_sha256 = hashlib.sha256(scan_payload).hexdigest()

    records: list[SourceRecord] = []
    tokens_by_family: dict[str, int] = {}
    for document in kept:
        raw_source = document.raw_source_sha256 or content_hashes[document.doc_id]
        record = SourceRecord(
            source_id=document.source_id,
            authorization_category=document.authorization_category,
            acquired_date=document.acquired_date,
            raw_sha256=raw_source,
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
        "raw_source_sha256": {
            document.doc_id: (document.raw_source_sha256 or content_hashes[document.doc_id])
            for document in kept
        },
        "processed_document_sha256": dict(sorted(processed_hashes.items())),
        "exact_duplicate_drops": dict(sorted(dropped.items())),
        "near_duplicate": near_stats,
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
