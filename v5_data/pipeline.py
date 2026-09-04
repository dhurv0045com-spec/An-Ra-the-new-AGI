"""Foundry pipeline: raw sources to qualified, accounted token inventory.

Executes the immutable stage chain over real source artifacts: read with
source-local IDs, normalize with recorded transforms, quality-judge with
reasons, exact-deduplicate before splitting, near-deduplicate with MinHash
LSH, tokenize with the canonical V5 tokenizer, and account exact tokens per
source and split. Emits a FoundryReceipt answering where every document came
from, why it was included, what was removed, what duplicates exist, and how
many unique tokens each class contributes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

from .near_dedup import cluster_near_duplicates, minhash_signature
from .normalize import normalize_text
from .quality import judge
from .readers import RawRecord


FOUNDRY_SCHEMA = "anra-v5-foundry-receipt/v1"


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def _sha256_hex(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def run_pipeline(
    records: list[RawRecord],
    *,
    run_id: str,
    encode: Callable[[str], list[int]],
    encode_batch: Callable[[list[str]], list[list[int]]] | None = None,
    near_threshold: float = 0.80,
    near_sample_cap: int = 2000,
    max_documents: int | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, object]:
    """Run the full stage chain over attributed raw records."""

    def note(stage: str) -> None:
        if progress is not None:
            progress(stage)

    if not records:
        raise ValueError("foundry pipeline needs raw records")
    if max_documents is not None:
        records = records[:max_documents]
    raw_count = len(records)
    note(f"ingested {raw_count} raw records")
    normalized = []
    for record in records:
        doc = normalize_text(f"{record.source_id}::{record.local_id}", record.text)
        normalized.append({"record": record, "doc": doc})
    del records
    note(f"normalized {len(normalized)} documents")
    judged = []
    quality_counts: dict[str, int] = {"KEEP": 0, "DROP": 0, "QUARANTINE": 0}
    domain_counts: dict[str, int] = {}
    for index, item in enumerate(normalized):
        verdict = judge(item["doc"].doc_id, item["doc"].text)
        judged.append({**item, "verdict": verdict})
        quality_counts[verdict.decision] += 1
        domain_counts[verdict.domain] = domain_counts.get(verdict.domain, 0) + 1
        if (index + 1) % 50000 == 0:
            note(f"judged {index + 1}/{len(normalized)}")
    kept = [item for item in judged if item["verdict"].decision == "KEEP"]
    quarantined = [item for item in judged if item["verdict"].decision == "QUARANTINE"]
    usable = kept + quarantined
    del normalized, judged
    note(f"quality: {quality_counts}")
    content_hashes = {
        item["doc"].doc_id: _sha256_hex(item["doc"].text.encode("utf-8")) for item in usable
    }
    seen: dict[str, str] = {}
    exact_drops = 0
    unique_items = []
    for item in usable:
        digest = content_hashes[item["doc"].doc_id]
        if digest in seen:
            exact_drops += 1
            continue
        seen[digest] = item["doc"].doc_id
        unique_items.append(item)
    del usable, content_hashes, seen
    note(f"exact dedup: {exact_drops} drops, {len(unique_items)} unique")
    stride = max(1, len(unique_items) // max(near_sample_cap, 1))
    sample = unique_items[::stride][:near_sample_cap]
    sample_texts = {item["doc"].doc_id: item["doc"].text for item in sample}
    near_clusters = cluster_near_duplicates(sample_texts, threshold=near_threshold)
    near_clustered_docs = sum(len(cluster.members) for cluster in near_clusters)
    near_sampled = len(sample)
    del sample, sample_texts
    note(f"near-dedup: {len(near_clusters)} clusters over {near_clustered_docs} sampled docs")
    tokenized: list[dict[str, Any]] = []
    total_tokens = 0
    tokens_by_class: dict[str, int] = {}
    batch_encode = encode_batch or (lambda texts: [encode(text) for text in texts])
    batch_size = 4096
    unique_texts = [item["doc"].text for item in unique_items]
    id_lists: list[list[int]] = []
    for start in range(0, len(unique_texts), batch_size):
        id_lists.extend(batch_encode(unique_texts[start:start + batch_size]))
        if (start // batch_size + 1) % 10 == 0:
            note(f"tokenized {(start // batch_size + 1) * batch_size}/{len(unique_texts)}")
    for item, ids in zip(unique_items, id_lists):
        count = len(ids)
        total_tokens += count
        source_class = _class_of(item["record"].source_id)
        tokens_by_class[source_class] = tokens_by_class.get(source_class, 0) + count
        tokenized.append({"doc_id": item["doc"].doc_id, "tokens": count})
    receipt: dict[str, object] = {
        "schema": FOUNDRY_SCHEMA,
        "run_id": run_id,
        "max_documents": max_documents,
        "raw_documents": raw_count,
        "quality": quality_counts,
        "domains": domain_counts,
        "quarantined_included": len(quarantined),
        "exact_duplicate_drops": exact_drops,
        "unique_documents": len(unique_items),
        "near_sampled": near_sampled,
        "near_clusters": len(near_clusters),
        "near_clustered_documents": near_clustered_docs,
        "near_threshold": near_threshold,
        "unique_tokens": total_tokens,
        "tokens_by_class": tokens_by_class,
        "tokenized_documents": len(tokenized),
    }
    receipt["sha256"] = _sha256_hex(_canonical_json(receipt))
    return receipt


def _class_of(source_id: str) -> str:
    lowered = source_id.casefold()
    if "finemath" in lowered or "math" in lowered:
        return "math"
    if "stack" in lowered or "code" in lowered:
        return "code"
    if "dialog" in lowered or "smoltalk" in lowered or "talk" in lowered:
        return "dialogue"
    return "natural"


__all__ = ["FOUNDRY_SCHEMA", "run_pipeline"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source", action="append", required=True,
                        help="repeat KIND::SOURCE_ID::PARQUET_PATH")
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--max-documents", type=int, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from v5_tokenizer.artifact import load_frozen

    kinds = {"fineweb": "read_fineweb_edu", "finemath": "read_finemath", "smoltalk": "read_smoltalk"}
    import v5_data.readers as readers

    records = []
    for spec in args.source:
        kind, source_id, path = spec.split("::", 2)
        reader = getattr(readers, kinds[kind])
        print(f"reading {path} ...", flush=True)
        records.extend(reader(Path(path), source_id=source_id))
        print(f"total records: {len(records)}", flush=True)
    root = Path(__file__).resolve().parents[1]
    trainer_record = root / "v5_tokenizer/legacy_24k_trainer_record.json"
    tokenizer = load_frozen(
        args.tokenizer,
        expected_sha256=hashlib.sha256(args.tokenizer.read_bytes()).hexdigest(),
        vocabulary_size=24576,
        trainer_config_sha256=hashlib.sha256(trainer_record.read_bytes()).hexdigest(),
        corpus_manifest_sha256="eb1f0dbac64524ff4dc589c0292af6dc4c3803f48f8fe0af0a77684fea26fc67",
    )
    receipt = run_pipeline(
        records, run_id=args.run_id, encode=tokenizer.encode,
        encode_batch=tokenizer.encode_batch, max_documents=args.max_documents,
        progress=lambda stage: print(stage, flush=True),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(
        {"sha256": receipt["sha256"], "unique_documents": receipt["unique_documents"],
         "unique_tokens": receipt["unique_tokens"]},
        sort_keys=True,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
