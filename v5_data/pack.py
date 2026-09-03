"""Deterministic sharded pack writer and sampler for V5 training data.

Documents become BOS-content-EOS segments with block-diagonal attention
between segments and position IDs reset per segment (enforced downstream by
the packed layout). Documents longer than the native context split into
deterministic non-overlapping chunks with their own boundaries. Padding never
enters the token ledger. Every shard is content-addressed; sampler order
derives from ``(run_seed, epoch, shard_hash)`` and persists as a cursor.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass


BUCKETS = (512, 1024, 2048, 4096)
NATIVE_CONTEXT = 4096


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


@dataclass(frozen=True, slots=True)
class PackedShard:
    shard_id: str
    bucket: int
    sequences: tuple[tuple[int, ...], ...]
    real_tokens: int

    def payload_bytes(self) -> bytes:
        return _canonical_json(
            {"shard_id": self.shard_id, "bucket": self.bucket,
             "sequences": [list(sequence) for sequence in self.sequences]}
        )

    def sha256(self) -> str:
        return hashlib.sha256(self.payload_bytes()).hexdigest()


def chunk_document(token_ids: list[int], *, bos: int, eos: int) -> list[list[int]]:
    """Split over-long content into bounded BOS/EOS chunks."""

    if any(not isinstance(token, int) or token < 0 for token in token_ids):
        raise ValueError("content token ids must be nonnegative integers")
    capacity = NATIVE_CONTEXT - 2
    if len(token_ids) <= capacity:
        return [[bos, *token_ids, eos]]
    return [
        [bos, *token_ids[start:start + capacity], eos]
        for start in range(0, len(token_ids), capacity)
    ]


def bucket_for(length: int) -> int:
    """Return the smallest bucket that fits a segment."""

    for bucket in BUCKETS:
        if length <= bucket:
            return bucket
    raise ValueError("segment exceeds the native context")


def build_shards(
    documents: list[tuple[str, list[int], str]],
    *,
    bos: int,
    eos: int,
    pad: int,
    sequences_per_shard: int,
) -> list[PackedShard]:
    """Pack documents into padded, content-addressed shards.

    Each document is ``(doc_id, content_token_ids, source)``. Documents sort
    by ``doc_id`` before packing so output is independent of input order.
    """

    if sequences_per_shard <= 0:
        raise ValueError("sequences per shard must be positive")
    if len({bos, eos, pad}) != 3:
        raise ValueError("boundary markers must be distinct")
    bucketed: dict[int, list[tuple[int, ...]]] = {bucket: [] for bucket in BUCKETS}
    seen_ids: set[str] = set()
    for doc_id, content, _source in sorted(documents, key=lambda item: item[0]):
        if doc_id in seen_ids:
            raise ValueError(f"duplicate document id: {doc_id}")
        seen_ids.add(doc_id)
        for segment in chunk_document(content, bos=bos, eos=eos):
            bucket = bucket_for(len(segment))
            padded = tuple(segment + [pad] * (bucket - len(segment)))
            bucketed[bucket].append(padded)
    shards: list[PackedShard] = []
    for bucket in BUCKETS:
        sequences = bucketed[bucket]
        for index in range(0, len(sequences), sequences_per_shard):
            group = tuple(sequences[index:index + sequences_per_shard])
            shard_id = f"bucket{bucket}-{index // sequences_per_shard:06d}"
            real = sum(len([t for t in sequence if t != pad]) for sequence in group)
            shards.append(PackedShard(shard_id, bucket, group, real))
    return shards


def pack_ledger(shards: list[PackedShard], *, pad: int) -> dict[str, int]:
    """Count exact real (non-padding) tokens across shards."""

    total = 0
    for shard in shards:
        for sequence in shard.sequences:
            total += sum(1 for token in sequence if token != pad)
    if total != sum(shard.real_tokens for shard in shards):
        raise ValueError("shard real-token receipts disagree with the ledger")
    return {"real_nonpad_tokens": total, "shards": len(shards)}


def sampler_order(shard_hashes: list[str], *, run_seed: int, epoch: int) -> list[int]:
    """Derive a deterministic shard visit order from seed, epoch, and hashes."""

    if run_seed < 0 or epoch < 0:
        raise ValueError("seed and epoch cannot be negative")
    if len(set(shard_hashes)) != len(shard_hashes):
        raise ValueError("shard hashes must be distinct")
    keyed = [
        (hashlib.sha256(f"{run_seed}/{epoch}/{digest}".encode()).hexdigest(), index)
        for index, digest in enumerate(shard_hashes)
    ]
    return [index for _, index in sorted(keyed)]


__all__ = [
    "BUCKETS",
    "NATIVE_CONTEXT",
    "PackedShard",
    "bucket_for",
    "build_shards",
    "chunk_document",
    "pack_ledger",
    "sampler_order",
]
