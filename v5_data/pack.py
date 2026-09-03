"""Deterministic sharded pack writer and sampler for V5 training data.

True packing: multiple short BOS-content-EOS document segments share one
padded sequence, each segment carrying a segment ID; the model applies
block-diagonal causal attention with per-segment RoPE resets downstream.
Documents longer than the native context split into deterministic
non-overlapping chunks with their own boundaries. Padding never enters the
token ledger. Every shard is content-addressed; sampler order derives from
``(run_seed, epoch, shard_hash)`` and persists as a cursor.
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


@dataclass(frozen=True, slots=True)
class PackedSequence:
    """One padded sequence holding multiple document segments."""

    tokens: tuple[int, ...]
    segment_ids: tuple[int, ...]
    sources: tuple[str, ...]
    real_tokens: int

    def __iter__(self):
        """Iterate tokens so cursor arithmetic treats the sequence as tokens."""

        return iter(self.tokens)

    def payload_bytes(self) -> bytes:
        return _canonical_json(
            {
                "tokens": list(self.tokens),
                "segment_ids": list(self.segment_ids),
                "sources": list(self.sources),
            }
        )


@dataclass(frozen=True, slots=True)
class MultiPackedShard:
    """Content-addressed shard of truly packed multi-segment sequences."""

    shard_id: str
    bucket: int
    sequences: tuple[PackedSequence, ...]
    real_tokens: int

    def payload_bytes(self) -> bytes:
        return _canonical_json(
            {
                "shard_id": self.shard_id,
                "bucket": self.bucket,
                "sequences": [
                    {
                        "tokens": list(sequence.tokens),
                        "segment_ids": list(sequence.segment_ids),
                        "sources": list(sequence.sources),
                    }
                    for sequence in self.sequences
                ],
            }
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


def pack_documents(
    documents: list[tuple[str, list[int], str]],
    *,
    bos: int,
    eos: int,
    pad: int,
    sequences_per_shard: int,
) -> tuple[list[MultiPackedShard], dict[str, int]]:
    """Pack documents into exactly-full, multi-segment, content-addressed shards.

    Each document is ``(doc_id, content_token_ids, source)``. Documents sort by
    ``doc_id`` so output is independent of input order. Chunks (BOS content
    EOS) are assigned to the smallest bucket that fits and stream-filled into
    exactly-full sequences: a chunk that overflows the remaining capacity is
    split at the token boundary, the head receiving EOS and the tail
    restarting with BOS in a later sequence. Segment isolation is preserved
    (the model applies block-diagonal attention) and the ledger stays exact.
    A sequence closes with trailing padding only when its remaining capacity
    does not divide evenly -- which surfaces as the run's final partial
    update. Returns the shards plus an audit with exact per-source totals.
    """

    if sequences_per_shard <= 0:
        raise ValueError("sequences per shard must be positive")
    if len({bos, eos, pad}) != 3:
        raise ValueError("boundary markers must be distinct")
    chunks: list[tuple[str, tuple[int, ...]]] = []
    seen_ids: set[str] = set()
    for doc_id, content, source in sorted(documents, key=lambda item: item[0]):
        if doc_id in seen_ids:
            raise ValueError(f"duplicate document id: {doc_id}")
        seen_ids.add(doc_id)
        if not source:
            raise ValueError("every document needs a source attribution")
        for chunk in chunk_document(content, bos=bos, eos=eos):
            chunks.append((source, tuple(chunk)))

    streams: dict[int, list[tuple[str, tuple[int, ...]]]] = {bucket: [] for bucket in BUCKETS}
    for source, chunk in chunks:
        streams[bucket_for(len(chunk))].append((source, chunk))

    shards: list[MultiPackedShard] = []
    ledger: dict[str, int] = {}
    total_real = 0
    total_segments = 0
    for bucket in BUCKETS:
        queue = streams[bucket]
        bucket_sequences: list[PackedSequence] = []
        while queue:
            emitted: list[tuple[str, tuple[int, ...]]] = []
            remaining = bucket
            while remaining > 0 and queue:
                source, chunk = queue[0]
                if len(chunk) > remaining:
                    if remaining >= 3:
                        # split at the token boundary; the tail restarts with
                        # BOS and the sequence closes exactly
                        head = chunk[: remaining - 1]
                        emitted.append((source, (*head, eos)))
                        queue[0] = (source, (bos, *chunk[remaining - 1 :]))
                        remaining = 0
                        continue
                    break  # remaining in {1,2}: close padded
                leftover = remaining - len(chunk)
                if leftover == 0:
                    emitted.append((source, chunk))
                    remaining = 0
                    queue.pop(0)
                elif leftover >= 3 or leftover < 0:
                    emitted.append((source, chunk))
                    remaining = leftover
                    queue.pop(0)
                else:
                    # emitting the chunk whole would strand capacity 1-2,
                    # which cannot host a valid segment: close padded
                    break
            tokens: list[int] = []
            segment_ids: list[int] = []
            sources: list[str] = []
            for index, (source, segment) in enumerate(emitted):
                tokens.extend(segment)
                segment_ids.extend([index] * len(segment))
                sources.append(source)
            if remaining:
                tokens.extend([pad] * remaining)
                segment_ids.extend([-1] * remaining)
            sequence = PackedSequence(
                tokens=tuple(tokens),
                segment_ids=tuple(segment_ids),
                sources=tuple(sources),
                real_tokens=sum(1 for segment in segment_ids if segment >= 0),
            )
            bucket_sequences.append(sequence)
            total_real += sequence.real_tokens
            total_segments += len(sources)
            for index, source in enumerate(sequence.sources):
                ledger[source] = ledger.get(source, 0) + sum(
                    1 for segment in sequence.segment_ids if segment == index
                )
        for index in range(0, len(bucket_sequences), sequences_per_shard):
            group = tuple(bucket_sequences[index:index + sequences_per_shard])
            shard_id = f"packed{bucket}-{index // sequences_per_shard:06d}"
            shards.append(
                MultiPackedShard(
                    shard_id,
                    bucket,
                    group,
                    sum(sequence.real_tokens for sequence in group),
                )
            )
    if total_real != sum(ledger.values()):
        raise ValueError("per-source ledger disagrees with packed real tokens")
    if total_real != sum(shard.real_tokens for shard in shards):
        raise ValueError("shard real-token receipts disagree with the packing ledger")
    capacity = sum(shard.bucket * len(shard.sequences) for shard in shards)
    all_sequences = [sequence for shard in shards for sequence in shard.sequences]
    full_count = sum(1 for sequence in all_sequences if -1 not in sequence.segment_ids)
    padded_count = len(all_sequences) - full_count
    audit = {
        "real_nonpad_tokens": total_real,
        "padded_capacity_tokens": capacity,
        "pack_efficiency": (total_real / capacity) if capacity else 0.0,
        "sequences": len(all_sequences),
        "full_sequences": full_count,
        "padded_sequences": padded_count,
        "segments": total_segments,
        "tokens_by_source": dict(sorted(ledger.items())),
    }
    return shards, audit


def multi_pack_ledger(shards: list[MultiPackedShard]) -> dict[str, int]:
    """Count exact real tokens and per-source totals across packed shards."""

    total = 0
    by_source: dict[str, int] = {}
    for shard in shards:
        for sequence in shard.sequences:
            total += sequence.real_tokens
            for index, source in enumerate(sequence.sources):
                count = sum(1 for segment in sequence.segment_ids if segment == index)
                if count <= 0:
                    raise ValueError("packed segment carries no tokens")
                by_source[source] = by_source.get(source, 0) + count
    if total != sum(by_source.values()) or total != sum(
        shard.real_tokens for shard in shards
    ):
        raise ValueError("multi-pack ledger disagrees with shard receipts")
    return {"real_nonpad_tokens": total, "shards": len(shards), **by_source}


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
    "MultiPackedShard",
    "PackedSequence",
    "PackedShard",
    "bucket_for",
    "build_shards",
    "chunk_document",
    "multi_pack_ledger",
    "pack_documents",
    "pack_ledger",
    "sampler_order",
]
