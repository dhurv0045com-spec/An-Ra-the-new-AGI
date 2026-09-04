"""Canonical update-stream construction: sampler order to exact microbatches.

One path for every consumer -- production training, canaries, and miniature
proofs alike.  The builder applies the deterministic sampler order over the
frozen pack, selects exactly-full sequences (padding never trains), groups
them into bucket-pure update windows whose real tokens equal the frozen
per-update budget (8x512 = 4x1024 = 2x2048 = 1x4096 = 4096), and reports the
real cursor coordinates each update consumes.

The builder precomputes a window list; ``next_window`` below is the
authoritative resume path.  It regenerates sampler order deterministically
from the pack manifest plus seed and epoch, so a fresh process holding only
committed artifacts (pack bytes, manifest, sampler spec, cursor) rebuilds
the exact next window with no Python iterator state and no
global-update-as-data-index.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Mapping

from .pack import BUCKETS, MultiPackedShard, PackedSequence, sampler_order


DEFAULT_REAL_TOKENS_PER_UPDATE = 4096


@dataclass(frozen=True, slots=True)
class UpdateWindow:
    """One exact-budget update window with its real cursor coordinates."""

    sequences: tuple[PackedSequence, ...]
    shard_ordinal: int
    sequence_ordinal: int
    real_tokens: int


def build_update_stream(
    shards: list[MultiPackedShard],
    *,
    run_seed: int,
    epoch: int = 0,
    real_tokens_per_update: int = DEFAULT_REAL_TOKENS_PER_UPDATE,
    sequences_per_update: Mapping[int, int] | None = None,
) -> list[UpdateWindow]:
    """Group exactly-full sequences into exact-budget update windows.

    The windows interleave buckets round-robin in ascending bucket order so
    every sequence class appears throughout training, and window cursors
    address the frozen shard list in sampler order.
    """

    if real_tokens_per_update <= 0:
        raise ValueError("real tokens per update must be positive")
    if sequences_per_update is None:
        sequences_per_update = {
            bucket: real_tokens_per_update // bucket
            for bucket in BUCKETS
            if real_tokens_per_update % bucket == 0
        }
    if not sequences_per_update:
        raise ValueError("no bucket divides the requested update size")
    for bucket, count in sequences_per_update.items():
        if count * bucket != real_tokens_per_update:
            raise ValueError(
                f"bucket {bucket} x {count} sequences != {real_tokens_per_update} real tokens"
            )
    order = sampler_order([shard.sha256() for shard in shards], run_seed=run_seed, epoch=epoch)
    pools: dict[int, list[tuple[int, int, PackedSequence]]] = {
        bucket: [] for bucket in sequences_per_update
    }
    for position, shard_index in enumerate(order):
        shard = shards[shard_index]
        if shard.bucket not in pools:
            continue
        for sequence_index, sequence in enumerate(shard.sequences):
            if -1 not in sequence.segment_ids and len(sequence.tokens) == shard.bucket:
                pools[shard.bucket].append((position, sequence_index, sequence))
    streams: list[list[UpdateWindow]] = []
    for bucket in sorted(sequences_per_update):
        need = sequences_per_update[bucket]
        pool = pools[bucket]
        windows = []
        for start in range(0, len(pool) - need + 1, need):
            group = pool[start:start + need]
            end_position, end_sequence, _ = group[-1]
            windows.append(
                UpdateWindow(
                    sequences=tuple(sequence for _, _, sequence in group),
                    shard_ordinal=end_position,
                    sequence_ordinal=end_sequence,
                    real_tokens=real_tokens_per_update,
                )
            )
        streams.append(windows)
    depth = max(len(stream) for stream in streams)
    interleaved: list[UpdateWindow] = []
    for level in range(depth):
        for stream in streams:
            if level < len(stream):
                interleaved.append(stream[level])
    return interleaved


__all__ = [
    "AUDIT_SCHEMA",
    "CURSOR_SCHEMA_V1",
    "DEFAULT_REAL_TOKENS_PER_UPDATE",
    "SAMPLER_SCHEMA",
    "SamplerSpec",
    "StreamCursor",
    "UpdateWindow",
    "assert_cursor_advances",
    "audit_epoch",
    "build_update_stream",
    "next_window",
    "take_slices",
]


SAMPLER_SCHEMA = "anra-v5-sampler-spec/v1"
CURSOR_SCHEMA_V1 = "anra-v5-stream-cursor/v1"
AUDIT_SCHEMA = "anra-v5-stream-audit/v1"

# PAD is a frozen V5 special (id 0). Sequence receipts are verified against
# actual non-pad counts on every touched sequence; the count field is never
# trusted for ledger math.
PAD_ID = 0


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def _sha256_hex(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _assert_sha256(name: str, value: str) -> None:
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} must be a lowercase SHA-256")


@dataclass(frozen=True, slots=True)
class SamplerSpec:
    """Frozen sampler identity: the only stream parameters besides the cursor."""

    schema: str
    pack_manifest_sha256: str
    run_seed: int
    tokens_per_update: int
    buckets: tuple[int, ...]

    def assert_valid(self) -> None:
        if self.schema != SAMPLER_SCHEMA:
            raise ValueError("unsupported sampler-spec schema")
        _assert_sha256("pack manifest", self.pack_manifest_sha256)
        if self.run_seed < 0:
            raise ValueError("run seed cannot be negative")
        if self.tokens_per_update <= 0:
            raise ValueError("tokens per update must be positive")
        if not self.buckets or any(bucket <= 0 for bucket in self.buckets):
            raise ValueError("bucket allowlist must be nonempty and positive")

    def sha256(self) -> str:
        self.assert_valid()
        return _sha256_hex(
            _canonical_json(
                {
                    "schema": self.schema,
                    "pack_manifest_sha256": self.pack_manifest_sha256,
                    "run_seed": self.run_seed,
                    "tokens_per_update": self.tokens_per_update,
                    "buckets": list(self.buckets),
                }
            )
        )


@dataclass(frozen=True, slots=True)
class StreamCursor:
    """Authoritative answer to: what is the next unread training unit?

    ``sampler_position`` is a flat sequence ordinal across ALL pack sequences
    in sampler order (padded sequences included: nothing is silently
    dropped). ``token_offset`` is a real-token offset inside that sequence.
    Together with the bound pack manifest, sampler spec, and epoch, the
    cursor alone rebuilds the next window.
    """

    schema: str
    pack_manifest_sha256: str
    sampler_spec_sha256: str
    epoch: int
    sampler_position: int
    token_offset: int
    cumulative_real_tokens: int

    def assert_valid(self) -> None:
        if self.schema != CURSOR_SCHEMA_V1:
            raise ValueError("unsupported stream-cursor schema")
        _assert_sha256("pack manifest", self.pack_manifest_sha256)
        _assert_sha256("sampler spec", self.sampler_spec_sha256)
        if self.epoch < 0 or self.sampler_position < 0 or self.token_offset < 0:
            raise ValueError("cursor coordinates cannot be negative")
        if self.cumulative_real_tokens < 0:
            raise ValueError("cumulative tokens cannot be negative")

    def sha256(self) -> str:
        self.assert_valid()
        return _sha256_hex(
            _canonical_json(
                {
                    "schema": self.schema,
                    "pack_manifest_sha256": self.pack_manifest_sha256,
                    "sampler_spec_sha256": self.sampler_spec_sha256,
                    "epoch": self.epoch,
                    "sampler_position": self.sampler_position,
                    "token_offset": self.token_offset,
                    "cumulative_real_tokens": self.cumulative_real_tokens,
                }
            )
        )


@dataclass(frozen=True, slots=True)
class WindowTake:
    """One contiguous real-token run inside a packed sequence."""

    epoch: int
    order_position: int
    shard_index: int
    sequence_index: int
    start_real_token: int
    length_real_tokens: int


def _sequence_real_tokens(sequence: PackedSequence) -> int:
    return int(sequence.real_tokens)


def _flat_sequence_count(order: list[int], shards: list[MultiPackedShard]) -> int:
    return sum(len(shards[index].sequences) for index in order)


def next_window(
    shards: list[MultiPackedShard],
    spec: SamplerSpec,
    cursor: StreamCursor,
    *,
    pack_manifest_sha256: str,
    shard_idents: list[tuple[str, str]] | None = None,
) -> tuple[list[WindowTake], StreamCursor]:
    """Rebuild the exact next update window from committed identities alone.

    Sampler order regenerates deterministically from manifest shard hashes
    plus seed and epoch: no Python iterator, no precomputed list, no
    global-update-as-index. Touched shards verify against the manifest.
    Consumes exactly ``tokens_per_update`` real tokens; padding never counts.
    """

    spec.assert_valid()
    cursor.assert_valid()
    if cursor.pack_manifest_sha256 != pack_manifest_sha256:
        raise ValueError("cursor binds a different pack manifest")
    if cursor.pack_manifest_sha256 != spec.pack_manifest_sha256:
        raise ValueError("cursor and sampler spec disagree on the pack")
    if cursor.sampler_spec_sha256 != spec.sha256():
        raise ValueError("cursor binds a different sampler specification")
    idents = (
        list(shard_idents)
        if shard_idents is not None
        else [(shard.shard_id, shard.sha256()) for shard in shards]
    )
    if len(idents) != len(shards):
        raise ValueError("manifest identity count disagrees with pack shards")
    order = sampler_order(
        [digest for _, digest in idents], run_seed=spec.run_seed, epoch=cursor.epoch
    )
    total_sequences = _flat_sequence_count(order, shards)
    if total_sequences == 0:
        raise ValueError("pack holds no addressable sequences")
    flat: list[tuple[int, int]] = [
        (order_position, sequence_index)
        for order_position in order
        for sequence_index in range(len(shards[order_position].sequences))
    ]
    epoch = cursor.epoch
    position = cursor.sampler_position
    offset = cursor.token_offset
    if position > len(flat) or (position == len(flat) and offset != 0):
        raise ValueError("cursor position is past the end of the pack epoch")
    if position == len(flat):
        epoch += 1
        position = 0
        offset = 0
        order = sampler_order(
            [digest for _, digest in idents], run_seed=spec.run_seed, epoch=epoch
        )
        flat = [
            (order_position, sequence_index)
            for order_position in order
            for sequence_index in range(len(shards[order_position].sequences))
        ]
    takes: list[WindowTake] = []
    remaining = spec.tokens_per_update
    touched: set[int] = set()
    iterations = 0
    while remaining > 0:
        iterations += 1
        if iterations > 2 * (len(flat) + 1):
            raise ValueError("pack holds no consumable real tokens")
        if position >= len(flat):
            epoch += 1
            position = 0
            offset = 0
            order = sampler_order(
                [digest for _, digest in idents], run_seed=spec.run_seed, epoch=epoch
            )
            flat = [
                (order_position, sequence_index)
                for order_position in order
                for sequence_index in range(len(shards[order_position].sequences))
            ]
        order_position, sequence_index = flat[position]
        shard = shards[order_position]
        sequence = shard.sequences[sequence_index]
        actual = sum(1 for token in sequence.tokens if token != PAD_ID)
        if actual != _sequence_real_tokens(sequence):
            raise ValueError("touched sequence receipt disagrees with its bytes")
        real = actual
        if offset >= real:
            if real == 0:
                position += 1
                offset = 0
                continue
            raise ValueError("cursor token offset is past the sequence end")
        take = min(remaining, real - offset)
        takes.append(
            WindowTake(
                epoch=epoch,
                order_position=order_position,
                shard_index=order_position,
                sequence_index=sequence_index,
                start_real_token=offset,
                length_real_tokens=take,
            )
        )
        touched.add(order_position)
        remaining -= take
        offset += take
        if offset >= real:
            position += 1
            offset = 0
    for shard_index in sorted(touched):
        shard = shards[shard_index]
        manifest_digest = dict(idents)[shard.shard_id]
        if shard.sha256() != manifest_digest:
            raise ValueError(f"touched shard failed manifest verification: {shard.shard_id}")
    next_cursor = StreamCursor(
        schema=CURSOR_SCHEMA_V1,
        pack_manifest_sha256=cursor.pack_manifest_sha256,
        sampler_spec_sha256=cursor.sampler_spec_sha256,
        epoch=epoch,
        sampler_position=position,
        token_offset=offset,
        cumulative_real_tokens=cursor.cumulative_real_tokens + spec.tokens_per_update,
    )
    next_cursor.assert_valid()
    return takes, next_cursor


def assert_cursor_advances(
    before: StreamCursor, after: StreamCursor, *, tokens_per_update: int
) -> None:
    """Reject rollback, reuse, jumps, epoch/spec/pack drift, and ledger mismatch."""

    before.assert_valid()
    after.assert_valid()
    if tokens_per_update <= 0:
        raise ValueError("tokens per update must be positive")
    if (
        before.pack_manifest_sha256 != after.pack_manifest_sha256
        or before.sampler_spec_sha256 != after.sampler_spec_sha256
    ):
        raise ValueError("cursor advance changed pack or sampler identity")
    before_key = (before.epoch, before.sampler_position, before.token_offset)
    after_key = (after.epoch, after.sampler_position, after.token_offset)
    if after_key <= before_key:
        raise ValueError("cursor did not advance: rollback or reuse refused")
    if after.cumulative_real_tokens - before.cumulative_real_tokens != tokens_per_update:
        raise ValueError("cursor ledger must advance by exactly one update budget")


def take_slices(
    sequence: PackedSequence, start_real_token: int, length_real_tokens: int
) -> tuple[list[int], list[int]]:
    """Resolve a real-token take to array slices, skipping padding.

    Real-token offsets count non-pad tokens only; the returned token and
    segment-ID slices address the underlying arrays so masks rebuild exactly.
    """

    if start_real_token < 0 or length_real_tokens <= 0:
        raise ValueError("take bounds must address a positive real-token run")
    indices = [index for index, token in enumerate(sequence.tokens) if token != PAD_ID]
    if start_real_token + length_real_tokens > len(indices):
        raise ValueError("take runs past the sequence real tokens")
    chosen = indices[start_real_token:start_real_token + length_real_tokens]
    return (
        [sequence.tokens[index] for index in chosen],
        [sequence.segment_ids[index] for index in chosen],
    )


def audit_epoch(
    shards: list[MultiPackedShard],
    spec: SamplerSpec,
    *,
    pack_manifest_sha256: str,
    shard_idents: list[tuple[str, str]] | None = None,
    epoch: int = 0,
    seeds: tuple[int, ...] = (0, 1),
) -> dict[str, object]:
    """Prove full single coverage per epoch and deterministic seed variants."""

    spec.assert_valid()
    if epoch < 0:
        raise ValueError("epoch cannot be negative")
    idents = (
        list(shard_idents)
        if shard_idents is not None
        else [(shard.shard_id, shard.sha256()) for shard in shards]
    )
    order = sampler_order(
        [digest for _, digest in idents], run_seed=spec.run_seed, epoch=epoch
    )
    expected = _flat_sequence_count(order, shards)
    cursor = StreamCursor(
        schema=CURSOR_SCHEMA_V1,
        pack_manifest_sha256=pack_manifest_sha256,
        sampler_spec_sha256=spec.sha256(),
        epoch=epoch,
        sampler_position=0,
        token_offset=0,
        cumulative_real_tokens=0,
    )
    seen: dict[tuple[int, int], int] = {}
    windows = 0
    consumed_this_epoch = 0
    while cursor.epoch == epoch:
        if windows > 0 and consumed_this_epoch == 0:
            raise ValueError("audit made no progress; pack holds no real tokens")
        takes, cursor = next_window(
            shards, spec, cursor,
            pack_manifest_sha256=pack_manifest_sha256, shard_idents=idents,
        )
        windows += 1
        for take in takes:
            if take.epoch != epoch:
                continue
            consumed_this_epoch += take.length_real_tokens
            key = (take.order_position, take.sequence_index)
            seen[key] = seen.get(key, 0) + take.length_real_tokens
        if windows > expected + 2 * len(order) + 4:
            raise ValueError("audit failed to terminate within one epoch")
    coverage_ok = len(seen) == expected and all(value > 0 for value in seen.values())
    variants = []
    base = [digest for _, digest in idents]
    for seed in seeds:
        probe = sampler_order(base, run_seed=seed, epoch=epoch)
        variants.append(
            {
                "seed": seed,
                "differs_from_spec_order": probe != order,
                "valid_permutation": sorted(probe) == sorted(order),
            }
        )
    receipt: dict[str, object] = {
        "schema": AUDIT_SCHEMA,
        "pack_manifest_sha256": pack_manifest_sha256,
        "sampler_spec_sha256": spec.sha256(),
        "epoch": epoch,
        "expected_sequences": expected,
        "covered_sequences": len(seen),
        "windows": windows,
        "single_coverage": coverage_ok,
        "seed_variants_differ": variants,
        "status": "PASS" if coverage_ok else "FAIL",
    }
    return receipt
