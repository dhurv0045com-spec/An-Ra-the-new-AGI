"""Bucket-pure data lanes and checkpointable bucket cursor.

A production microstep for bucket B consumes real tokens from bucket-B
sequences ONLY, so every row sent to the model has native width <= B and the
assembled tensor has exactly width B with the frozen per-replica sequence
count. Lanes are deterministic functions of (pack, run_seed, pattern, epoch,
cell map): resume rebuilds them identically (bounded by pack size, never by
campaign length) and only per-cell positions plus mixture counters are
checkpointed. A lane running dry never substitutes another bucket: the take
raises LaneExhausted and the campaign either performs an explicitly permitted
epoch replay or fails closed with DATA_NOT_READY.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass


BUCKET_CURSOR_SCHEMA = "anra-v5-bucket-cursor/v1"
LANES_SCHEMA = "anra-v5-bucket-lanes/v1"
CELL_SEP = "|"


def cell_key(bucket: int, family: str, sub: str) -> str:
    return f"{int(bucket)}{CELL_SEP}{family}{CELL_SEP}{sub}"


def parse_cell_key(key: str) -> tuple[int, str, str]:
    parts = key.split(CELL_SEP)
    if len(parts) != 3:
        raise ValueError(f"malformed lane cell key: {key!r}")
    return int(parts[0]), parts[1], parts[2]


class LaneExhausted(ValueError):
    """A bucket/family lane cannot supply the requested real tokens."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def _assert_sha256(name: str, value: str) -> None:
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} must be lowercase SHA-256")


@dataclass(frozen=True, slots=True)
class BucketCursorState:
    """Checkpointed position of the bucket-lane walk plus schedule counters."""

    schema: str
    pack_manifest_sha256: str
    lanes_sha256: str
    positions: dict
    mixture_consumed: dict
    sub_consumed: dict
    epoch: int
    replay_count: int

    def assert_valid(self) -> None:
        if self.schema != BUCKET_CURSOR_SCHEMA:
            raise ValueError("unsupported bucket-cursor schema")
        _assert_sha256("cursor pack manifest", self.pack_manifest_sha256)
        _assert_sha256("cursor lanes", self.lanes_sha256)
        for key, value in self.positions.items():
            parse_cell_key(key)
            if (not isinstance(value, (list, tuple)) or len(value) != 2
                    or min(int(value[0]), int(value[1])) < 0):
                raise ValueError(f"lane position for {key!r} must be [index, offset]")
        for name, consumed in list(self.mixture_consumed.items()) + list(self.sub_consumed.items()):
            if not isinstance(name, str) or int(consumed) < 0:
                raise ValueError("mixture counters require string names and nonnegative counts")
        if min(self.epoch, self.replay_count) < 0:
            raise ValueError("epoch and replay count cannot be negative")

    def canonical(self) -> dict[str, object]:
        self.assert_valid()
        return {
            "schema": self.schema,
            "pack_manifest_sha256": self.pack_manifest_sha256,
            "lanes_sha256": self.lanes_sha256,
            "positions": {key: (int(value[0]), int(value[1]))
                          for key, value in sorted(self.positions.items())},
            "mixture_consumed": dict(sorted(self.mixture_consumed.items())),
            "sub_consumed": dict(sorted(self.sub_consumed.items())),
            "epoch": self.epoch,
            "replay_count": self.replay_count,
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "BucketCursorState":
        expected = {"schema", "pack_manifest_sha256", "lanes_sha256",
                    "positions", "mixture_consumed", "sub_consumed",
                    "epoch", "replay_count"}
        if set(value) != expected:
            raise ValueError("bucket-cursor fields do not match schema")
        state = cls(
            schema=str(value["schema"]),
            pack_manifest_sha256=str(value["pack_manifest_sha256"]),
            lanes_sha256=str(value["lanes_sha256"]),
            positions={str(key): (int(item[0]), int(item[1]))
                       for key, item in dict(value["positions"]).items()},
            mixture_consumed={str(key): int(item)
                              for key, item in dict(value["mixture_consumed"]).items()},
            sub_consumed={str(key): int(item)
                          for key, item in dict(value["sub_consumed"]).items()},
            epoch=int(value["epoch"]),
            replay_count=int(value["replay_count"]),
        )
        state.assert_valid()
        return state


@dataclass(frozen=True, slots=True)
class LaneWindow:
    """One exact real-token window from a single bucket/family lane."""

    tokens: tuple[tuple[int, ...], ...]
    segment_ids: tuple[tuple[int, ...], ...]
    eligible: tuple[tuple[bool, ...], ...]
    tokens_by_source: dict[str, int]
    tokens_by_family: dict[str, int]
    real_tokens: int
    row_widths: tuple[int, ...]
    end_lane_index: int
    end_token_offset: int


def build_bucket_lanes(packed, *, run_seed: int, pattern: list[int],
                       epoch: int = 0,
                       cell_of_source=None,
                       required_buckets: set[int] | None = None) -> tuple[dict, dict[str, object]]:
    """Group packed sequences into deterministic per-(bucket, family, sub) lanes.

    Within a lane, shards follow the epoch sampler permutation and sequences
    keep shard order. Every REQUIRED bucket must have supply or the build
    fails closed with the shortfall (default: every distinct pattern
    bucket). ``cell_of_source`` maps a segment source id to (family,
    subfamily); without it every sequence pools into ("", ""). A sequence
    mixing cells fails closed (pack with segregation for the mixture path).
    """

    from .pack import sampler_order

    if not packed:
        raise ValueError("bucket lanes need a nonempty pack")
    if run_seed < 0 or epoch < 0:
        raise ValueError("run seed and epoch cannot be negative")
    if not pattern or any(bucket <= 0 for bucket in pattern):
        raise ValueError("supercycle pattern must hold positive buckets")
    shard_hashes = [shard.sha256() for shard in packed]
    if len(set(shard_hashes)) != len(shard_hashes):
        raise ValueError("pack shard hashes must be distinct")
    order = sampler_order(shard_hashes, run_seed=run_seed, epoch=epoch)
    lanes: dict[str, list[tuple[int, int]]] = {}
    supply: dict[str, dict[str, object]] = {}
    for shard_index in order:
        shard = packed[shard_index]
        for sequence_index, sequence in enumerate(shard.sequences):
            if cell_of_source is None:
                family, sub = "", ""
            else:
                cells = {tuple(cell_of_source[source]) for source in sequence.sources}
                if len(cells) != 1:
                    raise ValueError(
                        "mixed-cell packed sequence: pack with family segregation "
                        "for the mixture path")
                family, sub = cells.pop()
            key = cell_key(shard.bucket, family, sub)
            lanes.setdefault(key, []).append((shard_index, sequence_index))
            cell_supply = supply.setdefault(
                key, {"bucket": shard.bucket, "family": family, "sub": sub,
                      "sequences": 0, "real_tokens": 0, "shards": set()})
            cell_supply["sequences"] = int(cell_supply["sequences"]) + 1
            cell_supply["real_tokens"] = (
                int(cell_supply["real_tokens"]) + sequence.real_tokens)
            cell_supply["shards"].add(shard_index)
    buckets_with_supply = {parse_cell_key(key)[0] for key in lanes}
    required = set(pattern) if required_buckets is None else set(required_buckets)
    missing = sorted(required - buckets_with_supply)
    if missing:
        raise ValueError(
            f"bucket lane supply missing for required buckets {missing}; "
            "no silent substitution across buckets")
    for cell_supply in supply.values():
        cell_supply["shards"] = sorted(cell_supply["shards"])
    lanes_digest = hashlib.sha256(_canonical_json(
        {"pattern": list(pattern), "run_seed": run_seed, "epoch": epoch,
         "lanes": {key: [list(item) for item in lane]
                   for key, lane in sorted(lanes.items())}})).hexdigest()
    receipt: dict[str, object] = {
        "schema": LANES_SCHEMA,
        "run_seed": run_seed,
        "epoch": epoch,
        "pattern": list(pattern),
        "cells": {key: dict(supply[key]) for key in sorted(supply)},
        "lanes_sha256": lanes_digest,
    }
    return lanes, receipt


def lane_remainder(packed, lane, lane_index: int, token_offset: int,
                   *, pad: int) -> int:
    """Exact real tokens available in a lane from a position to its end."""

    if lane_index < 0 or token_offset < 0:
        raise ValueError("lane coordinates cannot be negative")
    total = 0
    for cursor in range(lane_index, len(lane)):
        shard_index, sequence_index = lane[cursor]
        sequence = packed[shard_index].sequences[sequence_index]
        real = [index for index, token in enumerate(sequence.tokens) if token != pad]
        if cursor == lane_index:
            if token_offset > len(real):
                raise ValueError("lane token offset is past the sequence real tokens")
            total += len(real) - token_offset
        else:
            total += len(real)
    return total


def take_cell_window(packed, lane, lane_index: int, token_offset: int, *,
                     real_tokens: int, pad: int, bucket: int,
                     cell_of_source=None) -> LaneWindow:
    """Take exactly ``real_tokens`` real tokens from one lane position.

    Whole sequences transfer while they fit; the final sequence is cut at
    the exact boundary with an eligibility mask over kept real positions.
    Every row must fit the lane bucket or the pack is rejected. Exhaustion
    raises LaneExhausted (replay or DATA_NOT_READY is the caller's choice).
    """

    if real_tokens <= 0:
        raise ValueError("lane window needs a positive real-token budget")
    if lane_index < 0 or token_offset < 0:
        raise ValueError("lane coordinates cannot be negative")

    def family_of(source: str) -> str:
        if cell_of_source is None:
            return ""
        return str(cell_of_source[source][0])

    rows_tokens: list[list[int]] = []
    rows_segments: list[list[int]] = []
    rows_eligible: list[list[bool]] = []
    rows_widths: list[int] = []
    by_source: dict[str, int] = {}
    by_family: dict[str, int] = {}
    consumed = 0
    cursor = lane_index
    pending_offset = token_offset
    end_index, end_sequence, end_offset = lane_index, 0, token_offset
    while consumed < real_tokens:
        if cursor >= len(lane):
            raise LaneExhausted(
                f"lane exhausted after {consumed} of {real_tokens} real tokens")
        shard_index, sequence_index = lane[cursor]
        sequence = packed[shard_index].sequences[sequence_index]
        if len(sequence.tokens) > bucket:
            raise ValueError("packed row exceeds its lane bucket")
        rows_widths.append(len(sequence.tokens))
        real_positions = [
            index for index, token in enumerate(sequence.tokens) if token != pad
        ]
        if pending_offset > len(real_positions):
            raise ValueError("lane token offset is past the sequence real tokens")
        need = real_tokens - consumed
        take = real_positions[pending_offset:pending_offset + need]
        kept = set(take)
        rows_tokens.append(list(sequence.tokens))
        rows_segments.append(list(sequence.segment_ids))
        rows_eligible.append([index in kept for index in range(len(sequence.tokens))])
        for index in take:
            segment = sequence.segment_ids[index]
            source = (sequence.sources[segment]
                      if 0 <= segment < len(sequence.sources) else None)
            if source is None:
                raise ValueError("eligible position lacks source attribution")
            by_source[source] = by_source.get(source, 0) + 1
            family = family_of(source)
            by_family[family] = by_family.get(family, 0) + 1
        consumed += len(take)
        if pending_offset + len(take) < len(real_positions):
            end_index, end_offset = cursor, pending_offset + len(take)
        else:
            end_index, end_offset = cursor + 1, 0
        pending_offset = 0
        cursor += 1
    if consumed != real_tokens or sum(by_source.values()) != real_tokens:
        raise ValueError("lane window ledger disagrees with the exact budget")
    return LaneWindow(
        tokens=tuple(tuple(row) for row in rows_tokens),
        segment_ids=tuple(tuple(row) for row in rows_segments),
        eligible=tuple(tuple(row) for row in rows_eligible),
        tokens_by_source=dict(sorted(by_source.items())),
        tokens_by_family=dict(sorted(by_family.items())),
        real_tokens=consumed,
        row_widths=tuple(rows_widths),
        end_lane_index=end_index,
        end_token_offset=end_offset,
    )


__all__ = ["BUCKET_CURSOR_SCHEMA", "BucketCursorState", "LANES_SCHEMA",
           "LaneExhausted", "LaneWindow", "build_bucket_lanes", "cell_key",
           "lane_remainder", "parse_cell_key", "take_cell_window"]
