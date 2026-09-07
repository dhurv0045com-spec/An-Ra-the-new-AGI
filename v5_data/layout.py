"""Deterministic campaign shard layout with supercycle-weighted lanes.

Exact-token microstep windows can start anywhere, so bucket proportions
cannot be enforced per microstep with a single flat cursor. Instead the
campaign layout arranges shards into lane runs that follow the frozen
bucket supercycle: each 20-slot cycle contributes roughly one microstep
worth of tokens per slot, so the token mix over time tracks the pattern
while every shard appears exactly once. Microstep windows cut exactly
wherever the cursor lands; per-microstep bucket mix is measured and
receipted, never assumed.

Layout is a pure function of (pack, run_seed, epoch, pattern): resume
recomputes it identically with memory bounded by pack size, never by
campaign length. Buckets absent from the pack skip their slots (recorded);
a pack supplying none of the pattern buckets fails closed.
"""

from __future__ import annotations

import hashlib
import json

from .pack import MultiPackedShard


LAYOUT_SCHEMA = "anra-v5-campaign-layout/v1"


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def campaign_layout(
    packed: list[MultiPackedShard],
    *,
    run_seed: int,
    pattern: list[int],
    epoch: int = 0,
    slot_tokens: int = 32768,
    base_order: list[int] | None = None,
) -> tuple[list[int], dict[str, object]]:
    """Arrange shard indices into supercycle-weighted lane runs.

    Returns ``(order, receipt)`` where ``order`` is a permutation of shard
    indices and ``receipt`` binds the pattern, per-slot fills, skips, and
    layout hash for the campaign receipt. ``base_order`` (the epoch sampler
    permutation) orders the per-bucket queues when supplied, so the layout
    consumes the sampler order instead of recomputing its own shuffle; the
    receipt binds its SHA. Without it, queues fall back to the seeded
    content-hash shuffle.
    """

    if not packed:
        raise ValueError("campaign layout needs a nonempty pack")
    if run_seed < 0 or epoch < 0:
        raise ValueError("run seed and epoch cannot be negative")
    if not pattern or any(bucket <= 0 for bucket in pattern):
        raise ValueError("supercycle pattern must hold positive buckets")
    if slot_tokens <= 0:
        raise ValueError("slot token target must be positive")
    sampler_order_sha256: str | None = None
    if base_order is not None:
        if sorted(base_order) != list(range(len(packed))):
            raise ValueError("base order must permute every shard exactly once")
        sampler_order_sha256 = hashlib.sha256(
            _canonical_json(list(base_order))).hexdigest()
        rank = {shard_index: position for position, shard_index in enumerate(base_order)}
    else:
        rank = {
            index: int(hashlib.sha256(
                f"{run_seed}/{epoch}/{packed[index].sha256()}".encode()
            ).hexdigest(), 16)
            for index in range(len(packed))
        }
    by_bucket: dict[int, list[int]] = {}
    for index, shard in enumerate(packed):
        by_bucket.setdefault(shard.bucket, []).append(index)
    for bucket in by_bucket:
        by_bucket[bucket].sort(key=lambda index: rank[index])
    positions: dict[int, int] = {bucket: 0 for bucket in by_bucket}
    order: list[int] = []
    slots: list[dict[str, object]] = []
    skipped: list[int] = []
    slot_number = 0
    remaining = sum(len(indices) for indices in by_bucket.values())
    while remaining > 0:
        progressed = False
        for bucket in pattern:
            queue = by_bucket.get(bucket, [])
            position = positions.get(bucket, 0)
            if position >= len(queue):
                if bucket not in skipped:
                    skipped.append(bucket)
                continue
            gathered = 0
            taken: list[int] = []
            while position < len(queue) and gathered < slot_tokens:
                shard_index = queue[position]
                taken.append(shard_index)
                gathered += packed[shard_index].real_tokens
                position += 1
            positions[bucket] = position
            remaining -= len(taken)
            order.extend(taken)
            slots.append(
                {
                    "slot": slot_number,
                    "bucket": bucket,
                    "shards": taken,
                    "slot_tokens": gathered,
                }
            )
            slot_number += 1
            progressed = True
        if not progressed:
            raise ValueError("campaign layout stalled with unplaced shards")
    if sorted(order) != list(range(len(packed))):
        raise ValueError("campaign layout must place every shard exactly once")
    receipt: dict[str, object] = {
        "schema": LAYOUT_SCHEMA,
        "run_seed": run_seed,
        "epoch": epoch,
        "pattern": list(pattern),
        "slot_tokens": slot_tokens,
        "sampler_order_sha256": sampler_order_sha256,
        "slots": slots,
        "skipped_buckets": sorted(skipped),
        "order": list(order),
    }
    receipt["layout_sha256"] = hashlib.sha256(
        _canonical_json(receipt)
    ).hexdigest()
    return order, receipt


__all__ = ["LAYOUT_SCHEMA", "campaign_layout"]
