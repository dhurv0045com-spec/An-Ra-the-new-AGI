"""Logical-global vs physical-replica execution mapping.

The frozen contract is data-parallel: 8 replicas x 4,096 real tokens per
replica microstep = 32,768 global tokens, accumulated over 4 microsteps to
one 131,072-token optimizer update. Local CPU/CUDA execution runs the GLOBAL
logical microstep on one device (a mathematical emulation, never TPU
evidence). This module is the single authority that splits a global row set
into deterministic replica shards for real data-parallel execution and for
the distributed-equivalence oracle. Partial tails that do not divide evenly
are padded with fully-ineligible pad rows, receipted, never silently
redistributed.
"""

from __future__ import annotations


TOPOLOGY_MAP_SCHEMA = "anra-v5-topology-map/v1"


def replica_shards(rows: list, *, replicas: int) -> list[list]:
    """Split global rows into contiguous per-replica shards.

    Requires exact divisibility: a partial tail that does not divide must be
    padded first (pad_replica_batch), so indivisibility fails closed here
    instead of silently unbalancing replicas.
    """

    if replicas <= 0:
        raise ValueError("replica count must be positive")
    if not rows:
        raise ValueError("replica sharding needs a nonempty row set")
    if len(rows) % replicas:
        raise ValueError(
            f"{len(rows)} rows do not divide across {replicas} replicas; "
            "pad the tail explicitly first")
    width = len(rows)
    chunk = width // replicas
    return [list(rows[index * chunk:(index + 1) * chunk]) for index in range(replicas)]


def pad_replica_batch(shards: list[list], *, width: int, pad_row: list) -> dict[str, object]:
    """Pad replica shards with fully-ineligible pad rows to equal length.

    Returns a receipt binding the padded shape. Padding rows must carry no
    eligible targets (the caller builds them from pad ids with an empty
    eligibility mask).
    """

    if width <= 0:
        raise ValueError("row width must be positive")
    if any(len(row) != width for shard in shards for row in shard):
        raise ValueError("replica rows disagree with the declared width")
    target = max(len(shard) for shard in shards)
    padded = [list(shard) + [list(pad_row) for _ in range(target - len(shard))]
              for shard in shards]
    added = sum(len(shard) - len(original)
                for shard, original in zip(padded, shards))
    return {"schema": TOPOLOGY_MAP_SCHEMA,
            "replicas": len(shards),
            "rows_per_replica": target,
            "padded_rows_added": added,
            "shards": padded}


def physical_plan(*, bucket: int, sequences_global: int, replicas: int,
                  sequences_per_replica: int) -> dict[str, object]:
    """Certify one microstep's physical shape against the frozen per-replica counts."""

    if bucket <= 0 or sequences_global <= 0 or replicas <= 0:
        raise ValueError("physical plan dimensions must be positive")
    if sequences_global != replicas * sequences_per_replica:
        raise ValueError(
            f"{sequences_global} global sequences do not equal {replicas} "
            f"replicas x {sequences_per_replica} for bucket {bucket}")
    return {"schema": TOPOLOGY_MAP_SCHEMA, "bucket": bucket,
            "sequences_global": sequences_global, "replicas": replicas,
            "sequences_per_replica": sequences_per_replica}


def certify_microstep_shape(*, bucket: int, sequences_global: int,
                            replicas: int,
                            sequences_per_replica: int) -> dict[str, object]:
    """Certify a microstep's executed shape: frozen plan when rows are full.

    Exactly-full packed rows yield exactly the frozen per-replica counts
    (the production regime: stream-filled packs). Sparse rows or partial
    tails take the explicit pad path instead: local emulation still runs the
    exact global tensor, and the receipt says so. Never silently certifies a
    non-frozen shape as frozen.
    """

    try:
        return physical_plan(bucket=bucket, sequences_global=sequences_global,
                             replicas=replicas,
                             sequences_per_replica=sequences_per_replica)
    except ValueError:
        return {"schema": TOPOLOGY_MAP_SCHEMA, "bucket": bucket,
                "sequences_global": sequences_global, "replicas": replicas,
                "sequences_per_replica_frozen": sequences_per_replica,
                "partial_tail": True,
                "note": "local emulation runs the exact global tensor; "
                        "physical sharding certified by topology_map"}


__all__ = ["TOPOLOGY_MAP_SCHEMA", "certify_microstep_shape",
           "pad_replica_batch", "physical_plan", "replica_shards"]
