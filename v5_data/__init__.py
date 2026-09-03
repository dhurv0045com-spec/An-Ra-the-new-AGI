"""Deterministic V5 data pipeline: mixture, split, pack, and cursor."""

from .cursor import advance, sequence_count
from .mixture import (
    allocate,
    bucket_plan,
    cognition_allocation,
    slice_allocation,
)
from .pack import (
    PackedShard,
    build_shards,
    pack_ledger,
    sampler_order,
)
from .split import (
    assign_split,
    exact_clusters,
    normalize_text,
    scan_contamination,
)

__all__ = [
    "PackedShard",
    "advance",
    "allocate",
    "assign_split",
    "bucket_plan",
    "build_shards",
    "cognition_allocation",
    "exact_clusters",
    "normalize_text",
    "pack_ledger",
    "sampler_order",
    "scan_contamination",
    "sequence_count",
    "slice_allocation",
]
