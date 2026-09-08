"""Exact, bounded teaching controls for delayed-information inquiry."""

from .teaching import (
    PairedTeachingData,
    build_paired_teaching_data,
    depth_two_scores,
    stratified_semantic_split,
)

__all__ = [
    "PairedTeachingData",
    "build_paired_teaching_data",
    "depth_two_scores",
    "stratified_semantic_split",
]

