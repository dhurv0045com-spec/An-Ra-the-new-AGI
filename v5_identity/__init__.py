"""HORM-001 V5 sibling: hormonal state plus bounded attention-scale overlay."""

from __future__ import annotations

from .appraisal import appraise_committed
from .hormonal_config import V5A_250M_HORMONAL_V1, HormonalOverlay
from .hormonal_projection import SCALE_BOUNDS, HormonalProjection
from .hormonal_state import (
    BASELINES,
    BOUNDS,
    DECAY_RATES,
    HORMONES,
    HormonalState,
)

__all__ = [
    "appraise_committed",
    "BASELINES",
    "BOUNDS",
    "DECAY_RATES",
    "HORMONES",
    "HormonalOverlay",
    "HormonalProjection",
    "HormonalState",
    "SCALE_BOUNDS",
    "V5A_250M_HORMONAL_V1",
]
