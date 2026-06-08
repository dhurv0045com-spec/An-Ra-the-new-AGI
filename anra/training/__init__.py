"""Training subsystem: RLVR, STaR curriculum, dynamic regret, mixed precision."""

from __future__ import annotations

from training.rlvr import RLVRTrainer
from training.star import STaRLoop

__all__ = ["RLVRTrainer", "STaRLoop"]
