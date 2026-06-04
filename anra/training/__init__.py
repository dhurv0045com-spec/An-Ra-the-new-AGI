"""Training subsystem: RLVR, STaR, curriculum, dynamic regret."""

from training.rlvr import RLVRTrainer
from training.star import STaRTrainer

__all__ = ["RLVRTrainer", "STaRTrainer"]
