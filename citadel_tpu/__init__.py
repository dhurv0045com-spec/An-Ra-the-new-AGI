"""Citadel Kaggle-TPU bootstrap path (source of truth; notebook is a thin launcher).

Target: Kaggle free TPU (expected v5e-8, never hard-coded — see environment.py).
Stack: PyTorch/XLA first. No JAX/TF rewrite. Cymek model/objective/checkpoint
logic is reused, never copied invisibly: Cymek SHA is recorded in every receipt.

Execution rule: if no Kaggle TPU is present, fail closed. A CPU run or a CUDA
run is NEVER recorded as a TPU receipt.
"""

__all__ = []
