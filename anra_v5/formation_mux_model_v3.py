"""FORMATION-MUX-001 Amendment-1B model binding.

The S2 row-aware optimizer is retained. The only scientific repair here is
that M2/M3 extra-row gradients are NOT zeroed before the global clip. Frozen
rows remain immutable because they are absent from the row optimizer's
trainable-row index, not because their gradients are removed pre-clip.
"""

from anra_v5.formation_mux_model_v2 import *  # noqa: F401,F403


def mask_frozen_gradients_before_clip(model, arm):
    """Intentional no-op under Amendment-1B.

    Keeping the gradient through clipping holds whole-model global-clip
    semantics common for M1 vs M2. The row optimizer still excludes extra
    rows in M2/M3, so those rows receive no update/state/decay.
    """

    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm}")
    grad = model.embedding.weight.grad
    if grad is None:
        raise RuntimeError("embedding gradient missing before global clip")
    return None
