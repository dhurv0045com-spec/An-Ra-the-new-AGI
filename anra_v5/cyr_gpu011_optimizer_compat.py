"""Strict compatibility shim for the CYR-GPU-011 research runner.

The V5 optimizer constructor deliberately freezes AdamW betas/epsilon/weight
decay and exposes only ``lr``. Early CYR-GPU-011 code passed those frozen
values explicitly. Rather than changing canonical production optimizer code,
this research-only shim accepts the redundant keywords, verifies that they are
exactly the V5 constants, and delegates to the canonical constructor.

The shim changes no optimizer semantics and refuses any non-canonical value.
"""
from __future__ import annotations

from typing import Any

_INSTALLED = False
_ORIGINAL = None


def install() -> None:
    global _INSTALLED, _ORIGINAL
    if _INSTALLED:
        return
    import v5_training.optimizer as module

    original = module.build_adamw_optimizer
    _ORIGINAL = original

    def compatible_build_adamw_optimizer(
        model: Any,
        *,
        lr: float = module.PEAK_LEARNING_RATE,
        torch_module: Any | None = None,
        betas: tuple[float, float] | None = None,
        eps: float | None = None,
        weight_decay: float | None = None,
    ) -> Any:
        if betas is not None and tuple(float(x) for x in betas) != (module.BETA1, module.BETA2):
            raise ValueError("CYR-GPU-011 refuses non-canonical AdamW betas")
        if eps is not None and float(eps) != module.EPSILON:
            raise ValueError("CYR-GPU-011 refuses non-canonical AdamW epsilon")
        if weight_decay is not None and float(weight_decay) != module.WEIGHT_DECAY:
            raise ValueError("CYR-GPU-011 refuses non-canonical AdamW weight decay")
        return original(model, lr=float(lr), torch_module=torch_module)

    module.build_adamw_optimizer = compatible_build_adamw_optimizer
    _INSTALLED = True


def installed() -> bool:
    return _INSTALLED


__all__ = ["install", "installed"]
