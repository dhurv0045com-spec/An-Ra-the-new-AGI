"""Scoped optimizer-API compatibility for CYR-GPU-011 research only.

Cymek's canonical optimizer constructor deliberately freezes AdamW betas,
epsilon and weight decay and therefore only exposes ``lr`` (plus the injectable
``torch_module``). Early CYR-GPU-011 code redundantly passes the frozen values
explicitly. This module accepts those keywords only while a scoped context is
active, verifies that they equal the canonical V5 constants, delegates to the
canonical constructor, then restores the original function immediately.

Nothing here changes production optimizer semantics or permits a non-canonical
optimizer configuration.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator


def _compatible_wrapper(module: Any, original: Any):
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

    return compatible_build_adamw_optimizer


@contextmanager
def canonical_optimizer_compat() -> Iterator[None]:
    """Temporarily accept redundant canonical AdamW keywords, then restore.

    The guard is intentionally process-local and scoped. Nested use is safe:
    only the outermost caller replaces/restores the function it observed.
    """
    import v5_training.optimizer as module

    original = module.build_adamw_optimizer
    wrapper = _compatible_wrapper(module, original)
    module.build_adamw_optimizer = wrapper
    try:
        yield
    finally:
        if module.build_adamw_optimizer is wrapper:
            module.build_adamw_optimizer = original


__all__ = ["canonical_optimizer_compat"]
