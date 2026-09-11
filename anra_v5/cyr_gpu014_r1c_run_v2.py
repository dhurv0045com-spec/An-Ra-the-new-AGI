"""CYR-GPU-014 / R1C pre-execution compatibility wrapper.

This wrapper repairs one implementation-only API mismatch discovered before any
R1C scientific execution: the frozen R1C runner called
``build_adamw_optimizer`` with explicit ``betas``, ``eps`` and
``weight_decay`` keywords, while the canonical V5 constructor freezes those
values internally and accepts only ``model``, ``lr`` and ``torch_module``.

Scientific semantics are unchanged. The canonical constructor's frozen values
are exactly AdamW betas=(0.9, 0.95), eps=1e-8 and weight_decay=0.1, matching
the values requested by the original runner. All R1C data, arms, seeds,
horizons, thresholds, diagnostics and evaluation code remain in the original
frozen runner.
"""
from __future__ import annotations

from anra_v5 import cyr_gpu014_r1c_run as frozen
from v5_training.optimizer import build_adamw_optimizer


def _compatible_build_optimizer(model, *, torch):
    """Use the canonical V5 AdamW API with R1C's unchanged learning rate."""
    return build_adamw_optimizer(
        model,
        lr=frozen.base.CYR11_HIGH_LR,
        torch_module=torch,
    )


# Patch only the incompatible call boundary; all scientific logic stays frozen.
frozen.build_optimizer = _compatible_build_optimizer


def main() -> int:
    return frozen.main()


if __name__ == "__main__":
    raise SystemExit(main())
