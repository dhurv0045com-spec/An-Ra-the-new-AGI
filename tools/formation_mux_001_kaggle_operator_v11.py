"""Canonical FORMATION-MUX operator v11.

Engineering hardening over v10: the S5 development-only X-factor and S5 sealed
finalizer explicitly rebind the shared audited training module back to Science
S5 before loading any S5 checkpoint. This prevents coordinator-local protocol
state left by TIE-ROLE diagnostics/finalization from crossing experiment
boundaries. Scientific protocols and exposures are unchanged.
"""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import formation_mux_001_kaggle_operator_v10 as v10
from tools import formation_mux_001_kaggle_operator_v9 as v9

OPERATOR_NAME = "tools/formation_mux_001_kaggle_operator_v11.py"
_ORIGINAL_RUN_XFACTOR = v9.run_xfactor
_ORIGINAL_S5_FINALIZER = v9._ORIGINAL_FINALIZE_SEALED


def _rebind_s5_training() -> None:
    from anra_v5 import formation_mux_train_v5 as train

    train._base.fxm = train.fxm
    train._base.proto = train.proto


def run_xfactor_s5_bound(*args, **kwargs):
    _rebind_s5_training()
    return _ORIGINAL_RUN_XFACTOR(*args, **kwargs)


def finalize_s5_bound(*args, **kwargs):
    _rebind_s5_training()
    return _ORIGINAL_S5_FINALIZER(*args, **kwargs)


def _bind() -> None:
    v10.OPERATOR_NAME = OPERATOR_NAME
    v9.run_xfactor = run_xfactor_s5_bound
    v9._ORIGINAL_FINALIZE_SEALED = finalize_s5_bound


def main(argv=None) -> int:
    _bind()
    return v10.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
