"""Canonical operator entry points for CYR-GPU-011.

Two narrow research-only compatibility layers are scoped to CYR-GPU-011:
1. Early runner call-sites redundantly supplied AdamW constants already frozen
   by Cymek. The optimizer shim accepts only the exact canonical constants and
   restores the production API immediately afterwards.
2. The final scientific verdict is exposure-aware. A production-tokenizer null
   cannot be called a representation divergence unless it received at least the
   semantic exposure at which the compact bridge qualified. Controller G90 must
   also agree with the larger DEV_MEASUREMENT STANDARD score.

Neither layer changes Cymek production training semantics.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, Mapping

from anra_v5 import cyr_gpu011_run as _runner
from anra_v5.cyr_gpu011_optimizer_compat import canonical_optimizer_compat
from v5_experiments import cyr_gpu011 as _core

BUNDLE_NAME = _runner.BUNDLE_NAME
production_tokenizer = _runner.production_tokenizer
write_json = _runner.write_json
read_json = _runner.read_json


def _measurement_standard(receipt: Mapping[str, Any] | None) -> float:
    if not receipt:
        return 0.0
    return float(
        receipt.get("reasoning_battery_final", {})
        .get("STANDARD", {})
        .get("complete_exact_with_valid_stop", 0.0)
    )


def _qualified_g90(receipt: Mapping[str, Any] | None) -> bool:
    return bool(
        receipt
        and receipt.get("g90_confirm_update") is not None
        and _measurement_standard(receipt) >= _core.CYR11_G90
    )


def _g90_exposure_fraction(receipt: Mapping[str, Any] | None) -> float | None:
    if not receipt or receipt.get("g90_confirm_update") is None:
        return None
    updates = int(receipt["g90_confirm_update"])
    batch = int(receipt.get("batch_rows", 0))
    if batch <= 0:
        return None
    return updates * batch / _core.CYR11_ARK_MAX_ROW_PRESENTATIONS


def exposure_aware_final_decision(
    *,
    compact: Mapping[str, Any] | None,
    production_primary: Mapping[str, Any] | None,
    production_replication: Mapping[str, Any] | None,
) -> dict[str, Any]:
    c = _qualified_g90(compact)
    p1 = _qualified_g90(production_primary)
    p2 = _qualified_g90(production_replication)
    c_g90_fraction = _g90_exposure_fraction(compact)
    p1_fraction = float((production_primary or {}).get("ark_exposure_fraction", 0.0))
    c_fraction = float((compact or {}).get("ark_exposure_fraction", 0.0))

    if p1 and p2:
        verdict = "PRODUCTION_REPRESENTATION_G90_REPLICATED_DEVELOPMENT"
    elif p1:
        verdict = "PRODUCTION_REPRESENTATION_G90_SINGLE_SEED_DEVELOPMENT"
    elif c and c_g90_fraction is not None and p1_fraction + 1e-12 < c_g90_fraction:
        verdict = "PRODUCTION_UNDEREXPOSED_RELATIVE_TO_COMPACT_G90"
    elif c:
        verdict = "EXPOSURE_MATCHED_BRIDGE_DIVERGENCE_COMPACT_G90_PRODUCTION_NO_G90"
    elif not c and not p1 and min(c_fraction, p1_fraction) >= 0.95:
        verdict = "NO_G90_AT_NEAR_ARK_REFERENCE_EXPOSURE"
    elif not c and not p1:
        verdict = "NO_G90_WITH_INCOMPLETE_EXPOSURE"
    else:
        verdict = "PRODUCTION_G90_WITHOUT_COMPACT_G90"

    return {
        "schema": "anra-cyr-gpu011-decision/v2",
        "verdict": verdict,
        "compact_controller_and_measurement_g90": c,
        "production_primary_controller_and_measurement_g90": p1,
        "production_replication_controller_and_measurement_g90": p2,
        "compact_measurement_standard_final": _measurement_standard(compact),
        "production_primary_measurement_standard_final": _measurement_standard(production_primary),
        "production_replication_measurement_standard_final": _measurement_standard(production_replication),
        "compact_g90_ark_exposure_fraction": c_g90_fraction,
        "compact_final_ark_exposure_fraction": c_fraction,
        "production_primary_final_ark_exposure_fraction": p1_fraction,
        "exposure_matched_for_divergence": bool(
            c and c_g90_fraction is not None and p1_fraction + 1e-12 >= c_g90_fraction
        ),
        "interpretation": (
            "A representation-divergence label is allowed only when the production bridge received at least "
            "the semantic exposure at which the compact bridge qualified. G90 claims require sustained "
            "DEV_CONTROLLER performance and >=0.90 final DEV_MEASUREMENT STANDARD exact-with-EOS."
        ),
        "broad_reasoning_claim_authorized": False,
        "production_promotion_authorized": False,
        "pre500m_authorized": False,
        "training_500m_authorized": False,
    }


@contextmanager
def _decision_scope() -> Iterator[None]:
    original = _core.final_decision
    _core.final_decision = exposure_aware_final_decision
    try:
        yield
    finally:
        if _core.final_decision is exposure_aware_final_decision:
            _core.final_decision = original


def calibrate_all(*args: Any, **kwargs: Any):
    with canonical_optimizer_compat():
        return _runner.calibrate_all(*args, **kwargs)


def run_campaign(*args: Any, **kwargs: Any):
    with canonical_optimizer_compat(), _decision_scope():
        return _runner.run_campaign(*args, **kwargs)


__all__ = [
    "BUNDLE_NAME",
    "calibrate_all",
    "exposure_aware_final_decision",
    "production_tokenizer",
    "read_json",
    "run_campaign",
    "write_json",
]
