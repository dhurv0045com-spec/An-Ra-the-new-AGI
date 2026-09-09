"""Canonical operator entry points for CYR-GPU-011.

CYR-GPU-011 is a research bridge, not a production-training API change. This
module scopes compatibility controls around the V11 runner:

1. Early runner call-sites redundantly supplied AdamW constants already frozen
   by Cymek. The optimizer shim accepts only the canonical constants and is
   restored immediately afterwards.
2. Exposure/decision logic is semantic-row aware. Batch 32/16 may run up to
   36k/72k updates so they can, wall permitting, target the same 1,152,000 row
   presentations as ARK-002B batch64 x 18k. A production null is not called a
   representation divergence unless it reached the compact G90 exposure.
3. The same six-minute finalization allowance used by the calibration resolver
   is enforced before each stage's outer deadline, leaving time for final
   candidate-free batteries/checkpoints while preserving the campaign wall.

The compact bridge uses the exact ARK-002B data and 19-symbol vocabulary but
keeps Cymek's canonical causal objective. Arkenstone additionally supervised a
BOS answer-prefix token; Cymek's objective hard-excludes BOS targets, so V11
records that as an intentional residual difference rather than faking a match.

All patches are process-local, scoped, fail closed, and restored after the
research call. Nothing changes Cymek production optimizer/tokenizer/objective
semantics.
"""
from __future__ import annotations

import math
import time
from contextlib import contextmanager
from typing import Any, Iterator, Mapping

from anra_v5 import cyr_gpu011_run as _runner
from anra_v5.cyr_gpu011_optimizer_compat import canonical_optimizer_compat
from v5_experiments import cyr_gpu011 as _core

BUNDLE_NAME = _runner.BUNDLE_NAME
production_tokenizer = _runner.production_tokenizer
write_json = _runner.write_json
read_json = _runner.read_json
CYR11_ACQUISITION_FINALIZE_RESERVE_SECONDS = 360.0


def _measurement_standard(receipt: Mapping[str, Any] | None) -> float:
    if not receipt:
        return 0.0
    return float(receipt.get("reasoning_battery_final", {}).get("STANDARD", {}).get(
        "complete_exact_with_valid_stop", 0.0
    ))


def _qualified_g90(receipt: Mapping[str, Any] | None) -> bool:
    return bool(receipt and receipt.get("g90_confirm_update") is not None
                and _measurement_standard(receipt) >= _core.CYR11_G90)


def _g90_exposure_fraction(receipt: Mapping[str, Any] | None) -> float | None:
    if not receipt or receipt.get("g90_confirm_update") is None:
        return None
    updates = int(receipt["g90_confirm_update"])
    batch = int(receipt.get("batch_rows", 0))
    if batch <= 0:
        return None
    return updates * batch / _core.CYR11_ARK_MAX_ROW_PRESENTATIONS


def exposure_aware_final_decision(*, compact: Mapping[str, Any] | None,
                                  production_primary: Mapping[str, Any] | None,
                                  production_replication: Mapping[str, Any] | None) -> dict[str, Any]:
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
        "schema": "anra-cyr-gpu011-decision/v2", "verdict": verdict,
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
            "A representation-divergence label is allowed only when production received at least the "
            "semantic exposure at which compact qualified. G90 claims require sustained DEV_CONTROLLER "
            "performance and >=0.90 final DEV_MEASUREMENT STANDARD exact-with-EOS."
        ),
        "broad_reasoning_claim_authorized": False,
        "production_promotion_authorized": False,
        "pre500m_authorized": False,
        "training_500m_authorized": False,
    }


def _target_updates(batch_rows: int) -> int:
    batch = int(batch_rows)
    if batch not in (16, 32, 64):
        raise ValueError(f"unsupported CYR-GPU-011 batch size: {batch}")
    return int(math.ceil(_core.CYR11_ARK_MAX_ROW_PRESENTATIONS / batch))


def _project_rows(rec: Mapping[str, Any], budget_seconds: float) -> tuple[int, int]:
    batch = int(rec["batch_rows"])
    ups = max(float(rec["training_updates_per_sec"]), 1e-9)
    eps = max(float(rec["generation_examples_per_sec"]), 1e-9)
    target_updates = _target_updates(batch)
    basic_eval_examples = 64 + 85 + 100
    evals_per_update = batch / _core.CYR11_EVAL_EVERY_ROW_PRESENTATIONS
    seconds_per_update = 1.0 / ups + (basic_eval_examples / eps) * evals_per_update
    usable = max(0.0, float(budget_seconds) - CYR11_ACQUISITION_FINALIZE_RESERVE_SECONDS)
    projected_updates = min(target_updates, int(usable / max(seconds_per_update, 1e-9)))
    return projected_updates, projected_updates * batch


def resolve_from_calibrations(calibrations: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Choose batches by projected semantic exposure, without feasibility refusal."""
    compact = [rec for key, rec in calibrations.items()
               if key.startswith("COMPACT_B") and rec.get("status") == "PASS"]
    production = [rec for key, rec in calibrations.items()
                  if key.startswith("PRODUCTION_B") and rec.get("status") == "PASS"]
    if not compact:
        raise ValueError("no healthy compact bridge calibration")

    compact_scored = []
    for rec in compact:
        updates, rows = _project_rows(rec, _core.CYR11_COMPACT_STAGE_CAP_MINUTES * 60.0)
        batch = int(rec["batch_rows"])
        compact_scored.append((rows * (1.03 if batch == 64 else 1.0), rows, batch, updates, rec))
    compact_scored.sort(reverse=True, key=lambda item: (item[0], item[1], item[2]))
    _cscore, compact_rows, compact_batch, compact_projected_updates, compact_pick = compact_scored[0]

    prod_pick = None; prod_rows = prod_batch = prod_projected_updates = 0
    if production:
        prod_budget = (_core.CYR11_WALL_MINUTES - _core.CYR11_PACKAGING_RESERVE_MINUTES
                       - _core.CYR11_COMPACT_STAGE_CAP_MINUTES) * 60.0
        prod_scored = []
        for rec in production:
            updates, rows = _project_rows(rec, prod_budget)
            batch = int(rec["batch_rows"])
            prod_scored.append((rows * (1.03 if batch == 64 else 1.0), rows, batch, updates, rec))
        prod_scored.sort(reverse=True, key=lambda item: (item[0], item[1], item[2]))
        _pscore, prod_rows, prod_batch, prod_projected_updates, prod_pick = prod_scored[0]

    resolved = {
        "schema": "anra-cyr-gpu011-resolved/v2", "experiment": _core.CYR11_ID,
        "wall_budget_minutes": _core.CYR11_WALL_MINUTES,
        "packaging_reserve_minutes": _core.CYR11_PACKAGING_RESERVE_MINUTES,
        "acquisition_finalize_reserve_seconds": CYR11_ACQUISITION_FINALIZE_RESERVE_SECONDS,
        "compact_stage_cap_minutes": _core.CYR11_COMPACT_STAGE_CAP_MINUTES,
        "compact_batch_rows": int(compact_batch),
        "compact_target_updates": _target_updates(compact_batch),
        "compact_projected_updates": int(compact_projected_updates),
        "compact_projected_row_presentations": int(compact_rows),
        "compact_projected_ark_exposure_fraction": compact_rows / _core.CYR11_ARK_MAX_ROW_PRESENTATIONS,
        "compact_updates_per_sec": float(compact_pick["training_updates_per_sec"]),
        "production_available": prod_pick is not None,
        "production_batch_rows": int(prod_batch) if prod_pick else None,
        "production_target_updates": _target_updates(prod_batch) if prod_pick else 0,
        "production_updates_per_sec": float(prod_pick["training_updates_per_sec"]) if prod_pick else 0.0,
        "production_generation_examples_per_sec": float(prod_pick["generation_examples_per_sec"]) if prod_pick else 0.0,
        "production_projected_updates": int(prod_projected_updates),
        "production_projected_row_presentations": int(prod_rows),
        "production_projected_ark_exposure_fraction": prod_rows / _core.CYR11_ARK_MAX_ROW_PRESENTATIONS,
        "ark_reference_batch_rows": _core.CYR11_ARK_BATCH,
        "ark_reference_max_updates": _core.CYR11_MAX_UPDATES,
        "ark_reference_max_row_presentations": _core.CYR11_ARK_MAX_ROW_PRESENTATIONS,
        "eval_every_row_presentations": _core.CYR11_EVAL_EVERY_ROW_PRESENTATIONS,
        "second_seed_launch_minutes": _core.CYR11_SECOND_SEED_LAUNCH_MINUTES,
        "claim_ceiling": "CONTROLLED_TASK_DEVELOPMENT_ONLY",
        "production_promotion_authorized": False, "pre500m_authorized": False,
        "training_500m_authorized": False,
    }
    _core.validate_resolved(resolved)
    if resolved["compact_target_updates"] * resolved["compact_batch_rows"] < _core.CYR11_ARK_MAX_ROW_PRESENTATIONS:
        raise AssertionError("compact target underexposes ARK reference")
    if prod_pick and resolved["production_target_updates"] * resolved["production_batch_rows"] < _core.CYR11_ARK_MAX_ROW_PRESENTATIONS:
        raise AssertionError("production target underexposes ARK reference")
    return resolved


@contextmanager
def _semantic_exposure_scope() -> Iterator[None]:
    """Target ARK semantic exposure while enforcing per-stage finalization time."""
    original = _runner.run_acquisition

    def exposure_matched_run(*args: Any, **kwargs: Any):
        batch = int(kwargs["batch_rows"])
        old_max = _core.CYR11_MAX_UPDATES
        outer_deadline = float(kwargs["deadline"])
        kwargs = dict(kwargs)
        kwargs["deadline"] = max(time.monotonic(), outer_deadline - CYR11_ACQUISITION_FINALIZE_RESERVE_SECONDS)
        _core.CYR11_MAX_UPDATES = _target_updates(batch)
        try:
            return original(*args, **kwargs)
        finally:
            _core.CYR11_MAX_UPDATES = old_max

    _runner.run_acquisition = exposure_matched_run
    try:
        yield
    finally:
        if _runner.run_acquisition is exposure_matched_run:
            _runner.run_acquisition = original


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
    with canonical_optimizer_compat(), _semantic_exposure_scope(), _decision_scope():
        return _runner.run_campaign(*args, **kwargs)


__all__ = ["BUNDLE_NAME", "calibrate_all", "exposure_aware_final_decision",
           "production_tokenizer", "read_json", "resolve_from_calibrations",
           "run_campaign", "write_json"]
