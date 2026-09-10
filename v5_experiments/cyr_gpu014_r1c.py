"""CYR-GPU-014 / R1C fixed-matrix softmax-competition mechanism experiment.

Every scientific arm uses the same physical V24576 Cymek V5 model.  Only the
training-logit treatment changes.  Full-vocabulary candidate-free evaluation
is never masked, so a positive result can be attributed to training-time
normalizer/competition changes rather than a smaller executable model.
"""
from __future__ import annotations

import math
from typing import Any, Mapping

EXPERIMENT = "CYR-GPU-014-R1C"
PHYSICAL_VOCAB = 24_576
ACTIVE_VOCAB = 19
BATCH_ROWS = 64
UPDATES = 3_000
ROW_PRESENTATIONS = BATCH_ROWS * UPDATES
EVAL_EVERY = 150
DIAGNOSTIC_UPDATES = (0, 300, 600, 900, 1500, 2100, 3000)
CHECKPOINT_EVERY = 500
MODEL_SEEDS = (3711, 3712, 3713, 3714)
ORDER_SEEDS = (6001, 6002, 6003, 6004)
ARMS = (
    "FULL_24576",
    "MASK_19",
    "MASK_4096",
    "MASK_8192",
    "MASK_16384",
    "OFFSET_EQ4096",
)
ARM_ORDERS = (
    ARMS,
    tuple(reversed(ARMS)),
    ("MASK_4096", "FULL_24576", "MASK_16384", "MASK_19", "OFFSET_EQ4096", "MASK_8192"),
    ("OFFSET_EQ4096", "MASK_8192", "MASK_19", "MASK_16384", "FULL_24576", "MASK_4096"),
)
PRIMARY_ARM = "MASK_4096"
REFERENCE_ARM = "FULL_24576"
WALL_MINUTES = 420.0
PACKAGING_RESERVE_MINUTES = 10.0
RUNTIME_SAFETY_FACTOR = 1.25
MIN_FINALIZE_SECONDS = 180.0
PRIMARY_SEED_GAP = 0.20
PRIMARY_MEAN_GAP = 0.25
NOT_SUFFICIENT_MEAN_GAP = 0.10
G50 = 0.50
OFFSET_EQ4096 = math.log((PHYSICAL_VOCAB - ACTIVE_VOCAB) / (4096 - ACTIVE_VOCAB))


def arm_label(seed_index: int, arm: str) -> str:
    if arm not in ARMS:
        raise ValueError(f"unknown R1C arm {arm}")
    return f"S{seed_index + 1}_{arm}"


def candidate_count(arm: str) -> int | None:
    if arm.startswith("MASK_"):
        return int(arm.split("_", 1)[1])
    return None


def treatment_spec(arm: str) -> dict[str, Any]:
    if arm not in ARMS:
        raise ValueError(arm)
    if arm == "FULL_24576":
        return {"kind": "full", "candidate_count": PHYSICAL_VOCAB, "inactive_logit_offset": 0.0}
    if arm.startswith("MASK_"):
        k = candidate_count(arm)
        assert k is not None
        if not ACTIVE_VOCAB <= k <= PHYSICAL_VOCAB:
            raise ValueError("invalid mask candidate count")
        return {"kind": "hard_mask", "candidate_count": k, "inactive_logit_offset": None}
    if arm == "OFFSET_EQ4096":
        return {
            "kind": "inactive_offset",
            "candidate_count": PHYSICAL_VOCAB,
            "inactive_logit_offset": OFFSET_EQ4096,
            "effective_equal_logit_inactive_count": 4096 - ACTIVE_VOCAB,
        }
    raise AssertionError(arm)


def apply_training_logits(logits: Any, arm: str, *, torch_module: Any) -> Any:
    """Apply the frozen training-only treatment without changing tensor shape."""
    spec = treatment_spec(arm)
    if spec["kind"] == "full":
        return logits
    out = logits.clone()
    if spec["kind"] == "hard_mask":
        k = int(spec["candidate_count"])
        if k < PHYSICAL_VOCAB:
            # Use a large finite negative value rather than -inf so BF16/FP32 CE
            # and diagnostic reductions remain finite on supported CUDA kernels.
            out[..., k:] = -1.0e4
        return out
    if spec["kind"] == "inactive_offset":
        out[..., ACTIVE_VOCAB:] = out[..., ACTIVE_VOCAB:] - float(spec["inactive_logit_offset"])
        return out
    raise AssertionError(spec)


def _eval_points(trace: list[Mapping[str, Any]]) -> list[tuple[int, float]]:
    points: list[tuple[int, float]] = []
    for row in trace:
        update = int(row.get("update", -1))
        if update < 0:
            continue
        metric = row.get("dev_measurement", {})
        if "complete_exact_with_valid_stop" in metric:
            points.append((update, float(metric["complete_exact_with_valid_stop"])))
    points.sort()
    return points


def formation_metrics(acquisition: Mapping[str, Any]) -> dict[str, Any]:
    points = _eval_points(list(acquisition.get("trace", [])))
    if int(acquisition.get("updates", -1)) != UPDATES:
        return {"complete": False}
    eligible = [(u, s) for u, s in points if 600 <= u <= UPDATES]
    if not eligible:
        return {"complete": False}
    values = [s for _u, s in eligible]
    sustained = None
    for i in range(len(eligible) - 2):
        tri = eligible[i:i + 3]
        if all(s >= G50 for _u, s in tri):
            sustained = tri[0][0]
            break
    peak3 = 0.0
    if len(values) >= 3:
        peak3 = max(sum(values[i:i + 3]) / 3.0 for i in range(len(values) - 2))
    else:
        peak3 = max(values)
    endpoint = None
    for u, s in points:
        if u == UPDATES:
            endpoint = s
    if endpoint is None:
        endpoint = float(acquisition.get("reasoning_battery_final", {}).get("STANDARD", {}).get(
            "complete_exact_with_valid_stop", 0.0
        ))
    return {
        "complete": True,
        "formation_auc": sum(values) / len(values),
        "sustained_g50_update": sustained,
        "endpoint_standard": endpoint,
        "peak_3eval_standard": peak3,
        "eval_count": len(values),
    }


def decision(arms: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    per_seed: list[dict[str, Any]] = []
    for i in range(len(MODEL_SEEDS)):
        row: dict[str, Any] = {
            "seed_index": i + 1,
            "model_seed": MODEL_SEEDS[i],
            "order_seed": ORDER_SEEDS[i],
            "arms": {},
        }
        complete = True
        for arm in ARMS:
            body = arms.get(arm_label(i, arm))
            if body is None:
                complete = False
                break
            acq = body.get("acquisition", body)
            fm = formation_metrics(acq)
            if not fm.get("complete"):
                complete = False
                break
            row["arms"][arm] = fm
        if complete:
            primary = row["arms"][PRIMARY_ARM]
            ref = row["arms"][REFERENCE_ARM]
            row["primary_auc_gap"] = primary["formation_auc"] - ref["formation_auc"]
            per_seed.append(row)

    if len(per_seed) < len(MODEL_SEEDS):
        verdict = "INCONCLUSIVE_INCOMPLETE_FOUR_SEED_CAMPAIGN"
    else:
        gaps = [float(r["primary_auc_gap"]) for r in per_seed]
        primary_g50 = sum(r["arms"][PRIMARY_ARM]["sustained_g50_update"] is not None for r in per_seed)
        ref_g50 = sum(r["arms"][REFERENCE_ARM]["sustained_g50_update"] is not None for r in per_seed)
        support = (
            sum(g >= PRIMARY_SEED_GAP for g in gaps) >= 3
            and sum(gaps) / len(gaps) >= PRIMARY_MEAN_GAP
            and primary_g50 >= 2
            and ref_g50 <= 1
        )
        not_sufficient = (
            sum(gaps) / len(gaps) < NOT_SUFFICIENT_MEAN_GAP
            and primary_g50 <= ref_g50 + 1
        )
        if support:
            verdict = "SOFTMAX_COMPETITION_CAUSALLY_SUPPORTED"
        elif not_sufficient:
            verdict = "SOFTMAX_COMPETITION_NOT_SUFFICIENT"
        else:
            verdict = "MIXED_SOFTMAX_COMPETITION_EFFECT"

    mass_rescue = False
    if len(per_seed) == len(MODEL_SEEDS):
        offset_gaps = [
            r["arms"]["OFFSET_EQ4096"]["formation_auc"] - r["arms"][REFERENCE_ARM]["formation_auc"]
            for r in per_seed
        ]
        offset_close = [
            abs(r["arms"]["OFFSET_EQ4096"]["formation_auc"] - r["arms"][PRIMARY_ARM]["formation_auc"])
            for r in per_seed
        ]
        mass_rescue = sum(g >= 0.20 for g in offset_gaps) >= 3 and (sum(offset_close) / 4.0) <= 0.15
    return {
        "schema": "anra-cyr-gpu014-r1c-decision/v1",
        "verdict": verdict,
        "per_seed": per_seed,
        "primary_pair": [PRIMARY_ARM, REFERENCE_ARM],
        "primary_thresholds": {
            "seed_auc_gap": PRIMARY_SEED_GAP,
            "required_seed_wins": "3/4",
            "mean_auc_gap": PRIMARY_MEAN_GAP,
            "mask4096_sustained_g50_min_seeds": 2,
            "full24576_sustained_g50_max_seeds": 1,
        },
        "inactive_partition_mass_rescue_supported": mass_rescue,
        "claim_ceiling": "CONTROLLED_DEVELOPMENT_MECHANISM_ONLY",
        "production_tokenizer_change_authorized": False,
        "pre500m_authorized": False,
        "training_500m_authorized": False,
        "broad_reasoning_claim_authorized": False,
        "agi_claim_authorized": False,
    }


def estimate_arm_seconds(cal: Mapping[str, Any]) -> float:
    ups = max(float(cal["training_updates_per_sec"]), 1e-9)
    eps = max(float(cal["generation_examples_per_sec"]), 1e-9)
    diag = max(float(cal.get("diagnostic_seconds", 0.0)), 0.0)
    eval_count = UPDATES // EVAL_EVERY
    basic_eval_examples = 64 + 85 + 100
    # Full structural battery is intentionally sparse: baseline, 1500, final.
    structural_examples = 85 + 85 + 96 + 64 + 48 + 48
    raw = (
        UPDATES / ups
        + eval_count * basic_eval_examples / eps
        + 3 * structural_examples / eps
        + len(DIAGNOSTIC_UPDATES) * diag
    )
    return raw * RUNTIME_SAFETY_FACTOR + MIN_FINALIZE_SECONDS


def resolve_from_calibrations(calibrations: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    estimates: dict[str, float] = {}
    for arm in ARMS:
        rec = calibrations.get(arm)
        if not rec or rec.get("status") != "PASS":
            raise RuntimeError(f"R1C missing healthy calibration for {arm}")
        if int(rec.get("batch_rows", -1)) != BATCH_ROWS:
            raise RuntimeError(f"R1C batch drift in calibration for {arm}")
        estimates[arm] = estimate_arm_seconds(rec)
    one_curve = sum(estimates.values())
    need = len(MODEL_SEEDS) * one_curve
    have = (WALL_MINUTES - PACKAGING_RESERVE_MINUTES) * 60.0
    if need > have:
        raise RuntimeError(
            f"R1C four complete six-arm curves do not conservatively fit: "
            f"need {need/60:.1f} min, have {have/60:.1f} min"
        )
    return {
        "schema": "anra-cyr-gpu014-r1c-resolved/v1",
        "experiment": EXPERIMENT,
        "arms": list(ARMS),
        "model_seeds": list(MODEL_SEEDS),
        "order_seeds": list(ORDER_SEEDS),
        "batch_rows": BATCH_ROWS,
        "updates_per_arm": UPDATES,
        "row_presentations_per_arm": ROW_PRESENTATIONS,
        "eval_every": EVAL_EVERY,
        "diagnostic_updates": list(DIAGNOSTIC_UPDATES),
        "estimated_arm_seconds": estimates,
        "estimated_curve_seconds": one_curve,
        "estimated_campaign_seconds": need,
        "wall_minutes": WALL_MINUTES,
        "packaging_reserve_minutes": PACKAGING_RESERVE_MINUTES,
        "runtime_safety_factor": RUNTIME_SAFETY_FACTOR,
        "pre500m_authorized": False,
        "training_500m_authorized": False,
    }


__all__ = [
    "ACTIVE_VOCAB", "ARMS", "ARM_ORDERS", "BATCH_ROWS", "CHECKPOINT_EVERY",
    "DIAGNOSTIC_UPDATES", "EVAL_EVERY", "EXPERIMENT", "MODEL_SEEDS", "OFFSET_EQ4096",
    "ORDER_SEEDS", "PHYSICAL_VOCAB", "ROW_PRESENTATIONS", "UPDATES", "WALL_MINUTES",
    "apply_training_logits", "arm_label", "candidate_count", "decision", "estimate_arm_seconds",
    "formation_metrics", "resolve_from_calibrations", "treatment_spec",
]
