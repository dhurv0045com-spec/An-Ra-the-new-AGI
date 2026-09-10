"""CYR-GPU-013 / R1B: replicated vocabulary response-curve screen.

R1B follows the one-seed CYR-GPU-012 result by mapping intermediate tied
embedding/output class-space sizes with fresh matched seeds. It is an early
capability-formation experiment, not a production-tokenizer or AGI claim.
"""
from __future__ import annotations

from typing import Any, Mapping

EXPERIMENT = "CYR-GPU-013-R1B"
VOCABS = (19, 1024, 4096, 8192, 16384, 24576)
MODEL_SEEDS = (3611, 3612, 3613)
ORDER_SEEDS = (5901, 5902, 5903)
ARM_ORDERS = (
    VOCABS,
    tuple(reversed(VOCABS)),
    (4096, 16384, 19, 8192, 24576, 1024),
)
BATCH_ROWS = 64
UPDATES = 2_000
ROW_PRESENTATIONS = BATCH_ROWS * UPDATES
WALL_MINUTES = 175.0
PACKAGING_RESERVE_MINUTES = 5.0
RUNTIME_SAFETY_FACTOR = 1.35
MIN_FINALIZE_SECONDS = 120.0
INTERMEDIATE = (1024, 4096, 8192, 16384)
EXTREMES = (19, 24576)
MIN_INTERMEDIATE_SIGNAL = 0.60
STRONG_GAP = 0.30
EQUIVALENT_GAP = 0.10


def arm_label(seed_index: int, vocab: int) -> str:
    return f"S{seed_index + 1}_CHAR_V{int(vocab)}"


def estimate_arm_seconds(cal: Mapping[str, Any]) -> float:
    """Conservative pre-outcome estimate for one fixed 2k-update arm."""
    ups = max(float(cal["training_updates_per_sec"]), 1e-9)
    eps = max(float(cal["generation_examples_per_sec"]), 1e-9)
    eval_count = UPDATES // 200
    basic_eval_examples = 64 + 85 + 100
    structural_examples = 85 + 85 + 96 + 64 + 48 + 48
    # Baseline/final plus periodic/milestone structural batteries.
    structural_budget = 7 * structural_examples
    final_examples = structural_examples + 64
    raw = (UPDATES / ups
           + eval_count * basic_eval_examples / eps
           + structural_budget / eps
           + final_examples / eps)
    return raw * RUNTIME_SAFETY_FACTOR + MIN_FINALIZE_SECONDS


def resolve_from_calibrations(calibrations: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    estimates: dict[str, float] = {}
    for vocab in VOCABS:
        key = f"V{vocab}"
        rec = calibrations.get(key)
        if not rec or rec.get("status") != "PASS" or int(rec.get("batch_rows", -1)) != BATCH_ROWS:
            raise RuntimeError(f"R1B missing healthy fixed-batch calibration for {key}")
        estimates[key] = estimate_arm_seconds(rec)
    one_curve = sum(estimates[f"V{v}"] for v in VOCABS)
    science_seconds = (WALL_MINUTES - PACKAGING_RESERVE_MINUTES) * 60.0
    mandatory = 2 * one_curve
    if mandatory > science_seconds:
        raise RuntimeError(
            f"R1B two complete response curves do not conservatively fit: "
            f"need {mandatory/60:.1f} min, have {science_seconds/60:.1f} min"
        )
    curves = 3 if 3 * one_curve <= science_seconds else 2
    return {
        "schema": "anra-cyr-gpu013-r1b-resolved/v1",
        "experiment": EXPERIMENT,
        "vocabs": list(VOCABS),
        "batch_rows": BATCH_ROWS,
        "updates_per_arm": UPDATES,
        "row_presentations_per_arm": ROW_PRESENTATIONS,
        "mandatory_curves": 2,
        "curves_to_run": curves,
        "estimated_arm_seconds": estimates,
        "estimated_curve_seconds": one_curve,
        "wall_minutes": WALL_MINUTES,
        "packaging_reserve_minutes": PACKAGING_RESERVE_MINUTES,
        "runtime_safety_factor": RUNTIME_SAFETY_FACTOR,
        "pre500m_authorized": False,
        "training_500m_authorized": False,
    }


def endpoint_score(acquisition: Mapping[str, Any]) -> float | None:
    if int(acquisition.get("row_presentations", -1)) != ROW_PRESENTATIONS:
        return None
    final = acquisition.get("reasoning_battery_final", {}).get("STANDARD", {})
    if "complete_exact_with_valid_stop" not in final:
        return None
    return float(final["complete_exact_with_valid_stop"])


def decision(arms: Mapping[str, Mapping[str, Any]], curves_run: int) -> dict[str, Any]:
    per_seed: list[dict[str, Any]] = []
    for i in range(int(curves_run)):
        scores: dict[int, float] = {}
        complete = True
        for vocab in VOCABS:
            body = arms.get(arm_label(i, vocab))
            if not body:
                complete = False
                break
            acq = body.get("acquisition", body)
            score = endpoint_score(acq)
            if score is None:
                complete = False
                break
            scores[vocab] = score
        if not complete:
            continue
        ibest_vocab = max(INTERMEDIATE, key=lambda v: scores[v])
        ibest = scores[ibest_vocab]
        ebest = max(scores[v] for v in EXTREMES)
        per_seed.append({
            "seed_index": i + 1,
            "model_seed": MODEL_SEEDS[i],
            "order_seed": ORDER_SEEDS[i],
            "scores": {str(k): v for k, v in scores.items()},
            "intermediate_best_vocab": ibest_vocab,
            "intermediate_best": ibest,
            "extreme_best": ebest,
            "intermediate_gap": ibest - ebest,
            "argmax_vocab": max(VOCABS, key=lambda v: scores[v]),
        })

    mandatory = per_seed[:2]
    if len(mandatory) < 2:
        verdict = "INCONCLUSIVE_NO_TWO_COMPLETE_CURVES"
    else:
        supported = [
            x["intermediate_best"] >= MIN_INTERMEDIATE_SIGNAL
            and x["intermediate_gap"] >= STRONG_GAP
            for x in mandatory
        ]
        no_advantage = [
            x["intermediate_best"] < MIN_INTERMEDIATE_SIGNAL
            or x["intermediate_gap"] <= EQUIVALENT_GAP
            for x in mandatory
        ]
        if all(supported):
            verdict = "REPLICATED_INTERMEDIATE_CLASS_SPACE_ADVANTAGE"
        elif all(no_advantage):
            verdict = "NO_REPLICATED_INTERMEDIATE_ADVANTAGE"
        else:
            verdict = "MIXED_OR_SEED_SENSITIVE_RESPONSE_CURVE"

    return {
        "schema": "anra-cyr-gpu013-r1b-decision/v1",
        "verdict": verdict,
        "primary_endpoint_rows": ROW_PRESENTATIONS,
        "mandatory_seed_count": 2,
        "per_seed": per_seed,
        "thresholds": {
            "min_intermediate_signal": MIN_INTERMEDIATE_SIGNAL,
            "strong_gap": STRONG_GAP,
            "equivalent_gap": EQUIVALENT_GAP,
        },
        "claim_ceiling": "CONTROLLED_DEVELOPMENT_MECHANISM_ONLY",
        "production_tokenizer_change_authorized": False,
        "pre500m_authorized": False,
        "training_500m_authorized": False,
        "agi_claim_authorized": False,
    }


__all__ = [
    "ARM_ORDERS", "BATCH_ROWS", "EXPERIMENT", "MODEL_SEEDS", "ORDER_SEEDS",
    "ROW_PRESENTATIONS", "UPDATES", "VOCABS", "WALL_MINUTES", "arm_label",
    "decision", "endpoint_score", "estimate_arm_seconds", "resolve_from_calibrations",
]
