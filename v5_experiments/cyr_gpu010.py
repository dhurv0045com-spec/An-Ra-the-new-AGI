"""CYR-GPU-010: long ResearchSmall capability-emergence experiment.

The experiment is motivated by two pieces of evidence:
1. CYR-GPU-009: Cymek TINY (2L/64w, 1.647M params) memorized its T2 train
   probe but did not reach candidate-free held-out G90 in 2M real tokens / ~15.6k updates.
2. Arkenstone ARK-002B: a 4L/128w Micro subject showed a replicated
   memorize-first -> generalize-later transition with sustained G90 emerging
   in the ~9k-18k update regime. ARK-003 found no reliable acceleration from
   simple curriculum/teacher suffixes. ARK-015 showed that learned invariances
   can be brittle under narrow continuation and that invariant-supporting data
   can preserve them.

CYR-GPU-010 therefore spends one long Colab session on the closest Cymek V5
scale match: RESEARCH_SMALL (4L/128w, production tokenizer). It measures dense
candidate-free emergence and structural diagnostics. If G90 appears early,
remaining wall time is spent on a matched same-parent invariance-support stress.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from typing import Any, Mapping

from v5_experiments import cyr_gpu005 as base

CYR10_ID = "CYR-GPU-010"
CYR10_PARENT_SEED = 3101
CYR10_WALL_MINUTES = 175.0
CYR10_PACKAGING_RESERVE_MINUTES = 5.0
CYR10_MAX_ACQUISITION_UPDATES = 18_000
CYR10_EVAL_EVERY_UPDATES = 200
CYR10_G50 = 0.50
CYR10_G90 = 0.90
CYR10_CONFIRMATIONS = 3
CYR10_BATCH_ROWS = 16
CYR10_HIGH_LR = 1e-3
CYR10_STRESS_MIN_STEPS_PER_ARM = 1_200
CYR10_STRESS_MAX_STEPS_PER_ARM = 3_000
CYR10_STRESS_EVAL_EVERY = 200
CYR10_ARKENSTONE_SHA = "59e1b805b7d93b7f2e1e9d3ea66b34c4fabca9c8"
CYR10_V9_BUNDLE_SHA256 = "dc15f14d3bc81551b1f0b00285faa4b23c9e68f1341405377959a7aba108f216"
CYR10_ARK002B_SPLIT_SHA = "0dd9305697045b0fbf4e7f268b46a4d7276e4794af5d78b60e999df914ae4236"
CYR10_ARK015_BINDING_MANIFEST = "fbc8605dc1cfc19ac8692d2b2c6338fcb2af3e02b24907f3b5cca3571947fee3"

proxy_registry = base.proxy_registry
render_t2_worlds = base.render_t2_worlds
build_data_manifest = base.build_data_manifest
assert_manifest_sha = base.assert_manifest_sha
commutation_audit = base.commutation_audit
assert_proxy_in_registry = base.assert_proxy_in_registry


def stable_sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def resolve_from_calibrations(calibrations: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    science_seconds = (CYR10_WALL_MINUTES - CYR10_PACKAGING_RESERVE_MINUTES) * 60.0
    candidates = (("RESEARCH_SMALL", CYR10_MAX_ACQUISITION_UPDATES), ("TINY", 36_000))
    passing = []
    for name, max_updates in candidates:
        rec = calibrations.get(name, {})
        if rec.get("status") != "PASS":
            continue
        ups = float(rec.get("training_updates_per_sec", 0.0))
        eps = float(rec.get("generation_examples_per_sec", 0.0))
        if ups <= 0 or eps <= 0:
            continue
        eval_events = math.ceil(max_updates / CYR10_EVAL_EVERY_UPDATES)
        predicted = max_updates / ups + eval_events * 224 / eps + 180.0
        passing.append((name, max_updates, rec, predicted))
    if not passing:
        raise ValueError("no healthy calibrated proxy for CYR-GPU-010")
    target = next((row for row in passing if row[0] == "RESEARCH_SMALL" and row[3] <= science_seconds), None)
    if target is None:
        target = next((row for row in passing if row[0] == "TINY"), passing[0])
    name, max_updates, rec, predicted = target
    return {
        "schema": "anra-cyr-gpu010-resolved/v1", "experiment": CYR10_ID,
        "mode": "full", "proxy": name, "parent_seed": CYR10_PARENT_SEED,
        "max_acquisition_updates": int(max_updates), "eval_every_updates": CYR10_EVAL_EVERY_UPDATES,
        "batch_rows": CYR10_BATCH_ROWS, "wall_budget_minutes": CYR10_WALL_MINUTES,
        "packaging_reserve_minutes": CYR10_PACKAGING_RESERVE_MINUTES,
        "training_updates_per_sec": float(rec["training_updates_per_sec"]),
        "training_real_tokens_per_sec": float(rec["training_real_tokens_per_sec"]),
        "generation_examples_per_sec": float(rec["generation_examples_per_sec"]),
        "predicted_acquisition_seconds": round(predicted, 1),
        "scale_match_to_arkenstone": name == "RESEARCH_SMALL",
        "claim_ceiling": "SINGLE_SEED_4L128W_CAPABILITY_EMERGENCE_DEVELOPMENT" if name == "RESEARCH_SMALL" else "TINY_LONG_DOSE_DEVELOPMENT_ONLY",
        "production_promotion_authorized": False, "pre500m_authorized": False, "training_500m_authorized": False,
    }


def validate_resolved(resolved: Mapping[str, Any]) -> dict[str, Any]:
    required = {"schema", "experiment", "mode", "proxy", "parent_seed", "max_acquisition_updates",
                "eval_every_updates", "batch_rows", "wall_budget_minutes", "packaging_reserve_minutes",
                "training_updates_per_sec", "generation_examples_per_sec", "claim_ceiling"}
    missing = sorted(required - set(resolved))
    if missing:
        raise ValueError(f"CYR-GPU-010 resolution missing {missing}")
    if resolved["experiment"] != CYR10_ID or resolved.get("mode") != "full":
        raise ValueError("CYR-GPU-010 identity/mode drift")
    if int(resolved["parent_seed"]) != CYR10_PARENT_SEED:
        raise ValueError("CYR-GPU-010 seed drift")
    if int(resolved["batch_rows"]) != CYR10_BATCH_ROWS:
        raise ValueError("CYR-GPU-010 batch drift")
    if float(resolved["wall_budget_minutes"]) > CYR10_WALL_MINUTES:
        raise ValueError("CYR-GPU-010 exceeds 175-minute hard wall")
    if resolved.get("production_promotion_authorized"):
        raise ValueError("CYR-GPU-010 cannot authorize production")
    return dict(resolved)


def make_reasoning_battery(splits: Mapping[str, list[dict[str, Any]]]) -> dict[str, Any]:
    standard = [dict(row) for row in splits["dev_measurement"]]
    commuted = []
    locality = []
    for row in standard[:64]:
        a, b = int(row["a"]), int(row["b"])
        commuted.append({**row, "world_id": row["world_id"] + "/commuted", "prompt": f"{b} + {a} = "})
        ua, ub = a % 10, b % 10
        delta = 1 if ua < 9 and ua + 1 + ub <= 9 else (-1 if ua > 0 and ua - 1 + ub <= 9 else None)
        if delta is not None:
            a2 = a + delta
            locality.append({"world_id": row["world_id"] + "/locality/base", "pair_id": row["world_id"],
                             "role": "base", "prompt": row["prompt"], "answer": row["answer"],
                             "a": a, "b": b, "expected_delta": delta})
            locality.append({"world_id": row["world_id"] + "/locality/cf", "pair_id": row["world_id"],
                             "role": "counterfactual", "prompt": f"{a2} + {b} = ", "answer": str(a2 + b),
                             "a": a2, "b": b, "expected_delta": delta})
    rendering = [{**row, "world_id": row["world_id"] + "/render", "prompt": f"What is {row['a']} plus {row['b']}? Answer: "}
                 for row in standard[:48]]
    rng = random.Random(1010)
    three_digit = []
    while len(three_digit) < 48:
        da = [rng.randrange(1, 6), rng.randrange(10), rng.randrange(10)]
        db = [rng.randrange(1, 4), rng.randrange(10), rng.randrange(10)]
        if any(x + y > 9 for x, y in zip(da, db)):
            continue
        a = da[0] * 100 + da[1] * 10 + da[2]; b = db[0] * 100 + db[1] * 10 + db[2]
        three_digit.append({"world_id": f"t3/diag/{len(three_digit)}", "prompt": f"{a} + {b} = ", "answer": str(a + b), "a": a, "b": b})
    battery: dict[str, Any] = {"STANDARD": standard, "COMMUTED": commuted, "LOCALITY": locality,
                              "RENDERING": rendering, "THREE_DIGIT": three_digit}
    battery["_manifest"] = [{"name": k, "count": len(v), "sha256": stable_sha(v)} for k, v in battery.items()]
    return battery


def stress_steps_from_remaining(*, remaining_seconds: float, updates_per_sec: float) -> int:
    if updates_per_sec <= 0:
        return 0
    usable = max(0.0, remaining_seconds - 120.0)
    each = int(usable * updates_per_sec * 0.46)
    if each < CYR10_STRESS_MIN_STEPS_PER_ARM:
        return 0
    return min(each, CYR10_STRESS_MAX_STEPS_PER_ARM)


def classify_acquisition(*, g90_confirm_update: int | None, final_standard: float,
                         final_locality: float | None, proxy: str, updates: int) -> dict[str, Any]:
    if g90_confirm_update is None:
        verdict = "NO_SUSTAINED_G90_IN_BOX"
    elif final_locality is not None and final_locality >= 0.80:
        verdict = "G90_WITH_COUNTERFACTUAL_LOCALITY"
    else:
        verdict = "G90_WITHOUT_STRONG_LOCALITY_EVIDENCE"
    return {"schema": "anra-cyr-gpu010-acquisition-decision/v1", "verdict": verdict,
            "proxy": proxy, "updates": int(updates), "g90_confirm_update": g90_confirm_update,
            "final_standard": float(final_standard), "final_locality_consistency": final_locality,
            "production_promotion_authorized": False, "pre500m_authorized": False, "training_500m_authorized": False}
