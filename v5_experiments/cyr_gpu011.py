"""CYR-GPU-011: exposure-matched capability-emergence bridge.

This experiment exists because CYR-GPU-009 and the first CYR-GPU-010 design
were underexposed relative to the Arkenstone ARK-002B protocol they were being
compared with. ARK-002B used batch 64 for up to 18k optimizer updates: up to
1,152,000 semantic row presentations. CYR-GPU-009 used batch 16 and stopped at
~15.6k updates (~250k row presentations); the original CYR-GPU-010 design used
batch 16 and capped at 18k (~288k rows). That is only one quarter of the
ARK-002B maximum exposure.

CYR-GPU-011 removes that ambiguity. It uses the exact frozen commutation-free
ARK-002B manifest and runs two prospectively defined bridges on the real Cymek
V5 transformer geometry:

A. COMPACT_BRIDGE: 4L/128w Cymek V5 with the 19-symbol arithmetic vocabulary,
   batch 64 when possible. This tests whether Cymek's core dynamics can
   reproduce the delayed memorize->generalize transition under a compact task
   representation close to Arkenstone's.
B. PRODUCTION_BRIDGE: the same 4L/128w Cymek V5 geometry with Cymek's frozen
   24,576-token production tokenizer. Batch size is selected by pre-outcome
   hardware calibration, preferring semantic exposure while mildly preferring
   batch 64 for protocol comparability.

Both stages use candidate-free generation. Structural batteries are
measurement-only and cannot alter training. If the first production subject
reaches sustained G90 with >=40 minutes left, a second independent production
subject is launched for replication. Nothing in this module authorizes
production changes, PRE500M, or the 500M campaign.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from v5_experiments import cyr_gpu005 as base

CYR11_ID = "CYR-GPU-011"
CYR11_WALL_MINUTES = 175.0
CYR11_PACKAGING_RESERVE_MINUTES = 5.0
CYR11_COMPACT_STAGE_CAP_MINUTES = 25.0
CYR11_SECOND_SEED_LAUNCH_MINUTES = 40.0
CYR11_MAX_UPDATES = 18_000
CYR11_ARK_BATCH = 64
CYR11_ARK_MAX_ROW_PRESENTATIONS = CYR11_MAX_UPDATES * CYR11_ARK_BATCH
CYR11_EVAL_EVERY_ROW_PRESENTATIONS = 200 * CYR11_ARK_BATCH  # ARK-002B cadence
CYR11_G50 = 0.50
CYR11_G90 = 0.90
CYR11_CONFIRMATIONS = 3
CYR11_HIGH_LR = 1e-3
CYR11_MODEL_SEEDS = {"compact": 3301, "production_primary": 3401, "production_replication": 3402}
CYR11_ORDER_SEEDS = {"compact": 4701, "production_primary": 4701, "production_replication": 4702}
CYR11_ARK002B_SPLIT_SHA = "0dd9305697045b0fbf4e7f268b46a4d7276e4794af5d78b60e999df914ae4236"
CYR11_ARK002B_BLOB_SHA = "6c46fdf90139526b00e9041af2d511ed0ac24270"
CYR11_V9_BUNDLE_SHA256 = "dc15f14d3bc81551b1f0b00285faa4b23c9e68f1341405377959a7aba108f216"
CYR11_ARKENSTONE_AUDIT_SHA = "c16718a7841c3cc3eba2b4b2c0388a0e36c0b530"
CYR11_BRAMASTRA_AUDIT_SHA = "415250f44179f3310e5dc55addb21722290604fa"
CYR11_ARK015_MANIFEST = "fbc8605dc1cfc19ac8692d2b2c6338fcb2af3e02b24907f3b5cca3571947fee3"

COMPACT_TOKENS = ["<pad>", "<bos>", "<eos>", "0", "1", "2", "3", "4", "5",
                  "6", "7", "8", "9", "+", "-", "*", "/", "=", " "]


class CompactCharTokenizer:
    """The ARK-001/002B arithmetic alphabet with Cymek render semantics.

    Unlike Arkenstone's helper, ``encode`` does not prepend BOS because Cymek's
    render_batch owns BOS insertion. The symbol table and special IDs are
    otherwise identical.
    """

    vocabulary_size = len(COMPACT_TOKENS)
    pad_id, bos_id, eos_id = 0, 1, 2

    def __init__(self) -> None:
        self.table = {token: i for i, token in enumerate(COMPACT_TOKENS)}
        self.inverse = {i: token for token, i in self.table.items()}

    def encode(self, text: str) -> list[int]:
        try:
            return [self.table[ch] for ch in text]
        except KeyError as exc:
            raise ValueError(f"compact tokenizer cannot encode {exc.args[0]!r}") from exc

    def decode(self, ids: list[int]) -> str:
        return "".join(self.inverse.get(int(i), "") for i in ids
                       if int(i) not in (self.pad_id, self.bos_id, self.eos_id))

    @property
    def special(self) -> dict[str, int]:
        return {"pad_id": self.pad_id, "bos_id": self.bos_id, "eos_id": self.eos_id}


def stable_sha(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                 ensure_ascii=False).encode("utf-8")).hexdigest()


def research_small_spec(vocab_size: int):
    registry = base.proxy_registry(vocab_size=24_576)
    return replace(registry["RESEARCH_SMALL"]["spec"], vocabulary_size=int(vocab_size))


def model_receipts() -> dict[str, Any]:
    compact = research_small_spec(len(COMPACT_TOKENS))
    production = research_small_spec(24_576)
    return {
        "COMPACT_BRIDGE": {"spec": compact.canonical(), "parameters": compact.parameter_receipt().total},
        "PRODUCTION_BRIDGE": {"spec": production.canonical(), "parameters": production.parameter_receipt().total},
    }


def _parse_prompt(prompt: str) -> tuple[int, int]:
    match = re.fullmatch(r"\s*(\d+) \+ (\d+) =\s*", prompt)
    if not match:
        raise ValueError(f"unexpected ARK-002B prompt: {prompt!r}")
    return int(match.group(1)), int(match.group(2))


def _rowify(pair: list[str] | tuple[str, str], role: str, index: int) -> dict[str, Any]:
    prompt, answer = str(pair[0]), str(pair[1])
    a, b = _parse_prompt(prompt)
    if str(a + b) != answer:
        raise ValueError(f"bad arithmetic row {prompt!r} -> {answer!r}")
    lo, hi = sorted((a, b))
    return {"world_id": f"ark002b/{role}/{index}", "prompt": prompt, "answer": answer,
            "a": a, "b": b, "canonical_pair": [lo, hi], "source_role": role}


def load_ark002b_manifest(path: str | Path) -> dict[str, Any]:
    body = json.loads(Path(path).read_text("utf-8"))
    if body.get("schema") != "arkenstone-task-manifest/v1" or body.get("task") != "t2-no-carry-add":
        raise ValueError("wrong ARK-002B manifest identity")
    if body.get("split_sha256") != CYR11_ARK002B_SPLIT_SHA:
        raise ValueError("ARK-002B split SHA drift")
    if body.get("counts") != {"train": 500, "test": 197}:
        raise ValueError("ARK-002B counts drift")
    train = [_rowify(row, "train", i) for i, row in enumerate(body["train"])]
    test = [_rowify(row, "test", i) for i, row in enumerate(body["test"])]
    train_pairs = {tuple(row["canonical_pair"]) for row in train}
    test_pairs = {tuple(row["canonical_pair"]) for row in test}
    overlap = train_pairs & test_pairs
    if overlap or int(body.get("pair_overlap_train_test", -1)) != 0:
        raise ValueError(f"ARK-002B commutation firewall violated: {len(overlap)} pairs")

    ordered = sorted(test, key=lambda row: hashlib.sha256(
        (row["prompt"] + "\0" + row["answer"]).encode()).hexdigest())
    controller = [dict(row, eval_role="DEV_CONTROLLER") for row in ordered[:64]]
    measurement = [dict(row, eval_role="DEV_MEASUREMENT") for row in ordered[64:149]]
    sealed = [dict(row, eval_role="SEALED_RESERVED") for row in ordered[149:]]
    if (len(controller), len(measurement), len(sealed)) != (64, 85, 48):
        raise AssertionError("evaluation firewall count drift")
    return {
        "schema": "anra-cyr-gpu011-ark002b/v1", "source_split_sha256": body["split_sha256"],
        "source_blob_sha": CYR11_ARK002B_BLOB_SHA, "train": train,
        "dev_controller": controller, "dev_measurement": measurement,
        "sealed_reserved": sealed,
        "train_test_canonical_overlap": 0,
        "role_sha256": stable_sha({"controller": [r["world_id"] for r in controller],
                                    "measurement": [r["world_id"] for r in measurement],
                                    "sealed": [r["world_id"] for r in sealed]}),
    }


def _avoid_train_pair(a: int, b: int, train_pairs: set[tuple[int, int]]) -> bool:
    return tuple(sorted((a, b))) not in train_pairs


def make_reasoning_battery(data: Mapping[str, Any]) -> dict[str, Any]:
    standard = [dict(row) for row in data["dev_measurement"]]
    train_pairs = {tuple(row["canonical_pair"]) for row in data["train"]}

    commuted = [{**row, "world_id": row["world_id"] + "/commuted",
                 "prompt": f"{row['b']} + {row['a']} = "} for row in standard]

    locality: list[dict[str, Any]] = []
    for row in standard:
        if len(locality) >= 96:
            break
        a, b = int(row["a"]), int(row["b"])
        candidates = []
        if a % 10 < 9 and (a % 10) + 1 + (b % 10) <= 9:
            candidates.append(1)
        if a % 10 > 0 and (a % 10) - 1 + (b % 10) <= 9:
            candidates.append(-1)
        for delta in candidates:
            a2 = a + delta
            if not _avoid_train_pair(a2, b, train_pairs):
                continue
            pair_id = row["world_id"] + f"/d{delta:+d}"
            locality.extend([
                {**row, "world_id": pair_id + "/base", "pair_id": pair_id,
                 "role": "base", "expected_delta": delta},
                {"world_id": pair_id + "/cf", "pair_id": pair_id, "role": "counterfactual",
                 "expected_delta": delta, "prompt": f"{a2} + {b} = ", "answer": str(a2 + b),
                 "a": a2, "b": b, "canonical_pair": list(sorted((a2, b)))},
            ])
            break

    carry: list[dict[str, Any]] = []
    for ta in (6, 7):
        for tb in (1, 2):
            for ua in range(1, 10):
                for ub in range(1, 10):
                    a, b = ta * 10 + ua, tb * 10 + ub
                    if ua + ub < 10 or a + b > 99 or not _avoid_train_pair(a, b, train_pairs):
                        continue
                    carry.append({"world_id": f"carry/{len(carry)}", "prompt": f"{a} + {b} = ",
                                  "answer": str(a + b), "a": a, "b": b})
                    if len(carry) >= 64:
                        break
                if len(carry) >= 64: break
            if len(carry) >= 64: break
        if len(carry) >= 64: break

    triple: list[dict[str, Any]] = []
    for a in range(60, 80):
        for b in range(10, 20):
            for c in range(1, 10):
                if (a % 10) + (b % 10) + c > 9 or a + b + c > 99:
                    continue
                triple.append({"world_id": f"triple/{len(triple)}", "prompt": f"{a} + {b} + {c} = ",
                               "answer": str(a + b + c), "a": a, "b": b, "c": c})
                if len(triple) >= 48: break
            if len(triple) >= 48: break
        if len(triple) >= 48: break

    three_digit: list[dict[str, Any]] = []
    for a in range(600, 680):
        for b in range(100, 140):
            if any(int(x) + int(y) > 9 for x, y in zip(str(a), str(b))):
                continue
            three_digit.append({"world_id": f"three_digit/{len(three_digit)}", "prompt": f"{a} + {b} = ",
                                "answer": str(a + b), "a": a, "b": b})
            if len(three_digit) >= 48: break
        if len(three_digit) >= 48: break

    verbal = [{**row, "world_id": row["world_id"] + "/verbal",
               "prompt": f"What is {row['a']} plus {row['b']}? Answer: "} for row in standard[:48]]

    battery = {"STANDARD": standard, "COMMUTED": commuted, "LOCALITY": locality,
               "CARRY": carry, "TRIPLE_ADD": triple, "THREE_DIGIT": three_digit,
               "VERBAL": verbal}
    battery["_manifest"] = [{"name": name, "count": len(rows), "sha256": stable_sha(rows)}
                            for name, rows in battery.items()]
    return battery


def calibration_key(regime: str, batch_rows: int) -> str:
    return f"{regime}_B{int(batch_rows)}"


def resolve_from_calibrations(calibrations: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    compact = [rec for key, rec in calibrations.items()
               if key.startswith("COMPACT_B") and rec.get("status") == "PASS"]
    production = [rec for key, rec in calibrations.items()
                  if key.startswith("PRODUCTION_B") and rec.get("status") == "PASS"]
    if not compact:
        raise ValueError("no healthy compact bridge calibration")

    compact.sort(key=lambda r: (int(r["batch_rows"]) == CYR11_ARK_BATCH, int(r["batch_rows"])), reverse=True)
    compact_pick = compact[0]

    prod_pick = None
    if production:
        prod_budget_s = (CYR11_WALL_MINUTES - CYR11_PACKAGING_RESERVE_MINUTES - CYR11_COMPACT_STAGE_CAP_MINUTES) * 60.0
        scored = []
        for rec in production:
            ups = float(rec["training_updates_per_sec"])
            eps = max(float(rec["generation_examples_per_sec"]), 1e-9)
            batch = int(rec["batch_rows"])
            usable = max(60.0, prod_budget_s - 480.0)
            per_update = 1.0 / max(ups, 1e-9) + (64.0 / eps) / (CYR11_EVAL_EVERY_ROW_PRESENTATIONS / batch)
            max_updates = min(CYR11_MAX_UPDATES, int(usable / per_update))
            projected_rows = max_updates * batch
            score = projected_rows * (1.10 if batch == CYR11_ARK_BATCH else 1.0)
            scored.append((score, projected_rows, batch, max_updates, rec))
        scored.sort(reverse=True, key=lambda x: (x[0], x[1], x[2]))
        _score, projected_rows, batch, max_updates, prod_pick = scored[0]
    else:
        projected_rows = batch = max_updates = 0

    return {
        "schema": "anra-cyr-gpu011-resolved/v1", "experiment": CYR11_ID,
        "wall_budget_minutes": CYR11_WALL_MINUTES,
        "packaging_reserve_minutes": CYR11_PACKAGING_RESERVE_MINUTES,
        "compact_stage_cap_minutes": CYR11_COMPACT_STAGE_CAP_MINUTES,
        "compact_batch_rows": int(compact_pick["batch_rows"]),
        "compact_updates_per_sec": float(compact_pick["training_updates_per_sec"]),
        "production_available": prod_pick is not None,
        "production_batch_rows": int(batch) if prod_pick else None,
        "production_updates_per_sec": float(prod_pick["training_updates_per_sec"]) if prod_pick else 0.0,
        "production_generation_examples_per_sec": float(prod_pick["generation_examples_per_sec"]) if prod_pick else 0.0,
        "production_projected_updates": int(max_updates),
        "production_projected_row_presentations": int(projected_rows),
        "production_projected_ark_exposure_fraction": float(projected_rows / CYR11_ARK_MAX_ROW_PRESENTATIONS),
        "ark_reference_batch_rows": CYR11_ARK_BATCH,
        "ark_reference_max_updates": CYR11_MAX_UPDATES,
        "ark_reference_max_row_presentations": CYR11_ARK_MAX_ROW_PRESENTATIONS,
        "eval_every_row_presentations": CYR11_EVAL_EVERY_ROW_PRESENTATIONS,
        "second_seed_launch_minutes": CYR11_SECOND_SEED_LAUNCH_MINUTES,
        "claim_ceiling": "CONTROLLED_TASK_DEVELOPMENT_ONLY",
        "production_promotion_authorized": False, "pre500m_authorized": False,
        "training_500m_authorized": False,
    }


def validate_resolved(resolved: Mapping[str, Any]) -> dict[str, Any]:
    if resolved.get("experiment") != CYR11_ID:
        raise ValueError("CYR-GPU-011 identity drift")
    if float(resolved.get("wall_budget_minutes", 999)) > CYR11_WALL_MINUTES:
        raise ValueError("CYR-GPU-011 exceeds hard wall")
    if int(resolved.get("compact_batch_rows", 0)) not in (16, 32, 64):
        raise ValueError("invalid compact batch")
    if resolved.get("production_available") and int(resolved.get("production_batch_rows", 0)) not in (16, 32, 64):
        raise ValueError("invalid production batch")
    if resolved.get("production_promotion_authorized") or resolved.get("pre500m_authorized") or resolved.get("training_500m_authorized"):
        raise ValueError("CYR-GPU-011 cannot authorize production")
    return dict(resolved)


def structural_flags(battery_result: Mapping[str, Any]) -> dict[str, bool]:
    def exact(name: str) -> float:
        return float(battery_result.get(name, {}).get("complete_exact_with_valid_stop", 0.0))
    locality = float(battery_result.get("LOCALITY", {}).get("structural", {}).get("counterfactual_relation_consistency", 0.0))
    usable = float(battery_result.get("LOCALITY", {}).get("structural", {}).get("numeric_usable_fraction", 0.0))
    return {
        "STANDARD_G90": exact("STANDARD") >= 0.90,
        "COMMUTATION_INVARIANCE": exact("COMMUTED") >= 0.90,
        "COUNTERFACTUAL_LOCALITY": locality >= 0.80 and usable >= 0.80,
        "CARRY_TRANSFER": exact("CARRY") >= 0.50,
        "OPERATION_COMPOSITION": exact("TRIPLE_ADD") >= 0.50,
        "LENGTH_EXTRAPOLATION": exact("THREE_DIGIT") >= 0.50,
        "SURFACE_TRANSFER": exact("VERBAL") >= 0.50,
    }


def final_decision(*, compact: Mapping[str, Any] | None,
                   production_primary: Mapping[str, Any] | None,
                   production_replication: Mapping[str, Any] | None) -> dict[str, Any]:
    c = bool(compact and compact.get("g90_confirm_update") is not None)
    p1 = bool(production_primary and production_primary.get("g90_confirm_update") is not None)
    p2 = bool(production_replication and production_replication.get("g90_confirm_update") is not None)
    prod_fraction = float((production_primary or {}).get("ark_exposure_fraction", 0.0))
    if p1 and p2:
        verdict = "PRODUCTION_REPRESENTATION_G90_REPLICATED_DEVELOPMENT"
    elif p1:
        verdict = "PRODUCTION_REPRESENTATION_G90_SINGLE_SEED_DEVELOPMENT"
    elif c and not p1:
        verdict = "BRIDGE_DIVERGENCE_COMPACT_G90_PRODUCTION_NO_G90"
    elif not c and not p1:
        verdict = "NO_G90_IN_COMPACT_OR_PRODUCTION_BOX"
    else:
        verdict = "PRODUCTION_G90_WITHOUT_COMPACT_G90"
    return {
        "schema": "anra-cyr-gpu011-decision/v1", "verdict": verdict,
        "compact_g90": c, "production_primary_g90": p1, "production_replication_g90": p2,
        "production_primary_ark_exposure_fraction": prod_fraction,
        "interpretation": (
            "If compact reaches G90 but the production tokenizer does not after substantial exposure, "
            "vocabulary/representation-dependent optimization burden is implicated but not isolated. "
            "If production reaches G90, its structural battery determines what kind of generalization emerged."
        ),
        "broad_reasoning_claim_authorized": False,
        "production_promotion_authorized": False, "pre500m_authorized": False,
        "training_500m_authorized": False,
    }
