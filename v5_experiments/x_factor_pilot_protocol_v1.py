from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

CAMPAIGN = "X-FACTOR-PILOT-001"
VERSION = "x-factor-pilot/v1"
PHYSICAL_VOCAB = 24_576
LATENT_VOCAB = 4_096
WIDTH = 256
LAYERS = 8
QUERY_HEADS = 4
KV_HEADS = 2
HEAD_DIMENSION = 64
FFN_WIDTH = 1_024
CONTEXT_LENGTH = 1_024
BATCH_ROWS = 16
LEARNING_RATE = 1e-3
MAX_GENERATION_TOKENS = 32
OFFICIAL_UPDATES = 2_000
OFFICIAL_EVAL_EVERY = 100
OFFICIAL_CHECKPOINT_EVERY = 200
OFFICIAL_ELIGIBLE_FROM = 600
CONTROL_UPDATES = 400
CONTROL_EVAL_EVERY = 50
CONTROL_CHECKPOINT_EVERY = 100
CONTROL_ELIGIBLE_FROM = 50
SURFACE_SEED = 930_111
MODEL_SEEDS = (930_121, 930_122, 930_123, 930_124)
CONTROL_SEED = 930_099
CALIBRATION_SEED = 930_090
ARMS = ("TIED_DENSE", "UNTIED_DENSE", "LEV_UNTIED")
PRIMARY_ARM = "LEV_UNTIED"
CONTROL_SYMBOL_COUNT = 32
CONTROL_TRAIN_CONTEXTS = 16
CONTROL_DEV_CONTEXTS = 4
CONTROL_MIN_ENDPOINT = 0.50
CONTROL_MIN_AUC = 0.50
AUC_GAP_THRESHOLD = 0.05
ENDPOINT_GAP_THRESHOLD = 0.05
SIGN_CONSISTENCY = 3
STORAGE_RESERVE_BYTES = 2 * 1024 ** 3
MAX_SESSION_SECONDS = 9 * 60 * 60
SESSION_GUARD_RESERVE_SECONDS = 10 * 60
MAX_TRAINABLE_PARAMETERS = 20_454_656
LEV_TRAINABLE_PARAMETERS = 14_210_048
SOURCES = (
    "https://arxiv.org/abs/2601.22040",
    "https://arxiv.org/abs/2605.29459",
    "https://aclanthology.org/2026.findings-acl.2027/",
)
CONTROL_SYMBOLS = tuple(range(1_000, 1_032))
CONTROL_OUTPUTS = tuple(range(2_000, 2_032))
CONTROL_DISTRACTORS = tuple(range(1_100, 1_164))
COPY = 260
QUERY = 269


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def protocol_payload() -> dict[str, Any]:
    return {
        "schema": "anra.x-factor-pilot-protocol/v1",
        "campaign": CAMPAIGN,
        "version": VERSION,
        "physical_vocabulary": PHYSICAL_VOCAB,
        "latent_vocabulary": LATENT_VOCAB,
        "model": {
            "width": WIDTH,
            "layers": LAYERS,
            "query_heads": QUERY_HEADS,
            "kv_heads": KV_HEADS,
            "head_dimension": HEAD_DIMENSION,
            "ffn_width": FFN_WIDTH,
            "context_length": CONTEXT_LENGTH,
            "dropout": 0.0,
            "bias": False,
        },
        "arms": list(ARMS),
        "primary_arm": PRIMARY_ARM,
        "surface_seed": SURFACE_SEED,
        "model_seeds": list(MODEL_SEEDS),
        "control_seed": CONTROL_SEED,
        "batch_rows": BATCH_ROWS,
        "learning_rate": LEARNING_RATE,
        "official": {
            "updates": OFFICIAL_UPDATES,
            "eligible_from": OFFICIAL_ELIGIBLE_FROM,
            "eval_every": OFFICIAL_EVAL_EVERY,
            "checkpoint_every": OFFICIAL_CHECKPOINT_EVERY,
        },
        "positive_control": {
            "updates": CONTROL_UPDATES,
            "eligible_from": CONTROL_ELIGIBLE_FROM,
            "eval_every": CONTROL_EVAL_EVERY,
            "checkpoint_every": CONTROL_CHECKPOINT_EVERY,
            "symbols": CONTROL_SYMBOL_COUNT,
            "train_contexts": CONTROL_TRAIN_CONTEXTS,
            "development_contexts": CONTROL_DEV_CONTEXTS,
            "minimum_endpoint": CONTROL_MIN_ENDPOINT,
            "minimum_auc": CONTROL_MIN_AUC,
        },
        "decision": {
            "auc_gap": AUC_GAP_THRESHOLD,
            "endpoint_gap": ENDPOINT_GAP_THRESHOLD,
            "sign_consistency": SIGN_CONSISTENCY,
            "primary_contrast": [PRIMARY_ARM, "TIED_DENSE"],
        },
        "x_factor": {
            "name": "Leviathan-inspired compositional continuous input",
            "base": 30,
            "components": 3,
            "seed_width": 64,
            "heads": 2,
            "output_head": "independent_dense_v24576",
            "implementation_scope": "factorized continuous generator; not a reproduction of the paper spline head",
        },
        "controls": ["TIED_DENSE", "UNTIED_DENSE"],
        "official_rendering": "R0_PRODUCTION_BPE",
        "max_trainable_parameters": MAX_TRAINABLE_PARAMETERS,
        "lev_trainable_parameters": LEV_TRAINABLE_PARAMETERS,
        "remote_execution": {
            "required_devices": 2,
            "required_name": "T4",
            "local_science_execution": False,
            "sealed_rows_passed_to_workers": False,
            "sealed_access_model": "trusted_worker_processes; sealed rows are coordinator-only by data custody",
            "exact_resume": True,
            "session_guard_reserve_seconds": SESSION_GUARD_RESERVE_SECONDS,
        },
        "sources": list(SOURCES),
    }


def protocol_sha256() -> str:
    return sha256_bytes(canonical(protocol_payload()))


def arm_parameterization(arm: str) -> dict[str, Any]:
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    if arm == "TIED_DENSE":
        return {
            "input": "dense_embedding",
            "output": "tied_dense",
            "input_parameters": PHYSICAL_VOCAB * WIDTH,
            "output_parameters": 0,
            "structured_code": False,
        }
    if arm == "UNTIED_DENSE":
        return {
            "input": "dense_embedding",
            "output": "independent_dense",
            "input_parameters": PHYSICAL_VOCAB * WIDTH,
            "output_parameters": PHYSICAL_VOCAB * WIDTH,
            "structured_code": False,
        }
    return {
        "input": "lev_factorized_continuous",
        "output": "independent_dense",
        "input_parameters": 3 * 30 * 64 + 2 * 64 + 2 * (64 * 64 + 64 * WIDTH),
        "output_parameters": PHYSICAL_VOCAB * WIDTH,
        "structured_code": True,
    }


def _derangement(seed: int, source: tuple[int, ...], target: tuple[int, ...]) -> tuple[int, ...]:
    ordered = tuple(sorted(target, key=lambda value: hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()))
    for shift in range(1, len(ordered)):
        candidate = ordered[shift:] + ordered[:shift]
        if all(left != right for left, right in zip(source, candidate)):
            return candidate
    raise RuntimeError("unable to construct a derangement")


def _control_row(symbol: int, mapped: int, distractor: int, split: str, ordinal: int) -> dict[str, Any]:
    prompt = [COPY, distractor, symbol, QUERY]
    answer = [mapped]
    key = f"{split}:{symbol}:{distractor}"
    return {
        "example_id": hashlib.sha256(key.encode()).hexdigest()[:16],
        "group_id": hashlib.sha256(f"group:{key}".encode()).hexdigest()[:16],
        "family": "identity",
        "split": split,
        "template_id": "control-copy-permutation",
        "prompt_ids": prompt,
        "answer_ids": answer,
    }


def build_positive_control_surface(*, seed: int = CONTROL_SEED) -> dict[str, Any]:
    mapping = _derangement(seed, CONTROL_SYMBOLS, CONTROL_OUTPUTS)
    mapping_by_symbol = dict(zip(CONTROL_SYMBOLS, mapping))
    training: list[dict[str, Any]] = []
    development: list[dict[str, Any]] = []
    for symbol_index, symbol in enumerate(CONTROL_SYMBOLS):
        for context in range(CONTROL_TRAIN_CONTEXTS):
            distractor = CONTROL_DISTRACTORS[(symbol_index * 19 + context * 7) % len(CONTROL_DISTRACTORS)]
            training.append(_control_row(symbol, mapping_by_symbol[symbol], distractor, "training", context))
        for context in range(CONTROL_TRAIN_CONTEXTS, CONTROL_TRAIN_CONTEXTS + CONTROL_DEV_CONTEXTS):
            distractor = CONTROL_DISTRACTORS[(symbol_index * 19 + context * 7) % len(CONTROL_DISTRACTORS)]
            development.append(_control_row(symbol, mapping_by_symbol[symbol], distractor, "development", context))
    training_keys = {tuple(row["prompt_ids"]) + tuple(row["answer_ids"]) for row in training}
    development_keys = {tuple(row["prompt_ids"]) + tuple(row["answer_ids"]) for row in development}
    if training_keys & development_keys:
        raise RuntimeError("positive-control train/development overlap")
    if any(row["answer_ids"][0] == row["prompt_ids"][-2] for row in training + development):
        raise RuntimeError("positive-control last-symbol shortcut")
    body: dict[str, Any] = {
        "schema": "anra.x-factor-positive-control/v1",
        "seed": int(seed),
        "physical_vocabulary": PHYSICAL_VOCAB,
        "symbols": list(CONTROL_SYMBOLS),
        "outputs": list(CONTROL_OUTPUTS),
        "mapping": {str(key): value for key, value in mapping_by_symbol.items()},
        "training": training,
        "development": development,
        "counts": {"training": len(training), "development": len(development)},
        "shortcut_screens": {
            "train_development_overlap": 0.0,
            "last_symbol_answer_rate": 0.0,
            "mapping_is_derangement": all(key != value for key, value in mapping_by_symbol.items()),
        },
    }
    body["sha256"] = sha256_bytes(canonical(body))
    return body


def validate_positive_control_surface(manifest: Mapping[str, Any]) -> None:
    if manifest.get("schema") != "anra.x-factor-positive-control/v1":
        raise RuntimeError("positive-control schema mismatch")
    claimed = manifest.get("sha256")
    body = {key: value for key, value in manifest.items() if key != "sha256"}
    if claimed != sha256_bytes(canonical(body)):
        raise RuntimeError("positive-control manifest hash mismatch")
    if int(manifest.get("seed", -1)) != CONTROL_SEED:
        raise RuntimeError("positive-control seed mismatch")
    expected = build_positive_control_surface(seed=CONTROL_SEED)
    if claimed != expected["sha256"]:
        raise RuntimeError("positive-control manifest is not the registered surface")
    if set(manifest.get("splits", {})) == {"training", "development"}:
        splits = manifest["splits"]
    else:
        splits = {"training": manifest.get("training"), "development": manifest.get("development")}
    if len(splits["training"]) != CONTROL_SYMBOL_COUNT * CONTROL_TRAIN_CONTEXTS:
        raise RuntimeError("positive-control training count mismatch")
    if len(splits["development"]) != CONTROL_SYMBOL_COUNT * CONTROL_DEV_CONTEXTS:
        raise RuntimeError("positive-control development count mismatch")
    if not all(0 <= int(row["prompt_ids"][2]) < PHYSICAL_VOCAB for row in splits["training"] + splits["development"]):
        raise RuntimeError("positive-control symbol outside physical vocabulary")
    if not all(0 <= int(row["answer_ids"][0]) < PHYSICAL_VOCAB for row in splits["training"] + splits["development"]):
        raise RuntimeError("positive-control answer outside physical vocabulary")
    train_keys = {tuple(row["prompt_ids"]) + tuple(row["answer_ids"]) for row in splits["training"]}
    dev_keys = {tuple(row["prompt_ids"]) + tuple(row["answer_ids"]) for row in splits["development"]}
    if train_keys & dev_keys:
        raise RuntimeError("positive-control split overlap")


def positive_control_gate(results: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    complete_results = [results[arm] for arm in ARMS if arm in results and results[arm].get("status") == "COMPLETE"]
    matched_exposure = len(complete_results) == len(ARMS) and len({int(result.get("processed_tokens", -1)) for result in complete_results}) == 1 and len({int(result.get("supervised_tokens", -1)) for result in complete_results}) == 1 and len({str(result.get("data_order_sha256", "")) for result in complete_results}) == 1
    passed = matched_exposure
    for arm in ARMS:
        result = results.get(arm)
        endpoint = None if result is None else float(result.get("formation", {}).get("endpoint", 0.0))
        auc = None if result is None else float(result.get("formation", {}).get("formation_auc", 0.0))
        arm_pass = result is not None and result.get("status") == "COMPLETE" and endpoint >= CONTROL_MIN_ENDPOINT and auc >= CONTROL_MIN_AUC
        checks[arm] = {"status": "PASS" if arm_pass else "FAIL", "endpoint": endpoint, "auc": auc}
        passed = passed and arm_pass
    return {"status": "PASS" if passed else "FAIL", "matched_exposure": matched_exposure, "checks": checks}


def paired_verdict(*, formation_auc_deltas: Mapping[int, float], sealed_endpoint_gaps: Mapping[int, float]) -> dict[str, Any]:
    seeds = list(MODEL_SEEDS)
    if any(seed not in formation_auc_deltas or seed not in sealed_endpoint_gaps for seed in seeds):
        return {"verdict": "INCONCLUSIVE", "reason": "missing matched seed"}
    auc = [float(formation_auc_deltas[seed]) for seed in seeds]
    endpoint = [float(sealed_endpoint_gaps[seed]) for seed in seeds]
    mean_auc = sum(auc) / len(auc)
    mean_endpoint = sum(endpoint) / len(endpoint)
    positive_auc = sum(value > 0 for value in auc)
    negative_auc = sum(value < 0 for value in auc)
    positive_endpoint = sum(value > 0 for value in endpoint)
    negative_endpoint = sum(value < 0 for value in endpoint)
    sign = 1 if positive_auc >= SIGN_CONSISTENCY else (-1 if negative_auc >= SIGN_CONSISTENCY else 0)
    endpoint_consistent = (positive_endpoint >= SIGN_CONSISTENCY) if sign > 0 else ((negative_endpoint >= SIGN_CONSISTENCY) if sign < 0 else False)
    if sign != 0 and abs(mean_auc) >= AUC_GAP_THRESHOLD and abs(mean_endpoint) >= ENDPOINT_GAP_THRESHOLD and endpoint_consistent:
        verdict = "SUCCESS" if sign > 0 else "REVERSE_EFFECT"
    elif abs(mean_auc) < AUC_GAP_THRESHOLD and abs(mean_endpoint) < ENDPOINT_GAP_THRESHOLD:
        verdict = "NULL"
    else:
        verdict = "PARTIAL_OR_INTERACTION"
    return {
        "verdict": verdict,
        "mean_formation_auc_delta": round(mean_auc, 6),
        "mean_sealed_endpoint_delta": round(mean_endpoint, 6),
        "formation_auc_deltas": {str(seed): value for seed, value in zip(seeds, auc)},
        "sealed_endpoint_deltas": {str(seed): value for seed, value in zip(seeds, endpoint)},
        "positive_auc_signs": positive_auc,
        "positive_endpoint_signs": positive_endpoint,
        "thresholds": {"auc": AUC_GAP_THRESHOLD, "endpoint": ENDPOINT_GAP_THRESHOLD, "signs": SIGN_CONSISTENCY},
    }


def aggregate_verdicts(verdicts: Mapping[str, str]) -> str:
    values = list(verdicts.values())
    if not values or any(value in {"INCONCLUSIVE", "FAILED", "INCONCLUSIVE_POSITIVE_CONTROL"} for value in values):
        return "INCONCLUSIVE"
    if all(value == "NULL" for value in values):
        return "NULL"
    if "SUCCESS" in values and all(value in {"SUCCESS", "NULL"} for value in values):
        return "SUCCESS"
    if "REVERSE_EFFECT" in values:
        return "REVERSE_OR_INTERACTION"
    return "PARTIAL_OR_INTERACTION"


def queue_assignment() -> dict[str, list[dict[str, Any]]]:
    queues = {"GPU0": [], "GPU1": []}
    for seed_index, seed in enumerate(MODEL_SEEDS):
        gpu = f"GPU{seed_index % 2}"
        for arm in ARMS:
            queues[gpu].append({"arm": arm, "seed": seed, "seed_label": f"S{seed_index + 1}", "gpu": gpu})
    return queues


def seed_label(seed: int) -> str:
    if seed in MODEL_SEEDS:
        return f"S{MODEL_SEEDS.index(seed) + 1}"
    if seed == CONTROL_SEED:
        return "CONTROL"
    return f"CAL{seed}"


def load_positive_control_surface(path: str | Path) -> dict[str, Any]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_positive_control_surface(manifest)
    return manifest
