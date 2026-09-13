"""CS-TRANSFER-001 physical class-space causal experiment.

Operator surface:

    python -m anra_v5.cs_transfer_001_run --mode prepare
    python -m anra_v5.cs_transfer_001_run --mode preflight --cuda
    python -m anra_v5.cs_transfer_001_run --mode scan
    python -m anra_v5.cs_transfer_001_run --mode run-arm --pair-index 0 --arm PHYS_4096 --cuda
    python -m anra_v5.cs_transfer_001_run --mode finalize --cuda

No command changes the frozen endpoint or the physical-vocabulary treatment.
Every scientific arm uses the V5 production pack/stream/backend/optimizer/
checkpoint transaction.  Sealed rows are consumed only by finalization after
all eight arms have durable endpoint checkpoints and a frozen development
aggregate.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from anra_v5 import v51_canary_run as base
from anra_v5.cs_transfer_001_data import (
    COMMON_VOCAB,
    SELECT_COUNTS,
    TokenRow,
    build_shared_surface,
    deserialize_rows,
    serialize_rows,
)
from anra_v5.cs_transfer_001_model import build_matched_pair, select_arm
from v5_model.core import packed_layout
from v5_training.checkpoint import CheckpointStore

REPO = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get(
    "CS_TRANSFER_001_ROOT",
    str(REPO / "experiments" / "CS_TRANSFER_001" / "runtime"),
))
PREREG_PATH = REPO / "experiments" / "CS_TRANSFER_001" / "PREREGISTRATION.json"
DATA_DIR = ROOT / "data"
RECEIPTS = ROOT / "receipts"
RUNS = ROOT / "runs"
SEALED_LOCK = ROOT / "SEALED_CONSUMPTION.json"
DEVELOPMENT_AGGREGATE = RECEIPTS / "DEVELOPMENT_AGGREGATE.json"
FINAL_RESULT = RECEIPTS / "FINAL_RESULT.json"
ARMS = ("PHYS_4096", "PHYS_24576")


def load_prereg() -> dict[str, Any]:
    p = json.loads(PREREG_PATH.read_text(encoding="utf-8"))
    if p.get("schema") != "anra-cs-transfer-001-preregistration/v1":
        raise SystemExit("FAIL_CLOSED: wrong CS-TRANSFER-001 preregistration")
    return p


def source_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def _canonical_sha(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode()).hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_receipt(name: str, payload: dict[str, Any], *, arm_dir: Path | None = None) -> str:
    root = (arm_dir / "receipts") if arm_dir is not None else RECEIPTS
    body = {**payload, "receipt_name": name}
    body["receipt_sha256"] = _canonical_sha(body)
    _atomic_json(root / f"{name}.json", body)
    return body["receipt_sha256"]


def _prereg_sha() -> str:
    return hashlib.sha256(PREREG_PATH.read_bytes()).hexdigest()


def prepare_data() -> dict[str, Any]:
    p = load_prereg()
    tokenizer, tokenizer_receipt = base.load_tokenizer()
    surface = build_shared_surface(
        tokenizer=tokenizer,
        seed=int(p["data"]["fresh_seed"]),
        candidate_worlds_per_family=int(p["data"]["candidate_worlds_per_family"]),
        select_counts={k: int(v) for k, v in p["data"]["selected_rows_per_family"].items()},
    )
    ROOT.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for split, rows in surface["rows"].items():
        (DATA_DIR / f"{split}.jsonl").write_text(serialize_rows(rows), encoding="utf-8")

    documents = [
        (r.example_id, list(r.prompt_ids + r.answer_ids), r.family)
        for r in surface["rows"]["training"]
    ]
    shards, audit = base.pack_documents(
        documents, bos=2, eos=3, pad=0, sequences_per_shard=64,
    )
    shard_hashes = [s.sha256() for s in shards]
    pack_manifest = {
        "schema": "anra-cs-transfer-001-pack/v1",
        "shared_surface_sha256": surface["manifest_sha256"],
        "shard_hashes": shard_hashes,
        "audit": audit,
        "tokenizer_artifact_sha256": tokenizer_receipt["artifact_sha256"],
        "maximum_training_content_id": max(
            token for r in surface["rows"]["training"] for token in r.content_ids
        ),
    }
    pack_sha = _canonical_sha(pack_manifest)
    receipt = {
        "schema": "anra-cs-transfer-001-data-receipt/v1",
        "source_commit": source_commit(),
        "preregistration_sha256": _prereg_sha(),
        "tokenizer_identity": tokenizer_receipt,
        "shared_surface": surface["manifest"],
        "shared_surface_sha256": surface["manifest_sha256"],
        "pack_manifest": pack_manifest,
        "pack_manifest_sha256": pack_sha,
        "identical_token_sequences_required": True,
        "maximum_allowed_content_id": COMMON_VOCAB - 1,
    }
    _write_receipt("DATA", receipt)
    return {"surface": surface, "shards": shards, "pack_manifest": pack_manifest,
            "pack_manifest_sha256": pack_sha, "tokenizer": tokenizer,
            "tokenizer_receipt": tokenizer_receipt, "receipt": receipt}


def _load_rows(split: str) -> list[TokenRow]:
    path = DATA_DIR / f"{split}.jsonl"
    if not path.is_file():
        raise SystemExit(f"FAIL_CLOSED DATA: missing {path}")
    return deserialize_rows(path.read_text(encoding="utf-8"))


def load_prepared() -> dict[str, Any]:
    receipt_path = RECEIPTS / "DATA.json"
    if not receipt_path.is_file():
        raise SystemExit("FAIL_CLOSED DATA: run prepare first")
    old = _read_json(receipt_path)
    p = load_prereg()
    if old["preregistration_sha256"] != _prereg_sha():
        raise SystemExit("FAIL_CLOSED DATA: preregistration changed after data preparation")
    tokenizer, tokenizer_receipt = base.load_tokenizer()
    if tokenizer_receipt["artifact_sha256"] != old["tokenizer_identity"]["artifact_sha256"]:
        raise SystemExit("FAIL_CLOSED DATA: tokenizer identity drift")
    rows = {s: _load_rows(s) for s in ("training", "development", "sealed")}
    # Regenerate deterministically and require exact selected-row bytes.  This
    # makes the Drive JSONL a cache, not a hidden source of experiment state.
    regenerated = build_shared_surface(
        tokenizer=tokenizer,
        seed=int(p["data"]["fresh_seed"]),
        candidate_worlds_per_family=int(p["data"]["candidate_worlds_per_family"]),
        select_counts={k: int(v) for k, v in p["data"]["selected_rows_per_family"].items()},
    )
    for split in rows:
        if serialize_rows(rows[split]) != serialize_rows(regenerated["rows"][split]):
            raise SystemExit(f"FAIL_CLOSED DATA: persisted {split} rows differ from regeneration")
    documents = [
        (r.example_id, list(r.prompt_ids + r.answer_ids), r.family)
        for r in rows["training"]
    ]
    shards, audit = base.pack_documents(
        documents, bos=2, eos=3, pad=0, sequences_per_shard=64,
    )
    pack_manifest = {
        "schema": "anra-cs-transfer-001-pack/v1",
        "shared_surface_sha256": regenerated["manifest_sha256"],
        "shard_hashes": [s.sha256() for s in shards],
        "audit": audit,
        "tokenizer_artifact_sha256": tokenizer_receipt["artifact_sha256"],
        "maximum_training_content_id": max(
            token for r in rows["training"] for token in r.content_ids
        ),
    }
    pack_sha = _canonical_sha(pack_manifest)
    if pack_sha != old["pack_manifest_sha256"]:
        raise SystemExit("FAIL_CLOSED DATA: regenerated pack identity differs")
    return {"rows": rows, "shards": shards, "pack_manifest": pack_manifest,
            "pack_manifest_sha256": pack_sha, "tokenizer": tokenizer,
            "tokenizer_receipt": tokenizer_receipt, "receipt": old}


def _arm_dir(pair_index: int, arm: str) -> Path:
    return RUNS / f"pair_{pair_index}" / arm


def _lineage(pair_index: int, arm: str) -> str:
    return f"cs-transfer-001-p{pair_index}-{arm.lower()}"


def _matched_model(*, pair_index: int, arm: str, device=None):
    import torch
    p = load_prereg()
    seeds = p["matching"]["model_seeds"]
    if not 0 <= pair_index < len(seeds):
        raise SystemExit("FAIL_CLOSED: pair-index out of range")
    pair = build_matched_pair(seed=int(seeds[pair_index]), torch_module=torch)
    small, full, init_receipt = pair
    model, _ = select_arm(pair, arm)
    other = full if arm == "PHYS_4096" else small
    del other
    gc.collect()
    if device is not None:
        model = model.to(device)
    optimizer = base.build_adamw_optimizer(model)
    base.validate_parameter_ownership(model, optimizer)
    return model, optimizer, init_receipt


def _schedule(p: dict[str, Any]):
    plan = base.canary_wsd_receipt(token_budget=int(p["training"]["token_budget"]))
    return plan, base.canary_lr_at(plan)


def _make_backend(*, pair_index: int, arm: str, device=None):
    from v5_training.production_backend import ProductionTrainingBackend
    p = load_prereg()
    model, optimizer, init_receipt = _matched_model(
        pair_index=pair_index, arm=arm, device=device,
    )
    plan, schedule = _schedule(p)
    backend = ProductionTrainingBackend(
        model=model, optimizer=optimizer, bos_id=2, pad_id=0,
        device=device, bfloat16_autocast=False, schedule=schedule,
    )
    expected = int(p["model"]["arms"][arm]["expected_parameters"])
    actual = sum(int(x.numel()) for x in model.parameters())
    if actual != expected:
        raise SystemExit(f"FAIL_CLOSED MODEL: {arm} parameters {actual} != {expected}")
    return backend, init_receipt, plan


def _stream_layout(prepared: dict[str, Any], *, order_seed: int, target: int) -> tuple[int, list[list[Any]]]:
    cache: list[list[Any]] = []
    epoch = 0
    total = 0
    per_epoch = None
    while total < target:
        windows = base.build_update_stream(
            prepared["shards"], run_seed=order_seed, epoch=epoch,
            real_tokens_per_update=4096,
        )
        if not windows:
            raise SystemExit("FAIL_CLOSED DATA: no complete 4096-token update windows")
        if per_epoch is None:
            per_epoch = len(windows)
        elif len(windows) != per_epoch:
            raise SystemExit("FAIL_CLOSED DATA: sampler epoch cardinality drift")
        cache.append(windows)
        total += len(windows)
        epoch += 1
    return int(per_epoch), cache


def _evaluate(backend, rows: list[TokenRow]) -> dict[str, Any]:
    import torch
    device = next(backend.model.parameters()).device
    eos = 3
    family: dict[str, dict[str, int]] = {}
    full_extra_mass_sum = 0.0
    full_extra_mass_n = 0
    with torch.no_grad():
        for row in rows:
            ids = torch.tensor([[2, *row.prompt_ids]], dtype=torch.long, device=device)
            segments = torch.zeros(1, ids.shape[1], dtype=torch.int32, device=device)
            generated: list[int] = []
            stopped = False
            for step in range(26):
                positions, mask = packed_layout(segments, torch_module=torch)
                logits = backend.model(ids, positions, mask)
                last = logits[0, -1]
                if step == 0 and last.shape[0] > COMMON_VOCAB:
                    probs = torch.softmax(last.float(), dim=-1)
                    full_extra_mass_sum += float(probs[COMMON_VOCAB:].sum().item())
                    full_extra_mass_n += 1
                nxt = int(torch.argmax(last).item())
                if nxt == eos:
                    stopped = True
                    break
                generated.append(nxt)
                ids = torch.cat([ids, torch.tensor([[nxt]], dtype=torch.long, device=device)], dim=1)
                segments = torch.cat([
                    segments, torch.zeros(1, 1, dtype=torch.int32, device=device)
                ], dim=1)
            expected = list(row.answer_ids)
            exact_answer = int(generated == expected)
            exact_eos = int(exact_answer and stopped)
            prefix = int(generated[:len(expected)] == expected)
            scores = family.setdefault(row.family, {
                "n": 0, "exact_eos": 0, "answer_exact": 0,
                "prefix_exact": 0, "eos_stop": 0,
            })
            scores["n"] += 1
            scores["exact_eos"] += exact_eos
            scores["answer_exact"] += exact_answer
            scores["prefix_exact"] += prefix
            scores["eos_stop"] += int(stopped)
    out: dict[str, Any] = {}
    for name, s in sorted(family.items()):
        n = s["n"]
        out[name] = {
            "n": n,
            "exact_token_with_valid_eos": s["exact_eos"] / n,
            "answer_token_exact_ignoring_eos": s["answer_exact"] / n,
            "answer_prefix_token_exact": s["prefix_exact"] / n,
            "eos_stop_rate": s["eos_stop"] / n,
        }
    total = sum(v["n"] for v in family.values())
    overall = sum(v["exact_eos"] for v in family.values()) / total
    return {
        "per_family": out,
        "overall_exact_token_with_valid_eos": overall,
        "mean_first_answer_step_probability_mass_on_ids_ge_4096": (
            full_extra_mass_sum / full_extra_mass_n if full_extra_mass_n else 0.0
        ),
    }


def _state_and_store(*, pair_index: int, arm: str, prepared: dict[str, Any], backend, plan):
    import torch
    p = load_prereg()
    arm_dir = _arm_dir(pair_index, arm)
    store = CheckpointStore(arm_dir / "state", _lineage(pair_index, arm))
    data_sha = _canonical_sha(prepared["receipt"])
    run_spec = {
        "preregistration_sha256": _prereg_sha(), "pair_index": pair_index,
        "arm": arm, "model_seed": p["matching"]["model_seeds"][pair_index],
        "order_seed": p["matching"]["order_seeds"][pair_index],
    }
    state = base.initial_state(
        lineage_id=_lineage(pair_index, arm),
        pack_manifest_sha256=prepared["pack_manifest_sha256"],
        token_budget=int(p["training"]["token_budget"]), tokens_per_update=4096,
        identities=base.identity_bindings(
            pack_manifest_sha256=prepared["pack_manifest_sha256"],
            model_spec_sha=backend.model.spec.sha256(),
            tokenizer_artifact_sha=prepared["tokenizer_receipt"]["artifact_sha256"],
            data_receipt_sha=data_sha, canary_config_sha=_canonical_sha(run_spec),
            wsd_sha=plan["sha256"],
        ),
        rng_state_sha256=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
    )
    return state, store, arm_dir, run_spec


def _latest_trace(arm_dir: Path) -> list[dict[str, Any]]:
    path = arm_dir / "receipts" / "TRAINING.json"
    if not path.is_file():
        return []
    return list(_read_json(path).get("trace", []))


def _latest_development(arm_dir: Path) -> list[dict[str, Any]]:
    path = arm_dir / "receipts" / "DEVELOPMENT_TRACE.json"
    if not path.is_file():
        return []
    return list(_read_json(path).get("evaluations", []))


def _persist_training(arm_dir: Path, trace: list[dict[str, Any]], status: str) -> None:
    _write_receipt("TRAINING", {
        "schema": "anra-cs-transfer-001-training/v1",
        "status": status, "trace": trace,
    }, arm_dir=arm_dir)


def _persist_dev(arm_dir: Path, values: list[dict[str, Any]]) -> None:
    _write_receipt("DEVELOPMENT_TRACE", {
        "schema": "anra-cs-transfer-001-development-trace/v1",
        "evaluations": values,
    }, arm_dir=arm_dir)


def run_arm(*, pair_index: int, arm: str, cuda: bool) -> dict[str, Any]:
    import torch
    if arm not in ARMS:
        raise SystemExit(f"FAIL_CLOSED: arm must be one of {ARMS}")
    if cuda and not torch.cuda.is_available():
        raise SystemExit("FAIL_CLOSED HARDWARE: CUDA requested but unavailable")
    p = load_prereg()
    prepared = load_prepared()
    device = torch.device("cuda") if cuda else None
    backend, init_receipt, plan = _make_backend(pair_index=pair_index, arm=arm, device=device)
    state, store, arm_dir, run_spec = _state_and_store(
        pair_index=pair_index, arm=arm, prepared=prepared, backend=backend, plan=plan,
    )
    latest = store.latest_sha256()
    if latest is not None:
        restored, payloads = store.restore(latest)
        if restored.identities != state.identities:
            raise SystemExit("FAIL_CLOSED RESUME: checkpoint identity drift")
        base.restore_production(backend, payloads=payloads)
        state = restored
    elif _latest_trace(arm_dir):
        raise SystemExit("FAIL_CLOSED RESUME: trace exists without checkpoint")

    target = int(p["training"]["target_updates"])
    checkpoint_every = int(p["training"]["checkpoint_every_updates"])
    eval_updates = set(int(x) for x in p["training"]["development_eval_updates"])
    order_seed = int(p["matching"]["order_seeds"][pair_index])
    per_epoch, epochs = _stream_layout(prepared, order_seed=order_seed, target=target)
    trace = [r for r in _latest_trace(arm_dir) if int(r["update"]) <= int(state.global_update)]
    if [int(r["update"]) for r in trace] != list(range(1, int(state.global_update) + 1)):
        if state.global_update:
            raise SystemExit("FAIL_CLOSED RESUME: durable trace does not cover checkpoint")
        trace = []
    dev_trace = [x for x in _latest_development(arm_dir)
                 if int(x["update"]) <= int(state.global_update)]

    # Update-zero development evaluation is mandatory and outcome-blind.
    if state.global_update == 0 and not any(int(x["update"]) == 0 for x in dev_trace):
        dev_trace.append({"update": 0, "metrics": _evaluate(backend, prepared["rows"]["development"])})
        _persist_dev(arm_dir, dev_trace)

    parent_sha = latest
    t0 = time.time()
    for update_index in range(int(state.global_update), target):
        epoch, position = divmod(update_index, per_epoch)
        window = epochs[epoch][position]
        batch = base.batch_from_window(
            window, pack_manifest_sha256=prepared["pack_manifest_sha256"],
            update_ordinal=update_index, device=device,
        )
        pre_tokens = int(state.schedule_tokens)
        report = backend.step(state, batch)
        state = state.advance(
            tokens_by_source=report.tokens_by_source, cursor=report.cursor,
            rng_state_sha256=report.rng_state_sha256,
            parent_checkpoint_sha256=parent_sha,
        )
        base.certify_update(
            before=state.__class__.from_dict({**state.canonical(),
                "generation": state.generation - 1,
                "global_update": state.global_update - 1,
                "optimizer_step_min": state.optimizer_step_min - 1,
                "optimizer_step_max": state.optimizer_step_max - 1,
                "cumulative_tokens": state.cumulative_tokens - 4096,
                "schedule_tokens": state.schedule_tokens - 4096,
            }),
            after=state, tokens_by_source=report.tokens_by_source,
            loss_finite=report.loss_finite, grad_finite=report.grad_finite,
            grad_norm_post_clip=report.grad_norm_post_clip,
            tied_preserved=report.tied_preserved,
        )
        expected_lr = float(base.canary_lr_at(plan)(cumulative_tokens=pre_tokens))
        actual_lr = float(backend.optimizer.param_groups[0]["lr"])
        if expected_lr != actual_lr:
            raise SystemExit("FAIL_CLOSED SCHEDULE: actual LR differs from frozen WSD")
        trace.append({
            "update": int(state.global_update), "tokens_seen": int(state.cumulative_tokens),
            "epoch": int(epoch), "epoch_update": int(position + 1),
            "loss": float(backend.last_receipt["loss"]),
            "grad_norm_post_clip": float(report.grad_norm_post_clip),
            "lr_expected": expected_lr, "lr_actual": actual_lr,
        })

        due_checkpoint = state.global_update % checkpoint_every == 0
        due_eval = state.global_update in eval_updates
        if due_checkpoint:
            _persist_training(arm_dir, trace, "IN_PROGRESS")
            published = store.publish(
                state=state, payloads=base.production_payloads(backend, state=state),
                expected_parent_sha256=parent_sha,
            )
            parent_sha = published
            trace[-1]["checkpoint_sha256"] = published
        if due_eval:
            # All preregistered dev checkpoints are checkpoint boundaries.
            if not due_checkpoint:
                raise SystemExit("FAIL_CLOSED: development evaluation lacks durable checkpoint")
            if not any(int(x["update"]) == int(state.global_update) for x in dev_trace):
                dev_trace.append({
                    "update": int(state.global_update),
                    "checkpoint_sha256": parent_sha,
                    "metrics": _evaluate(backend, prepared["rows"]["development"]),
                })
                _persist_dev(arm_dir, dev_trace)

    _persist_training(arm_dir, trace, "COMPLETE_ENDPOINT")
    _persist_dev(arm_dir, dev_trace)
    _write_receipt("ARM_RESULT", {
        "schema": "anra-cs-transfer-001-arm-result/v1",
        "pair_index": pair_index, "arm": arm, "run_spec": run_spec,
        "matched_init": init_receipt,
        "global_update": int(state.global_update),
        "cumulative_tokens": int(state.cumulative_tokens),
        "checkpoint_sha256": store.latest_sha256(),
        "development_trace_updates": [int(x["update"]) for x in dev_trace],
        "wall_seconds_this_invocation": round(time.time() - t0, 3),
        "status": "COMPLETE",
    }, arm_dir=arm_dir)
    return {"pair_index": pair_index, "arm": arm, "status": "COMPLETE",
            "global_update": state.global_update, "checkpoint": store.latest_sha256()}


def _auc(dev_trace: list[dict[str, Any]], family: str) -> float:
    points = []
    for item in sorted(dev_trace, key=lambda x: int(x["update"])):
        u = int(item["update"])
        y = float(item["metrics"]["per_family"][family]["exact_token_with_valid_eos"])
        points.append((u, y))
    if not points or points[0][0] != 0 or points[-1][0] != 480:
        raise SystemExit("FAIL_CLOSED ANALYSIS: development trace does not span 0..480")
    area = 0.0
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        area += (x1 - x0) * (y0 + y1) / 2.0
    return area / 480.0


def development_aggregate() -> dict[str, Any]:
    p = load_prereg()
    pairs = []
    for i in range(len(p["matching"]["model_seeds"])):
        arms: dict[str, Any] = {}
        for arm in ARMS:
            arm_dir = _arm_dir(i, arm)
            result_path = arm_dir / "receipts" / "ARM_RESULT.json"
            if not result_path.is_file() or _read_json(result_path).get("status") != "COMPLETE":
                raise SystemExit(f"FAIL_CLOSED: pair {i} {arm} is not complete")
            dev = _latest_development(arm_dir)
            endpoint = next(x for x in dev if int(x["update"]) == 480)
            arms[arm] = {
                "identity_auc": _auc(dev, "identity"),
                "identity_endpoint": endpoint["metrics"]["per_family"]["identity"]["exact_token_with_valid_eos"],
                "composition_endpoint": endpoint["metrics"]["per_family"]["composition"]["exact_token_with_valid_eos"],
                "termination_endpoint": endpoint["metrics"]["per_family"]["termination"]["exact_token_with_valid_eos"],
                "overall_endpoint": endpoint["metrics"]["overall_exact_token_with_valid_eos"],
                "dev_trace": dev,
            }
        gap_auc = float(arms["PHYS_4096"]["identity_auc"] - arms["PHYS_24576"]["identity_auc"])
        gap_end = float(arms["PHYS_4096"]["identity_endpoint"] - arms["PHYS_24576"]["identity_endpoint"])
        pairs.append({"pair_index": i, "arms": arms,
                      "paired_identity_auc_gap": gap_auc,
                      "paired_identity_endpoint_gap": gap_end})
    mean_auc = sum(x["paired_identity_auc_gap"] for x in pairs) / len(pairs)
    mean_end = sum(x["paired_identity_endpoint_gap"] for x in pairs) / len(pairs)
    positive = sum(x["paired_identity_auc_gap"] > 0 for x in pairs)
    near = sum(abs(x["paired_identity_auc_gap"]) <= 0.10 for x in pairs)
    control_ok = all(
        max(float(x["arms"][arm]["composition_endpoint"]),
            float(x["arms"][arm]["termination_endpoint"])) >= 0.50
        for x in pairs for arm in ARMS
    )
    if not control_ok:
        verdict = "INCONCLUSIVE_FORMATION"
    elif mean_auc >= 0.10 and positive >= 3 and mean_end >= 0.10:
        verdict = "SUPPORTED_PHYSICAL_CLASS_SPACE"
    elif mean_auc <= -0.10 and positive <= 1:
        verdict = "REVERSE_EFFECT"
    elif abs(mean_auc) < 0.05 and abs(mean_end) < 0.05 and near >= 3:
        verdict = "NULL_AT_THIS_SCALE"
    else:
        verdict = "PARTIAL_OR_INTERACTION"
    payload = {
        "schema": "anra-cs-transfer-001-development-aggregate/v1",
        "pairs": pairs, "mean_paired_identity_auc_gap": mean_auc,
        "mean_paired_identity_endpoint_gap": mean_end,
        "positive_auc_pairs": positive, "pairs_within_abs_0p10_auc": near,
        "control_formation_floor_pass": control_ok,
        "development_verdict": verdict,
        "sealed_unseen": True,
    }
    _atomic_json(DEVELOPMENT_AGGREGATE, payload)
    return payload


def _restore_endpoint(pair_index: int, arm: str, device):
    p = load_prereg()
    prepared = load_prepared()
    backend, init_receipt, plan = _make_backend(pair_index=pair_index, arm=arm, device=device)
    state, store, _, _ = _state_and_store(
        pair_index=pair_index, arm=arm, prepared=prepared, backend=backend, plan=plan,
    )
    latest = store.latest_sha256()
    if latest is None:
        raise SystemExit("FAIL_CLOSED FINALIZE: missing endpoint checkpoint")
    restored, payloads = store.restore(latest)
    if restored.global_update != int(p["training"]["target_updates"]):
        raise SystemExit("FAIL_CLOSED FINALIZE: endpoint checkpoint not at update 480")
    if restored.identities != state.identities:
        raise SystemExit("FAIL_CLOSED FINALIZE: checkpoint identity drift")
    base.restore_production(backend, payloads=payloads)
    return backend, latest


def finalize(*, cuda: bool) -> dict[str, Any]:
    import torch
    if FINAL_RESULT.exists():
        return _read_json(FINAL_RESULT)
    if SEALED_LOCK.exists():
        raise SystemExit(
            "FAIL_CLOSED SEALED: consumption marker exists without final result; do not look again"
        )
    dev = development_aggregate()
    prepared = load_prepared()
    marker = {
        "schema": "anra-cs-transfer-001-sealed-consumption/v1",
        "status": "STARTED", "development_aggregate_sha256": _canonical_sha(dev),
        "sealed_split_sha256": prepared["receipt"]["shared_surface"]["split_hashes"]["sealed"],
        "started_at_unix": time.time(),
    }
    _atomic_json(SEALED_LOCK, marker)
    device = torch.device("cuda") if cuda and torch.cuda.is_available() else None
    sealed: dict[str, Any] = {}
    p = load_prereg()
    for i in range(len(p["matching"]["model_seeds"])):
        sealed[str(i)] = {}
        for arm in ARMS:
            backend, checkpoint = _restore_endpoint(i, arm, device)
            sealed[str(i)][arm] = {
                "checkpoint_sha256": checkpoint,
                "metrics": _evaluate(backend, prepared["rows"]["sealed"]),
            }
            del backend
            gc.collect()
            if device is not None:
                torch.cuda.empty_cache()
    sealed_gaps = [
        sealed[str(i)]["PHYS_4096"]["metrics"]["per_family"]["identity"]["exact_token_with_valid_eos"]
        - sealed[str(i)]["PHYS_24576"]["metrics"]["per_family"]["identity"]["exact_token_with_valid_eos"]
        for i in range(len(p["matching"]["model_seeds"]))
    ]
    mean_sealed_gap = sum(sealed_gaps) / len(sealed_gaps)
    verdict = dev["development_verdict"]
    if (verdict == "SUPPORTED_PHYSICAL_CLASS_SPACE"
            and dev["mean_paired_identity_auc_gap"] >= 0.20
            and mean_sealed_gap >= 0.10):
        verdict = "STRONG_PHYSICAL_CLASS_SPACE"
    final = {
        "schema": "anra-cs-transfer-001-final-result/v1",
        "source_commit": source_commit(),
        "preregistration_sha256": _prereg_sha(),
        "development": dev,
        "sealed": sealed,
        "sealed_identity_endpoint_gaps": sealed_gaps,
        "mean_sealed_identity_endpoint_gap": mean_sealed_gap,
        "verdict": verdict,
        "claim_ceiling": p["claim_ceiling"],
        "next_action": p["post_result_actions"].get(verdict, p["post_result_actions"].get(dev["development_verdict"])),
    }
    _atomic_json(FINAL_RESULT, final)
    marker["status"] = "CONSUMED_AND_FINALIZED"
    marker["final_result_sha256"] = _canonical_sha(final)
    _atomic_json(SEALED_LOCK, marker)
    return final


def scan() -> dict[str, Any]:
    p = load_prereg()
    rows = []
    for i in range(len(p["matching"]["model_seeds"])):
        for arm in ARMS:
            arm_dir = _arm_dir(i, arm)
            result = arm_dir / "receipts" / "ARM_RESULT.json"
            status = "COMPLETE" if result.is_file() and _read_json(result).get("status") == "COMPLETE" else "PENDING"
            latest = None
            state_update = 0
            state_root = arm_dir / "state"
            store = CheckpointStore(state_root, _lineage(i, arm))
            try:
                latest = store.latest_sha256()
                if latest:
                    state, _ = store.restore(latest)
                    state_update = int(state.global_update)
                    if status != "COMPLETE":
                        status = "RESUME"
            except Exception as exc:
                status = f"FAIL_CLOSED:{type(exc).__name__}:{exc}"
            rows.append({"pair_index": i, "arm": arm, "status": status,
                         "checkpoint": latest, "global_update": state_update})
    if FINAL_RESULT.exists():
        action = "COMPLETE"
    elif any(str(r["status"]).startswith("FAIL_CLOSED") for r in rows):
        action = "FAIL_CLOSED"
    elif all(r["status"] == "COMPLETE" for r in rows):
        action = "FINALIZE"
    elif any(r["status"] == "RESUME" for r in rows):
        action = "RESUME"
    else:
        action = "START"
    return {"schema": "anra-cs-transfer-001-scan/v1", "action": action, "arms": rows}


def preflight(*, cuda: bool) -> dict[str, Any]:
    import torch
    if cuda and not torch.cuda.is_available():
        raise SystemExit("FAIL_CLOSED HARDWARE: CUDA unavailable")
    prepared = load_prepared()
    device = torch.device("cuda") if cuda else None
    result: dict[str, Any] = {"schema": "anra-cs-transfer-001-preflight/v1", "arms": {}}
    # Temporary checkpoint roots ensure smoke updates can never become scientific state.
    for arm in ARMS:
        backend, init_receipt, plan = _make_backend(pair_index=0, arm=arm, device=device)
        windows = base.build_update_stream(
            prepared["shards"], run_seed=7811, epoch=0, real_tokens_per_update=4096,
        )
        if not windows:
            raise SystemExit("FAIL_CLOSED PREFLIGHT: no update window")
        with tempfile.TemporaryDirectory(prefix="cs-transfer-preflight-") as tmp:
            state = base.initial_state(
                lineage_id=f"preflight-{arm.lower()}",
                pack_manifest_sha256=prepared["pack_manifest_sha256"],
                token_budget=4096, tokens_per_update=4096,
                identities=base.identity_bindings(
                    pack_manifest_sha256=prepared["pack_manifest_sha256"],
                    model_spec_sha=backend.model.spec.sha256(),
                    tokenizer_artifact_sha=prepared["tokenizer_receipt"]["artifact_sha256"],
                    data_receipt_sha=_canonical_sha(prepared["receipt"]),
                    canary_config_sha=_canonical_sha({"preflight": arm}),
                    wsd_sha=plan["sha256"],
                ), rng_state_sha256="0" * 64,
            )
            batch = base.batch_from_window(
                windows[0], pack_manifest_sha256=prepared["pack_manifest_sha256"],
                update_ordinal=0, device=device,
            )
            report = backend.step(state, batch)
            result["arms"][arm] = {
                "loss_finite": bool(report.loss_finite),
                "grad_finite": bool(report.grad_finite),
                "grad_norm_post_clip": float(report.grad_norm_post_clip),
                "parameters": sum(int(x.numel()) for x in backend.model.parameters()),
                "matched_init_exact": init_receipt["shared_initialization_exact"],
                "scientific_state_untouched": True,
            }
    _write_receipt("PREFLIGHT", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True,
                        choices=("prepare", "preflight", "scan", "run-arm", "development", "finalize"))
    parser.add_argument("--pair-index", type=int, default=0)
    parser.add_argument("--arm", choices=ARMS, default="PHYS_4096")
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()
    if args.mode == "prepare":
        result = prepare_data()["receipt"]
    elif args.mode == "preflight":
        result = preflight(cuda=args.cuda)
    elif args.mode == "scan":
        result = scan()
    elif args.mode == "run-arm":
        result = run_arm(pair_index=args.pair_index, arm=args.arm, cuda=args.cuda)
    elif args.mode == "development":
        result = development_aggregate()
    else:
        result = finalize(cuda=args.cuda)
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
