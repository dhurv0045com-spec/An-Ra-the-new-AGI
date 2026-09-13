"""V5.1 Canary-v2: extended-exposure formation qualification.

This is deliberately a thin extension of :mod:`anra_v5.v51_canary_run`.
V1 already qualified the production model/data/optimizer/checkpoint spine; V2
changes only the preregistered exposure regime and adds stronger sealed-test
and multi-epoch accounting.  The canonical output path remains tied full
softmax after R1C's ``SOFTMAX_COMPETITION_NOT_SUFFICIENT`` verdict.

Operator entry point::

    python -m anra_v5.v51_canary_v2_run --mode prepare
    python -m anra_v5.v51_canary_v2_run --mode preflight --cuda
    python -m anra_v5.v51_canary_v2_run --mode scan
    python -m anra_v5.v51_canary_v2_run --mode run --cuda
    python -m anra_v5.v51_canary_v2_run --mode evaluate --cuda
    python -m anra_v5.v51_canary_v2_run --mode finalize --cuda

Persistent state is isolated under ``V51_CANARY_V2_ROOT``.  The default is a
repo-local fixture path; the Colab launcher binds it to a dedicated Drive
folder before importing this module.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

from anra_v5 import v51_canary_run as base

REPO = Path(__file__).resolve().parents[1]
CANARY_ROOT = Path(
    os.environ.get("V51_CANARY_V2_ROOT", str(REPO / "experiments" / "V5_1_CANARY_V2"))
)
RECEIPTS = CANARY_ROOT / "receipts"
STATE_ROOT = CANARY_ROOT / "state"
LINEAGE_ID = "v51-canary-v2"
PREREG_PATH = REPO / "experiments" / "V5_1_CANARY_V2" / "PREREGISTRATION.json"
SEALED_LOCK = CANARY_ROOT / "SEALED_CONSUMPTION.json"

# Rebind the proven V1 machinery to the isolated V2 state root.  Functions in
# the V1 module resolve these globals at call time, so this does not fork the
# production backend implementation.
base.CANARY_ROOT = CANARY_ROOT
base.RECEIPTS = RECEIPTS
base.STATE_ROOT = STATE_ROOT
base.LINEAGE_ID = LINEAGE_ID


def _canonical_sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def load_prereg() -> dict[str, Any]:
    if not PREREG_PATH.is_file():
        raise SystemExit(f"FAIL_CLOSED: V2 preregistration missing at {PREREG_PATH}")
    payload = json.loads(PREREG_PATH.read_text(encoding="utf-8"))
    if payload.get("schema") != "anra-v51-canary-v2-preregistration/v1":
        raise SystemExit("FAIL_CLOSED: wrong Canary-v2 preregistration schema")
    return payload


def flat_config(prereg: dict[str, Any]) -> dict[str, int]:
    t = prereg["training"]
    return {
        "seed": int(prereg["seed"]),
        "worlds_per_family": int(prereg["dataset"]["worlds_per_family"]),
        "token_budget": int(t["token_budget"]),
        "tokens_per_update": int(t["tokens_per_update"]),
        "checkpoint_every": int(t["checkpoint_every"]),
    }


def _prereg_rung(prereg: dict[str, Any], rung: str) -> dict[str, Any]:
    if rung != "A":
        raise SystemExit("FAIL_CLOSED: Canary-v2 preregisters Rung A only")
    return prereg["model"]


# Make every reused V1 helper consume the frozen V2 scientific inputs.
base.load_prereg = load_prereg
base.flat_config = flat_config
base._prereg_rung = _prereg_rung


def _shortcut_violations(shortcuts: dict[str, dict[str, float]], threshold: float) -> list[dict[str, Any]]:
    bad: list[dict[str, Any]] = []
    for family, scores in sorted(shortcuts.items()):
        for heuristic, score in sorted(scores.items()):
            if float(score) >= threshold:
                bad.append({"family": family, "heuristic": heuristic, "score": float(score)})
    return bad


def _data_receipt_payload(pack: dict[str, Any], prereg: dict[str, Any]) -> dict[str, Any]:
    payload = base.generator_receipt(pack["dataset"], pack["tokenizer_receipt"])
    payload["schema"] = "anra-v51-canary-v2-data-receipt/v1"
    payload["contamination_screen"] = pack["screen"]
    payload["shortcut_baselines_dev"] = pack["shortcuts"]
    payload["shortcut_fail_threshold"] = float(
        prereg["dataset"]["screening"]["shortcut_fail_threshold"]
    )
    payload["v2_fresh_seed"] = int(prereg["seed"])
    return payload


def mode_prepare(args: argparse.Namespace) -> int:
    prereg = load_prereg()
    cfg = flat_config(prereg)
    pack = base.build_pack(seed=cfg["seed"], worlds_per_family=cfg["worlds_per_family"])
    if not pack["screen"]["clean"]:
        print(json.dumps({"action": "FAIL_CLOSED", "reason": "cross-split collision",
                          "collisions": pack["screen"]["collisions"]}, indent=1))
        return 1
    threshold = float(prereg["dataset"]["screening"]["shortcut_fail_threshold"])
    violations = _shortcut_violations(pack["shortcuts"], threshold)
    if violations:
        print(json.dumps({"action": "FAIL_CLOSED", "reason": "shortcut baseline threshold",
                          "threshold": threshold, "violations": violations}, indent=1))
        return 1

    CANARY_ROOT.mkdir(parents=True, exist_ok=True)
    for split, rows in pack["dataset"]["splits"].items():
        (CANARY_ROOT / f"{split}.jsonl").write_text(
            "\n".join(json.dumps(row.__dict__, sort_keys=True) for row in rows),
            encoding="utf-8",
        )
    receipt = _data_receipt_payload(pack, prereg)
    receipt_sha = base.write_receipt("DATA", receipt)
    print(json.dumps({
        "mode": "prepare",
        "data_receipt_sha256": receipt_sha,
        "split_sizes": receipt["split_sizes"],
        "split_hashes": receipt["split_hashes"],
        "clean": True,
        "max_shortcut": max(
            float(v) for scores in pack["shortcuts"].values() for v in scores.values()
        ),
    }, indent=1))
    return 0


def _epoch_stream(pack: dict[str, Any], *, seed: int, epoch: int, tokens_per_update: int):
    return base.build_update_stream(
        pack["shards"], run_seed=seed, epoch=epoch,
        real_tokens_per_update=tokens_per_update,
    )


def _stream_layout(pack: dict[str, Any], prereg: dict[str, Any], target_updates: int) -> dict[str, Any]:
    cfg = flat_config(prereg)
    first = _epoch_stream(
        pack, seed=cfg["seed"], epoch=0, tokens_per_update=cfg["tokens_per_update"]
    )
    if not first:
        raise SystemExit("FAIL_CLOSED DATA: zero complete update windows in epoch 0")
    per_epoch = len(first)
    epochs_needed = (target_updates + per_epoch - 1) // per_epoch
    lengths = []
    for epoch in range(epochs_needed):
        n = len(_epoch_stream(
            pack, seed=cfg["seed"], epoch=epoch,
            tokens_per_update=cfg["tokens_per_update"],
        ))
        lengths.append(n)
        if n != per_epoch:
            raise SystemExit(
                f"FAIL_CLOSED DATA: epoch {epoch} has {n} windows; epoch 0 has {per_epoch}"
            )
    return {"windows_per_epoch": per_epoch, "epochs_needed": epochs_needed,
            "epoch_window_counts": lengths}


def train_updates(*, backend, state, pack, prereg, rung: str, updates: int,
                  checkpoint_every: int, store, wsd_receipt: dict) -> dict[str, Any]:
    """Production training with deterministic sampler epochs.

    The only V2 execution change versus V1 is the mapping from global update to
    ``(epoch, update-within-epoch)``.  Every update still goes through the same
    ``batch_from_window -> ProductionTrainingBackend.step -> certify_update``
    transaction and the same CheckpointStore.
    """
    cfg = flat_config(prereg)
    target = int(prereg["training"]["target_updates"])
    layout = _stream_layout(pack, prereg, target)
    per_epoch = int(layout["windows_per_epoch"])
    cache: dict[int, list[Any]] = {}

    def stream(epoch: int):
        if epoch not in cache:
            cache[epoch] = _epoch_stream(
                pack, seed=cfg["seed"], epoch=epoch,
                tokens_per_update=cfg["tokens_per_update"],
            )
        return cache[epoch]

    trace_rows: list[dict[str, Any]] = []
    start_update = int(state.global_update)
    parent_sha = store.latest_sha256()
    t0 = time.time()

    for offset in range(updates):
        update_index = start_update + offset
        if update_index >= target:
            raise SystemExit("FAIL_CLOSED: update exceeds frozen 360-update endpoint")
        epoch = update_index // per_epoch
        epoch_update = update_index % per_epoch
        window = stream(epoch)[epoch_update]
        batch = base.batch_from_window(
            window,
            pack_manifest_sha256=pack["pack_manifest_sha256"],
            update_ordinal=update_index,
        )
        pre_tokens = int(state.schedule_tokens)
        report = backend.step(state, batch)
        after = state.advance(
            tokens_by_source=report.tokens_by_source,
            cursor=report.cursor,
            rng_state_sha256=report.rng_state_sha256,
            parent_checkpoint_sha256=parent_sha,
        )
        base.certify_update(
            before=state, after=after,
            tokens_by_source=report.tokens_by_source,
            loss_finite=report.loss_finite, grad_finite=report.grad_finite,
            grad_norm_post_clip=report.grad_norm_post_clip,
            tied_preserved=report.tied_preserved,
        )
        state = after
        expected_lr = float(base.canary_lr_at(wsd_receipt)(cumulative_tokens=pre_tokens))
        actual_lr = float(backend.optimizer.param_groups[0]["lr"])
        if actual_lr != expected_lr:
            raise SystemExit(
                f"FAIL_CLOSED SCHEDULE: update {state.global_update} expected {expected_lr} got {actual_lr}"
            )
        row: dict[str, Any] = {
            "update": int(state.global_update),
            "epoch": int(epoch),
            "epoch_update": int(epoch_update + 1),
            "tokens_seen": int(state.cumulative_tokens),
            "optimizer_step": int(state.optimizer_step_max),
            "lr_expected": expected_lr,
            "lr_actual": actual_lr,
            "loss": float(backend.last_receipt["loss"]),
            "grad_norm_post_clip": float(report.grad_norm_post_clip),
            "consumed_real_tokens": int(backend.last_receipt["consumed_real_tokens"]),
        }
        if checkpoint_every and state.global_update % checkpoint_every == 0:
            published = store.publish(
                state=state,
                payloads=base.production_payloads(backend, state=state),
                expected_parent_sha256=parent_sha,
            )
            parent_sha = published
            row["checkpoint_sha256"] = published
        trace_rows.append(row)

    wall = time.time() - t0
    metrics = {
        "updates_this_invocation": len(trace_rows),
        "start_update": start_update,
        "final_update": int(state.global_update),
        "wall_seconds": round(wall, 3),
        "tokens_per_second": round(
            sum(r["consumed_real_tokens"] for r in trace_rows) / max(1e-9, wall), 1
        ),
        "final_loss": trace_rows[-1]["loss"] if trace_rows else None,
        "final_grad_norm_post_clip": trace_rows[-1]["grad_norm_post_clip"] if trace_rows else None,
        **layout,
    }
    return {"state": state, "trace": trace_rows, "metrics": metrics,
            "pack": pack, "wsd_receipt": wsd_receipt}


base.train_updates = train_updates


def _existing_training_trace() -> list[dict[str, Any]]:
    path = RECEIPTS / "TRAINING.json"
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload.get("trace", []))


def _merge_trace(old: list[dict[str, Any]], new: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged = {int(row["update"]): row for row in old}
    for row in new:
        u = int(row["update"])
        if u in merged and merged[u] != row:
            raise SystemExit(f"FAIL_CLOSED: conflicting training trace at update {u}")
        merged[u] = row
    out = [merged[k] for k in sorted(merged)]
    if out and [int(r["update"]) for r in out] != list(range(1, int(out[-1]["update"]) + 1)):
        raise SystemExit("FAIL_CLOSED: training trace contains a gap")
    return out


def mode_run(args: argparse.Namespace) -> int:
    import torch

    prereg = load_prereg()
    _prereg_rung(prereg, args.rung)
    target = int(prereg["training"]["target_updates"])
    if int(args.updates) != target:
        print(json.dumps({"action": "FAIL_CLOSED",
                          "reason": f"scientific endpoint is frozen at {target} updates"}))
        return 1
    if args.bfloat16:
        print(json.dumps({"action": "FAIL_CLOSED",
                          "reason": "V2 substantive run is preregistered CUDA/CPU FP32; BF16 is not authorized"}))
        return 1
    if not args.cuda and not args.allow_cpu:
        print(json.dumps({"action": "FAIL_CLOSED",
                          "reason": "substantive run requires --cuda; use --allow-cpu only for explicit fallback"}))
        return 1
    if args.cuda and not torch.cuda.is_available():
        print(json.dumps({"action": "FAIL_CLOSED", "reason": "--cuda requested but CUDA unavailable"}))
        return 1

    cfg = flat_config(prereg)
    if target * cfg["tokens_per_update"] != cfg["token_budget"]:
        raise SystemExit("FAIL_CLOSED: update endpoint and token budget disagree")
    pack = base.build_pack(seed=cfg["seed"], worlds_per_family=cfg["worlds_per_family"])
    threshold = float(prereg["dataset"]["screening"]["shortcut_fail_threshold"])
    if not pack["screen"]["clean"] or _shortcut_violations(pack["shortcuts"], threshold):
        raise SystemExit("FAIL_CLOSED DATA: integrity/shortcut gate not satisfied")

    plan = base.canary_wsd_receipt(token_budget=cfg["token_budget"])
    device = torch.device("cuda") if args.cuda else None
    backend, expected_params = base.make_backend(
        rung="A", device=device, bfloat16=False,
        schedule=base.canary_lr_at(plan), seed=prereg["seed"],
    )
    instantiated = sum(int(p.numel()) for p in backend.model.parameters())
    if instantiated != int(prereg["model"]["parameter_count"]):
        raise SystemExit(
            f"FAIL_CLOSED MODEL: instantiated {instantiated} != prereg {prereg['model']['parameter_count']}"
        )
    if int(expected_params["total"]) != instantiated:
        raise SystemExit("FAIL_CLOSED MODEL: analytic and instantiated parameter counts disagree")

    store = base.CheckpointStore(STATE_ROOT, LINEAGE_ID)
    scan = base._scan(store)
    if scan["action"] == "FAIL_CLOSED":
        print(json.dumps(scan, indent=1)); return 1
    if scan["action"] == "COMPLETE":
        print(json.dumps({**scan, "note": "V2 already finalized; no rerun performed"}, indent=1))
        return 0

    data_identity = _canonical_sha(_data_receipt_payload(pack, prereg))
    state = base.initial_state(
        lineage_id=LINEAGE_ID,
        pack_manifest_sha256=pack["pack_manifest_sha256"],
        token_budget=cfg["token_budget"],
        tokens_per_update=cfg["tokens_per_update"],
        identities=base.identity_bindings(
            pack_manifest_sha256=pack["pack_manifest_sha256"],
            model_spec_sha=backend.model.spec.sha256(),
            tokenizer_artifact_sha=pack["tokenizer_receipt"]["artifact_sha256"],
            data_receipt_sha=data_identity,
            canary_config_sha=_canonical_sha({**prereg, "rung": "A"}),
            wsd_sha=plan["sha256"],
        ),
        rng_state_sha256="0" * 64,
    )
    if scan["action"] == "RESUME":
        restored_state, payloads = store.restore(scan["checkpoint"])
        if restored_state.identities != state.identities:
            print(json.dumps({"action": "FAIL_CLOSED",
                              "reason": "identity drift versus frozen V2 executable"}, indent=1))
            return 1
        base.restore_production(backend, payloads=payloads)
        state = restored_state

    remaining = target - int(state.global_update)
    if remaining < 0:
        raise SystemExit("FAIL_CLOSED: checkpoint is beyond the frozen endpoint")
    if remaining == 0:
        print(json.dumps({"mode": "run", "note": "fixed endpoint already reached",
                          "state_sha256": state.sha256()}, indent=1))
        return 0

    old_trace = _existing_training_trace()
    result = train_updates(
        backend=backend, state=state, pack=pack, prereg=prereg, rung="A",
        updates=remaining, checkpoint_every=int(args.checkpoint_every),
        store=store, wsd_receipt=plan,
    )
    full_trace = _merge_trace(old_trace, result["trace"])
    base.write_receipt("TRAINING", {
        "schema": "anra-v51-canary-v2-training/v1",
        "rung": "A",
        "target_updates": target,
        "trace": full_trace,
        "metrics": result["metrics"],
        "wsd_receipt": plan,
        "all_lr_match": all(float(r["lr_expected"]) == float(r["lr_actual"]) for r in full_trace),
        "epoch_transitions": [
            {"update": r["update"], "epoch": r["epoch"]}
            for i, r in enumerate(full_trace)
            if i == 0 or int(r["epoch"]) != int(full_trace[i - 1]["epoch"])
        ],
        "frozen_schedule_domain_check": {
            "note": "frozen 5B lr_at verified on its real domain; V2 uses the same WSD shape interface",
            "probe_points": {str(t): base.lr_at(cumulative_tokens=t)
                             for t in (0, 25_000_000, 49_999_999, 50_000_000,
                                       4_499_999_999, 4_999_999_999)},
        },
    })
    print(json.dumps({"mode": "run", "rung": "A",
                      "state_sha256": result["state"].sha256(),
                      "trace_rows_total": len(full_trace),
                      **result["metrics"]}, indent=1))
    return 0


def mode_resume(args: argparse.Namespace) -> int:
    return mode_run(args)


def evaluate_split(backend, tokenizer, rows, *, max_answer_tokens: int = 12) -> dict[str, Any]:
    """Candidate-free generation plus explicit EOS-collapse diagnostics."""
    import torch
    from v5_model.core import packed_layout

    eos = tokenizer.identity.special_token_ids["eos"]
    per_family: dict[str, dict[str, int]] = {}
    device = next(backend.model.parameters()).device
    with torch.no_grad():
        for row in rows:
            prompt_ids = tokenizer.encode(row.prompt)
            answer_ids = tokenizer.encode(row.answer)
            ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            segments = torch.zeros(1, ids.shape[1], dtype=torch.int32, device=device)
            generated: list[int] = []
            stopped_on_eos = False
            for _ in range(max_answer_tokens):
                positions, mask = packed_layout(segments, torch_module=torch)
                logits = backend.model(ids, positions, mask)
                next_id = int(torch.argmax(logits[0, -1]).item())
                if next_id == eos:
                    stopped_on_eos = True
                    break
                generated.append(next_id)
                ids = torch.cat([ids, torch.tensor([[next_id]], device=device)], dim=1)
                segments = torch.cat([
                    segments, torch.zeros(1, 1, dtype=torch.int32, device=device)
                ], dim=1)
            text = tokenizer.decode(generated).strip()
            answer_exact = int(text == row.answer)
            prefix_exact = int(generated[:len(answer_ids)] == answer_ids)
            exact_valid = int(answer_exact and stopped_on_eos)
            eos_after_answer = int(stopped_on_eos and len(generated) == len(answer_ids))
            overgen_after_prefix = int(prefix_exact and not exact_valid)
            scores = per_family.setdefault(row.family, {
                "n": 0,
                "exact_with_valid_eos_count": 0,
                "answer_exact_ignoring_eos_count": 0,
                "answer_prefix_exact_count": 0,
                "eos_stop_count": 0,
                "eos_immediately_after_answer_count": 0,
                "overgeneration_after_correct_prefix_count": 0,
            })
            scores["n"] += 1
            scores["exact_with_valid_eos_count"] += exact_valid
            scores["answer_exact_ignoring_eos_count"] += answer_exact
            scores["answer_prefix_exact_count"] += prefix_exact
            scores["eos_stop_count"] += int(stopped_on_eos)
            scores["eos_immediately_after_answer_count"] += eos_after_answer
            scores["overgeneration_after_correct_prefix_count"] += overgen_after_prefix

    out: dict[str, Any] = {}
    for family, s in per_family.items():
        n = int(s["n"])
        out[family] = {
            "n": n,
            "exact_with_valid_eos": s["exact_with_valid_eos_count"] / n,
            "answer_exact_ignoring_eos": s["answer_exact_ignoring_eos_count"] / n,
            "answer_prefix_exact": s["answer_prefix_exact_count"] / n,
            "eos_stop_rate": s["eos_stop_count"] / n,
            "eos_immediately_after_answer": s["eos_immediately_after_answer_count"] / n,
            "overgeneration_after_correct_prefix": s["overgeneration_after_correct_prefix_count"] / n,
        }
    return out


base.evaluate_split = evaluate_split


def _weighted_exact(scores: dict[str, Any]) -> float:
    total_n = sum(int(v["n"]) for v in scores.values())
    if total_n <= 0:
        return 0.0
    successes = sum(float(v["exact_with_valid_eos"]) * int(v["n"]) for v in scores.values())
    return successes / total_n


def formation_gates(prereg: dict[str, Any], dev: dict[str, Any]) -> dict[str, bool]:
    t = prereg["evaluation"]["v1_thresholds_retained_without_change"]
    return {
        "identity_acquisition_dev": float(dev.get("identity", {}).get("exact_with_valid_eos", 0.0)) >= float(t["identity_dev_min"]),
        "binding_acquisition_dev": float(dev.get("binding", {}).get("exact_with_valid_eos", 0.0)) >= float(t["binding_dev_min"]),
        "dev_transfer": _weighted_exact(dev) >= float(t["dev_overall_min"]),
        "formation_positive": any(
            float(v["exact_with_valid_eos"]) >= float(t["formation_min_any_family"])
            for v in dev.values()
        ),
    }


def mode_evaluate(args: argparse.Namespace) -> int:
    # Reuse V1's restore/eval path with V2 roots/prereg/evaluator.  Development
    # only; sealed rows are not touched here.
    return base.mode_evaluate(args)


def _load_receipt(name: str) -> dict[str, Any]:
    p = RECEIPTS / f"{name}.json"
    if not p.is_file():
        raise SystemExit(f"FAIL_CLOSED: required receipt missing: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def mode_finalize(args: argparse.Namespace) -> int:
    import torch

    prereg = load_prereg()
    _prereg_rung(prereg, args.rung)
    final_path = RECEIPTS / "FINALIZATION.json"
    if final_path.is_file():
        payload = json.loads(final_path.read_text(encoding="utf-8"))
        print(json.dumps({"action": "COMPLETE", "verdict": payload.get("verdict"),
                          "note": "sealed split is not consumed again"}, indent=1))
        return 0
    if SEALED_LOCK.exists():
        print(json.dumps({
            "action": "FAIL_CLOSED",
            "reason": "sealed-consumption marker exists without FINALIZATION; sealed status is ambiguous",
            "required_action": "invalidate/regenerate a new sealed split under an explicit amendment before retry",
        }, indent=1))
        return 1

    cfg = flat_config(prereg)
    pack = base.build_pack(seed=cfg["seed"], worlds_per_family=cfg["worlds_per_family"])
    plan = base.canary_wsd_receipt(token_budget=cfg["token_budget"])
    if args.cuda and not torch.cuda.is_available():
        raise SystemExit("FAIL_CLOSED HARDWARE: CUDA requested but unavailable")
    device = torch.device("cuda") if args.cuda else None
    backend, expected_params = base.make_backend(
        rung="A", device=device, bfloat16=False,
        schedule=base.canary_lr_at(plan), seed=prereg["seed"],
    )
    store = base.CheckpointStore(STATE_ROOT, LINEAGE_ID)
    latest = store.latest_sha256()
    if latest is None:
        raise SystemExit("FAIL_CLOSED CHECKPOINT: no checkpoint exists")
    state, payloads = store.restore(latest)
    base.restore_production(backend, payloads=payloads)

    target = int(prereg["training"]["target_updates"])
    endpoint_ok = (
        int(state.global_update) == target
        and int(state.cumulative_tokens) == int(prereg["training"]["token_budget"])
    )
    training = _load_receipt("TRAINING")
    data = _load_receipt("DATA")
    trace = list(training.get("trace", []))
    trace_updates = [int(r["update"]) for r in trace]
    trace_complete = trace_updates == list(range(1, target + 1))
    lr_match = trace_complete and all(
        float(r["lr_expected"]) == float(r["lr_actual"]) for r in trace
    )
    threshold = float(prereg["dataset"]["screening"]["shortcut_fail_threshold"])
    shortcut_bad = _shortcut_violations(data.get("shortcut_baselines_dev", {}), threshold)
    data_clean = bool(data.get("contamination_screen", {}).get("clean", False)) and not shortcut_bad
    instantiated = sum(int(p.numel()) for p in backend.model.parameters())
    params_ok = (
        instantiated == int(prereg["model"]["parameter_count"])
        and int(expected_params["total"]) == instantiated
    )

    sealed_rows, sealed_hash = base._generate_eval_rows("sealed", prereg)
    dev_rows, dev_hash = base._generate_eval_rows("development", prereg)

    CANARY_ROOT.mkdir(parents=True, exist_ok=True)
    SEALED_LOCK.write_text(json.dumps({
        "schema": "anra-v51-canary-v2-sealed-consumption/v1",
        "status": "STARTED",
        "timestamp": time.time(),
        "executable_commit": base.source_commit(),
        "checkpoint_sha256": latest,
        "sealed_split_sha256": sealed_hash,
    }, indent=1), encoding="utf-8")

    # From this point the sealed split is considered consumed even if the
    # process crashes.  A retry must not silently look at it again.
    sealed = evaluate_split(backend, pack["tokenizer"], sealed_rows)
    dev = evaluate_split(backend, pack["tokenizer"], dev_rows)
    formation = formation_gates(prereg, dev)
    mechanical = {
        "fixed_endpoint_reached": endpoint_ok,
        "parameter_accounting_exact": params_ok,
        "data_integrity_and_shortcuts": data_clean,
        "training_trace_complete": trace_complete,
        "wsd_expected_equals_actual": lr_match,
        "no_wsd_rewarm": lr_match,
        "canonical_full_softmax": prereg["model"]["output_path"] == "tied full softmax, canonical only",
        "checkpoint_identity_valid": state.identities.source_commit == base.source_commit(),
        "eos_contract_exercised": all("eos_stop_rate" in f for f in dev.values()),
    }
    if not all(mechanical.values()):
        verdict = "CANARY_V2_FAIL_ENGINEERING"
    elif not all(formation.values()):
        verdict = "CANARY_V2_FAIL_FORMATION"
    else:
        verdict = "CANARY_V2_PASS"

    receipt = {
        "schema": "anra-v51-canary-v2-finalization/v1",
        "rung": "A",
        "executable_commit": base.source_commit(),
        "checkpoint_sha256": latest,
        "global_update": int(state.global_update),
        "cumulative_tokens": int(state.cumulative_tokens),
        "sealed_split_sha256": sealed_hash,
        "development_split_sha256": dev_hash,
        "sealed_per_family": sealed,
        "development_per_family": dev,
        "development_overall_exact_with_valid_eos": _weighted_exact(dev),
        "sealed_overall_exact_with_valid_eos": _weighted_exact(sealed),
        "mechanical_gates": mechanical,
        "formation_gates": formation,
        "verdict": verdict,
        "claim_ceiling": prereg["claim_ceiling"],
        "next_if_pass": prereg["next_if_pass"],
        "next_if_fail": prereg["next_if_fail"],
    }
    receipt_sha = base.write_receipt("FINALIZATION", receipt)
    SEALED_LOCK.write_text(json.dumps({
        "schema": "anra-v51-canary-v2-sealed-consumption/v1",
        "status": "CONSUMED_AND_FINALIZED",
        "timestamp": time.time(),
        "executable_commit": base.source_commit(),
        "checkpoint_sha256": latest,
        "sealed_split_sha256": sealed_hash,
        "finalization_receipt_sha256": receipt_sha,
    }, indent=1), encoding="utf-8")
    print(json.dumps({**receipt, "receipt_sha256": receipt_sha}, indent=1))
    return 0


def mode_scan(args: argparse.Namespace) -> int:
    store = base.CheckpointStore(STATE_ROOT, LINEAGE_ID)
    payload = base._scan(store)
    payload["root"] = str(CANARY_ROOT)
    payload["lineage"] = LINEAGE_ID
    payload["target_updates"] = int(load_prereg()["training"]["target_updates"])
    print(json.dumps(payload, indent=1))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True,
                        choices=("prepare", "preflight", "run", "resume",
                                 "evaluate", "finalize", "scan"))
    parser.add_argument("--rung", default="A", choices=("A", "B"))
    parser.add_argument("--updates", type=int, default=360,
                        help="TOTAL frozen V2 endpoint; substantive runs must be 360")
    parser.add_argument("--checkpoint-every", type=int, default=24)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--bfloat16", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true",
                        help="explicit fallback for substantive run; scientific default is CUDA FP32")
    args = parser.parse_args()

    if args.mode in {"run", "resume", "evaluate", "finalize"} and args.rung != "A":
        raise SystemExit("FAIL_CLOSED: Canary-v2 authorizes Rung A only")
    if args.mode == "prepare":
        return mode_prepare(args)
    if args.mode == "preflight":
        # V1 preflight is the already-qualified production path, now rebound to
        # the V2 prereg/root.  It remains a one-update engineering smoke.
        return base.mode_preflight(args)
    if args.mode == "scan":
        return mode_scan(args)
    if args.mode == "run":
        return mode_run(args)
    if args.mode == "resume":
        return mode_resume(args)
    if args.mode == "evaluate":
        return mode_evaluate(args)
    if args.mode == "finalize":
        return mode_finalize(args)
    raise AssertionError(args.mode)


if __name__ == "__main__":
    raise SystemExit(main())
