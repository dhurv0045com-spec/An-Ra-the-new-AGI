from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import random
import subprocess
import sys
import time
import traceback
import zipfile
from pathlib import Path
from typing import Any

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments" / "COLAB"))

from discovery_v7_common import (
    ARK014_BINDING_MANIFEST_SHA,
    bind_ark11_runtime,
    cpu_tree,
    current_device,
    file_sha256,
    flat_params,
    generate_indices,
    git_head,
    gradient_norm_and_clip,
    load_ark11,
    load_ark14,
    model_state_equal,
    optimizer_step_with_delta,
    order_sha256,
    parameter_sha,
    sha_json,
)

BASE_PLAN_SHA = "4d145288215c304310253a93881073d5d1a03800"
V2_ADDENDUM_SHA = "2cd849d3be106bf1036cc13a3c2e004fdf843e1b"
MASTER_V8_PLAN_SHA = "6f0a38088e966494bb7caf2f3b49ea235c84f971"
MASTER_V8_V2_ADDENDUM_SHA = "8b5c6a895e5a17a8ab2f344cc9612cb376c10235"
RUNNER_PATH = Path(__file__)
BASE_RUNNER_PATH = REPO / "experiments" / "ARK-017" / "run_ark017.py"
RESULTS_DIR = Path("/content/arkenstone_ark017_v2_results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ACQ_SEEDS = [2601, 2702, 2803]
CONT_SEEDS = [10801, 10802]
ACQ_MAX_STEPS = 16000
CONT_STEPS = 8000
EVAL_EVERY = 200
BATCH_SIZE = 64
HIGH_LR = 1e-3
LOW_LR = 1e-5
DEFAULT_BUDGET_MINUTES = 240.0
SPARSE_COUNTS = {
    "REPLAY_1OF16": 4,
    "REPLAY_1OF32": 2,
    "REPLAY_1OF64": 1,
}


def load_base():
    spec = importlib.util.spec_from_file_location("arkenstone_ark017_base_v1", BASE_RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import base ARK-017 runner: {BASE_RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = load_base()
BASE_RUN_ARM = BASE.run_arm


class Context:
    def __init__(self, device: torch.device, head: str, budget_minutes: float):
        self.device = device
        self.head = head
        self.budget_minutes = float(budget_minutes)
        self.started = time.time()

    @property
    def minutes_used(self) -> float:
        return (time.time() - self.started) / 60.0

    @property
    def minutes_left(self) -> float:
        return self.budget_minutes - self.minutes_used


def save_json(ctx: Context, name: str, payload: dict) -> Path:
    out = dict(payload)
    out["base_plan_commit_sha"] = BASE_PLAN_SHA
    out["v2_addendum_commit_sha"] = V2_ADDENDUM_SHA
    out["master_v8_plan_commit_sha"] = MASTER_V8_PLAN_SHA
    out["master_v8_v2_addendum_commit_sha"] = MASTER_V8_V2_ADDENDUM_SHA
    out["runner_commit_sha"] = ctx.head
    out["runner_source_sha256"] = file_sha256(RUNNER_PATH)
    out["base_runner_source_sha256"] = file_sha256(BASE_RUNNER_PATH)
    out["device"] = str(ctx.device)
    out["torch"] = torch.__version__
    out["campaign_minutes_used"] = ctx.minutes_used
    body = dict(out)
    body.pop("receipt_sha256", None)
    out["receipt_sha256"] = sha_json(body)
    path = RESULTS_DIR / name
    path.write_text(json.dumps(out, indent=2, default=str) + "\n", encoding="utf-8")
    print("saved:", path, flush=True)
    return path


def package_results(download: bool = True) -> Path:
    zip_path = RESULTS_DIR / "ARKENSTONE_ARK017_V2_RESULTS.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(RESULTS_DIR.glob("*.json")):
            zf.write(p, p.name)
    print("RESULT ZIP:", zip_path, flush=True)
    if download:
        try:
            from google.colab import files
            files.download(str(zip_path))
        except Exception as exc:
            print("auto-download skipped:", repr(exc), flush=True)
    return zip_path


def sparse_positions(acq_seed: int, order_seed: int, absolute_step: int, count: int) -> tuple[int, ...]:
    digest = hashlib.sha256(
        f"ark017-v2-replay-pos:{acq_seed}:{order_seed}:{absolute_step}:{count}".encode("utf-8")
    ).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    return tuple(sorted(rng.sample(range(BATCH_SIZE), count)))


def nonidentity_perm(ark14, item: dict, *, acq_seed: int, order_seed: int, absolute_step: int, batch_position: int):
    identity = tuple(range(3))
    nonidentity = [tuple(p) for p in ark14.PERMS if tuple(p) != identity]
    if len(nonidentity) != 5:
        raise RuntimeError("expected exactly five non-identity 3-fact permutations")
    key = (
        f"ark017-v2-replay-perm:{acq_seed}:{order_seed}:{absolute_step}:"
        f"{batch_position}:{int(item['semantic_id'])}"
    ).encode("utf-8")
    idx = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % len(nonidentity)
    return nonidentity[idx]


def sparse_schedule_entry(ark14, train_meta, semantic_ids, *, acq_seed: int, order_seed: int,
                          absolute_step: int, count: int):
    positions = sparse_positions(acq_seed, order_seed, absolute_step, count)
    entry = []
    for pos in positions:
        item = train_meta[int(semantic_ids[pos])]
        perm = nonidentity_perm(
            ark14,
            item,
            acq_seed=acq_seed,
            order_seed=order_seed,
            absolute_step=absolute_step,
            batch_position=pos,
        )
        entry.append((int(pos), tuple(int(x) for x in perm), int(item["semantic_id"])))
    return tuple(entry)


def make_rows_v2(ark14, train_meta, semantic_ids, *, acq_seed: int, order_seed: int,
                 absolute_step: int, mode: str):
    canonical = ark14.make_batch(
        train_meta,
        semantic_ids,
        regime="CANONICAL_TRAIN",
        seed=acq_seed,
        step=absolute_step,
    )
    if mode == "CANONICAL":
        return canonical, ()
    if mode == "AUGMENTED":
        augmented = ark14.make_batch(
            train_meta,
            semantic_ids,
            regime="ORDER_AUGMENTED",
            seed=acq_seed,
            step=absolute_step,
        )
        return augmented, tuple(range(BATCH_SIZE))
    if mode not in SPARSE_COUNTS:
        raise ValueError(f"unknown mode {mode}")

    count = SPARSE_COUNTS[mode]
    schedule = sparse_schedule_entry(
        ark14,
        train_meta,
        semantic_ids,
        acq_seed=acq_seed,
        order_seed=order_seed,
        absolute_step=absolute_step,
        count=count,
    )
    rows = list(canonical)
    for pos, perm, _semantic_id in schedule:
        item = train_meta[int(semantic_ids[pos])]
        facts = tuple(item["facts"][i] for i in perm)
        candidate = ark14.render(facts, int(item["query"]))
        if candidate == canonical[pos]:
            raise RuntimeError("sparse replay selected a canonical/identity presentation")
        rows[pos] = candidate
    return rows, tuple(pos for pos, _perm, _sid in schedule)


def schedule_sha(ark14, train_meta, semantic_indices, *, acq_seed: int, order_seed: int,
                 base_step: int, mode: str) -> str | None:
    if mode not in SPARSE_COUNTS:
        return None
    h = hashlib.sha256()
    count = SPARSE_COUNTS[mode]
    for step in range(1, CONT_STEPS + 1):
        entry = sparse_schedule_entry(
            ark14,
            train_meta,
            semantic_indices[step - 1],
            acq_seed=acq_seed,
            order_seed=order_seed,
            absolute_step=base_step + step,
            count=count,
        )
        h.update(json.dumps(entry, separators=(",", ":")).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def run_arm_v2(ark11, ark14, **kwargs):
    mode = kwargs["mode"]
    out = BASE_RUN_ARM(ark11, ark14, **kwargs)
    if mode in SPARSE_COUNTS:
        expected = SPARSE_COUNTS[mode] * CONT_STEPS
        out["replay_examples"] = expected
        out["noncanonical_replay_examples"] = expected
        out["replay_fraction"] = expected / (CONT_STEPS * BATCH_SIZE)
        out["replay_schedule_sha256"] = schedule_sha(
            ark14,
            kwargs["train_meta"],
            kwargs["semantic_indices"],
            acq_seed=int(kwargs["acq"]["seed"]),
            order_seed=int(kwargs["order_seed"]),
            base_step=int(kwargs["acq"]["qualification_confirmation_step"]),
            mode=mode,
        )
    return out


# Patch module-level dynamic lookups used by the base run_arm.
BASE.make_rows = make_rows_v2
BASE.run_arm = run_arm_v2
BASE.RESULTS_DIR = RESULTS_DIR
BASE.CONT_STEPS = CONT_STEPS
BASE.save_json = save_json
BASE.package_results = package_results


def nested_equal(a: Any, b: Any) -> bool:
    if torch.is_tensor(a) and torch.is_tensor(b):
        return torch.equal(a.detach().cpu(), b.detach().cpu())
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(nested_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(nested_equal(x, y) for x, y in zip(a, b))
    return a == b


def optimizer_finite(optimizer) -> bool:
    for state in optimizer.state.values():
        for value in state.values():
            if torch.is_tensor(value) and not torch.isfinite(value).all():
                return False
    return True


def smoke_test(ctx: Context) -> dict:
    print("\n=== ARK-017 V2 GPU SMOKE TEST ===", flush=True)
    ark11 = load_ark11()
    bind_ark11_runtime(ark11, ctx.device, ctx.head)
    ark14 = load_ark14()
    train_meta, control, sealed, manifest = ark14.build_binding_manifest()
    if manifest.get("manifest_sha256") != ARK014_BINDING_MANIFEST_SHA:
        raise RuntimeError("binding manifest drift")

    ids = list(range(BATCH_SIZE))
    rows1, pos1 = make_rows_v2(
        ark14, train_meta, ids,
        acq_seed=2601, order_seed=10801, absolute_step=17, mode="REPLAY_1OF16"
    )
    rows2, pos2 = make_rows_v2(
        ark14, train_meta, ids,
        acq_seed=2601, order_seed=10801, absolute_step=17, mode="REPLAY_1OF16"
    )
    canonical, _ = make_rows_v2(
        ark14, train_meta, ids,
        acq_seed=2601, order_seed=10801, absolute_step=17, mode="CANONICAL"
    )
    if rows1 != rows2 or pos1 != pos2 or len(pos1) != 4:
        raise RuntimeError("sparse replay determinism/count failure")
    if any(rows1[p] == canonical[p] for p in pos1):
        raise RuntimeError("sparse replay identity-permutation leak")
    if any(rows1[p] != canonical[p] for p in set(range(BATCH_SIZE)) - set(pos1)):
        raise RuntimeError("non-replay row changed")

    vocab = ark11.CompactVocab()
    torch.manual_seed(17017)
    torch.cuda.manual_seed_all(17017)
    model = ark11.Micro(vocab.size, 128).to(ctx.device)
    opt = torch.optim.AdamW(model.parameters(), lr=HIGH_LR, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    train_rows = ark14.make_batch(train_meta, ids, regime="ORDER_AUGMENTED", seed=17017, step=1)
    loss, count = ark11.loss_and_positions(model, vocab, train_rows, ctx.device)
    if not torch.isfinite(loss) or int(count.item()) <= 0:
        raise RuntimeError("GPU forward/loss smoke failure")
    opt.zero_grad(set_to_none=True)
    loss.backward()
    gradient_norm_and_clip(model, 1.0)
    optimizer_step_with_delta(model, opt)

    snap = ark11.snapshot_state(model, opt)
    _, restored_model, restored_opt = ark11.load_fork(snap, HIGH_LR)
    if not model_state_equal(model.state_dict(), restored_model.state_dict()):
        raise RuntimeError("snapshot model reload mismatch")
    if not nested_equal(snap["optimizer"], cpu_tree(restored_opt.state_dict())):
        raise RuntimeError("snapshot optimizer reload mismatch")

    v1, m1, o1 = ark11.load_fork(snap, HIGH_LR)
    v2, m2, o2 = ark11.load_fork(snap, HIGH_LR)
    ark11.train_step(m1, o1, v1, train_rows)
    ark11.train_step(m2, o2, v2, train_rows)
    if parameter_sha(m1) != parameter_sha(m2):
        raise RuntimeError("identical fork/minibatch did not reproduce next update")

    # Multi-step LOW-reference -> capped-HIGH trace consumption.
    vl, ml, ol = ark11.load_fork(snap, LOW_LR)
    vh, mh, oh = ark11.load_fork(snap, HIGH_LR)
    cap_trace = []
    for _ in range(3):
        ll, _ = ark11.loss_and_positions(ml, vl, train_rows, ctx.device)
        ol.zero_grad(set_to_none=True)
        ll.backward()
        gradient_norm_and_clip(ml, 1.0)
        dlow = optimizer_step_with_delta(ml, ol)
        cap = float(dlow["applied_delta_norm"])
        cap_trace.append(cap)

        lh, _ = ark11.loss_and_positions(mh, vh, train_rows, ctx.device)
        oh.zero_grad(set_to_none=True)
        lh.backward()
        gradient_norm_and_clip(mh, 1.0)
        dhigh = optimizer_step_with_delta(mh, oh, cap_norm=cap)
        if float(dhigh["applied_delta_norm"]) > cap * 1.00001 + 1e-12:
            raise RuntimeError("capped HIGH exceeded LOW reference movement")
        if not optimizer_finite(oh):
            raise RuntimeError("nonfinite optimizer state after capped HIGH")

    payload = {
        "status": "PASS",
        "binding_manifest_sha256": manifest["manifest_sha256"],
        "sparse_replay_positions": list(pos1),
        "sparse_replay_exact_fraction": len(pos1) / BATCH_SIZE,
        "sparse_replay_all_noncanonical": True,
        "non_replay_rows_byte_identical": True,
        "gpu_forward_backward": True,
        "snapshot_model_reload_exact": True,
        "snapshot_optimizer_reload_exact": True,
        "identical_next_update": True,
        "multi_step_cap_trace": cap_trace,
        "capped_optimizer_state_finite": True,
    }
    save_json(ctx, "ARK-017_V2_SMOKE_TEST.json", payload)
    print("ARK-017 V2 GPU SMOKE TEST PASS", flush=True)
    return payload


def public_acquisition(a: dict) -> dict:
    return {k: v for k, v in a.items() if k not in {"snapshot", "reference_flat"}}


def run_campaign(ctx: Context) -> dict:
    ark11 = load_ark11()
    bind_ark11_runtime(ark11, ctx.device, ctx.head)
    ark14 = load_ark14()
    train_meta, control, sealed, manifest = ark14.build_binding_manifest()
    if manifest.get("manifest_sha256") != ARK014_BINDING_MANIFEST_SHA:
        raise RuntimeError("ARK-017 V2 binding manifest drift")
    save_json(ctx, "ARK-017_V2_TASK_MANIFEST.json", manifest)

    acq_private: dict[int, dict] = {}
    acquisitions: list[dict] = []
    results: list[dict] = []
    cache: dict[tuple[int, int], dict] = {}

    # Breadth-first: acquire all independent parents before expensive continuation sets.
    for seed in ACQ_SEEDS:
        if ctx.minutes_left < 30:
            acquisitions.append({"seed": seed, "status": "BUDGET_BLOCKED"})
            continue
        print(f"\n=== ARK-017 V2 acquire seed={seed} ===", flush=True)
        a = BASE.acquire(ark11, ark14, seed=seed, train_meta=train_meta, control=control, sealed=sealed)
        acq_private[seed] = a
        acquisitions.append(public_acquisition(a))
        save_json(ctx, "ARK-017_V2_PARTIAL.json", {"acquisitions": acquisitions, "results": results})

    for order_seed in CONT_SEEDS:
        for seed in ACQ_SEEDS:
            a = acq_private.get(seed)
            if not a or a.get("status") != "QUALIFIED":
                continue
            if ctx.minutes_left < 28:
                results.append({
                    "acquisition_seed": seed,
                    "order_seed": order_seed,
                    "status": "BUDGET_BLOCKED_BEFORE_SET",
                })
                save_json(ctx, "ARK-017_V2_PARTIAL.json", {"acquisitions": acquisitions, "results": results})
                continue

            print(f"\n--- ARK-017 V2 seed={seed} order={order_seed} ---", flush=True)
            semantic = generate_indices(order_seed, CONT_STEPS, BATCH_SIZE, len(train_meta))
            common = dict(
                acq=a,
                train_meta=train_meta,
                control=control,
                sealed=sealed,
                semantic_indices=semantic,
                order_seed=order_seed,
            )
            low = run_arm_v2(
                ark11, ark14, **common,
                arm="NARROW_LOW_REFERENCE", lr=LOW_LR, mode="CANONICAL", record_trace=True,
            )
            low_trace = list(low.pop("applied_delta_trace"))
            low_trace_sha = sha_json(low_trace)

            high = run_arm_v2(
                ark11, ark14, **common,
                arm="NARROW_HIGH", lr=HIGH_LR, mode="CANONICAL",
            )
            cap = run_arm_v2(
                ark11, ark14, **common,
                arm="NARROW_HIGH_CAP1X", lr=HIGH_LR, mode="CANONICAL", cap_trace=low_trace,
            )
            replay = run_arm_v2(
                ark11, ark14, **common,
                arm="NARROW_HIGH_REPLAY_1OF16", lr=HIGH_LR, mode="REPLAY_1OF16",
            )
            joint = run_arm_v2(
                ark11, ark14, **common,
                arm="NARROW_HIGH_CAP1X_REPLAY_1OF16", lr=HIGH_LR, mode="REPLAY_1OF16", cap_trace=low_trace,
            )
            aug = run_arm_v2(
                ark11, ark14, **common,
                arm="AUGMENTED_HIGH_REFERENCE", lr=HIGH_LR, mode="AUGMENTED",
            )

            result = {
                "acquisition_seed": seed,
                "order_seed": order_seed,
                "status": "SET_EXECUTED",
                "sealed_at_fork": a["sealed_at_fork"],
                "semantic_order_sha256": order_sha256(semantic),
                "low_applied_delta_trace_sha256": low_trace_sha,
                "arms": {
                    "NARROW_HIGH": high,
                    "NARROW_LOW_REFERENCE": low,
                    "NARROW_HIGH_CAP1X": cap,
                    "NARROW_HIGH_REPLAY_1OF16": replay,
                    "NARROW_HIGH_CAP1X_REPLAY_1OF16": joint,
                    "AUGMENTED_HIGH_REFERENCE": aug,
                },
            }
            results.append(result)
            cache[(seed, order_seed)] = {
                "acq": a,
                "semantic": semantic,
                "low_trace": low_trace,
            }
            save_json(ctx, "ARK-017_V2_PARTIAL.json", {"acquisitions": acquisitions, "results": results})

    core_summary = BASE.summarize(acquisitions, results)
    secondary = {
        "status": "NOT_ELIGIBLE",
        "source_primary_verdict": core_summary.get("primary_verdict"),
        "results": [],
    }

    verdict = core_summary.get("primary_verdict")
    event_sufficient = bool(core_summary.get("event_sufficient"))
    movement_ok = verdict in {"UPDATE_MAGNITUDE_SUFFICIENT", "BOTH_LEVERS_SUFFICIENT"}
    replay_ok = verdict in {"DIVERSITY_SUPPORT_SUFFICIENT", "BOTH_LEVERS_SUFFICIENT"}

    if event_sufficient and (movement_ok or replay_ok):
        selected = []
        if movement_ok and replay_ok and ctx.minutes_left < 45:
            paths = core_summary.get("mean_applied_paths") or {}
            low_path = max(float(paths.get("NARROW_LOW_REFERENCE") or 0.0), 1e-30)
            cap_ratio = float(paths.get("NARROW_HIGH_CAP1X") or 0.0) / low_path
            selected = ["MOVEMENT"] if cap_ratio <= 1.5 else ["REPLAY"]
        else:
            if movement_ok:
                selected.append("MOVEMENT")
            if replay_ok:
                selected.append("REPLAY")

        secondary = {
            "status": "ELIGIBLE",
            "source_primary_verdict": verdict,
            "selected_screens": selected,
            "results": [],
        }
        order_seed = CONT_SEEDS[0]
        for screen in selected:
            if ctx.minutes_left < 12:
                secondary["results"].append({"screen": screen, "status": "BUDGET_BLOCKED"})
                continue
            if screen == "MOVEMENT":
                arm_specs = [
                    ("NARROW_HIGH_CAP4X", 4.0),
                    ("NARROW_HIGH_CAP16X", 16.0),
                ]
                for arm_name, mult in arm_specs:
                    for seed in ACQ_SEEDS:
                        item = cache.get((seed, order_seed))
                        if item is None:
                            continue
                        if ctx.minutes_left < 5:
                            secondary["results"].append({"screen": screen, "arm": arm_name, "seed": seed, "status": "BUDGET_BLOCKED"})
                            continue
                        common = dict(
                            acq=item["acq"], train_meta=train_meta, control=control, sealed=sealed,
                            semantic_indices=item["semantic"], order_seed=order_seed,
                        )
                        trace = [mult * float(x) for x in item["low_trace"]]
                        arm = run_arm_v2(
                            ark11, ark14, **common,
                            arm=arm_name, lr=HIGH_LR, mode="CANONICAL", cap_trace=trace,
                        )
                        secondary["results"].append({
                            "screen": screen, "arm": arm_name, "cap_multiplier": mult,
                            "acquisition_seed": seed, "order_seed": order_seed, "status": "EXECUTED",
                            "result": arm,
                        })
                        save_json(ctx, "ARK-017_V2_SECONDARY_PARTIAL.json", secondary)
            elif screen == "REPLAY":
                arm_specs = [
                    ("NARROW_HIGH_REPLAY_1OF64", "REPLAY_1OF64"),
                    ("NARROW_HIGH_REPLAY_1OF32", "REPLAY_1OF32"),
                ]
                for arm_name, mode in arm_specs:
                    for seed in ACQ_SEEDS:
                        item = cache.get((seed, order_seed))
                        if item is None:
                            continue
                        if ctx.minutes_left < 5:
                            secondary["results"].append({"screen": screen, "arm": arm_name, "seed": seed, "status": "BUDGET_BLOCKED"})
                            continue
                        common = dict(
                            acq=item["acq"], train_meta=train_meta, control=control, sealed=sealed,
                            semantic_indices=item["semantic"], order_seed=order_seed,
                        )
                        arm = run_arm_v2(
                            ark11, ark14, **common,
                            arm=arm_name, lr=HIGH_LR, mode=mode,
                        )
                        secondary["results"].append({
                            "screen": screen, "arm": arm_name,
                            "acquisition_seed": seed, "order_seed": order_seed, "status": "EXECUTED",
                            "result": arm,
                        })
                        save_json(ctx, "ARK-017_V2_SECONDARY_PARTIAL.json", secondary)

        secondary["status"] = "EXECUTED_OR_BUDGETED_PARTIAL"
        save_json(ctx, "ARK-017_V2_SECONDARY.json", secondary)

    payload = {
        "status": "EXECUTED_OR_BUDGETED_PARTIAL",
        "binding_manifest_sha256": manifest["manifest_sha256"],
        "acquisition_seeds": ACQ_SEEDS,
        "continuation_seeds": CONT_SEEDS,
        "continuation_steps": CONT_STEPS,
        "acquisitions": acquisitions,
        "results": results,
        "summary": core_summary,
        "secondary_efficiency_screen": secondary,
    }
    save_json(ctx, "ARK-017_V2_RESULT.json", payload)
    return payload


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--budget-minutes", type=float, default=DEFAULT_BUDGET_MINUTES)
    p.add_argument("--expected-head", type=str, default="")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    device = current_device()
    head = git_head()
    if args.expected_head and head != args.expected_head:
        raise RuntimeError(f"checked-out HEAD {head} != expected {args.expected_head}")
    ctx = Context(device, head, args.budget_minutes)
    print("DEVICE:", torch.cuda.get_device_name(0), "| torch", torch.__version__, flush=True)
    print("HEAD:", head, flush=True)
    print("ARK-017 BASE PLAN:", BASE_PLAN_SHA, flush=True)
    print("ARK-017 V2 ADDENDUM:", V2_ADDENDUM_SHA, flush=True)
    print("MASTER V8 V2 ADDENDUM:", MASTER_V8_V2_ADDENDUM_SHA, flush=True)
    print("RUNNER SHA256:", file_sha256(RUNNER_PATH), flush=True)
    try:
        if args.smoke_test:
            smoke_test(ctx)
        else:
            run_campaign(ctx)
        return 0
    except Exception as exc:
        save_json(ctx, "ARK-017_V2_FAILURE_RECEIPT.json", {
            "status": "FAILED",
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "traceback": traceback.format_exc(),
        })
        raise
    finally:
        package_results(download=not args.smoke_test)


if __name__ == "__main__":
    raise SystemExit(main())
