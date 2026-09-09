from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
import time
import traceback
import zipfile
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments" / "COLAB"))

from discovery_v7_common import (
    ARK014_BINDING_MANIFEST_SHA,
    bind_ark11_runtime,
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

PLAN_SHA = "4d145288215c304310253a93881073d5d1a03800"
MASTER_V8_PLAN_SHA = "6f0a38088e966494bb7caf2f3b49ea235c84f971"
RUNNER_PATH = Path(__file__)
RESULTS_DIR = Path("/content/arkenstone_ark017_results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ACQ_SEEDS = [2601, 2702, 2803]
CONT_SEEDS = [10801, 10802]
ACQ_MAX_STEPS = 16000
CONT_STEPS = 8000
EVAL_EVERY = 200
BATCH_SIZE = 64
HIGH_LR = 1e-3
LOW_LR = 1e-5
DEFAULT_BUDGET_MINUTES = 180.0


def robust_qualified(metrics: dict) -> bool:
    return bool(
        float(metrics["canonical"]) >= 0.90
        and float(metrics["order_only"]) >= 0.85
        and float(metrics["query_order"]) >= 0.85
    )


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
    out["plan_commit_sha"] = PLAN_SHA
    out["master_v8_plan_commit_sha"] = MASTER_V8_PLAN_SHA
    out["runner_commit_sha"] = ctx.head
    out["runner_source_sha256"] = file_sha256(RUNNER_PATH)
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
    zip_path = RESULTS_DIR / "ARKENSTONE_ARK017_RESULTS.zip"
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


def replay_positions(acq_seed: int, order_seed: int, absolute_step: int) -> tuple[int, ...]:
    digest = hashlib.sha256(
        f"ark017-replay:{acq_seed}:{order_seed}:{absolute_step}".encode("utf-8")
    ).digest()
    seed = int.from_bytes(digest[:8], "big")
    rng = random.Random(seed)
    return tuple(sorted(rng.sample(range(BATCH_SIZE), 4)))


def make_rows(
    ark14,
    train_meta,
    semantic_ids,
    *,
    acq_seed: int,
    order_seed: int,
    absolute_step: int,
    mode: str,
):
    canonical = ark14.make_batch(
        train_meta,
        semantic_ids,
        regime="CANONICAL_TRAIN",
        seed=acq_seed,
        step=absolute_step,
    )
    if mode == "CANONICAL":
        return canonical, ()
    augmented = ark14.make_batch(
        train_meta,
        semantic_ids,
        regime="ORDER_AUGMENTED",
        seed=acq_seed,
        step=absolute_step,
    )
    if mode == "AUGMENTED":
        return augmented, tuple(range(BATCH_SIZE))
    if mode != "REPLAY_1OF16":
        raise ValueError(f"unknown mode {mode}")
    pos = replay_positions(acq_seed, order_seed, absolute_step)
    rows = list(canonical)
    for i in pos:
        rows[i] = augmented[i]
    return rows, pos


def acquire(ark11, ark14, *, seed: int, train_meta, control, sealed) -> dict:
    vocab = ark11.CompactVocab()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = ark11.Micro(vocab.size, 128).to(ark11.dev())
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=HIGH_LR, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1
    )
    indices = generate_indices(seed, ACQ_MAX_STEPS, BATCH_SIZE, len(train_meta))
    trajectory = []
    q_history = []
    tokens = 0

    for step in range(1, ACQ_MAX_STEPS + 1):
        rows = ark14.make_batch(
            train_meta,
            indices[step - 1],
            regime="ORDER_AUGMENTED",
            seed=seed,
            step=step,
        )
        loss, count = ark11.loss_and_positions(model, vocab, rows, ark11.dev())
        if not torch.isfinite(loss):
            raise RuntimeError(f"nonfinite acquisition loss seed={seed} step={step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm_and_clip(model, 1.0)
        optimizer.step()
        tokens += int(count.item())

        if step == 1 or step % EVAL_EVERY == 0:
            c = ark14.evaluate(ark11, model, vocab, control)
            q = robust_qualified(c)
            q_history.append((step, q))
            trajectory.append({"step": step, **c, "qualified": q, "loss": float(loss.detach().item())})
            streak = 0
            onset = None
            confirm = None
            for s, hit in q_history:
                if hit:
                    if streak == 0:
                        onset = s
                    streak += 1
                    if streak >= 3:
                        confirm = s
                        break
                else:
                    streak = 0
                    onset = None
            print(
                f"[ARK-017 ACQ s{seed}] step={step} can={c['canonical']:.3f} "
                f"ord={c['order_only']:.3f} qord={c['query_order']:.3f}",
                flush=True,
            )
            if confirm is not None:
                s = ark14.evaluate(ark11, model, vocab, sealed)
                return {
                    "seed": seed,
                    "status": "QUALIFIED",
                    "qualification_onset_step": onset,
                    "qualification_confirmation_step": confirm,
                    "supervised_tokens": tokens,
                    "acquisition_order_sha256": order_sha256(indices),
                    "trajectory": trajectory,
                    "sealed_at_fork": s,
                    "snapshot": ark11.snapshot_state(model, optimizer),
                    "reference_flat": flat_params(model).detach().cpu(),
                }

    return {
        "seed": seed,
        "status": "NO_QUALIFICATION",
        "supervised_tokens": tokens,
        "acquisition_order_sha256": order_sha256(indices),
        "trajectory": trajectory,
        "sealed_final": ark14.evaluate(ark11, model, vocab, sealed),
    }


def retention_metrics(trajectory: list[dict], prefix: str) -> dict:
    if not trajectory:
        return {"status": "EMPTY"}
    qualified = [bool(x[f"{prefix}_qualified"]) for x in trajectory]
    streak = 0
    onset = None
    confirm = None
    for x in trajectory:
        hit = not bool(x[f"{prefix}_qualified"])
        if hit:
            if streak == 0:
                onset = int(x["step"])
            streak += 1
            if streak >= 3:
                confirm = int(x["step"])
                break
        else:
            streak = 0
            onset = None

    out = {
        "RET_QUALIFIED": sum(qualified) / len(qualified),
        "FAILURE_ONSET": onset,
        "FAILURE_CONFIRM": confirm,
        "failed": confirm is not None,
    }
    for metric in ["canonical", "order_only", "query_only", "query_order"]:
        vals = [float(x[f"{prefix}_{metric}"]) for x in trajectory]
        out[f"{metric.upper()}_AREA"] = sum(vals) / len(vals)
        out[f"{metric.upper()}_FINAL"] = vals[-1]
        out[f"{metric.upper()}_PEAK"] = max(vals)
    return out


def run_arm(
    ark11,
    ark14,
    *,
    acq: dict,
    train_meta,
    control,
    sealed,
    semantic_indices,
    order_seed: int,
    arm: str,
    lr: float,
    mode: str,
    cap_trace: list[float] | None = None,
    record_trace: bool = False,
) -> dict:
    if cap_trace is not None and len(cap_trace) != CONT_STEPS:
        raise RuntimeError(f"{arm}: cap trace length mismatch")
    vocab, model, optimizer = ark11.load_fork(acq["snapshot"], lr)
    reference = acq["reference_flat"].to(ark11.dev())
    base_step = int(acq["qualification_confirmation_step"])
    trajectory = []
    tokens = 0
    raw_path = 0.0
    applied_path = 0.0
    grad_sum = 0.0
    grad_max = 0.0
    cap_fires = 0
    applied_trace = []
    replay_count = 0

    for step in range(1, CONT_STEPS + 1):
        abs_step = base_step + step
        rows, replay_pos = make_rows(
            ark14,
            train_meta,
            semantic_indices[step - 1],
            acq_seed=int(acq["seed"]),
            order_seed=order_seed,
            absolute_step=abs_step,
            mode=mode,
        )
        replay_count += len(replay_pos) if mode == "REPLAY_1OF16" else 0
        loss, count = ark11.loss_and_positions(model, vocab, rows, ark11.dev())
        if not torch.isfinite(loss):
            raise RuntimeError(f"{arm}: nonfinite loss at step={step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad = gradient_norm_and_clip(model, 1.0)
        cap = None if cap_trace is None else float(cap_trace[step - 1])
        delta = optimizer_step_with_delta(model, optimizer, cap_norm=cap)
        tokens += int(count.item())
        raw_path += float(delta["raw_delta_norm"])
        applied_path += float(delta["applied_delta_norm"])
        grad_sum += grad
        grad_max = max(grad_max, grad)
        cap_fires += int(bool(delta["cap_fired"]))
        if record_trace:
            applied_trace.append(float(delta["applied_delta_norm"]))

        if step % EVAL_EVERY == 0:
            c = ark14.evaluate(ark11, model, vocab, control)
            s = ark14.evaluate(ark11, model, vocab, sealed)
            current = flat_params(model)
            disp = float((current - reference).norm().item())
            row = {
                "step": step,
                "loss": float(loss.detach().item()),
                "lr": lr,
                "relative_displacement": disp / max(float(reference.norm().item()), 1e-12),
                "cumulative_raw_path": raw_path,
                "cumulative_applied_path": applied_path,
                "cap_fired_fraction_so_far": cap_fires / step,
            }
            for k, v in c.items():
                if k != "qualified":
                    row[f"control_{k}"] = float(v)
            for k, v in s.items():
                if k != "qualified":
                    row[f"sealed_{k}"] = float(v)
            row["control_qualified"] = robust_qualified(c)
            row["sealed_qualified"] = robust_qualified(s)
            trajectory.append(row)

    out = {
        "arm": arm,
        "lr": lr,
        "mode": mode,
        "order_seed": order_seed,
        "semantic_order_sha256": order_sha256(semantic_indices),
        "supervised_tokens": tokens,
        "raw_path": raw_path,
        "applied_path": applied_path,
        "mean_raw_step_delta_norm": raw_path / CONT_STEPS,
        "mean_applied_step_delta_norm": applied_path / CONT_STEPS,
        "mean_preclip_gradient_norm": grad_sum / CONT_STEPS,
        "max_preclip_gradient_norm": grad_max,
        "cap_fired_fraction": cap_fires / CONT_STEPS,
        "replay_examples": replay_count,
        "replay_fraction": replay_count / (CONT_STEPS * BATCH_SIZE),
        "final_parameter_sha256": parameter_sha(model),
        "control_retention": retention_metrics(trajectory, "control"),
        "sealed_retention": retention_metrics(trajectory, "sealed"),
        "trajectory": trajectory,
    }
    if record_trace:
        out["applied_delta_trace"] = applied_trace
    return out


def summarize(acquisitions: list[dict], results: list[dict]) -> dict:
    primary = [
        r for r in results
        if r.get("status") == "SET_EXECUTED" and robust_qualified(r["sealed_at_fork"])
    ]
    n = len(primary)
    seed_count = len({int(r["acquisition_seed"]) for r in primary})
    names = [
        "NARROW_HIGH",
        "NARROW_LOW_REFERENCE",
        "NARROW_HIGH_CAP1X",
        "NARROW_HIGH_REPLAY_1OF16",
        "NARROW_HIGH_CAP1X_REPLAY_1OF16",
        "AUGMENTED_HIGH_REFERENCE",
    ]
    risks = {}
    fails = {}
    paths = {}
    for arm in names:
        fails[arm] = sum(bool(r["arms"][arm]["sealed_retention"]["failed"]) for r in primary)
        risks[arm] = (fails[arm] / n) if n else None
        paths[arm] = (
            sum(float(r["arms"][arm]["applied_path"]) for r in primary) / n if n else None
        )

    high_risk = risks["NARROW_HIGH"]
    low_risk = risks["NARROW_LOW_REFERENCE"]
    cap_risk = risks["NARROW_HIGH_CAP1X"]
    replay_risk = risks["NARROW_HIGH_REPLAY_1OF16"]
    joint_risk = risks["NARROW_HIGH_CAP1X_REPLAY_1OF16"]
    ref_risk = risks["AUGMENTED_HIGH_REFERENCE"]

    event_sufficient = bool(n >= 5 and seed_count == 3 and high_risk is not None and high_risk >= 0.60)
    flags = []
    if event_sufficient:
        movement_rescue = high_risk - cap_risk
        support_rescue = high_risk - replay_risk
        joint_rescue = high_risk - joint_risk

        low_path = max(float(paths["NARROW_LOW_REFERENCE"]), 1e-30)
        cap_path_ratio = float(paths["NARROW_HIGH_CAP1X"]) / low_path
        high_path = max(float(paths["NARROW_HIGH"]), 1e-30)
        replay_path_ratio = float(paths["NARROW_HIGH_REPLAY_1OF16"]) / high_path

        movement_ok = (
            movement_rescue >= 0.50
            and cap_risk <= low_risk + 0.20
            and 0.70 <= cap_path_ratio <= 1.30
        )
        support_ok = (
            support_rescue >= 0.50
            and replay_risk <= low_risk + 0.20
            and replay_path_ratio >= 0.50
        )
        joint_ok = joint_rescue >= 0.50 and joint_risk <= low_risk + 0.20

        if movement_ok:
            flags.append("UPDATE_MAGNITUDE_SUFFICIENT")
        if support_ok:
            flags.append("DIVERSITY_SUPPORT_SUFFICIENT")
        if movement_ok and support_ok:
            primary_verdict = "BOTH_LEVERS_SUFFICIENT"
        elif movement_ok:
            primary_verdict = "UPDATE_MAGNITUDE_SUFFICIENT"
        elif support_ok:
            primary_verdict = "DIVERSITY_SUPPORT_SUFFICIENT"
        elif joint_ok:
            primary_verdict = "JOINT_CONTROL_REQUIRED"
            flags.append("JOINT_CONTROL_REQUIRED")
        else:
            primary_verdict = "MECHANISM_MIXED_OR_UNRESOLVED"
        if ref_risk is not None and ref_risk >= 0.40:
            flags.append("REFERENCE_INSTABILITY_WARNING")
    else:
        movement_rescue = support_rescue = joint_rescue = None
        primary_verdict = "INCONCLUSIVE_LOW_EVENT_RATE"

    by_seed = {}
    for seed in sorted({int(r["acquisition_seed"]) for r in primary}):
        rows = [r for r in primary if int(r["acquisition_seed"]) == seed]
        by_seed[str(seed)] = {
            "n": len(rows),
            **{
                f"{arm}_failures": sum(bool(x["arms"][arm]["sealed_retention"]["failed"]) for x in rows)
                for arm in names
            },
        }

    return {
        "acquisitions_qualified": sum(a.get("status") == "QUALIFIED" for a in acquisitions),
        "primary_matched_sets": n,
        "independent_acquisition_seeds": seed_count,
        "failure_counts": fails,
        "failure_risks": risks,
        "mean_applied_paths": paths,
        "movement_rescue": movement_rescue,
        "support_rescue": support_rescue,
        "joint_rescue": joint_rescue,
        "by_acquisition_seed": by_seed,
        "event_sufficient": event_sufficient,
        "mechanism_flags": flags,
        "primary_verdict": primary_verdict,
    }


def smoke_test(ctx: Context) -> dict:
    print("\n=== ARK-017 GPU SMOKE TEST ===", flush=True)
    ark11 = load_ark11()
    bind_ark11_runtime(ark11, ctx.device, ctx.head)
    ark14 = load_ark14()
    train_meta, control, sealed, manifest = ark14.build_binding_manifest()
    if manifest.get("manifest_sha256") != ARK014_BINDING_MANIFEST_SHA:
        raise RuntimeError("binding manifest drift")

    ids = list(range(BATCH_SIZE))
    r1, p1 = make_rows(
        ark14, train_meta, ids,
        acq_seed=2601, order_seed=10801, absolute_step=17, mode="REPLAY_1OF16"
    )
    r2, p2 = make_rows(
        ark14, train_meta, ids,
        acq_seed=2601, order_seed=10801, absolute_step=17, mode="REPLAY_1OF16"
    )
    if p1 != p2 or len(p1) != 4 or r1 != r2:
        raise RuntimeError("replay determinism/count failure")

    vocab = ark11.CompactVocab()
    torch.manual_seed(17017)
    torch.cuda.manual_seed_all(17017)
    model = ark11.Micro(vocab.size, 128).to(ctx.device)
    opt = torch.optim.AdamW(model.parameters(), lr=HIGH_LR, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    rows = ark14.make_batch(train_meta, ids, regime="ORDER_AUGMENTED", seed=17017, step=1)
    loss, count = ark11.loss_and_positions(model, vocab, rows, ctx.device)
    if not torch.isfinite(loss) or int(count.item()) <= 0:
        raise RuntimeError("GPU forward/loss smoke failure")
    opt.zero_grad(set_to_none=True)
    loss.backward()
    gradient_norm_and_clip(model, 1.0)
    delta = optimizer_step_with_delta(model, opt)
    if float(delta["raw_delta_norm"]) <= 0:
        raise RuntimeError("zero update in smoke")

    snap = ark11.snapshot_state(model, opt)
    _, restored, _ = ark11.load_fork(snap, HIGH_LR)
    if not model_state_equal(model.state_dict(), restored.state_dict()):
        raise RuntimeError("snapshot model reload mismatch")

    v, m, o = ark11.load_fork(snap, HIGH_LR)
    loss2, _ = ark11.loss_and_positions(m, v, rows, ctx.device)
    o.zero_grad(set_to_none=True)
    loss2.backward()
    gradient_norm_and_clip(m, 1.0)
    uncapped = optimizer_step_with_delta(m, o)
    raw = float(uncapped["raw_delta_norm"])

    v, m, o = ark11.load_fork(snap, HIGH_LR)
    loss3, _ = ark11.loss_and_positions(m, v, rows, ctx.device)
    o.zero_grad(set_to_none=True)
    loss3.backward()
    gradient_norm_and_clip(m, 1.0)
    cap = raw * 0.25
    capped = optimizer_step_with_delta(m, o, cap_norm=cap)
    if not bool(capped["cap_fired"]) or float(capped["applied_delta_norm"]) > cap * 1.00001:
        raise RuntimeError("cap primitive smoke failure")

    payload = {
        "status": "PASS",
        "binding_manifest_sha256": manifest["manifest_sha256"],
        "replay_positions": list(p1),
        "replay_exact_fraction": len(p1) / BATCH_SIZE,
        "gpu_forward_backward": True,
        "snapshot_model_reload_exact": True,
        "cap_raw_delta_norm": raw,
        "cap_requested": cap,
        "cap_applied": float(capped["applied_delta_norm"]),
    }
    save_json(ctx, "ARK-017_SMOKE_TEST.json", payload)
    print("ARK-017 GPU SMOKE TEST PASS", flush=True)
    return payload


def run_campaign(ctx: Context) -> dict:
    ark11 = load_ark11()
    bind_ark11_runtime(ark11, ctx.device, ctx.head)
    ark14 = load_ark14()
    train_meta, control, sealed, manifest = ark14.build_binding_manifest()
    if manifest.get("manifest_sha256") != ARK014_BINDING_MANIFEST_SHA:
        raise RuntimeError("ARK-017 binding manifest drift")
    save_json(ctx, "ARK-017_TASK_MANIFEST.json", manifest)

    acq_private = {}
    acquisitions = []
    results = []

    for seed in ACQ_SEEDS:
        if ctx.minutes_left < 25:
            acquisitions.append({"seed": seed, "status": "BUDGET_BLOCKED"})
            continue
        print(f"\n=== ARK-017 acquire seed={seed} ===", flush=True)
        a = acquire(ark11, ark14, seed=seed, train_meta=train_meta, control=control, sealed=sealed)
        acq_private[seed] = a
        acquisitions.append({k: v for k, v in a.items() if k not in {"snapshot", "reference_flat"}})
        save_json(ctx, "ARK-017_PARTIAL.json", {"acquisitions": acquisitions, "results": results})

    for order_seed in CONT_SEEDS:
        for seed in ACQ_SEEDS:
            a = acq_private.get(seed)
            if not a or a.get("status") != "QUALIFIED":
                continue
            if ctx.minutes_left < 20:
                results.append({
                    "acquisition_seed": seed,
                    "order_seed": order_seed,
                    "status": "BUDGET_BLOCKED_BEFORE_SET",
                })
                save_json(ctx, "ARK-017_PARTIAL.json", {"acquisitions": acquisitions, "results": results})
                continue

            print(f"\n--- ARK-017 seed={seed} order={order_seed} ---", flush=True)
            semantic = generate_indices(order_seed, CONT_STEPS, BATCH_SIZE, len(train_meta))

            low = run_arm(
                ark11, ark14,
                acq=a, train_meta=train_meta, control=control, sealed=sealed,
                semantic_indices=semantic, order_seed=order_seed,
                arm="NARROW_LOW_REFERENCE", lr=LOW_LR, mode="CANONICAL",
                record_trace=True,
            )
            low_trace = list(low.pop("applied_delta_trace"))
            low_trace_sha = sha_json(low_trace)

            high = run_arm(
                ark11, ark14,
                acq=a, train_meta=train_meta, control=control, sealed=sealed,
                semantic_indices=semantic, order_seed=order_seed,
                arm="NARROW_HIGH", lr=HIGH_LR, mode="CANONICAL",
            )
            cap = run_arm(
                ark11, ark14,
                acq=a, train_meta=train_meta, control=control, sealed=sealed,
                semantic_indices=semantic, order_seed=order_seed,
                arm="NARROW_HIGH_CAP1X", lr=HIGH_LR, mode="CANONICAL",
                cap_trace=low_trace,
            )
            replay = run_arm(
                ark11, ark14,
                acq=a, train_meta=train_meta, control=control, sealed=sealed,
                semantic_indices=semantic, order_seed=order_seed,
                arm="NARROW_HIGH_REPLAY_1OF16", lr=HIGH_LR, mode="REPLAY_1OF16",
            )
            joint = run_arm(
                ark11, ark14,
                acq=a, train_meta=train_meta, control=control, sealed=sealed,
                semantic_indices=semantic, order_seed=order_seed,
                arm="NARROW_HIGH_CAP1X_REPLAY_1OF16", lr=HIGH_LR, mode="REPLAY_1OF16",
                cap_trace=low_trace,
            )
            aug = run_arm(
                ark11, ark14,
                acq=a, train_meta=train_meta, control=control, sealed=sealed,
                semantic_indices=semantic, order_seed=order_seed,
                arm="AUGMENTED_HIGH_REFERENCE", lr=HIGH_LR, mode="AUGMENTED",
            )

            results.append({
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
            })
            save_json(ctx, "ARK-017_PARTIAL.json", {"acquisitions": acquisitions, "results": results})

    payload = {
        "status": "EXECUTED_OR_BUDGETED_PARTIAL",
        "binding_manifest_sha256": manifest["manifest_sha256"],
        "acquisition_seeds": ACQ_SEEDS,
        "continuation_seeds": CONT_SEEDS,
        "continuation_steps": CONT_STEPS,
        "acquisitions": acquisitions,
        "results": results,
        "summary": summarize(acquisitions, results),
    }
    save_json(ctx, "ARK-017_RESULT.json", payload)
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
    print("PLAN:", PLAN_SHA, "| MASTER V8:", MASTER_V8_PLAN_SHA, flush=True)
    try:
        if args.smoke_test:
            smoke_test(ctx)
        else:
            run_campaign(ctx)
        return 0
    except Exception as exc:
        save_json(ctx, "ARK-017_FAILURE_RECEIPT.json", {
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
