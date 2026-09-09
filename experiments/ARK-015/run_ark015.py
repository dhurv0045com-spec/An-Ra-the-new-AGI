from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments" / "COLAB"))

from discovery_v7_common import (
    ARK014_BINDING_MANIFEST_SHA,
    ARK015_PLAN_SHA,
    ReceiptWriter,
    RunContext,
    bind_ark11_runtime,
    detect_sustained,
    flat_params,
    generate_indices,
    gradient_norm_and_clip,
    load_ark11,
    load_ark14,
    optimizer_step_with_delta,
    order_sha256,
    parameter_sha,
)

RUNNER_PATH = Path(__file__)
ACQ_SEEDS = [2301, 2402, 2503]
CONT_SEEDS = [8801, 8802, 8803]
ACQ_MAX_STEPS = 16000
CONT_STEPS = 12000
EVAL_EVERY = 200


def robust_qualified(metrics: dict) -> bool:
    return bool(
        float(metrics["canonical"]) >= 0.90
        and float(metrics["order_only"]) >= 0.85
        and float(metrics["query_order"]) >= 0.85
    )


def acquire_robust_binding(ark11, ark14, *, seed: int, train_meta, control, sealed) -> dict:
    vocab = ark11.CompactVocab()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = ark11.Micro(vocab.size, 128).to(ark11.dev())
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1
    )
    indices = generate_indices(seed, ACQ_MAX_STEPS, 64, len(train_meta))
    qualification_evals: list[tuple[int, float]] = []
    trajectory = []
    supervised_tokens = 0
    started = time.time()

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
            raise RuntimeError(f"ARK-015 acquisition nonfinite loss seed={seed} step={step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm_and_clip(model, 1.0)
        optimizer.step()
        supervised_tokens += int(count.item())

        if step == 1 or step % EVAL_EVERY == 0:
            c = ark14.evaluate(ark11, model, vocab, control)
            q = robust_qualified(c)
            qualification_evals.append((step, 1.0 if q else 0.0))
            onset, confirm = detect_sustained(qualification_evals, 1.0, 3)
            trajectory.append({"step": step, **c, "qualified": q, "loss": float(loss.detach().item())})
            print(
                f"[ARK-015 ACQ s{seed}] step={step} can={c['canonical']:.3f} "
                f"ord={c['order_only']:.3f} qord={c['query_order']:.3f}",
                flush=True,
            )
            if confirm is not None:
                sealed_once = ark14.evaluate(ark11, model, vocab, sealed)
                return {
                    "seed": seed,
                    "status": "QUALIFIED",
                    "qualification_onset_step": onset,
                    "qualification_confirmation_step": confirm,
                    "supervised_tokens": supervised_tokens,
                    "acquisition_order_sha256": order_sha256(indices),
                    "trajectory": trajectory,
                    "sealed_at_fork": sealed_once,
                    "snapshot": ark11.snapshot_state(model, optimizer),
                    "reference_flat": flat_params(model).detach().cpu(),
                }

        if time.time() - started > 1800:
            return {
                "seed": seed,
                "status": "ACQUISITION_WALLTIME_STOP",
                "supervised_tokens": supervised_tokens,
                "acquisition_order_sha256": order_sha256(indices),
                "trajectory": trajectory,
            }

    sealed_final = ark14.evaluate(ark11, model, vocab, sealed)
    return {
        "seed": seed,
        "status": "NO_QUALIFICATION",
        "supervised_tokens": supervised_tokens,
        "acquisition_order_sha256": order_sha256(indices),
        "trajectory": trajectory,
        "sealed_final": sealed_final,
    }


def retention_metrics(trajectory: list[dict], prefix: str) -> dict:
    if not trajectory:
        return {"status": "EMPTY"}
    qualified = [bool(x[f"{prefix}_qualified"]) for x in trajectory]
    failure_evals = [
        (int(x["step"]), 1.0 if not bool(x[f"{prefix}_qualified"]) else 0.0)
        for x in trajectory
    ]
    onset, confirm = detect_sustained(failure_evals, 1.0, 3)
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


def run_continuation_arm(
    ark11,
    ark14,
    *,
    acq: dict,
    train_meta,
    control,
    sealed,
    semantic_indices,
    arm: str,
    lr: float,
    render_regime: str,
) -> dict:
    vocab, model, optimizer = ark11.load_fork(acq["snapshot"], lr)
    reference = acq["reference_flat"].to(ark11.dev())
    base_step = int(acq["qualification_confirmation_step"])
    trajectory = []
    supervised_tokens = 0
    cumulative_path = 0.0
    raw_delta_sum = 0.0
    grad_norm_sum = 0.0
    grad_norm_max = 0.0

    for step in range(1, CONT_STEPS + 1):
        rows = ark14.make_batch(
            train_meta,
            semantic_indices[step - 1],
            regime=render_regime,
            seed=int(acq["seed"]),
            step=base_step + step,
        )
        loss, count = ark11.loss_and_positions(model, vocab, rows, ark11.dev())
        if not torch.isfinite(loss):
            raise RuntimeError(f"ARK-015 {arm}: nonfinite loss at step={step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = gradient_norm_and_clip(model, 1.0)
        delta = optimizer_step_with_delta(model, optimizer)
        supervised_tokens += int(count.item())
        cumulative_path += float(delta["applied_delta_norm"])
        raw_delta_sum += float(delta["raw_delta_norm"])
        grad_norm_sum += grad_norm
        grad_norm_max = max(grad_norm_max, grad_norm)

        if step % EVAL_EVERY == 0:
            c = ark14.evaluate(ark11, model, vocab, control)
            s = ark14.evaluate(ark11, model, vocab, sealed)
            current = flat_params(model)
            disp = float((current - reference).norm().item())
            row = {
                "step": step,
                "lr": lr,
                "loss": float(loss.detach().item()),
                "relative_displacement": disp / max(float(reference.norm().item()), 1e-12),
                "cumulative_path_length": cumulative_path,
            }
            for k, v in c.items():
                if k == "qualified":
                    continue
                row[f"control_{k}"] = float(v)
            for k, v in s.items():
                if k == "qualified":
                    continue
                row[f"sealed_{k}"] = float(v)
            row["control_qualified"] = robust_qualified(c)
            row["sealed_qualified"] = robust_qualified(s)
            trajectory.append(row)

    return {
        "arm": arm,
        "lr": lr,
        "render_regime": render_regime,
        "supervised_tokens": supervised_tokens,
        "cumulative_path_length": cumulative_path,
        "mean_raw_step_delta_norm": raw_delta_sum / CONT_STEPS,
        "mean_preclip_gradient_norm": grad_norm_sum / CONT_STEPS,
        "max_preclip_gradient_norm": grad_norm_max,
        "final_parameter_sha256": parameter_sha(model),
        "control_retention": retention_metrics(trajectory, "control"),
        "sealed_retention": retention_metrics(trajectory, "sealed"),
        "trajectory": trajectory,
    }


def summarize(acquisitions: list[dict], results: list[dict]) -> dict:
    qualified_rows = [
        r for r in results
        if r.get("status") == "TRIPLET_EXECUTED" and robust_qualified(r["sealed_at_fork"])
    ]
    n = len(qualified_rows)
    high_fail = sum(bool(r["arms"]["NARROW_HIGH"]["sealed_retention"]["failed"]) for r in qualified_rows)
    low_fail = sum(bool(r["arms"]["NARROW_LOW"]["sealed_retention"]["failed"]) for r in qualified_rows)
    ref_fail = sum(bool(r["arms"]["AUGMENTED_HIGH_REFERENCE"]["sealed_retention"]["failed"]) for r in qualified_rows)
    risk_diff = ((low_fail - high_fail) / n) if n else None
    high_bad_low_good = sum(
        bool(r["arms"]["NARROW_HIGH"]["sealed_retention"]["failed"])
        and not bool(r["arms"]["NARROW_LOW"]["sealed_retention"]["failed"])
        for r in qualified_rows
    )
    reverse = sum(
        not bool(r["arms"]["NARROW_HIGH"]["sealed_retention"]["failed"])
        and bool(r["arms"]["NARROW_LOW"]["sealed_retention"]["failed"])
        for r in qualified_rows
    )

    grouped = {}
    for seed in sorted({int(r["acquisition_seed"]) for r in qualified_rows}):
        rows = [r for r in qualified_rows if int(r["acquisition_seed"]) == seed]
        grouped[str(seed)] = {
            "n": len(rows),
            "narrow_high_failures": sum(bool(r["arms"]["NARROW_HIGH"]["sealed_retention"]["failed"]) for r in rows),
            "narrow_low_failures": sum(bool(r["arms"]["NARROW_LOW"]["sealed_retention"]["failed"]) for r in rows),
            "augmented_high_reference_failures": sum(bool(r["arms"]["AUGMENTED_HIGH_REFERENCE"]["sealed_retention"]["failed"]) for r in rows),
        }

    enough_pairs = n >= 6 and len(grouped) == 3
    enough_high_events = high_fail >= 3
    nonreversed_groups = sum(v["narrow_low_failures"] <= v["narrow_high_failures"] for v in grouped.values())
    majority_nonreversed = bool(grouped) and nonreversed_groups >= (len(grouped) // 2 + 1)

    if not enough_pairs or not enough_high_events:
        verdict = "INCONCLUSIVE_LOW_EVENT_RATE"
    elif risk_diff is not None and risk_diff <= -0.33 and reverse <= 1 and majority_nonreversed:
        verdict = "SUPPORTED_NONARITHMETIC_INVARIANCE_PROTECTION"
    else:
        verdict = "TRANSFER_NOT_SUPPORTED"

    stress_flag = "STRESS_SPECIFICITY_UNRESOLVED"
    if n:
        high_rate = high_fail / n
        ref_rate = ref_fail / n
        if abs(ref_rate - high_rate) <= 0.15:
            stress_flag = "STRESS_NOT_SPECIFIC"
        elif high_rate - ref_rate >= 0.30:
            stress_flag = "STRESS_SPECIFIC_TO_NARROWING"

    path_ratios = []
    for r in qualified_rows:
        high_path = float(r["arms"]["NARROW_HIGH"]["cumulative_path_length"])
        low_path = float(r["arms"]["NARROW_LOW"]["cumulative_path_length"])
        if low_path > 0:
            path_ratios.append(high_path / low_path)

    return {
        "acquisitions_qualified": sum(a.get("status") == "QUALIFIED" for a in acquisitions),
        "sealed_qualified_primary_pairs": n,
        "independent_acquisition_seeds_in_primary": len(grouped),
        "narrow_high_failures": high_fail,
        "narrow_low_failures": low_fail,
        "augmented_high_reference_failures": ref_fail,
        "risk_difference_low_minus_high": risk_diff,
        "high_fail_low_stable": high_bad_low_good,
        "reverse_discordance": reverse,
        "by_acquisition_seed": grouped,
        "median_high_to_low_path_ratio": statistics.median(path_ratios) if path_ratios else None,
        "verdict": verdict,
        "stress_specificity_flag": stress_flag,
    }


def run_campaign(ctx: RunContext) -> dict:
    writer = ReceiptWriter(ctx, experiment_id="ARK-015", plan_sha=ARK015_PLAN_SHA, runner_path=RUNNER_PATH)
    ark11 = load_ark11()
    bind_ark11_runtime(ark11, ctx.device, ctx.head)
    ark14 = load_ark14()

    train_meta, control, sealed, manifest = ark14.build_binding_manifest()
    if manifest.get("manifest_sha256") != ARK014_BINDING_MANIFEST_SHA:
        raise RuntimeError(
            f"ARK-015 binding manifest drift: {manifest.get('manifest_sha256')} != {ARK014_BINDING_MANIFEST_SHA}"
        )
    writer.save("ARK-015_TASK_MANIFEST.json", manifest)

    acquisitions: list[dict] = []
    runtime_acq: dict[int, dict] = {}
    results: list[dict] = []

    # Acquire all fresh subjects before expensive continuation triplets so a budget
    # shortfall cannot silently turn a 3-seed design into a single-seed result.
    for seed in ACQ_SEEDS:
        if ctx.minutes_left < 18:
            acquisitions.append({"seed": seed, "status": "BUDGET_BLOCKED"})
            writer.save("ARK-015_PARTIAL.json", {"acquisitions": acquisitions, "results": results})
            continue
        print(f"\n=== ARK-015 robust acquisition seed={seed} ===", flush=True)
        acq = acquire_robust_binding(ark11, ark14, seed=seed, train_meta=train_meta, control=control, sealed=sealed)
        if acq["status"] == "QUALIFIED":
            runtime_acq[seed] = acq
        acquisitions.append({k: v for k, v in acq.items() if k not in {"snapshot", "reference_flat"}})
        writer.save("ARK-015_PARTIAL.json", {"acquisitions": acquisitions, "results": results})

    # Breadth-first orders: order 8801 is attempted for all qualified subjects before
    # any subject receives 8802, preserving independent-seed coverage under budget.
    for order_seed in CONT_SEEDS:
        for seed in ACQ_SEEDS:
            acq = runtime_acq.get(seed)
            if acq is None:
                continue
            if ctx.minutes_left < 18:
                results.append({
                    "acquisition_seed": seed,
                    "order_seed": order_seed,
                    "status": "BUDGET_BLOCKED_BEFORE_TRIPLET",
                })
                writer.save("ARK-015_PARTIAL.json", {"acquisitions": acquisitions, "results": results})
                continue

            print(f"\n--- ARK-015 seed={seed} order={order_seed} ---", flush=True)
            semantic_indices = generate_indices(order_seed, CONT_STEPS, 64, len(train_meta))
            common = {
                "ark11": ark11,
                "ark14": ark14,
                "acq": acq,
                "train_meta": train_meta,
                "control": control,
                "sealed": sealed,
                "semantic_indices": semantic_indices,
            }
            narrow_high = run_continuation_arm(
                **common,
                arm="NARROW_HIGH",
                lr=1e-3,
                render_regime="CANONICAL_TRAIN",
            )
            narrow_low = run_continuation_arm(
                **common,
                arm="NARROW_LOW",
                lr=1e-5,
                render_regime="CANONICAL_TRAIN",
            )
            augmented_high = run_continuation_arm(
                **common,
                arm="AUGMENTED_HIGH_REFERENCE",
                lr=1e-3,
                render_regime="ORDER_AUGMENTED",
            )
            results.append({
                "acquisition_seed": seed,
                "order_seed": order_seed,
                "semantic_order_sha256": order_sha256(semantic_indices),
                "status": "TRIPLET_EXECUTED",
                "sealed_at_fork": acq["sealed_at_fork"],
                "arms": {
                    "NARROW_HIGH": narrow_high,
                    "NARROW_LOW": narrow_low,
                    "AUGMENTED_HIGH_REFERENCE": augmented_high,
                },
            })
            writer.save("ARK-015_PARTIAL.json", {"acquisitions": acquisitions, "results": results})

    payload = {
        "status": "EXECUTED_OR_BUDGETED_PARTIAL",
        "binding_manifest_sha256": manifest["manifest_sha256"],
        "acquisition_seeds": ACQ_SEEDS,
        "continuation_seeds": CONT_SEEDS,
        "acquisitions": acquisitions,
        "results": results,
        "summary": summarize(acquisitions, results),
    }
    writer.save("ARK-015_RESULT.json", payload)
    return payload
