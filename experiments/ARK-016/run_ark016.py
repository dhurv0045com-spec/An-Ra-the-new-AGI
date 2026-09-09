from __future__ import annotations

import statistics
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments" / "COLAB"))

from discovery_v7_common import (
    ARK016_PLAN_SHA,
    CANONICAL_T2_SHA,
    ReceiptWriter,
    RunContext,
    bind_ark11_runtime,
    flat_params,
    gradient_norm_and_clip,
    load_ark11,
    optimizer_step_with_delta,
    order_sha256,
    parameter_sha,
    sha_json,
)

RUNNER_PATH = Path(__file__)
ACQ_SEEDS = [1919, 2020, 2121]
CONT_SEEDS = [9901, 9902, 9903, 9904]
FORK_STEPS = 6000
EVAL_EVERY = 200


def sealed_recollapse_metrics(trajectory: list[dict], key: str = "sealed_exact") -> dict:
    if not trajectory:
        return {"status": "EMPTY"}
    vals = [float(x[key]) for x in trajectory]
    evals = [(int(x["step"]), 1.0 if float(x[key]) < 0.90 else 0.0) for x in trajectory]
    onset, confirm = _detect_boolean_streak(evals)
    return {
        "RET90": sum(v >= 0.90 for v in vals) / len(vals),
        "AREA": sum(vals) / len(vals),
        "FINAL": vals[-1],
        "PEAK": max(vals),
        "T_RECOLLAPSE_ONSET": onset,
        "T_RECOLLAPSE_CONFIRM": confirm,
        "recollapsed": confirm is not None,
    }


def _detect_boolean_streak(evals: list[tuple[int, float]]) -> tuple[int | None, int | None]:
    streak = 0
    onset = None
    for step, hit in evals:
        if hit >= 1.0:
            if streak == 0:
                onset = step
            streak += 1
            if streak >= 3:
                return onset, step
        else:
            streak = 0
            onset = None
    return None, None


def evaluate_snapshot(ark11, snapshot: dict, rows) -> float:
    vocab, model, _ = ark11.load_fork(snapshot, 1e-3)
    exact, _ = ark11.greedy_exact(model, vocab, rows, ark11.dev())
    return float(exact)


def run_post_recovery_arm(
    ark11,
    *,
    snapshot: dict,
    reference_flat_cpu: torch.Tensor,
    indices,
    offset: int,
    train,
    control,
    sealed,
    arm: str,
    lr: float,
    cap_trace: list[float] | None = None,
    cap_multiplier: float | None = None,
    record_delta_trace: bool = False,
) -> dict:
    if offset + FORK_STEPS > len(indices):
        raise RuntimeError(f"ARK-016 {arm}: continuation tail too short")
    if cap_trace is not None and len(cap_trace) != FORK_STEPS:
        raise RuntimeError(f"ARK-016 {arm}: cap trace length {len(cap_trace)} != {FORK_STEPS}")
    if (cap_trace is None) != (cap_multiplier is None):
        raise RuntimeError(f"ARK-016 {arm}: cap trace/multiplier must be supplied together")

    vocab, model, optimizer = ark11.load_fork(snapshot, lr)
    reference = reference_flat_cpu.to(ark11.dev())
    trajectory = []
    supervised_tokens = 0
    cumulative_path = 0.0
    raw_path = 0.0
    grad_sum = 0.0
    grad_max = 0.0
    cap_fires = 0
    applied_delta_trace: list[float] = []

    for step in range(1, FORK_STEPS + 1):
        rows = [train[i] for i in indices[offset + step - 1]]
        loss, count = ark11.loss_and_positions(model, vocab, rows, ark11.dev())
        if not torch.isfinite(loss):
            raise RuntimeError(f"ARK-016 {arm}: nonfinite loss step={step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = gradient_norm_and_clip(model, 1.0)
        cap = None
        if cap_trace is not None:
            cap = float(cap_multiplier) * float(cap_trace[step - 1])
        delta = optimizer_step_with_delta(model, optimizer, cap_norm=cap)

        supervised_tokens += int(count.item())
        raw_path += float(delta["raw_delta_norm"])
        cumulative_path += float(delta["applied_delta_norm"])
        grad_sum += grad_norm
        grad_max = max(grad_max, grad_norm)
        cap_fires += int(bool(delta["cap_fired"]))
        if record_delta_trace:
            applied_delta_trace.append(float(delta["applied_delta_norm"]))

        if step % EVAL_EVERY == 0:
            control_exact, _ = ark11.greedy_exact(model, vocab, control, ark11.dev())
            sealed_exact, _ = ark11.greedy_exact(model, vocab, sealed, ark11.dev())
            current = flat_params(model)
            l2 = float((current - reference).norm().item())
            trajectory.append({
                "step": step,
                "absolute_continuation_step": offset + step,
                "control_exact": float(control_exact),
                "sealed_exact": float(sealed_exact),
                "lr": lr,
                "loss": float(loss.detach().item()),
                "relative_displacement": l2 / max(float(reference.norm().item()), 1e-12),
                "cumulative_applied_path_length": cumulative_path,
                "cumulative_raw_path_length": raw_path,
                "cap_fired_fraction_so_far": cap_fires / step,
            })

    out = {
        "arm": arm,
        "lr": lr,
        "supervised_tokens": supervised_tokens,
        "cumulative_applied_path_length": cumulative_path,
        "cumulative_raw_path_length": raw_path,
        "mean_applied_step_delta_norm": cumulative_path / FORK_STEPS,
        "mean_raw_step_delta_norm": raw_path / FORK_STEPS,
        "mean_preclip_gradient_norm": grad_sum / FORK_STEPS,
        "max_preclip_gradient_norm": grad_max,
        "cap_fired_fraction": cap_fires / FORK_STEPS,
        "cap_multiplier": cap_multiplier,
        "final_parameter_sha256": parameter_sha(model),
        "control_retention": sealed_recollapse_metrics(trajectory, "control_exact"),
        "sealed_retention": sealed_recollapse_metrics(trajectory, "sealed_exact"),
        "trajectory": trajectory,
    }
    if record_delta_trace:
        out["applied_delta_trace"] = applied_delta_trace
    return out


def summarize(results: list[dict]) -> dict:
    qualified = [
        r for r in results
        if r.get("status") == "FORK_EXECUTED" and float(r.get("sealed_at_recovery_fork", 0.0)) >= 0.90
    ]
    n = len(qualified)
    seeds = sorted({int(r["acquisition_seed"]) for r in qualified})
    arms = ["LOW_REFERENCE", "HIGH_UNCAPPED", "HIGH_CAP_1X", "HIGH_CAP_10X"]

    fail_counts = {
        arm: sum(bool(r["arms"][arm]["sealed_retention"]["recollapsed"]) for r in qualified)
        for arm in arms
    }
    risks = {arm: (fail_counts[arm] / n if n else None) for arm in arms}
    mean_ret90 = {
        arm: (
            sum(float(r["arms"][arm]["sealed_retention"]["RET90"]) for r in qualified) / n
            if n else None
        )
        for arm in arms
    }
    mean_paths = {
        arm: (
            sum(float(r["arms"][arm]["cumulative_applied_path_length"]) for r in qualified) / n
            if n else None
        )
        for arm in arms
    }
    median_paths = {
        arm: (
            statistics.median(float(r["arms"][arm]["cumulative_applied_path_length"]) for r in qualified)
            if n else None
        )
        for arm in arms
    }

    cap1_ratios = []
    cap10_ratios = []
    for r in qualified:
        low_path = float(r["arms"]["LOW_REFERENCE"]["cumulative_applied_path_length"])
        if low_path > 0:
            cap1_ratios.append(float(r["arms"]["HIGH_CAP_1X"]["cumulative_applied_path_length"]) / low_path)
            cap10_ratios.append(float(r["arms"]["HIGH_CAP_10X"]["cumulative_applied_path_length"]) / low_path)

    grouped = {}
    for seed in seeds:
        rows = [r for r in qualified if int(r["acquisition_seed"]) == seed]
        grouped[str(seed)] = {
            "n": len(rows),
            **{
                f"{arm}_failures": sum(bool(r["arms"][arm]["sealed_retention"]["recollapsed"]) for r in rows)
                for arm in arms
            },
        }

    event_sufficient = n >= 4 and len(seeds) >= 2 and fail_counts["HIGH_UNCAPPED"] >= 2
    flags: list[str] = []

    if event_sufficient:
        high_risk = float(risks["HIGH_UNCAPPED"])
        low_risk = float(risks["LOW_REFERENCE"])
        cap1_risk = float(risks["HIGH_CAP_1X"])
        cap10_risk = float(risks["HIGH_CAP_10X"])
        med_cap1_ratio = statistics.median(cap1_ratios) if cap1_ratios else None
        med_cap10_ratio = statistics.median(cap10_ratios) if cap10_ratios else None
        mean_cap10_ratio = (
            float(mean_paths["HIGH_CAP_10X"]) / max(float(mean_paths["LOW_REFERENCE"]), 1e-30)
            if mean_paths["HIGH_CAP_10X"] is not None and mean_paths["LOW_REFERENCE"] is not None
            else None
        )

        if (
            high_risk - cap1_risk >= 0.30
            and abs(cap1_risk - low_risk) <= 0.20
            and med_cap1_ratio is not None
            and 0.5 <= med_cap1_ratio <= 1.5
        ):
            flags.append("UPDATE_MAGNITUDE_MAJOR_MEDIATOR")

        cap10_worse_groups = sum(
            v["HIGH_CAP_10X_failures"] > v["HIGH_UNCAPPED_failures"]
            for v in grouped.values()
        )
        majority_cap10_worse = cap10_worse_groups >= (len(grouped) // 2 + 1)
        if (
            high_risk - cap10_risk >= 0.30
            and med_cap10_ratio is not None
            and med_cap10_ratio >= 3.0
            and mean_cap10_ratio is not None
            and mean_cap10_ratio >= 3.0
            and float(mean_ret90["HIGH_CAP_10X"]) >= float(mean_ret90["LOW_REFERENCE"]) - 0.10
            and not majority_cap10_worse
        ):
            flags.append("TRUST_REGION_CANDIDATE")

        if (
            low_risk <= 0.25
            and cap1_risk - low_risk >= 0.30
            and med_cap1_ratio is not None
            and med_cap1_ratio <= 1.5
        ):
            flags.append("LOW_LR_SPECIFIC_BEYOND_STEP_NORM")

    if not event_sufficient:
        primary_verdict = "INCONCLUSIVE_LOW_EVENT_RATE"
    elif "TRUST_REGION_CANDIDATE" in flags:
        primary_verdict = "TRUST_REGION_CANDIDATE"
    elif "LOW_LR_SPECIFIC_BEYOND_STEP_NORM" in flags:
        primary_verdict = "LOW_LR_SPECIFIC_BEYOND_STEP_NORM"
    elif "UPDATE_MAGNITUDE_MAJOR_MEDIATOR" in flags:
        primary_verdict = "UPDATE_MAGNITUDE_MAJOR_MEDIATOR"
    else:
        primary_verdict = "MECHANISM_MIXED_OR_UNRESOLVED"

    return {
        "sealed_qualified_recovery_forks": n,
        "independent_acquisition_seeds": len(seeds),
        "failure_counts": fail_counts,
        "risks": risks,
        "mean_RET90": mean_ret90,
        "mean_cumulative_applied_path": mean_paths,
        "median_cumulative_applied_path": median_paths,
        "median_cap1_to_low_path_ratio": statistics.median(cap1_ratios) if cap1_ratios else None,
        "median_cap10_to_low_path_ratio": statistics.median(cap10_ratios) if cap10_ratios else None,
        "by_acquisition_seed": grouped,
        "event_sufficient": event_sufficient,
        "mechanism_flags": flags,
        "primary_verdict": primary_verdict,
    }


def evaluate_snapshot(ark11, snapshot: dict, rows) -> float:
    vocab, model, _ = ark11.load_fork(snapshot, 1e-3)
    exact, _ = ark11.greedy_exact(model, vocab, rows, ark11.dev())
    return float(exact)


def run_campaign(ctx: RunContext) -> dict:
    writer = ReceiptWriter(ctx, experiment_id="ARK-016", plan_sha=ARK016_PLAN_SHA, runner_path=RUNNER_PATH)
    ark11 = load_ark11()
    bind_ark11_runtime(ark11, ctx.device, ctx.head)

    manifest = ark11.load_manifest()
    if manifest.get("split_sha256") != CANONICAL_T2_SHA:
        raise RuntimeError("ARK-016 canonical T2 manifest drift")
    train = [(p, a) for p, a in manifest["train"]]
    test = [(p, a) for p, a in manifest["test"]]
    control, sealed, split_manifest = ark11.build_control_sealed_split(test)
    writer.save("ARK-016_TASK_MANIFEST.json", {
        "canonical_t2_split_sha256": manifest["split_sha256"],
        "control_sealed_split": split_manifest,
    })

    acquisitions: list[dict] = []
    runtime_acq: dict[int, dict] = {}
    results: list[dict] = []
    cap_traces: dict[str, dict] = {}

    # Acquire every fresh parent first so later event work is breadth-first across
    # independent parents rather than exhausting the budget on one seed.
    for seed in ACQ_SEEDS:
        if ctx.minutes_left < 22:
            acquisitions.append({"seed": seed, "status": "BUDGET_BLOCKED"})
            writer.save("ARK-016_PARTIAL.json", {"acquisitions": acquisitions, "results": results})
            continue
        print(f"\n=== ARK-016 acquire T2 seed={seed} ===", flush=True)
        acq = ark11.acquire(seed, train, control)
        if acq["status"] == "ACQUIRED":
            runtime_acq[seed] = acq
        acquisitions.append({
            "seed": seed,
            "status": acq["status"],
            "control_g90_onset": acq.get("onset_step"),
            "control_g90_confirm": acq.get("confirmation_step"),
            "supervised_tokens": acq.get("supervised_tokens"),
        })
        writer.save("ARK-016_PARTIAL.json", {"acquisitions": acquisitions, "results": results})

    # Breadth-first continuation orders preserve acquisition-seed diversity under
    # a finite Colab budget. A started recovery fork always completes all four arms.
    for order_seed in CONT_SEEDS:
        for seed in ACQ_SEEDS:
            acq = runtime_acq.get(seed)
            if acq is None:
                continue
            if ctx.minutes_left < 20:
                results.append({
                    "acquisition_seed": seed,
                    "order_seed": order_seed,
                    "status": "BUDGET_BLOCKED_BEFORE_EVENT",
                })
                writer.save("ARK-016_PARTIAL.json", {"acquisitions": acquisitions, "results": results})
                continue

            print(f"\n--- ARK-016 seed={seed} order={order_seed} ---", flush=True)
            indices = ark11.generate_continuation_indices(order_seed, 16000, 64, len(train))
            full_order_hash = order_sha256(indices)

            collapse = ark11.run_to_threshold(
                phase_name=f"ARK016_COLLAPSE_s{seed}_o{order_seed}",
                snapshot=acq["snapshot"],
                indices=indices,
                offset=0,
                max_steps=6000,
                train=train,
                control=control,
                lr=1e-3,
                bar=0.90,
                below=True,
            )
            if collapse["status"] != "TRIGGERED":
                results.append({
                    "acquisition_seed": seed,
                    "order_seed": order_seed,
                    "continuation_order_sha256": full_order_hash,
                    "status": "NO_CONTROL_COLLAPSE",
                    "collapse": {k: v for k, v in collapse.items() if k != "snapshot"},
                })
                writer.save("ARK-016_PARTIAL.json", {"acquisitions": acquisitions, "results": results})
                continue

            recovery = ark11.run_to_threshold(
                phase_name=f"ARK016_RECOVERY_s{seed}_o{order_seed}",
                snapshot=collapse["snapshot"],
                indices=indices,
                offset=int(collapse["confirmation_absolute_step"]),
                max_steps=4000,
                train=train,
                control=control,
                lr=1e-3,
                bar=0.90,
                below=False,
            )
            if recovery["status"] != "TRIGGERED":
                results.append({
                    "acquisition_seed": seed,
                    "order_seed": order_seed,
                    "continuation_order_sha256": full_order_hash,
                    "status": "NO_CONTROL_RECOVERY",
                    "collapse": {k: v for k, v in collapse.items() if k != "snapshot"},
                    "recovery": {k: v for k, v in recovery.items() if k != "snapshot"},
                })
                writer.save("ARK-016_PARTIAL.json", {"acquisitions": acquisitions, "results": results})
                continue

            fork_offset = int(recovery["confirmation_absolute_step"])
            sealed_at_fork = evaluate_snapshot(ark11, recovery["snapshot"], sealed)
            _, recovery_model, _ = ark11.load_fork(recovery["snapshot"], 1e-3)
            recovery_flat = flat_params(recovery_model).detach().cpu()

            common = {
                "ark11": ark11,
                "snapshot": recovery["snapshot"],
                "reference_flat_cpu": recovery_flat,
                "indices": indices,
                "offset": fork_offset,
                "train": train,
                "control": control,
                "sealed": sealed,
            }

            low = run_post_recovery_arm(
                **common,
                arm="LOW_REFERENCE",
                lr=1e-5,
                record_delta_trace=True,
            )
            low_trace = list(low.pop("applied_delta_trace"))
            trace_hash = sha_json(low_trace)

            high = run_post_recovery_arm(
                **common,
                arm="HIGH_UNCAPPED",
                lr=1e-3,
            )
            cap1 = run_post_recovery_arm(
                **common,
                arm="HIGH_CAP_1X",
                lr=1e-3,
                cap_trace=low_trace,
                cap_multiplier=1.0,
            )
            cap10 = run_post_recovery_arm(
                **common,
                arm="HIGH_CAP_10X",
                lr=1e-3,
                cap_trace=low_trace,
                cap_multiplier=10.0,
            )

            event_key = f"s{seed}_o{order_seed}"
            cap_traces[event_key] = {
                "acquisition_seed": seed,
                "order_seed": order_seed,
                "low_delta_trace_sha256": trace_hash,
                "low_applied_delta_trace": low_trace,
            }
            writer.save("ARK-016_CAP_TRACES.json", {"traces": cap_traces})

            results.append({
                "acquisition_seed": seed,
                "order_seed": order_seed,
                "continuation_order_sha256": full_order_hash,
                "status": "FORK_EXECUTED",
                "collapse": {k: v for k, v in collapse.items() if k != "snapshot"},
                "recovery": {k: v for k, v in recovery.items() if k != "snapshot"},
                "sealed_at_recovery_fork": sealed_at_fork,
                "fork_absolute_continuation_step": fork_offset,
                "low_delta_trace_sha256": trace_hash,
                "arms": {
                    "LOW_REFERENCE": low,
                    "HIGH_UNCAPPED": high,
                    "HIGH_CAP_1X": cap1,
                    "HIGH_CAP_10X": cap10,
                },
            })
            writer.save("ARK-016_PARTIAL.json", {"acquisitions": acquisitions, "results": results})

    payload = {
        "status": "EXECUTED_OR_BUDGETED_PARTIAL",
        "canonical_t2_split_sha256": manifest["split_sha256"],
        "control_sealed_assignment_sha256": split_manifest["assignment_sha256"],
        "acquisition_seeds": ACQ_SEEDS,
        "continuation_seeds": CONT_SEEDS,
        "acquisitions": acquisitions,
        "results": results,
        "summary": summarize(results),
    }
    writer.save("ARK-016_RESULT.json", payload)
    return payload
