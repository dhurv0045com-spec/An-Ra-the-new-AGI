from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments" / "COLAB"))

from discovery_v6_common import (
    ReceiptWriter,
    RunContext,
    bind_ark11_runtime,
    detect_sustained,
    ensure_budget,
    flat_params,
    load_ark11,
    order_sha256,
    trajectory_metrics,
)

PLAN_SHA = "324b762b3cd3e362e1a28a34bd609c30fbc1f171"
RUNNER_PATH = Path(__file__)
SOURCES = [(909, 2702), (1010, 2702), (1111, 2703)]
SCHEDULES = [
    ("HIGH_CONTINUE", None),
    ("LOW_IMMEDIATE", 0.0),
    ("SWITCH_75", 0.75),
    ("SWITCH_85", 0.85),
    ("SWITCH_90", 0.90),
    ("SWITCH_95", 0.95),
]
SCHEDULE_NAMES = {name for name, _ in SCHEDULES}
FIXED_HORIZON = 8000


def _complete_source(row: dict) -> bool:
    if (row.get("acquisition_seed"), row.get("continuation_seed")) not in SOURCES:
        return False
    schedules = row.get("schedules")
    if row.get("status") != "EXECUTED" or not isinstance(schedules, dict):
        return False
    if set(schedules) != SCHEDULE_NAMES:
        return False
    for arm in schedules.values():
        if not isinstance(arm, dict) or arm.get("completed_steps") != FIXED_HORIZON:
            return False
        metrics = arm.get("sealed_metrics")
        if not isinstance(metrics, dict) or not all(k in metrics for k in ("AREA", "FINAL")):
            return False
    return True


def _run_schedule(ark11, *, snapshot, indices, offset, train, control, sealed,
                  name: str, threshold: float | None, steps: int = 8000,
                  ctx: RunContext | None = None) -> dict:
    if offset + steps > len(indices):
        raise RuntimeError(f"{name}: continuation stream too short")

    initial_lr = 1e-5 if name == "LOW_IMMEDIATE" else 1e-3
    vocab, model, optimizer = ark11.load_fork(snapshot, initial_lr)
    reference = flat_params(model).detach().clone()
    trajectory = []
    control_evals: list[tuple[int, float]] = []
    switched = name == "LOW_IMMEDIATE"
    switch_onset = 0 if switched else None
    switch_confirm = 0 if switched else None
    supervised_tokens = 0

    for step in range(1, steps + 1):
        if ctx is not None:
            ensure_budget(ctx)
        rows = [train[i] for i in indices[offset + step - 1]]
        loss, count = ark11.loss_and_positions(model, vocab, rows, ark11.dev())
        if not torch.isfinite(loss):
            raise RuntimeError(f"{name}: nonfinite loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        supervised_tokens += int(count.item())

        if step % 200 == 0:
            control_exact, _ = ark11.greedy_exact(model, vocab, control, ark11.dev())
            sealed_exact, _ = ark11.greedy_exact(model, vocab, sealed, ark11.dev())
            current = flat_params(model)
            displacement = float((current - reference).norm().item())
            trajectory.append({
                "step": step,
                "absolute_continuation_step": offset + step,
                "control_exact": control_exact,
                "sealed_exact": sealed_exact,
                "lr": float(optimizer.param_groups[0]["lr"]),
                "relative_displacement": displacement / max(float(reference.norm().item()), 1e-12),
                "loss": float(loss.detach().item()),
            })

            if threshold not in (None, 0.0) and not switched:
                control_evals.append((step, control_exact))
                onset, confirm = detect_sustained(control_evals, float(threshold), 3)
                if confirm is not None:
                    switch_onset = onset
                    switch_confirm = confirm
                    for group in optimizer.param_groups:
                        group["lr"] = 1e-5
                    switched = True

    return {
        "schedule": name,
        "threshold": threshold,
        "switched": switched,
        "switch_onset_step": switch_onset,
        "switch_confirmation_step": switch_confirm,
        "supervised_tokens": supervised_tokens,
        "completed_steps": steps,
        "control_metrics": trajectory_metrics(trajectory, "control_exact"),
        "sealed_metrics": trajectory_metrics(trajectory, "sealed_exact"),
        "trajectory": trajectory,
    }


def _summarize(rows: list[dict]) -> dict:
    candidates = [r for r in rows if _complete_source(r)]
    identities = [(r["acquisition_seed"], r["continuation_seed"]) for r in candidates]
    executed = [r for r, identity in zip(candidates, identities) if identities.count(identity) == 1]
    summary = {"event_sources_executed": len(executed), "selected_event_design": True}
    summary["rule_status"] = "CONSERVATIVE_OPERATIONAL_INTERPRETATION_NOT_NEW_PREREGISTRATION"
    summary["excluded_rows"] = len(rows) - len(executed)
    means = {}
    for name, _ in SCHEDULES:
        vals = [r["schedules"][name]["sealed_metrics"]["AREA"] for r in executed if name in r.get("schedules", {})]
        if vals:
            means[name] = sum(vals) / len(vals)
    summary["mean_sealed_area_by_schedule"] = means

    if len(executed) < 2:
        summary["verdict"] = "INCONCLUSIVE_LOW_EVENT_RATE"
        return summary

    needed = ["LOW_IMMEDIATE", "SWITCH_75", "SWITCH_85", "SWITCH_90", "SWITCH_95", "HIGH_CONTINUE"]
    if not all(k in means for k in needed):
        summary["verdict"] = "INCONCLUSIVE_PARTIAL_SCHEDULES"
        return summary

    ordered_sources = []
    late_gain_sources = []
    late_not_more_unstable = []
    for row in executed:
        arms = row["schedules"]
        switch_arms = [arms[f"SWITCH_{bar}"] for bar in (75, 85, 90, 95)]
        ordered_sources.append(all(
            switch_arms[i]["sealed_metrics"]["AREA"]
            <= switch_arms[i + 1]["sealed_metrics"]["AREA"] for i in range(3)
        ))
        late_gain_sources.append(
            max(a["sealed_metrics"]["AREA"] for a in switch_arms[2:])
            > arms["LOW_IMMEDIATE"]["sealed_metrics"]["AREA"]
        )
        # The same late-switch arm must both gain over LOW_IMMEDIATE and
        # avoid recurrent sealed instability. HIGH_CONTINUE collapse alone is
        # not evidence of protection.
        protected_late_gain = any(
            a["sealed_metrics"]["AREA"] > arms["LOW_IMMEDIATE"]["sealed_metrics"]["AREA"]
            and a["sealed_metrics"].get("G90_CONFIRM") is not None
            and a["sealed_metrics"].get("DROP90_CONFIRM") is None
            for a in switch_arms[2:]
        )
        late_not_more_unstable.append(protected_late_gain)

    def dominates(candidate: str) -> bool:
        comparisons = [
            (row["schedules"][candidate]["sealed_metrics"]["AREA"],
             row["schedules"][other]["sealed_metrics"]["AREA"])
            for row in executed for other in needed if other != candidate
        ]
        return all(left >= right for left, right in comparisons) and any(left > right for left, right in comparisons)

    summary["state_ordered_by_source"] = ordered_sources
    summary["late_gain_over_immediate_by_source"] = late_gain_sources
    summary["late_not_more_unstable_than_high_by_source"] = late_not_more_unstable
    if all(ordered_sources) and all(late_gain_sources) and all(late_not_more_unstable):
        verdict = "STATE_THRESHOLD_SUPPORTED_SCREEN"
    elif dominates("LOW_IMMEDIATE"):
        verdict = "LOW_ALWAYS_BEST_SCREEN"
    elif dominates("HIGH_CONTINUE"):
        verdict = "HIGH_ALWAYS_BEST_SCREEN"
    else:
        verdict = "TIME_NOT_STATE_SCREEN"
    summary["verdict"] = verdict
    return summary


def run_campaign(ctx: RunContext) -> dict:
    writer = ReceiptWriter(ctx, experiment_id="ARK-012", plan_sha=PLAN_SHA, runner_path=RUNNER_PATH)
    if ctx.minutes_left < 35:
        payload = {"status": "BUDGET_BLOCKED", "results": [{"acquisition_seed": a, "continuation_seed": o, "status": "BUDGET_BLOCKED"} for a, o in SOURCES], "summary": {"event_sources_executed": 0, "verdict": "BUDGET_BLOCKED"}}
        writer.save("ARK-012_RESULT.json", payload)
        return payload
    ark11 = load_ark11()
    bind_ark11_runtime(ark11, ctx.device, ctx.head, ctx=ctx)

    manifest = ark11.load_manifest()
    train = [(p, a) for p, a in manifest["train"]]
    source_test = [(p, a) for p, a in manifest["test"]]
    control, sealed, split_manifest = ark11.build_control_sealed_split(source_test)

    results = []
    for acq_seed, order_seed in SOURCES:
        if ctx.minutes_left < 15:
            results.append({
                "acquisition_seed": acq_seed,
                "continuation_seed": order_seed,
                "status": "BUDGET_BLOCKED",
            })
            continue

        print(f"\n=== ARK-012 source seed={acq_seed} order={order_seed} ===", flush=True)
        acq = ark11.acquire(acq_seed, train, control)
        if acq["status"] != "ACQUIRED":
            results.append({
                "acquisition_seed": acq_seed,
                "continuation_seed": order_seed,
                "status": "BLOCKED_BY_ACQUISITION",
                "acquisition_status": acq["status"],
            })
            writer.save("ARK-012_PARTIAL.json", {"results": results})
            continue

        indices = ark11.generate_continuation_indices(order_seed, 18000, 64, len(train))
        order_hash = order_sha256(indices)
        collapse = ark11.run_to_threshold(
            phase_name=f"ARK012_COLLAPSE_s{acq_seed}_o{order_seed}",
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
                "acquisition_seed": acq_seed,
                "continuation_seed": order_seed,
                "continuation_order_sha256": order_hash,
                "status": "NO_CONTROL_COLLAPSE",
                "collapse": {k: v for k, v in collapse.items() if k != "snapshot"},
            })
            writer.save("ARK-012_PARTIAL.json", {"results": results})
            continue

        offset = int(collapse["confirmation_absolute_step"])
        schedules = {}
        event = {
            "acquisition_seed": acq_seed,
            "continuation_seed": order_seed,
            "continuation_order_sha256": order_hash,
            "status": "SCHEDULES_PARTIAL",
            "selection_note": "historical high-instability source; selected-event mechanistic screen only",
            "acquisition_control_g90_onset": acq["onset_step"],
            "acquisition_control_g90_confirm": acq["confirmation_step"],
            "collapse": {k: v for k, v in collapse.items() if k != "snapshot"},
            "schedules": schedules,
        }
        results.append(event)
        for name, threshold in SCHEDULES:
            schedules[name] = _run_schedule(
                ark11,
                snapshot=collapse["snapshot"],
                indices=indices,
                offset=offset,
                train=train,
                control=control,
                sealed=sealed,
                name=name,
                threshold=threshold,
                steps=FIXED_HORIZON,
                ctx=ctx,
            )
            writer.save("ARK-012_PARTIAL.json", {"results": results})
        event["status"] = "EXECUTED"
        writer.save("ARK-012_PARTIAL.json", {"results": results})

    payload = {
        "status": "EXECUTED_OR_BUDGETED_PARTIAL",
        "source_manifest_sha256": manifest["split_sha256"],
        "controller_sealed_split": split_manifest,
        "results": results,
        "summary": _summarize(results),
    }
    writer.save("ARK-012_RESULT.json", payload)
    return payload
