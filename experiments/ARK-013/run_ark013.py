from __future__ import annotations

import hashlib
import random
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
    sha_json,
    trajectory_metrics,
)

PLAN_SHA = "fefbe39350d39e25e998a24e81e1caf0090ea8e4"
ADDENDUM_SHA = "8d086a501ef579574947b506fec6e0c71bd80711"
RUNNER_PATH = Path(__file__)
ACQ_SEEDS = [1717, 1818]
ORDER_SEEDS = [6801, 6802]
ARM_NAMES = {"FIXED_HIGH", "FIXED_LOW", "ADAPTIVE_HIGH_LOW"}
FIXED_HORIZON = 12000


def _complete_triplet(row: dict) -> bool:
    if row.get("acquisition_seed") not in ACQ_SEEDS or row.get("order_seed") not in ORDER_SEEDS:
        return False
    arms = row.get("arms")
    if row.get("status") != "TRIPLET_EXECUTED" or not isinstance(arms, dict) or set(arms) != ARM_NAMES:
        return False
    for arm in arms.values():
        if not isinstance(arm, dict) or arm.get("completed_steps") != FIXED_HORIZON:
            return False
        for key in ("t3_control_metrics", "t3_sealed_metrics", "t2_control_metrics", "t2_sealed_metrics"):
            metrics = arm.get(key)
            if not isinstance(metrics, dict) or not all(k in metrics for k in ("AREA", "FINAL")):
                return False
    return True


def _pair_from_prompt(prompt: str) -> tuple[int, int]:
    left = prompt.split("=")[0]
    a_s, b_s = left.split("+")
    return int(a_s.strip()), int(b_s.strip())


def _row_hash(row) -> str:
    return hashlib.sha256((row[0] + "\0" + row[1]).encode("utf-8")).hexdigest()


def build_t3carry_manifest() -> tuple[list[tuple[str, str]], list[tuple[str, str]], list[tuple[str, str]], dict]:
    def candidates(ta_values):
        rows = []
        for ta in ta_values:
            for tb in range(1, 10):
                for ua in range(10):
                    for ub in range(10):
                        if ua + ub < 10:
                            continue
                        a = ta * 10 + ua
                        b = tb * 10 + ub
                        rows.append((f"{a} + {b} = ", str(a + b)))
        return rows

    train_candidates = candidates(range(1, 6))
    ood_candidates = candidates(range(6, 8))
    rng_train = random.Random(131313)
    rng_ood = random.Random(131314)
    rng_train.shuffle(train_candidates)
    rng_ood.shuffle(ood_candidates)
    train = train_candidates[:500]
    train_pairs = {tuple(sorted(_pair_from_prompt(p))) for p, _ in train}
    ood = []
    excluded = 0
    for row in ood_candidates:
        pair = tuple(sorted(_pair_from_prompt(row[0])))
        if pair in train_pairs:
            excluded += 1
            continue
        ood.append(row)
        if len(ood) == 200:
            break
    if len(ood) != 200:
        raise RuntimeError(f"T3CARRY OOD feasibility failure: {len(ood)}")

    control = []
    sealed = []
    band_counts = {}
    for band in [6, 7]:
        band_rows = [r for r in ood if _pair_from_prompt(r[0])[0] // 10 == band]
        ordered = sorted(band_rows, key=_row_hash)
        c = [r for i, r in enumerate(ordered) if i % 2 == 0]
        s = [r for i, r in enumerate(ordered) if i % 2 == 1]
        control.extend(c)
        sealed.extend(s)
        band_counts[str(band)] = {"source": len(ordered), "control": len(c), "sealed": len(s)}

    if set(control) & set(sealed):
        raise RuntimeError("T3 CONTROL/SEALED overlap")
    if set(control) | set(sealed) != set(ood):
        raise RuntimeError("T3 CONTROL/SEALED union mismatch")

    for prompt, _ in train + ood:
        a, b = _pair_from_prompt(prompt)
        if a % 10 + b % 10 < 10:
            raise RuntimeError("T3 forced-carry invariant violated")

    manifest = {
        "schema": "arkenstone-ark013-t3carry/v1",
        "dataset_seed_train": 131313,
        "dataset_seed_ood": 131314,
        "semantics": "two-digit operands; forced ones-column carry; final answer may be 2 or 3 digits per preexecution addendum",
        "train_a_tens_band": [1, 5],
        "ood_a_tens_band": [6, 7],
        "counts": {"train": len(train), "ood": len(ood), "control": len(control), "sealed": len(sealed)},
        "commutation_excluded": excluded,
        "commutation_overlap": len(
            {tuple(sorted(_pair_from_prompt(p))) for p, _ in train}
            & {tuple(sorted(_pair_from_prompt(p))) for p, _ in ood}
        ),
        "band_counts": band_counts,
        "train_sha256": sha_json(train),
        "ood_sha256": sha_json(ood),
        "control_sha256": sha_json(control),
        "sealed_sha256": sha_json(sealed),
        "train": [list(x) for x in train],
        "ood": [list(x) for x in ood],
        "control": [list(x) for x in control],
        "sealed": [list(x) for x in sealed],
    }
    manifest["manifest_sha256"] = sha_json({k: v for k, v in manifest.items() if k != "manifest_sha256"})
    if manifest["commutation_overlap"] != 0:
        raise RuntimeError("T3 commutation overlap survived")
    return train, control, sealed, manifest


def _run_arm(ark11, *, snapshot, t3_indices, t3_train, t3_control, t3_sealed,
             t2_control, t2_sealed, arm: str, steps: int = 12000,
             ctx: RunContext | None = None) -> dict:
    initial_lr = 1e-5 if arm == "FIXED_LOW" else 1e-3
    vocab, model, optimizer = ark11.load_fork(snapshot, initial_lr)
    reference = flat_params(model).detach().clone()
    t3_evals: list[tuple[int, float]] = []
    switched = arm == "FIXED_LOW"
    switch_onset = 0 if switched else None
    switch_confirm = 0 if switched else None
    trajectory = []
    supervised_tokens = 0

    for step in range(1, steps + 1):
        if ctx is not None:
            ensure_budget(ctx)
        rows = [t3_train[i] for i in t3_indices[step - 1]]
        loss, count = ark11.loss_and_positions(model, vocab, rows, ark11.dev())
        if not torch.isfinite(loss):
            raise RuntimeError(f"{arm}: nonfinite loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        supervised_tokens += int(count.item())

        if step % 200 == 0:
            t3_c, _ = ark11.greedy_exact(model, vocab, t3_control, ark11.dev())
            t3_s, _ = ark11.greedy_exact(model, vocab, t3_sealed, ark11.dev())
            t2_c, _ = ark11.greedy_exact(model, vocab, t2_control, ark11.dev())
            t2_s, _ = ark11.greedy_exact(model, vocab, t2_sealed, ark11.dev())
            current = flat_params(model)
            disp = float((current - reference).norm().item())
            trajectory.append({
                "step": step,
                "t3_control_exact": t3_c,
                "t3_sealed_exact": t3_s,
                "t2_control_exact": t2_c,
                "t2_sealed_exact": t2_s,
                "lr": float(optimizer.param_groups[0]["lr"]),
                "relative_displacement": disp / max(float(reference.norm().item()), 1e-12),
                "loss": float(loss.detach().item()),
            })

            if arm == "ADAPTIVE_HIGH_LOW" and not switched:
                t3_evals.append((step, t3_c))
                onset, confirm = detect_sustained(t3_evals, 0.90, 3)
                if confirm is not None:
                    switch_onset = onset
                    switch_confirm = confirm
                    for group in optimizer.param_groups:
                        group["lr"] = 1e-5
                    switched = True

    return {
        "arm": arm,
        "switched": switched,
        "switch_onset_step": switch_onset,
        "switch_confirmation_step": switch_confirm,
        "supervised_tokens": supervised_tokens,
        "t3_control_metrics": trajectory_metrics(trajectory, "t3_control_exact"),
        "t3_sealed_metrics": trajectory_metrics(trajectory, "t3_sealed_exact"),
        "t2_control_metrics": trajectory_metrics(trajectory, "t2_control_exact"),
        "t2_sealed_metrics": trajectory_metrics(trajectory, "t2_sealed_exact"),
        "trajectory": trajectory,
        "completed_steps": steps,
    }


def _summarize(results: list[dict]) -> dict:
    candidates = [r for r in results if _complete_triplet(r)]
    identities = [(r["acquisition_seed"], r["order_seed"]) for r in candidates]
    triplets = [r for r, identity in zip(candidates, identities) if identities.count(identity) == 1]
    summary = {
        "matched_triplets": len(triplets),
        "excluded_rows": len(results) - len(triplets),
        "rule_status": "CONSERVATIVE_FINAL_AND_AREA_AGREEMENT_NOT_NEW_PREREGISTRATION",
    }
    if not triplets:
        summary["verdict"] = "INCONCLUSIVE_NO_MATCHED_TRIPLETS"
        return summary

    high_acquired = [
        r for r in triplets
        if r["arms"]["FIXED_HIGH"]["t3_control_metrics"].get("G90_CONFIRM") is not None
    ]
    summary["fixed_high_t3_acquired_triplets"] = len(high_acquired)
    seeds_high_acquired = sorted({r["acquisition_seed"] for r in high_acquired})
    summary["fixed_high_t3_acquired_seeds"] = seeds_high_acquired

    summary["plasticity_cost_paired_deltas"] = [
        {
            "acquisition_seed": r["acquisition_seed"],
            "order_seed": r["order_seed"],
            "t3_final_low_minus_high": r["arms"]["FIXED_LOW"]["t3_sealed_metrics"]["FINAL"] - r["arms"]["FIXED_HIGH"]["t3_sealed_metrics"]["FINAL"],
            "t2_final_low_minus_high": r["arms"]["FIXED_LOW"]["t2_sealed_metrics"]["FINAL"] - r["arms"]["FIXED_HIGH"]["t2_sealed_metrics"]["FINAL"],
        }
        for r in triplets
    ]
    summary["plasticity_cost_classification"] = "UNRESOLVED_MATERIALITY_NOT_QUANTIFIED_IN_PLAN"

    adaptive_wins = []
    for r in triplets:
        hi = r["arms"]["FIXED_HIGH"]
        ad = r["arms"]["ADAPTIVE_HIGH_LOW"]
        # Conservative implementation interpretation of the frozen wording:
        # final and area must agree in the same direction. This is not a new
        # preregistered law.
        final_a = (ad["t3_sealed_metrics"]["FINAL"] >= hi["t3_sealed_metrics"]["FINAL"] - 0.05
                   and ad["t2_sealed_metrics"]["FINAL"] >= hi["t2_sealed_metrics"]["FINAL"] + 0.10)
        area_a = (ad["t3_sealed_metrics"]["AREA"] >= hi["t3_sealed_metrics"]["AREA"] - 0.05
                  and ad["t2_sealed_metrics"]["AREA"] >= hi["t2_sealed_metrics"]["AREA"] + 0.10)
        final_b = (ad["t3_sealed_metrics"]["FINAL"] >= hi["t3_sealed_metrics"]["FINAL"] + 0.10
                   and ad["t2_sealed_metrics"]["FINAL"] >= hi["t2_sealed_metrics"]["FINAL"] - 0.05)
        area_b = (ad["t3_sealed_metrics"]["AREA"] >= hi["t3_sealed_metrics"]["AREA"] + 0.10
                  and ad["t2_sealed_metrics"]["AREA"] >= hi["t2_sealed_metrics"]["AREA"] - 0.05)
        if (final_a and area_a) or (final_b and area_b):
            adaptive_wins.append((r["acquisition_seed"], r["order_seed"]))
    summary["adaptive_pareto_win_triplets"] = adaptive_wins
    summary["adaptive_pareto_seeds"] = sorted({s for s, _ in adaptive_wins})

    if len(triplets) == len(ACQ_SEEDS) * len(ORDER_SEEDS) and not high_acquired:
        verdict = "INCONCLUSIVE_NEW_SKILL_NOT_ACQUIRED"
    elif len(adaptive_wins) >= 2 and len({s for s, _ in adaptive_wins}) >= 2:
        verdict = "ADAPTIVE_PARETO_IMPROVEMENT"
    elif len(triplets) < len(ACQ_SEEDS) * len(ORDER_SEEDS):
        verdict = "INCONCLUSIVE_PARTIAL_TRIPLETS"
    else:
        verdict = "ADAPTIVE_NO_BENEFIT"
    summary["verdict"] = verdict
    return summary


def run_campaign(ctx: RunContext) -> dict:
    writer = ReceiptWriter(
        ctx,
        experiment_id="ARK-013",
        plan_sha=PLAN_SHA,
        runner_path=RUNNER_PATH,
        extra_plan_shas={"preexecution_addendum_commit_sha": ADDENDUM_SHA},
    )
    if ctx.minutes_left < 50:
        payload = {"status": "BUDGET_BLOCKED", "acquisitions": [{"seed": s, "status": "BUDGET_BLOCKED"} for s in ACQ_SEEDS], "results": [{"acquisition_seed": s, "order_seed": o, "status": "BUDGET_BLOCKED"} for s in ACQ_SEEDS for o in ORDER_SEEDS], "summary": {"matched_triplets": 0, "verdict": "BUDGET_BLOCKED"}}
        writer.save("ARK-013_RESULT.json", payload)
        return payload
    ark11 = load_ark11()
    bind_ark11_runtime(ark11, ctx.device, ctx.head, ctx=ctx)

    t2_manifest = ark11.load_manifest()
    t2_train = [(p, a) for p, a in t2_manifest["train"]]
    t2_test = [(p, a) for p, a in t2_manifest["test"]]
    t2_control, t2_sealed, t2_split = ark11.build_control_sealed_split(t2_test)

    t3_train, t3_control, t3_sealed, t3_manifest = build_t3carry_manifest()
    writer.save("ARK-013_TASK_MANIFEST.json", t3_manifest)

    results = []
    acquisitions = []
    for seed in ACQ_SEEDS:
        if ctx.minutes_left < 18:
            acquisitions.append({"seed": seed, "status": "BUDGET_BLOCKED"})
            continue

        print(f"\n=== ARK-013 acquire T2 seed={seed} ===", flush=True)
        acq = ark11.acquire(seed, t2_train, t2_control)
        acquisitions.append({
            "seed": seed,
            "status": acq["status"],
            "control_g90_onset": acq.get("onset_step"),
            "control_g90_confirm": acq.get("confirmation_step"),
            "supervised_tokens": acq.get("supervised_tokens"),
        })
        writer.save("ARK-013_PARTIAL.json", {"acquisitions": acquisitions, "results": results})
        if acq["status"] != "ACQUIRED":
            continue

        for order_seed in ORDER_SEEDS:
            if ctx.minutes_left < 18:
                results.append({
                    "acquisition_seed": seed,
                    "order_seed": order_seed,
                    "status": "BUDGET_BLOCKED_BEFORE_TRIPLET",
                })
                continue

            print(f"\n--- ARK-013 seed={seed} T3 order={order_seed} ---", flush=True)
            indices = ark11.generate_continuation_indices(order_seed, 12000, 64, len(t3_train))
            arms = {}
            triplet = {
                "acquisition_seed": seed,
                "order_seed": order_seed,
                "order_sha256": order_sha256(indices),
                "status": "ARMS_PARTIAL",
                "arms": arms,
            }
            results.append(triplet)
            for arm in ["FIXED_HIGH", "FIXED_LOW", "ADAPTIVE_HIGH_LOW"]:
                arms[arm] = _run_arm(
                    ark11,
                    snapshot=acq["snapshot"],
                    t3_indices=indices,
                    t3_train=t3_train,
                    t3_control=t3_control,
                    t3_sealed=t3_sealed,
                    t2_control=t2_control,
                    t2_sealed=t2_sealed,
                    arm=arm,
                    steps=FIXED_HORIZON,
                    ctx=ctx,
                )
                writer.save("ARK-013_PARTIAL.json", {"acquisitions": acquisitions, "results": results})
            triplet["status"] = "TRIPLET_EXECUTED"
            writer.save("ARK-013_PARTIAL.json", {"acquisitions": acquisitions, "results": results})

    payload = {
        "status": "EXECUTED_OR_BUDGETED_PARTIAL",
        "t2_manifest_sha256": t2_manifest["split_sha256"],
        "t2_controller_sealed_split": t2_split,
        "t3_manifest_sha256": t3_manifest["manifest_sha256"],
        "acquisitions": acquisitions,
        "results": results,
        "summary": _summarize(results),
    }
    writer.save("ARK-013_RESULT.json", payload)
    return payload
