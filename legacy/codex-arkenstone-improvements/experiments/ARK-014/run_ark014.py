"""ARK-014 runner: order-robust non-arithmetic binding + LR-retention screen.

Implements the frozen contract in experiments/ARK-014/PLAN.md:

- matched acquisition arms CANONICAL_TRAIN and ORDER_AUGMENTED at seed 2201
  (identical initialization, optimizer, semantic minibatch stream and update
  budget; only the presented fact order differs);
- a qualification controller that reads BIND_CONTROL diagnostics only —
  BIND_SEALED is measured after qualification decisions, for analysis only;
- paired HIGH (1e-3) / LOW (1e-5) retention forks from the qualified snapshot
  for frozen continuation orders 7701, 7702, 7703, 6,000 updates each;
- plan verdicts, computed from completed matched evidence only.

Every receipt binds source identities, task manifest, stream hashes, measured
supervised positions/examples/steps and checkpoint hashes. Incomplete arms are
labeled explicitly and are never counted as comparisons.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import time
import traceback
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments" / "COLAB"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from discovery_v6_common import (  # noqa: E402
    BudgetExhausted, ReceiptWriter, RunContext, bind_ark11_runtime, cpu_tree, detect_sustained,
    ensure_budget, file_sha256, generate_indices, git_head, load_ark11, order_sha256,
    parameter_sha,
)
import ark014_binding as binding  # noqa: E402

PLAN_COMMIT_SHA = "59e01a33a11aea1222368553d0819764341e2577"
RUNNER_PATH = Path(__file__)

# Frozen experiment constants (experiments/ARK-014/PLAN.md).
ACQUISITION_SEED = 2201
CONTINUATION_ORDER_SEEDS = [7701, 7702, 7703]
MAX_ACQUISITION_STEPS = 24_000
RETENTION_STEPS = 6_000
BATCH_SIZE = 64
EVAL_EVERY = 200
LR_HIGH, LR_LOW = 1e-3, 1e-5
ENTRY_RESERVE_MINUTES = 40.0
CONTINUATION_RESERVE_MINUTES = 20.0

# Fork policy, stated before any BIND_SEALED inspection (PLAN.md qualification
# section): prefer ORDER_AUGMENTED when both regimes qualify; a regime that
# does not qualify on BIND_CONTROL is never forked.
FORK_POLICY = "PREFER_ORDER_AUGMENTED_IF_BOTH_QUALIFY_ELSE_ONLY_QUALIFYING_REGIME"

# Prospective operational definition of "materially improves" (frozen before
# execution; only consulted when BOTH arms qualify, so the primary contrast is
# unavailable). ORDER_AUGMENTED must exceed CANONICAL_TRAIN by >= 0.05 on BOTH
# BIND_CONTROL ORDER_ONLY and BIND_CONTROL QUERY_ORDER, each measured as the
# mean over the last three evals of the respective arm's trajectory. Raw deltas
# are reported alongside the decision.
MATERIAL_IMPROVEMENT_DELTA = 0.05
MATERIAL_IMPROVEMENT_WINDOW = 3

CHECKPOINT_DIRNAME = "checkpoints"
DEFAULT_GPU_DUTY_CAP = 0.8


class GpuThrottle:
    """Cooperative GPU duty-cycle cap (wall-clock pacing only; it consumes no
    RNG state and cannot change any scientific quantity).

    After each optimizer step it sleeps so GPU kernel time is at most `duty`
    of the step period, keeping the device's reported utilization (the
    fraction of time kernels occupy the GPU) at or below that level. Evals
    (a few ms every 200 steps) are not throttled and add negligible duty.
    """

    def __init__(self, duty: float | None) -> None:
        self.duty = duty
        self.enabled = duty is not None and 0.0 < float(duty) < 1.0
        self._busy_start: float | None = None
        self.seconds_busy = 0.0
        self.seconds_slept = 0.0

    def begin(self) -> None:
        if self.enabled:
            self._busy_start = time.perf_counter()

    def pause(self) -> None:
        if not self.enabled or self._busy_start is None:
            return
        busy = time.perf_counter() - self._busy_start
        self._busy_start = None
        self.seconds_busy += busy
        sleep_for = busy * (1.0 / self.duty - 1.0)
        if sleep_for > 0:
            time.sleep(sleep_for)
            self.seconds_slept += sleep_for

    def payload(self) -> dict:
        return {
            "enabled": self.enabled,
            "target_max_utilization": self.duty if self.enabled else None,
            "busy_seconds": round(self.seconds_busy, 3),
            "slept_seconds": round(self.seconds_slept, 3),
            "note": "wall-clock pacing cap only; no effect on streams, updates or metrics",
        }


def _row_meta(task):
    """Semantic identity of each canonical train row, in expansion order."""
    meta = []
    for facts in task["train_factsets"]:
        mapping = dict((int(k), int(v)) for k, v in facts)
        for k, _ in facts:
            meta.append({"facts": tuple((int(k), int(v)) for k, v in facts),
                         "query": int(k), "answer": str(mapping[int(k)])})
    rows = task["train_rows"]
    if len(meta) != len(rows):
        raise RuntimeError("train meta expansion drift")
    for i, (prompt, answer) in enumerate(rows):
        if binding.render_prompt(meta[i]["facts"], meta[i]["query"]) != prompt or meta[i]["answer"] != answer:
            raise RuntimeError(f"train meta row mismatch at {i}")
    return meta


def qualification_indicator(diagnostics: dict) -> bool:
    """Frozen BIND_CONTROL gate; consumes CONTROL diagnostics only."""
    return all(
        float(diagnostics[name]) >= threshold
        for name, threshold in binding.QUALIFICATION_THRESHOLDS.items()
    )


def qualification_decision(control_indicator_evals: list[tuple[int, float]]):
    """Pure controller: 3 consecutive qualifying evals. Sealed values cannot
    enter this function; they are measured only after a decision fires."""
    return detect_sustained(control_indicator_evals, 0.5, binding.QUALIFICATION_CONSECUTIVE)


def _snapshot_equal(a, b) -> bool:
    """Structural equality of runtime snapshots, nested tensors included."""
    if isinstance(a, dict):
        return (isinstance(b, dict) and a.keys() == b.keys()
                and all(_snapshot_equal(a[k], b[k]) for k in a))
    if isinstance(a, (list, tuple)):
        return (isinstance(b, type(a)) and len(a) == len(b)
                and all(_snapshot_equal(x, y) for x, y in zip(a, b)))
    if torch.is_tensor(a):
        return torch.is_tensor(b) and torch.equal(a, b)
    return a == b


def _rows_as_pairs(rows) -> list[tuple[str, str]]:
    return [(row["prompt"], row["answer"]) for row in rows]


def _evaluate_diagnostics(ark11, model, vocab, rows_by_diagnostic, device) -> dict:
    out = {}
    for diagnostic, rows in rows_by_diagnostic.items():
        exact, _ = ark11.greedy_exact(model, vocab, rows, device)
        out[diagnostic] = float(exact)
    return out


def _augmented_rows(regime, meta, idx, optimizer_step, acq_seed, augmentation_hasher, perm_counts):
    rows = []
    for pos, i in enumerate(idx):
        example_id = int(i)
        item = meta[example_id]
        if regime == "ORDER_AUGMENTED":
            facts, perm_index = binding.augment_facts_with_index(
                item["facts"], acq_seed, optimizer_step, pos, example_id)
            if augmentation_hasher is not None:
                augmentation_hasher.update(
                    f"{acq_seed},{optimizer_step},{pos},{example_id},{perm_index}\n".encode("utf-8"))
                perm_counts[perm_index] = perm_counts.get(perm_index, 0) + 1
            prompt = binding.render_prompt(facts, item["query"])
        elif regime == "CANONICAL_TRAIN":
            prompt = binding.render_prompt(item["facts"], item["query"])
        else:
            raise ValueError(f"unknown regime: {regime}")
        rows.append((prompt, item["answer"]))
    return rows


def _save_checkpoint(ark11, model, optimizer, path: Path, step: int, arm: str,
                     extra: dict | None = None) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "arkenstone-ark014-checkpoint/v1",
        "arm": arm,
        "step": int(step),
        "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        "optimizer": _cpu_optimizer_state(optimizer),
        "torch_rng": torch.get_rng_state().cpu(),
        "parameter_sha256": parameter_sha(model),
        "extra": extra or {},
    }
    torch.save(payload, path)
    return {
        "filename": path.name,
        "sha256": file_sha256(path),
        "parameter_sha256": payload["parameter_sha256"],
        "step": int(step),
    }


def _cpu_optimizer_state(optimizer):
    state = optimizer.state_dict()
    out = {"state": {}, "param_groups": state["param_groups"]}
    for key, value in state["state"].items():
        out["state"][key] = {
            k: (v.detach().cpu().clone() if torch.is_tensor(v) else v)
            for k, v in value.items()
        }
    return out


def acquire_arm(ark11, *, regime, task, meta, device, ctx, writer, checkpoints_dir,
                acq_seed=ACQUISITION_SEED, max_steps=MAX_ACQUISITION_STEPS,
                eval_every=EVAL_EVERY, log=print, throttle: GpuThrottle | None = None) -> dict:
    """One matched acquisition arm. BIND_SEALED is evaluated only after the
    qualification decision and snapshot, and is recorded for analysis only."""
    control_rows = {d: _rows_as_pairs(task["diagnostics"][d]["BIND_CONTROL"])
                    for d in binding.DIAGNOSTICS}
    torch.manual_seed(acq_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(acq_seed)
    vocab = ark11.CompactVocab()
    model = ark11.Micro(vocab.size, 128).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95),
                                  eps=1e-8, weight_decay=0.1)
    rng = torch.Generator().manual_seed(acq_seed)
    control_evals: list[tuple[int, float]] = []
    trajectory = []
    supervised_positions = 0
    examples_seen = 0
    augmentation_hasher = hashlib.sha256()
    perm_counts: dict[int, int] = {}
    started = time.perf_counter()
    status = "INCOMPLETE_EXCEPTION"

    try:
        for step in range(1, max_steps + 1):
            ensure_budget(ctx)
            idx = torch.randint(0, len(meta), (BATCH_SIZE,), generator=rng)
            batch_rows = _augmented_rows(regime, meta, idx, step, acq_seed,
                                         augmentation_hasher, perm_counts)
            if throttle is not None:
                throttle.begin()
            loss, count = ark11.loss_and_positions(model, vocab, batch_rows, device)
            if not torch.isfinite(loss):
                raise RuntimeError(f"nonfinite {regime} acquisition loss step={step}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if throttle is not None:
                throttle.pause()
            supervised_positions += int(count.item())
            examples_seen += BATCH_SIZE

            if step == 1 or step % eval_every == 0:
                diagnostics = _evaluate_diagnostics(ark11, model, vocab, control_rows, device)
                indicator = qualification_indicator(diagnostics)
                control_evals.append((step, 1.0 if indicator else 0.0))
                onset, confirm = qualification_decision(control_evals)
                trajectory.append({"step": step, "loss": float(loss.detach().item()),
                                   **{f"control_{d}": v for d, v in diagnostics.items()},
                                   "control_qualified": bool(indicator)})
                log(f"[ARK-014 {regime}] step={step} "
                    + " ".join(f"{d}={diagnostics[d]:.3f}" for d in binding.DIAGNOSTICS)
                    + (f"  QUALIFIED at {confirm}" if confirm is not None else ""), flush=True)
                if confirm is not None:
                    snapshot = ark11.snapshot_state(model, optimizer)
                    checkpoint = _save_checkpoint(
                        ark11, model, optimizer,
                        checkpoints_dir / f"{regime}_qualified.pt", step, regime,
                        extra={"kind": "qualification_snapshot"})
                    # Sealed measurement strictly after the decision + snapshot.
                    sealed = _evaluate_diagnostics(
                        ark11, model, vocab,
                        {d: _rows_as_pairs(task["diagnostics"][d]["BIND_SEALED"])
                         for d in binding.DIAGNOSTICS},
                        device)
                    status = "QUALIFIED"
                    return _arm_payload(regime, status, acq_seed, max_steps, step,
                                        supervised_positions, examples_seen, started,
                                        trajectory, control_evals, onset, confirm,
                                        augmentation_hasher, perm_counts, len(meta),
                                        checkpoint=checkpoint, sealed_at_qualification=sealed,
                                        eval_every=eval_every)
        status = "NO_QUALIFICATION"
        checkpoint = _save_checkpoint(
            ark11, model, optimizer,
            checkpoints_dir / f"{regime}_final.pt", max_steps, regime,
            extra={"kind": "final_state"})
        return _arm_payload(regime, status, acq_seed, max_steps, max_steps,
                            supervised_positions, examples_seen, started, trajectory,
                            control_evals, None, None, augmentation_hasher, perm_counts,
                            len(meta), checkpoint=checkpoint, eval_every=eval_every)
    except BudgetExhausted:
        status = "INCOMPLETE_BUDGET_STOP"
        raise
    except (Exception, KeyboardInterrupt):
        status = "INCOMPLETE_EXCEPTION"
        raise
    finally:
        if status in ("INCOMPLETE_BUDGET_STOP", "INCOMPLETE_EXCEPTION"):
            # Any interrupted arm preserves its partial evidence under an
            # explicit incomplete status; summarize() never counts these.
            checkpoint = _save_checkpoint(
                ark11, model, optimizer,
                checkpoints_dir / f"{regime}_incomplete.pt", supervised_positions // BATCH_SIZE,
                regime, extra={"kind": "incomplete_state"})
            payload = _arm_payload(regime, status, acq_seed, max_steps,
                                   supervised_positions // BATCH_SIZE,
                                   supervised_positions, examples_seen, started, trajectory,
                                   control_evals, None, None, augmentation_hasher,
                                   perm_counts, len(meta), checkpoint=checkpoint,
                                   eval_every=eval_every)
            writer.save(f"ARK-014_ARM_{regime}_INCOMPLETE.json", payload)


def _arm_payload(regime, status, acq_seed, max_steps, steps_run, supervised_positions,
                 examples_seen, started, trajectory, control_evals, onset, confirm,
                 augmentation_hasher, perm_counts, pool_size, *, checkpoint, eval_every,
                 sealed_at_qualification=None) -> dict:
    stream = _regenerate_minibatch_stream(acq_seed, steps_run, pool_size)
    payload = {
        "regime": regime,
        "status": status,
        "acquisition_seed": acq_seed,
        "max_steps": max_steps,
        "steps_run": steps_run,
        "completed_steps": steps_run if status in ("QUALIFIED", "NO_QUALIFICATION") else 0,
        "eval_every": eval_every,
        "supervised_positions": supervised_positions,
        "examples_seen": examples_seen,
        "wall_seconds": time.perf_counter() - started,
        "trajectory": trajectory,
        "control_evals": [list(x) for x in control_evals],
        "qualification_onset_step": onset,
        "qualification_confirmation_step": confirm,
        "minibatch_stream": {
            "generator": "torch.Generator().manual_seed(acq_seed); torch.randint(pool)",
            "seed": acq_seed,
            "pool_size": pool_size,
            "batches": steps_run,
            "batch_size": BATCH_SIZE,
            "order_sha256": order_sha256(stream),
        },
        "order_augmentation": {
            "enabled": regime == "ORDER_AUGMENTED",
            "spec": binding.AUGMENTATION_SPEC,
            "stream_sha256": augmentation_hasher.hexdigest(),
            "permutation_counts": {str(k): v for k, v in sorted(perm_counts.items())},
        },
        "checkpoint": checkpoint,
    }
    if sealed_at_qualification is not None:
        payload["sealed_at_qualification"] = sealed_at_qualification
    return payload


def _regenerate_minibatch_stream(acq_seed: int, batches: int, pool_size: int) -> list[list[int]]:
    """Rebuild the semantic minibatch stream for hashing; identical to the
    in-loop torch.randint draw by construction (same generator/seed/shape)."""
    if batches <= 0:
        return []
    rng = torch.Generator().manual_seed(acq_seed)
    return torch.randint(0, pool_size, (batches, BATCH_SIZE), generator=rng).tolist()


def retention_fork(ark11, *, regime, snapshot, order_seed, lr, task, meta, device, ctx,
                   checkpoints_dir, acq_seed=ACQUISITION_SEED, steps=RETENTION_STEPS,
                   eval_every=EVAL_EVERY, log=print, throttle: GpuThrottle | None = None) -> dict:
    """One matched continuation arm at a fixed LR. Identical semantic stream and
    (for ORDER_AUGMENTED) identical order permutations for both LRs."""
    indices = generate_indices(order_seed, steps, BATCH_SIZE, len(meta))
    control_rows = {d: _rows_as_pairs(task["diagnostics"][d]["BIND_CONTROL"])
                    for d in binding.DIAGNOSTICS}
    sealed_rows = {d: _rows_as_pairs(task["diagnostics"][d]["BIND_SEALED"])
                   for d in binding.DIAGNOSTICS}
    vocab, model, optimizer = ark11.load_fork(snapshot, lr)
    augmentation_hasher = hashlib.sha256()
    perm_counts: dict[int, int] = {}
    trajectory = []
    supervised_positions = 0
    examples_seen = 0
    started = time.perf_counter()
    reference = parameter_sha(model)
    lr_label = "HIGH" if lr == LR_HIGH else "LOW"

    for rel_step in range(1, steps + 1):
        ensure_budget(ctx)
        idx = indices[rel_step - 1]
        batch_rows = _augmented_rows(regime, meta, idx, rel_step, acq_seed,
                                     augmentation_hasher, perm_counts)
        if throttle is not None:
            throttle.begin()
        loss, count = ark11.loss_and_positions(model, vocab, batch_rows, device)
        if not torch.isfinite(loss):
            raise RuntimeError(f"nonfinite retention loss {regime}/{lr_label}/{order_seed} step={rel_step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if throttle is not None:
            throttle.pause()
        supervised_positions += int(count.item())
        examples_seen += BATCH_SIZE

        if rel_step % eval_every == 0:
            control = _evaluate_diagnostics(ark11, model, vocab, control_rows, device)
            # Sealed is recorded for the frozen endpoint analysis only; nothing
            # in this loop reads it back for decisions (there are none here).
            sealed = _evaluate_diagnostics(ark11, model, vocab, sealed_rows, device)
            c_indicator = qualification_indicator(control)
            s_indicator = qualification_indicator(sealed)
            trajectory.append({
                "step": rel_step, "loss": float(loss.detach().item()),
                **{f"control_{d}": v for d, v in control.items()},
                **{f"sealed_{d}": v for d, v in sealed.items()},
                "control_qualified": bool(c_indicator), "sealed_qualified": bool(s_indicator),
            })
            log(f"[ARK-014 RET {regime} o{order_seed} {lr_label}] rel={rel_step} "
                f"control_q={c_indicator} sealed_q={s_indicator}", flush=True)

    completed = len(trajectory) * eval_every == steps
    checkpoint = _save_checkpoint(
        ark11, model, optimizer,
        checkpoints_dir / f"retention_{regime}_o{order_seed}_{lr_label}.pt", steps,
        f"retention_{regime}_{lr_label}", extra={"kind": "retention_final"})
    return {
        "regime": regime, "order_seed": order_seed, "lr": lr, "lr_label": lr_label,
        "status": "COMPLETED" if completed else "INCOMPLETE_EVAL_GRID",
        "completed_steps": steps,
        "supervised_positions": supervised_positions,
        "examples_seen": examples_seen,
        "wall_seconds": time.perf_counter() - started,
        "trajectory": trajectory,
        "order_stream": {"seed": order_seed, "batches": steps, "batch_size": BATCH_SIZE,
                         "order_sha256": order_sha256(indices)},
        "order_augmentation": {
            "enabled": regime == "ORDER_AUGMENTED", "spec": binding.AUGMENTATION_SPEC,
            "stream_sha256": augmentation_hasher.hexdigest(),
            "permutation_counts": {str(k): v for k, v in sorted(perm_counts.items())},
        },
        "fork_parameter_sha256": reference,
        "final_parameter_sha256": parameter_sha(model),
        "checkpoint": checkpoint,
        "control_metrics": {d: _diag_endpoint(trajectory, f"control_{d}") for d in binding.DIAGNOSTICS},
        "sealed_metrics": {d: _diag_endpoint(trajectory, f"sealed_{d}") for d in binding.DIAGNOSTICS},
        "sealed_qualification_failures": _first_sustained_failures(trajectory),
        "sealed_qualification_at_fork": _sealed_at_fork(trajectory),
    }


def _diag_endpoint(trajectory, key) -> dict:
    if not trajectory:
        return {"status": "EMPTY"}
    vals = [float(x[key]) for x in trajectory]
    return {"AREA": sum(vals) / len(vals), "FINAL": vals[-1], "PEAK": max(vals),
            "MIN": min(vals)}


def _first_sustained_failures(trajectory, consecutive: int = 3) -> dict:
    """First 3 consecutive failing evals of the sealed qualification."""
    if not trajectory:
        return {"occurred": False, "onset_step": None, "confirmation_step": None}
    evals = [(int(x["step"]), 1.0 if x["sealed_qualified"] else 0.0) for x in trajectory]
    onset, confirm = detect_sustained(evals, 0.5, consecutive, below=True)
    return {"occurred": confirm is not None, "onset_step": onset,
            "confirmation_step": confirm}


def _sealed_at_fork(trajectory) -> bool:
    return bool(trajectory) and bool(trajectory[0]["sealed_qualified"])


def summarize(acquisitions: list[dict], retention: list[dict], *, protocol_scale: str) -> dict:
    """Frozen plan verdicts. Only matched completed evidence can count."""
    by_regime = {a.get("regime"): a for a in acquisitions}
    aug = by_regime.get("ORDER_AUGMENTED") or {}
    canon = by_regime.get("CANONICAL_TRAIN") or {}
    aug_q = aug.get("status") == "QUALIFIED"
    canon_q = canon.get("status") == "QUALIFIED"

    if not aug_q and not canon_q:
        verdict = "ROBUST_BINDING_NOT_ACQUIRED"
    else:
        matched = _matched_retention_orders(retention)
        high_failures = sum(1 for _, arms in matched.items()
                            if arms["HIGH"]["sealed_qualification_failures"]["occurred"])
        low_failures = sum(1 for _, arms in matched.items()
                           if arms["LOW"]["sealed_qualification_failures"]["occurred"])
        reverse = sum(1 for _, arms in matched.items()
                      if arms["LOW"]["sealed_qualification_failures"]["occurred"]
                      and not arms["HIGH"]["sealed_qualification_failures"]["occurred"])
        sealed_qualified_at_fork = sum(
            1 for _, arms in matched.items()
            if arms["HIGH"]["sealed_qualification_at_fork"] and arms["LOW"]["sealed_qualification_at_fork"])
        risk_diff = ((low_failures - high_failures) / len(matched)) if matched else None
        retention_status = _retention_execution_status(retention, len(matched))
        verdict = _verdict(aug_q, canon_q, aug, canon, matched, high_failures,
                           low_failures, reverse, sealed_qualified_at_fork, retention_status)

    summary = {
        "verdict": verdict,
        "verdict_scale": protocol_scale,
        "order_robustness_repaired": _repair_criterion(aug, canon, aug_q, canon_q),
        "regime_qualification": {
            "ORDER_AUGMENTED": aug_q, "CANONICAL_TRAIN": canon_q,
            "rule": "BIND_CONTROL-only, 3 consecutive evals, frozen thresholds",
        },
        "material_improvement_rule": {
            "delta": MATERIAL_IMPROVEMENT_DELTA, "window_evals": MATERIAL_IMPROVEMENT_WINDOW,
            "consulted_only_when_both_arms_qualify": True,
        },
        "fork_policy": FORK_POLICY,
        "matched_retention_orders": len(_matched_retention_orders(retention)),
        "limitations": ["acquisition seed n=1; a positive screen is not independent replication"],
    }
    if aug_q or canon_q:
        summary["paired_retention"] = _paired_retention_summary(retention)
    return summary


def _repair_criterion(aug, canon, aug_q, canon_q) -> dict:
    """The plan's primary ORDER_ROBUSTNESS_REPAIRED criterion, recorded
    separately from the retention-branch verdict so a repaired acquisition is
    never hidden by an inconclusive retention screen."""
    if aug_q and not canon_q:
        return {"met": True,
                "basis": ("ORDER_AUGMENTED QUALIFIED on BIND_CONTROL while CANONICAL_TRAIN "
                          "did not qualify"),
                "augmented_confirmation_step": aug.get("qualification_confirmation_step"),
                "canonical_status": (canon or {}).get("status")}
    if aug_q and canon_q:
        met, deltas = _materially_improves(aug, canon)
        return {"met": met,
                "basis": "both regimes qualified; frozen material-improvement rule consulted",
                "deltas": deltas}
    return {"met": False,
            "basis": ("ORDER_AUGMENTED did not qualify on BIND_CONTROL"
                      if not aug_q else "both regimes qualified but the rule was not met"),
            "canonical_status": (canon or {}).get("status")}


def _retention_execution_status(retention, matched_count) -> str:
    if not retention:
        return "NOT_EXECUTED"
    if all(str(r.get("status", "")).startswith("BUDGET_BLOCKED") for r in retention):
        return "NOT_EXECUTED"
    if any(r.get("status") != "COMPLETED" for r in retention):
        return "INCOMPLETE_ARMS"
    return "EXECUTED" if matched_count else "NO_MATCHED_ORDERS"


def _matched_retention_orders(retention) -> dict:
    """Order seeds where BOTH LRs completed the full frozen horizon."""
    by_order: dict[int, dict] = {}
    for row in retention:
        if row.get("status") != "COMPLETED" or row.get("completed_steps") != RETENTION_STEPS:
            continue
        by_order.setdefault(row["order_seed"], {})[row["lr_label"]] = row
    return {order: arms for order, arms in sorted(by_order.items())
            if set(arms) == {"HIGH", "LOW"}}


def _paired_retention_summary(retention) -> dict:
    matched = _matched_retention_orders(retention)
    high_failures = sum(1 for _, arms in matched.items()
                        if arms["HIGH"]["sealed_qualification_failures"]["occurred"])
    low_failures = sum(1 for _, arms in matched.items()
                       if arms["LOW"]["sealed_qualification_failures"]["occurred"])
    reverse = sum(1 for _, arms in matched.items()
                  if arms["LOW"]["sealed_qualification_failures"]["occurred"]
                  and not arms["HIGH"]["sealed_qualification_failures"]["occurred"])
    sealed_qualified_at_fork = sum(
        1 for _, arms in matched.items()
        if arms["HIGH"]["sealed_qualification_at_fork"] and arms["LOW"]["sealed_qualification_at_fork"])
    per_order = []
    for order, arms in matched.items():
        per_order.append({
            "order_seed": order,
            "HIGH": {"failure_occurred": arms["HIGH"]["sealed_qualification_failures"]["occurred"],
                     "failure_confirmation_step": arms["HIGH"]["sealed_qualification_failures"]["confirmation_step"],
                     "sealed_metrics": arms["HIGH"]["sealed_metrics"]},
            "LOW": {"failure_occurred": arms["LOW"]["sealed_qualification_failures"]["occurred"],
                    "failure_confirmation_step": arms["LOW"]["sealed_qualification_failures"]["confirmation_step"],
                    "sealed_metrics": arms["LOW"]["sealed_metrics"]},
        })
    return {
        "endpoint": "first 3 consecutive BIND_SEALED qualification failures; paired HIGH vs LOW",
        "orders_compared": sorted(matched),
        "sealed_qualified_orders_at_fork": sealed_qualified_at_fork,
        "high_failure_orders": high_failures,
        "low_failure_orders": low_failures,
        "risk_difference_low_minus_high": ((low_failures - high_failures) / len(matched)) if matched else None,
        "reverse_discordance_orders": reverse,
        "per_order": per_order,
    }


def _verdict(aug_q, canon_q, aug, canon, matched, high_failures, low_failures,
             reverse, sealed_qualified_at_fork, retention_status) -> str:
    # The repair question is decided before any retention outcome is consulted.
    if aug_q and not canon_q:
        repaired = True
    elif aug_q and canon_q:
        repaired = _materially_improves(aug, canon)[0]
    else:
        # Only CANONICAL_TRAIN qualified: the order-augmentation hypothesis is
        # not supported, regardless of retention outcomes.
        return "ORDER_AUGMENTATION_NOT_SUPPORTED_CANONICAL_QUALIFIED"
    if not repaired:
        return "ORDER_AUGMENTATION_NOT_SUPPORTED_CANONICAL_QUALIFIED"
    if retention_status == "NOT_EXECUTED":
        return "ROBUST_BINDING_ACQUIRED_BUT_RETENTION_NOT_EXECUTED"
    if len(matched) < 2:
        return "ROBUST_BINDING_ACQUIRED_BUT_TRANSFER_INCONCLUSIVE"
    if sealed_qualified_at_fork >= 2 and high_failures >= 1 and low_failures == 0 and reverse == 0:
        return "NONARITHMETIC_LR_PROTECTION_SCREEN"
    # Frozen interpretation (prospective protocol note): TRANSFER_NOT_SUPPORTED
    # requires clean discordance-free evidence that LOW does not protect; a
    # reverse discordance makes the directional evidence mixed, not negative.
    if high_failures >= 1 and low_failures >= 1 and reverse == 0:
        return "TRANSFER_NOT_SUPPORTED_SCREEN"
    return "ROBUST_BINDING_ACQUIRED_BUT_TRANSFER_INCONCLUSIVE"


def _materially_improves(aug, canon) -> tuple[bool, dict]:
    """Frozen rule (pure): mean over each arm's last window evals of the
    BIND_CONTROL ORDER_ONLY and QUERY_ORDER trajectories; ORDER_AUGMENTED must
    exceed CANONICAL_TRAIN by >= delta on both. Returns (met, raw deltas)."""

    def tail_mean(arm, key):
        vals = [float(x[key]) for x in arm.get("trajectory", [])][-MATERIAL_IMPROVEMENT_WINDOW:]
        return sum(vals) / len(vals) if vals else float("nan")

    deltas = {}
    met = True
    for diagnostic in ("ORDER_ONLY", "QUERY_ORDER"):
        d = tail_mean(aug, f"control_{diagnostic}") - tail_mean(canon, f"control_{diagnostic}")
        deltas[diagnostic] = d
        if not (d >= MATERIAL_IMPROVEMENT_DELTA):
            met = False
    return met, deltas


def preflight(ark11, ctx: RunContext, task) -> dict:
    """Bounded correctness box for the exact device the campaign will use."""
    device = ctx.device
    meta = _row_meta(task)
    vocab = ark11.CompactVocab()
    torch.manual_seed(9191)
    model = ark11.Micro(vocab.size, 128).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95),
                                  eps=1e-8, weight_decay=0.1)
    probe_rng = torch.Generator().manual_seed(4242)
    rows = _augmented_rows("ORDER_AUGMENTED", meta,
                           torch.randint(0, len(meta), (8,), generator=probe_rng).tolist(),
                           1, 1, None, {})
    loss, count = ark11.loss_and_positions(model, vocab, rows, device)
    assert torch.isfinite(loss)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    assert int(count.item()) == 3 * len(rows), "supervised positions must be answer-BOS+digit+EOS"

    snapshot = ark11.snapshot_state(model, optimizer)
    original = cpu_tree(snapshot)
    # Identical semantic batch and identical augmentation for both forks:
    idx = torch.randint(0, len(meta), (8,), generator=torch.Generator().manual_seed(5150)).tolist()
    fork_rows = _augmented_rows("ORDER_AUGMENTED", meta, idx, 1, 1, None, {})
    fork_hashes = []
    for _ in range(2):
        _, fork_model, fork_optimizer = ark11.load_fork(snapshot, 1e-3)
        loss, _ = ark11.loss_and_positions(fork_model, vocab, fork_rows, device)
        fork_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(fork_model.parameters(), 1.0)
        fork_optimizer.step()
        fork_hashes.append(parameter_sha(fork_model))
        assert _snapshot_equal(snapshot, original), "fork mutated its source snapshot"
    assert fork_hashes[0] == fork_hashes[1], "matched fork updates diverged"

    # Augmentation purity: no torch RNG consumption, deterministic, in-range.
    state_before = torch.get_rng_state().clone()
    seen = set()
    for probe in range(200):
        item = meta[probe % len(meta)]
        f, index = binding.augment_facts_with_index(item["facts"], 2201, probe, probe % 64, probe)
        seen.add(index)
        assert tuple(sorted(f)) == tuple(sorted(item["facts"]))
        assert binding.augment_facts_with_index(
            item["facts"], 2201, probe, probe % 64, probe) == (f, index)
    assert len(seen) == 6, "augmentation must cover all six permutations"
    assert torch.equal(state_before, torch.get_rng_state()), "augmentation consumed torch RNG"

    # Controller is a pure function of the CONTROL indicator stream: a stream
    # whose fourth eval still qualifies confirms at the fourth eval; inserting
    # one failing eval resets the streak and no confirmation fires.
    base = [(200, 1.0), (400, 1.0), (600, 1.0), (800, 1.0)]
    assert qualification_decision(base) == (200, 600)
    # A failing eval resets the streak: two passes, one failure, one pass is not
    # 3 consecutive qualifying evals.
    assert qualification_decision([(200, 1.0), (400, 1.0), (800, 0.0), (1000, 1.0)]) == (None, None)

    return {
        "status": "PASS",
        "scope": "bounded engineering integrity; no capability evidence",
        "task_manifest_sha256": task["manifest"]["manifest_sha256"],
        "supervised_positions_per_row": 3,
        "matched_fork_next_update_sha256": fork_hashes[0],
        "augmentation_permutations_covered": sorted(seen),
        "augmentation_torch_rng_untouched": True,
        "unverified": ["full training horizons", "capability gain"],
    }


def run_campaign(ctx: RunContext, *, max_steps: int = MAX_ACQUISITION_STEPS,
                 retention_steps: int = RETENTION_STEPS, log=print,
                 max_gpu_duty: float | None = DEFAULT_GPU_DUTY_CAP) -> dict:
    protocol_scale = ("PREREGISTERED_FROZEN"
                      if max_steps == MAX_ACQUISITION_STEPS and retention_steps == RETENTION_STEPS
                      else "DIAGNOSTIC_SCALE_NOT_PREREGISTERED")
    throttle = GpuThrottle(max_gpu_duty if ctx.device.type == "cuda" else None)
    ark11 = load_ark11()
    bind_ark11_runtime(ark11, ctx.device, ctx.head, ctx=ctx)
    task = binding.build_binding_task()
    meta = _row_meta(task)
    writer = _receipt_writer(ctx, "ARK-014")
    checkpoints = ctx.output_dir / CHECKPOINT_DIRNAME

    if ctx.minutes_left < ENTRY_RESERVE_MINUTES and protocol_scale == "PREREGISTERED_FROZEN":
        payload = {"status": "BUDGET_BLOCKED",
                   "required_entry_reserve_minutes": ENTRY_RESERVE_MINUTES,
                   "summary": {"verdict": "BUDGET_BLOCKED"}}
        writer.save("ARK-014_RESULT.json", payload)
        return payload

    writer.save("ARK-014_TASK_MANIFEST.json", task["manifest"])
    writer.save("ARK-014_PREFLIGHT.json", preflight(ark11, ctx, task))
    # The GPU duty cap, fork policy and material-improvement rule are published
    # before any BIND_SEALED value exists in this run.
    writer.save("ARK-014_FROZEN_POLICY.json", {
        "fork_policy": FORK_POLICY,
        "gpu_duty_cap": throttle.payload(),
        "material_improvement_rule": {
            "delta": MATERIAL_IMPROVEMENT_DELTA, "window_evals": MATERIAL_IMPROVEMENT_WINDOW,
            "definition": ("consulted only when both regimes qualify on BIND_CONTROL; "
                           "ORDER_AUGMENTED must exceed CANONICAL_TRAIN by >= delta on both "
                           "ORDER_ONLY and QUERY_ORDER, mean over each arm's last window evals"),
        },
        "protocol_scale": protocol_scale,
        "sealed_usage": "measurement-only; never selects checkpoints, LR, stopping or regime",
    })

    acquisitions, retention = [], []
    try:
        for regime in ("CANONICAL_TRAIN", "ORDER_AUGMENTED"):
            if ctx.minutes_left <= 0:
                acquisitions.append({"regime": regime, "status": "BUDGET_BLOCKED_BEFORE_START"})
                continue
            log(f"\n=== ARK-014 acquisition {regime} ===", flush=True)
            arm = acquire_arm(ark11, regime=regime, task=task, meta=meta, device=ctx.device,
                              ctx=ctx, writer=writer, checkpoints_dir=checkpoints,
                              max_steps=max_steps, log=log, throttle=throttle)
            acquisitions.append(arm)
            writer.save("ARK-014_PARTIAL.json", {
                "protocol_scale": protocol_scale, "acquisitions": _public(acquisitions),
                "retention": _public(retention)})

        aug = next((a for a in acquisitions if a.get("regime") == "ORDER_AUGMENTED"), {})
        canon = next((a for a in acquisitions if a.get("regime") == "CANONICAL_TRAIN"), {})
        aug_q = aug.get("status") == "QUALIFIED"
        canon_q = canon.get("status") == "QUALIFIED"
        if aug_q and canon_q:
            fork_regime = "ORDER_AUGMENTED"  # frozen policy
        elif aug_q:
            fork_regime = "ORDER_AUGMENTED"
        elif canon_q:
            fork_regime = "CANONICAL_TRAIN"
        else:
            fork_regime = None

        if fork_regime is None:
            log("no regime qualified on BIND_CONTROL; retention blocked by plan", flush=True)
        elif ctx.minutes_left < CONTINUATION_RESERVE_MINUTES and protocol_scale == "PREREGISTERED_FROZEN":
            retention.append({"regime": fork_regime, "status": "BUDGET_BLOCKED_BEFORE_RETENTION"})
            writer.save("ARK-014_PARTIAL.json", {
                "protocol_scale": protocol_scale, "acquisitions": _public(acquisitions),
                "retention": _public(retention)})
        else:
            snapshot = _snapshot_from_checkpoint(ark11, ctx, acquisitions, fork_regime, checkpoints)
            for order_seed in CONTINUATION_ORDER_SEEDS:
                for lr, label in ((LR_HIGH, "HIGH"), (LR_LOW, "LOW")):
                    if ctx.minutes_left <= 0:
                        retention.append({"regime": fork_regime, "order_seed": order_seed,
                                          "lr_label": label, "status": "BUDGET_BLOCKED_BEFORE_ARM"})
                        continue
                    log(f"\n--- ARK-014 retention {fork_regime} order={order_seed} {label} ---", flush=True)
                    arm = retention_fork(ark11, regime=fork_regime, snapshot=snapshot,
                                         order_seed=order_seed, lr=lr, task=task, meta=meta,
                                         device=ctx.device, ctx=ctx,
                                         checkpoints_dir=checkpoints, steps=retention_steps,
                                         log=log, throttle=throttle)
                    retention.append(arm)
                    writer.save("ARK-014_PARTIAL.json", {
                        "protocol_scale": protocol_scale, "acquisitions": _public(acquisitions),
                        "retention": _public(retention)})

        summary = summarize(acquisitions, retention, protocol_scale=protocol_scale)
        payload = {
            "status": "EXECUTED_OR_PARTIAL_BUDGETED",
            "protocol_scale": protocol_scale,
            "compute_box": _compute_box(ctx, throttle=throttle),
            "acquisitions": _public(acquisitions),
            "retention": _public(retention),
            "summary": summary,
            "campaign_minutes_used": ctx.minutes_used,
        }
        writer.save("ARK-014_RESULT.json", payload)
        return payload
    except (Exception, KeyboardInterrupt) as exc:
        summary = summarize(acquisitions, retention, protocol_scale=protocol_scale)
        writer.save("ARK-014_FAILURE.json", {
            "status": "BUDGET_EXHAUSTED" if isinstance(exc, BudgetExhausted) else "FAILED",
            "exception_type": type(exc).__name__, "exception": str(exc),
            "traceback": traceback.format_exc(),
            "protocol_scale": protocol_scale,
            "acquisitions": _public(acquisitions),
            "retention": _public(retention),
            "summary": summary,
        })
        raise


def _public(arms):
    out = []
    for arm in arms:
        arm = dict(arm)
        arm.pop("snapshot", None)
        out.append(arm)
    return out


def _snapshot_from_checkpoint(ark11, ctx, acquisitions, regime, checkpoints):
    """Fork the retention phase from the recorded qualified checkpoint."""
    arm = next(a for a in acquisitions if a.get("regime") == regime)
    path = checkpoints / arm["checkpoint"]["filename"]
    if file_sha256(path) != arm["checkpoint"]["sha256"]:
        raise RuntimeError("qualified checkpoint changed since acquisition")
    payload = torch.load(path, map_location=ctx.device, weights_only=False)
    vocab = ark11.CompactVocab()
    model = ark11.Micro(vocab.size, 128).to(ctx.device)
    model.load_state_dict(payload["model"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95),
                                  eps=1e-8, weight_decay=0.1)
    snapshot = {
        "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "optimizer": payload["optimizer"],
        "torch_rng": payload["torch_rng"],
        "cuda_rng": [x.cpu() for x in torch.cuda.get_rng_state_all()] if ctx.device.type == "cuda" else [],
    }
    if parameter_sha(model) != payload["parameter_sha256"]:
        raise RuntimeError("checkpoint parameter hash mismatch")
    return snapshot


def _compute_box(ctx: RunContext, throttle: GpuThrottle | None = None) -> dict:
    from discovery_v6_common import device_name
    return {
        "device": str(ctx.device),
        "device_name": device_name(ctx.device),
        "torch_num_threads": torch.get_num_threads(),
        "budget_minutes": ctx.budget_minutes,
        "gpu_duty_cap": throttle.payload() if throttle is not None else None,
        "authorization_note": ("local run on owner-provided hardware (CPU or the owner's "
                               "single local GPU after availability check, capped at "
                               "80% GPU utilization by owner instruction); no paid "
                               "accelerator, hosted service or network access"),
    }


def _receipt_writer(ctx: RunContext, experiment_id: str) -> ReceiptWriter:
    return ReceiptWriter(
        ctx, experiment_id=experiment_id, plan_sha=PLAN_COMMIT_SHA, runner_path=RUNNER_PATH,
        extra_plan_shas={"ark014_plan_commit_sha": PLAN_COMMIT_SHA},
        extra_source_paths=[RUNNER_PATH.parent / "ark014_binding.py"],
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--budget-minutes", type=float, default=2.0)
    parser.add_argument("--output-dir", type=Path, help="must be a new directory")
    parser.add_argument("--max-steps", type=int, default=MAX_ACQUISITION_STEPS)
    parser.add_argument("--retention-steps", type=int, default=RETENTION_STEPS)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--max-gpu-duty", type=float, default=DEFAULT_GPU_DUTY_CAP,
                        help="CUDA only: cap GPU utilization at this duty cycle (0<d<=1); "
                             "owner instruction caps it at 0.8")
    args = parser.parse_args(argv)
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda requested but CUDA is not available")

    ctx = RunContext(torch.device(args.device), git_head(), time.time(), args.budget_minutes,
                     args.output_dir)
    result = 1
    try:
        if args.preflight_only:
            ark11 = load_ark11()
            bind_ark11_runtime(ark11, ctx.device, ctx.head, ctx=ctx)
            task = binding.build_binding_task()
            writer = _receipt_writer(ctx, "ARK-014_PREFLIGHT")
            writer.save("ARK-014_TASK_MANIFEST.json", task["manifest"])
            writer.save("ARK-014_PREFLIGHT.json", preflight(ark11, ctx, task))
        else:
            # The campaign owns its receipts, task construction and failure
            # packaging; this entry point only wraps it with the final ZIP.
            run_campaign(ctx, max_steps=args.max_steps, retention_steps=args.retention_steps,
                         max_gpu_duty=args.max_gpu_duty)
        result = 0
    except (Exception, KeyboardInterrupt) as exc:
        try:
            _receipt_writer(ctx, "ARK-014").save("FAILURE_RECEIPT.json", {
                "status": "BUDGET_EXHAUSTED" if isinstance(exc, BudgetExhausted) else "FAILED",
                "exception_type": type(exc).__name__, "exception": str(exc),
                "traceback": traceback.format_exc(),
            })
        except Exception as receipt_exc:
            print(f"failure receipt could not be written: {receipt_exc!r}", flush=True)
        print(f"FAILED: {type(exc).__name__}: {exc}", flush=True)
    finally:
        from discovery_v6_common import package_all
        package_all(download=False, ctx=ctx)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
