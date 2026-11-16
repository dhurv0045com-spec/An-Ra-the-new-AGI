"""Conditional V8 forward transfer engine (Scenarios R12, C24) with right-censored curves."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

from anra_v5.elendril.codec import digest
from anra_v5.elendril.foundation_training import Recipe, StageTransition
from v5_training.checkpoint import CheckpointStore

from .contracts import V8ModelSpec
from .protocol import ADVANCEMENT_RULES
from .training import V8TypedSample, V8TypedTrainer

TRANSFER_HISTORIES = ("fresh", "unrelated_history", "ab_no_replay", "ab_qualified_replay")


def verify_transfer_prerequisites(*, retention_qualified: bool, c_achievable_accuracy: float | None) -> dict[str, object]:
    """Require qualified A->B retention and Task C >= 80% qualification before reporting C24 transfer."""

    threshold = float(ADVANCEMENT_RULES["transfer"]["c_qualification_min"])
    reasons: list[str] = []
    if not retention_qualified:
        reasons.append("A->B retention stage is not qualified")
    if c_achievable_accuracy is None or c_achievable_accuracy < threshold:
        reasons.append(f"Task C threshold {threshold:.2f} not reached (observed={c_achievable_accuracy})")
    return {
        "qualified": len(reasons) == 0,
        "status": "QUALIFIED" if not reasons else "PREREQUISITE_NOT_QUALIFIED",
        "reasons": reasons,
    }


def right_censored_acquisition(
    curve: list[dict[str, float]],
    *,
    threshold: float = 0.80,
) -> dict[str, object]:
    """Compute exact update/example cost to reach threshold, or explicit right-censoring notice."""

    if not curve:
        return {"reached": False, "right_censored": True, "right_censored_at_updates": 0, "final_accuracy": 0.0}
    for point in curve:
        if float(point["accuracy"]) >= threshold:
            return {
                "reached": True,
                "right_censored": False,
                "updates_to_threshold": int(point["updates"]),
                "examples_to_threshold": int(point.get("examples", point["updates"])),
                "accuracy_at_cross": float(point["accuracy"]),
            }
    last = curve[-1]
    return {
        "reached": False,
        "right_censored": True,
        "right_censored_at_updates": int(last["updates"]),
        "right_censored_at_examples": int(last.get("examples", last["updates"])),
        "final_accuracy": float(last["accuracy"]),
    }


def build_task_c_child_trainer(
    task_c_samples: tuple[V8TypedSample, ...],
    *,
    v8_spec: V8ModelSpec,
    recipe: Recipe,
    order_seed: int,
    source_commit: str,
    root: Path,
    history_name: str,
    data_provenance: dict[str, object],
    parent_store: CheckpointStore | None = None,
    parent_checkpoint: str | None = None,
    device: str = "cpu",
    cpu_canary: bool = False,
) -> V8TypedTrainer:
    """Create a Task C trainer from fresh init or explicit StageTransition with optimizer reset."""

    if history_name not in TRANSFER_HISTORIES:
        raise ValueError(f"unsupported transfer history: {history_name!r}")
    if history_name == "fresh":
        return V8TypedTrainer(
            task_c_samples,
            v8_spec=v8_spec,
            recipe=recipe,
            order_seed=order_seed,
            source_commit=source_commit,
            root=root,
            lineage=f"transfer-C-{history_name}",
            data_provenance=data_provenance,
            device=device,
            cpu_canary=cpu_canary,
        )
    if parent_store is None or parent_checkpoint is None:
        raise ValueError(f"transfer history {history_name!r} requires explicit parent_store and parent_checkpoint")

    # Compute exact data_sha for binding transition cleanly without partial state mutation
    parent_state, payloads = parent_store.restore(parent_checkpoint)
    _ = parent_state
    parent_recipe = json.loads(payloads["scheduler.json"])["metadata"]["recipe"]
    data_sha = digest({"samples": [asdict(s) for s in task_c_samples], "provenance": data_provenance})
    transition = StageTransition(
        parent_checkpoint=parent_checkpoint,
        parent_recipe=digest(parent_recipe),
        child_recipe=digest(asdict(recipe)),
        child_data=data_sha,
        optimizer_policy="reset",
    )
    return V8TypedTrainer(
        task_c_samples,
        v8_spec=v8_spec,
        recipe=recipe,
        order_seed=order_seed,
        source_commit=source_commit,
        root=root,
        lineage=f"transfer-C-{history_name}",
        data_provenance=data_provenance,
        device=device,
        cpu_canary=cpu_canary,
        transition=transition,
        parent_store=parent_store,
    )


def evaluate_transfer_gain(curves_by_history: dict[str, list[dict[str, float]]]) -> dict[str, object]:
    """Compare C24 acquisition curves across histories using >=10% example/work reduction rule."""

    threshold = float(ADVANCEMENT_RULES["transfer"]["c_qualification_min"])
    min_rel = float(ADVANCEMENT_RULES["transfer"]["acquisition_example_or_work_reduction_min"])
    acquisitions = {h: right_censored_acquisition(curve, threshold=threshold) for h, curve in curves_by_history.items()}
    fresh_acq = acquisitions.get("fresh")
    ab_rep = acquisitions.get("ab_qualified_replay")
    reduction = None
    qualified = False
    if fresh_acq and ab_rep and ab_rep.get("reached"):
        if fresh_acq.get("reached"):
            f_ex = float(fresh_acq["examples_to_threshold"])
            a_ex = float(ab_rep["examples_to_threshold"])
            reduction = (f_ex - a_ex) / max(f_ex, 1.0)
            qualified = reduction >= min_rel
        else:
            # Transfer reached threshold while fresh remained right-censored before threshold
            qualified = True
    return {
        "scenario": "C24",
        "threshold": threshold,
        "acquisitions": acquisitions,
        "relative_reduction_vs_fresh": reduction,
        "meets_transfer_rule": qualified,
    }


__all__ = [
    "TRANSFER_HISTORIES",
    "build_task_c_child_trainer",
    "evaluate_transfer_gain",
    "right_censored_acquisition",
    "verify_transfer_prerequisites",
]
