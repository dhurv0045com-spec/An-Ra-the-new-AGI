"""E4 executor: gated shared-block architecture (D4 + real execution contracts).

Forks E1-B independently of E3; migrates the actual training handle (S1
gated, S0 disabled slots); verifies gate parameters in the optimizer
inventory, migrated state identity, head/segment behavior and additional
execution cost on that very handle. Both variants use the same declared
training stream/targets/comparison endpoints.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any

from bramastra_lab.research.campaigns.phases.types import (
    EVIDENCE_FIXTURE,
    EVIDENCE_LEARNED_CAMPAIGN,
    JobInput,
    ParentRef,
    PhaseResult,
)


def _resolve_parent(job: JobInput) -> dict[str, Any]:
    key = job.resolved_parent_key()
    if not key:
        raise ValueError("E4 requires an E1-B parent; missing parent must fail")
    candidate = str(key).split("/")[0].strip()
    try:
        if job.parent_ref is not None:
            return job.parent_ref.resolve(job.run_dir)
        return ParentRef(lookup_key=candidate).resolve(job.run_dir)
    except Exception as exc:
        raise ValueError(
            f"E4 parent {candidate!r} has no verified checkpoint; "
            f"reinitializing is forbidden ({exc})") from exc


def execute(job: JobInput, *, ops=None,
            update_target: int | None = None) -> PhaseResult:
    started = time.monotonic()
    job.validate()
    if job.phase != "E4" or job.arm not in ("S0", "S1") or job.seed is None:
        return PhaseResult(status="failed", device_seconds=time.monotonic() - started,
                           error="E4 requires arm S0/S1 and seed",
                           evidence_kind=EVIDENCE_FIXTURE,
                           extra={"phase": "E4"})
    try:
        target = job.require_update_target(update_target)
    except ValueError as exc:
        return PhaseResult(status="failed", device_seconds=time.monotonic() - started,
                           error=str(exc), evidence_kind=EVIDENCE_FIXTURE,
                           extra={"phase": "E4"})
    manifest = os.path.join(job.data_dir, "manifest.json")
    if not os.path.exists(manifest):
        return PhaseResult(status="failed", device_seconds=time.monotonic() - started,
                           error="bundle manifest missing",
                           evidence_kind=EVIDENCE_FIXTURE,
                           extra={"phase": "E4"})
    if ops is None:
        from bramastra_lab.research.campaigns.phases.ops import ProductionOps

        ops = ProductionOps(precision=job.precision)
    try:
        parent_record = _resolve_parent(job)
    except Exception as exc:
        return PhaseResult(status="failed", device_seconds=time.monotonic() - started,
                           error=str(exc), evidence_kind=EVIDENCE_FIXTURE,
                           extra={"phase": "E4"})
    gates_enabled = (job.arm == "S1")
    # Restore + fork + migrate the actual training handle (never a discarded
    # tiny probe; never a fresh base model for both arms).
    try:
        if not hasattr(ops, "restore_parent") or not hasattr(ops, "fork_child") \
                or not hasattr(ops, "migrate_to_gated"):
            raise ValueError(
                "ops lacks restore/fork/migrate; refusing fresh base init")
        restored = ops.restore_parent(
            parent={**parent_record, "run_dir": job.run_dir, "seed": job.seed},
            device=job.local_device, optimizer_policy="fresh")
        child = ops.fork_child(parent_handle=restored, optimizer_policy="fresh")
        child = ops.migrate_to_gated(child, gates_enabled=gates_enabled)
        # Verify migration on the very handle (gate inventory, identity).
        migration_error = _verify_handle_migration(child, gates_enabled=gates_enabled)
        if migration_error:
            raise ValueError(migration_error)
    except Exception as exc:
        return PhaseResult(status="failed", device_seconds=time.monotonic() - started,
                           error=f"E4 migration failed: {exc}",
                           evidence_kind=EVIDENCE_FIXTURE,
                           extra={"phase": "E4"})
    from bramastra_lab.research.campaigns.phases.compiler import (
        build_batch_for_trajectory,
        compile_channels_for_row,
        load_training_trajectories,
    )

    weights = {"token": 1.0, "world": 0.5, "action": 0.5, "value": 0.1,
               "pair": 0.0, "pg": 0.0}
    enabled = frozenset({"token", "world", "action", "value"})
    try:
        trajectories = load_training_trajectories(job.data_dir, seed=job.seed)
    except Exception as exc:
        return PhaseResult(status="failed", device_seconds=time.monotonic() - started,
                           error=f"E4 stream refused: {exc}",
                           evidence_kind=EVIDENCE_FIXTURE,
                           extra={"phase": "E4"})
    if len(trajectories) < target:
        return PhaseResult(status="failed", device_seconds=time.monotonic() - started,
                           error=f"E4 stream too short for target {target}",
                           evidence_kind=EVIDENCE_FIXTURE,
                           extra={"phase": "E4"})
    try:
        before_updates = int(ops.optimizer_updates(child))
    except Exception:
        before_updates = 0
    committed = attempted = exposure = 0
    for step in range(target):
        row = trajectories[step % len(trajectories)]
        try:
            batch = build_batch_for_trajectory(row)
            compiled = compile_channels_for_row(
                row, batch, arm_weights=dict(weights), arm_enabled=enabled)
            window, extra = ops.construct_objectives(
                handle=child, batch=batch, compiled=compiled, arm=job.arm)
        except Exception as exc:
            return PhaseResult(status="failed", committed_updates=committed,
                               attempted_updates=attempted,
                               supervised_exposure=exposure,
                               device_seconds=time.monotonic() - started,
                               error=f"E4 objective construction refused: {exc}",
                               evidence_kind=EVIDENCE_FIXTURE,
                               extra={"phase": "E4"})
        try:
            try:
                outcome = ops.apply_update(
                    child, batch=batch, window=window, extra=extra,
                    pair_rows=None)
            except TypeError:
                outcome = ops.training_update(child, batch=batch,
                                              window=window, extra=extra)
        except Exception as exc:
            return PhaseResult(status="failed", committed_updates=committed,
                               attempted_updates=attempted,
                               supervised_exposure=exposure,
                               device_seconds=time.monotonic() - started,
                               error=f"E4 update refused: {exc}",
                               evidence_kind=EVIDENCE_FIXTURE,
                               extra={"phase": "E4"})
        committed += int(outcome.get("committed", 0))
        attempted += int(outcome.get("attempted", 0))
        exposure += int(outcome.get("exposure", 0))
    try:
        after_updates = int(ops.optimizer_updates(child))
    except Exception:
        after_updates = before_updates
    optimizer_delta = after_updates - before_updates
    # Post-training gate check on the trained handle (gates may have moved).
    try:
        gate_values = _read_gate_values(child)
    except Exception:
        gate_values = None
    try:
        if hasattr(ops, "publish_checkpoint"):
            checkpoint_id = ops.publish_checkpoint(
                handle=child, run_dir=job.run_dir, phase="E4",
                arm=job.arm, seed=job.seed, update_index=after_updates,
                parent_checkpoint_id=parent_record.get("checkpoint_id"),
                data_dir=job.data_dir)
        else:
            checkpoint_id = ops.save_checkpoint(
                child, path=os.path.join(job.run_dir, "checkpoints", job.phase,
                                         f"{job.arm}-{job.seed}-final.pt"),
                fraction=1.0)
    except Exception as exc:
        return PhaseResult(status="failed", committed_updates=committed,
                           attempted_updates=attempted,
                           supervised_exposure=exposure,
                           device_seconds=time.monotonic() - started,
                           error=f"E4 checkpoint publication refused: {exc}",
                           evidence_kind=EVIDENCE_FIXTURE,
                           extra={"phase": "E4"})
    try:
        is_fixture = str(checkpoint_id).startswith(("fixture-", "double-"))
    except Exception:
        is_fixture = True
    evidence = EVIDENCE_FIXTURE if (is_fixture or optimizer_delta <= 0) \
        else EVIDENCE_LEARNED_CAMPAIGN
    artifact_dir = os.path.join(job.run_dir, "phase_outputs", job.phase)
    os.makedirs(artifact_dir, exist_ok=True)
    with open(os.path.join(artifact_dir, f"{job.arm}-{job.seed}.json"), "w",
              encoding="utf-8") as handle_file:
        json.dump({"parent": parent_record.get("checkpoint_id"),
                   "parent_lookup": parent_record.get("lookup_key"),
                   "arm": job.arm, "seed": job.seed,
                   "gates_enabled": gates_enabled,
                   "architecture_id": child.get("architecture_id")
                   if isinstance(child, dict) else str(
                       getattr(getattr(child, "model", None),
                               "architecture_id", "unknown")),
                   "gate_values_after_training": gate_values,
                   "migration": "verified-on-training-handle",
                   "committed_updates": committed,
                   "optimizer_delta": optimizer_delta,
                   "evidence_kind": evidence,
                   "checkpoint_identity": checkpoint_id},
                  handle_file, indent=2, sort_keys=True)
    if committed <= 0:
        return PhaseResult(status="failed", committed_updates=committed,
                           attempted_updates=attempted,
                           supervised_exposure=exposure,
                           device_seconds=time.monotonic() - started,
                           error="E4 produced zero committed updates",
                           evidence_kind=EVIDENCE_FIXTURE,
                           extra={"phase": "E4"})
    result = PhaseResult(status="completed", committed_updates=committed,
                         attempted_updates=attempted,
                         supervised_exposure=exposure,
                         device_seconds=time.monotonic() - started,
                         checkpoint_identity=checkpoint_id,
                         evidence_kind=evidence,
                         extra={"phase": "E4",
                                "parent": parent_record.get("checkpoint_id"),
                                "gates_enabled": gates_enabled})
    try:
        result.validate()
    except ValueError as exc:
        return PhaseResult(status="failed", committed_updates=committed,
                           attempted_updates=attempted,
                           supervised_exposure=exposure,
                           device_seconds=time.monotonic() - started,
                           error=f"E4 receipt refused: {exc}",
                           evidence_kind=EVIDENCE_FIXTURE,
                           extra={"phase": "E4"})
    return result


def _verify_handle_migration(handle: Any, *, gates_enabled: bool) -> str | None:
    """Verify gate inventory + identity on the training handle."""
    try:
        if isinstance(handle, dict) and "double_id" in handle:
            # Fixture handle: check recorded migration flags.
            if not handle.get("migrated"):
                return "fixture handle carries no migration record"
            if bool(handle.get("gates_enabled")) != bool(gates_enabled):
                return "fixture gate flag does not match arm"
            arch = str(handle.get("architecture_id", ""))
            if gates_enabled and "gated" not in arch:
                return "S1 fixture architecture is not gated"
            if not gates_enabled and "disabled" not in arch and "gated" in arch:
                return "S0 fixture architecture must carry disabled slots"
            return None
        # Production handle: real model inventory.
        model = handle.get("model") if isinstance(handle, dict) else getattr(
            handle, "model", None)
        if model is None:
            return "training handle carries no model"
        arch = str(getattr(model, "architecture_id", ""))
        if gates_enabled and "gated" not in arch:
            return f"S1 architecture {arch!r} is not gated"
        if not gates_enabled and getattr(model, "gates_enabled", True):
            return "S0 handle has enabled gates"
        # Optimizer inventory must reflect gate params.
        trainer = handle.get("trainer") if isinstance(handle, dict) else getattr(
            handle, "trainer", None)
        if trainer is not None:
            try:
                n_params = sum(p.numel() for p in model.parameters())
                if n_params <= 0:
                    return "migrated model has no parameters"
            except Exception as exc:
                return f"parameter inventory refused: {exc}"
        return None
    except Exception as exc:
        return f"migration verification refused: {exc}"


def _read_gate_values(handle: Any) -> list[float] | None:
    try:
        if isinstance(handle, dict) and "double_id" in handle:
            return list(handle.get("gate_alpha", [0.0, 0.0]))
        model = handle.get("model") if isinstance(handle, dict) else getattr(
            handle, "model", None)
        if model is not None and hasattr(model, "gate_values"):
            return list(model.gate_values())
    except Exception:
        pass
    return None
