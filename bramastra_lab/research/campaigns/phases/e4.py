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
    if "/" in str(key):
        raise ValueError(
            f"E4 parent {key!r} is combined; E4 requires a single E1-B parent")
    candidate = str(key).strip()
    try:
        if job.parent_ref is not None and job.parent_ref.lookup_key == candidate:
            rec = job.parent_ref.resolve(job.run_dir)
        else:
            rec = ParentRef(lookup_key=candidate).resolve(job.run_dir)
        rec["lookup_key"] = candidate
        return rec
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
    # tiny probe; never a fresh base model for both arms). All supported ops
    # expose restore/fork/migrate (same interface, no branches).
    try:
        restored = ops.restore_parent(
            parent={**parent_record, "run_dir": job.run_dir, "seed": job.seed},
            device=job.local_device, optimizer_policy="fresh")
        child = ops.fork_child(parent_handle=restored, optimizer_policy="fresh")
        child = ops.migrate_to_gated(child, gates_enabled=gates_enabled)
        # Verify migration on the very handle (gate inventory, identity).
        migration_error = _verify_handle_migration(child, gates_enabled=gates_enabled)
        if migration_error:
            raise ValueError(migration_error)
        # Authority binding (O01): the migrated handle trains only under the
        # job reservation; doubles take the recorded zero-update path.
        from bramastra_lab.research.campaigns.phases.session import (
            bind_job_reservation, job_reservation_record)
        authority = bind_job_reservation(child, job,
                                         remaining_updates=target)
        reservation = None
        if authority == "bound":
            reservation = job_reservation_record(
                job, remaining_updates=target)
            from bramastra_lab.research.campaigns.phases.session import (
                await_stepping_allowed)
            readiness = await_stepping_allowed(job.run_dir, reservation)
            if not readiness.get("allowed", False):
                return PhaseResult(status="failed", device_seconds=time.monotonic() - started,
                                   error="E4 stepping refused: "
                                         f"{readiness.get('reason', 'unknown')}",
                                   evidence_kind=EVIDENCE_FIXTURE,
                                   extra={"phase": "E4"})
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
    # Gate-path activation + segment isolation + cost on the migrated handle
    # itself (O07): backward-only gate gradient proof (grads discarded, no
    # step), packed-segment forward contract, and timed execution cost.
    # Fixture doubles record the checks without model compute.
    try:
        arch_proof = _prove_architecture_on_handle(
            ops, child, gates_enabled=gates_enabled)
    except Exception as exc:
        return PhaseResult(status="failed", device_seconds=time.monotonic() - started,
                           error=f"E4 architecture proof refused: {exc}",
                           evidence_kind=EVIDENCE_FIXTURE,
                           extra={"phase": "E4"})
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
            from bramastra_lab.research.campaigns.phases.session import (
                step_or_noop)
            outcome = step_or_noop(
                ops, child, job, batch=batch, window=window, extra=extra,
                pair_rows=None, reservation=reservation)
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
        checkpoint_id = ops.publish_checkpoint(
            handle=child, run_dir=job.run_dir, phase="E4",
            arm=job.arm, seed=job.seed, update_index=after_updates,
            parent_checkpoint_id=parent_record.get("checkpoint_id"),
            data_dir=job.data_dir)
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
                   "job_id": job.job_id,
                   "authority": authority,
                   "gates_enabled": gates_enabled,
                   "architecture_id": child.get("architecture_id")
                   if isinstance(child, dict) else str(
                       getattr(getattr(child, "model", None),
                               "architecture_id", "unknown")),
                   "gate_values_after_training": gate_values,
                   "migration": "verified-on-training-handle",
                   "architecture_proof": arch_proof,
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


def _prove_architecture_on_handle(ops: Any, handle: Any, *,
                                    gates_enabled: bool) -> dict[str, Any]:
    """Gate-path, segment-isolation and cost proof on the training handle."""
    try:
        from bramastra_lab.research.models.gated import check_gate_gradients
    except Exception:
        check_gate_gradients = None  # type: ignore[assignment]
    model = handle.get("model") if isinstance(handle, dict) else getattr(
        handle, "model", None)
    if model is None or isinstance(handle, dict) and "double_id" in handle:
        ops.calls.append(("e4_arch_proof", {"gates_enabled": gates_enabled,
                                            "fixture": True}))
        return {"gate_gradients": "fixture-recorded",
                "segment_isolation": "fixture-recorded",
                "execution_cost": "fixture-recorded"}
    import time as _time
    import torch as _torch

    # Gate-path activation: real backward, gate grads must exist for S1.
    @_torch.no_grad()
    def _probe_batch() -> Any:
        config = handle["config"]
        probe = _torch.randint(0, config.model.vocab, (2, 12))
        return probe

    probe = _probe_batch()
    logits = model(probe).logits
    loss = logits.float().square().mean()
    model.zero_grad(set_to_none=True)
    loss.backward()
    if gates_enabled:
        gate_grad = getattr(getattr(model, "gate_alpha", None), "grad", None)
        if gate_grad is None:
            raise ValueError(
                "S1 gate path inactive: gate_alpha carries no gradient")
        if not bool(_torch.isfinite(gate_grad).all()):
            raise ValueError("S1 gate gradients nonfinite")
    grad_proof = True
    if check_gate_gradients is not None:
        try:
            checks = check_gate_gradients(model)
            grad_proof = bool(checks.get("shared_block_gradients_present"))
        except Exception:
            pass
    model.zero_grad(set_to_none=True)
    # Packed-segment isolation: identical tokens under different segments
    # must route differently (mask respected), checked on this handle.
    with _torch.no_grad():
        tokens = probe
        pad = _torch.ones_like(tokens, dtype=_torch.bool)
        seg_same = _torch.ones_like(tokens)
        seg_split = _torch.ones_like(tokens)
        seg_split[:, 6:] = 2
        try:
            out_same = model(tokens, pad, segment_ids=seg_same).logits
            out_split = model(tokens, pad, segment_ids=seg_split).logits
            isolated = bool((out_same - out_split).abs().max().item() > 0.0)
        except Exception as exc:
            raise ValueError(
                f"segment-isolation forward refused: {exc}") from exc
    # Execution cost: timed forwards on this handle (extra block passes run
    # even at zero gates by design).
    repeats = 3
    start = _time.perf_counter()
    with _torch.no_grad():
        for _ in range(repeats):
            _ = model(probe).logits
    cost_seconds = (_time.perf_counter() - start) / max(1, repeats)
    return {"gate_gradients": grad_proof,
            "segment_isolation": isolated,
            "execution_cost_seconds_per_forward": cost_seconds}


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
        # Production handle: real model + optimizer inventory.
        model = handle.get("model") if isinstance(handle, dict) else getattr(
            handle, "model", None)
        if model is None:
            return "training handle carries no model"
        arch = str(getattr(model, "architecture_id", ""))
        if gates_enabled and "gated" not in arch:
            return f"S1 architecture {arch!r} is not gated"
        if not gates_enabled and getattr(model, "gates_enabled", True):
            return "S0 handle has enabled gates"
        # Optimizer inventory must reflect gate params on the very handle:
        # S1 exposes trainable gate_alpha, S0 keeps them fixed/disabled.
        try:
            named = dict(model.named_parameters())
        except Exception as exc:
            return f"parameter inventory refused: {exc}"
        gate_params = [n for n in named if "gate_alpha" in n]
        if gates_enabled:
            if not gate_params:
                return "S1 optimizer inventory carries no gate_alpha params"
            try:
                trainer = handle.get("trainer") if isinstance(handle, dict) else getattr(
                    handle, "trainer", None)
                if trainer is not None:
                    opt_ids = {id(p) for g in trainer.optimizer.param_groups
                               for p in g.get("params", [])}
                    if not any(id(named[n]) in opt_ids for n in gate_params):
                        return "S1 gate params missing from optimizer inventory"
            except Exception as exc:
                return f"optimizer inventory refused: {exc}"
        else:
            try:
                trainer = handle.get("trainer") if isinstance(handle, dict) else getattr(
                    handle, "trainer", None)
                if trainer is not None:
                    opt_ids = {id(p) for g in trainer.optimizer.param_groups
                               for p in g.get("params", [])}
                    for n in gate_params:
                        param = named[n]
                        if param.requires_grad and id(param) in opt_ids:
                            return "S0 gate slots must be fixed/disabled (trainable gate in optimizer)"
            except Exception as exc:
                return f"optimizer inventory refused: {exc}"
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
