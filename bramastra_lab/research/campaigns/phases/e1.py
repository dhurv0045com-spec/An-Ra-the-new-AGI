"""E1 executor: paired from-scratch formation (D4 + real execution contracts).

Random initialization for paired seeds via frozen K8 campaign config
(context 512, frozen optimizer); A/B objective package from the frozen
protocol; identical per-seed starting state hash and deterministic
exact-split stream; real compiled world/action/value/pair targets;
checkpoints at 25/50/75/100% via real publication; held-out evaluation via
free generation + independent verifier (EOS is not accuracy).
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any

from bramastra_lab.research.campaigns.phases.types import (
    EVIDENCE_FIXTURE,
    EVIDENCE_LEARNED_CAMPAIGN,
    EVIDENCE_LOCAL_INTEGRATION,
    JobInput,
    PhaseResult,
)

ARM_WEIGHTS = {
    "A": {"token": 1.0, "world": 0.0, "action": 0.0, "value": 0.0,
          "pair": 0.0, "pg": 0.0},
    "B": {"token": 1.0, "world": 0.5, "action": 0.5, "value": 0.1,
          "pair": 0.1, "pg": 0.0},
}


def _write_partial_receipt(job: JobInput, *, committed: int, attempted: int,
                           exposure: int, checkpoint_ids: list[str],
                           parent_for_next: str | None, step: int,
                           stream_id: str, noop_boundaries: int = 0,
                           authority: str | None = None) -> None:
    """Best-effort partial receipt: interruption never loses completed steps.

    Written alongside every mid-phase checkpoint publication (~200 steps).
    A killed session resumes from ledger receipts, but this file is the
    human-readable proof of how far the arm reached. Never raises: receipt
    IO must not fail training.
    """
    try:
        artifact_dir = os.path.join(job.run_dir, "phase_outputs", job.phase)
        os.makedirs(artifact_dir, exist_ok=True)
        partial = {"partial": True, "job": {"phase": job.phase, "arm": job.arm,
                                            "seed": job.seed, "job_id": job.job_id,
                                            "update_target": job.update_target},
                   "stream_id": stream_id,
                   "completed_steps": step + 1,
                   "committed_updates": committed, "attempted_updates": attempted,
                   "supervised_exposure": exposure,
                   "noop_boundaries": noop_boundaries,
                   "authority": authority,
                   "checkpoint_identities": list(checkpoint_ids),
                   "checkpoint_identity": parent_for_next}
        with open(os.path.join(artifact_dir, f"{job.arm}-{job.seed}-partial.json"),
                  "w", encoding="utf-8") as handle_file:
            json.dump(partial, handle_file, indent=2, sort_keys=True)
    except Exception:
        pass
ARM_ENABLED = {
    "A": frozenset({"token"}),
    "B": frozenset({"token", "world", "action", "value", "pair"}),
}


def _hash_snapshot(snapshot: Any) -> str:
    try:
        import torch

        if isinstance(snapshot, dict):
            digest = hashlib.sha256()
            for key in sorted(snapshot.keys()):
                tensor = snapshot[key]
                try:
                    digest.update(key.encode())
                    digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
                except Exception:
                    digest.update(str(snapshot[key])[:256].encode())
            return digest.hexdigest()
    except Exception:
        pass
    return hashlib.sha256(str(snapshot).encode()).hexdigest()[:64]


def _derive_evidence_kind(*, committed: int, optimizer_delta: int,
                          checkpoint_id: str | None) -> str:
    if committed <= 0 or not checkpoint_id:
        return EVIDENCE_FIXTURE
    is_fixture_id = str(checkpoint_id).startswith(
        ("fixture-", "double-", "double-ckpt-"))
    if optimizer_delta > 0 and not is_fixture_id:
        return EVIDENCE_LEARNED_CAMPAIGN
    # Local integration (real target construction + backward, no step) or
    # fixture double path: never learned.
    if is_fixture_id:
        return EVIDENCE_FIXTURE
    return EVIDENCE_LOCAL_INTEGRATION


def execute(job: JobInput, *, ops=None,
            update_target: int | None = None) -> PhaseResult:
    """Run one E1 arm/seed job with actual counts (test doubles injectable)."""
    started = time.monotonic()
    job.validate()
    if job.phase != "E1" or job.arm not in ("A", "B") or job.seed is None:
        return PhaseResult(status="failed", device_seconds=time.monotonic() - started,
                           error="E1 requires arm A/B and seed",
                           evidence_kind=EVIDENCE_FIXTURE,
                           extra={"phase": "E1"})
    try:
        target = job.require_update_target(update_target)
    except ValueError as exc:
        return PhaseResult(status="failed", device_seconds=time.monotonic() - started,
                           error=str(exc), evidence_kind=EVIDENCE_FIXTURE,
                           extra={"phase": "E1"})
    manifest = os.path.join(job.data_dir, "manifest.json")
    if not os.path.exists(manifest):
        return PhaseResult(status="failed", device_seconds=time.monotonic() - started,
                           error=f"bundle manifest missing: {manifest}",
                           evidence_kind=EVIDENCE_FIXTURE,
                           extra={"phase": "E1"})
    if ops is None:
        from bramastra_lab.research.campaigns.phases.ops import ProductionOps

        ops = ProductionOps(precision=job.precision)
    from bramastra_lab.research.campaigns.phases.compiler import (
        build_batch_for_trajectory,
        build_pair_rows,
        compile_channels_for_row,
        load_training_trajectories,
    )

    # Frozen campaign init (never bare development/256). All supported ops
    # expose initialize_random (same validated interface, no type branches).
    # Authority binding (O01): real trainers admit only through the
    # supervisor reservation; doubles take the recorded zero-update path.
    try:
        handle = ops.initialize_random(seed=job.seed, device=job.local_device,
                                       reservation=None)
        from bramastra_lab.research.campaigns.phases.session import (
            bind_job_reservation, job_reservation_record)
        authority = bind_job_reservation(handle, job,
                                         remaining_updates=target)
        reservation = None
        if authority == "bound":
            # Stepping additionally requires the live ledger allocation
            # (backstop: well-formed fields alone never authorize steps).
            reservation = job_reservation_record(
                job, remaining_updates=target)
            # Fail fast on dead authority: a job that would noop every step
            # must refuse here with the concrete ledger cause, not burn its
            # wall and die later on checkpoint confusion.
            from bramastra_lab.research.campaigns.phases.session import (
                stepping_readiness)
            readiness = stepping_readiness(job.run_dir, reservation)
            if not readiness.get("allowed", False):
                return PhaseResult(
                    status="failed", committed_updates=0, attempted_updates=0,
                    supervised_exposure=0,
                    device_seconds=time.monotonic() - started,
                    error="E1 stepping refused: "
                          f"{readiness.get('reason', 'unknown')}",
                    evidence_kind=EVIDENCE_FIXTURE, extra={"phase": "E1"})
    except Exception as exc:
        return PhaseResult(status="failed", device_seconds=time.monotonic() - started,
                           error=f"E1 authority refused: {exc}",
                           evidence_kind=EVIDENCE_FIXTURE,
                           extra={"phase": "E1"})
    except Exception as exc:
        return PhaseResult(status="failed", device_seconds=time.monotonic() - started,
                           error=f"E1 authority refused: {exc}",
                           evidence_kind=EVIDENCE_FIXTURE,
                           extra={"phase": "E1"})
    except Exception as exc:
        return PhaseResult(status="failed", device_seconds=time.monotonic() - started,
                           error=f"E1 init refused: {exc}",
                           evidence_kind=EVIDENCE_FIXTURE,
                           extra={"phase": "E1"})
    try:
        init_snapshot = ops.snapshot_state(handle)
    except Exception:
        init_snapshot = None
    init_hash = _hash_snapshot(init_snapshot) if init_snapshot is not None else "unavailable"
    try:
        before_updates = int(ops.optimizer_updates(handle))
    except Exception:
        before_updates = 0
    # Exact-split deterministic stream (same seed -> same batches across arms).
    try:
        trajectories = load_training_trajectories(
            job.data_dir, seed=job.seed)
    except Exception as exc:
        return PhaseResult(status="failed", device_seconds=time.monotonic() - started,
                           error=f"E1 stream refused: {exc}",
                           evidence_kind=EVIDENCE_FIXTURE,
                           extra={"phase": "E1"})
    if len(trajectories) < target:
        return PhaseResult(status="failed", device_seconds=time.monotonic() - started,
                           error=f"bundle stream too short ({len(trajectories)}) "
                                 f"for E1 update target {target}; failing, never "
                                 "synthesizing rows",
                           evidence_kind=EVIDENCE_FIXTURE,
                           extra={"phase": "E1"})
    stream_id = f"e1-{job.seed}-{len(trajectories)}-training-exact"
    weights = ARM_WEIGHTS[job.arm]
    enabled = ARM_ENABLED[job.arm]
    committed = attempted = exposure = 0
    noop_boundaries = 0
    checkpoint_ids: list[str] = []
    parent_for_next: str | None = None
    fractions = {max(1, target * p // 100) for p in (25, 50, 75, 100)}
    # Declared mid-phase save cadence (owner request: "every ~200 steps"):
    # a checkpoint is published each time 200 loop steps elapse or 200
    # optimizer updates accumulate since the last publication, in addition
    # to the registered 25/50/75/100% lineage fractions. A session that
    # dies loses at most this many steps/updates of the active arm.
    checkpoint_every_updates = 200
    checkpoint_every_steps = 200
    committed_since_checkpoint = 0
    steps_since_checkpoint = 0
    # Pair support: need distinct-answer lookahead for B.
    for step in range(target):
        row = trajectories[step % len(trajectories)]
        try:
            batch = build_batch_for_trajectory(row)
            compiled = compile_channels_for_row(
                row, batch, arm_weights=dict(weights), arm_enabled=enabled)
        except Exception as exc:
            return PhaseResult(status="failed", committed_updates=committed,
                               attempted_updates=attempted,
                               supervised_exposure=exposure,
                               device_seconds=time.monotonic() - started,
                               error=f"E1 target compilation refused at step "
                                     f"{step}: {exc}",
                               evidence_kind=EVIDENCE_FIXTURE,
                               extra={"phase": "E1"})
        # Same validated interface for prod and doubles (no type branch).
        try:
            window, extra = ops.construct_objectives(
                handle=handle, batch=batch, compiled=compiled, arm=job.arm)
        except Exception as exc:
            return PhaseResult(status="failed", committed_updates=committed,
                               attempted_updates=attempted,
                               supervised_exposure=exposure,
                               device_seconds=time.monotonic() - started,
                               error=f"E1 objective construction refused: {exc}",
                               evidence_kind=EVIDENCE_FIXTURE,
                               extra={"phase": "E1"})
        pair_rows = None
        if "pair" in enabled and float(weights.get("pair", 0.0)) > 0:
            # Find a real distinct-answer partner (never invented).
            partner = None
            for lookahead in range(1, len(trajectories)):
                candidate = trajectories[(step + lookahead) % len(trajectories)]
                if str(candidate.get("answer")) != str(row.get("answer")):
                    partner = candidate
                    break
            if partner is None:
                return PhaseResult(
                    status="failed", committed_updates=committed,
                    attempted_updates=attempted, supervised_exposure=exposure,
                    device_seconds=time.monotonic() - started,
                    error="E1 pair enabled but no distinct-answer partner in "
                          "training split; refusing silent zero pair loss",
                    evidence_kind=EVIDENCE_FIXTURE, extra={"phase": "E1"})
            try:
                pair_rows = build_pair_rows(row, partner)
            except Exception as exc:
                return PhaseResult(
                    status="failed", committed_updates=committed,
                    attempted_updates=attempted, supervised_exposure=exposure,
                    device_seconds=time.monotonic() - started,
                    error=f"E1 pair row construction refused: {exc}",
                    evidence_kind=EVIDENCE_FIXTURE, extra={"phase": "E1"})
        try:
            # Real steps only under a live ledger allocation; otherwise the
            # admission-checked no-op boundary (O01 backstop: fields alone
            # never authorize local training). Doubles record fixture counts.
            from bramastra_lab.research.campaigns.phases.session import (
                step_or_noop)
            outcome = step_or_noop(
                ops, handle, job, batch=batch, window=window,
                extra=extra, pair_rows=pair_rows, reservation=reservation)
        except Exception as exc:
            return PhaseResult(status="failed", committed_updates=committed,
                               attempted_updates=attempted,
                               supervised_exposure=exposure,
                               device_seconds=time.monotonic() - started,
                               error=f"E1 update refused at step {step}: {exc}",
                               evidence_kind=EVIDENCE_FIXTURE,
                               extra={"phase": "E1"})
        committed += int(outcome.get("committed", 0))
        attempted += int(outcome.get("attempted", 0))
        exposure += int(outcome.get("exposure", 0))
        noop_boundaries += 1 if outcome.get("boundary") == "noop" else 0
        committed_since_checkpoint += int(outcome.get("committed", 0))
        steps_since_checkpoint += 1
        if (step + 1) in fractions or (step + 1) == target or (
                steps_since_checkpoint >= checkpoint_every_steps) or (
                committed_since_checkpoint >= checkpoint_every_updates
                and committed_since_checkpoint > 0):
            try:
                checkpoint_id = ops.publish_checkpoint(
                    handle=handle, run_dir=job.run_dir, phase="E1",
                    arm=job.arm, seed=job.seed,
                    update_index=int(ops.optimizer_updates(handle)),
                    parent_checkpoint_id=parent_for_next,
                    data_dir=job.data_dir)
                checkpoint_ids.append(checkpoint_id)
                parent_for_next = checkpoint_id
            except Exception as exc:
                return PhaseResult(
                    status="failed", committed_updates=committed,
                    attempted_updates=attempted, supervised_exposure=exposure,
                    device_seconds=time.monotonic() - started,
                    error=f"E1 checkpoint publication refused: {exc}",
                    evidence_kind=EVIDENCE_FIXTURE, extra={"phase": "E1"})
            committed_since_checkpoint = 0
            steps_since_checkpoint = 0
            _write_partial_receipt(job, committed=committed, attempted=attempted,
                                   exposure=exposure, checkpoint_ids=checkpoint_ids,
                                   parent_for_next=parent_for_next, step=step,
                                   stream_id=stream_id,
                                   noop_boundaries=noop_boundaries,
                                   authority=authority)
    # Final checkpoint (100%) if not already published.
    try:
        after_updates = int(ops.optimizer_updates(handle))
    except Exception:
        after_updates = before_updates
    optimizer_delta = after_updates - before_updates
    final_id = parent_for_next
    if not final_id:
        try:
            final_id = ops.publish_checkpoint(
                handle=handle, run_dir=job.run_dir, phase="E1",
                arm=job.arm, seed=job.seed, update_index=after_updates,
                parent_checkpoint_id=parent_for_next,
                data_dir=job.data_dir)
            checkpoint_ids.append(final_id)
        except Exception as exc:
            return PhaseResult(status="failed", committed_updates=committed,
                               attempted_updates=attempted,
                               supervised_exposure=exposure,
                               device_seconds=time.monotonic() - started,
                               error=f"E1 final publication refused: {exc}",
                               evidence_kind=EVIDENCE_FIXTURE,
                               extra={"phase": "E1"})
    # Held-out evaluation via free generation + independent verifier.
    eval_report = _evaluate_heldout(job, ops, handle)
    evidence = _derive_evidence_kind(committed=committed,
                                     optimizer_delta=optimizer_delta,
                                     checkpoint_id=final_id)
    artifact_dir = os.path.join(job.run_dir, "phase_outputs", job.phase)
    os.makedirs(artifact_dir, exist_ok=True)
    receipt = {"job": {"phase": job.phase, "arm": job.arm, "seed": job.seed,
                       "slot": job.slot, "parent": job.parent,
                       "job_id": job.job_id,
                       "update_target": target,
                       "reservation_id": job.reservation_id,
                       "allocation_id": job.allocation_id},
               "authority": authority,
               "noop_boundaries": noop_boundaries,
               "stream_id": stream_id,
               "treatment_weights": weights,
               "enabled_terms": sorted(enabled),
               "init_state_hash": init_hash,
               "committed_updates": committed, "attempted_updates": attempted,
               "optimizer_delta": optimizer_delta,
               "checkpoint_identities": checkpoint_ids,
               "checkpoint_identity": final_id,
               "evaluation": eval_report,
               "evidence_kind": evidence}
    with open(os.path.join(artifact_dir, f"{job.arm}-{job.seed}.json"), "w",
              encoding="utf-8") as handle_file:
        json.dump(receipt, handle_file, indent=2, sort_keys=True)
    if committed <= 0:
        return PhaseResult(status="failed", committed_updates=committed,
                           attempted_updates=attempted,
                           supervised_exposure=exposure,
                           device_seconds=time.monotonic() - started,
                           error="E1 produced zero committed updates",
                           evidence_kind=EVIDENCE_FIXTURE,
                           extra={"phase": "E1", "stream_id": stream_id})
    result = PhaseResult(status="completed", committed_updates=committed,
                         attempted_updates=attempted,
                         supervised_exposure=exposure,
                         device_seconds=time.monotonic() - started,
                         checkpoint_identity=final_id,
                         evidence_kind=evidence,
                         extra={"phase": "E1", "stream_id": stream_id,
                                "treatment_weights": weights,
                                "init_state_hash": init_hash,
                                "checkpoint_identities": checkpoint_ids,
                                "authority": authority,
                                "evaluation": eval_report})
    try:
        result.validate()
    except ValueError as exc:
        return PhaseResult(status="failed", committed_updates=committed,
                           attempted_updates=attempted,
                           supervised_exposure=exposure,
                           device_seconds=time.monotonic() - started,
                           error=f"E1 receipt refused: {exc}",
                           evidence_kind=EVIDENCE_FIXTURE,
                           extra={"phase": "E1"})
    return result


def _evaluate_heldout(job: JobInput, ops, handle, *,
                      heldout_eval_rows: int = 32) -> dict[str, Any]:
    """Held-out development-measurement evaluation (verifier decides).

    The evaluation scale is DECLARED (heldout_eval_rows), never a silent
    sample; prompts use the exact training representation via
    `prompt_tokens_for_row` (no JSON/char/token slicing); generation allows
    the full typed answer length. The declared cap and evaluated count are
    recorded in the receipt.
    """
    import glob as _glob

    episode_dir = os.path.join(job.data_dir, "episodes")
    heldout_rows: list[dict[str, Any]] = []
    for path in sorted(_glob.glob(os.path.join(episode_dir, "*.jsonl"))):
        base = os.path.basename(path)
        # Exact pool suffix (never substring): development-measurement only.
        # Sealed-confirmation is never used for checkpoint selection.
        if not base.endswith("-development-measurement.jsonl"):
            continue
        try:
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    if row.get("pool") == "development-measurement":
                        heldout_rows.append(row)
        except OSError:
            continue
        if len(heldout_rows) >= heldout_eval_rows:
            break
    heldout_rows = heldout_rows[:heldout_eval_rows]
    if not heldout_rows:
        return {"evaluated": 0, "declared_eval_rows": heldout_eval_rows,
                "note": "no heldout rows; evaluation empty"}
    try:
        from bramastra_lab.research.data.k8_bundle import FAMILY_VERIFIERS
    except Exception:
        FAMILY_VERIFIERS = {}
    evaluated = 0
    successes = 0
    details: list[dict[str, Any]] = []
    for row in heldout_rows:
        family = str(row.get("family", ""))
        verifier = FAMILY_VERIFIERS.get(family)
        if verifier is None:
            continue
        # Exact training representation (boundary + goal + received
        # history) — never a sliced JSON blob of the goal alone.
        from bramastra_lab.research.campaigns.phases.compiler import (
            prompt_tokens_for_row)
        try:
            prompt = prompt_tokens_for_row(row)
        except Exception as exc:
            details.append({"mechanism_id": row.get("mechanism_id"),
                            "family": family,
                            "error": f"prompt compilation refused: {exc}",
                            "success": False})
            continue
        try:
            outcome = ops.evaluate_episode(
                handle=handle,
                episode={"mechanism": row, "verifier": verifier,
                         "prompt_tokens": prompt, "max_new_tokens": 24})
            answer = str(outcome.get("answer", ""))
            stopped = bool(outcome.get("stopped_on_eos", False))
            success = outcome.get("success")
            if success is None:
                # Fall back to verifier on the generated answer.
                try:
                    success = bool(verifier(
                        row, {"answer": answer, "sum": answer}))
                except Exception:
                    success = False
            evaluated += 1
            successes += 1 if success else 0
            details.append({"mechanism_id": row.get("mechanism_id"),
                            "family": family, "answer": answer,
                            "stopped_on_eos": stopped,
                            "success": bool(success),
                            "model_calls": int(outcome.get("model_calls", 1)),
                            "cost": float(outcome.get("cost", 1.0))})
        except Exception as exc:
            details.append({"mechanism_id": row.get("mechanism_id"),
                            "family": family, "error": str(exc)[:120],
                            "success": False})
    return {"evaluated": evaluated, "successes": successes,
            "declared_eval_rows": heldout_eval_rows,
            "stopped_eos_count": sum(1 for d in details if d.get("stopped_on_eos")),
            "note": "EOS is stop evidence only; success comes from the "
                    "independent family verifier",
            "cases": details}
