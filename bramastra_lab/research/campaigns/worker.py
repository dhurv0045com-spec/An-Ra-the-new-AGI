"""Campaign worker (I05/R01/R05): one process per GPU with explicit device
visibility set BEFORE torch import. Each worker runs one phase/arm on its
assigned device and checkpoints at boundaries.

Fail-closed: unknown or unimplemented phases refuse (never inherit success).
E1-E5 require qualified E0 evidence (enforced by the runner gate) and a real
executor; without one they return refused_missing_executor with an explicit
reason. E6 validates export requirements. E0 uses one canonical update
function in both uninterrupted and resume branches with a single scaled
backward boundary, strong state identity and separate-process restore proof.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from typing import Any

SUPPORTED_PHASES = frozenset({"E0", "E1", "E2", "E3", "E4", "E5", "E6"})
E0_MIN_CONTEXT = 256


def _e0_probe_config(profile: str):
    """Keep the lightweight E0 model but fit its complete typed probe prompt.

    The tiny profile's default 128-token context is too short for the
    canonical public-state prefix plus the two real action candidates used
    by the resume-equivalence test.  Increasing context does not increase
    decoder parameter count and keeps uninterrupted/resumed configs equal.
    """
    from bramastra_lab.research.config import BuildConfig

    base = BuildConfig.from_dict({"model": {"profile": profile}})
    if base.model.max_seq >= E0_MIN_CONTEXT:
        return base
    return BuildConfig.from_dict({
        "model": {"profile": profile, "max_seq": E0_MIN_CONTEXT}})


def run_worker_phase(*, phase: str, device: str, arm: str | None,
                     seed: int | None, data_dir: str, run_dir: str,
                     precision: str, deadline: float,
                     physical_device: str | None = None,
                     slot: int | None = None,
                     parent: str | None = None,
                     job_id: str | None = None,
                     update_target: int | None = None,
                     eval_cases: int | None = None,
                     tasks_per_block: int | None = None,
                     reservation_id: str | None = None,
                     allocation_id: str | None = None,
                     deadline_unix: float | None = None,
                     remaining_updates: int | None = None,
                     learning_boundary: str | None = None) -> dict[str, Any]:
    """Execute one phase on one device (D4: real executors behind dispatch).

    Runs in a fresh spawn subprocess with CUDA_VISIBLE_DEVICES set before
    torch import. `device` is the worker-LOCAL device (e.g. cuda:0 after
    isolation); `physical_device` preserves the ledger identity (e.g. cuda:1).
    Each phase dispatches to its repository executor module; expensive model
    operations use production ops on GPU and explicit test doubles in tests.
    """
    started = time.monotonic()
    physical = physical_device or device
    if phase not in SUPPORTED_PHASES:
        return {
            "phase": phase, "device": device, "arm": arm, "seed": seed,
            "status": "failed", "error": f"unknown phase {phase!r}; refusing",
            "committed_updates": 0, "attempted_updates": 0,
            "supervised_exposure": 0, "device_seconds": time.monotonic() - started,
            "checkpoint_identity": None,
        }
    if time.time() > deadline:
        return {
            "phase": phase, "device": device, "arm": arm, "seed": seed,
            "status": "failed", "error": "deadline already exceeded; refusing",
            "committed_updates": 0, "attempted_updates": 0,
            "supervised_exposure": 0, "device_seconds": time.monotonic() - started,
            "checkpoint_identity": None,
        }
    if phase == "E0":
        return _run_e0(device=device, arm=arm, seed=seed or 1701,
                       data_dir=data_dir, run_dir=run_dir,
                       precision=precision, deadline=deadline)
    if phase in ("E1", "E2", "E3", "E4", "E5"):
        return _dispatch_phase_executor(
            phase=phase, device=device, physical_device=physical,
            arm=arm, seed=seed, slot=slot, parent=parent,
            job_id=job_id,
            update_target=update_target, eval_cases=eval_cases,
            tasks_per_block=tasks_per_block,
            reservation_id=reservation_id, allocation_id=allocation_id,
            deadline_unix=deadline_unix,
            remaining_updates=remaining_updates,
            learning_boundary=learning_boundary,
            data_dir=data_dir, run_dir=run_dir, precision=precision,
            deadline=deadline, started=started)
    if phase == "E6":
        return _run_e6(device=device, data_dir=data_dir, run_dir=run_dir,
                       deadline=deadline, started=started)
    return {
        "phase": phase, "device": device, "arm": arm, "seed": seed,
        "status": "failed", "error": "unreachable branch",
        "committed_updates": 0, "attempted_updates": 0,
        "supervised_exposure": 0, "device_seconds": time.monotonic() - started,
        "checkpoint_identity": None,
    }


def _dispatch_phase_executor(*, phase: str, device: str, physical_device: str,
                             arm: str | None, seed: int | None,
                             slot: int | None, parent: str | None,
                             job_id: str | None = None,
                             update_target: int | None = None,
                             eval_cases: int | None = None,
                             tasks_per_block: int | None = None,
                             reservation_id: str | None = None,
                             allocation_id: str | None = None,
                             deadline_unix: float | None = None,
                             remaining_updates: int | None = None,
                             learning_boundary: str | None = None,
                             data_dir: str, run_dir: str, precision: str,
                             deadline: float, started: float) -> dict[str, Any]:
    """Dispatch to the repository phase executor (D4 + contracts S1).

    Frozen protocol targets come from the runner spec (E0-calibrated when
    available, otherwise campaign minimums). Missing training targets fail
    here (never silent function defaults in the worker either).
    """
    from bramastra_lab.research.campaigns.phases.e1 import execute as execute_e1
    from bramastra_lab.research.campaigns.phases.e2 import execute as execute_e2
    from bramastra_lab.research.campaigns.phases.e3 import execute as execute_e3
    from bramastra_lab.research.campaigns.phases.e4 import execute as execute_e4
    from bramastra_lab.research.campaigns.phases.e5 import execute as execute_e5
    from bramastra_lab.research.campaigns.phases.types import JobInput

    elapsed = time.monotonic() - started
    manifest = os.path.join(data_dir, "manifest.json")
    if not os.path.exists(manifest):
        return {"phase": phase, "device": device, "arm": arm, "seed": seed,
                "status": "failed", "error": f"bundle manifest missing: {manifest}",
                "committed_updates": 0, "attempted_updates": 0,
                "supervised_exposure": 0, "device_seconds": elapsed,
                "checkpoint_identity": None,
                "evidence_kind": "fixture"}
    executors = {"E1": execute_e1, "E2": execute_e2, "E3": execute_e3,
                 "E4": execute_e4, "E5": execute_e5}
    executor = executors.get(phase)
    if executor is None:
        return {"phase": phase, "device": device, "arm": arm, "seed": seed,
                "status": "failed", "error": f"no executor for phase {phase!r}",
                "committed_updates": 0, "attempted_updates": 0,
                "supervised_exposure": 0, "device_seconds": elapsed,
                "checkpoint_identity": None,
                "evidence_kind": "fixture"}
    # Training targets must come from the spec (runner binds frozen protocol
    # minimums; E0 calibration refines them). The worker never invents 200/80
    # silently: missing targets for training phases fail here.
    if phase in ("E1", "E3", "E4") and update_target is None:
        return {"phase": phase, "device": device, "arm": arm, "seed": seed,
                "status": "failed",
                "error": f"{phase} requires explicit update_target from the "
                         "frozen protocol spec; refusing silent worker default",
                "committed_updates": 0, "attempted_updates": 0,
                "supervised_exposure": 0, "device_seconds": elapsed,
                "checkpoint_identity": None,
                "evidence_kind": "fixture"}
    if phase == "E2" and eval_cases is None:
        return {"phase": phase, "device": device, "arm": arm, "seed": seed,
                "status": "failed",
                "error": "E2 requires explicit eval_cases from the spec",
                "committed_updates": 0, "attempted_updates": 0,
                "supervised_exposure": 0, "device_seconds": elapsed,
                "checkpoint_identity": None,
                "evidence_kind": "fixture"}
    if phase == "E5" and tasks_per_block is None:
        return {"phase": phase, "device": device, "arm": arm, "seed": seed,
                "status": "failed",
                "error": "E5 requires explicit tasks_per_block from the spec",
                "committed_updates": 0, "attempted_updates": 0,
                "supervised_exposure": 0, "device_seconds": elapsed,
                "checkpoint_identity": None,
                "evidence_kind": "fixture"}
    from bramastra_lab.research.campaigns.phases.types import ParentRef

    resolved_job_id = job_id or (
        f"{phase}-{arm}-{seed}" if arm and seed is not None else
        (f"{phase}-{seed}" if seed is not None else phase))
    parent_ref = ParentRef(lookup_key=parent) if parent else None
    try:
        from bramastra_lab.research.campaigns.phases.ops import k8_identities

        identities = k8_identities(data_dir=data_dir)
    except Exception:
        identities = {}
    job = JobInput(phase=phase, slot=slot, arm=arm, seed=seed, parent=parent,
                   physical_device=physical_device, local_device=device,
                   data_dir=data_dir, run_dir=run_dir, precision=precision,
                   deadline=deadline, job_id=resolved_job_id,
                   update_target=update_target,
                   source_hash=identities.get("source_hash"),
                   data_hash=identities.get("data_identity"),
                   tokenizer_identity=identities.get("tokenizer_identity"),
                   config_identity=identities.get("config_identity"),
                   reservation_id=reservation_id,
                   allocation_id=allocation_id,
                   reservation_deadline_unix=deadline_unix,
                   parent_ref=parent_ref)
    try:
        if phase == "E1":
            result = executor(job, update_target=update_target)
        elif phase in ("E3", "E4"):
            result = executor(job, update_target=update_target)
        elif phase == "E2":
            result = executor(job, eval_cases=eval_cases)
        elif phase == "E5":
            result = executor(
                job, tasks_per_block=tasks_per_block,
                learning_boundary=learning_boundary
                if learning_boundary is not None else "test_substitute")
        else:
            result = executor(job)
    except Exception as exc:  # noqa: BLE001 - executor failure is a result
        return {"phase": phase, "device": device, "arm": arm, "seed": seed,
                "status": "failed", "error": f"{type(exc).__name__}: {exc}",
                "committed_updates": 0, "attempted_updates": 0,
                "supervised_exposure": 0,
                "device_seconds": time.monotonic() - started,
                "checkpoint_identity": None,
                "evidence_kind": "fixture"}
    out = result.to_dict()
    out.setdefault("phase", phase)
    out.setdefault("device", device)
    out["physical_device"] = physical_device
    out["device_seconds"] = float(out.get("device_seconds") or 0.0) or \
        (time.monotonic() - started)
    return out


def _run_learned_phase(*, phase: str, device: str, arm: str | None,
                       seed: int | None, data_dir: str, run_dir: str,
                       precision: str, deadline: float,
                       started: float) -> dict[str, Any]:
    """Legacy refusal entry (kept for backwards compatibility; dispatches)."""
    return _dispatch_phase_executor(
        phase=phase, device=device, physical_device=device, arm=arm,
        seed=seed, slot=None, parent=None, data_dir=data_dir,
        run_dir=run_dir, precision=precision, deadline=deadline,
        started=started)


def _run_e6(*, device: str, data_dir: str, run_dir: str,
            deadline: float, started: float) -> dict[str, Any]:
    """E6 production export behind dispatch (D4/D5)."""
    from bramastra_lab.research.campaigns.phases.e6 import execute as execute_e6
    from bramastra_lab.research.campaigns.phases.types import JobInput

    job = JobInput(phase="E6", slot=None, arm=None, seed=None, parent=None,
                   physical_device=device, local_device=device,
                   data_dir=data_dir, run_dir=run_dir, precision="fp32",
                   deadline=deadline)
    try:
        result = execute_e6(job)
    except Exception as exc:  # noqa: BLE001
        return {"phase": "E6", "device": device, "arm": None, "seed": None,
                "status": "failed", "error": f"{type(exc).__name__}: {exc}",
                "committed_updates": 0, "attempted_updates": 0,
                "supervised_exposure": 0,
                "device_seconds": time.monotonic() - started,
                "checkpoint_identity": None,
                "evidence_kind": "fixture"}
    out = result.to_dict()
    out.setdefault("phase", "E6")
    out["device_seconds"] = float(out.get("device_seconds") or 0.0) or \
        (time.monotonic() - started)
    return out


def full_profile_descriptor() -> dict[str, Any]:
    """Full K8 campaign profile calibration descriptor (no training).

    The E0 probe runs the tiny profile locally only as an orchestration
    check; full-profile parameter counts and heaviest-arm update targets are
    declared here for GPU calibration without consuming local updates.
    """
    from bramastra_lab.research.config import BuildConfig

    descriptors: dict[str, Any] = {}
    for profile in ("tiny", "development", "future_capacity"):
        config = BuildConfig.from_dict({"model": {"profile": profile}})
        descriptors[profile] = {
            "parameter_count": config.parameter_count(),
            "max_seq": config.model.max_seq,
        }
    return {
        "probe_profile": "tiny",
        "full_campaign_profile": "development",
        "profiles": descriptors,
        "note": "E0 probe uses tiny for orchestration only; GPU calibration "
                "must measure the development profile and heaviest active arms.",
    }


def _run_e0(*, device: str, arm: str | None, seed: int, data_dir: str,
            run_dir: str, precision: str, deadline: float,
            profile: str = "tiny") -> dict[str, Any]:
    """E0: hardware check, actual-path 3-update vs 1+2-resume comparison on
    this device, pilot throughput. Six real updates per worker, charged to
    the K8 allocation. GPU-only in production; local orchestration tests must
    use doubles, never this path (no local optimizer updates per policy).
    """
    started = time.monotonic()
    import torch

    from bramastra_lab.research.config import BuildConfig, seed_everything
    from bramastra_lab.research.experience.sequences import (
        build_answer_row,
        collocate,
    )
    from bramastra_lab.research.learning.k8_trainer import AllocationContext, K8Trainer
    from bramastra_lab.research.models import IntegratedModel

    torch.manual_seed(seed)
    config = _e0_probe_config(profile)
    model = IntegratedModel(config).to(device)
    trainer = K8Trainer(config, model, device=device,
                        precision="fp32" if not device.startswith("cuda") else precision,
                        require_allocation=True)
    allocation = AllocationContext(
        allocation_id=f"e0-{device}", device=device,
        deadline_unix=deadline, remaining_updates=128,
        job_id=f"E0-{device}", phase="E0")
    trainer.begin_campaign(allocation)

    def make_batch(question: str, answer: str):
        row = build_answer_row(
            [("goal", {"question": question})], answer,
            provenance={"kind": "trajectory", "episode_id": f"e0-{question[:4]}",
                        "task_semantic_id": "e0", "split": "training",
                        "source": "e0-pilot", "collection_policy": "e0",
                        "family": "e0"},
            max_tokens=64)
        return collocate([row], max_seq=64)

    e0_questions = ("2+2?", "3+3?", "5+1?")
    e0_answers = ("4", "6", "6")
    batches = [make_batch(question, answer)
               for question, answer in zip(e0_questions, e0_answers)]

    def training_step_full(batch, trainer_ref, question):
        """One explicit canonical update (D2): accumulate then finalize once.

        Both uninterrupted and resume branches call THIS function with the
        same treatment: answer window plus differentiable action/value/world
        terms combined by route_window into ONE scaled backward, then ONE
        optimizer step at the explicit boundary. Committed updates derive
        from actual counters (never a hardcoded 6).
        """
        from bramastra_lab.research.experience.supervision import SupervisionWindow
        from bramastra_lab.research.learning.k8_scoring import (
            score_candidates_trainable,
            value_estimate_trainable,
            world_transition_token_loss,
        )
        trainer_ref.accumulate_full_window(
            batch,
            window_builder=lambda target_count: _e0_window(target_count),
            extra_terms_fn=lambda: _e0_extra_terms(
                trainer_ref.model, trainer_ref.config, batch, question),
        )
        return trainer_ref.finalize_update()

    def _e0_window(target_count: int):
        from bramastra_lab.research.experience.supervision import SupervisionWindow

        window = SupervisionWindow(
            weights={"token": 1.0, "world": 0.01, "action": 0.01,
                     "value": 0.01, "pair": 0.0, "pg": 0.0},
            enabled_terms=frozenset({"token", "world", "action", "value"}))
        window.add("token", target_count)
        window.add("world", 1)
        window.add("action", 1)
        window.add("value", 1)
        return window

    def _e0_extra_terms(model_ref, config_ref, batch_ref, question):
        from bramastra_lab.research.campaigns.phases.compiler import (
            goal_prefix_tokens)
        from bramastra_lab.research.experience.codec import encode_text
        from bramastra_lab.research.learning.k8_scoring import (
            score_candidates_trainable,
            value_estimate_trainable,
            world_transition_token_loss,
        )

        # Complete prompt prefix (boundary + goal event), matching the
        # compiled batch — never a sliced fragment of the input row.
        tokens = goal_prefix_tokens({"question": question})
        # Real typed action candidates at full length (no arbitrary token
        # lists): the diagnostic exercises the same encode path as training.
        candidates = [
            encode_text(json.dumps(
                {"kind": "e0", "op": op, "question": question},
                sort_keys=True))
            for op in ("add", "echo")]
        scored = score_candidates_trainable(model_ref, config_ref, tokens, candidates)
        action_sum = -scored["log_probs"].sum()
        value = value_estimate_trainable(model_ref, config_ref, tokens)
        value_sum = value.square()
        world_sum = world_transition_token_loss(
            model_ref, config_ref, tokens, action={"kind": "e0"},
            target_feedback={"result": "ok"})
        return {"action": action_sum, "value": value_sum, "world": world_sum}

    # Uninterrupted: 3 explicit updates (accumulate + finalize each).
    seed_everything(seed)
    for batch, question in zip(batches, e0_questions):
        training_step_full(batch, trainer, question)
    uninterrupted_checksum = _strong_checksum(trainer)
    uninterrupted_committed = int(trainer.counters.optimizer_updates)
    uninterrupted_attempted = int(trainer.attempted_updates)
    uninterrupted_exposure = int(trainer.counters.supervised_targets_seen)
    state_payload = trainer.state_payload()
    payload_identity = _payload_identity(state_payload)
    config_identity = state_payload.get("config_identity")
    del trainer, model

    # Interrupted: 1 explicit update + checkpoint + child 2 updates.
    # The resumed continuation runs IN THE CHILD with the same configured
    # device/profile/precision (D2), not as another object in this process.
    torch.manual_seed(seed)
    model_b = IntegratedModel(config).to(device)
    trainer_b = K8Trainer(config, model_b, device=device,
                          precision="fp32" if not device.startswith("cuda") else precision,
                          require_allocation=True)
    trainer_b.begin_campaign(AllocationContext(
        allocation_id=f"e0-resume-{device}", device=device,
        deadline_unix=deadline, remaining_updates=128,
        job_id=f"E0-resume-{device}", phase="E0"))
    training_step_full(batches[0], trainer_b, e0_questions[0])
    payload = trainer_b.state_payload()
    interrupted_committed_1 = int(trainer_b.counters.optimizer_updates)
    child_result = _run_resume_in_child(
        payload, seed=seed, device=device, profile=profile,
        precision="fp32" if not device.startswith("cuda") else precision,
        deadline=deadline, config_identity=config_identity)
    resumed_checksum = child_result.get("resumed_checksum")
    resumed_committed = int(child_result.get("committed_updates", 0))
    resumed_attempted = int(child_result.get("attempted_updates", 0))
    resumed_exposure = int(child_result.get("supervised_exposure", 0))
    restore_proof = child_result.get("restore_proof", {})
    agrees = (uninterrupted_checksum == resumed_checksum
              and restore_proof.get("restored_ok", False)
              and uninterrupted_committed == resumed_committed == 3)
    committed = uninterrupted_committed + resumed_committed
    attempted = uninterrupted_attempted + resumed_attempted
    if not restore_proof.get("restored_ok", False):
        detail = child_result.get("error") or restore_proof.get("error")
        if not detail:
            detail = restore_proof.get("stderr_tail") or \
                restore_proof.get("stdout_tail") or \
                f"resume child exited with return code {restore_proof.get('returncode')}"
        return {
            "status": "failed",
            "error": f"E0 resume child failed: {str(detail)[:2000]}",
            "committed_updates": committed,
            "attempted_updates": attempted,
            "supervised_exposure": uninterrupted_exposure + resumed_exposure,
            "device_seconds": time.monotonic() - started,
            "checkpoint_identity": payload_identity,
            "resume_agrees": False,
            "uninterrupted_checksum": uninterrupted_checksum,
            "resumed_checksum": resumed_checksum,
            "restore_proof": restore_proof,
            "device_used": child_result.get("device_used", device),
            "profile": profile,
        }
    if not agrees:
        return {
            "status": "failed",
            "error": "E0 resume checksum divergence: uninterrupted and "
                     "resumed trainer states differ",
            "committed_updates": committed,
            "attempted_updates": attempted,
            "supervised_exposure": uninterrupted_exposure + resumed_exposure,
            "device_seconds": time.monotonic() - started,
            "checkpoint_identity": payload_identity,
            "resume_agrees": False,
            "uninterrupted_checksum": uninterrupted_checksum,
            "resumed_checksum": resumed_checksum,
            "restore_proof": restore_proof,
            "device_used": child_result.get("device_used", device),
            "profile": profile,
        }
    # Full-profile pilot timing for E0 calibration (O03): heaviest-arm
    # update seconds + eval-loop throughput on the K8 configuration.
    # Accelerator-only: locally recorded as skipped (no local training and
    # no large-model timing claims from CPU). Pilot updates are charged to
    # the E0 allowance like every other attempt.
    calibration_samples = _pilot_full_profile(
        device=device, data_dir=data_dir, precision=precision,
        deadline=deadline, seed=seed)
    return {
        "status": "completed" if agrees else "resume_divergence",
        "committed_updates": committed, "attempted_updates": attempted,
        "supervised_exposure": uninterrupted_exposure + resumed_exposure,
        "device_seconds": time.monotonic() - started,
        "checkpoint_identity": payload_identity,
        "resume_agrees": agrees,
        "uninterrupted_checksum": uninterrupted_checksum,
        "resumed_checksum": resumed_checksum,
        "restore_proof": restore_proof,
        "device_used": child_result.get("device_used", device),
        "profile": profile,
        "calibration_samples": calibration_samples,
    }


def _pilot_full_profile(*, device: str, data_dir: str, precision: str,
                        deadline: float, seed: int) -> dict[str, Any]:
    """Time heaviest-arm updates + eval throughput on the K8 config (O03)."""
    import time as _time

    if not str(device).startswith("cuda"):
        return {"full_profile_pilot": "skipped-local-no-accelerator",
                "worst_update_seconds": None,
                "eval_cases_per_second": None}
    from bramastra_lab.research.campaigns.phases.ops import (
        ProductionOps, k8_campaign_config)

    started = _time.monotonic()
    ops = ProductionOps(precision=precision)
    handle = ops.initialize_random(seed=seed, device=device)
    from bramastra_lab.research.learning.k8_trainer import AllocationContext

    pilot_allocation = AllocationContext(
        allocation_id=f"e0-pilot-{device}", device=device,
        deadline_unix=deadline, remaining_updates=8,
        job_id=f"E0-pilot-{device}", phase="E0")
    handle["trainer"].begin_campaign(pilot_allocation)
    # Heaviest arm (B package) on a real training trajectory.
    from bramastra_lab.research.campaigns.phases.compiler import (
        build_batch_for_trajectory, build_pair_rows,
        compile_channels_for_row, load_training_trajectories)
    from bramastra_lab.research.campaigns.phases.e1 import (
        ARM_ENABLED, ARM_WEIGHTS)

    trajectories = load_training_trajectories(data_dir, seed=seed)[:4]
    per_update: list[float] = []
    for step, row in enumerate(trajectories[:3]):
        batch = build_batch_for_trajectory(row)
        compiled = compile_channels_for_row(
            row, batch, arm_weights=dict(ARM_WEIGHTS["B"]),
            arm_enabled=ARM_ENABLED["B"])
        window, extra = ops.construct_objectives(
            handle=handle, batch=batch, compiled=compiled, arm="B")
        partner = next(
            r for r in trajectories
            if str(r.get("answer")) != str(row.get("answer")))
        pair_rows = build_pair_rows(row, partner)
        tick = _time.monotonic()
        ops.apply_update(handle, batch=batch, window=window, extra=extra,
                         pair_rows=pair_rows)
        per_update.append(_time.monotonic() - tick)
    worst_update_seconds = max(per_update) if per_update else None
    # Eval-loop throughput: live episodes on the pilot model (frozen eval,
    # no training; random-init generations score ~nothing — throughput only).
    eval_cases_per_second = None
    try:
        from bramastra_lab.research.cognition import episode as kernel
        from bramastra_lab.research.cognition.episode import (
            FreeGenerationModel)
        from bramastra_lab.research.environments.k8_live import (
            build_live_env, generate_live_mechanism)

        model = FreeGenerationModel(handle["model"], handle["config"])
        tick = _time.monotonic()
        episodes = 0
        for index, family in enumerate(("rule-inquiry", "inventory")):
            mechanism = generate_live_mechanism(family, index, seed=seed)
            env = build_live_env(mechanism, budget=6, seed=seed)
            # Learned-policy path on the pilot model (random-init outputs
            # mostly fail parsing; throughput — not success — is measured).
            kernel.run_episode(
                env, kernel.LearnedPolicyAdapter(), model=model, seed=seed,
                mechanism=mechanism, session_job_id=f"E0-pilot-{device}",
                checkpoint_id=None)
            episodes += 1
        elapsed = _time.monotonic() - tick
        eval_cases_per_second = episodes / elapsed if elapsed > 0 else None
    except Exception as exc:
        eval_cases_per_second = f"refused:{exc}"[:120]
    return {"full_profile_pilot": "measured",
            "device": device,
            "worst_update_seconds": worst_update_seconds,
            "eval_cases_per_second": eval_cases_per_second,
            "pilot_updates_charged": len(per_update),
            "pilot_wall_seconds": round(_time.monotonic() - started, 1)}


def _strong_checksum(trainer) -> str:
    """Full state identity: model + optimizer + counters + schedule (R05).

    A parameter sum is insufficient; this hashes the complete trainer payload
    (model state bytes, optimizer state, counters, controller) so a changed
    optimizer cursor, target or counter is detected.
    """
    import torch

    payload = trainer.state_payload()
    digest = hashlib.sha256()
    for name, tensor in sorted(payload["model"].items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    digest.update(json.dumps(payload["counters"], sort_keys=True).encode())
    digest.update(json.dumps({
        "controller_multiplier": payload.get("controller_multiplier"),
        "controller_reason": payload.get("controller_reason"),
        "schedule": payload.get("schedule"),
    }, sort_keys=True, default=str).encode())
    # Optimizer state (moments) included: different trajectories diverge here.
    try:
        opt_state = payload.get("optimizer", {})
        digest.update(json.dumps(_jsonable(opt_state), sort_keys=True,
                                 default=str).encode())
    except Exception:
        pass
    return digest.hexdigest()


def _jsonable(value):
    import torch

    if isinstance(value, torch.Tensor):
        return {"__tensor__": value.detach().cpu().tolist()}
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _payload_identity(payload: dict) -> str:
    digest = hashlib.sha256()
    for name in sorted(payload.get("model", {})):
        tensor = payload["model"][name]
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    digest.update(json.dumps(payload.get("counters", {}), sort_keys=True).encode())
    return digest.hexdigest()[:32]


def _run_resume_in_child(payload: dict, *, seed: int, device: str,
                         profile: str, precision: str, deadline: float,
                         config_identity: str | None) -> dict[str, Any]:
    """Perform the resumed 2-update continuation IN THE CHILD (D2).

    Restores the exact configured device/profile/precision with allocation +
    config-identity validation, runs the same explicit accumulate/finalize
    function for the remaining batches, and returns the strong checksum +
    actual counters. The parent never continues a restored trainer in-process.
    """
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory(prefix="e0-resume-") as tmp:
        path = os.path.join(tmp, "payload.pt")
        import torch

        torch.save(payload, path)
        code = (
            "import sys, torch\n"
            f"payload = torch.load({path!r}, weights_only=False, map_location='cpu')\n"
            "from bramastra_lab.research.config import BuildConfig, seed_everything\n"
            "from bramastra_lab.research.experience.sequences import build_answer_row, collocate\n"
            "from bramastra_lab.research.learning.k8_trainer import AllocationContext, K8Trainer\n"
            "from bramastra_lab.research.models import IntegratedModel\n"
            "from bramastra_lab.research.campaigns.worker import _strong_checksum, _e0_probe_config\n"
            "from bramastra_lab.research.campaigns.phases.compiler import goal_prefix_tokens\n"
            "from bramastra_lab.research.experience.codec import encode_text\n"
            f"profile={profile!r}; device={device!r}; precision={precision!r}; "
            f"seed={int(seed)}; deadline={float(deadline)!r}\n"
            f"config_identity={config_identity!r}\n"
            "config = _e0_probe_config(profile)\n"
            "torch.manual_seed(seed)\n"
            "model = IntegratedModel(config).to(device)\n"
            "trainer = K8Trainer(config, model, device=device, precision=precision, require_allocation=True)\n"
            "allocation = AllocationContext(allocation_id=f'e0-resume-{device}', device=device, "
            "deadline_unix=deadline, remaining_updates=128, job_id=f'E0-resume-{device}', phase='E0')\n"
            "trainer.begin_campaign(allocation)\n"
            "trainer.load_state_payload(payload, expected_allocation=allocation, expected_config_identity=config_identity)\n"
            "trainer.model.to(device)\n"
            "def make_batch(question, answer):\n"
            "    row = build_answer_row([('goal', {'question': question})], answer, "
            "provenance={'kind': 'trajectory', 'episode_id': f'e0-{question[:4]}', 'task_semantic_id': 'e0', "
            "'split': 'training', 'source': 'e0-pilot', 'collection_policy': 'e0', 'family': 'e0'}, max_tokens=64)\n"
            "    return collocate([row], max_seq=64)\n"
            "batches = [make_batch('2+2?', '4'), make_batch('3+3?', '6'), make_batch('5+1?', '6')]\n"
            "from bramastra_lab.research.learning.k8_scoring import score_candidates_trainable, value_estimate_trainable, world_transition_token_loss\n"
            "from bramastra_lab.research.experience.supervision import SupervisionWindow\n"
            "def window_for(n):\n"
            "    window = SupervisionWindow(weights={'token': 1.0, 'world': 0.01, 'action': 0.01, 'value': 0.01, 'pair': 0.0, 'pg': 0.0}, "
            "enabled_terms=frozenset({'token', 'world', 'action', 'value'}))\n"
            "    window.add('token', n); window.add('world', 1); window.add('action', 1); window.add('value', 1)\n"
            "    return window\n"
            "import json\n"
            "questions = ['3+3?', '5+1?']\n"
            "for batch, question in zip(batches[1:], questions):\n"
            "    def extra_terms(batch_ref=batch, q=question):\n"
            "        tokens = goal_prefix_tokens({'question': q})\n"
            "        candidates = [encode_text(json.dumps({'kind': 'e0', 'op': op, 'question': q}, sort_keys=True)) for op in ('add', 'echo')]\n"
            "        scored = score_candidates_trainable(trainer.model, trainer.config, tokens, candidates)\n"
            "        action_sum = -scored['log_probs'].sum()\n"
            "        value = value_estimate_trainable(trainer.model, trainer.config, tokens)\n"
            "        world_sum = world_transition_token_loss(trainer.model, trainer.config, tokens, action={'kind': 'e0'}, target_feedback={'result': 'ok'})\n"
            "        return {'action': action_sum, 'value': value.square(), 'world': world_sum}\n"
            "    trainer.accumulate_full_window(batch, window_builder=window_for, extra_terms_fn=extra_terms)\n"
            "    trainer.finalize_update()\n"
            "checksum = _strong_checksum(trainer)\n"
            "print('RESUME_CHILD:' + json.dumps({'checksum': checksum, "
            "'committed': trainer.counters.optimizer_updates, 'attempted': trainer.attempted_updates, "
            "'exposure': trainer.counters.supervised_targets_seen, "
            "'device': str(next(trainer.model.parameters()).device)}))\n"
        )
        try:
            child_env = os.environ.copy()
            repo_root = os.path.abspath(os.getcwd())
            child_env["PYTHONPATH"] = repo_root + os.pathsep + \
                child_env.get("PYTHONPATH", "")
            proc = subprocess.run(
                [sys.executable, "-c", code], cwd=repo_root, env=child_env,
                capture_output=True, text=True, timeout=600)
        except Exception as exc:  # noqa: BLE001
            return {"resumed_checksum": None, "committed_updates": 0,
                    "attempted_updates": 0, "supervised_exposure": 0,
                    "device_used": device,
                    "error": f"{type(exc).__name__}: {exc}",
                    "restore_proof": {"restored_ok": False,
                                      "error": str(exc)}}
        marker = "RESUME_CHILD:"
        resumed_checksum = None
        committed = attempted = exposure = 0
        device_used = device
        restored_ok = False
        for line in proc.stdout.splitlines():
            if marker in line:
                try:
                    data = json.loads(line.split(marker, 1)[1])
                    resumed_checksum = data.get("checksum")
                    committed = int(data.get("committed", 0))
                    attempted = int(data.get("attempted", 0))
                    exposure = int(data.get("exposure", 0))
                    device_used = str(data.get("device", device))
                    restored_ok = proc.returncode == 0 and bool(resumed_checksum)
                except Exception:
                    pass
        result = {"resumed_checksum": resumed_checksum,
                "committed_updates": committed,
                "attempted_updates": attempted,
                "supervised_exposure": exposure,
                "device_used": device_used,
                "restore_proof": {"restored_ok": restored_ok,
                                  "returncode": proc.returncode,
                                  "stdout_tail": proc.stdout[-500:],
                                  "stderr_tail": proc.stderr[-500:]}}
        if proc.returncode != 0:
            result["error"] = (proc.stderr or proc.stdout or
                                f"resume child exited with code {proc.returncode}")[-2000:]
        elif not resumed_checksum:
            result["error"] = "resume child produced no RESUME_CHILD result marker"
        return result


def _verify_payload_in_subprocess(payload: dict) -> dict[str, Any]:
    """Prove fresh-process restore: reload payload in a spawn subprocess."""
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory(prefix="e0-restore-") as tmp:
        path = os.path.join(tmp, "payload.pt")
        import torch

        torch.save(payload, path)
        code = (
            "import sys, torch; "
            f"payload = torch.load({path!r}, weights_only=False, map_location='cpu'); "
            "from bramastra_lab.research.config import BuildConfig; "
            "from bramastra_lab.research.models import IntegratedModel; "
            "from bramastra_lab.research.learning.k8_trainer import K8Trainer; "
            "profile = 'tiny'; "
            "config = BuildConfig.from_dict({'model': {'profile': profile}}); "
            "model = IntegratedModel(config); "
            "trainer = K8Trainer(config, model, device='cpu'); "
            "trainer.load_state_payload(payload); "
            "print('RESTORE_OK:' + str(trainer.counters.optimizer_updates))"
        )
        try:
            child_env = os.environ.copy()
            repo_root = os.path.abspath(os.getcwd())
            child_env["PYTHONPATH"] = repo_root + os.pathsep + \
                child_env.get("PYTHONPATH", "")
            proc = subprocess.run(
                [sys.executable, "-c", code], cwd=repo_root, env=child_env,
                capture_output=True, text=True, timeout=120)
        except Exception as exc:  # noqa: BLE001
            return {"restored_ok": False, "error": str(exc)}
        ok = proc.returncode == 0 and "RESTORE_OK:" in proc.stdout
        return {"restored_ok": bool(ok),
                "returncode": proc.returncode,
                "stdout_tail": proc.stdout[-500:],
                "stderr_tail": proc.stderr[-500:]}
