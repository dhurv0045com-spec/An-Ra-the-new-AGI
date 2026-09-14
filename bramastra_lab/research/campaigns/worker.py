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


def run_worker_phase(*, phase: str, device: str, arm: str | None,
                     seed: int | None, data_dir: str, run_dir: str,
                     precision: str, deadline: float) -> dict[str, Any]:
    """Execute one phase on one device. In the real campaign this runs in a
    fresh spawn subprocess with CUDA_VISIBLE_DEVICES set before importing
    torch (see process_supervision); in-process execution records the device
    for E0 verification."""
    started = time.monotonic()
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
        return _run_learned_phase(phase=phase, device=device, arm=arm,
                                  seed=seed, data_dir=data_dir,
                                  run_dir=run_dir, precision=precision,
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


def _run_learned_phase(*, phase: str, device: str, arm: str | None,
                       seed: int | None, data_dir: str, run_dir: str,
                       precision: str, deadline: float,
                       started: float) -> dict[str, Any]:
    """Real learned-phase handoff point (R01).

    Validates the bundle manifest, allocation deadline and E0 qualification
    signal (runner enforces the gate; the worker re-checks the manifest and
    refuses without an executor). Expensive model execution is delegated to
    an injected executor in tests; production GPU execution remains within
    the owner's future allocation. Without an executor this returns an
    explicit refused status — never blocked_pending_e0 masquerading as
    success and never inherited completed with zero work (old E2 defect).
    """
    elapsed = time.monotonic() - started
    manifest = os.path.join(data_dir, "manifest.json")
    if not os.path.exists(manifest):
        return {
            "phase": phase, "device": device, "arm": arm, "seed": seed,
            "status": "failed", "error": f"bundle manifest missing: {manifest}",
            "committed_updates": 0, "attempted_updates": 0,
            "supervised_exposure": 0, "device_seconds": elapsed,
            "checkpoint_identity": None,
        }
    # No executor is bound in this build: refuse explicitly so the runner
    # records failure and returns nonzero (fail-closed). A test double or a
    # future GPU executor supplies the learned execution behind this gate.
    return {
        "phase": phase, "device": device, "arm": arm, "seed": seed,
        "status": "failed",
        "error": (f"{phase} learned executor not bound in this build; "
                  "refusing (no zero-work success). Provide the GPU executor "
                  "within the owner's allocation."),
        "reason": "refused_missing_executor",
        "committed_updates": 0, "attempted_updates": 0,
        "supervised_exposure": 0, "device_seconds": elapsed,
        "checkpoint_identity": None,
    }


def _run_e6(*, device: str, data_dir: str, run_dir: str,
            deadline: float, started: float) -> dict[str, Any]:
    """E6 export verification (R01/R08): refuse unless ledger + bundle exist."""
    elapsed = time.monotonic() - started
    ledger_path = os.path.join(run_dir, "campaign_ledger.sqlite")
    manifest = os.path.join(data_dir, "manifest.json")
    missing = [path for path in (ledger_path, manifest)
               if not os.path.exists(path)]
    if missing:
        return {
            "phase": "E6", "device": device, "arm": None, "seed": None,
            "status": "failed",
            "error": f"E6 export requirements missing: {missing}",
            "committed_updates": 0, "attempted_updates": 0,
            "supervised_exposure": 0, "device_seconds": elapsed,
            "checkpoint_identity": None,
        }
    return {
        "phase": "E6", "device": device, "arm": None, "seed": None,
        "status": "completed",
        "committed_updates": 0, "attempted_updates": 0,
        "supervised_exposure": 0, "device_seconds": elapsed,
        "checkpoint_identity": "e6-export-verified",
    }


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
    config = BuildConfig.from_dict({"model": {"profile": profile}})
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

    batches = [make_batch("2+2?", "4"), make_batch("3+3?", "6"),
               make_batch("5+1?", "6")]

    def training_step_full(batch, trainer_ref):
        """One canonical update through the shared objective boundary (R04).

        Both uninterrupted and resume branches call THIS function with the
        same treatment: answer window plus differentiable action/value/world
        terms combined by route_window into ONE scaled backward (no mixed
        scaled/unscaled gradients, no divergent treatments).
        """
        from bramastra_lab.research.experience.supervision import SupervisionWindow
        from bramastra_lab.research.learning.k8_scoring import (
            score_candidates_trainable,
            value_estimate_trainable,
            world_transition_token_loss,
        )
        return trainer_ref.accumulate_full_window(
            batch,
            window_builder=lambda target_count: _e0_window(target_count),
            extra_terms_fn=lambda: _e0_extra_terms(
                trainer_ref.model, trainer_ref.config, batch),
        )

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

    def _e0_extra_terms(model_ref, config_ref, batch_ref):
        from bramastra_lab.research.learning.k8_scoring import (
            score_candidates_trainable,
            value_estimate_trainable,
            world_transition_token_loss,
        )

        tokens = batch_ref.input_ids[0][:8].tolist()
        candidates = [[70, 71], [80, 81]]
        scored = score_candidates_trainable(model_ref, config_ref, tokens, candidates)
        action_sum = -scored["log_probs"].sum()
        value = value_estimate_trainable(model_ref, config_ref, tokens[:6])
        value_sum = value.square()
        world_sum = world_transition_token_loss(
            model_ref, config_ref, tokens[:6], action={"kind": "e0"},
            target_feedback={"result": "ok"})
        return {"action": action_sum, "value": value_sum, "world": world_sum}

    # Uninterrupted: 3 canonical updates.
    seed_everything(seed)
    for batch in batches:
        training_step_full(batch, trainer)
    uninterrupted_checksum = _strong_checksum(trainer)
    uninterrupted_counters = dict(trainer.counters.to_dict())
    state_payload = trainer.state_payload()
    payload_identity = _payload_identity(state_payload)
    del trainer, model

    # Interrupted: 1 canonical update + checkpoint + fresh-model 2 canonical updates.
    torch.manual_seed(seed)
    model_b = IntegratedModel(config).to(device)
    trainer_b = K8Trainer(config, model_b, device=device,
                          precision="fp32" if not device.startswith("cuda") else precision,
                          require_allocation=True)
    trainer_b.begin_campaign(AllocationContext(
        allocation_id=f"e0-resume-{device}", device=device,
        deadline_unix=deadline, remaining_updates=128,
        job_id=f"E0-resume-{device}", phase="E0"))
    training_step_full(batches[0], trainer_b)
    payload = trainer_b.state_payload()
    # Fresh-process restore proof: payload round-trips through a separate
    # spawn process before continuing (not another object in-process).
    restore_proof = _verify_payload_in_subprocess(payload)
    model_c = IntegratedModel(config).to(device)
    trainer_c = K8Trainer(config, model_c, device=device,
                          precision="fp32" if not device.startswith("cuda") else precision,
                          require_allocation=True)
    trainer_c.load_state_payload(payload)
    trainer_c.model.to(device)
    for batch in batches[1:]:
        training_step_full(batch, trainer_c)
    resumed_checksum = _strong_checksum(trainer_c)
    agrees = uninterrupted_checksum == resumed_checksum and restore_proof.get(
        "restored_ok", False)
    return {
        "status": "completed" if agrees else "resume_divergence",
        "committed_updates": 6, "attempted_updates": 6,
        "supervised_exposure": sum(b.target_count for b in batches) * 2,
        "device_seconds": time.monotonic() - started,
        "checkpoint_identity": payload_identity,
        "resume_agrees": agrees,
        "uninterrupted_checksum": uninterrupted_checksum,
        "resumed_checksum": resumed_checksum,
        "restore_proof": restore_proof,
        "device_used": str(next(model_c.parameters()).device),
        "profile": profile,
    }


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
            proc = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True, text=True, timeout=120)
        except Exception as exc:  # noqa: BLE001
            return {"restored_ok": False, "error": str(exc)}
        ok = proc.returncode == 0 and "RESTORE_OK:" in proc.stdout
        return {"restored_ok": bool(ok),
                "returncode": proc.returncode,
                "stdout_tail": proc.stdout[-500:],
                "stderr_tail": proc.stderr[-500:]}
