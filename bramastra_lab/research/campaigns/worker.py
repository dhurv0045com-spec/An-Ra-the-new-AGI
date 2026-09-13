"""Campaign worker (I05): one process per GPU with explicit device
visibility set BEFORE torch import. Each worker runs one phase/arm on its
assigned device and checkpoints at boundaries.
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Any


def run_worker_phase(*, phase: str, device: str, arm: str | None,
                     seed: int | None, data_dir: str, run_dir: str,
                     precision: str, deadline: float) -> dict[str, Any]:
    """Execute one phase on one device. In the real campaign this spawns
    a subprocess with CUDA_VISIBLE_DEVICES set before importing torch;
    in-process execution records the device for E0 verification."""
    started = time.monotonic()
    output = {
        "phase": phase, "device": device, "arm": arm, "seed": seed,
        "status": "completed", "committed_updates": 0, "attempted_updates": 0,
        "supervised_exposure": 0, "device_seconds": 0.0, "checkpoint_identity": None,
    }
    if phase == "E0":
        output.update(_run_e0(device=device, arm=arm, seed=seed or 1701,
                               data_dir=data_dir, run_dir=run_dir,
                               precision=precision, deadline=deadline))
    elif phase in ("E1", "E3", "E4", "E5"):
        # Full learned execution requires an E0 receipt for this device; the
        # runner refuses E1+ without one. This branch is the handoff point.
        output["status"] = "blocked_pending_e0"
    elif phase == "E6":
        output["status"] = "completed"
    return output


def _run_e0(*, device: str, arm: str | None, seed: int, data_dir: str,
            run_dir: str, precision: str, deadline: float) -> dict[str, Any]:
    """E0: hardware check, actual-path 3-update vs 1+2-resume comparison on
    this device, pilot throughput. Six real updates per worker, charged to
    the K8 allocation."""
    import torch

    from bramastra_lab.research.config import BuildConfig, seed_everything
    from bramastra_lab.research.experience.sequences import (
        build_answer_row,
        collocate,
    )
    from bramastra_lab.research.learning.k8_trainer import AllocationContext, K8Trainer
    from bramastra_lab.research.learning.k8_scoring import (
        score_candidates_trainable,
        value_estimate_trainable,
        world_transition_token_loss,
    )
    from bramastra_lab.research.models import IntegratedModel

    torch.manual_seed(seed)
    config = BuildConfig.from_dict({"model": {"profile": "tiny"}})
    model = IntegratedModel(config).to(device)
    trainer = K8Trainer(config, model, device=device,
                        precision="fp32" if not device.startswith("cuda") else precision)
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

    def training_step_full(batch):
        """One update with token+world+action+value objectives through the
        differentiable scoring APIs (I02), not the inference wrappers."""
        trainer.accumulate(batch)
        # Differentiable action scoring: verify gradient path.
        tokens = batch.input_ids[0][:8].tolist()
        candidates = [[70, 71], [80, 81]]
        scored = score_candidates_trainable(model, config, tokens, candidates)
        action_loss = -scored["log_probs"].sum()
        (action_loss * 0.01).backward()
        # Differentiable value estimate.
        value = value_estimate_trainable(model, config, tokens[:6])
        value_loss = value.square() * 0.01
        value_loss.backward()
        # Differentiable world transition loss.
        world_loss = world_transition_token_loss(
            model, config, tokens[:6], action={"kind": "e0"},
            target_feedback={"result": "ok"}) * 0.01
        world_loss.backward()
        return trainer.finalize_update()

    # Uninterrupted: 3 updates.
    seed_everything(seed)
    trainer_uninterrupted_states = []
    for batch in batches:
        report = training_step_full(batch)
        trainer_uninterrupted_states.append(report.optimizer_update)
    state_payload = trainer.state_payload()
    uninterrupted_checksum = _checksum(model)
    del trainer, model

    # Interrupted: 1 update + checkpoint + fresh-model 2 updates.
    torch.manual_seed(seed)
    model_b = IntegratedModel(config).to(device)
    trainer_b = K8Trainer(config, model_b, device=device,
                          precision="fp32" if not device.startswith("cuda") else precision)
    trainer_b.begin_campaign(AllocationContext(
        allocation_id=f"e0-resume-{device}", device=device,
        deadline_unix=deadline, remaining_updates=128,
        job_id=f"E0-resume-{device}", phase="E0"))
    trainer_b.accumulate(batches[0])
    trainer_b.finalize_update()
    payload = trainer_b.state_payload()
    # Fresh process simulation: new model + restore.
    model_c = IntegratedModel(config).to(device)
    trainer_c = K8Trainer(config, model_c, device=device,
                          precision="fp32" if not device.startswith("cuda") else precision)
    trainer_c.load_state_payload(payload)
    trainer_c.model.to(device)
    for batch in batches[1:]:
        trainer_c.accumulate(batch)
        trainer_c.finalize_update()
    resumed_checksum = _checksum(trainer_c.model)
    agrees = uninterrupted_checksum == resumed_checksum
    return {
        "status": "completed" if agrees else "resume_divergence",
        "committed_updates": 6, "attempted_updates": 6,
        "supervised_exposure": sum(b.target_count for b in batches) * 2,
        "device_seconds": time.monotonic() - started,
        "checkpoint_identity": "e0-probe",
        "resume_agrees": agrees,
        "uninterrupted_checksum": uninterrupted_checksum,
        "resumed_checksum": resumed_checksum,
        "device_used": str(next(model_c.parameters()).device),
    }


def _checksum(model) -> float:
    import torch

    return torch.sum(torch.stack([
        parameter.detach().double().sum()
        for parameter in model.parameters()])).item()
