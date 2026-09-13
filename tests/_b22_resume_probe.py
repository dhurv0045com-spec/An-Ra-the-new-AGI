"""B2.2 C5 final acceptance probe: uninterrupted three versus interrupted one
plus resumed two, through the actual shared train/resume path, with replay
and controller scheduling active at a nontrivial boundary.

Budget: exactly six real optimizer updates (3 + 1 + 2) counted in the build
smoke ledger. All learned work happens in fresh subprocesses.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PHASE_SCRIPT = r'''
import json, os, sys

import torch

os.environ.setdefault("BRAMASTRA_LEARNED_CHECKS", "1")
from bramastra_lab.research.collection.runner import collect
from bramastra_lab.research.commands import (
    _collocate_records, _load_rows, _read_prepared, _sampler_units, prepare_data)
from bramastra_lab.research.config import BuildConfig, seed_everything
from bramastra_lab.research.data.sampler import GroupSampler, SamplerState
from bramastra_lab.research.learning.plasticity import ControllerState
from bramastra_lab.research.learning.trainer import Trainer, TrainerDiagnostics
from bramastra_lab.research.models import IntegratedModel
from bramastra_lab.research.experience.ledger import ExperienceLedger
from bramastra_lab.research.experience.replay import ReplaySchedule
from bramastra_lab.research.environments.oracles import failed_baseline_policy
from bramastra_lab.research.environments.worlds import InventoryWorld, SwitchWorld
from bramastra_lab.research.runtime import checkpoint as ckpt
from bramastra_lab.research.runtime.resume import (
    RunContext, append_event, checkpoint_run, create_run, read_run_manifest,
    restore_run)

phase, workdir = sys.argv[1], sys.argv[2]
RAW_CONFIG = {
    "model": {"profile": "tiny"},
    "training": {"seed": 4242, "learning_rate": 0.01, "grad_accum_steps": 2,
                  "pair_loss_weight": 0.5},
    "controller": {"mode": "evidence_driven", "controller_pool_id": "probe-controller",
                    "formation_threshold": 0.95, "stabilize_window": 2,
                    "controller_eval_every": 2, "cooldown_updates": 1,
                    "collapse_confirmation_evaluations": 2,
                    "recovery_confirmation_evaluations": 2},
    "replay": {"enabled": True, "proportion": 0.5, "on_empty": "skip",
                "family_weights": {"switch-world": 1.0, "inventory-world": 1.0}},
}

def build_fixtures():
    prepared = os.path.join(workdir, "prepared")
    if not os.path.exists(prepared):
        rows_dir = os.path.join(workdir, "data")
        os.makedirs(rows_dir, exist_ok=True)
        training = [
            {"example_id": "t1", "prompt_events": [["goal", {"q": "2+2?"}]], "answer": "4",
             "family": "arithmetic"},
            {"example_id": "t2", "prompt_events": [["goal", {"q": "3+3?"}]], "answer": "6",
             "family": "arithmetic"},
            {"example_id": "p1a", "prompt_events": [["goal", {"q": "5+1?"}]], "answer": "6",
             "group": "swap-1", "family": "arithmetic"},
            {"example_id": "p1b", "prompt_events": [["goal", {"q": "1+4?"}]], "answer": "5",
             "group": "swap-1", "family": "arithmetic"},
            {"example_id": "t3", "prompt_events": [["goal", {"q": "2+5?"}]], "answer": "7",
             "family": "arithmetic"},
            {"example_id": "t4", "prompt_events": [["goal", {"q": "4+1?"}]], "answer": "5",
             "family": "arithmetic"},
        ]
        open(os.path.join(rows_dir, "train.jsonl"), "w", newline="\n").write(
            "".join(json.dumps(r) + "\n" for r in training))
        controller = [
            {"example_id": "c1", "prompt_events": [["goal", {"q": "8+1?"}]], "answer": "9",
             "family": "arithmetic"},
            {"example_id": "c2", "prompt_events": [["goal", {"q": "2+7?"}]], "answer": "9",
             "family": "arithmetic"},
        ]
        open(os.path.join(rows_dir, "controller.jsonl"), "w", newline="\n").write(
            "".join(json.dumps(r) + "\n" for r in controller))
        manifest = {"schema": "bramastra-dataset-manifest/v1", "name": "b22-probe",
                     "license": "probe-fixture", "provenance": "b22-acceptance",
                     "entries": [
                         {"path": "train.jsonl", "split": "training", "kind": "trajectory"},
                         {"path": "controller.jsonl", "split": "controller",
                          "kind": "trajectory"}]}
        open(os.path.join(rows_dir, "manifest.json"), "w").write(json.dumps(manifest))
        config_path = os.path.join(workdir, "config.json")
        open(config_path, "w").write(json.dumps(RAW_CONFIG))
        from bramastra_lab.research.cli import CommandError
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            prepare_data(os.path.join(rows_dir, "manifest.json"), prepared, config_path)
        # Replay episodes through the real collection runner (zero updates).
        ledger = ExperienceLedger(os.path.join(prepared, "episodes.jsonl"))
        environments = [SwitchWorld(budget=6, seed=11), InventoryWorld(budget=6, seed=12)]
        collect(environments, ledger, failed_baseline_policy,
                policy_identity="failed-baseline/v1", episodes_per_environment=2,
                episode_prefix="probe")
    return os.path.join(workdir, "config.json"), prepared

def state_identities(trainer):
    model_checksum = torch.sum(torch.stack([
        parameter.detach().double().sum() for parameter in trainer.model.parameters()])).item()
    optimizer_terms = []
    for state in trainer.optimizer.state_dict()["state"].values():
        for value in state.values():
            if isinstance(value, torch.Tensor):
                optimizer_terms.append(value.detach().double().sum().item())
            else:
                optimizer_terms.append(float(value))
    return {"model_checksum": model_checksum,
            "optimizer_checksum": sum(optimizer_terms),
            "counters": trainer.counters.to_dict()}

def run_shared(run_dir, config_path, prepared, target_updates, *, resume_from=None):
    config = BuildConfig.from_dict(json.load(open(config_path, encoding="utf-8")))
    if resume_from is None:
        seed_everything(config.training.seed)
        create_run(run_dir, config, data_identity=_read_prepared(prepared, config)["identity"])
        writer_token = ckpt.acquire_writer_fence(run_dir)
        trainer = Trainer(config, IntegratedModel(config))
        sampler = GroupSampler(_sampler_units(_load_rows(prepared, "training")),
                               batch_size=min(4, 6), seed=config.training.seed)
        controller_state = ControllerState()
        schedule = ReplaySchedule(config.replay.proportion)
        engine = None
        from bramastra_lab.research.commands import _load_replay_engine
        engine = _load_replay_engine(config, prepared)
        expected_parent = None
    else:
        ctx = restore_run(run_dir, config)
        trainer = ctx.trainer
        sampler = GroupSampler(_sampler_units(_load_rows(prepared, "training")),
                               batch_size=min(4, 6), seed=config.training.seed)
        if ctx.sampler_state:
            sampler.restore(SamplerState.from_dict(ctx.sampler_state))
        controller_state = ControllerState.from_dict(ctx.controller_state)
        engine = None
        schedule = None
        if ctx.replay_cursor:
            from bramastra_lab.research.commands import _load_replay_engine, _restore_replay
            engine, schedule = _restore_replay(config, prepared, ctx.replay_cursor)
        expected_parent = ctx.expected_parent
        writer_token = ctx.writer_token
        seed_everything(0)  # restore_run already restored RNG streams
    from bramastra_lab.research.commands import (
        _collocate_records, _collocate_rows, _controller_boundary, _pair_rows_for,
        _publish, _rows_from_receipt)
    trainer.set_diagnostics(TrainerDiagnostics(
        trainer.model, _collocate_records(_load_rows(prepared, "training")[:1],
                                          config.model.max_seq)))
    run_manifest = read_run_manifest(run_dir)
    ctx = RunContext(run_dir=run_dir, run_id=run_manifest["run_id"], config=config,
                     data_identity=run_manifest["data_identity"],
                     writer_token=writer_token,
                     trainer=trainer, sampler_state=None, controller_state=None,
                     replay_cursor=None, expected_parent=expected_parent)
    import atexit
    atexit.register(ckpt.release_writer_fence, run_dir, writer_token)
    publication = {"last_update": None, "last_manifest": None, "last_directory": None}
    updates_before = trainer.counters.optimizer_updates
    while trainer.counters.optimizer_updates < target_updates:
        if config.controller.mode == "evidence_driven":
            controller_state, decision = _controller_boundary(
                trainer, ctx, config, prepared, controller_state,
                run_dir=run_dir)
            if decision is not None and decision.pause_updates:
                break
        for _micro in range(trainer.grad_accum_steps):
            if engine is not None and schedule.slot_due():
                replay_batch = engine.sample()
                rendered = [row for row in
                            (_rows_from_receipt(receipt, max_tokens=config.model.max_seq)[0]
                             for receipt in replay_batch.entries) if row is not None]
                if rendered:
                    trainer.counters.replay_entries_consumed += len(rendered)
                    trainer.accumulate(_collocate_rows(rendered, config.model.max_seq))
                    continue
            refs = sampler.take_batch()
            batch = _collocate_records([ref.record for ref in refs], config.model.max_seq)
            pair_rows = None
            from bramastra_lab.research.commands import _pair_rows_for
            own, swapped = _pair_rows_for([ref.record for ref in refs])
            if own:
                pair_rows = (own, swapped)
            trainer.accumulate(batch, pair_rows=pair_rows)
        trainer.finalize_update()
    ctx.sampler_state = sampler.state().to_dict()
    ctx.controller_state = controller_state.to_dict()
    ctx.replay_cursor = {"engine": engine.state() if engine else None,
                          "schedule": schedule.state() if schedule else None}
    manifest = checkpoint_run(trainer, run_dir, ctx, milestone="probe")
    result = state_identities(trainer)
    result["sampler_state"] = ctx.sampler_state
    result["controller_state"] = ctx.controller_state
    result["replay_cursor"] = ctx.replay_cursor
    # Future event sequence from the persisted state (non-mutating peeks).
    restored = ReplaySchedule.from_state(ctx.replay_cursor["schedule"])
    result["future_replay_slots"] = [restored.slot_due() for _ in range(3)]
    result["next_controller_eval_due_at"] = (
        None if ctx.controller_state["last_controller_eval_update"] is None
        else ctx.controller_state["last_controller_eval_update"]
        + config.controller.controller_eval_every)
    return result

if phase == "A":
    config_path, prepared = build_fixtures()
    run_dir = os.path.join(workdir, "uninterrupted")
    print(json.dumps(run_shared(run_dir, config_path, prepared, 3)))
elif phase == "B":
    config_path, prepared = build_fixtures()
    run_dir = os.path.join(workdir, "interrupted")
    print(json.dumps(run_shared(run_dir, config_path, prepared, 1)))
elif phase == "C":
    config_path, prepared = build_fixtures()
    run_dir = os.path.join(workdir, "interrupted")
    print(json.dumps(run_shared(run_dir, config_path, prepared, 3, resume_from=True)))
'''


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--ledger", default=None,
                        help="session ledger path (defaults to the repo build ledger)")
    args = parser.parse_args()
    workdir = os.path.abspath(args.workdir)
    os.makedirs(workdir, exist_ok=True)
    # Hard pre-execution gate: subprocess launches cannot bypass the shared
    # allowance merely because they run outside train --smoke (B2.2 chief F6).
    from bramastra_lab.research.runtime.smoke import SessionLedger

    ledger_path = args.ledger or os.path.join(REPO_ROOT, "engineering", "reports",
                                              "B2", "SESSION_LEDGER.json")
    SessionLedger(ledger_path).require_learned_allowance(
        updates=6, seconds=60.0, what="B2.2 acceptance comparison")
    env = {**os.environ, "PYTHONPATH": REPO_ROOT, "BRAMASTRA_LEARNED_CHECKS": "1"}

    def run_phase(name: str) -> dict:
        completed = subprocess.run(
            [sys.executable, "-c", PHASE_SCRIPT, name, workdir],
            capture_output=True, text=True, env=env, cwd=REPO_ROOT)
        if completed.returncode != 0:
            print(completed.stdout + completed.stderr, file=sys.stderr)
            raise SystemExit(f"phase {name} failed")
        return json.loads(completed.stdout.strip().splitlines()[-1])

    uninterrupted = run_phase("A")
    run_phase("B")
    resumed = run_phase("C")

    checksum_equal = (
        uninterrupted["model_checksum"] == resumed["model_checksum"]
        and uninterrupted["optimizer_checksum"] == resumed["optimizer_checksum"])
    counters_equal = uninterrupted["counters"] == resumed["counters"]
    state_equal = (
        uninterrupted["sampler_state"] == resumed["sampler_state"]
        and uninterrupted["controller_state"] == resumed["controller_state"]
        and uninterrupted["replay_cursor"] == resumed["replay_cursor"])
    future_equal = (
        uninterrupted["future_replay_slots"] == resumed["future_replay_slots"]
        and uninterrupted["next_controller_eval_due_at"]
        == resumed["next_controller_eval_due_at"])
    verdict = {
        "agrees": bool(checksum_equal and counters_equal and state_equal
                       and future_equal),
        "checksum_equal": checksum_equal,
        "counters_equal": counters_equal,
        "state_equal": state_equal,
        "future_event_sequence_equal": future_equal,
        "uninterrupted": uninterrupted,
        "resumed": resumed,
    }
    print(json.dumps(verdict, indent=2, sort_keys=True))
    return 0 if verdict["agrees"] else 1


if __name__ == "__main__":
    sys.exit(main())
