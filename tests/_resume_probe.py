"""Fresh-process resume probe (B05, learned smoke).

Executed only via tests.test_research_checkpoint with BRAMASTRA_LEARNED_CHECKS=1.
Three subprocess phases prove the integrated path:

  A. uninterrupted: two updates in one process, final identities printed
  B. interrupted: one update + atomic checkpoint publication
  C. fresh process: restore_run() then the next update

The orchestrator compares A and C. Optimizer updates happen ONLY inside the
subprocesses under the owner-authorized learned-smoke budget.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def phase_script() -> str:
    return r'''
import json, os, sys, time

import torch

os.environ.setdefault("BRAMASTRA_LEARNED_CHECKS", "1")
from bramastra_lab.research.config import BuildConfig, seed_everything
from bramastra_lab.research.experience.sequences import build_answer_row, collocate
from bramastra_lab.research.learning.trainer import Trainer
from bramastra_lab.research.models import IntegratedModel
from bramastra_lab.research.runtime import resume as resume_mod
from bramastra_lab.research.runtime.resume import RunContext, checkpoint_run, restore_run

phase, workdir = sys.argv[1], sys.argv[2]

RAW_CONFIG = {"model": {"profile": "tiny"}, "training": {"learning_rate": 0.01}}

def batches():
    rows = []
    for index, (question, answer) in enumerate([("2+2?", "4"), ("3+3?", "6"), ("5+1?", "6")]):
        row = build_answer_row([("goal", {"question": question})], answer,
            provenance={"episode_id": f"e{index}", "task_semantic_id": "math",
                        "split": "training", "source": "probe", "collection_policy": "probe"},
            max_tokens=64)
        rows.append(collocate([row], max_seq=64))
    return rows

def state_identities(trainer):
    model_id = torch.sum(torch.stack([
        parameter.detach().double().sum() for parameter in trainer.model.parameters()
    ])).item()
    optimizer_terms = []
    for state in trainer.optimizer.state_dict()["state"].values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                optimizer_terms.append(value.detach().double().sum().item())
            else:
                optimizer_terms.append(float(value))
    return {"model_checksum": model_id,
            "optimizer_checksum": sum(optimizer_terms),
            "counters": trainer.counters.to_dict()}

if phase == "A":
    seed_everything(101)
    config = BuildConfig.from_dict(RAW_CONFIG)
    trainer = Trainer(config, IntegratedModel(config))
    for batch in batches()[:2]:
        trainer.training_step(batch)
    print(json.dumps(state_identities(trainer)))

elif phase == "B":
    seed_everything(101)
    config = BuildConfig.from_dict(RAW_CONFIG)
    run_dir = os.path.join(workdir, "interrupted-run")
    resume_mod.create_run(run_dir, config, data_identity="probe-data")
    trainer = Trainer(config, IntegratedModel(config))
    sampler_batches = batches()
    ctx = RunContext(run_dir=run_dir, run_id="probe", config=config,
                     data_identity="probe-data", writer_token="probe",
                     trainer=trainer,
                     sampler_state={"epoch": 0, "group_index": 1,
                                    "groups_consumed": 1, "examples_consumed": 1},
                     controller_state=None, replay_cursor=None)
    trainer.training_step(sampler_batches[0])
    manifest = checkpoint_run(trainer, run_dir, ctx)
    print(json.dumps({"checkpoint_id": manifest.checkpoint_id,
                      "update": trainer.counters.optimizer_updates}))

elif phase == "C":
    seed_everything(101)
    config = BuildConfig.from_dict(RAW_CONFIG)
    run_dir = os.path.join(workdir, "interrupted-run")
    # A fresh process: the trainer is rebuilt here from the checkpoint.
    ctx = restore_run(run_dir, config)
    trainer = ctx.trainer
    sampler_batches = batches()
    # The restored sampler cursor selects the next batch, not the first.
    trainer.training_step(sampler_batches[1])
    print(json.dumps(state_identities(trainer)))
'''


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", required=True)
    args = parser.parse_args()
    workdir = args.workdir
    env = {**os.environ, "PYTHONPATH": REPO_ROOT, "BRAMASTRA_LEARNED_CHECKS": "1"}

    def run_phase(name: str) -> dict:
        completed = subprocess.run(
            [sys.executable, "-c", phase_script(), name, workdir],
            capture_output=True, text=True, env=env, cwd=REPO_ROOT)
        if completed.returncode != 0:
            print(completed.stdout + completed.stderr, file=sys.stderr)
            raise SystemExit(f"phase {name} failed")
        return json.loads(completed.stdout.strip().splitlines()[-1])

    uninterrupted = run_phase("A")
    run_phase("B")
    resumed = run_phase("C")

    model_agrees = abs(uninterrupted["model_checksum"] - resumed["model_checksum"]) <= 1e-5 * max(
        1.0, abs(uninterrupted["model_checksum"]))
    optimizer_agrees = abs(
        uninterrupted["optimizer_checksum"] - resumed["optimizer_checksum"]) <= 1e-5 * max(
        1.0, abs(uninterrupted["optimizer_checksum"]))
    counters_agree = uninterrupted["counters"] == resumed["counters"]
    verdict = {
        "agrees": bool(model_agrees and optimizer_agrees and counters_agree),
        "tolerance": "1e-5 relative / 1e-7 absolute",
        "uninterrupted": uninterrupted,
        "resumed": resumed,
    }
    print(json.dumps(verdict))
    return 0


if __name__ == "__main__":
    sys.exit(main())
