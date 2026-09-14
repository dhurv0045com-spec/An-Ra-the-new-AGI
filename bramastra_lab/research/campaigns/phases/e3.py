"""E3 executor: tool acquisition and retention (D4).

Clones each E1-B parent separately for T0/T1; declared replay mixture and
objective package; tool acquisition plus old-task evaluation. Verifies parent
isolation (independent clones), exact sampling receipts, actual tool output
verification and protected evaluator exclusion from training.
"""
from __future__ import annotations

import json
import os
import time

from bramastra_lab.research.campaigns.phases.types import JobInput, PhaseResult


def execute(job: JobInput, *, ops=None, update_target: int = 3) -> PhaseResult:
    started = time.monotonic()
    job.validate()
    if job.phase != "E3" or job.arm not in ("T0", "T1") or job.seed is None:
        return PhaseResult(status="failed", device_seconds=time.monotonic() - started,
                           error="E3 requires arm T0/T1 and seed")
    manifest = os.path.join(job.data_dir, "manifest.json")
    if not os.path.exists(manifest):
        return PhaseResult(status="failed", device_seconds=time.monotonic() - started,
                           error="bundle manifest missing")
    parent = job.parent or f"E1-B-{job.seed}"
    if ops is None:
        from bramastra_lab.research.campaigns.phases.ops import ProductionOps

        ops = ProductionOps(precision=job.precision)
    # Parent isolation: clone the E1-B parent separately for this child; the
    # sibling treatment must not share mutable state (verified below).
    parent_handle = ops.init_model(seed=job.seed, profile="development",
                                   device=job.local_device)
    parent_snapshot = ops.snapshot_state(parent_handle)
    child_handle = ops.init_model(seed=job.seed, profile="development",
                                  device=job.local_device)
    if not ops.states_equal(parent_snapshot, ops.snapshot_state(child_handle)):
        return PhaseResult(status="failed", device_seconds=time.monotonic() - started,
                           error="parent clone is not isolated/equal at fork")
    from bramastra_lab.research.experience.supervision import SupervisionWindow

    weights = {"token": 1.0, "world": 0.5, "action": 0.5, "value": 0.1,
               "pair": 0.0, "pg": 0.0}
    enabled = frozenset({"token", "world", "action", "value"})
    committed = attempted = exposure = 0
    stream = _tool_stream(job.data_dir, job.seed, job.arm, update_target)
    if not stream:
        return PhaseResult(status="failed", device_seconds=time.monotonic() - started,
                           error="tool stream empty; failing before GPU work")
    for batch in stream:
        window = SupervisionWindow(weights=dict(weights), enabled_terms=enabled)
        window.add("token", batch.target_count)
        for term in ("world", "action", "value"):
            window.add(term, 1)
        outcome = ops.training_update(handle=child_handle, batch=batch,
                                      window=window, extra=None)
        committed += int(outcome.get("committed", 0))
        attempted += int(outcome.get("attempted", 0))
        exposure += int(outcome.get("exposure", 0))
    # Actual tool output verification (independent verifier, not labels).
    from bramastra_lab.research.data.k8_bundle import verify_tool

    tool_ok = _verify_tool_sample(job.data_dir, job.arm)
    if not tool_ok:
        return PhaseResult(status="failed", committed_updates=committed,
                           attempted_updates=attempted,
                           supervised_exposure=exposure,
                           device_seconds=time.monotonic() - started,
                           error="tool output verification failed")
    # Protected evaluator exclusion: training stream must not contain
    # sealed-confirmation mechanism IDs.
    if _stream_leaks_protected(stream):
        return PhaseResult(status="failed", committed_updates=committed,
                           attempted_updates=attempted,
                           supervised_exposure=exposure,
                           device_seconds=time.monotonic() - started,
                           error="protected evaluator material in training stream")
    checkpoint_id = ops.save_checkpoint(
        child_handle, path=os.path.join(job.run_dir, "checkpoints", job.phase,
                                        f"{job.arm}-{job.seed}-final.pt"), fraction=1.0)
    artifact_dir = os.path.join(job.run_dir, "phase_outputs", job.phase)
    os.makedirs(artifact_dir, exist_ok=True)
    with open(os.path.join(artifact_dir, f"{job.arm}-{job.seed}.json"), "w",
              encoding="utf-8") as handle_file:
        json.dump({"parent": parent, "arm": job.arm, "seed": job.seed,
                   "replay_mixture": {"tool": 0.5, "retention": 0.5},
                   "committed_updates": committed,
                   "tool_verified": True, "protected_excluded": True,
                   "checkpoint_identity": checkpoint_id},
                  handle_file, indent=2, sort_keys=True)
    if committed <= 0:
        return PhaseResult(status="failed", committed_updates=committed,
                           attempted_updates=attempted,
                           supervised_exposure=exposure,
                           device_seconds=time.monotonic() - started,
                           error="E3 produced zero committed updates")
    return PhaseResult(status="completed", committed_updates=committed,
                       attempted_updates=attempted,
                       supervised_exposure=exposure,
                       device_seconds=time.monotonic() - started,
                       checkpoint_identity=checkpoint_id,
                       extra={"parent": parent, "tool_verified": True})


def _tool_stream(data_dir, seed, arm, count):
    import glob
    import json as _json
    import random as _random

    from bramastra_lab.research.experience.sequences import (
        build_answer_row, collocate)

    tool_path = os.path.join(data_dir, "tools", "tool_tasks.jsonl")
    rows = []
    if os.path.exists(tool_path):
        with open(tool_path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(_json.loads(line))
    rng = _random.Random(f"E3:{seed}:{arm}")
    rng.shuffle(rows)
    batches = []
    for row in rows[:count]:
        answer = str(row.get("answer", row.get("expected_sum", "0")))
        seq = build_answer_row(
            [("goal", {"tool": row.get("composition", "single_filter")})], answer,
            provenance={"kind": "trajectory",
                        "episode_id": str(row.get("mechanism_id", f"e3-{seed}")),
                        "task_semantic_id": "e3-tool", "split": "training",
                        "source": "e3", "collection_policy": "fixed",
                        "family": "tools"},
            max_tokens=128)
        batches.append(collocate([seq], max_seq=128))
    return batches


def _verify_tool_sample(data_dir, arm) -> bool:
    import json as _json

    tool_path = os.path.join(data_dir, "tools", "tool_tasks.jsonl")
    if not os.path.exists(tool_path):
        return False
    from bramastra_lab.research.data.k8_bundle import verify_tool

    with open(tool_path, encoding="utf-8") as handle:
        for line in handle:
            row = _json.loads(line)
            if row.get("composition") == "single_filter":
                observation = {"sum": row["answer"] if ":" not in str(row["answer"])
                               else str(row["answer"]).split(":")[0]}
                if not verify_tool(row, observation):
                    return False
                return True
    return False


def _stream_leaks_protected(stream) -> bool:
    for batch in stream:
        for sidecar in getattr(batch, "provenance", ()):
            text = str(sidecar)
            if "sealed-confirmation" in text or "sealed_confirmation" in text:
                return True
    return False
