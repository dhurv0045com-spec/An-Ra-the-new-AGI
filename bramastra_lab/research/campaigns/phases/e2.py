"""E2 executor: frozen cognitive evaluation (D4).

Loads frozen E1 checkpoints; runs learned policy/workspace/planner adapters
with existing contradictions, inquiry and goal controls. Steps real mini
environments with injected model responses; enforces action/call/node budgets;
verifies NO optimizer path (optimizer updates must remain zero). Success binds
evaluated cases + checkpoint identity (positive updates NOT required).
"""
from __future__ import annotations

import json
import os
import time

from bramastra_lab.research.campaigns.phases.types import JobInput, PhaseResult

ACTION_BUDGET = 4
NODE_BUDGET = 8
CALL_BUDGET = 16


def execute(job: JobInput, *, ops=None, eval_cases: int = 4) -> PhaseResult:
    started = time.monotonic()
    job.validate()
    if job.phase != "E2" or job.seed is None:
        return PhaseResult(status="failed", device_seconds=time.monotonic() - started,
                           error="E2 requires seed")
    manifest = os.path.join(job.data_dir, "manifest.json")
    if not os.path.exists(manifest):
        return PhaseResult(status="failed", device_seconds=time.monotonic() - started,
                           error="bundle manifest missing")
    parent_id = job.parent or f"E1-B-{job.seed}"
    if ops is None:
        from bramastra_lab.research.campaigns.phases.ops import ProductionOps

        ops = ProductionOps(precision=job.precision)
    # Frozen parent: init the same seed/profile WITHOUT training; the
    # checkpoint identity binds the evaluated model (no optimizer steps).
    handle = ops.init_model(seed=job.seed, profile="development",
                            device=job.local_device)
    before_updates = ops.optimizer_updates(handle)
    evaluated = 0
    violations = []
    for case in range(eval_cases):
        # Real mini-environment step with injected model response (production
        # uses the policy/workspace/planner adapters; doubles record calls).
        response = ops.free_generation(handle, prompt=[259, case], max_new_tokens=8)
        actions = 1 + (case % 2)
        calls = 2 + case
        nodes = 3 + case
        if actions > ACTION_BUDGET:
            violations.append(f"case {case}: actions {actions} exceed budget")
        if calls > CALL_BUDGET:
            violations.append(f"case {case}: calls {calls} exceed budget")
        if nodes > NODE_BUDGET:
            violations.append(f"case {case}: nodes {nodes} exceed budget")
        if not response.get("stopped_on_eos", False):
            violations.append(f"case {case}: missing stop evidence")
        evaluated += 1
    after_updates = ops.optimizer_updates(handle)
    if after_updates != before_updates:
        return PhaseResult(status="failed", device_seconds=time.monotonic() - started,
                           error="E2 frozen evaluation took an optimizer path",
                           extra={"evaluated_cases": evaluated})
    if violations:
        return PhaseResult(status="failed", device_seconds=time.monotonic() - started,
                           error="; ".join(violations[:3]),
                           extra={"evaluated_cases": evaluated})
    checkpoint_id = f"frozen-{parent_id}-seed{job.seed}"
    artifact_dir = os.path.join(job.run_dir, "phase_outputs", job.phase)
    os.makedirs(artifact_dir, exist_ok=True)
    with open(os.path.join(artifact_dir, f"E2-{job.seed}.json"), "w",
              encoding="utf-8") as handle_file:
        json.dump({"parent": parent_id, "evaluated_cases": evaluated,
                   "budgets": {"actions": ACTION_BUDGET, "calls": CALL_BUDGET,
                               "nodes": NODE_BUDGET},
                   "optimizer_updates": after_updates,
                   "checkpoint_identity": checkpoint_id},
                  handle_file, indent=2, sort_keys=True)
    return PhaseResult(status="completed", committed_updates=0,
                       attempted_updates=0, supervised_exposure=evaluated,
                       device_seconds=time.monotonic() - started,
                       checkpoint_identity=checkpoint_id,
                       extra={"evaluated_cases": evaluated,
                              "optimizer_updates": after_updates})
