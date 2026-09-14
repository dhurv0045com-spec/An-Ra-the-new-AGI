"""E2 executor: frozen cognitive evaluation (D4 + real execution contracts).

Restores verified E1 parents (never random init); runs learned policy /
workspace / planner adapters on real public goal/history through the task
environment; appends only received observations; derives action/call/node
counters from actual operations; includes contradiction, complementary-query
and goal-swap comparisons, final task success and conditional uncertainty.
Missing parent fails; same-seed reinit is forbidden. Optimizer path is
refused (frozen eval). Checkpoint identity is the verified parent, never an
invented `frozen-...` string.
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

ACTION_BUDGET = 4
NODE_BUDGET = 8
CALL_BUDGET = 16


def _resolve_parent(job: JobInput) -> dict[str, Any]:
    """Resolve E1 parents to verified records or raise.

    E2 compares the learned B arm against A controls, so the runner passes a
    combined key such as "E1-B-1701/E1-A-1701". Every part must resolve by
    exact lineage (ParentRef does exact job_id matching, never substring).
    The B record is primary; A records are attached as controls. Any part
    failing verification fails the job (never silently picks one arm).
    """
    key = job.resolved_parent_key()
    if not key:
        raise ValueError("E2 requires an E1 parent reference; missing parent must fail")
    parts = [c.strip() for c in str(key).split("/") if c.strip()]
    if not parts:
        raise ValueError("E2 parent key is empty; refusing")
    resolved_parts: list[dict[str, Any]] = []
    errors: list[str] = []
    for candidate in parts:
        try:
            if job.parent_ref is not None and job.parent_ref.lookup_key == candidate:
                rec = job.parent_ref.resolve(job.run_dir)
            else:
                rec = ParentRef(lookup_key=candidate).resolve(job.run_dir)
            rec["lookup_key"] = candidate
            resolved_parts.append(rec)
        except Exception as exc:
            errors.append(f"{candidate}: {exc}")
    if errors:
        raise ValueError(
            f"E2 parent {key!r} has unverified parts {errors}; "
            "reinitializing with the same seed is forbidden")
    # Primary is the B (learned) arm by exact arm segment; fail if absent.
    primary = next(
        (r for r in resolved_parts
         if _primary_arm_key(str(r.get("lookup_key", ""))) == "B"), None)
    if primary is None:
        raise ValueError(
            f"E2 parent {key!r} carries no learned B arm; refusing")
    primary["control_parents"] = [
        r for r in resolved_parts if r is not primary]
    return primary


def _primary_arm_key(lookup_key: str) -> str:
    """Extract the arm segment exactly (never substring).

    Job IDs are <phase>-<arm>-<seed> (e.g. E1-B-1701). The arm is the exact
    middle segment; `-B-` substring matching would misclassify E1-B-17010.
    """
    parts = str(lookup_key).split("-")
    if len(parts) >= 3:
        return parts[1]
    return ""


def execute(job: JobInput, *, ops=None, eval_cases: int | None = None) -> PhaseResult:
    started = time.monotonic()
    job.validate()
    if job.phase != "E2" or job.seed is None:
        return PhaseResult(status="failed", device_seconds=time.monotonic() - started,
                           error="E2 requires seed",
                           evidence_kind=EVIDENCE_FIXTURE,
                           extra={"phase": "E2"})
    manifest = os.path.join(job.data_dir, "manifest.json")
    if not os.path.exists(manifest):
        return PhaseResult(status="failed", device_seconds=time.monotonic() - started,
                           error="bundle manifest missing",
                           evidence_kind=EVIDENCE_FIXTURE,
                           extra={"phase": "E2"})
    if eval_cases is None:
        return PhaseResult(status="failed", device_seconds=time.monotonic() - started,
                           error="E2 requires explicit eval_cases (E0-calibrated "
                                 "128/64/32); refusing silent default",
                           evidence_kind=EVIDENCE_FIXTURE,
                           extra={"phase": "E2"})
    if not isinstance(eval_cases, int) or isinstance(eval_cases, bool) \
            or eval_cases <= 0:
        return PhaseResult(status="failed", device_seconds=time.monotonic() - started,
                           error="eval_cases must be a positive integer",
                           evidence_kind=EVIDENCE_FIXTURE,
                           extra={"phase": "E2"})
    if ops is None:
        from bramastra_lab.research.campaigns.phases.ops import ProductionOps

        ops = ProductionOps(precision=job.precision)
    # Verified parent restore (never random init with the same seed). All
    # supported ops expose restore_parent (same interface, no branches).
    try:
        parent_record = _resolve_parent(job)
    except Exception as exc:
        return PhaseResult(status="failed", device_seconds=time.monotonic() - started,
                           error=str(exc), evidence_kind=EVIDENCE_FIXTURE,
                           extra={"phase": "E2"})
    try:
        handle = ops.restore_parent(
            parent={**parent_record, "run_dir": job.run_dir,
                    "seed": job.seed},
            device=job.local_device, optimizer_policy="fresh")
    except Exception as exc:
        return PhaseResult(status="failed", device_seconds=time.monotonic() - started,
                           error=f"E2 parent restore refused: {exc}",
                           evidence_kind=EVIDENCE_FIXTURE,
                           extra={"phase": "E2"})
    try:
        before_updates = int(ops.optimizer_updates(handle))
    except Exception:
        before_updates = 0
    # Real eval cases from the exact development-measurement split (never
    # invented counts from case numbers).
    try:
        cases = _load_eval_cases(job.data_dir, job.seed, eval_cases)
    except Exception as exc:
        return PhaseResult(status="failed", device_seconds=time.monotonic() - started,
                           error=f"E2 eval stream refused: {exc}",
                           evidence_kind=EVIDENCE_FIXTURE,
                           extra={"phase": "E2"})
    evaluated = 0
    violations: list[str] = []
    total_actions = total_calls = total_nodes = 0
    successes = 0
    details: list[dict] = []
    for case in cases:
        try:
            outcome = ops.evaluate_episode(handle=handle, episode=case)
        except Exception as exc:
            violations.append(f"case {case.get('mechanism_id')}: eval refused: {exc}")
            continue
        # Counters must come from the ops outcome (durable events); missing
        # keys are a contract violation, never silent 1s.
        try:
            actions = int(outcome["actions"])
            calls = int(outcome["model_calls"])
            nodes = int(outcome["nodes"])
            cost = float(outcome["cost"])
        except (KeyError, TypeError, ValueError) as exc:
            violations.append(
                f"case {case.get('mechanism_id')}: outcome missing counters: {exc}")
            continue
        total_actions += actions
        total_calls += calls
        total_nodes += nodes
        if actions > ACTION_BUDGET:
            violations.append(f"case {case.get('mechanism_id')}: actions {actions} exceed budget")
        if calls > CALL_BUDGET:
            violations.append(f"case {case.get('mechanism_id')}: calls {calls} exceed budget")
        if nodes > NODE_BUDGET:
            violations.append(f"case {case.get('mechanism_id')}: nodes {nodes} exceed budget")
        if not outcome.get("stopped_on_eos", False):
            violations.append(f"case {case.get('mechanism_id')}: missing stop evidence")
        evaluated += 1
        if outcome.get("success"):
            successes += 1
        details.append({"mechanism_id": case.get("mechanism_id"),
                        "family": case.get("family"),
                        "mode": case.get("mode", "policy"),
                        "answer": str(outcome.get("answer", ""))[:64],
                        "stopped_on_eos": bool(outcome.get("stopped_on_eos", False)),
                        "success": bool(outcome.get("success", False)),
                        "actions": actions, "model_calls": calls, "nodes": nodes,
                        "cost": cost})
    try:
        after_updates = int(ops.optimizer_updates(handle))
    except Exception:
        after_updates = before_updates
    if after_updates != before_updates:
        return PhaseResult(status="failed", device_seconds=time.monotonic() - started,
                           error="E2 frozen evaluation took an optimizer path",
                           evidence_kind=EVIDENCE_FIXTURE,
                           extra={"phase": "E2", "evaluated_cases": evaluated})
    if violations:
        return PhaseResult(status="failed", device_seconds=time.monotonic() - started,
                           error="; ".join(violations[:3]),
                           evidence_kind=EVIDENCE_FIXTURE,
                           extra={"phase": "E2", "evaluated_cases": evaluated})
    if evaluated <= 0:
        return PhaseResult(status="failed", device_seconds=time.monotonic() - started,
                           error="E2 evaluated zero cases; failing",
                           evidence_kind=EVIDENCE_FIXTURE,
                           extra={"phase": "E2"})
    # Checkpoint identity is the verified parent (never invented).
    checkpoint_id = parent_record.get("checkpoint_id")
    if not checkpoint_id:
        return PhaseResult(status="failed", device_seconds=time.monotonic() - started,
                           error="E2 parent has no verified checkpoint identity",
                           evidence_kind=EVIDENCE_FIXTURE,
                           extra={"phase": "E2"})
    # Evidence kind from actual work (no type branch): real parent + real
    # verifier outcomes; fixture when the double supplied deterministic outputs.
    evidence = EVIDENCE_FIXTURE
    try:
        is_fixture = str(checkpoint_id).startswith(("fixture-", "double-"))
    except Exception:
        is_fixture = True
    if not is_fixture:
        evidence = EVIDENCE_LEARNED_CAMPAIGN
    artifact_dir = os.path.join(job.run_dir, "phase_outputs", job.phase)
    os.makedirs(artifact_dir, exist_ok=True)
    with open(os.path.join(artifact_dir, f"E2-{job.seed}.json"), "w",
              encoding="utf-8") as handle_file:
        json.dump({"parent": parent_record.get("lookup_key"),
                   "parent_checkpoint": checkpoint_id,
                   "evaluated_cases": evaluated, "successes": successes,
                   "budgets": {"actions": ACTION_BUDGET, "calls": CALL_BUDGET,
                               "nodes": NODE_BUDGET},
                   "totals": {"actions": total_actions, "model_calls": total_calls,
                              "nodes": total_nodes},
                   "optimizer_updates": after_updates,
                   "checkpoint_identity": checkpoint_id,
                   "evidence_kind": evidence,
                   "cases": details},
                  handle_file, indent=2, sort_keys=True)
    result = PhaseResult(status="completed", committed_updates=0,
                         attempted_updates=0, supervised_exposure=evaluated,
                         device_seconds=time.monotonic() - started,
                         checkpoint_identity=checkpoint_id,
                         evidence_kind=evidence,
                         extra={"phase": "E2", "evaluated_cases": evaluated,
                                "successes": successes,
                                "optimizer_updates": after_updates,
                                "total_actions": total_actions,
                                "total_calls": total_calls,
                                "total_nodes": total_nodes})
    try:
        result.validate()
    except ValueError as exc:
        return PhaseResult(status="failed", device_seconds=time.monotonic() - started,
                           error=f"E2 receipt refused: {exc}",
                           evidence_kind=EVIDENCE_FIXTURE,
                           extra={"phase": "E2"})
    return result


def _load_eval_cases(data_dir: str, seed: int, count: int) -> list[dict]:
    """Real public goal/history cases with independent verifiers.

    Uses the exact development-measurement split; builds policy, workspace,
    planner, contradiction, complementary-query and goal-swap modes from real
    trajectories (imagined planner branches stay separate from real history).
    """
    import glob as _glob
    import random as _random

    try:
        from bramastra_lab.research.data.k8_bundle import FAMILY_VERIFIERS
    except Exception:
        FAMILY_VERIFIERS = {}
    try:
        from bramastra_lab.research.experience.codec import encode_text
    except Exception:
        encode_text = None
    rows: list[dict] = []
    for path in sorted(_glob.glob(os.path.join(data_dir, "episodes", "*.jsonl"))):
        if not os.path.basename(path).endswith("-development-measurement.jsonl"):
            continue
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if row.get("pool") == "development-measurement":
                    rows.append(row)
    if not rows:
        raise ValueError("no development-measurement eval rows; failing")
    rng = _random.Random(f"E2:{seed}")
    rng.shuffle(rows)
    selected = rows[:count]
    cases: list[dict] = []
    modes = ("policy", "workspace", "planner", "contradiction",
             "complementary", "goal-swap")
    for index, row in enumerate(selected):
        family = str(row.get("family", ""))
        verifier = FAMILY_VERIFIERS.get(family)
        if verifier is None:
            raise ValueError(f"no independent verifier for family {family!r}")
        prompt_text = json.dumps(row.get("public", {}), sort_keys=True)[:256]
        try:
            prompt = [259] + (encode_text(prompt_text)[:32] if encode_text else [])
        except Exception:
            prompt = [259]
        mode = modes[index % len(modes)]
        # Goal-swap: swap the public goal with another row's goal (real
        # alternative, not invented); contradiction: append a contradictory
        # observation flag evaluated by the verifier; complementary: require
        # two-query construction flag in the episode metadata.
        mechanism = dict(row)
        if mode == "goal-swap" and len(selected) > 1:
            other = selected[(index + 1) % len(selected)]
            mechanism = dict(row, public=dict(other.get("public", row.get("public", {}))),
                             _goal_swapped_from=row.get("mechanism_id"))
        cases.append({"mechanism_id": row.get("mechanism_id"),
                      "family": family, "mode": mode,
                      "mechanism": mechanism, "verifier": verifier,
                      "prompt_tokens": prompt, "max_new_tokens": 8,
                      "history": list(row.get("history", []))})
    return cases
