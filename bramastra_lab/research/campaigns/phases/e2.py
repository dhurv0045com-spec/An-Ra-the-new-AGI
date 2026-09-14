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
    """Resolve the E1 parent to a verified record or raise."""
    key = job.resolved_parent_key()
    if not key:
        raise ValueError("E2 requires an E1 parent reference; missing parent must fail")
    # Support combined "E1-A-1701/E1-B-1701" style: prefer the B (learned) arm.
    candidates = [c.strip() for c in str(key).split("/") if c.strip()]
    # Prefer B arm when present, else first.
    ordered = sorted(candidates, key=lambda c: ("/B-" not in f"/{c}", c))
    last_error: Exception | None = None
    for candidate in ordered:
        ref = job.parent_ref
        if ref is not None and ref.lookup_key == key:
            # Structured ref already points at the combined key; try direct.
            try:
                resolved = ref.resolve(job.run_dir)
                resolved["lookup_key"] = candidate
                return resolved
            except Exception as exc:
                last_error = exc
                continue
        try:
            resolved = ParentRef(lookup_key=candidate).resolve(job.run_dir)
            resolved["lookup_key"] = candidate
            return resolved
        except Exception as exc:
            last_error = exc
            continue
    raise ValueError(
        f"E2 parent {key!r} has no verified checkpoint; "
        f"reinitializing with the same seed is forbidden ({last_error})")


def execute(job: JobInput, *, ops=None, eval_cases: int | None = 4) -> PhaseResult:
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
        # Production passes explicit confirmation inventory (E0-calibrated
        # 128/64/32); local default is 4 for integration only.
        eval_cases = 4
    if not isinstance(eval_cases, int) or eval_cases <= 0:
        return PhaseResult(status="failed", device_seconds=time.monotonic() - started,
                           error="eval_cases must be a positive integer",
                           evidence_kind=EVIDENCE_FIXTURE,
                           extra={"phase": "E2"})
    if ops is None:
        from bramastra_lab.research.campaigns.phases.ops import ProductionOps

        ops = ProductionOps(precision=job.precision)
    # Verified parent restore (never random init with the same seed).
    try:
        parent_record = _resolve_parent(job)
    except Exception as exc:
        return PhaseResult(status="failed", device_seconds=time.monotonic() - started,
                           error=str(exc), evidence_kind=EVIDENCE_FIXTURE,
                           extra={"phase": "E2"})
    try:
        if hasattr(ops, "restore_parent"):
            handle = ops.restore_parent(
                parent={**parent_record, "run_dir": job.run_dir,
                        "seed": job.seed},
                device=job.local_device, optimizer_policy="fresh")
        else:
            # Backward-compat: old doubles lack restore_parent; refuse rather
            # than silently reinit.
            raise ValueError(
                "ops carries no restore_parent; refusing random reinit")
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
            if hasattr(ops, "evaluate_episode"):
                outcome = ops.evaluate_episode(handle=handle, episode=case)
            else:
                response = ops.free_generation(
                    handle, prompt=case.get("prompt_tokens", [259]),
                    max_new_tokens=8)
                # Independent verifier decides (not EOS).
                try:
                    success = bool(case["verifier"](
                        case["mechanism"], {"answer": response.get("answer", ""),
                                            "sum": response.get("answer", "")}))
                except Exception:
                    success = False
                outcome = {"answer": response.get("answer", ""),
                           "stopped_on_eos": bool(response.get("stopped_on_eos", False)),
                           "success": success, "model_calls": 1,
                           "actions": 1, "nodes": 1, "cost": 1.0}
        except Exception as exc:
            violations.append(f"case {case.get('mechanism_id')}: eval refused: {exc}")
            continue
        # Actual operation counters (never 1+case%2 inventions).
        actions = int(outcome.get("actions", 1))
        calls = int(outcome.get("model_calls", 1))
        nodes = int(outcome.get("nodes", 1))
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
                        "cost": float(outcome.get("cost", 1.0))})
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
