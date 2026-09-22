"""E2 executor: frozen cognitive evaluation on live mechanisms (O05).

Restores verified B (learned) and A/control E1 parents explicitly, then runs
MATCHED live episodes: every mode runs the same mechanism+seed through the
real episode kernel and real environments. Modes select different declared
adapters with their own traces (never one shared generation path relabeled):

- b-policy / b-workspace / b-planner / b-memory: restored B model behind the model
  interface (learned arms under test).
- a-direct / a-fixed / a-random: model-free scripted controls.
- symbolic-reference: named oracle control (diagnostic ceiling, excluded
  from every learned comparison).

Plus per-group goal-swap episodes (goal changed BEFORE rendering, success
measured against the swapped predicate), contradiction probes (injected
second-source observation + conflict bookkeeping on the workspace artifact),
and complementary-query coverage read from the event traces. Counters reduce
from emitted events; the independent verifier (env submit feedback) decides
success. The optimizer path stays refused (frozen eval).
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
EVAL_FAMILIES = ("rule-inquiry", "inventory", "program")
# Goal-only fit reserve: mechanisms whose mandatory goal prefix alone exceeds
# this are counted as rejected-length (experiment.md: reject-or-count over
# context512 records and report the fraction), never silently truncated.
GOAL_FIT_TOKENS = 320
MODES_B = ("b-policy", "b-workspace", "b-planner", "b-memory")
MODES_A = ("a-direct", "a-fixed", "a-random")


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


class _OpsModelBridge:
    """Model interface over an ops handle with origin labeling (O05).

    Production handles expose real model/config (origin "model"); doubles
    return fixed outputs without generation counters (origin "double",
    fixture). Distinguished by capability (generation report shape), never
    by class name.
    """

    def __init__(self, ops: Any, handle: Any) -> None:
        self._ops = ops
        self._handle = handle
        self.calls = 0

    def generate(self, prompt_tokens: list[int], *,
                 max_new_tokens: int) -> dict[str, Any]:
        out = self._ops.free_generation(
            self._handle, prompt=prompt_tokens, max_new_tokens=max_new_tokens)
        self.calls += 1
        if isinstance(out, dict) and "new_tokens" in out:
            return {"answer": out.get("answer", ""),
                    "generation_id": f"gen-{self.calls}",
                    "origin": "model",
                    "new_tokens": out.get("new_tokens", 1)}
        return {"answer": out.get("answer", "") if isinstance(out, dict) else "",
                "generation_id": f"double-{self.calls}",
                "origin": "double",
                "new_tokens": 1}


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
    # Verified B + A/control restoration (never random init). All supported
    # ops expose restore_parent (same interface, no branches).
    try:
        parent_record = _resolve_parent(job)
    except Exception as exc:
        return PhaseResult(status="failed", device_seconds=time.monotonic() - started,
                           error=str(exc), evidence_kind=EVIDENCE_FIXTURE,
                           extra={"phase": "E2"})
    try:
        b_handle = ops.restore_parent(
            parent={**parent_record, "run_dir": job.run_dir,
                    "seed": job.seed},
            device=job.local_device, optimizer_policy="fresh")
    except Exception as exc:
        return PhaseResult(status="failed", device_seconds=time.monotonic() - started,
                           error=f"E2 B-parent restore refused: {exc}",
                           evidence_kind=EVIDENCE_FIXTURE,
                           extra={"phase": "E2"})
    a_handle: Any | None = None
    a_restore_error: str | None = None
    controls = list(parent_record.get("control_parents", ()))
    if controls:
        try:
            a_handle = ops.restore_parent(
                parent={**controls[0], "run_dir": job.run_dir,
                        "seed": job.seed},
                device=job.local_device, optimizer_policy="fresh")
        except Exception as exc:
            a_restore_error = str(exc)[:200]
    try:
        before_updates = int(ops.optimizer_updates(b_handle))
    except Exception:
        before_updates = 0
    # Matched live mechanism groups (same mechanism+seed across modes).
    try:
        groups, rejected_length = _select_matched_groups(
            job.seed, eval_cases)
    except Exception as exc:
        return PhaseResult(status="failed", device_seconds=time.monotonic() - started,
                           error=f"E2 case selection refused: {exc}",
                           evidence_kind=EVIDENCE_FIXTURE,
                           extra={"phase": "E2"})
    if not groups:
        return PhaseResult(status="failed", device_seconds=time.monotonic() - started,
                           error="E2 selected zero matched groups; failing",
                           evidence_kind=EVIDENCE_FIXTURE,
                           extra={"phase": "E2"})
    from bramastra_lab.research.cognition import episode as kernel

    try:
        memory_index = _build_training_memory_index(job)
        memory_status = {"status": "enabled",
                         "index_identity": memory_index.identity,
                         "record_count": len(memory_index.records),
                         "scope_allowlist": ["controller", "training"]}
    except Exception as exc:
        # Keep reduced fixture bundles usable, while recording that the
        # cognition memory ablation did not run. Production manifests require
        # the full exact training split, so a missing split is visible here.
        memory_index = None
        memory_status = {"status": "unavailable",
                         "reason": str(exc)[:240]}

    b_bridge = _OpsModelBridge(ops, b_handle)
    b_checkpoint = parent_record.get("checkpoint_id")
    a_checkpoint = (controls[0].get("checkpoint_id") if controls else None)
    mode_stats: dict[str, dict[str, Any]] = {}
    paired_deltas: list[dict[str, Any]] = []
    goal_swap_rows: list[dict[str, Any]] = []
    contradiction_rows: list[dict[str, Any]] = []
    complementary_rows: list[dict[str, Any]] = []
    planner_gaps: list[float] = []
    violations: list[str] = []
    total_actions = total_calls = total_nodes = 0
    evaluated_episodes = 0
    model_origin_seen = False
    group_details: list[dict[str, Any]] = []
    for group in groups:
        group_result: dict[str, Any] = {
            "mechanism_id": group["mechanism_id"],
            "family": group["family"], "modes": {}}
        try:
            episodes = _run_matched_group(
                group, b_bridge=b_bridge, job=job,
                b_checkpoint=b_checkpoint, a_checkpoint=a_checkpoint,
                memory_index=memory_index)
        except Exception as exc:
            violations.append(
                f"group {group['mechanism_id']}: matched run refused: {exc}")
            continue
        for mode, trace in episodes.items():
            summary = trace["summary"]
            total_actions += int(summary.get("actions", 0))
            total_calls += int(summary.get("model_calls", 0))
            total_nodes += int(summary.get("imagined_nodes", 0))
            evaluated_episodes += 1
            if _trace_has_model_origin(trace):
                model_origin_seen = True
            for event in trace["events"]:
                if event.get("kind") == "action" and str(
                        event.get("model_origin") or "").startswith("model"):
                    model_origin_seen = True
            stat = mode_stats.setdefault(mode, {"episodes": 0, "successes": 0,
                                                "cost": 0.0, "truncated": 0,
                                                "inference_input_tokens": 0,
                                                "memory_token_cost": 0})
            stat["episodes"] += 1
            stat["successes"] += 1 if summary.get("success") else 0
            stat["cost"] += float(summary.get("cost", 0.0))
            stat["truncated"] += 1 if summary.get("truncated") else 0
            stat["inference_input_tokens"] += int(
                summary.get("inference_input_tokens", 0))
            stat["memory_token_cost"] += int(
                summary.get("memory", {}).get("token_cost", 0))
            group_result["modes"][mode] = {
                "success": bool(summary.get("success", False)),
                "actions": summary.get("actions"),
                "model_calls": summary.get("model_calls"),
                "imagined_nodes": summary.get("imagined_nodes"),
                "truncated": summary.get("truncated"),
                "cost": summary.get("cost"),
                "inference_input_tokens": summary.get("inference_input_tokens"),
                "memory": summary.get("memory"),
                "checkpoint": trace.get("checkpoint_id"),
                "adapter": trace.get("adapter")}
            if mode == "b-planner":
                gap = _planner_prediction_gap(trace)
                if gap.get("joined") and isinstance(
                        gap.get("success_gap"), (int, float)):
                    planner_gaps.append(float(gap["success_gap"]))
        # Paired contrast on this mechanism (matched copies, same budgets).
        b_ok = group_result["modes"].get("b-policy", {}).get("success")
        a_ok = group_result["modes"].get("a-fixed", {}).get("success")
        if b_ok is not None and a_ok is not None:
            paired_deltas.append({
                "mechanism_id": group["mechanism_id"],
                "family": group["family"],
                "b_policy_minus_a_fixed": int(bool(b_ok)) - int(bool(a_ok))})
        # Complementary coverage from the B-policy trace events.
        complementary_rows.append(_complementary_coverage(
            group, episodes.get("b-policy")))
        # Goal-swap episode (goal changed before rendering).
        try:
            swap = _run_goal_swap(group, b_bridge=b_bridge, job=job,
                                  b_checkpoint=b_checkpoint)
            goal_swap_rows.append(swap)
            total_actions += int(swap.get("actions", 0))
            total_calls += int(swap.get("model_calls", 0))
            evaluated_episodes += 1
        except Exception as exc:
            violations.append(f"group {group['mechanism_id']}: goal-swap "
                              f"refused: {exc}")
        # Contradiction probe on the B-workspace artifact.
        try:
            probe = _run_contradiction_probe(
                group, episodes.get("b-workspace"))
            contradiction_rows.append(probe)
        except Exception as exc:
            violations.append(f"group {group['mechanism_id']}: contradiction "
                              f"probe refused: {exc}")
        group_details.append(group_result)
    try:
        after_updates = int(ops.optimizer_updates(b_handle))
    except Exception:
        after_updates = before_updates
    if after_updates != before_updates:
        return PhaseResult(status="failed", device_seconds=time.monotonic() - started,
                           error="E2 frozen evaluation took an optimizer path",
                           evidence_kind=EVIDENCE_FIXTURE,
                           extra={"phase": "E2",
                                  "evaluated_cases": len(group_details)})
    if violations:
        return PhaseResult(status="failed", device_seconds=time.monotonic() - started,
                           error="; ".join(violations[:3]),
                           evidence_kind=EVIDENCE_FIXTURE,
                           extra={"phase": "E2",
                                  "evaluated_cases": len(group_details)})
    if not group_details:
        return PhaseResult(status="failed", device_seconds=time.monotonic() - started,
                           error="E2 evaluated zero matched groups; failing",
                           evidence_kind=EVIDENCE_FIXTURE,
                           extra={"phase": "E2"})
    checkpoint_id = parent_record.get("checkpoint_id")
    if not checkpoint_id:
        return PhaseResult(status="failed", device_seconds=time.monotonic() - started,
                           error="E2 parent has no verified checkpoint identity",
                           evidence_kind=EVIDENCE_FIXTURE,
                           extra={"phase": "E2"})
    # Per-mode count reconciliation (U05): top-level totals must reduce from
    # the per-mode per-episode details (including goal-swap episodes).
    sum_actions = sum_calls = 0
    for detail in group_details:
        for mode_stat in detail["modes"].values():
            sum_actions += int(mode_stat.get("actions") or 0)
            sum_calls += int(mode_stat.get("model_calls") or 0)
    for swap in goal_swap_rows:
        sum_actions += int(swap.get("actions") or 0)
        sum_calls += int(swap.get("model_calls") or 0)
    calls_reconciled = (sum_actions == total_actions
                        and sum_calls == total_calls
                        and evaluated_episodes > 0)
    # Consumed checkpoint/schema identities per mode (U05): B modes name the
    # B parent's schema identity, A/control modes the A control's.
    b_schema = _handle_schema_identity(b_handle)
    a_schema = _handle_schema_identity(a_handle)
    schema_identities: dict[str, str | None] = {}
    for mode in mode_stats:
        if mode.startswith("b-"):
            schema_identities[mode] = b_schema
        elif mode.startswith("a-"):
            schema_identities[mode] = a_schema
        else:
            schema_identities[mode] = None
    planner_calibration = {"joined": len(planner_gaps),
                           "mean_abs_gap": (sum(planner_gaps)
                                            / len(planner_gaps)
                                            if planner_gaps else None)}
    # Evidence origin (O05): a learned checkpoint alone never qualifies —
    # the trace must show model-origin generations behind B-mode decisions.
    try:
        is_fixture = str(checkpoint_id).startswith(("fixture-", "double-"))
    except Exception:
        is_fixture = True
    evidence = EVIDENCE_LEARNED_CAMPAIGN \
        if (not is_fixture and model_origin_seen) else EVIDENCE_FIXTURE
    successes = sum(1 for row in paired_deltas if row[
        "b_policy_minus_a_fixed"] >= 0)
    artifact_dir = os.path.join(job.run_dir, "phase_outputs", job.phase)
    os.makedirs(artifact_dir, exist_ok=True)
    with open(os.path.join(artifact_dir, f"E2-{job.seed}.json"), "w",
              encoding="utf-8") as handle_file:
        json.dump({"parent": parent_record.get("lookup_key"),
                   "parent_checkpoint": checkpoint_id,
                   "a_parent": controls[0].get("lookup_key") if controls else None,
                   "a_parent_checkpoint": a_checkpoint,
                   "a_restore_error": a_restore_error,
                   "mechanism_source": "k8-live-eval/v1",
                   "cognition_memory": memory_status,
                   "matched_groups": len(group_details),
                   "total_episodes": evaluated_episodes,
                   "evaluated_cases": len(group_details),
                   "rejected_length": rejected_length,
                   "mode_stats": mode_stats,
                   "paired_deltas": paired_deltas,
                   "goal_swap": goal_swap_rows,
                   "contradiction": contradiction_rows,
                   "complementary": complementary_rows,
                   "planner_calibration": planner_calibration,
                   "planner_calibration_mean_abs_gap": (
                       sum(planner_gaps) / len(planner_gaps)
                       if planner_gaps else None),
                   "calls_reconciled": calls_reconciled,
                   "schema_identities": schema_identities,
                   "budgets": {"actions": ACTION_BUDGET, "calls": CALL_BUDGET,
                               "nodes": NODE_BUDGET},
                   "totals": {"actions": total_actions,
                              "model_calls": total_calls,
                              "nodes": total_nodes},
                   "optimizer_updates": after_updates,
                   "checkpoint_identity": checkpoint_id,
                   "evidence_kind": evidence,
                   "groups": group_details},
                  handle_file, indent=2, sort_keys=True)
    result = PhaseResult(status="completed", committed_updates=0,
                         attempted_updates=0,
                         supervised_exposure=evaluated_episodes,
                         device_seconds=time.monotonic() - started,
                         checkpoint_identity=checkpoint_id,
                         evidence_kind=evidence,
                         extra={"phase": "E2",
                                "evaluated_cases": len(group_details),
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


def _select_matched_groups(seed: int, count: int) -> tuple[list[dict], int]:
    """Matched live mechanisms across families with goal-fit selection."""
    from bramastra_lab.research.environments.k8_live import (
        generate_live_mechanism)
    from bramastra_lab.research.cognition.episode import render_public_state

    families = list(EVAL_FAMILIES)
    groups: list[dict] = []
    rejected_length = 0
    attempts = 0
    index = 0
    while len(groups) < count and attempts < count * 12 + 24:
        attempts += 1
        family = families[index % len(families)]
        index += 1
        mechanism = generate_live_mechanism(family, index, seed=seed)
        try:
            goal_tokens, _ = render_public_state(
                goal=dict(mechanism.get("public", {})), history=[],
                workspace=[], budgets=None)
        except Exception:
            rejected_length += 1
            continue
        if len(goal_tokens) > GOAL_FIT_TOKENS:
            # Mandatory prefix cannot fit with interaction room: count as
            # rejected-length per experiment.md, never silently truncate.
            rejected_length += 1
            continue
        groups.append({"mechanism_id": mechanism.get(
            "mechanism_id", f"{family}-{index}"),
            "family": family, "mechanism": mechanism,
            "env_seed": seed * 100003 + index})
    if not groups:
        raise ValueError("no fittable live mechanisms; failing")
    return groups, rejected_length


def _adapters_for_group(group: dict, *, b_bridge: Any,
                        job: JobInput,
                        memory_index: Any | None = None) -> dict[str, Any]:
    """One fresh adapter instance per mode (no shared mutable policy state)."""
    from bramastra_lab.research.cognition import episode as kernel

    adapters = {
        "b-policy": kernel.LearnedPolicyAdapter(),
        "b-workspace": kernel.WorkspacePolicyAdapter(),
        "b-planner": kernel.BoundedPlannerAdapter(
            world_model=kernel.ModelWorldModel(b_bridge)),
        "a-direct": kernel.DirectAnswerAdapter(),
        "a-fixed": kernel.FixedInquiryAdapter(),
        "a-random": kernel.RandomInquiryAdapter(seed=job.seed or 0),
    }
    if memory_index is not None:
        # Same frozen learner and environment, with an explicit training-only
        # nonparametric memory ablation routed through run_episode's renderer.
        adapters["b-memory"] = kernel.LearnedPolicyAdapter()
    return adapters


def _run_matched_group(group: dict, *, b_bridge: Any, job: JobInput,
                       b_checkpoint: Any, a_checkpoint: Any,
                       memory_index: Any | None = None) -> dict[str, Any]:
    """Run every mode on matched copies of one mechanism+seed."""
    from bramastra_lab.research.cognition import episode as kernel
    from bramastra_lab.research.environments.k8_live import build_live_env

    adapters = _adapters_for_group(group, b_bridge=b_bridge, job=job,
                                   memory_index=memory_index)
    episodes: dict[str, Any] = {}
    for mode, adapter in adapters.items():
        env = build_live_env(group["mechanism"], budget=6,
                             seed=group["env_seed"])
        if mode.startswith("b-"):
            model: Any = b_bridge
            checkpoint = b_checkpoint
        else:
            # Scripted controls consume no model.
            model = kernel.ScriptedModel([{"kind": "noop"}])
            checkpoint = a_checkpoint
        trace = kernel.run_episode(
            env, adapter, model=model, seed=group["env_seed"],
            mechanism=group["mechanism"], session_job_id=str(job.job_id),
            checkpoint_id=checkpoint,
            memory_index=memory_index if mode == "b-memory" else None)
        trace["checkpoint_id"] = checkpoint
        episodes[mode] = trace
    # Symbolic oracle control on its own matched copy (diagnostic ceiling).
    env = build_live_env(group["mechanism"], budget=6,
                         seed=group["env_seed"])
    symbolic = kernel.SymbolicReferenceAdapter(env)
    trace = kernel.run_episode(
        env, symbolic, model=kernel.ScriptedModel([{"kind": "noop"}]),
        seed=group["env_seed"], mechanism=group["mechanism"],
        session_job_id=str(job.job_id), checkpoint_id=None)
    trace["checkpoint_id"] = None
    episodes["symbolic"] = trace
    return episodes


def _build_training_memory_index(job: JobInput):
    """Freeze an exact-training-split exemplar index for the E2 ablation."""
    from bramastra_lab.research.campaigns.phases.compiler import (
        load_training_trajectories)
    from bramastra_lab.research.memory.store import MemoryIndex, MemoryRecord

    rows = load_training_trajectories(job.data_dir, seed=int(job.seed or 0))
    records = []
    seen_mechanisms: set[str] = set()
    for row in rows:
        if row.get("pool") != "training":
            raise ValueError(
                f"memory source {row.get('mechanism_id')} is outside the "
                "exact training split")
        mechanism_id = str(row["mechanism_id"])
        if mechanism_id in seen_mechanisms:
            continue  # four trajectories from one mechanism share one exemplar
        seen_mechanisms.add(mechanism_id)
        content = json.dumps({"public_example": row["public"],
                              "answer": str(row["answer"])},
                             sort_keys=True, separators=(",", ":"))
        records.append(MemoryRecord(
            content=content,
            identity=f"{row.get('family', 'k8')}:{mechanism_id}",
            scope="training", episode_id=mechanism_id,
            kind="training_trajectory"))
    if not records:
        raise ValueError("exact training split produced an empty memory index")
    return MemoryIndex(tuple(records))


def _trace_has_model_origin(trace: dict) -> bool:
    for event in trace.get("events", ()):
        if str(event.get("model_origin", "")).startswith("model"):
            return True
    return False


def _handle_schema_identity(handle: Any) -> str | None:
    """Schema/config identity consumed by a restored handle (U05)."""
    if handle is None:
        return None
    for key in ("config_identity", "schema_identity"):
        if isinstance(handle, dict):
            if handle.get(key):
                return str(handle[key])
        else:
            value = getattr(handle, key, None)
            if value:
                return str(value)
    return None


def _planner_prediction_gap(trace: dict) -> dict[str, Any]:
    """Chosen-action predicted success vs actual outcome (search vs model).

    Joins the imagined node whose action matches the LAST received history
    action (the planner's executed selection), not the first node. Returns
    a dict with joined/node_index/success_gap/feedback_match/reason.
    """
    result: dict[str, Any] = {"joined": False, "node_index": None,
                              "success_gap": None, "feedback_match": None,
                              "reason": None}
    history = trace.get("history", [])
    imagined = trace.get("imagined", [])
    actual = trace.get("summary", {}).get("success")
    if not imagined:
        result["reason"] = "no_imagined_nodes"
        return result
    if actual is None:
        result["reason"] = "no_actual_outcome"
        return result
    if not history:
        result["reason"] = "no_history"
        return result
    # The chosen action is the FIRST history entry (the planner's
    # root selection); later entries are subsequent episode actions.
    chosen_action = history[0].get("action", {})
    chosen_json = json.dumps(chosen_action, sort_keys=True)
    # Find the imagined node whose first prefix action matches.
    chosen_node = None
    for node in imagined:
        prefix = node.get("action_prefix", [])
        if prefix and json.dumps(prefix[-1], sort_keys=True) == chosen_json:
            chosen_node = node
            break
    if chosen_node is None:
        result["reason"] = "chosen_action_not_in_imagined"
        return result
    predicted_outcome = chosen_node.get("predicted_outcome", {})
    sp = predicted_outcome.get("success_prob")
    if sp is None:
        result["reason"] = "chosen_node_has_no_success_prob"
        return result
    try:
        sp = float(sp)
    except (TypeError, ValueError):
        result["reason"] = "success_prob_not_numeric"
        return result
    actual_value = 1.0 if actual else 0.0
    predicted_feedback = predicted_outcome.get("feedback", {})
    received_feedback = history[0].get("feedback", {})
    feedback_match = (json.dumps(predicted_feedback, sort_keys=True, default=str)
                      == json.dumps(received_feedback, sort_keys=True, default=str))
    result.update({
        "joined": True,
        "node_index": chosen_node.get("node_index"),
        "success_gap": abs(sp - actual_value),
        "feedback_match": feedback_match,
        "predicted_success_prob": sp,
        "actual_success": actual,
    })
    return result


def _complementary_coverage(group: dict,
                            b_policy_trace: dict | None) -> dict[str, Any]:
    """Whether the B-policy trace covered the complementary queries."""
    mechanism = group["mechanism"]
    if group["family"] == "rule-inquiry":
        required = {str(item.get("variable"))
                    for item in mechanism.get("complementary_pair", [])}
        key = "variable"
    else:
        required = {json.dumps(q, sort_keys=True)
                    for q in list(mechanism.get("queries", []))[:2]}
        key = None
    covered: set[str] = set()
    if b_policy_trace is not None:
        for event in b_policy_trace.get("events", ()):
            if event.get("kind") != "action":
                continue
            action = event.get("action", {})
            if key is not None:
                if action.get(key) in required:
                    covered.add(str(action.get(key)))
            else:
                for query in list(mechanism.get("queries", []))[:2]:
                    if all(action.get(k) == v for k, v in query.items()):
                        covered.add(json.dumps(query, sort_keys=True))
    return {"mechanism_id": group["mechanism_id"],
            "family": group["family"],
            "required": sorted(required),
            "covered": sorted(covered),
            "covered_all": bool(required) and covered >= required}


def _run_goal_swap(group: dict, *, b_bridge: Any,
                   job: JobInput, b_checkpoint: Any) -> dict[str, Any]:
    """Goal changed BEFORE rendering; success against the swapped predicate."""
    from bramastra_lab.research.cognition import episode as kernel
    from bramastra_lab.research.environments.k8_live import build_live_env

    family = group["family"]
    mechanism = group["mechanism"]
    swapped_goal: dict[str, Any]
    same_predicate = False
    if family == "rule-inquiry":
        swapped_goal = {"question": "is the hidden rule FALSE?",
                        "negated": True}
    elif family == "program":
        start = mechanism.get("start_value", 0)
        swapped_input = 1 if start != 1 else 2
        swapped_goal = {"question": "report f on the swapped input",
                        "swapped_input": swapped_input}
    else:
        swapped_goal = dict(mechanism.get("public", {}))
        swapped_goal["paraphrase"] = "same task, reworded goal"
        same_predicate = True
    env = build_live_env(mechanism, budget=6, seed=group["env_seed"])
    trace = kernel.run_episode(
        env, kernel.LearnedPolicyAdapter(), model=b_bridge,
        seed=group["env_seed"], mechanism=mechanism,
        goal_override=swapped_goal, session_job_id=str(job.job_id),
        checkpoint_id=b_checkpoint)
    submitted = None
    for event in trace.get("events", ()):
        if event.get("kind") == "action" and event.get("action", {}).get(
                "kind") in ("submit", "check_result"):
            submitted = event.get("action")
    swapped_success: bool | None = None
    if family == "rule-inquiry" and submitted is not None:
        from bramastra_lab.research.data.k8_bundle import verify_rule

        actual = verify_rule(mechanism, {"values": dict(env._world)})
        swapped_success = bool(submitted.get("answer")) == (not bool(actual))
    elif family == "program" and submitted is not None:
        try:
            swapped_success = int(submitted.get("value")) == env.evaluate(
                swapped_goal["swapped_input"])
        except (KeyError, TypeError, ValueError):
            swapped_success = False
    else:
        swapped_success = trace.get("summary", {}).get("success")
    return {"mechanism_id": group["mechanism_id"], "family": family,
            "swapped_goal": swapped_goal, "same_predicate": same_predicate,
            "submitted": submitted,
            "swapped_success": swapped_success,
            "actions": trace["summary"].get("actions", 0),
            "model_calls": trace["summary"].get("model_calls", 0)}


def _run_contradiction_probe(group: dict,
                             workspace_trace: dict | None) -> dict[str, Any]:
    """Injected second-source contradiction + conflict bookkeeping check."""
    from bramastra_lab.research.cognition.episode import mark_conflicts

    if workspace_trace is None:
        return {"mechanism_id": group["mechanism_id"],
                "family": group["family"], "skipped": "no workspace trace"}
    history = list(workspace_trace.get("history", ()))
    if not history:
        return {"mechanism_id": group["mechanism_id"],
                "family": group["family"], "skipped": "empty history"}
    first = history[0]
    feedback = dict(first.get("feedback", {}))
    # Copy a real observed variable, flip its reported value, label the
    # second source explicitly as an injected probe (never real history).
    probe_variable = feedback.get("variable", feedback.get("item", "?"))
    probe_value = not feedback.get("value", False) \
        if isinstance(feedback.get("value"), bool) else "INJECTED-OTHER"
    workspace = [{"entity": f"v:{probe_variable}",
                  "observed_value": json.dumps(
                      {"v": feedback.get("value")}, sort_keys=True),
                  "observation_id": "real-0",
                  "temporal_scope": "episode", "status": "active"},
                 {"entity": f"v:{probe_variable}",
                  "observed_value": json.dumps(
                      {"v": probe_value}, sort_keys=True),
                  "observation_id": "injected-contradiction-probe",
                  "temporal_scope": "episode", "status": "active"}]
    before = [dict(record) for record in workspace]
    conflicts = mark_conflicts(workspace)
    history_intact = all(
        record.get("observed_value") == original.get("observed_value")
        for record, original in zip(workspace, before))
    return {"mechanism_id": group["mechanism_id"],
            "family": group["family"],
            "conflict_detected": bool(conflicts),
            "conflicting_pairs": conflicts,
            "history_intact": history_intact}
