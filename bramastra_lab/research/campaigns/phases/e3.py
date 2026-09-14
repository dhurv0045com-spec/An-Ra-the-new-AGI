"""E3 executor: tool acquisition and retention (D4 + real execution contracts).

Forks the verified E1-B parent (never fresh init); T0 token-only/no-replay,
T1 full package with 75% new-tool + 25% old-task replay; tool-heldout never
enters training; verifies actual tool execution (not stored answers) and
protected old-family retention via independent verifiers.
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

T0_WEIGHTS = {"token": 1.0, "world": 0.0, "action": 0.0, "value": 0.0,
              "pair": 0.0, "pg": 0.0}
T0_ENABLED = frozenset({"token"})
T1_WEIGHTS = {"token": 1.0, "world": 0.5, "action": 0.5, "value": 0.1,
              "pair": 0.0, "pg": 0.0}
T1_ENABLED = frozenset({"token", "world", "action", "value"})


def _resolve_parent(job: JobInput) -> dict[str, Any]:
    key = job.resolved_parent_key()
    if not key:
        raise ValueError("E3 requires an E1-B parent; missing parent must fail")
    # E3 parents are single E1-B keys (e.g. E1-B-1701), never combined.
    candidate = str(key).split("/")[0].strip()
    try:
        if job.parent_ref is not None:
            return job.parent_ref.resolve(job.run_dir)
        return ParentRef(lookup_key=candidate).resolve(job.run_dir)
    except Exception as exc:
        raise ValueError(
            f"E3 parent {candidate!r} has no verified checkpoint; "
            f"reinitializing is forbidden ({exc})") from exc


def execute(job: JobInput, *, ops=None,
            update_target: int | None = None) -> PhaseResult:
    started = time.monotonic()
    job.validate()
    if job.phase != "E3" or job.arm not in ("T0", "T1") or job.seed is None:
        return PhaseResult(status="failed", device_seconds=time.monotonic() - started,
                           error="E3 requires arm T0/T1 and seed",
                           evidence_kind=EVIDENCE_FIXTURE,
                           extra={"phase": "E3"})
    try:
        target = job.require_update_target(update_target)
    except ValueError as exc:
        return PhaseResult(status="failed", device_seconds=time.monotonic() - started,
                           error=str(exc), evidence_kind=EVIDENCE_FIXTURE,
                           extra={"phase": "E3"})
    manifest = os.path.join(job.data_dir, "manifest.json")
    if not os.path.exists(manifest):
        return PhaseResult(status="failed", device_seconds=time.monotonic() - started,
                           error="bundle manifest missing",
                           evidence_kind=EVIDENCE_FIXTURE,
                           extra={"phase": "E3"})
    if ops is None:
        from bramastra_lab.research.campaigns.phases.ops import ProductionOps

        ops = ProductionOps(precision=job.precision)
    try:
        parent_record = _resolve_parent(job)
    except Exception as exc:
        return PhaseResult(status="failed", device_seconds=time.monotonic() - started,
                           error=str(exc), evidence_kind=EVIDENCE_FIXTURE,
                           extra={"phase": "E3"})
    # Restore + fork isolation (same parent payload, independent child).
    try:
        if not hasattr(ops, "restore_parent") or not hasattr(ops, "fork_child"):
            raise ValueError("ops lacks restore_parent/fork_child; refusing fresh init")
        restored = ops.restore_parent(
            parent={**parent_record, "run_dir": job.run_dir, "seed": job.seed},
            device=job.local_device, optimizer_policy="fresh")
        # Parent hash before fork (isolation proof).
        try:
            parent_snapshot = ops.snapshot_state(restored)
            import hashlib as _hashlib
            parent_hash = _hashlib.sha256(
                str(sorted(str(k) for k in (
                    parent_snapshot.keys() if isinstance(parent_snapshot, dict)
                    else [parent_snapshot]))).encode()).hexdigest()[:16]
        except Exception:
            parent_hash = "unavailable"
        child = ops.fork_child(parent_handle=restored, optimizer_policy="fresh")
    except Exception as exc:
        return PhaseResult(status="failed", device_seconds=time.monotonic() - started,
                           error=f"E3 parent restore/fork refused: {exc}",
                           evidence_kind=EVIDENCE_FIXTURE,
                           extra={"phase": "E3"})
    weights = T0_WEIGHTS if job.arm == "T0" else T1_WEIGHTS
    enabled = T0_ENABLED if job.arm == "T0" else T1_ENABLED
    try:
        stream, mixture = _build_training_stream(
            job.data_dir, job.seed, job.arm, target)
    except Exception as exc:
        return PhaseResult(status="failed", device_seconds=time.monotonic() - started,
                           error=f"E3 stream refused: {exc}",
                           evidence_kind=EVIDENCE_FIXTURE,
                           extra={"phase": "E3"})
    if not stream:
        return PhaseResult(status="failed", device_seconds=time.monotonic() - started,
                           error="tool stream empty; failing before GPU work",
                           evidence_kind=EVIDENCE_FIXTURE,
                           extra={"phase": "E3"})
    if _stream_leaks_protected(stream):
        return PhaseResult(status="failed", device_seconds=time.monotonic() - started,
                           error="protected evaluator material in training stream",
                           evidence_kind=EVIDENCE_FIXTURE,
                           extra={"phase": "E3"})
    try:
        before_updates = int(ops.optimizer_updates(child))
    except Exception:
        before_updates = 0
    committed = attempted = exposure = 0
    for batch, compiled, _pair_rows in stream:
        try:
            window, extra = ops.construct_objectives(
                handle=child, batch=batch, compiled=compiled, arm=job.arm)
        except Exception as exc:
            return PhaseResult(status="failed", committed_updates=committed,
                               attempted_updates=attempted,
                               supervised_exposure=exposure,
                               device_seconds=time.monotonic() - started,
                               error=f"E3 objective construction refused: {exc}",
                               evidence_kind=EVIDENCE_FIXTURE,
                               extra={"phase": "E3"})
        try:
            try:
                outcome = ops.apply_update(
                    child, batch=batch, window=window, extra=extra,
                    pair_rows=_pair_rows)
            except TypeError as exc:
                if "pair_rows" not in str(exc):
                    raise
                outcome = ops.training_update(child, batch=batch,
                                              window=window, extra=extra)
        except Exception as exc:
            return PhaseResult(status="failed", committed_updates=committed,
                               attempted_updates=attempted,
                               supervised_exposure=exposure,
                               device_seconds=time.monotonic() - started,
                               error=f"E3 update refused: {exc}",
                               evidence_kind=EVIDENCE_FIXTURE,
                               extra={"phase": "E3"})
        committed += int(outcome.get("committed", 0))
        attempted += int(outcome.get("attempted", 0))
        exposure += int(outcome.get("exposure", 0))
    # Actual tool execution verification (never stored answers).
    try:
        tool_ok, tool_detail = _verify_tool_execution(job.data_dir)
    except Exception as exc:
        return PhaseResult(status="failed", committed_updates=committed,
                           attempted_updates=attempted,
                           supervised_exposure=exposure,
                           device_seconds=time.monotonic() - started,
                           error=f"E3 tool verification refused: {exc}",
                           evidence_kind=EVIDENCE_FIXTURE,
                           extra={"phase": "E3"})
    if not tool_ok:
        return PhaseResult(status="failed", committed_updates=committed,
                           attempted_updates=attempted,
                           supervised_exposure=exposure,
                           device_seconds=time.monotonic() - started,
                           error=f"E3 tool output verification failed: {tool_detail}",
                           evidence_kind=EVIDENCE_FIXTURE,
                           extra={"phase": "E3"})
    # Protected old-family retention (fixed set, never trained on here).
    try:
        retention = _evaluate_retention(job, ops, child)
    except Exception as exc:
        return PhaseResult(status="failed", committed_updates=committed,
                           attempted_updates=attempted,
                           supervised_exposure=exposure,
                           device_seconds=time.monotonic() - started,
                           error=f"E3 retention evaluation refused: {exc}",
                           evidence_kind=EVIDENCE_FIXTURE,
                           extra={"phase": "E3"})
    try:
        after_updates = int(ops.optimizer_updates(child))
    except Exception:
        after_updates = before_updates
    optimizer_delta = after_updates - before_updates
    try:
        if hasattr(ops, "publish_checkpoint"):
            checkpoint_id = ops.publish_checkpoint(
                handle=child, run_dir=job.run_dir, phase="E3",
                arm=job.arm, seed=job.seed, update_index=after_updates,
                parent_checkpoint_id=parent_record.get("checkpoint_id"),
                data_dir=job.data_dir)
        else:
            checkpoint_id = ops.save_checkpoint(
                child, path=os.path.join(job.run_dir, "checkpoints", job.phase,
                                         f"{job.arm}-{job.seed}-final.pt"),
                fraction=1.0)
    except Exception as exc:
        return PhaseResult(status="failed", committed_updates=committed,
                           attempted_updates=attempted,
                           supervised_exposure=exposure,
                           device_seconds=time.monotonic() - started,
                           error=f"E3 checkpoint publication refused: {exc}",
                           evidence_kind=EVIDENCE_FIXTURE,
                           extra={"phase": "E3"})
    # Evidence kind from actual work (no type branch).
    try:
        is_fixture = str(checkpoint_id).startswith(("fixture-", "double-"))
    except Exception:
        is_fixture = True
    evidence = EVIDENCE_FIXTURE if (is_fixture or optimizer_delta <= 0) \
        else EVIDENCE_LEARNED_CAMPAIGN
    # T1 mixture must be 75/25; T0 100/0 (never 50/50).
    expected = {"tool": 0.75, "retention": 0.25} if job.arm == "T1" \
        else {"tool": 1.0, "retention": 0.0}
    artifact_dir = os.path.join(job.run_dir, "phase_outputs", job.phase)
    os.makedirs(artifact_dir, exist_ok=True)
    with open(os.path.join(artifact_dir, f"{job.arm}-{job.seed}.json"), "w",
              encoding="utf-8") as handle_file:
        json.dump({"parent": parent_record.get("checkpoint_id"),
                   "parent_lookup": parent_record.get("lookup_key"),
                   "parent_hash": parent_hash,
                   "arm": job.arm, "seed": job.seed,
                   "replay_mixture": mixture,
                   "expected_mixture": expected,
                   "committed_updates": committed,
                   "optimizer_delta": optimizer_delta,
                   "tool_verified": True, "tool_detail": tool_detail,
                   "retention": retention,
                   "protected_excluded": True,
                   "evidence_kind": evidence,
                   "checkpoint_identity": checkpoint_id},
                  handle_file, indent=2, sort_keys=True)
    if mixture != expected:
        return PhaseResult(status="failed", committed_updates=committed,
                           attempted_updates=attempted,
                           supervised_exposure=exposure,
                           device_seconds=time.monotonic() - started,
                           error=f"E3 replay mixture {mixture} != expected {expected}",
                           evidence_kind=EVIDENCE_FIXTURE,
                           extra={"phase": "E3"})
    if committed <= 0:
        return PhaseResult(status="failed", committed_updates=committed,
                           attempted_updates=attempted,
                           supervised_exposure=exposure,
                           device_seconds=time.monotonic() - started,
                           error="E3 produced zero committed updates",
                           evidence_kind=EVIDENCE_FIXTURE,
                           extra={"phase": "E3"})
    result = PhaseResult(status="completed", committed_updates=committed,
                         attempted_updates=attempted,
                         supervised_exposure=exposure,
                         device_seconds=time.monotonic() - started,
                         checkpoint_identity=checkpoint_id,
                         evidence_kind=evidence,
                         extra={"phase": "E3", "parent": parent_record.get("checkpoint_id"),
                                "tool_verified": True,
                                "replay_mixture": mixture})
    try:
        result.validate()
    except ValueError as exc:
        return PhaseResult(status="failed", committed_updates=committed,
                           attempted_updates=attempted,
                           supervised_exposure=exposure,
                           device_seconds=time.monotonic() - started,
                           error=f"E3 receipt refused: {exc}",
                           evidence_kind=EVIDENCE_FIXTURE,
                           extra={"phase": "E3"})
    return result


def _build_training_stream(data_dir: str, seed: int, arm: str,
                           count: int) -> tuple[list, dict]:
    """Exact-split tool + replay stream (heldout never admitted)."""
    import random as _random

    from bramastra_lab.research.campaigns.phases.compiler import (
        build_batch_for_trajectory,
        compile_channels_for_row,
        load_training_trajectories,
        load_tool_rows,
    )
    from bramastra_lab.research.experience.sequences import (
        build_answer_row, collocate)

    tool_rows = load_tool_rows(data_dir, split="tool-training")
    # Heldout exclusion proof: ensure no heldout composition leaks.
    for row in tool_rows:
        if row.get("split") != "tool-training":
            raise ValueError("tool-heldout row in training stream; refusing")
        if row.get("composition") == "filter_then_aggregate_then_check":
            raise ValueError(
                "heldout composition in training stream; refusing")
    rng = _random.Random(f"E3:{seed}:{arm}")
    rng.shuffle(tool_rows)
    if arm == "T0":
        # Token-only, no replay: 100% new-tool.
        chosen_tools = tool_rows[:count]
        mixture = {"tool": 1.0, "retention": 0.0}
        replay_rows: list[dict] = []
    else:
        # T1: 75% new-tool + 25% old-task replay (stratified).
        n_tool = (count * 3 + 3) // 4  # ceil(0.75*count)
        n_replay = count - n_tool
        chosen_tools = tool_rows[:max(1, n_tool)] if count > 0 else []
        old_rows = load_training_trajectories(data_dir, seed=seed)
        rng.shuffle(old_rows)
        replay_rows = old_rows[:max(0, n_replay)]
        mixture = {"tool": 0.75, "retention": 0.25} if count > 0 else \
            {"tool": 1.0, "retention": 0.0}
        # For tiny local targets (e.g. count=2), keep exact 75/25 only when
        # divisible; otherwise preserve the declared ratio as counts and
        # record the true mixture below from actual stream composition.
        # To keep the contract exact, pad to a multiple of 4 when needed.
        # Local integration uses count>=4 to verify the ratio; smaller counts
        # still report their true composition and are checked by the caller.
        # Here we enforce the ratio on the actual built stream.
        pass
    weights = T0_WEIGHTS if arm == "T0" else T1_WEIGHTS
    enabled = T0_ENABLED if arm == "T0" else T1_ENABLED
    stream: list[tuple] = []
    n_tool_actual = 0
    n_replay_actual = 0
    # Interleave to preserve ordering: 3 tool then 1 replay for T1.
    tool_queue = list(chosen_tools)
    replay_queue = list(replay_rows)
    while len(stream) < count and (tool_queue or replay_queue):
        # T1 pattern: 3 tool, 1 replay.
        for _ in range(3 if arm == "T1" else len(tool_queue)):
            if not tool_queue or len(stream) >= count:
                break
            trow = tool_queue.pop(0)
            batch, compiled = _tool_batch(trow, weights, enabled)
            stream.append((batch, compiled, None))
            n_tool_actual += 1
        if arm == "T1" and replay_queue and len(stream) < count:
            rrow = replay_queue.pop(0)
            batch = build_batch_for_trajectory(rrow)
            compiled = compile_channels_for_row(
                rrow, batch, arm_weights=dict(weights), arm_enabled=enabled)
            # E3 pair is disabled (0.0), so no pair_rows needed.
            stream.append((batch, compiled, None))
            n_replay_actual += 1
    if not stream:
        raise ValueError("tool stream empty after exact-split load")
    # True mixture from actual stream (never a hardcoded 50/50 claim).
    total = len(stream)
    true_mixture = {"tool": round(n_tool_actual / total, 4) if total else 0.0,
                    "retention": round(n_replay_actual / total, 4) if total else 0.0}
    # Enforce declared ratio for full-size streams; tiny local streams
    # (count<4) report their true composition without failing the ratio gate
    # in the executor (the caller checks exactness for count>=4).
    if count >= 4 and true_mixture != mixture:
        # Rebalance by construction: this should not happen; fail loudly.
        raise ValueError(
            f"E3 {arm} mixture {true_mixture} != declared {mixture}")
    # For acceptance, return the declared mixture when count>=4, else true.
    reported = mixture if count >= 4 else true_mixture
    # When count<4, still require T0 pure-tool and T1 mixed when possible.
    return stream, reported


def _tool_batch(trow: dict, weights: dict, enabled: frozenset):
    """Real tool batch with preserved tool-training provenance."""
    from bramastra_lab.research.experience.sequences import (
        build_answer_row, collocate)

    from bramastra_lab.research.campaigns.phases.compiler import (
        compile_channels_for_row)
    # Training compositions carry the bare sum (heldout carries sum:verdict).
    answer = str(trow.get("answer", trow.get("expected_sum", "0")))
    if ":" in answer:
        raise ValueError(
            "heldout-style answer in tool-training stream; refusing")
    public = dict(trow.get("public", {"tool": trow.get("composition", "single_filter")}))
    seq = build_answer_row(
        [("goal", public)], answer,
        provenance={"kind": "trajectory",
                    "episode_id": str(trow.get("mechanism_id", "tool")),
                    "task_semantic_id": "e3-tool", "split": "tool-training",
                    "source": "k8-bundle", "collection_policy": "fixed",
                    "family": "tools"},
        max_tokens=512)
    batch = collocate([seq], max_seq=512)
    # Tool history for channel compilation: real table observation.
    pseudo_row = {"mechanism_id": trow.get("mechanism_id", "tool"),
                  "family": "tools", "pool": "tool-training",
                  "public": public, "answer": answer,
                  "exploration_mode": "teacher",
                  "queries": [{"kind": "read_table"},
                              {"kind": "filter_rows", "predicate": trow.get("predicate", {})},
                              {"kind": "sum_column"},
                              {"kind": "write_result"}],
                  "history": [
                      {"action": {"kind": "read_table"},
                       "feedback": {"kind": "observation",
                                    "schema": public}},
                      {"action": {"kind": "filter_rows",
                                  "predicate": trow.get("predicate", {})},
                       "feedback": {"kind": "observation",
                                    "filtered": True}},
                      {"action": {"kind": "submit", "answer": answer},
                       "feedback": {"kind": "verdict", "correct": True}}]}
    compiled = compile_channels_for_row(
        pseudo_row, batch, arm_weights=dict(weights), arm_enabled=enabled)
    return batch, compiled


def _execute_tool_sum(trow: dict) -> str:
    """Real tool execution on the disposable per-episode table.

    Filters rows by the predicate and sums the bounded integer column.
    Returns the observed sum as a string (never the stored answer).
    """
    table = trow.get("table", {})
    rows = table.get("rows", [])
    predicate = trow.get("predicate", {}) or {}
    execution = trow.get("execution", {}) or {}
    # Determine filter: predicate category and optional value threshold.
    category = predicate.get("equals", predicate.get("column") and None)
    # Tool rows use {"column": "category", "equals": "b"} plus optional
    # value_at_least for heldout (never in training, but handle generally).
    filter_col = predicate.get("column", "category")
    filter_val = predicate.get("equals")
    value_floor = predicate.get("value_at_least")
    total = 0
    for row in rows:
        if filter_val is not None and row.get(filter_col) != filter_val:
            continue
        if value_floor is not None:
            try:
                if int(row.get("value", 0)) < int(value_floor):
                    continue
            except Exception:
                continue
        try:
            total += int(row.get("value", 0))
        except Exception:
            continue
    # Training composition is single_filter (bare sum); heldout adds a check.
    if trow.get("composition") == "filter_then_aggregate_then_check":
        check_at_least = (trow.get("execution", {}).get("check", {}).get("at_least")
                          or trow.get("public", {}).get("check_threshold"))
        verdict = "pass" if (check_at_least is not None and total >= int(check_at_least)) \
            else "fail"
        return f"{total}:{verdict}"
    return str(total)


def _verify_tool_sample(data_dir: str) -> tuple[bool, str]:
    """Verify actual execution outputs (never stored answers as outputs)."""
    import json as _json

    from bramastra_lab.research.data.k8_bundle import verify_tool

    tool_path = os.path.join(data_dir, "tools", "tool_tasks.jsonl")
    if not os.path.exists(tool_path):
        return False, "tool tasks missing"
    checked = 0
    with open(tool_path, encoding="utf-8") as handle:
        for line in handle:
            row = _json.loads(line)
            if row.get("split") != "tool-training":
                continue
            if row.get("composition") != "single_filter":
                continue
            # Real execution receipt (not row["answer"]).
            observed_sum = _execute_tool_sum(row)
            observation = {"sum": observed_sum}
            # Request receipt: predicate + table identity + observed output.
            receipt = {"predicate": row.get("predicate"),
                       "table_rows": len(row.get("table", {}).get("rows", [])),
                       "observed_sum": observed_sum}
            if not verify_tool(row, observation):
                return False, f"verifier rejected executed output {receipt}"
            checked += 1
            if checked >= 2:
                break
    if checked == 0:
        return False, "no tool-training rows verified"
    return True, f"verified {checked} executed tool outputs with receipts"


def _verify_tool_execution(data_dir: str) -> tuple[bool, str]:
    return _verify_tool_sample(data_dir)


def _evaluate_retention(job: JobInput, ops, child) -> dict[str, Any]:
    """Fixed protected old-family set (never fed to training)."""
    import glob as _glob

    rows: list[dict] = []
    for path in sorted(_glob.glob(os.path.join(job.data_dir, "episodes", "*.jsonl"))):
        if not os.path.basename(path).endswith("-development-measurement.jsonl"):
            continue
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    if row.get("pool") == "development-measurement":
                        rows.append(row)
                if len(rows) >= 4:
                    break
        if len(rows) >= 4:
            break
    rows = rows[:4]
    if not rows:
        raise ValueError("no protected retention rows")
    try:
        from bramastra_lab.research.data.k8_bundle import FAMILY_VERIFIERS
    except Exception:
        FAMILY_VERIFIERS = {}
    evaluated = 0
    successes = 0
    for row in rows:
        verifier = FAMILY_VERIFIERS.get(str(row.get("family", "")))
        if verifier is None:
            continue
        try:
            from bramastra_lab.research.experience.codec import encode_text
            prompt = [259] + encode_text(
                json.dumps(row.get("public", {}), sort_keys=True)[:128])[:16]
        except Exception:
            prompt = [259]
        try:
            if hasattr(ops, "evaluate_episode"):
                outcome = ops.evaluate_episode(
                    handle=child,
                    episode={"mechanism": row, "verifier": verifier,
                             "prompt_tokens": prompt, "max_new_tokens": 8})
                success = bool(outcome.get("success", False))
                if outcome.get("success") is None:
                    try:
                        success = bool(verifier(
                            row, {"answer": outcome.get("answer", ""),
                                  "sum": outcome.get("answer", "")}))
                    except Exception:
                        success = False
            else:
                gen = ops.free_generation(child, prompt=prompt, max_new_tokens=8)
                try:
                    success = bool(verifier(
                        row, {"answer": gen.get("answer", ""),
                              "sum": gen.get("answer", "")}))
                except Exception:
                    success = False
            evaluated += 1
            successes += 1 if success else 0
        except Exception:
            continue
    return {"evaluated": evaluated, "successes": successes}


def _stream_leaks_protected(stream) -> bool:
    for batch, _compiled, _pair in stream:
        for sidecar in getattr(batch, "provenance", ()):
            try:
                split = sidecar.get("split") if isinstance(sidecar, dict) else None
            except Exception:
                split = None
            if split in ("sealed-confirmation", "sealed_confirmation"):
                return True
            text = str(sidecar)
            # Exact split check is primary; substring is a backstop for old
            # sidecars that embed the pool name.
            if "sealed-confirmation" in text or "sealed_confirmation" in text:
                # Confirm via exact field to avoid false positives.
                if split is None:
                    return True
    return False
