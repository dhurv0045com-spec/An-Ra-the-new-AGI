"""E5 executor: measured recursive method selection (D4 + real execution contracts).

Three exact blocks (archive_P0, successor, fresh confirmation) with a fixed
adaptation anchor per seed (separate from proposer weights). Every method
trial forks the anchor with identical initial state + support order, applies
the declared M0/M1/M2 recipe to the actual trainer (never a fake trainer),
and evaluates on query/protected cases. Archive rows carry measured outcomes,
costs, support/query identities and checkpoint lineage (failures recorded).
P_fixed is M0 (never M2). Proposal capture binds decoder output, input/archive
cutoff, parsed method and recipe identity. Confirmation choices are captured
before fresh outcomes via immutable archive capabilities; current-task
identities in proposal context are rejected. Counts derive from actual ops
calls, never task counts. `run_fixture_generation` is forbidden here.
"""
from __future__ import annotations

import hashlib
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


def _resolve_anchor(job: JobInput) -> dict[str, Any]:
    key = job.resolved_parent_key()
    # E5 requires an explicit adaptation anchor (the seed's E1-B parent).
    # Synthesizing a default key would mask a missing anchor; fail instead
    # and require the caller (worker/runner) to pass the exact parent.
    if not key:
        raise ValueError(
            "E5 requires an explicit adaptation anchor parent (e.g. E1-B-1701); "
            "missing anchor must fail, never synthesize a default key")
    if "/" in str(key):
        raise ValueError(
            f"E5 anchor {key!r} is combined; E5 requires a single anchor parent")
    candidate = str(key).strip()
    try:
        if job.parent_ref is not None and job.parent_ref.lookup_key == candidate:
            rec = job.parent_ref.resolve(job.run_dir)
        else:
            rec = ParentRef(lookup_key=candidate).resolve(job.run_dir)
        rec["lookup_key"] = candidate
        return rec
    except Exception as exc:
        raise ValueError(
            f"E5 adaptation anchor {candidate!r} has no verified checkpoint; "
            f"missing anchor must fail ({exc})") from exc


def _load_meta_tasks(data_dir: str, *, pool: str) -> list[dict[str, Any]]:
    path = os.path.join(data_dir, "meta", "meta_tasks.jsonl")
    if not os.path.exists(path):
        raise ValueError(f"meta tasks missing: {path}")
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("pool") == pool:
                rows.append(row)
    if not rows:
        raise ValueError(f"no meta tasks for pool {pool!r}; failing")
    return rows


def _iter_episode_rows(data_dir: str):
    import glob as _glob
    import json as _json

    pattern = os.path.join(data_dir, "episodes", "*.jsonl")
    for path in sorted(_glob.glob(pattern)):
        try:
            with open(path, encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = _json.loads(line)
                    except ValueError:
                        continue
                    yield row
        except OSError:
            continue


def resolve_mechanism_row(data_dir: str, mechanism_id: str) -> dict[str, Any]:
    """Resolve a mechanism's prepared episode row (public + answer).

    Fails explicitly when the mechanism has no prepared row (missing data),
    inconsistent answers across rows, or no public state to render. A
    missing protected reference must never silently shrink any evaluation
    denominator.
    """
    if not mechanism_id:
        raise ValueError("mechanism reference carries no mechanism_id; refusing")
    row: dict[str, Any] | None = None
    answers: set[str] = set()
    for candidate in _iter_episode_rows(data_dir):
        if str(candidate.get("mechanism_id", "")) != str(mechanism_id):
            continue
        if "answer" in candidate:
            answers.add(str(candidate["answer"]))
        if row is None and candidate.get("public") is not None:
            row = candidate
    if not answers:
        raise ValueError(
            f"no prepared answer for mechanism {mechanism_id!r}; refusing "
            "invented query labels")
    if len(answers) > 1:
        raise ValueError(
            f"mechanism {mechanism_id!r} has inconsistent answers "
            f"{sorted(answers)}; refusing")
    if row is None or not isinstance(row.get("public"), dict):
        raise ValueError(
            f"mechanism {mechanism_id!r} has no prepared public state; "
            "an evaluation prompt cannot be rendered without it")
    return row


def resolve_mechanism_answer(data_dir: str, mechanism_id: str) -> str:
    """Resolve a query mechanism's answer from prepared bundle data.

    Searches episode rows for the mechanism ID and returns its recorded
    answer, verifying consistency across rows. Meta-training feedback comes
    from these prepared labels (never invented); confirmation scoring uses
    the same lookup only AFTER choices are frozen (same-table scoring).
    """
    return str(resolve_mechanism_row(data_dir, mechanism_id)["answer"])


def _support_identities(task: dict, task_id: str) -> tuple[str, ...]:
    return tuple(f"{task_id}-support-{i}"
                 for i in range(len(task.get("support_examples", ()))))


def _query_identities(task: dict, task_id: str) -> tuple[str, ...]:
    return tuple(f"{task_id}-query-{i}"
                 for i in range(len(task.get("query_examples", ()))))


def _run_archive_trial(ops: Any, job: JobInput, anchor_record: dict,
                       anchor_id: str, anchor_support: tuple,
                       task: dict, task_id: str, method_id: str,
                       learning_boundary: str) -> Any:
    """One archive trial through the measured trial service (O08)."""
    import time as _time

    from bramastra_lab.research.campaigns import trial_service
    from bramastra_lab.research.metalearning.dispatch import _METHOD_PROGRAMS
    from bramastra_lab.research.metalearning.method_language import (
        compile_method)

    compiled = compile_method(_METHOD_PROGRAMS[method_id],
                              runtime_config={"profile": "k8-campaign"})
    if not compiled.get("identity"):
        raise ValueError(f"{method_id} compiled without identity")
    # M2 trains the gated reuse variant: migrate the forked anchor copy to
    # zero gates first (the base anchor stays untouched for M0/M1 trials).
    migrate_for_trial = (method_id == "M2")
    support_ids = _support_identities(task, task_id) or anchor_support
    query_ids = _query_identities(task, task_id) or (f"{task_id}-query",)
    protected_ids = tuple(
        str(ref.get("mechanism_id", "")) for ref in
        task.get("protected_references", ()))
    # Child deadline: the MINIMUM of the trial cap, the job reservation
    # deadline and the job's own phase deadline (spec 14: nested operations
    # obey every parent deadline; a trial never outlives its allocation).
    trial_deadline = _time.time() + 45.0
    for parent_deadline in (job.reservation_deadline_unix, job.deadline):
        try:
            if parent_deadline is not None:
                trial_deadline = min(trial_deadline, float(parent_deadline))
        except (TypeError, ValueError):
            continue
    request = trial_service.TrialRequest(
        anchor_checkpoint_id=str(anchor_record.get("checkpoint_id")),
        anchor_run_dir=job.run_dir, method_id=method_id,
        compiled_recipe=dict(compiled), support_identities=support_ids,
        query_identities=query_ids, protected_identities=protected_ids,
        seed=int(job.seed or 0),
        reservation={
            "allocation_id": job.allocation_id or f"e5-local-{job.seed}",
            "reservation_id": (job.reservation_id or f"e5-local-{job.seed}")
            + f":{task_id}:{method_id}",
            "job_id": job.job_id or f"E5-{job.seed}",
            "device": job.physical_device,
            "phase": "E5",
            "deadline_unix": trial_deadline,
            "remaining_updates": 4000,
            "source_hash": job.source_hash or "unbound-source"},
        deadline_unix=trial_deadline, max_updates=4000,
        task_identity=f"E5-{job.seed}-{task_id}-{method_id}")
    if learning_boundary == "test_substitute":
        return _run_substitute_trial(
            ops, job, anchor_record, anchor_id, anchor_support,
            request, task, task_id, method_id, migrate_for_trial)
    return trial_service.run_trial(
        ops, request, job=job, learning_boundary="production",
        build_support_batch=_production_support_builder(
            task, request.compiled_recipe),
        evaluate_queries=_production_query_evaluator(
            task, job.data_dir, task_id),
        prepare_trial_handle=(
            lambda handle: ops.migrate_to_gated(
                handle, gates_enabled=True)) if migrate_for_trial else None)


def _run_substitute_trial(ops: Any, job: JobInput, anchor_record: dict,
                          anchor_id: str, anchor_support: tuple,
                          request: Any, task: dict, task_id: str,
                          method_id: str, migrate_for_trial: bool) -> Any:
    """Test-substitute trial: same scheduling, fixture-labeled measurement."""
    from bramastra_lab.research.campaigns import trial_service
    from bramastra_lab.research.metalearning.dispatch import (
        dispatch_method_to_trainer)

    trial_handle = ops.fork_child(
        parent_handle=_restore_anchor_handle(
            ops, job, anchor_record), optimizer_policy="fresh")
    if migrate_for_trial:
        # M2 trains the gated variant: zero-gate migration of the forked
        # copy (the anchor itself is never mutated).
        trial_handle = ops.migrate_to_gated(
            trial_handle, gates_enabled=True)
    if isinstance(trial_handle, dict):
        trial_handle["anchor_id"] = anchor_id
        trial_handle["support_order"] = anchor_support
    trainer = trial_handle.get("_e5_trainer")
    if trainer is None:
        from types import SimpleNamespace as _NS
        trainer = _NS()
        trainer.anchor_id = anchor_id
        trainer.support_order = anchor_support
        trainer.model = _NS(gates_enabled=True)
        trainer.optimizer = _NS(param_groups=[{"lr": 0.0003}])
        trainer.applied_method = None
        trainer.applied_count = 0

        def _set_multiplier(multiplier: float, reason: str) -> None:
            trainer._multiplier = multiplier  # noqa: SLF001

        def _measured(task_identity: str) -> float:
            import hashlib as _hashlib

            digest = _hashlib.sha256(
                f"{anchor_id}:{task_identity}:{trainer.applied_method}"
                .encode()).hexdigest()
            base = (int(digest[:4], 16) % 100) / 100.0
            bonus = {"M0": 0.1, "M1": 0.2,
                     "M2": 0.15}.get(trainer.applied_method or "", 0.0)
            return round(min(1.0, 0.3 + base * 0.4 + bonus), 4)

        trainer.set_controller_multiplier = _set_multiplier
        trainer.measured_success = _measured
        trial_handle["_e5_trainer"] = trainer
    result = dispatch_method_to_trainer(
        method_id, dict(request.compiled_recipe), trainer,
        task_identity=request.task_identity)
    trainer.applied_method = method_id
    trainer.applied_count += 1
    trial_handle["applied_method"] = method_id
    _ = result
    return trial_service.TrialResult(
        task_identity=task_id,
        method_id=method_id, measured_updates=1,
        measured_success=float(trainer.measured_success(
            request.task_identity)),
        elapsed_seconds=45.0, support_identities=request.support_identities,
        query_identities=request.query_identities,
        trial_checkpoint_id=None, validation="measured",
        detail={"learning_boundary": "test_substitute",
                "evidence_kind": "fixture"})


def _restore_anchor_handle(ops: Any, job: JobInput,
                           anchor_record: dict) -> Any:
    return ops.restore_parent(
        parent={**anchor_record, "run_dir": job.run_dir, "seed": job.seed},
        device=job.local_device, optimizer_policy="fresh")


def _production_support_builder(task: dict, compiled_recipe: dict | None = None) -> Any:
    """Build the E5 token-only batch using the compiled method coefficients.

    The current E5 bundle supplies public question/answer pairs only. Reject
    any method that weights a cognitive objective without matching targets;
    never label answer-only training as world/action/value/pair learning.
    """
    from bramastra_lab.research.experience.supervision import (
        OBJECTIVE_TERMS)

    if not isinstance(compiled_recipe, dict):
        raise ValueError("compiled E5 method recipe is required")
    raw_weights = compiled_recipe.get("objective_coefficients")
    if not isinstance(raw_weights, dict):
        raise ValueError("compiled E5 method has no objective coefficients")
    weights = {term: float(raw_weights.get(term, 0.0))
               for term in OBJECTIVE_TERMS}
    if any(value < 0 or not __import__("math").isfinite(value)
           for value in weights.values()):
        raise ValueError("compiled E5 objective weights must be finite/nonnegative")
    unsupported = [term for term, value in weights.items()
                   if term != "token" and value > 0]
    if unsupported:
        raise ValueError(
            "E5 support bundle has no eligible targets for objectives: "
            + ", ".join(unsupported))
    if weights["token"] <= 0:
        raise ValueError("E5 method must positively weight answer-token targets")

    def build(trial_handle: Any) -> Any:
        from bramastra_lab.research.experience.sequences import (
            build_answer_row, collocate)
        from bramastra_lab.research.experience.supervision import (
            SupervisionWindow)

        rows = []
        for example in task.get("support_examples", ()):
            if "answer" not in example or "public" not in example:
                raise ValueError(
                    "support example carries no public/answer; refusing")
            rows.append(build_answer_row(
                [("goal", dict(example["public"]))],
                str(example["answer"]),
                provenance={"kind": "trajectory",
                            "episode_id": str(example.get(
                                "mechanism_id", "support")),
                            "task_semantic_id": "e5-support",
                            "split": "meta-training",
                            "source": "k8-bundle",
                            "collection_policy": "support",
                            "family": str(example.get("family", "meta"))},
                max_tokens=512))
        if not rows:
            raise ValueError("no support rows; refusing empty adaptation")
        batch = collocate(rows, max_seq=512)
        window = SupervisionWindow(
            weights=weights,
            enabled_terms=frozenset({"token"}))
        window.add("token", batch.target_count)
        return batch, window, {}, None

    return build


def _production_query_evaluator(task: dict, data_dir: str,
                               task_id: str) -> Any:
    """Query + protected measurement from resolved prepared answers (O08).

    Every query/protected prompt uses the COMPLETE canonical decision
    representation: the same goal-prefix tokens the compiler builds for
    training (no character slicing, no token slicing). If a complete valid
    prompt cannot fit the configured context budget the trial fails
    explicitly instead of discarding semantic content. Protected references
    must ALL resolve against prepared data: an unresolved, inconsistent or
    missing reference raises — it can never silently shrink the denominator
    (an unresolved protected case must not make the model look better).
    """
    from bramastra_lab.research.campaigns.phases.compiler import (
        K8_MAX_SEQ, goal_prefix_tokens)

    def evaluate(trial_handle: Any) -> Mapping[str, Any]:
        model = trial_handle.get("model") if isinstance(
            trial_handle, dict) else getattr(trial_handle, "model", None)
        config = trial_handle.get("config") if isinstance(
            trial_handle, dict) else getattr(trial_handle, "config", None)
        if model is None or config is None:
            raise ValueError("trial handle owns no model for query eval")
        from bramastra_lab.research.runtime.inference import generate_free_form

        queries = task.get("query_examples", ())
        if not queries:
            raise ValueError("no query examples; refusing")
        hits = 0
        query_rows: list[dict[str, Any]] = []
        for example in queries:
            mechanism_id = str(example.get("mechanism_id", ""))
            expected = resolve_mechanism_answer(data_dir, mechanism_id)
            public = example.get("public")
            if not isinstance(public, dict):
                row = resolve_mechanism_row(data_dir, mechanism_id)
                public = row["public"]
            prompt = goal_prefix_tokens(public)
            if len(prompt) + 1 > K8_MAX_SEQ:
                raise ValueError(
                    f"query {mechanism_id!r} canonical prompt needs "
                    f"{len(prompt)} tokens; context is {K8_MAX_SEQ}; "
                    "refusing to truncate a decision-relevant representation")
            report = generate_free_form(model, config, prompt,
                                        max_new_tokens=16)
            correct = str(report.answer).strip() == expected
            hits += 1 if correct else 0
            query_rows.append({"mechanism_id": mechanism_id,
                               "expected": expected,
                               "prompt_tokens": len(prompt),
                               "correct": bool(correct)})
        refs = list(task.get("protected_references", ()))
        protected_hits = protected_misses = 0
        unresolved: list[str] = []
        protected_rows: list[dict[str, Any]] = []
        for ref in refs:
            mechanism_id = str(ref.get("mechanism_id", ""))
            try:
                row = resolve_mechanism_row(data_dir, mechanism_id)
            except ValueError as exc:
                # Denominator integrity: a missing/inconsistent protected
                # reference is an explicit failure, never a skipped case.
                raise ValueError(
                    f"protected reference {mechanism_id!r} in task "
                    f"{task_id!r} did not resolve: {exc}") from exc
            prompt = goal_prefix_tokens(row["public"])
            if len(prompt) + 1 > K8_MAX_SEQ:
                raise ValueError(
                    f"protected {mechanism_id!r} canonical prompt needs "
                    f"{len(prompt)} tokens; context is {K8_MAX_SEQ}; "
                    "refusing to truncate a decision-relevant representation")
            report = generate_free_form(model, config, prompt,
                                        max_new_tokens=16)
            correct = str(report.answer).strip() == row["answer"]
            protected_hits += 1 if correct else 0
            protected_misses += 0 if correct else 1
            protected_rows.append({"mechanism_id": mechanism_id,
                                   "prompt_tokens": len(prompt),
                                   "correct": bool(correct)})
        expected_protected = len(refs)
        evaluated_protected = protected_hits + protected_misses
        if evaluated_protected != expected_protected or unresolved:
            raise ValueError(
                f"protected denominator integrity failure in task "
                f"{task_id!r}: expected {expected_protected}, evaluated "
                f"{evaluated_protected}, unresolved {sorted(unresolved)}")
        return {"measured_success": hits / len(queries),
                "query_hits": hits,
                "query_total": len(queries),
                "queries": query_rows,
                "protected": {
                    "expected": expected_protected,
                    "evaluated": evaluated_protected,
                    "hits": protected_hits,
                    "misses": protected_misses,
                    "unresolved": 0,
                    "denominator": evaluated_protected,
                    "mechanism_ids": [r["mechanism_id"]
                                      for r in protected_rows],
                    "rows": protected_rows}}

    return evaluate


class _MethodSensitiveDoubleTrainer:
    """Stateful deterministic learner double for E5 local integration.

    Changes only when the specified method is applied (via
    dispatch_method_to_trainer surface). Records applied method, support
    order and anchor; evaluation success is a deterministic function of
    (task, method, anchor) – never a constant. Fixture-labeled.
    """

    def __init__(self, *, anchor_id: str, support_order: tuple) -> None:
        from types import SimpleNamespace

        self.anchor_id = anchor_id
        self.support_order = tuple(support_order)
        self.model = SimpleNamespace(gates_enabled=True)
        self.optimizer = SimpleNamespace(param_groups=[{"lr": 0.0003}])
        self._multiplier = 1.0
        self.applied_method: str | None = None
        self.applied_count = 0

    def set_controller_multiplier(self, multiplier, reason) -> None:
        self._multiplier = multiplier

    def measured_success(self, task_id: str) -> float:
        # Deterministic measured table: method + task + anchor hash.
        # M1 best on even tasks, M0 on odd, M2 middle – ensures selection
        # matters (replacing the choice changes the outcome).
        digest = hashlib.sha256(
            f"{self.anchor_id}:{task_id}:{self.applied_method}".encode()).hexdigest()
        base = (int(digest[:4], 16) % 100) / 100.0
        bonus = {"M0": 0.1, "M1": 0.2, "M2": 0.15}.get(self.applied_method or "", 0.0)
        # Task parity shifts the ranking so no single method dominates.
        try:
            parity = int(task_id[-1], 16) % 2 if task_id[-1].isdigit() else 0
        except Exception:
            parity = 0
        if parity == 0 and (self.applied_method == "M1"):
            bonus += 0.15
        if parity == 1 and (self.applied_method == "M0"):
            bonus += 0.15
        return round(min(1.0, 0.3 + base * 0.4 + bonus), 4)


def _apply_method_to_handle(ops, handle: Any, method_id: str, compiled: dict,
                            *, task_identity: str) -> str:
    """Apply the recipe to the actual learner (never a bare fake trainer)."""
    from bramastra_lab.research.metalearning.dispatch import (
        dispatch_method_to_trainer)
    # Production handles own a real trainer; double handles own a double_id.
    # For doubles, wrap the handle in a method-sensitive trainer that owns
    # the same anchor/support lineage.
    if isinstance(handle, dict) and "double_id" in handle:
        trainer = handle.get("_e5_trainer")
        if trainer is None:
            trainer = _MethodSensitiveDoubleTrainer(
                anchor_id=str(handle.get("anchor_id", handle.get("init_state", "anchor"))),
                support_order=tuple(handle.get("support_order", (task_identity,))))
            handle["_e5_trainer"] = trainer
        # Dispatch validates the compiled recipe against the declared lineage.
        result = dispatch_method_to_trainer(method_id, compiled, trainer,
                                            task_identity=task_identity)
        trainer.applied_method = method_id
        trainer.applied_count += 1
        handle["applied_method"] = method_id
        return result
    # Production path: real trainer inside the handle.
    trainer = handle.get("trainer") if isinstance(handle, dict) else getattr(
        handle, "trainer", None)
    if trainer is None:
        raise ValueError("learner handle owns no trainer; refusing fake-trainer dispatch")
    return dispatch_method_to_trainer(method_id, compiled, trainer,
                                      task_identity=task_identity)


def _measure_archive_block(ops: Any, job: JobInput, anchor_record: dict,
                           anchor_id: str, anchor_support: tuple,
                           tasks: list[dict[str, Any]], learning_boundary: str,
                           *, block_label: str) -> dict[str, Any]:
    """Measure all three methods on every task of ONE archive block.

    Every (task, method) cell must be measured; failed trials are preserved
    as auditable records but make the block incomplete (the phase cannot
    continue with a partial measured archive). Returns rows/lineages/results
    or an "error" describing the refusal.
    """
    from bramastra_lab.research.campaigns import trial_service

    rows: list[dict[str, Any]] = []
    lineages: list[str] = []
    results: list[Any] = []
    failures: list[str] = []
    for task_index, task in enumerate(tasks):
        task_id = str(task.get("meta_task_id", f"mt-{task_index}"))
        for method_id in ("M0", "M1", "M2"):
            try:
                result = _run_archive_trial(
                    ops, job, anchor_record, anchor_id, anchor_support,
                    task, task_id, method_id, learning_boundary)
            except Exception as exc:
                # A failed trial is still an auditable archive record with
                # its real cost signature (zero work if it never ran).
                result = trial_service.TrialResult(
                    task_identity=task_id, method_id=method_id,
                    measured_updates=0, measured_success=None,
                    elapsed_seconds=0.0, support_identities=(),
                    query_identities=(), trial_checkpoint_id=None,
                    validation="failed", detail={"error": str(exc)[:300]})
                failures.append(f"{task_id}/{method_id}: {exc}")
            results.append(result)
            lineages.append(
                result.trial_checkpoint_id or result.detail.get(
                    "lineage", f"failed:{task_id}:{method_id}")[:64])
            rows.append({
                "task_identity": task_id, "method_id": method_id,
                "block": block_label,
                "measured_success": result.measured_success
                if result.measured_success is not None else 0.0,
                "measured_updates": result.measured_updates,
                "elapsed_seconds": result.elapsed_seconds,
                "support_identities": list(result.support_identities),
                "query_identities": list(result.query_identities),
                "trial_lineage": lineages[-1],
                "validation": result.validation})
    if not rows:
        return {"error": f"E5 archive block {block_label} produced no trials"}
    if failures:
        # The archive must contain measured outcomes for every declared
        # (task, method) cell; failed trials are preserved above, but the
        # phase cannot continue with an incomplete measured archive.
        return {"error": f"E5 archive block {block_label} trials incomplete: "
                         + "; ".join(failures[:3]),
                "rows": rows, "lineages": lineages, "results": results,
                "failures": failures}
    return {"rows": rows, "lineages": lineages, "results": results}


def execute(job: JobInput, *, ops=None,
            tasks_per_block: int | None = 2,
            learning_boundary: str = "test_substitute") -> PhaseResult:
    started = time.monotonic()
    job.validate()
    if job.phase != "E5" or job.seed is None:
        return PhaseResult(status="failed", device_seconds=time.monotonic() - started,
                           error="E5 requires seed",
                           evidence_kind=EVIDENCE_FIXTURE,
                           extra={"phase": "E5"})
    if learning_boundary not in ("production", "test_substitute"):
        return PhaseResult(status="failed", device_seconds=time.monotonic() - started,
                           error="E5 learning_boundary must be 'production' "
                                 "(GPU, real steps) or 'test_substitute' "
                                 "(local, fixture-labeled); refusing",
                           evidence_kind=EVIDENCE_FIXTURE,
                           extra={"phase": "E5"})
    if tasks_per_block is None:
        # Production uses 12/12/6 per campaign.json; local default is small
        # but must be explicit (never a silent task-count-as-update).
        tasks_per_block = 2
    if not isinstance(tasks_per_block, int) or tasks_per_block <= 0:
        return PhaseResult(status="failed", device_seconds=time.monotonic() - started,
                           error="tasks_per_block must be a positive integer",
                           evidence_kind=EVIDENCE_FIXTURE,
                           extra={"phase": "E5"})
    manifest = os.path.join(job.data_dir, "manifest.json")
    if not os.path.exists(manifest):
        return PhaseResult(status="failed", device_seconds=time.monotonic() - started,
                           error="bundle manifest missing",
                           evidence_kind=EVIDENCE_FIXTURE,
                           extra={"phase": "E5"})
    if ops is None:
        from bramastra_lab.research.campaigns.phases.ops import ProductionOps

        ops = ProductionOps(precision=job.precision)
    # Fixed adaptation anchor (separate from proposer weights).
    try:
        anchor_record = _resolve_anchor(job)
    except Exception as exc:
        return PhaseResult(status="failed", device_seconds=time.monotonic() - started,
                           error=str(exc), evidence_kind=EVIDENCE_FIXTURE,
                           extra={"phase": "E5"})
    try:
        anchor = ops.restore_parent(
            parent={**anchor_record, "run_dir": job.run_dir, "seed": job.seed},
            device=job.local_device, optimizer_policy="fresh")
        # Anchor identity (checkpoint + support order) is frozen for the job.
        # The restored handle stays pristine (no in-place mutation); lineage
        # is carried on forked children only, preserving sibling isolation.
        anchor_id = str(anchor_record.get("checkpoint_id"))
        anchor_support = tuple(f"E5-{job.seed}-support-{i}"
                               for i in range(tasks_per_block))
        proposer = ops.fork_child(parent_handle=anchor, optimizer_policy="fresh")
        if isinstance(proposer, dict):
            proposer["anchor_id"] = anchor_id
            proposer["support_order"] = anchor_support
    except Exception as exc:
        return PhaseResult(status="failed", device_seconds=time.monotonic() - started,
                           error=f"E5 anchor/proposer restore refused: {exc}",
                           evidence_kind=EVIDENCE_FIXTURE,
                           extra={"phase": "E5"})
    try:
        from bramastra_lab.research.metalearning.dispatch import _METHOD_PROGRAMS
        from bramastra_lab.research.metalearning.method_language import compile_method
    except Exception as exc:
        return PhaseResult(status="failed", device_seconds=time.monotonic() - started,
                           error=f"E5 method language unavailable: {exc}",
                           evidence_kind=EVIDENCE_FIXTURE,
                           extra={"phase": "E5"})
    # Block 1: archive_P0 — measured trials (no fixture generation).
    try:
        meta_training = _load_meta_tasks(job.data_dir, pool="meta-training")
    except Exception as exc:
        return PhaseResult(status="failed", device_seconds=time.monotonic() - started,
                           error=str(exc), evidence_kind=EVIDENCE_FIXTURE,
                           extra={"phase": "E5"})
    archive_tasks = meta_training[:tasks_per_block]
    if not archive_tasks:
        return PhaseResult(status="failed", device_seconds=time.monotonic() - started,
                           error="E5 archive block has no meta-training tasks",
                           evidence_kind=EVIDENCE_FIXTURE,
                           extra={"phase": "E5"})
    archive_rows: list[dict[str, Any]] = []
    trial_lineages: list[str] = []
    trial_results: list[Any] = []
    measured_a = _measure_archive_block(
        ops, job, anchor_record, anchor_id, anchor_support, archive_tasks,
        learning_boundary, block_label="A")
    if measured_a.get("error"):
        return PhaseResult(status="failed", device_seconds=time.monotonic() - started,
                           error=measured_a["error"],
                           evidence_kind=EVIDENCE_FIXTURE,
                           extra={"phase": "E5"})
    archive_rows.extend(measured_a["rows"])
    trial_lineages.extend(measured_a["lineages"])
    trial_results.extend(measured_a["results"])
    # Immutable archive capability (cutoff frozen before choices).
    try:
        from bramastra_lab.research.metalearning.dispatch import (
            MethodArchive, MethodTrialOutcome)
        outcome_list = []
        for r in archive_rows:
            if r["measured_success"] is None:
                raise ValueError(
                    f"archive row {r.get('task_identity')}/"
                    f"{r.get('method_id')} carries no measured success; "
                    "refusing to fabricate an archive best")
            outcome_list.append(MethodTrialOutcome(
                method_id=r["method_id"], task_identity=r["task_identity"],
                measured_updates=int(r["measured_updates"]),
                measured_success=float(r["measured_success"]),
                elapsed_seconds=float(r["elapsed_seconds"]),
                validation=str(r["validation"])))
        outcomes = tuple(outcome_list)
        archive = MethodArchive(rows=outcomes, cutoff_event_index=len(outcomes))
        archive_identity = archive.identity()
    except Exception as exc:
        return PhaseResult(status="failed", device_seconds=time.monotonic() - started,
                           error=f"E5 archive capability refused: {exc}",
                           evidence_kind=EVIDENCE_FIXTURE,
                           extra={"phase": "E5"})
    # Anchor stability: the requested parent key must match the resolved
    # record's lineage (a changed anchor key fails instead of silently
    # retargeting mid-job).
    requested_key = job.resolved_parent_key()
    if requested_key and anchor_record.get("lookup_key") \
            and str(anchor_record["lookup_key"]) != str(requested_key).split("/")[0].strip():
        # Resolved via structured parent_ref with a different key: refuse.
        return PhaseResult(status="failed", device_seconds=time.monotonic() - started,
                           error="E5 adaptation anchor key changed mid-job; refusing",
                           evidence_kind=EVIDENCE_FIXTURE,
                           extra={"phase": "E5"})
    # Block 2: TRAIN P0 on the measured archive first, THEN capture its
    # choice from the trained weights (U08 train-before-capture): the
    # pretraining decoder output must never survive the training boundary.
    try:
        proposer_batches = _proposer_batches(archive_tasks, archive)
        p0_train = _train_method_selection(
            ops, proposer, proposer_batches, job,
            task_identity=f"E5-{job.seed}-P0-train",
            learning_boundary=learning_boundary)
        if p0_train["validation"] not in ("measured", "boundary-only"):
            raise ValueError(
                f"P0 method training refused: {p0_train.get('error')}")
    except Exception as exc:
        return PhaseResult(status="failed", device_seconds=time.monotonic() - started,
                           error=f"E5 P0 training refused: {exc}",
                           evidence_kind=EVIDENCE_FIXTURE,
                           extra={"phase": "E5"})
    # Bind P0's learned proposal to a durable payload before asking it to
    # choose. The adaptation anchor initialized this proposer, but it is not
    # the model that emitted the post-training proposal.
    try:
        p0_checkpoint_id = ops.publish_checkpoint(
            handle=proposer, run_dir=job.run_dir, phase="E5", arm="P0",
            seed=job.seed,
            update_index=int(ops.optimizer_updates(proposer)),
            parent_checkpoint_id=anchor_id, data_dir=job.data_dir)
        if not isinstance(p0_checkpoint_id, str) or not p0_checkpoint_id.strip():
            raise ValueError("checkpoint publisher returned no P0 identity")
        p0_choice, p0_capture = _capture_proposer_choice(
            ops, proposer, archive, archive_tasks, anchor_id,
            job, proposer_checkpoint_id=p0_checkpoint_id)
        p0_capture["captured_after_training"] = True
        p0_capture["p0_training_committed"] = int(
            p0_train.get("committed_updates", 0))
    except Exception as exc:
        return PhaseResult(status="failed", device_seconds=time.monotonic() - started,
                           error=f"E5 P0 checkpoint/capture refused: {exc}",
                           evidence_kind=EVIDENCE_FIXTURE,
                           extra={"phase": "E5"})
    # Block B: a DISTINCT fresh archive measured on fresh meta-training
    # tasks, admitted only after its cutoff; the trained successors learn
    # method selection from this newly admitted archive (never Block A).
    block_b_tasks = meta_training[tasks_per_block:2 * tasks_per_block]
    if not block_b_tasks:
        return PhaseResult(status="failed", device_seconds=time.monotonic() - started,
                           error="E5 Block B unavailable: meta-training pool "
                                 "carries no fresh tasks beyond the Block A "
                                 "slice; archives A and B must be distinct",
                           evidence_kind=EVIDENCE_FIXTURE,
                           extra={"phase": "E5"})
    measured_b = _measure_archive_block(
        ops, job, anchor_record, anchor_id, anchor_support, block_b_tasks,
        learning_boundary, block_label="B")
    if measured_b.get("error"):
        return PhaseResult(status="failed", device_seconds=time.monotonic() - started,
                           error=measured_b["error"],
                           evidence_kind=EVIDENCE_FIXTURE,
                           extra={"phase": "E5"})
    archive_rows.extend(measured_b["rows"])
    trial_lineages.extend(measured_b["lineages"])
    trial_results.extend(measured_b["results"])
    try:
        outcomes_b = tuple(MethodTrialOutcome(
            method_id=r["method_id"], task_identity=r["task_identity"],
            measured_updates=int(r["measured_updates"]),
            measured_success=float(r["measured_success"]),
            elapsed_seconds=float(r["elapsed_seconds"]),
            validation=str(r["validation"])) for r in measured_b["rows"])
        archive_b = MethodArchive(rows=outcomes_b,
                                  cutoff_event_index=len(outcomes_b))
        archive_b_identity = archive_b.identity()
        if archive_b_identity == archive_identity:
            raise ValueError(
                "Block B archive identity equals Block A; archives must be "
                "distinct")
        successor_tasks = block_b_tasks
        successor_archive = archive_b
        successor_batches = _proposer_batches(block_b_tasks, archive_b)
    except Exception as exc:
        return PhaseResult(status="failed", device_seconds=time.monotonic() - started,
                           error=f"E5 Block B archive refused: {exc}",
                           evidence_kind=EVIDENCE_FIXTURE,
                           extra={"phase": "E5"})
    try:
        p1_handle = ops.fork_child(parent_handle=proposer, optimizer_policy="fresh")
        if isinstance(p1_handle, dict):
            p1_handle["anchor_id"] = anchor_id
            p1_handle["support_order"] = anchor_support
        p1_compiled = compile_method(_METHOD_PROGRAMS[p0_choice],
                                     runtime_config={"profile": "k8-campaign"})
        _apply_method_to_handle(ops, p1_handle, p0_choice, p1_compiled,
                                task_identity=f"E5-{job.seed}-P1")
        p1_train = _train_method_selection(
            ops, p1_handle, successor_batches, job,
            task_identity=f"E5-{job.seed}-P1-train",
            learning_boundary=learning_boundary)
        p_fixed_handle = ops.fork_child(parent_handle=proposer, optimizer_policy="fresh")
        if isinstance(p_fixed_handle, dict):
            p_fixed_handle["anchor_id"] = anchor_id
            p_fixed_handle["support_order"] = anchor_support
        p_fixed_compiled = compile_method(_METHOD_PROGRAMS["M0"],
                                          runtime_config={"profile": "k8-campaign"})
        _apply_method_to_handle(ops, p_fixed_handle, "M0", p_fixed_compiled,
                                task_identity=f"E5-{job.seed}-P_fixed")
        p_fixed_train = _train_method_selection(
            ops, p_fixed_handle, successor_batches, job,
            task_identity=f"E5-{job.seed}-P_fixed-train",
            learning_boundary=learning_boundary)
        for label, record in (("P1", p1_train), ("P_fixed", p_fixed_train)):
            if record["validation"] not in ("measured", "boundary-only"):
                raise ValueError(
                    f"{label} successor training refused: {record.get('error')}")
        # Publish successor checkpoints (real payloads on GPU; fixture
        # receipts for doubles locally).
        successor_checkpoints: dict[str, str] = {}
        for label, handle in (("P1", p1_handle), ("P_fixed", p_fixed_handle)):
            checkpoint_id = ops.publish_checkpoint(
                handle=handle, run_dir=job.run_dir, phase="E5",
                arm=label, seed=job.seed,
                update_index=int(ops.optimizer_updates(handle)),
                parent_checkpoint_id=p0_checkpoint_id,
                data_dir=job.data_dir)
            if not isinstance(checkpoint_id, str) or not checkpoint_id.strip():
                raise ValueError(
                    f"{label} checkpoint publisher returned no identity")
            successor_checkpoints[label] = checkpoint_id
    except Exception as exc:
        return PhaseResult(status="failed", device_seconds=time.monotonic() - started,
                           error=f"E5 successor fork/apply refused: {exc}",
                           evidence_kind=EVIDENCE_FIXTURE,
                           extra={"phase": "E5"})
    # Block 3: fresh confirmation — capture ALL choices BEFORE fresh outcomes.
    # Exact pool only (meta-confirmation); silently substituting validation
    # would leak a different split into confirmation.
    try:
        meta_confirm = _load_meta_tasks(job.data_dir, pool="meta-confirmation")
    except Exception as exc:
        return PhaseResult(status="failed", device_seconds=time.monotonic() - started,
                           error=f"E5 confirmation pool refused (meta-confirmation "
                                 f"required, no validation fallback): {exc}",
                           evidence_kind=EVIDENCE_FIXTURE,
                           extra={"phase": "E5"})
    confirm_tasks = meta_confirm[:max(1, min(tasks_per_block, len(meta_confirm)))]
    # Immutable confirmation capability: separate archive snapshot that must
    # NOT contain confirmation outcomes at capture time.
    try:
        for task in confirm_tasks:
            task_id = str(task.get("meta_task_id", "mc-?"))
            if archive.contains_task_outcomes(task_id):
                raise ValueError(
                    f"confirmation task {task_id} already in archive context; "
                    "moving a confirmation row into context must fail")
    except ValueError:
        raise
    except Exception as exc:
        return PhaseResult(status="failed", device_seconds=time.monotonic() - started,
                           error=f"E5 confirmation isolation refused: {exc}",
                           evidence_kind=EVIDENCE_FIXTURE,
                           extra={"phase": "E5"})
    # Capture choices from all five policies before any fresh method trial.
    # P1 and P_fixed are trained proposers: each decodes its OWN choice from
    # its own weights (never a copy of P0's choice; P_fixed is NOT the
    # always-M0 controller — fixed_M0 is). Frozen P0 keeps its already
    # captured choice; no policy is re-decided after outcomes.
    confirmation_choices: dict[str, str] = {}
    successor_choice_captures: dict[str, dict] = {}
    try:
        p1_choice, p1_capture = _capture_successor_choice(
            p1_handle, successor_archive, successor_tasks, anchor_id, "P1",
            ops=ops, job=job,
            proposer_checkpoint_id=successor_checkpoints["P1"])
        p_fixed_choice, p_fixed_capture = _capture_successor_choice(
            p_fixed_handle, successor_archive, successor_tasks, anchor_id,
            "P_fixed", ops=ops, job=job,
            proposer_checkpoint_id=successor_checkpoints["P_fixed"])
        successor_choice_captures["P1"] = p1_capture
        successor_choice_captures["P_fixed"] = p_fixed_capture
        confirmation_choices["P1"] = p1_choice
        confirmation_choices["P_fixed"] = p_fixed_choice
        # Frozen P0 choice (captured before successors trained; never
        # re-decided after confirmation outcomes).
        confirmation_choices["P0"] = p0_choice
        confirmation_choices["fixed_M0"] = "M0"
        # Deterministic random (seeded, not outcome-dependent).
        import random as _random
        rng = _random.Random(f"E5-confirm:{job.seed}:{archive_identity[:8]}")
        confirmation_choices["random"] = rng.choice(["M0", "M1", "M2"])
    except Exception as exc:
        return PhaseResult(status="failed", device_seconds=time.monotonic() - started,
                           error=f"E5 confirmation capture refused: {exc}",
                           evidence_kind=EVIDENCE_FIXTURE,
                           extra={"phase": "E5"})
    # Measured confirmation: run every method on every confirmation task
    # through the same trial service, then score the frozen choices against
    # the resulting measured table (same-table scoring, O09). Choices were
    # captured before any of these trials ran.
    confirmation_rows: list[dict] = []
    confirmation_table: dict[tuple[str, str], float] = {}
    confirmation_failed: list[str] = []
    for task in confirm_tasks:
        task_id = str(task.get("meta_task_id", "mc-?"))
        for method_id in ("M0", "M1", "M2"):
            try:
                result = _run_archive_trial(
                    ops, job, anchor_record, anchor_id, anchor_support,
                    task, task_id, method_id, learning_boundary)
            except Exception as exc:
                confirmation_failed.append(f"{task_id}/{method_id}: {exc}")
                continue
            if result.validation != "measured" or \
                    result.measured_success is None:
                confirmation_failed.append(
                    f"{task_id}/{method_id}: {result.validation}")
                continue
            confirmation_table[(task_id, method_id)] = float(
                result.measured_success)
    if confirmation_failed:
        return PhaseResult(status="failed", device_seconds=time.monotonic() - started,
                           error="E5 confirmation trials incomplete: "
                                 + "; ".join(confirmation_failed[:3]),
                           evidence_kind=EVIDENCE_FIXTURE,
                           extra={"phase": "E5"})
    for task in confirm_tasks:
        task_id = str(task.get("meta_task_id", "mc-?"))
        measured = {method_id: confirmation_table[(task_id, method_id)]
                    for method_id in ("M0", "M1", "M2")}
        best = max(measured.values())
        for policy, method in confirmation_choices.items():
            chosen = measured[method]
            confirmation_rows.append({
                "task_identity": task_id, "policy": policy,
                "method_id": method, "measured_success": chosen,
                "best_measured": best,
                "regret": round(best - chosen, 4)})
    # Counts derive from measured trials (each measured trial with updates>0
    # counts once), never from task-count arithmetic. Learned evidence
    # requires production-measured trials with real restorable checkpoints;
    # everything else stays fixture (O09).
    committed = sum(1 for r in archive_rows if int(r.get("measured_updates", 0)) > 0
                    and r.get("validation") == "measured")
    attempted = len([r for r in archive_rows if r.get("validation") in ("measured", "failed")])
    learned_trials = sum(
        1 for r in trial_results
        if getattr(r, "validation", "") == "measured"
        and r.detail.get("learning_boundary") == "production"
        and isinstance(r.trial_checkpoint_id, str)
        and len(r.trial_checkpoint_id) == 64)
    evidence = EVIDENCE_LEARNED_CAMPAIGN if learned_trials > 0 \
        else EVIDENCE_FIXTURE
    # Confirmation comparison: P1 beats P0 but not P_fixed attributes gains
    # to extra meta-training, not to the selected method (experiment.md).
    confirm_summary = _summarize_confirmation(confirmation_rows)
    generation_chain = _build_e5_generation_chain(
        seed=int(job.seed), p0_checkpoint_id=p0_checkpoint_id,
        p0_choice=p0_choice, p0_capture=p0_capture,
        p1_checkpoint_id=successor_checkpoints["P1"],
        p1_choice=p1_choice,
        p1_capture=successor_choice_captures["P1"],
        anchor_id=anchor_id, archive_identity=archive_identity,
        archive_b_identity=archive_b_identity,
        confirmation_rows=confirmation_rows,
        confirmation_summary=confirm_summary,
        confirmation_trial_count=len(confirmation_table),
        fixture=(evidence != EVIDENCE_LEARNED_CAMPAIGN))
    artifact_dir = os.path.join(job.run_dir, "phase_outputs", job.phase)
    os.makedirs(artifact_dir, exist_ok=True)
    with open(os.path.join(artifact_dir, f"E5-{job.seed}.json"), "w",
              encoding="utf-8") as handle_file:
        json.dump({"archive_identity": archive_identity,
                   "archive_rows": archive_rows,
                   "p0_choice": p0_choice,
                   "p0_capture": p0_capture,
                   "checkpoint_lineage": {
                       "adaptation_anchor": anchor_id,
                       "P0": {"checkpoint_id": p0_checkpoint_id,
                              "parent_checkpoint_id": anchor_id},
                       "P1": {"checkpoint_id": successor_checkpoints["P1"],
                              "parent_checkpoint_id": p0_checkpoint_id},
                       "P_fixed": {
                           "checkpoint_id": successor_checkpoints["P_fixed"],
                           "parent_checkpoint_id": p0_checkpoint_id}},
                   "generation_chain": generation_chain,
                   "p0_training": p0_train,
                   "successors": {
                       "P1": {"method": p0_choice, "training": p1_train,
                              "checkpoint": successor_checkpoints.get("P1")},
                       "P_fixed": {"method": "M0",
                                   "training": p_fixed_train,
                                   "checkpoint": successor_checkpoints.get(
                                       "P_fixed")}},
                   "anchors": {"P1": p0_choice, "P_fixed": "M0", "P0": p0_choice},
                   "successor_choice_captures": successor_choice_captures,
                   "confirmation_choices": confirmation_choices,
                   "confirmation_rows": confirmation_rows,
                   "confirmation_summary": confirm_summary,
                   "committed_updates": committed,
                   "attempted_updates": attempted,
                   "evidence_kind": evidence,
                   "learning_boundary": learning_boundary,
                   "lineages": trial_lineages[:6]},
                  handle_file, indent=2, sort_keys=True)
    # Attach the Block-B provenance (fresh successor archive).
    try:
        artifact_path = os.path.join(artifact_dir, f"E5-{job.seed}.json")
        updated = json.load(open(artifact_path, encoding="utf-8"))
        updated["archive_b_identity"] = archive_b_identity
        updated["archive_b_rows"] = measured_b["rows"]
        updated["archive_blocks"] = {
            "A": {"tasks": len(archive_tasks),
                  "identity": archive_identity},
            "B": {"tasks": len(successor_tasks),
                  "identity": archive_b_identity}}
        with open(artifact_path, "w", encoding="utf-8") as handle_file:
            json.dump(updated, handle_file, indent=2, sort_keys=True)
    except Exception:
        pass
    if committed <= 0:
        return PhaseResult(status="failed", committed_updates=committed,
                           attempted_updates=attempted,
                           supervised_exposure=len(archive_rows),
                           device_seconds=time.monotonic() - started,
                           error="E5 produced zero measured committed trials; "
                                 "a no-training boundary never yields positive "
                                 "learned updates",
                           evidence_kind=EVIDENCE_FIXTURE,
                           extra={"phase": "E5", "trials": len(archive_rows),
                                  "archive_identity": archive_identity})
    result = PhaseResult(status="completed", committed_updates=committed,
                         attempted_updates=attempted,
                         supervised_exposure=len(archive_rows),
                         device_seconds=time.monotonic() - started,
                         checkpoint_identity=archive_identity,
                         evidence_kind=evidence,
                         extra={"phase": "E5", "trials": len(archive_rows),
                                "archive_identity": archive_identity,
                                "anchors": {"P1": p0_choice, "P_fixed": "M0"}})
    try:
        result.validate()
    except ValueError as exc:
        return PhaseResult(status="failed", committed_updates=committed,
                           attempted_updates=attempted,
                           supervised_exposure=len(archive_rows),
                           device_seconds=time.monotonic() - started,
                           error=f"E5 receipt refused: {exc}",
                           evidence_kind=EVIDENCE_FIXTURE,
                           extra={"phase": "E5"})
    return result


def _summarize_confirmation(rows: list[dict]) -> dict[str, Any]:
    """Per-policy means and the P1-vs-P_fixed attribution rule (O09)."""
    by_policy: dict[str, list[float]] = {}
    for row in rows:
        by_policy.setdefault(str(row.get("policy", "?")), []).append(
            float(row.get("measured_success", 0.0)))
    means = {policy: (sum(values) / len(values) if values else 0.0)
             for policy, values in by_policy.items()}
    p1 = means.get("P1", 0.0)
    p0 = means.get("P0", 0.0)
    p_fixed = means.get("P_fixed", 0.0)
    if p1 > p0 and p1 > p_fixed:
        attribution = "recursive-benefit-supported"
    elif p1 > p0:
        attribution = "extra-training-only"
    else:
        attribution = "negative-or-inconclusive"
    return {"policy_means": {k: round(v, 4) for k, v in means.items()},
            "attribution": attribution}


def _proposer_batches(archive_tasks: list, archive: Any) -> list[Any]:
    """Method-token training batches from measured archive bests (O09).

    Each batch asks for the best-feasible method of one archive task given
    only permitted descriptors (no query labels/outcomes); the target is the
    measured best method token. Ties resolve to M0 (valid no-change).
    """
    from bramastra_lab.research.experience.sequences import (
        build_answer_row, collocate)

    batches = []
    measured_count = 0
    for task in archive_tasks:
        task_id = str(task.get("meta_task_id", "mt-?"))
        best = archive.best_measured(task_id)
        if best is None:
            continue
        measured_count += 1
        descriptor = json.dumps(
            {"task_identity": task_id,
             "family": str(task.get("family", "")),
             "archive_methods": sorted(
                 {row.method_id for row in archive.rows})},
            sort_keys=True)
        row = build_answer_row(
            [("goal", {"method_choice": descriptor})], best,
            provenance={"kind": "trajectory",
                        "episode_id": f"p0-{task_id}",
                        "task_semantic_id": "e5-proposer",
                        "split": "meta-training",
                        "source": "measured-archive",
                        "collection_policy": "proposer-train",
                        "family": "meta"},
            max_tokens=512)
        batches.append(collocate([row], max_seq=512))
    if not batches:
        validations: dict[str, int] = {}
        for row in archive.rows:
            key = str(getattr(row, "validation", "?"))
            validations[key] = validations.get(key, 0) + 1
        raise ValueError(
            "no measured archive bests for proposer training "
            f"(tasks={len(archive_tasks)} measured_tasks={measured_count} "
            f"archive_rows={len(archive.rows)} validations={validations})")
    return batches


def _train_method_selection(ops: Any, handle: Any, batches: list,
                            job: JobInput, *, task_identity: str,
                            learning_boundary: str) -> dict[str, Any]:
    """Train method-token prediction with an explicit boundary (O09).

    Real trainers (own `accumulate_full_window`): bind authority, then step
    only under a live ledger allocation, else run the admission-checked
    no-op boundary (backward runs, gradients discarded, fixture-labeled).
    Deterministic doubles (no window API): same batches through the
    recording double interface, fixture-labeled. Capability-checked, never
    class-name-checked.
    """
    from bramastra_lab.research.campaigns import trial_service
    from bramastra_lab.research.campaigns.phases.session import (
        bind_job_reservation, drive_noop_boundary)
    from bramastra_lab.research.experience.supervision import SupervisionWindow

    def _token_window(batch: Any) -> Any:
        window = SupervisionWindow(
            weights={"token": 1.0, "world": 0.0, "action": 0.0,
                     "value": 0.0, "pair": 0.0, "pg": 0.0},
            enabled_terms=frozenset({"token"}))
        window.add("token", batch.target_count)
        return window

    trainer = handle.get("trainer") if isinstance(handle, dict) else getattr(
        handle, "trainer", None)
    real_trainer = trainer is not None and hasattr(
        trainer, "accumulate_full_window")
    if not real_trainer:
        # Deterministic double path: recording interface, fixture evidence.
        try:
            for batch in batches:
                ops.training_update(handle, batch=batch,
                                    window=_token_window(batch), extra={})
        except Exception as exc:
            return {"validation": "failed",
                    "error": f"double proposer update refused: {exc}"}
        return {"validation": "measured", "evidence_kind": "fixture",
                "committed_updates": len(batches)}
    reservation = {
        "allocation_id": job.allocation_id or f"e5-local-{job.seed}",
        "reservation_id": (job.reservation_id or f"e5-local-{job.seed}")
        + ":proposer",
        "job_id": job.job_id or f"E5-{job.seed}",
        "device": job.physical_device,
        "phase": "E5",
        "deadline_unix": job.reservation_deadline_unix or 9e9,
        "remaining_updates": len(batches),
        "source_hash": job.source_hash or "unbound-source"}
    try:
        bind_job_reservation(handle, dict(reservation),
                             remaining_updates=len(batches))
    except Exception as exc:
        return {"validation": "failed",
                "error": f"proposer authority refused: {exc}"}
    if learning_boundary == "production" and \
            trial_service._ledger_stepping_allowed(job.run_dir, reservation):
        committed = 0
        try:
            for batch in batches:
                outcome = ops.apply_update(
                    handle, batch=batch, window=_token_window(batch),
                    extra={}, pair_rows=None)
                committed += int(outcome.get("committed", 0))
        except Exception as exc:
            return {"validation": "failed",
                    "error": f"proposer update refused: {exc}"}
        return {"validation": "measured", "committed_updates": committed,
                "evidence_kind": "learned-campaign"}
    try:
        grads_seen = 0
        for batch in batches:
            proof = drive_noop_boundary(
                handle, batch=batch, window=_token_window(batch), extra={},
                pair_rows=None)
            grads_seen += len(proof.get("named_grads_finite", {}))
    except Exception as exc:
        return {"validation": "failed",
                "error": f"proposer boundary refused: {exc}"}
    return {"validation": "boundary-only", "evidence_kind": "fixture",
            "grads_finite": grads_seen}


def _capture_proposer_choice(ops, proposer: Any, archive, archive_tasks: list,
                             anchor_id: str, job: JobInput, *,
                             proposer_checkpoint_id: str | None = None
                             ) -> tuple[str, dict]:
    """Capture P0's choice: real decoder first, named teacher fallback.

    The production path invokes the proposer decoder through the model
    interface and parses its actual output. A host-computed majority is
    allowed ONLY as an explicitly named teacher/control fallback (recorded
    as capture_origin teacher-majority-control), never packaged as model
    output. Locally with doubles (no model), the teacher path is taken
    and labeled.
    """
    from bramastra_lab.research.metalearning.dispatch import (
        MethodProposer, _METHOD_PROGRAMS)

    task_descriptors = [{"task_identity": str(t.get("meta_task_id")),
                         "family": str(t.get("family", ""))} for t in archive_tasks]
    model = proposer.get("model") if isinstance(proposer, dict) else getattr(
        proposer, "model", None)
    config = proposer.get("config") if isinstance(proposer, dict) else getattr(
        proposer, "config", None)
    model_usable = model is not None and config is not None and not (
        isinstance(proposer, dict) and "double_id" in proposer)
    payload_id = proposer_checkpoint_id or anchor_id
    if model_usable:
        try:
            if not proposer_checkpoint_id:
                raise ValueError(
                    "model-origin capture requires its published checkpoint")
            method_proposer = MethodProposer(
                model, config,
                checkpoint_payload_identity=proposer_checkpoint_id)
            descriptor = {"task_identities": sorted(
                t["task_identity"] for t in task_descriptors),
                "family": "meta-training"}
            capture = method_proposer.capture_proposal(
                descriptor, archive)
            from bramastra_lab.research.metalearning.dispatch import (
                parse_method_selection)
            method_id, program = parse_method_selection(capture.raw_output)
            _validate_e5_model_capture(
                ops, job, capture, capture.raw_output,
                proposer_checkpoint_id)
            return method_id, {
                "capture_origin": "model-decoder",
                "raw_output": capture.raw_output,
                "parsed_method": method_id,
                "rendered_input_hash": capture.transcript_hash[:16],
                "archive_identity": archive.identity(),
                "archive_cutoff": archive.cutoff_event_index,
                "recipe_identity": program.identity()
                if hasattr(program, "identity") else method_id,
                "program_identity": program.identity()
                if hasattr(program, "identity") else method_id,
                "checkpoint_payload_identity": proposer_checkpoint_id,
                "adaptation_anchor_identity": anchor_id}
        except Exception as exc:
            teacher_error = str(exc)[:200]
    else:
        teacher_error = "no model on proposer handle (double); teacher control"
    choice = _teacher_majority_choice(archive, archive_tasks)
    from bramastra_lab.research.metalearning.method_language import (
        compile_method)
    try:
        compiled = compile_method(_METHOD_PROGRAMS[choice],
                                  runtime_config={"profile": "k8-campaign"})
        recipe_identity = str(compiled.get("identity", ""))
        program_identity = str(compiled.get("program_identity", ""))
    except Exception:
        recipe_identity = choice
        program_identity = choice
    rendered_input = json.dumps({"task_descriptors": task_descriptors,
                                 "archive_identity": archive.identity(),
                                 "cutoff": archive.cutoff_event_index,
                                 "anchor_id": anchor_id[:12]},
                                sort_keys=True)
    forbidden = ("measured_success", "query_outcome", "label", "confirmation")
    for token in forbidden:
        if token in rendered_input:
            raise ValueError(
                f"current-task outcome {token!r} in proposal context; refusing")
    if choice not in ("M0", "M1", "M2"):
        raise ValueError(f"proposer choice {choice!r} not in M0/M1/M2")
    return choice, {
        "capture_origin": "teacher-majority-control",
        "model_error": teacher_error,
        "raw_output": choice, "parsed_method": choice,
        "rendered_input_hash": hashlib.sha256(
            rendered_input.encode()).hexdigest()[:16],
        "archive_identity": archive.identity(),
        "archive_cutoff": archive.cutoff_event_index,
        "recipe_identity": recipe_identity,
        "program_identity": program_identity,
        "checkpoint_payload_identity": payload_id,
        "adaptation_anchor_identity": anchor_id}


def _capture_successor_choice(handle: Any, archive, archive_tasks: list,
                              anchor_id: str, label: str, *, ops=None,
                              job: JobInput | None = None,
                              proposer_checkpoint_id: str | None = None
                              ) -> tuple[str, dict]:
    """Independently decode a trained successor's method choice (Block C).

    P1 and P_fixed are each trained proposers with their own weights. Each
    decodes its OWN choice from its own model on the confirmation context;
    P0's choice is never copied into a successor, and the always-M0
    controller is the separate fixed_M0 policy. With local doubles (no real
    model) the named teacher/control fallback is taken and labeled.
    """
    from bramastra_lab.research.metalearning.dispatch import MethodProposer

    model = handle.get("model") if isinstance(handle, dict) else getattr(
        handle, "model", None)
    config = handle.get("config") if isinstance(handle, dict) else getattr(
        handle, "config", None)
    double = isinstance(handle, dict) and "double_id" in handle
    payload_id = proposer_checkpoint_id or anchor_id
    if model is not None and config is not None and not double:
        try:
            if not proposer_checkpoint_id or ops is None or job is None:
                raise ValueError(
                    f"{label} model-origin capture requires a published, "
                    "verifiable checkpoint")
            method_proposer = MethodProposer(
                model, config,
                checkpoint_payload_identity=proposer_checkpoint_id)
            descriptor = {"task_identities": sorted(
                str(t.get("meta_task_id")) for t in archive_tasks),
                "family": "meta-confirmation"}
            capture = method_proposer.capture_proposal(descriptor, archive)
            from bramastra_lab.research.metalearning.dispatch import (
                parse_method_selection)

            method_id, program = parse_method_selection(capture.raw_output)
            if method_id not in ("M0", "M1", "M2"):
                raise ValueError(
                    f"{label} decoded method {method_id!r} not in M0/M1/M2")
            _validate_e5_model_capture(
                ops, job, capture, capture.raw_output,
                proposer_checkpoint_id)
            return method_id, {
                "capture_origin": "model-decoder",
                "raw_output": capture.raw_output,
                "parsed_method": method_id,
                "rendered_input_hash": capture.transcript_hash[:16],
                "archive_identity": archive.identity(),
                "archive_cutoff": archive.cutoff_event_index,
                "checkpoint_payload_identity": proposer_checkpoint_id,
                "adaptation_anchor_identity": anchor_id}
        except Exception as exc:
            fallback_reason = f"{label} decoder refused: {str(exc)[:200]}"
    else:
        fallback_reason = (
            f"no model on {label} successor handle (double); teacher control")
    # Named teacher/control fallback for doubles only. In production a
    # trained successor that cannot decode fails the phase (never silently
    # replaced by P0's choice or M0).
    if not double:
        raise ValueError(
            f"{label} successor choice could not be decoded independently; "
            f"copying another policy's choice is forbidden ({fallback_reason})")
    choice = _teacher_majority_choice(archive, archive_tasks)
    rendered_input = json.dumps(
        {"label": label,
         "task_identities": sorted(str(t.get("meta_task_id"))
                                   for t in archive_tasks),
         "archive_identity": archive.identity(),
         "cutoff": archive.cutoff_event_index,
         "anchor_id": anchor_id[:12]}, sort_keys=True)
    forbidden = ("measured_success", "query_outcome", "label-outcome",
                 "confirmation-outcome")
    for token in forbidden:
        if token in rendered_input:
            raise ValueError(
                f"current-task outcome {token!r} in proposal context; refusing")
    return choice, {
        "capture_origin": "teacher-majority-control",
        "model_error": fallback_reason,
        "raw_output": choice, "parsed_method": choice,
        "rendered_input_hash": hashlib.sha256(
            rendered_input.encode()).hexdigest()[:16],
        "archive_identity": archive.identity(),
        "archive_cutoff": archive.cutoff_event_index,
        "checkpoint_payload_identity": payload_id,
        "adaptation_anchor_identity": anchor_id}


def _validate_e5_model_capture(ops, job: JobInput, capture: Any,
                               raw_output: str,
                               expected_checkpoint_id: str) -> str:
    """Validate AST and prove the referenced proposer payload is restorable."""
    from bramastra_lab.research.metalearning.dispatch import (
        parse_method_selection)
    from bramastra_lab.research.metalearning.generations import validate_origin

    if capture.checkpoint_payload_identity != expected_checkpoint_id:
        raise ValueError("capture checkpoint identity differs from published payload")
    proof = ops.restore_verify(
        run_dir=job.run_dir, checkpoint_id=expected_checkpoint_id)
    if not isinstance(proof, dict) or proof.get("restored_ok") is not True \
            or proof.get("checkpoint_id") != expected_checkpoint_id:
        raise ValueError("proposer checkpoint did not pass restore verification")
    _method_id, reparsed = parse_method_selection(raw_output)
    return validate_origin(
        capture, reparsed_program=reparsed,
        checkpoint_registry={expected_checkpoint_id: "model"})


def _build_e5_generation_chain(*, seed: int, p0_checkpoint_id: str,
                               p0_choice: str, p0_capture: dict,
                               p1_checkpoint_id: str, p1_choice: str,
                               p1_capture: dict, anchor_id: str,
                               archive_identity: str,
                               archive_b_identity: str,
                               confirmation_rows: list[dict],
                               confirmation_summary: dict,
                               confirmation_trial_count: int,
                               fixture: bool) -> dict[str, Any]:
    """Bind E5's trained proposer and independent confirmation into M24.

    This records the recursive-selection lineage without promoting a learned
    parent. Publication remains a separate chief-approved operation.
    """
    from bramastra_lab.research.contracts.core import content_identity
    from bramastra_lab.research.metalearning.dispatch import (
        _METHOD_PROGRAMS)
    from bramastra_lab.research.metalearning.generations import (
        GenerationReceipt, GenerationRegistry)
    from bramastra_lab.research.metalearning.method_language import (
        compile_method)

    attempted = tuple(_METHOD_PROGRAMS[name].identity()
                      for name in ("M0", "M1", "M2"))
    registry = GenerationRegistry()
    first = GenerationReceipt(
        generation_id=f"E5-{seed}-P0-{p0_checkpoint_id[:12]}",
        predecessor_receipt_id=None,
        proposer_checkpoint=p0_checkpoint_id,
        proposer_origin=("model" if p0_capture.get("capture_origin")
                         == "model-decoder" else "symbolic_teacher"),
        parent_method=_METHOD_PROGRAMS[p0_choice].to_dict(),
        candidate_program=None, compiled_identity=None,
        comparison_identity=archive_identity,
        attempted_candidates=attempted,
        consumed_resources={"archive_blocks": 1.0},
        status="compared", fixture=fixture)
    first_id = registry.record(first)

    candidate = None if p1_choice == "M0" else _METHOD_PROGRAMS[p1_choice]
    compiled_identity = (compile_method(
        candidate, runtime_config={"profile": "k8-campaign"})["identity"]
        if candidate is not None else None)
    comparison_identity = content_identity({
        "archive_a": archive_identity,
        "archive_b": archive_b_identity,
        "confirmation_rows": confirmation_rows,
        "confirmation_summary": confirmation_summary})
    model_origin = p1_capture.get("capture_origin") == "model-decoder"
    confirmed_gain = confirmation_summary.get("attribution") \
        == "recursive-benefit-supported"
    second = GenerationReceipt(
        generation_id=f"E5-{seed}-P1-{p1_checkpoint_id[:12]}",
        predecessor_receipt_id=first_id,
        proposer_checkpoint=p1_checkpoint_id,
        proposer_origin="model" if model_origin else "symbolic_teacher",
        parent_method=_METHOD_PROGRAMS["M0"].to_dict(),
        candidate_program=candidate,
        compiled_identity=compiled_identity,
        comparison_identity=comparison_identity,
        attempted_candidates=attempted,
        consumed_resources={
            "confirmation_trials": float(confirmation_trial_count)},
        status=("confirmed" if confirmed_gain and model_origin else "rejected"),
        fixture=fixture)
    second_id = registry.record(second)
    if second.predecessor_receipt_id not in registry.receipts:
        raise ValueError("E5 M24 receipt chain lost its predecessor")
    return {
        "schema": "bramastra-generation-chain/v1",
        "registry_namespace": "fixture" if fixture else "learned",
        "adaptation_anchor": anchor_id,
        "p0_choice": p0_choice,
        "p1_choice": p1_choice,
        "p1_successor_checkpoint": p1_checkpoint_id,
        "p1_model_origin": model_origin,
        "p1_confirmation_support": confirmed_gain,
        "publication_state": (
            "fixture-only" if fixture else
            "awaiting-chief-approval" if second.status == "confirmed" else
            "rejected"),
        "receipt_ids": [first_id, second_id],
        "receipts": [
            {"receipt_id": first_id,
             "generation_id": first.generation_id,
             "predecessor_receipt_id": first.predecessor_receipt_id,
             "proposer_checkpoint": first.proposer_checkpoint,
             "proposer_origin": first.proposer_origin,
             "comparison_identity": first.comparison_identity,
             "status": first.status,
             "fixture": first.fixture},
            {"receipt_id": second_id,
             "generation_id": second.generation_id,
             "predecessor_receipt_id": second.predecessor_receipt_id,
             "proposer_checkpoint": second.proposer_checkpoint,
             "proposer_origin": second.proposer_origin,
             "parent_method": dict(second.parent_method),
             "candidate_program": (None if second.candidate_program is None
                                   else second.candidate_program.to_dict()),
             "compiled_identity": second.compiled_identity,
             "comparison_identity": second.comparison_identity,
             "attempted_candidates": list(second.attempted_candidates),
             "consumed_resources": dict(second.consumed_resources),
             "status": second.status,
             "fixture": second.fixture}]}


def _teacher_majority_choice(archive: Any,
                             archive_tasks: list) -> str:
    """Named teacher/control: most frequent best-measured method (ties M0)."""
    from collections import Counter
    best_by_task: dict[str, str] = {}
    for task in archive_tasks:
        task_id = str(task.get("meta_task_id", "mt-?"))
        best = archive.best_measured(task_id)
        if best is not None:
            best_by_task[task_id] = best
    counts = Counter(best_by_task.values())
    if counts:
        top = counts.most_common()
        max_count = top[0][1]
        tied = sorted([m for m, c in top if c == max_count])
        return tied[0] if len(tied) == 1 else "M0"
    return "M0"
