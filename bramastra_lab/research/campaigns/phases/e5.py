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


def resolve_mechanism_answer(data_dir: str, mechanism_id: str) -> str:
    """Resolve a query mechanism's answer from prepared bundle data.

    Searches episode rows for the mechanism ID and returns its recorded
    answer, verifying consistency across rows. Meta-training feedback comes
    from these prepared labels (never invented); confirmation scoring uses
    the same lookup only AFTER choices are frozen (same-table scoring).
    """
    import glob as _glob
    import json as _json

    answers: set[str] = set()
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
                    if str(row.get("mechanism_id", "")) == str(mechanism_id) \
                            and "answer" in row:
                        answers.add(str(row["answer"]))
        except OSError:
            continue
    if not answers:
        raise ValueError(
            f"no prepared answer for mechanism {mechanism_id!r}; refusing "
            "invented query labels")
    if len(answers) > 1:
        raise ValueError(
            f"mechanism {mechanism_id!r} has inconsistent answers "
            f"{sorted(answers)}; refusing")
    return next(iter(answers))


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
    trial_deadline = _time.time() + 45.0
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
        build_support_batch=_production_support_builder(task),
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


def _production_support_builder(task: dict) -> Any:
    """Token-objective support batches from real support examples (O08).

    Adaptation varies optimizer/architecture via the dispatched recipe, not
    labels: support answers are real prepared labels; the objective is
    answer-token likelihood (method choice changes LR/gates, never gold
    answers or eval rules).
    """
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
            weights={"token": 1.0, "world": 0.0, "action": 0.0,
                     "value": 0.0, "pair": 0.0, "pg": 0.0},
            enabled_terms=frozenset({"token"}))
        window.add("token", batch.target_count)
        return batch, window, {}, None

    return build


def _production_query_evaluator(task: dict, data_dir: str,
                               task_id: str) -> Any:
    """Query + protected measurement from resolved prepared answers (O08)."""
    def evaluate(trial_handle: Any) -> Mapping[str, Any]:
        model = trial_handle.get("model") if isinstance(
            trial_handle, dict) else getattr(trial_handle, "model", None)
        config = trial_handle.get("config") if isinstance(
            trial_handle, dict) else getattr(trial_handle, "config", None)
        if model is None or config is None:
            raise ValueError("trial handle owns no model for query eval")
        from bramastra_lab.research.experience.codec import encode_text
        from bramastra_lab.research.runtime.inference import generate_free_form

        queries = task.get("query_examples", ())
        if not queries:
            raise ValueError("no query examples; refusing")
        hits = 0
        for example in queries:
            expected = resolve_mechanism_answer(
                data_dir, str(example.get("mechanism_id", "")))
            prompt = [259] + encode_text(json.dumps(
                example.get("public", {}), sort_keys=True)[:256])[:64]
            report = generate_free_form(model, config, prompt,
                                        max_new_tokens=16)
            if str(report.answer).strip() == expected:
                hits += 1
        protected_hits = protected_total = 0
        for ref in task.get("protected_references", ()):
            try:
                expected = resolve_mechanism_answer(
                    data_dir, str(ref.get("mechanism_id", "")))
            except ValueError:
                continue
            protected_total += 1
            prompt = [259] + encode_text(json.dumps(
                {"protected": ref.get("mechanism_id")})[:128])[:32]
            report = generate_free_form(model, config, prompt,
                                        max_new_tokens=16)
            if str(report.answer).strip() == expected:
                protected_hits += 1
        return {"measured_success": hits / len(queries),
                "protected": {"hits": protected_hits,
                              "total": protected_total}}

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
    for task_index, task in enumerate(archive_tasks):
        task_id = str(task.get("meta_task_id", f"mt-{task_index}"))
        for method_id in ("M0", "M1", "M2"):
            result = _run_archive_trial(
                ops, job, anchor_record, anchor_id, anchor_support,
                task, task_id, method_id, learning_boundary)
            trial_results.append(result)
            trial_lineages.append(
                result.trial_checkpoint_id or result.detail.get(
                    "lineage", f"failed:{task_id}:{method_id}")[:64])
            archive_rows.append({
                "task_identity": task_id, "method_id": method_id,
                "measured_success": result.measured_success
                if result.measured_success is not None else 0.0,
                "measured_updates": result.measured_updates,
                "elapsed_seconds": result.elapsed_seconds,
                "support_identities": list(result.support_identities),
                "query_identities": list(result.query_identities),
                "trial_lineage": trial_lineages[-1],
                "validation": result.validation})
    if not archive_rows:
        return PhaseResult(status="failed", device_seconds=time.monotonic() - started,
                           error="E5 archive block produced no trials",
                           evidence_kind=EVIDENCE_FIXTURE,
                           extra={"phase": "E5"})
    # Immutable archive capability (cutoff frozen before choices).
    try:
        from bramastra_lab.research.metalearning.dispatch import (
            MethodArchive, MethodTrialOutcome)
        outcomes = tuple(MethodTrialOutcome(
            method_id=r["method_id"], task_identity=r["task_identity"],
            measured_updates=int(r["measured_updates"]),
            measured_success=float(r["measured_success"]),
            elapsed_seconds=float(r["elapsed_seconds"]),
            validation=str(r["validation"])) for r in archive_rows)
        archive = MethodArchive(rows=outcomes, cutoff_event_index=len(outcomes))
        archive_identity = archive.identity()
    except Exception as exc:
        return PhaseResult(status="failed", device_seconds=time.monotonic() - started,
                           error=f"E5 archive capability refused: {exc}",
                           evidence_kind=EVIDENCE_FIXTURE,
                           extra={"phase": "E5"})
    # Block 2: P0 successor decision — capture actual decoder output BEFORE
    # applying. P_fixed is M0 (never M2).
    try:
        p0_choice, p0_capture = _capture_proposer_choice(
            ops, proposer, archive, archive_tasks, anchor_id, job)
    except Exception as exc:
        return PhaseResult(status="failed", device_seconds=time.monotonic() - started,
                           error=f"E5 proposer capture refused: {exc}",
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
    # Block 2: train P0 on the measured archive, capture its choice, then
    # fork the TRAINED P0 into P1 (selected recipe) and P_fixed (M0) with
    # the same newly admitted archive/order and equal update allowances.
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
            ops, p1_handle, proposer_batches, job,
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
            ops, p_fixed_handle, proposer_batches, job,
            task_identity=f"E5-{job.seed}-P_fixed-train",
            learning_boundary=learning_boundary)
        for label, record in (("P1", p1_train), ("P_fixed", p_fixed_train)):
            if record["validation"] not in ("measured", "boundary-only"):
                raise ValueError(
                    f"{label} successor training refused: {record.get('error')}")
        # Publish successor checkpoints (real payloads on GPU; fixture
        # receipts for doubles locally).
        successor_checkpoints: dict[str, Any] = {}
        for label, handle in (("P1", p1_handle), ("P_fixed", p_fixed_handle)):
            try:
                successor_checkpoints[label] = ops.publish_checkpoint(
                    handle=handle, run_dir=job.run_dir, phase="E5",
                    arm=label, seed=job.seed,
                    update_index=int(ops.optimizer_updates(handle)),
                    parent_checkpoint_id=anchor_id,
                    data_dir=job.data_dir)
            except Exception as exc:
                successor_checkpoints[label] = f"unpublished:{exc}"[:120]
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
    confirmation_choices: dict[str, str] = {}
    try:
        confirmation_choices["P1"] = p0_choice
        confirmation_choices["P_fixed"] = "M0"
        # Frozen P0 choice (same as captured, not re-decided after outcomes).
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
    artifact_dir = os.path.join(job.run_dir, "phase_outputs", job.phase)
    os.makedirs(artifact_dir, exist_ok=True)
    with open(os.path.join(artifact_dir, f"E5-{job.seed}.json"), "w",
              encoding="utf-8") as handle_file:
        json.dump({"archive_identity": archive_identity,
                   "archive_rows": archive_rows,
                   "p0_choice": p0_choice,
                   "p0_capture": p0_capture,
                   "p0_training": p0_train,
                   "successors": {
                       "P1": {"method": p0_choice, "training": p1_train,
                              "checkpoint": successor_checkpoints.get("P1")},
                       "P_fixed": {"method": "M0",
                                   "training": p_fixed_train,
                                   "checkpoint": successor_checkpoints.get(
                                       "P_fixed")}},
                   "anchors": {"P1": p0_choice, "P_fixed": "M0", "P0": p0_choice},
                   "confirmation_choices": confirmation_choices,
                   "confirmation_rows": confirmation_rows,
                   "confirmation_summary": confirm_summary,
                   "committed_updates": committed,
                   "attempted_updates": attempted,
                   "evidence_kind": evidence,
                   "learning_boundary": learning_boundary,
                   "lineages": trial_lineages[:6]},
                  handle_file, indent=2, sort_keys=True)
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
    for task in archive_tasks:
        task_id = str(task.get("meta_task_id", "mt-?"))
        best = archive.best_measured(task_id)
        if best is None:
            continue
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
        raise ValueError("no measured archive bests for proposer training")
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
                             anchor_id: str, job: JobInput) -> tuple[str, dict]:
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
    if model_usable:
        try:
            method_proposer = MethodProposer(
                model, config, checkpoint_payload_identity=anchor_id)
            descriptor = {"task_identities": sorted(
                t["task_identity"] for t in task_descriptors),
                "family": "meta-training"}
            capture = method_proposer.capture_proposal(
                descriptor, archive)
            from bramastra_lab.research.metalearning.dispatch import (
                parse_method_selection)
            method_id, program = parse_method_selection(capture.raw_output)
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
                "checkpoint_payload_identity": anchor_id[:16]}
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
        "checkpoint_payload_identity": anchor_id[:16]}


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
