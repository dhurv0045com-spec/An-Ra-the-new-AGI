"""Pure V8 multi-phase Kaggle 2xT4 campaign orchestrator with admission, 2-lane scheduling, and verified export."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
import inspect
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from anra_v5.elendril.codec import digest
from anra_v5.elendril.v8.analysis import (
    build_cost_views,
    build_v8_report_payload,
    evaluate_advancement_gate,
    load_v717_archive_baseline,
    pair_condition_runs,
    paired_world_comparison,
    updated_scenario_ledger,
)
from anra_v5.elendril.v8.contracts import EvaluationOutcome, POLICY_MODES
from anra_v5.elendril.v8.data import (
    D1_FAMILIES,
    build_d1_dev_suite,
    build_d1_training_dataset,
    load_d0_dataset,
)
from anra_v5.elendril.v8.evaluation import EpisodeRecord
from anra_v5.elendril.v8.holdout import audit_split_disjointness
from anra_v5.elendril.v8.protocol import (
    EVALUATION_ALLOCATION,
    OPTIMIZER_DEFAULTS,
    PHASES,
    jobs as protocol_jobs,
)
from anra_v5.elendril.v8.teacher import audit_teacher_samples, compute_shortcut_baselines
from v5_training.checkpoint import CheckpointStore

from .elendril_v8_export import export_v8_archives, verify_v8_export
from .elendril_v8_worker import (
    prepare_verified_splits,
    rescore_saved_episode_jsonl,
    run_canary,
    run_profile,
)

DEFAULT_SAFETY_RESERVE_SECONDS = 600.0
DEFAULT_EXPORT_RESERVE_SECONDS = 180.0
DEFAULT_WORK_MARGIN = 2.0

REGISTERED_DATASET_SAMPLES = {
    "D0": 9472,
    "D1": 5888,
}
D0_FAMILIES = ("conflict", "known", "memory", "missing", "null", "route", "unavailable")
EXPECTED_COHORTS = tuple(EVALUATION_ALLOCATION["primary_final_cohorts"])
EXPECTED_MODES = tuple(POLICY_MODES)


def _is_strict_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def expected_updates_for_job(
    job_spec: dict[str, object],
    *,
    cpu_smoke: bool = False,
    updates_override: int | None = None,
) -> int:
    """Compute exact registered update count independently of self-reported job JSON."""

    if cpu_smoke:
        if updates_override is None:
            raise ValueError("cpu_smoke validation requires explicit updates_override")
        if not _is_strict_int(updates_override) or updates_override <= 0:
            raise ValueError(f"invalid updates_override: {updates_override!r}")
        return int(updates_override)

    dataset = str(job_spec["dataset"])
    if dataset not in REGISTERED_DATASET_SAMPLES:
        raise ValueError(f"unregistered dataset in job_spec: {dataset!r}")
    sample_count = REGISTERED_DATASET_SAMPLES[dataset]
    batch_size = int(job_spec.get("batch_size", OPTIMIZER_DEFAULTS["batch_size"]))
    epochs = int(job_spec.get("epochs", OPTIMIZER_DEFAULTS["epochs"]))
    if batch_size <= 0:
        raise ValueError(f"invalid batch_size in job_spec: {batch_size!r}")
    eff_batch = min(batch_size, sample_count)
    if epochs <= 0:
        return 1
    return max(1, math.ceil(sample_count / eff_batch) * epochs)


def _expected_milestones_for_updates(expected_updates: int) -> dict[str, int]:
    epoch_len = max(expected_updates // 4, 1)
    by_update = {
        epoch_len: "epoch-1",
        2 * epoch_len: "epoch-2",
        expected_updates: "epoch-4",
    }
    expected: dict[str, int] = {"initial": 0}
    for u_val in sorted(by_update):
        expected[by_update[u_val]] = u_val
    return expected


def _expected_families_for_dataset(dataset: str) -> set[str]:
    if dataset == "D1":
        return set(D1_FAMILIES)
    if dataset == "D0":
        return set(D0_FAMILIES)
    raise ValueError(f"unregistered dataset: {dataset!r}")


def _load_registered_cohort_worlds(
    cohorts_path: Path,
    *,
    job_id: str,
    label: str,
    violations: list[str],
) -> tuple[dict[str, list[dict[str, Any]]] | None, set[str] | None]:
    """Load and validate registered Dev/Confirmation cohort worlds from splits/*.json."""

    if not cohorts_path.exists() or not cohorts_path.is_file() or cohorts_path.stat().st_size == 0:
        violations.append(f"{job_id}: missing splits/{cohorts_path.name} for {label} validation")
        return None, None
    try:
        parsed = json.loads(cohorts_path.read_text(encoding="utf-8"))
    except Exception as exc:
        violations.append(f"{job_id}: unreadable splits/{cohorts_path.name} ({exc})")
        return None, None
    if not isinstance(parsed, dict) or not set(EXPECTED_COHORTS).issubset(set(parsed)):
        violations.append(f"{job_id}: splits/{cohorts_path.name} missing required cohorts")
        return None, None

    world_ids: set[str] = set()
    for cohort_name, worlds in parsed.items():
        if not isinstance(worlds, list):
            violations.append(f"{job_id}: splits/{cohorts_path.name} cohort {cohort_name!r} is not a list")
            return None, None
        for w in worlds:
            if not isinstance(w, dict) or w.get("world_id") is None:
                violations.append(f"{job_id}: splits/{cohorts_path.name} contains invalid world entry")
                return None, None
            world_ids.add(str(w["world_id"]))

    if not world_ids:
        violations.append(f"{job_id}: splits/{cohorts_path.name} has 0 worlds")
        return None, None
    return parsed, world_ids


def _validate_rescore_summary_fields(
    *,
    job_id: str,
    prefix: str,
    rep_rescore: object,
    actual_rescore: dict[str, Any],
    violations: list[str],
) -> None:
    if not isinstance(rep_rescore, dict) or rep_rescore.get("verified") is not True:
        violations.append(f"{job_id}: missing or unverified {prefix} in report")
        return
    if rep_rescore.get("episode_count") != actual_rescore.get("episode_count"):
        violations.append(
            f"{job_id}: {prefix} episode_count mismatch {rep_rescore.get('episode_count')!r} != {actual_rescore.get('episode_count')!r}"
        )
    if (
        rep_rescore.get("exact_accuracy") is None
        or abs(float(rep_rescore["exact_accuracy"]) - float(actual_rescore["exact_accuracy"])) > 1e-6
    ):
        violations.append(f"{job_id}: {prefix} exact_accuracy mismatch")
    if rep_rescore.get("by_mode") != actual_rescore.get("by_mode"):
        violations.append(f"{job_id}: {prefix} by_mode mismatch")
    if rep_rescore.get("by_cohort") != actual_rescore.get("by_cohort"):
        violations.append(f"{job_id}: {prefix} by_cohort mismatch")
    if (
        isinstance(actual_rescore.get("by_milestone"), dict)
        and isinstance(rep_rescore.get("by_milestone"), dict)
        and rep_rescore["by_milestone"] != actual_rescore["by_milestone"]
    ):
        violations.append(f"{job_id}: {prefix} by_milestone mismatch")


def validate_job_completion_artifacts(
    job_spec: dict[str, object],
    report: dict[str, object],
    work_dir: Path,
    *,
    source_commit: str,
    expected_updates: int,
    expected_split_audit_sha256: str | None = None,
    check_confirmation: bool = False,
    cpu_smoke: bool | None = None,
) -> list[str]:
    """Fail-closed shared validator for worker and test-runner job completion evidence."""

    violations: list[str] = []
    work_dir = Path(work_dir)
    job_id = str(job_spec.get("job_id", ""))
    arm = str(job_spec.get("arm", ""))
    dataset = str(job_spec.get("dataset", ""))
    expected_cond = f"{arm}-{dataset}"

    if not _is_strict_int(expected_updates) or expected_updates <= 0:
        return [f"{job_id}: expected_updates must be positive int, got {expected_updates!r}"]
    if not isinstance(report, dict):
        return [f"{job_id}: report must be a JSON object"]

    if report.get("job_id") != job_id:
        violations.append(f"{job_id}: job_id mismatch {report.get('job_id')!r} != {job_id!r}")
    if report.get("arm") != arm:
        violations.append(f"{job_id}: arm mismatch {report.get('arm')!r} != {arm!r}")
    if report.get("dataset") != dataset:
        violations.append(f"{job_id}: dataset mismatch {report.get('dataset')!r} != {dataset!r}")
    if report.get("condition_id") != expected_cond:
        violations.append(f"{job_id}: condition_id mismatch {report.get('condition_id')!r} != {expected_cond!r}")

    exp_init = int(job_spec["init_seed"])
    exp_order = int(job_spec["order_seed"])
    if not _is_strict_int(report.get("init_seed")) or report.get("init_seed") != exp_init:
        violations.append(f"{job_id}: init_seed mismatch {report.get('init_seed')!r} != {exp_init!r}")
    if not _is_strict_int(report.get("order_seed")) or report.get("order_seed") != exp_order:
        violations.append(f"{job_id}: order_seed mismatch {report.get('order_seed')!r} != {exp_order!r}")
    if report.get("source_commit") != source_commit:
        violations.append(f"{job_id}: source_commit mismatch {report.get('source_commit')!r} != {source_commit!r}")

    comp_u = report.get("completed_updates")
    targ_u = report.get("target_updates")
    if (
        not _is_strict_int(comp_u)
        or not _is_strict_int(targ_u)
        or comp_u != expected_updates
        or targ_u != expected_updates
    ):
        violations.append(
            f"{job_id}: update count mismatch completed={comp_u!r}, target={targ_u!r}, expected={expected_updates}"
        )

    # Split audit binding
    audit_path = work_dir / "splits" / "SPLIT_AUDIT.json"
    audit_obj: dict[str, Any] | None = None
    if audit_path.exists():
        try:
            parsed_audit = json.loads(audit_path.read_text(encoding="utf-8"))
            if isinstance(parsed_audit, dict):
                audit_obj = parsed_audit
        except Exception as exc:
            violations.append(f"{job_id}: unreadable SPLIT_AUDIT.json ({exc})")
    if expected_split_audit_sha256 is not None:
        if report.get("split_audit_sha256") != expected_split_audit_sha256:
            violations.append(
                f"{job_id}: split_audit_sha256 mismatch {report.get('split_audit_sha256')!r} != {expected_split_audit_sha256!r}"
            )
        if audit_obj is None:
            violations.append(f"{job_id}: missing splits/SPLIT_AUDIT.json")
        else:
            if audit_obj.get("audit_passed") is not True:
                violations.append(f"{job_id}: SPLIT_AUDIT.json audit_passed is not True")
            if audit_obj.get("split_audit_sha256") != expected_split_audit_sha256:
                violations.append(f"{job_id}: SPLIT_AUDIT.json sha256 mismatch")

    # Checkpoint validation
    ckpt_root = work_dir / "checkpoints"
    ckpt_dir = ckpt_root / job_id
    ckpt_state = None
    latest_sha: str | None = None
    if not ckpt_dir.exists() or not ckpt_dir.is_dir():
        violations.append(f"{job_id}: missing checkpoint directory {ckpt_dir}")
    else:
        try:
            store = CheckpointStore(ckpt_root, job_id)
            latest_sha = store.latest_sha256()
            if latest_sha is None:
                violations.append(f"{job_id}: checkpoint store has no committed LATEST")
            else:
                if report.get("checkpoint_sha256") != latest_sha:
                    violations.append(
                        f"{job_id}: checkpoint_sha256 mismatch report={report.get('checkpoint_sha256')!r} != store={latest_sha!r}"
                    )
                ckpt_state, _ = store.restore(latest_sha)
                if ckpt_state.lineage_id != job_id:
                    violations.append(f"{job_id}: checkpoint lineage_id {ckpt_state.lineage_id!r} != {job_id!r}")
                if ckpt_state.global_update != expected_updates:
                    violations.append(
                        f"{job_id}: checkpoint global_update {ckpt_state.global_update} != {expected_updates}"
                    )
                if ckpt_state.identities.source_commit != source_commit:
                    violations.append(f"{job_id}: checkpoint source_commit mismatch")
                if "checkpoint_generation" in report and report["checkpoint_generation"] is not None:
                    rep_gen = report["checkpoint_generation"]
                    if not _is_strict_int(rep_gen) or ckpt_state.generation != rep_gen:
                        violations.append(
                            f"{job_id}: checkpoint_generation mismatch report={rep_gen!r} != store={ckpt_state.generation}"
                        )
        except Exception as exc:
            violations.append(f"{job_id}: checkpoint restore failed ({exc})")

    # Steps history validation
    steps_path = work_dir / "steps" / f"{job_id}.jsonl"
    if not steps_path.exists() or not steps_path.is_file():
        violations.append(f"{job_id}: missing steps file {steps_path}")
    else:
        try:
            seen_updates: list[int] = []
            for line_no, raw_line in enumerate(steps_path.read_text(encoding="utf-8").splitlines(), start=1):
                line = raw_line.strip()
                if not line:
                    continue
                row = json.loads(line)
                u_val = row.get("global_update")
                if not _is_strict_int(u_val):
                    violations.append(f"{job_id}: steps line {line_no} invalid global_update {u_val!r}")
                    break
                seen_updates.append(u_val)
            expected_seq = list(range(1, expected_updates + 1))
            if seen_updates != expected_seq:
                violations.append(
                    f"{job_id}: steps global_update sequence invalid (count={len(seen_updates)}, expected=1..{expected_updates})"
                )
        except Exception as exc:
            violations.append(f"{job_id}: failed reading steps JSONL ({exc})")

    is_cpu_smoke = (
        bool(cpu_smoke)
        if cpu_smoke is not None
        else (bool(audit_obj.get("cpu_smoke")) if isinstance(audit_obj, dict) and "cpu_smoke" in audit_obj else False)
    )

    # Families & milestones validation
    try:
        expected_families = _expected_families_for_dataset(dataset)
    except ValueError as exc:
        violations.append(f"{job_id}: {exc}")
        expected_families = set()

    expected_milestones = _expected_milestones_for_updates(expected_updates)
    rep_milestones = report.get("milestones")
    if not isinstance(rep_milestones, dict):
        violations.append(f"{job_id}: report['milestones'] must be a dict")
        rep_milestones = {}

    milestones_dir = work_dir / "milestones" / job_id
    loaded_milestones: dict[str, dict[str, Any]] = {}
    for m_name, exp_u in expected_milestones.items():
        m_file = milestones_dir / f"{m_name}.json"
        if not m_file.exists() or not m_file.is_file():
            violations.append(f"{job_id}: missing milestone file {m_file}")
            continue
        try:
            m_disk = json.loads(m_file.read_text(encoding="utf-8"))
        except Exception as exc:
            violations.append(f"{job_id}: unreadable milestone {m_name}: {exc}")
            continue
        if not isinstance(m_disk, dict):
            violations.append(f"{job_id}: milestone {m_name} is not a JSON object")
            continue
        loaded_milestones[m_name] = m_disk
        if m_disk.get("milestone_name", m_name) != m_name:
            violations.append(f"{job_id}: milestone {m_name} name mismatch {m_disk.get('milestone_name')!r}")
        if not _is_strict_int(m_disk.get("global_update")) or m_disk["global_update"] != exp_u:
            violations.append(
                f"{job_id}: milestone {m_name} global_update {m_disk.get('global_update')!r} != {exp_u}"
            )

        suite_sum = m_disk.get("suite_summary")
        if not isinstance(suite_sum, dict) or not _is_strict_int(suite_sum.get("episodes")) or suite_sum["episodes"] <= 0:
            violations.append(f"{job_id}: milestone {m_name} missing valid non-null suite_summary")
        elif suite_sum.get("exact_episode_accuracy") is None:
            violations.append(f"{job_id}: milestone {m_name} suite_summary missing exact_episode_accuracy")

        m_probe = m_disk.get("probe")
        if not isinstance(m_probe, dict):
            violations.append(f"{job_id}: milestone {m_name} missing probe dict")
        else:
            m_fams = set(m_probe.get("families_covered") or ())
            if expected_families and not expected_families.issubset(m_fams):
                violations.append(
                    f"{job_id}: milestone {m_name} missing families {sorted(expected_families - m_fams)}"
                )

        if exp_u == 0:
            if m_disk.get("checkpoint_generation") != 0 or m_disk.get("checkpoint_sha256") is not None:
                violations.append(f"{job_id}: initial milestone checkpoint binding invalid")
        else:
            if not _is_sha256(m_disk.get("checkpoint_sha256")):
                violations.append(f"{job_id}: milestone {m_name} missing valid checkpoint_sha256")
            if not _is_strict_int(m_disk.get("checkpoint_generation")) or m_disk["checkpoint_generation"] <= 0:
                violations.append(f"{job_id}: milestone {m_name} invalid checkpoint_generation")
            if m_name == "epoch-4":
                if latest_sha is not None and m_disk.get("checkpoint_sha256") != latest_sha:
                    violations.append(f"{job_id}: epoch-4 checkpoint_sha256 != latest checkpoint")
                if ckpt_state is not None and m_disk.get("checkpoint_generation") != ckpt_state.generation:
                    violations.append(f"{job_id}: epoch-4 checkpoint_generation != checkpoint state generation")

        rep_m = rep_milestones.get(m_name)
        if not isinstance(rep_m, dict):
            violations.append(f"{job_id}: report['milestones'] missing {m_name!r}")
        else:
            if rep_m.get("global_update") != m_disk.get("global_update"):
                violations.append(f"{job_id}: report milestone {m_name} global_update != disk")
            if rep_m.get("checkpoint_sha256") != m_disk.get("checkpoint_sha256"):
                violations.append(f"{job_id}: report milestone {m_name} checkpoint_sha256 != disk")
            if rep_m.get("checkpoint_generation") != m_disk.get("checkpoint_generation"):
                violations.append(f"{job_id}: report milestone {m_name} checkpoint_generation != disk")

    # final_probe family coverage + full corpus denominator on non-smoke runs
    final_probe = report.get("final_probe")
    if not isinstance(final_probe, dict):
        violations.append(f"{job_id}: missing final_probe")
    else:
        fp_fams = set(final_probe.get("families_covered") or ())
        if expected_families and not expected_families.issubset(fp_fams):
            violations.append(f"{job_id}: final_probe missing families {sorted(expected_families - fp_fams)}")
        if not is_cpu_smoke and dataset in REGISTERED_DATASET_SAMPLES:
            expected_probe_steps = REGISTERED_DATASET_SAMPLES[dataset]
            fp_total_steps = final_probe.get("total_steps")
            if not _is_strict_int(fp_total_steps) or fp_total_steps != expected_probe_steps:
                violations.append(
                    f"{job_id}: final_probe total_steps {fp_total_steps!r} != {expected_probe_steps}"
                )

    # Load registered Dev worlds for exact world x mode x milestone validation
    dev_cohorts, expected_dev_world_ids = _load_registered_cohort_worlds(
        work_dir / "splits" / "dev_cohorts.json",
        job_id=job_id,
        label="dev",
        violations=violations,
    )
    expected_sub_world_ids: set[str] | None = None
    if dev_cohorts is not None:
        sub_cap = 5 * int(EVALUATION_ALLOCATION["intermediate_milestone_base_worlds"])
        expected_sub_world_ids = {
            str(w["world_id"])
            for rows in dev_cohorts.values()
            for w in rows[:sub_cap]
        }

    # Episodes JSONL + milestone coverage + independent rescore validation
    ep_path = work_dir / "episodes" / f"{job_id}.jsonl"
    if report.get("episodes_file") is not None:
        try:
            if Path(str(report["episodes_file"])).resolve() != ep_path.resolve():
                violations.append(f"{job_id}: episodes_file {report['episodes_file']!r} != {ep_path}")
        except Exception:
            violations.append(f"{job_id}: invalid episodes_file path {report['episodes_file']!r}")
    if not ep_path.exists() or not ep_path.is_file() or ep_path.stat().st_size == 0:
        violations.append(f"{job_id}: missing or empty episodes file {ep_path}")
    else:
        observed_pairs_by_milestone: dict[str, set[tuple[str, str]]] = defaultdict(set)
        duplicates_by_milestone: Counter[str] = Counter()
        foreign_worlds_by_milestone: dict[str, set[str]] = defaultdict(set)
        milestone_totals: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "correct": 0})
        milestone_mode_stats: dict[str, dict[str, dict[str, int]]] = defaultdict(
            lambda: defaultdict(lambda: {"n": 0, "correct": 0})
        )
        empty_steps_rows = 0
        saw_explicit_milestone = False
        saw_missing_milestone = False
        invalid_milestone_rows = 0
        parse_ok = True

        try:
            for line in ep_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    violations.append(f"{job_id}: non-object row in episodes JSONL")
                    parse_ok = False
                    break
                if "milestone" in row:
                    saw_explicit_milestone = True
                    raw_m = row["milestone"]
                    if not isinstance(raw_m, str) or not raw_m.strip():
                        invalid_milestone_rows += 1
                        continue
                    m_label = raw_m
                else:
                    saw_missing_milestone = True
                    m_label = "epoch-4"

                wid = str(row.get("world_id", ""))
                mode = str(row.get("mode", ""))
                steps_list = row.get("steps")
                if not isinstance(steps_list, list) or len(steps_list) == 0:
                    empty_steps_rows += 1
                if expected_dev_world_ids is not None and wid not in expected_dev_world_ids:
                    foreign_worlds_by_milestone[m_label].add(wid)

                pair = (wid, mode)
                if pair in observed_pairs_by_milestone[m_label]:
                    duplicates_by_milestone[m_label] += 1
                observed_pairs_by_milestone[m_label].add(pair)

                is_ok = bool(row.get("exact_match", row.get("correct", False)))
                milestone_totals[m_label]["n"] += 1
                milestone_totals[m_label]["correct"] += int(is_ok)
                milestone_mode_stats[m_label][mode]["n"] += 1
                milestone_mode_stats[m_label][mode]["correct"] += int(is_ok)
        except Exception as exc:
            violations.append(f"{job_id}: failed parsing episodes JSONL ({exc})")
            parse_ok = False

        if invalid_milestone_rows > 0:
            violations.append(f"{job_id}: {invalid_milestone_rows} episode rows have invalid milestone tag")
        if saw_explicit_milestone and saw_missing_milestone:
            violations.append(f"{job_id}: mixed rows with and without 'milestone' key in {ep_path.name}")
        if empty_steps_rows > 0:
            violations.append(f"{job_id}: {empty_steps_rows} episode rows have empty or missing steps")

        if parse_ok:
            extra_milestones = sorted(set(milestone_totals) - set(expected_milestones))
            if extra_milestones:
                violations.append(f"{job_id}: unexpected milestones in episodes JSONL: {extra_milestones}")

            for m_name in expected_milestones:
                dup_m = duplicates_by_milestone.get(m_name, 0)
                if dup_m > 0:
                    violations.append(f"{job_id}: {dup_m} duplicate (world_id, mode) rows at milestone {m_name}")
                foreign_m = foreign_worlds_by_milestone.get(m_name)
                if foreign_m:
                    violations.append(
                        f"{job_id}: {len(foreign_m)} foreign world_id(s) at milestone {m_name}: {sorted(foreign_m)[:3]}"
                    )

                if expected_dev_world_ids is not None:
                    if m_name == "epoch-4":
                        exp_pairs = {(wid, mode) for wid in expected_dev_world_ids for mode in EXPECTED_MODES}
                    elif m_name == "initial":
                        exp_pairs = {(wid, "raw") for wid in expected_dev_world_ids}
                    else:
                        sub_ids = expected_sub_world_ids if expected_sub_world_ids is not None else expected_dev_world_ids
                        exp_pairs = {(wid, "raw") for wid in sub_ids}

                    obs_pairs = observed_pairs_by_milestone.get(m_name, set())
                    missing_pairs = exp_pairs - obs_pairs
                    extra_pairs = obs_pairs - exp_pairs
                    if missing_pairs:
                        violations.append(
                            f"{job_id}: milestone {m_name} missing {len(missing_pairs)} (world_id, mode) pair(s), sample={sorted(missing_pairs)[:3]}"
                        )
                    if extra_pairs:
                        violations.append(
                            f"{job_id}: milestone {m_name} has {len(extra_pairs)} unexpected (world_id, mode) pair(s)"
                        )
                    if milestone_totals.get(m_name, {}).get("n", 0) != len(exp_pairs):
                        violations.append(
                            f"{job_id}: milestone {m_name} row count {milestone_totals.get(m_name, {}).get('n', 0)} != {len(exp_pairs)}"
                        )

        rescore = rescore_saved_episode_jsonl(ep_path)
        if rescore.get("verified") is not True:
            violations.append(f"{job_id}: rescore_saved_episode_jsonl failed: {rescore.get('errors', [])[:3]}")
        else:
            _validate_rescore_summary_fields(
                job_id=job_id,
                prefix="rescore_summary",
                rep_rescore=report.get("rescore_summary"),
                actual_rescore=rescore,
                violations=violations,
            )

            if isinstance(rescore.get("by_milestone"), dict):
                for m_name in expected_milestones:
                    if m_name not in rescore["by_milestone"]:
                        violations.append(f"{job_id}: rescore by_milestone missing {m_name!r}")
                    else:
                        bm = rescore["by_milestone"][m_name]
                        if isinstance(bm, dict):
                            bm_count = bm.get("episode_count", bm.get("n"))
                            m_tot = milestone_totals.get(m_name, {"n": 0, "correct": 0})
                            if _is_strict_int(bm_count) and m_tot["n"] and bm_count != m_tot["n"]:
                                violations.append(
                                    f"{job_id}: rescore by_milestone[{m_name!r}] count {bm_count} != {m_tot['n']}"
                                )
                            if bm.get("exact_accuracy") is not None and m_tot["n"] > 0:
                                exp_m_acc = m_tot["correct"] / m_tot["n"]
                                if abs(float(bm["exact_accuracy"]) - exp_m_acc) > 1e-6:
                                    violations.append(
                                        f"{job_id}: rescore by_milestone[{m_name!r}] exact_accuracy mismatch"
                                    )
                            raw_st = milestone_mode_stats.get(m_name, {}).get("raw", {"n": 0, "correct": 0})
                            if bm.get("raw_exact_accuracy") is not None and raw_st["n"] > 0:
                                exp_raw_acc = raw_st["correct"] / raw_st["n"]
                                if abs(float(bm["raw_exact_accuracy"]) - exp_raw_acc) > 1e-6:
                                    violations.append(
                                        f"{job_id}: rescore by_milestone[{m_name!r}] raw_exact_accuracy mismatch"
                                    )

            for m_name, m_disk in loaded_milestones.items():
                suite_sum = m_disk.get("suite_summary")
                raw_st = milestone_mode_stats.get(m_name, {}).get("raw", {"n": 0, "correct": 0})
                if isinstance(suite_sum, dict) and raw_st["n"] > 0:
                    if suite_sum.get("episodes") != raw_st["n"]:
                        violations.append(
                            f"{job_id}: milestone {m_name} suite_summary episodes {suite_sum.get('episodes')!r} != {raw_st['n']}"
                        )
                    if suite_sum.get("exact_episode_accuracy") is not None:
                        rescored_m_raw_acc = raw_st["correct"] / raw_st["n"]
                        if abs(float(suite_sum["exact_episode_accuracy"]) - rescored_m_raw_acc) > 1e-6:
                            violations.append(
                                f"{job_id}: milestone {m_name} suite_summary exact_episode_accuracy {suite_sum['exact_episode_accuracy']!r} != rescored {rescored_m_raw_acc!r}"
                            )

            if isinstance(report.get("summary"), dict) and report["summary"].get("exact_accuracy") is not None:
                if abs(float(report["summary"]["exact_accuracy"]) - float(rescore["exact_accuracy"])) > 1e-6:
                    violations.append(f"{job_id}: report['summary']['exact_accuracy'] != rescored exact_accuracy")

            e4_raw_st = milestone_mode_stats.get("epoch-4", {}).get("raw", {"n": 0, "correct": 0})
            if e4_raw_st["n"] > 0:
                e4_raw_acc = e4_raw_st["correct"] / e4_raw_st["n"]
                if report.get("raw_overall_accuracy") is not None:
                    if abs(float(report["raw_overall_accuracy"]) - e4_raw_acc) > 1e-6:
                        violations.append(
                            f"{job_id}: raw_overall_accuracy {report['raw_overall_accuracy']!r} != rescored {e4_raw_acc!r}"
                        )
                rep_sbm = report.get("summary_by_mode")
                if isinstance(rep_sbm, dict):
                    for mode in EXPECTED_MODES:
                        ms = rep_sbm.get(mode)
                        m_st = milestone_mode_stats.get("epoch-4", {}).get(mode, {"n": 0, "correct": 0})
                        if not isinstance(ms, dict):
                            violations.append(f"{job_id}: summary_by_mode missing {mode!r}")
                        elif m_st["n"] > 0:
                            if ms.get("episodes") != m_st["n"]:
                                violations.append(
                                    f"{job_id}: summary_by_mode[{mode!r}] episodes {ms.get('episodes')!r} != {m_st['n']}"
                                )
                            m_acc = m_st["correct"] / m_st["n"]
                            if ms.get("exact_episode_accuracy") is None or abs(float(ms["exact_episode_accuracy"]) - m_acc) > 1e-6:
                                violations.append(
                                    f"{job_id}: summary_by_mode[{mode!r}] exact_episode_accuracy mismatch"
                                )

    # Confirmation evidence validation for replication (always strict & required)
    if check_confirmation:
        _, expected_conf_world_ids = _load_registered_cohort_worlds(
            work_dir / "splits" / "conf_cohorts.json",
            job_id=job_id,
            label="confirmation",
            violations=violations,
        )

        conf_sum = report.get("confirmation_summary")
        if not isinstance(conf_sum, dict):
            violations.append(f"{job_id}: missing confirmation_summary")
        else:
            if not _is_strict_int(conf_sum.get("episodes")) or conf_sum["episodes"] <= 0:
                violations.append(f"{job_id}: confirmation_summary missing positive episodes count")
            if conf_sum.get("exact_episode_accuracy") is None:
                violations.append(f"{job_id}: confirmation_summary missing exact_episode_accuracy")
            conf_cohorts_map = conf_sum.get("by_cohort")
            if not isinstance(conf_cohorts_map, dict) or not set(EXPECTED_COHORTS).issubset(set(conf_cohorts_map)):
                violations.append(f"{job_id}: confirmation_summary missing required cohorts")
            conf_lcf = conf_sum.get("linked_counterfactuals")
            if (
                not isinstance(conf_lcf, dict)
                or conf_lcf.get("relevant_cf_both_correct") is None
                or conf_lcf.get("irrelevant_cf_stability") is None
            ):
                violations.append(f"{job_id}: confirmation_summary missing linked_counterfactuals")
            if expected_conf_world_ids is not None and conf_sum.get("episodes") != len(expected_conf_world_ids):
                violations.append(
                    f"{job_id}: confirmation episodes {conf_sum.get('episodes')!r} != {len(expected_conf_world_ids)}"
                )

        conf_ep_path = work_dir / "episodes" / f"{job_id}.confirmation.jsonl"
        if report.get("confirmation_episodes_file") is not None:
            try:
                if Path(str(report["confirmation_episodes_file"])).resolve() != conf_ep_path.resolve():
                    violations.append(
                        f"{job_id}: confirmation_episodes_file {report['confirmation_episodes_file']!r} != {conf_ep_path}"
                    )
            except Exception:
                violations.append(
                    f"{job_id}: invalid confirmation_episodes_file path {report['confirmation_episodes_file']!r}"
                )

        if not conf_ep_path.exists() or not conf_ep_path.is_file() or conf_ep_path.stat().st_size == 0:
            violations.append(f"{job_id}: missing or empty confirmation episodes file {conf_ep_path}")
        else:
            conf_observed_pairs: set[tuple[str, str]] = set()
            conf_duplicates = 0
            conf_foreign_worlds: set[str] = set()
            conf_empty_steps = 0
            conf_raw_cohort_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "correct": 0})
            conf_parse_ok = True
            try:
                for line in conf_ep_path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        violations.append(f"{job_id}: non-object row in confirmation episodes JSONL")
                        conf_parse_ok = False
                        break
                    wid = str(row.get("world_id", ""))
                    mode = str(row.get("mode", ""))
                    cohort = str(row.get("cohort", ""))
                    steps_list = row.get("steps")
                    if not isinstance(steps_list, list) or len(steps_list) == 0:
                        conf_empty_steps += 1
                    if expected_conf_world_ids is not None and wid not in expected_conf_world_ids:
                        conf_foreign_worlds.add(wid)
                    pair = (wid, mode)
                    if pair in conf_observed_pairs:
                        conf_duplicates += 1
                    conf_observed_pairs.add(pair)
                    if mode == "raw":
                        is_ok = bool(row.get("exact_match", row.get("correct", False)))
                        conf_raw_cohort_stats[cohort]["n"] += 1
                        conf_raw_cohort_stats[cohort]["correct"] += int(is_ok)
            except Exception as exc:
                violations.append(f"{job_id}: failed parsing confirmation episodes JSONL ({exc})")
                conf_parse_ok = False

            if conf_empty_steps > 0:
                violations.append(f"{job_id}: {conf_empty_steps} confirmation episode rows have empty or missing steps")
            if conf_duplicates > 0:
                violations.append(f"{job_id}: {conf_duplicates} duplicate (world_id, mode) rows in confirmation episodes")
            if conf_foreign_worlds:
                violations.append(
                    f"{job_id}: {len(conf_foreign_worlds)} foreign world_id(s) in confirmation episodes: {sorted(conf_foreign_worlds)[:3]}"
                )
            if conf_parse_ok and expected_conf_world_ids is not None:
                expected_conf_pairs = {(wid, mode) for wid in expected_conf_world_ids for mode in EXPECTED_MODES}
                missing_conf = expected_conf_pairs - conf_observed_pairs
                extra_conf = conf_observed_pairs - expected_conf_pairs
                if missing_conf:
                    violations.append(
                        f"{job_id}: confirmation episodes missing {len(missing_conf)} (world_id, mode) pair(s), sample={sorted(missing_conf)[:3]}"
                    )
                if extra_conf:
                    violations.append(
                        f"{job_id}: confirmation episodes has {len(extra_conf)} unexpected (world_id, mode) pair(s)"
                    )

            conf_rescore = rescore_saved_episode_jsonl(conf_ep_path)
            if conf_rescore.get("verified") is not True:
                violations.append(
                    f"{job_id}: confirmation episode rescore failed: {conf_rescore.get('errors', [])[:3]}"
                )
            else:
                if expected_conf_world_ids is not None:
                    exp_conf_rows = len(expected_conf_world_ids) * len(EXPECTED_MODES)
                    if conf_rescore.get("episode_count") != exp_conf_rows:
                        violations.append(
                            f"{job_id}: confirmation rescore episode_count {conf_rescore.get('episode_count')!r} != {exp_conf_rows}"
                        )
                if isinstance(report.get("confirmation_rescore_summary"), dict):
                    _validate_rescore_summary_fields(
                        job_id=job_id,
                        prefix="confirmation_rescore_summary",
                        rep_rescore=report.get("confirmation_rescore_summary"),
                        actual_rescore=conf_rescore,
                        violations=violations,
                    )
                if isinstance(conf_sum, dict):
                    conf_raw_acc = conf_rescore.get("raw_exact_accuracy")
                    if conf_raw_acc is None and isinstance(conf_rescore.get("by_mode"), dict):
                        conf_raw_acc = (conf_rescore["by_mode"].get("raw") or {}).get("exact_accuracy")
                    if (
                        conf_raw_acc is None
                        or conf_sum.get("exact_episode_accuracy") is None
                        or abs(float(conf_sum["exact_episode_accuracy"]) - float(conf_raw_acc)) > 1e-6
                    ):
                        violations.append(f"{job_id}: confirmation_summary exact_episode_accuracy mismatch with rescore")
                    if (
                        conf_sum.get("non_isolated_exact_accuracy") is not None
                        and conf_rescore.get("raw_non_isolated_exact_accuracy") is not None
                        and abs(
                            float(conf_sum["non_isolated_exact_accuracy"])
                            - float(conf_rescore["raw_non_isolated_exact_accuracy"])
                        )
                        > 1e-6
                    ):
                        violations.append(f"{job_id}: confirmation_summary non_isolated_exact_accuracy mismatch with rescore")
                    conf_cohorts_map = conf_sum.get("by_cohort")
                    if isinstance(conf_cohorts_map, dict):
                        for cohort in EXPECTED_COHORTS:
                            c_rep = conf_cohorts_map.get(cohort)
                            c_st = conf_raw_cohort_stats.get(cohort, {"n": 0, "correct": 0})
                            if not isinstance(c_rep, dict):
                                violations.append(f"{job_id}: confirmation_summary missing cohort {cohort!r}")
                            elif c_st["n"] > 0:
                                if c_rep.get("n") != c_st["n"]:
                                    violations.append(
                                        f"{job_id}: confirmation_summary cohort {cohort} n {c_rep.get('n')!r} != {c_st['n']}"
                                    )
                                exp_c_acc = c_st["correct"] / c_st["n"]
                                if c_rep.get("exact_accuracy") is None or abs(float(c_rep["exact_accuracy"]) - exp_c_acc) > 1e-6:
                                    violations.append(
                                        f"{job_id}: confirmation_summary cohort {cohort} exact_accuracy mismatch"
                                    )

    return violations


def estimate_campaign_storage_bytes(phase_jobs: tuple[dict[str, object], ...], *, gpu_mode: bool = True) -> dict[str, int]:
    """Compute conservative storage footprint from actual parameter inventories and archive overhead."""

    if not gpu_mode:
        # CPU bounded canary uses tiny fixtures (<100k params)
        return {
            "weights_and_optimizer_bytes": 16_000_000,
            "episode_and_report_bytes": 32_000_000,
            "archive_headroom_bytes": 64_000_000,
            "total_required_bytes": 112_000_000,
        }
    total_ckpt_bytes = 0
    for job in phase_jobs:
        params = int(job["total_parameters"])
        # FP32 weights (4B) + AdamW m/v (8B) = 12 B/param; up to 2 retained snapshots (LATEST + PREVIOUS) + staging
        per_snapshot = params * 12
        total_ckpt_bytes += 3 * per_snapshot
    episode_and_report = len(phase_jobs) * 64 * 1024 * 1024
    # Zip output alongside unpacked working directory
    subtotal = total_ckpt_bytes + episode_and_report
    archive_headroom = subtotal
    return {
        "weights_and_optimizer_bytes": total_ckpt_bytes,
        "episode_and_report_bytes": episode_and_report,
        "archive_headroom_bytes": archive_headroom,
        "total_required_bytes": subtotal + archive_headroom,
    }


def forecast_campaign_runtime(
    *,
    phase_jobs: tuple[dict[str, object], ...],
    seconds_per_update: float,
    updates_per_job: list[int],
    canary_seconds: float,
    profile_seconds: float,
    lanes: int = 2,
    eval_seconds_per_job: float = 45.0,
    eval_seconds_per_episode: float | None = None,
    episodes_per_job: int | None = None,
    per_job_train_seconds: list[float] | None = None,
    per_job_eval_seconds: list[float] | None = None,
    work_margin: float = DEFAULT_WORK_MARGIN,
    safety_reserve_seconds: float = DEFAULT_SAFETY_RESERVE_SECONDS,
    phase_budget_seconds: float | None = None,
) -> dict[str, float | bool | None]:
    """2-lane runtime forecast with measured eval timings, 2.0x work margin, and explicit reserve."""

    if per_job_train_seconds is not None:
        job_train_s = [float(x) for x in per_job_train_seconds]
    else:
        job_train_s = [float(u) * float(seconds_per_update) for u in updates_per_job]

    if per_job_eval_seconds is not None:
        job_eval_s = [float(x) for x in per_job_eval_seconds]
    elif eval_seconds_per_episode is not None and episodes_per_job is not None:
        per_eval = float(eval_seconds_per_episode) * float(episodes_per_job)
        job_eval_s = [per_eval for _ in phase_jobs]
    else:
        job_eval_s = [float(eval_seconds_per_job) for _ in phase_jobs]

    total_train_s = sum(job_train_s)
    total_eval_s = sum(job_eval_s)

    eff_lanes = max(int(lanes), 1)
    lane_loads = [0.0] * eff_lanes
    for jt, je in zip(job_train_s, job_eval_s):
        min_idx = min(range(eff_lanes), key=lambda idx: lane_loads[idx])
        lane_loads[min_idx] += jt + je
    scheduled_parallel_work_s = max(lane_loads) if lane_loads else 0.0
    margined_parallel_work_s = float(work_margin) * scheduled_parallel_work_s
    total_forecast_s = margined_parallel_work_s + float(canary_seconds) + float(profile_seconds) + float(safety_reserve_seconds)
    feasible = (total_forecast_s <= float(phase_budget_seconds)) if phase_budget_seconds is not None else None
    return {
        "lanes": float(lanes),
        "work_margin": float(work_margin),
        "seconds_per_update": float(seconds_per_update),
        "total_train_seconds_serial": float(total_train_s),
        "total_eval_seconds_serial": float(total_eval_s),
        "parallel_work_seconds": float(scheduled_parallel_work_s),
        "scheduled_parallel_work_seconds": float(scheduled_parallel_work_s),
        "margined_parallel_work_seconds": float(margined_parallel_work_s),
        "canary_seconds": float(canary_seconds),
        "profile_seconds": float(profile_seconds),
        "safety_reserve_seconds": float(safety_reserve_seconds),
        "total_forecast_seconds": float(total_forecast_s),
        "phase_budget_seconds": float(phase_budget_seconds) if phase_budget_seconds is not None else None,
        "feasible": feasible,
    }


def _call_with_accepted_kwargs(
    fn: Callable[..., Any],
    *,
    positional: tuple[Any, ...] = (),
    kwargs: dict[str, Any],
) -> Any:
    """Invoke a runner/preparer callback with only the arguments its signature accepts."""

    sig = inspect.signature(fn)
    params = list(sig.parameters.values())
    has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params)
    accepted_kw_names = {
        p.name
        for p in params
        if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    pos_capable = [
        p
        for p in params
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]

    if has_var_kw:
        leading_pos: list[Any] = []
        for idx, p in enumerate(pos_capable):
            if idx >= len(positional) or p.name in kwargs:
                break
            leading_pos.append(positional[idx])
        return fn(*leading_pos, **kwargs)

    filtered = {k: v for k, v in kwargs.items() if k in accepted_kw_names}
    if filtered:
        leading_pos = []
        for idx, p in enumerate(pos_capable):
            if idx >= len(positional) or p.name in filtered:
                break
            leading_pos.append(positional[idx])
        return fn(*leading_pos, **filtered)

    if positional and pos_capable:
        return fn(*positional[: len(pos_capable)])
    return fn()


def _save_state_machine(work_root: Path, state: dict[str, object]) -> None:
    work_root.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(work_root / "CAMPAIGN_STATE.json", state)


def _run_worker_subprocess(
    mode: str,
    *,
    job_id: str | None = None,
    arm: str | None = None,
    source_commit: str,
    work_dir: Path,
    gpu_lane: int,
    device: str = "cuda:0",
    cwd: Path | None = None,
    extra_args: list[str] | None = None,
) -> subprocess.Popen:
    cmd = [
        sys.executable,
        "-m",
        "tools.elendril_v8_worker",
        mode,
        "--source",
        source_commit,
        "--work-dir",
        str(work_dir),
        "--device",
        device,
    ]
    if job_id:
        cmd.extend(["--job-id", job_id])
    if arm:
        cmd.extend(["--arm", arm])
    if extra_args:
        cmd.extend(extra_args)
    env = dict(os.environ)
    env["CUBLAS_WORKSPACE_CONFIG"] = os.environ.get("CUBLAS_WORKSPACE_CONFIG") or ":4096:8"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env["OMP_NUM_THREADS"] = "2"
    env["MKL_NUM_THREADS"] = "2"
    env["TORCH_NUM_THREADS"] = "2"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    if cwd is not None:
        existing_pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{cwd}{os.pathsep}{existing_pp}" if existing_pp else str(cwd)
    if device != "cpu":
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_lane)
    log_dir = work_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_name = f"{mode}_{job_id or arm or 'worker'}.log"
    fh = (log_dir / log_name).open("w", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        env=env,
        cwd=str(cwd) if cwd is not None else None,
        stdout=fh,
        stderr=subprocess.STDOUT,
    )
    proc._v8_log_fh = fh  # type: ignore[attr-defined]
    return proc


def _close_proc_log(proc: subprocess.Popen) -> None:
    fh = getattr(proc, "_v8_log_fh", None)
    if fh is not None:
        fh.close()


def _run_parallel_gpu_workers(
    tasks: Sequence[dict[str, Any]],
    *,
    gpu_devices: Sequence[int],
    source_commit: str,
    repo_root: Path,
) -> tuple[dict[str, dict[str, object]], str | None]:
    """Run independent GPU worker subprocesses concurrently across gpu_devices (one per GPU lane)."""

    lanes = [int(d) for d in gpu_devices] or [0]
    pending = list(tasks)
    active: dict[int, tuple[dict[str, Any], subprocess.Popen]] = {}
    results: dict[str, dict[str, object]] = {}
    failed_reason: str | None = None

    try:
        while pending or active:
            for lane_idx, gpu_id in enumerate(lanes):
                if lane_idx in active or not pending or failed_reason is not None:
                    continue
                task = pending.pop(0)
                proc = _run_worker_subprocess(
                    str(task["mode"]),
                    job_id=task.get("job_id"),
                    arm=task.get("arm"),
                    source_commit=source_commit,
                    work_dir=Path(task["work_dir"]),
                    gpu_lane=gpu_id,
                    device="cuda:0",
                    cwd=repo_root,
                    extra_args=list(task["build_extra_args"](gpu_id)),
                )
                active[lane_idx] = (task, proc)

            finished = [lane_idx for lane_idx, (_, proc) in active.items() if proc.poll() is not None]
            for lane_idx in finished:
                task, proc = active.pop(lane_idx)
                _close_proc_log(proc)
                key = str(task["key"])
                json_path = Path(task["json_path"])
                if proc.returncode != 0:
                    failed_reason = f"{task['label']} subprocess failed for {key} (exit={proc.returncode})"
                    pending.clear()
                    break
                if not json_path.exists():
                    failed_reason = f"{task['label']} subprocess missing JSON output for {key}: {json_path}"
                    pending.clear()
                    break
                res = json.loads(json_path.read_text(encoding="utf-8"))
                results[key] = dict(res)
                err = task["validate_fn"](key, res)
                if err is not None:
                    failed_reason = err
                    pending.clear()
                    break

            if failed_reason is not None and active:
                for _, (_, proc) in list(active.items()):
                    if proc.poll() is None:
                        proc.terminate()
                for lane_idx, (_, proc) in list(active.items()):
                    proc.wait(timeout=30)
                    _close_proc_log(proc)
                    del active[lane_idx]
                break
            if active:
                time.sleep(0.05)
    finally:
        for _, proc in active.values():
            if proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=30)
            _close_proc_log(proc)

    return results, failed_reason


def _verify_job_result(
    job: dict[str, object],
    work_dir: Path,
    *,
    source_commit: str,
    expected_updates: int,
    expected_split_audit_sha256: str | None,
    check_confirmation: bool,
    cpu_smoke: bool | None = None,
) -> dict[str, object]:
    job_id = str(job["job_id"])
    job_json_path = work_dir / "jobs" / f"{job_id}.json"
    if not job_json_path.exists():
        raise ValueError(f"missing worker report for {job_id}: {job_json_path}")
    data = json.loads(job_json_path.read_text(encoding="utf-8"))
    violations = validate_job_completion_artifacts(
        job,
        data,
        work_dir,
        source_commit=source_commit,
        expected_updates=expected_updates,
        expected_split_audit_sha256=expected_split_audit_sha256,
        check_confirmation=check_confirmation,
        cpu_smoke=cpu_smoke,
    )
    if violations:
        raise ValueError("; ".join(violations))
    if not data.get("data_identity"):
        store = CheckpointStore(work_dir / "checkpoints", job_id)
        ckpt_state, _ = store.restore(str(data["checkpoint_sha256"]))
        data["data_identity"] = ckpt_state.identities.data_manifest_sha256
    return data


def _load_episodes_for_pairwise(ep_file: Path) -> list[EpisodeRecord]:
    out: list[EpisodeRecord] = []
    if not ep_file.exists():
        return out
    for line in ep_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("mode") != "raw" or row.get("milestone", "epoch-4") != "epoch-4":
            continue
        out.append(
            EpisodeRecord(
                world_id=str(row["world_id"]),
                base_world_id=str(row["base_world_id"]),
                cohort=str(row["cohort"]),
                family=str(row["family"]),
                form=str(row["form"]),
                mode="raw",
                model_identity="loaded",
                steps=(),
                outcome=EvaluationOutcome(correct=bool(row["correct"]), terminal_status=str(row["terminal_status"]), charged_cost=0.0),
                first_observable_divergence=row.get("first_observable_divergence"),
                query_count=0,
                retrieve_count=0,
                model_calls=int(row.get("model_calls", 0)),
                evaluated_positions=int(row.get("evaluated_positions", 0)),
                teacher_action_matches=0,
            )
        )
    return out


def _resolve_condition_comparison_pairs(
    phase: str,
    phase_jobs: Sequence[dict[str, object]],
    *,
    winner_arm: str | None = None,
    comparator_arm: str | None = None,
) -> list[tuple[str, str]]:
    """Determine canonical (candidate_condition, comparator_condition) pairs for a V8 phase."""

    conditions = sorted({f"{j['arm']}-{j['dataset']}" for j in phase_jobs})
    if phase == "data-control":
        arms_ordered = []
        for preferred in ("B0", "P", "A"):
            if f"{preferred}-D1" in conditions and f"{preferred}-D0" in conditions:
                arms_ordered.append(preferred)
        for j in phase_jobs:
            arm = str(j["arm"])
            if arm not in arms_ordered and f"{arm}-D1" in conditions and f"{arm}-D0" in conditions:
                arms_ordered.append(arm)
        return [(f"{arm}-D1", f"{arm}-D0") for arm in arms_ordered]

    if phase == "head-screen" and "F-D1" in conditions and "J-D1" in conditions:
        return [("F-D1", "J-D1")]

    if phase == "reader-screen" and "P-D1" in conditions and "F-cap-D1" in conditions:
        return [("P-D1", "F-cap-D1")]

    if phase == "replication":
        win = f"{winner_arm or 'F'}-D1"
        comp = f"{comparator_arm or 'J'}-D1"
        if win in conditions and comp in conditions and win != comp:
            return [(win, comp)]

    if "mechanism" in phase:
        pairs: list[tuple[str, str]] = []
        arms_d1 = [c for c in conditions if c.endswith("-D1")]
        for c in arms_d1:
            if c not in ("P-D1", "B0-D1"):
                if "P-D1" in conditions:
                    pairs.append((c, "P-D1"))
                if "B0-D1" in conditions:
                    pairs.append((c, "B0-D1"))
        if "P-D1" in conditions and "B0-D1" in conditions:
            pairs.append(("P-D1", "B0-D1"))
        if pairs:
            return pairs

    # Generic screen fallback
    candidates = [c for c in ("P-D1", "C-D1", "A-D1", "E-D1", "F-D1") if c in conditions]
    comparator = next((c for c in ("B0-D1", "J-D1", "F-cap-D1") if c in conditions), None)
    if comparator is not None and candidates:
        return [(cand, comparator) for cand in candidates if cand != comparator]
    if len(conditions) == 2:
        return [(conditions[1], conditions[0])]
    return []


def _extract_confirmation_pair_metrics(
    ra: dict[str, object],
    rb: dict[str, object],
    *,
    init_seed: int,
    order_seed: int,
) -> dict[str, object]:
    ca = ra.get("confirmation_summary")
    cb = rb.get("confirmation_summary")
    if not isinstance(ca, dict) or not isinstance(cb, dict):
        raise ValueError("replication confirmation gate requires confirmation_summary on both candidate and comparator")
    cand_ov = ca.get("exact_episode_accuracy")
    comp_ov = cb.get("exact_episode_accuracy")
    if cand_ov is None or comp_ov is None:
        raise ValueError("confirmation_summary missing exact_episode_accuracy")
    cand_r09 = (ca.get("by_cohort") or {}).get("R09", {}).get("exact_accuracy")  # type: ignore[union-attr]
    cand_lcf = ca.get("linked_counterfactuals") or {}
    cand_rel = cand_lcf.get("relevant_cf_both_correct")  # type: ignore[union-attr]
    cand_irr = cand_lcf.get("irrelevant_cf_stability")  # type: ignore[union-attr]
    return {
        "init_seed": init_seed,
        "order_seed": order_seed,
        "candidate_job_id": ra.get("job_id"),
        "comparator_job_id": rb.get("job_id"),
        "split": "confirmation",
        "split_name": "confirmation",
        "paired_gain": float(cand_ov) - float(comp_ov),
        "candidate_raw_overall": float(cand_ov),
        "comparator_raw_overall": float(comp_ov),
        "candidate_raw_r09": float(cand_r09) if cand_r09 is not None else 0.0,
        "candidate_relevant_cf_both_correct": float(cand_rel) if cand_rel is not None else 0.0,
        "candidate_irrelevant_cf_stability": float(cand_irr) if cand_irr is not None else 0.0,
    }


def _write_sidecars_and_export(
    *,
    work_root: Path,
    output_dir: Path,
    repo_root: Path,
    phase: str,
    status: str,
    source_commit: str,
    run_records: list[dict[str, object]],
    paired_comparisons: list[dict[str, object]],
    condition_pairings: list[dict[str, object]],
    confirmation_pairing: dict[str, object] | None,
    advancement: dict[str, object] | None,
    executed_scenarios: set[str],
    diagnostics_payload: dict[str, object],
    split_audit_sha256: str | None = None,
    forecast: dict[str, object] | None = None,
    notes: list[str] | None = None,
    status_extra: dict[str, object] | None = None,
    archive_baseline: dict[str, object] | None = None,
) -> dict[str, object]:
    """Write all mandatory V8 sidecars to work_root and export + verify archives."""

    cost_views = {
        "schema": "elendril-v8-cost-views-v1",
        "phase": phase,
        "source_commit": source_commit,
        **build_cost_views(run_records),
    }
    pairwise_payload = {
        "schema": "elendril-v8-pairwise-comparisons-v1",
        "phase": phase,
        "source_commit": source_commit,
        "split_audit_sha256": split_audit_sha256,
        "condition_pairings": condition_pairings,
        "world_comparisons": paired_comparisons,
        "confirmation_pairing": confirmation_pairing,
        "advancement": advancement,
    }
    ledger_payload = {
        "schema": "elendril-v8-scenario-ledger-v1",
        "phase": phase,
        "source_commit": source_commit,
        "executed_scenarios": sorted(executed_scenarios),
        "scenarios": updated_scenario_ledger(executed_scenarios),
    }
    baseline = archive_baseline if archive_baseline is not None else load_v717_archive_baseline(repo_root)
    metrics, md = build_v8_report_payload(
        phase=phase,
        status=status,
        source_commit=source_commit,
        run_records=run_records,
        paired_comparisons=paired_comparisons,
        advancement=advancement,
        executed_scenarios=executed_scenarios,
        archive_baseline=baseline,
        notes=notes,
    )
    if split_audit_sha256 is not None:
        metrics["split_audit_sha256"] = split_audit_sha256
    if forecast is not None:
        metrics["forecast"] = forecast
    if condition_pairings:
        metrics["condition_pairings"] = condition_pairings
    if confirmation_pairing is not None:
        metrics["confirmation_pairing"] = confirmation_pairing

    diag_path = work_root / "DIAGNOSTICS.json"
    cost_path = work_root / "COST_VIEWS.json"
    pair_path = work_root / "PAIRWISE_COMPARISONS.json"
    ledger_path = work_root / "SCENARIO_LEDGER.json"
    metrics_path = work_root / "METRICS.json"
    results_path = work_root / "RESULTS.md"

    _atomic_write_json(diag_path, diagnostics_payload)
    _atomic_write_json(cost_path, cost_views)
    _atomic_write_json(pair_path, pairwise_payload)
    _atomic_write_json(ledger_path, ledger_payload)
    _atomic_write_json(metrics_path, metrics)
    results_path.write_text(md, encoding="utf-8")

    status_payload: dict[str, object] = {
        "status": status,
        "phase": phase,
        "source_commit": source_commit,
        **({"split_audit_sha256": split_audit_sha256} if split_audit_sha256 is not None else {}),
        **(status_extra or {}),
    }
    _atomic_write_json(work_root / "FINAL_STATUS.json", status_payload)

    analysis_files: dict[str, Path] = {
        "DIAGNOSTICS.json": diag_path,
        "COST_VIEWS.json": cost_path,
        "PAIRWISE_COMPARISONS.json": pair_path,
        "SCENARIO_LEDGER.json": ledger_path,
        "METRICS.json": metrics_path,
        "RESULTS.md": results_path,
    }
    split_audit_file = work_root / "splits" / "SPLIT_AUDIT.json"
    if split_audit_file.exists():
        analysis_files["splits/SPLIT_AUDIT.json"] = split_audit_file

    export_res = export_v8_archives(
        work_root,
        output_dir,
        source_commit=source_commit,
        phase=phase,
        status_payload=status_payload,
        analysis_files=analysis_files,
        include_work_root_evidence=True,
    )
    verify_v8_export(output_dir, expected_source_commit=source_commit)
    return export_res


def _run_archive_diagnostic_phase(
    *,
    repo_root: Path,
    work_root: Path,
    output_dir: Path,
    source_commit: str,
) -> dict[str, object]:
    """Phase 0: audit local V7.17 archive baseline (no local weights), D0/D1 corpora, shortcuts & holdouts."""

    t0 = time.perf_counter()
    archive_info = load_v717_archive_baseline(repo_root)
    d0 = load_d0_dataset()
    d1 = build_d1_training_dataset(quota_per_family=64)
    dev_cohorts = build_d1_dev_suite(worlds_per_cohort=8, excluded_structure_hashes=d1.structural_hashes)
    dev_worlds = [w for rows in dev_cohorts.values() for w in rows]
    disjoint = audit_split_disjointness(list(d1.worlds) + dev_worlds, min_distinct_classes_per_split=4)
    audit = audit_teacher_samples(d1.steps)
    shortcuts = compute_shortcut_baselines(d1.worlds)

    diag = {
        "schema": "elendril-v8-phase0-diagnostic-v1",
        "source_commit": source_commit,
        "archive_baseline": archive_info,
        "d0_identity": d0.identity,
        "d0_samples": len(d0.samples),
        "d1_probe_audit": audit,
        "d1_shortcuts": shortcuts,
        "split_disjointness": disjoint,
        "wall_seconds": time.perf_counter() - t0,
    }
    export_res = _write_sidecars_and_export(
        work_root=work_root,
        output_dir=output_dir,
        repo_root=repo_root,
        phase="archive-diagnostic",
        status="COMPLETE",
        source_commit=source_commit,
        run_records=[],
        paired_comparisons=[],
        condition_pairings=[],
        confirmation_pairing=None,
        advancement=None,
        executed_scenarios=set(),
        diagnostics_payload=diag,
        archive_baseline=archive_info,
        notes=[
            "Phase 0 archive diagnostic verified locally without synthetic weights.",
            f"Parent V7.17 checkpoint status: {archive_info.get('weights_status')}.",
            f"D0 identity verified: {d0.identity} ({len(d0.samples)} samples).",
        ],
    )
    return {"status": "COMPLETE", "state": "COMPLETE", "export": export_res, "diagnostic": diag}


def run_campaign(
    *,
    phase: str,
    source_commit: str,
    work_root: Path,
    output_dir: Path,
    minutes: float = 330.0,
    owner_enabled: bool = False,
    winner_arm: str | None = None,
    comparator_arm: str | None = None,
    confirmation_seed: int | None = None,
    repo_root: Path | None = None,
    cpu_smoke: bool = False,
    updates_override: int | None = None,
    dev_worlds_per_cohort: int | None = None,
    conf_worlds_per_cohort: int | None = None,
    devices: tuple[int, ...] | list[int] | None = None,
    free_bytes_override: int | None = None,
    safety_reserve_seconds: float = DEFAULT_SAFETY_RESERVE_SECONDS,
    eval_seconds_per_job: float = 45.0,
    work_margin: float = DEFAULT_WORK_MARGIN,
    split_preparer: Callable[..., dict[str, Any]] | None = None,
    job_runner: Callable[[dict[str, object], Path], dict[str, object]] | None = None,
    canary_runner: Callable[..., dict[str, object]] | None = None,
    profile_runner: Callable[..., dict[str, object]] | None = None,
) -> dict[str, object]:
    """Execute one explicit V8 campaign phase with full admission, monitoring, and export verification."""

    if phase not in PHASES:
        raise ValueError(f"unknown V8 phase: {phase!r}")
    if len(source_commit) != 40 or any(c not in "0123456789abcdef" for c in source_commit):
        raise ValueError("source_commit must be a 40-char lowercase hex commit")

    work_root = Path(work_root)
    output_dir = Path(output_dir)
    work_root.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    repo = repo_root or Path(__file__).resolve().parents[1]

    if not owner_enabled:
        msg = "NOT LAUNCHED: owner switch is false"
        status_payload = {"status": "NOT_RUN", "reason": msg, "phase": phase, "source_commit": source_commit}
        _atomic_write_json(output_dir / "FINAL_STATUS.json", status_payload)
        _save_state_machine(work_root, {"state": "PREPARED", "status": "NOT_RUN", "reason": msg})
        return {"status": "NOT_RUN", "message": msg}

    if phase == "archive-diagnostic":
        _save_state_machine(work_root, {"state": "PREPARED", "phase": phase})
        return _run_archive_diagnostic_phase(
            repo_root=repo,
            work_root=work_root,
            output_dir=output_dir,
            source_commit=source_commit,
        )

    phase_jobs = protocol_jobs(phase, winner_arm=winner_arm, comparator_arm=comparator_arm)
    effective_updates_override = updates_override if updates_override is not None else (2 if cpu_smoke else None)
    eff_dev_wpc = int(dev_worlds_per_cohort) if dev_worlds_per_cohort is not None else (2 if cpu_smoke else 128)
    eff_conf_wpc = int(conf_worlds_per_cohort) if conf_worlds_per_cohort is not None else (2 if cpu_smoke else 256)
    phase_budget_seconds = float(minutes) * 60.0
    t_campaign_start = time.monotonic()
    deadline = t_campaign_start + phase_budget_seconds

    state_log: dict[str, object] = {
        "phase": phase,
        "source_commit": source_commit,
        "state": "PREPARED",
        "jobs": [j["job_id"] for j in phase_jobs],
    }
    _save_state_machine(work_root, state_log)

    if confirmation_seed is not None and not _is_strict_int(confirmation_seed):
        raise ValueError(f"confirmation_seed must be an integer, got {confirmation_seed!r}")

    if phase == "replication" and confirmation_seed is None:
        reason = "Stage 0 replication requires explicit owner --confirmation-seed"
        state_log["state"] = "NO_GO"
        state_log["reason"] = reason
        _save_state_machine(work_root, state_log)
        _write_sidecars_and_export(
            work_root=work_root,
            output_dir=output_dir,
            repo_root=repo,
            phase=phase,
            status="NO_GO",
            source_commit=source_commit,
            run_records=[],
            paired_comparisons=[],
            condition_pairings=[],
            confirmation_pairing=None,
            advancement=None,
            executed_scenarios=set(),
            diagnostics_payload={"schema": "elendril-v8-diagnostics-v1", "phase": phase, "status": "NO_GO", "reason": reason},
            notes=[reason],
            status_extra={"reason": reason},
        )
        return {"status": "NO_GO", "reason": reason}

    # 1. Storage Admission
    storage_est = estimate_campaign_storage_bytes(phase_jobs, gpu_mode=(not cpu_smoke))
    disk_free = int(free_bytes_override) if free_bytes_override is not None else shutil.disk_usage(work_root).free
    if disk_free < storage_est["total_required_bytes"]:
        reason = (
            f"RESOURCE_INFEASIBLE: storage admission failed: free={disk_free} < required={storage_est['total_required_bytes']} bytes"
        )
        state_log["state"] = "NO_GO"
        state_log["reason"] = reason
        _save_state_machine(work_root, state_log)
        _write_sidecars_and_export(
            work_root=work_root,
            output_dir=output_dir,
            repo_root=repo,
            phase=phase,
            status="NO_GO",
            source_commit=source_commit,
            run_records=[],
            paired_comparisons=[],
            condition_pairings=[],
            confirmation_pairing=None,
            advancement=None,
            executed_scenarios=set(),
            diagnostics_payload={"schema": "elendril-v8-diagnostics-v1", "phase": phase, "status": "NO_GO", "reason": reason, "storage": storage_est, "disk_free": disk_free},
            notes=[reason],
            status_extra={"reason": reason, "reason_code": "RESOURCE_INFEASIBLE"},
        )
        return {"status": "NO_GO", "reason": reason, "reason_code": "RESOURCE_INFEASIBLE", "storage": storage_est, "disk_free": disk_free}

    # 2. R1 — Upfront Verified Split Preparation (before canaries, profiling, or training)
    preparer = split_preparer or prepare_verified_splits
    split_res = _call_with_accepted_kwargs(
        preparer,
        positional=(work_root,),
        kwargs={
            "cpu_smoke": cpu_smoke,
            "dev_worlds_per_cohort": eff_dev_wpc,
            "conf_worlds_per_cohort": eff_conf_wpc,
            "confirmation_seed": confirmation_seed,
        },
    )
    split_audit_path = work_root / "splits" / "SPLIT_AUDIT.json"
    if not split_audit_path.exists():
        reason = f"Stage 0 split preparation failed: missing {split_audit_path}"
        state_log["state"] = "NO_GO"
        state_log["reason"] = reason
        _save_state_machine(work_root, state_log)
        _write_sidecars_and_export(
            work_root=work_root,
            output_dir=output_dir,
            repo_root=repo,
            phase=phase,
            status="NO_GO",
            source_commit=source_commit,
            run_records=[],
            paired_comparisons=[],
            condition_pairings=[],
            confirmation_pairing=None,
            advancement=None,
            executed_scenarios=set(),
            diagnostics_payload={"schema": "elendril-v8-diagnostics-v1", "phase": phase, "status": "NO_GO", "reason": reason},
            notes=[reason],
            status_extra={"reason": reason},
        )
        return {"status": "NO_GO", "reason": reason}

    audit_obj = json.loads(split_audit_path.read_text(encoding="utf-8"))
    split_audit_sha256 = str(audit_obj.get("split_audit_sha256") or "")
    body_for_hash = {k: v for k, v in audit_obj.items() if k != "split_audit_sha256"}
    split_ok = (
        audit_obj.get("audit_passed") is True
        and _is_sha256(split_audit_sha256)
        and digest(body_for_hash) == split_audit_sha256
        and (split_res is None or str(split_res.get("split_audit_sha256", split_audit_sha256)) == split_audit_sha256)
    )
    if not split_ok:
        reason = "Stage 0 split audit failed verification in splits/SPLIT_AUDIT.json"
        state_log["state"] = "NO_GO"
        state_log["reason"] = reason
        _save_state_machine(work_root, state_log)
        _write_sidecars_and_export(
            work_root=work_root,
            output_dir=output_dir,
            repo_root=repo,
            phase=phase,
            status="NO_GO",
            source_commit=source_commit,
            run_records=[],
            paired_comparisons=[],
            condition_pairings=[],
            confirmation_pairing=None,
            advancement=None,
            executed_scenarios=set(),
            diagnostics_payload={"schema": "elendril-v8-diagnostics-v1", "phase": phase, "status": "NO_GO", "reason": reason, "split_audit": audit_obj},
            split_audit_sha256=split_audit_sha256 or None,
            notes=[reason],
            status_extra={"reason": reason},
        )
        return {"status": "NO_GO", "reason": reason}

    state_log["split_audit_sha256"] = split_audit_sha256
    _save_state_machine(work_root, state_log)

    if devices is not None:
        gpu_devices = [int(d) for d in devices] or [0]
    elif not cpu_smoke and torch.cuda.is_available():
        gpu_devices = list(range(max(1, torch.cuda.device_count())))
    else:
        gpu_devices = [0]

    # 3. R4 — Stage A Positive Canary over ALL active arms (including B0)
    active_arms = sorted({str(j["arm"]) for j in phase_jobs})
    canary_results: dict[str, dict[str, object]] = {}
    canary_lr = None if cpu_smoke else float(OPTIMIZER_DEFAULTS["peak_lr"])
    canary_failed_reason: str | None = None

    def _validate_canary_result(arm_key: str, c_res: dict[str, object]) -> str | None:
        passed_flag = bool(c_res.get("passed"))
        exact_acc = c_res.get("exact_action_accuracy")
        acc_ok = True if cpu_smoke else (exact_acc is not None and float(exact_acc) >= 0.95)
        if not passed_flag or not acc_ok:
            return f"Stage A canary check failed for arm {arm_key}: {c_res}"
        return None

    if canary_runner is not None or cpu_smoke:
        for arm in active_arms:
            if canary_runner is not None:
                c_res = _call_with_accepted_kwargs(
                    canary_runner,
                    kwargs={
                        "arm": arm,
                        "source_commit": source_commit,
                        "work_dir": work_root / "canaries" / arm,
                        "device": "cpu" if cpu_smoke else "cuda:0",
                        "cpu_canary": cpu_smoke,
                        "lr": canary_lr,
                    },
                )
            else:
                c_res = run_canary(
                    arm=arm,
                    source_commit=source_commit,
                    work_dir=work_root / "canaries" / arm,
                    device="cpu" if cpu_smoke else "cuda:0",
                    cpu_canary=cpu_smoke,
                    lr=canary_lr,
                )
            canary_results[arm] = dict(c_res)
            canary_failed_reason = _validate_canary_result(arm, canary_results[arm])
            if canary_failed_reason is not None:
                break
    else:
        canary_tasks = [
            {
                "key": arm,
                "mode": "canary",
                "arm": arm,
                "work_dir": work_root / "canaries" / arm,
                "json_path": work_root / "canaries" / arm / "canary_result.json",
                "label": "Stage A canary",
                "build_extra_args": (
                    lambda _gpu_id, jp=work_root / "canaries" / arm / "canary_result.json": [
                        "--json-out",
                        str(jp),
                        *(["--lr", str(canary_lr)] if canary_lr is not None else []),
                    ]
                ),
                "validate_fn": _validate_canary_result,
            }
            for arm in active_arms
        ]
        canary_results, canary_failed_reason = _run_parallel_gpu_workers(
            canary_tasks,
            gpu_devices=gpu_devices,
            source_commit=source_commit,
            repo_root=repo,
        )

    if canary_failed_reason is not None:
        state_log["state"] = "NO_GO"
        state_log["reason"] = canary_failed_reason
        state_log["canaries"] = canary_results
        _save_state_machine(work_root, state_log)
        _write_sidecars_and_export(
            work_root=work_root,
            output_dir=output_dir,
            repo_root=repo,
            phase=phase,
            status="NO_GO",
            source_commit=source_commit,
            run_records=[],
            paired_comparisons=[],
            condition_pairings=[],
            confirmation_pairing=None,
            advancement=None,
            executed_scenarios=set(),
            diagnostics_payload={
                "schema": "elendril-v8-diagnostics-v1",
                "phase": phase,
                "status": "NO_GO",
                "reason": canary_failed_reason,
                "canaries": canary_results,
            },
            split_audit_sha256=split_audit_sha256,
            notes=[canary_failed_reason],
            status_extra={"reason": canary_failed_reason},
        )
        return {"status": "NO_GO", "reason": canary_failed_reason, "canaries": canary_results}

    canary_wall_seconds = sum(float(r.get("wall_seconds", 0.0)) for r in canary_results.values())

    # 4. R6 — Stage B Representative Multi-Condition Profiling + 2.0x Work Margin Forecast
    unique_conditions: list[tuple[str, str, dict[str, object]]] = []
    seen_conds: set[tuple[str, str]] = set()
    for j in phase_jobs:
        key = (str(j["arm"]), str(j["dataset"]))
        if key not in seen_conds:
            seen_conds.add(key)
            unique_conditions.append((key[0], key[1], j))

    profile_results: dict[str, dict[str, object]] = {}
    t_prof0 = time.perf_counter()
    profile_failed_reason: str | None = None

    def _validate_profile_result(cond_key: str, p_res: dict[str, object]) -> str | None:
        rec_parity = p_res.get("recovery_parity")
        if isinstance(rec_parity, dict):
            parity_ok = bool(rec_parity.get("all_match")) and bool(p_res.get("resume_parity_at_4_verified", True))
        else:
            parity_ok = bool(p_res.get("resume_parity_at_4_verified"))
        if not parity_ok:
            return f"Stage B profile failed recovery parity for {cond_key}: {p_res}"
        return None

    if profile_runner is not None or cpu_smoke:
        for idx, (arm, dataset, rep_job) in enumerate(unique_conditions):
            cond_key = f"{arm}-{dataset}"
            dev_idx = None if cpu_smoke else gpu_devices[idx % len(gpu_devices)]
            resolved_device = (
                "cpu"
                if cpu_smoke
                else (f"cuda:{dev_idx}" if len(gpu_devices) > 1 or dev_idx else "cuda:0")
            )
            if profile_runner is not None:
                p_res = _call_with_accepted_kwargs(
                    profile_runner,
                    positional=(rep_job,),
                    kwargs={
                        "arm": arm,
                        "dataset": dataset,
                        "device_index": dev_idx,
                        "job": rep_job,
                        "job_id": str(rep_job["job_id"]),
                        "source_commit": source_commit,
                        "work_dir": work_root / "profiles" / cond_key,
                        "device": resolved_device,
                        "cpu_canary": cpu_smoke,
                    },
                )
            else:
                p_res = run_profile(
                    job_id=str(rep_job["job_id"]),
                    arm=arm,
                    dataset=dataset,
                    device_index=dev_idx,
                    source_commit=source_commit,
                    work_dir=work_root / "profiles" / cond_key,
                    device=resolved_device,
                    cpu_canary=cpu_smoke,
                )
            profile_results[cond_key] = dict(p_res)
            profile_failed_reason = _validate_profile_result(cond_key, profile_results[cond_key])
            if profile_failed_reason is not None:
                break
    else:
        profile_tasks = [
            {
                "key": f"{arm}-{dataset}",
                "mode": "profile",
                "job_id": str(rep_job["job_id"]),
                "arm": arm,
                "work_dir": work_root / "profiles" / f"{arm}-{dataset}",
                "json_path": work_root / "profiles" / f"{arm}-{dataset}" / "profile_result.json",
                "label": "Stage B profile",
                "build_extra_args": (
                    lambda gpu_id, ds=dataset, jp=work_root / "profiles" / f"{arm}-{dataset}" / "profile_result.json": [
                        "--dataset",
                        ds,
                        "--device-index",
                        str(gpu_id),
                        "--json-out",
                        str(jp),
                    ]
                ),
                "validate_fn": _validate_profile_result,
            }
            for arm, dataset, rep_job in unique_conditions
        ]
        profile_results, profile_failed_reason = _run_parallel_gpu_workers(
            profile_tasks,
            gpu_devices=gpu_devices,
            source_commit=source_commit,
            repo_root=repo,
        )
    profile_wall = time.perf_counter() - t_prof0

    if profile_failed_reason is not None:
        state_log["state"] = "NO_GO"
        state_log["reason"] = profile_failed_reason
        state_log["profiles"] = profile_results
        _save_state_machine(work_root, state_log)
        _write_sidecars_and_export(
            work_root=work_root,
            output_dir=output_dir,
            repo_root=repo,
            phase=phase,
            status="NO_GO",
            source_commit=source_commit,
            run_records=[],
            paired_comparisons=[],
            condition_pairings=[],
            confirmation_pairing=None,
            advancement=None,
            executed_scenarios=set(),
            diagnostics_payload={
                "schema": "elendril-v8-diagnostics-v1",
                "phase": phase,
                "status": "NO_GO",
                "reason": profile_failed_reason,
                "canaries": canary_results,
                "profiles": profile_results,
            },
            split_audit_sha256=split_audit_sha256,
            notes=[profile_failed_reason],
            status_extra={"reason": profile_failed_reason},
        )
        return {"status": "NO_GO", "reason": profile_failed_reason, "profiles": profile_results}

    # Compute per-job expected updates and measured train/eval timings from matched condition profiles
    if cpu_smoke:
        episodes_per_job = (6 * eff_dev_wpc * 5) * (1 + 4) + (
            (6 * eff_conf_wpc * 5) if (phase == "replication" or confirmation_seed is not None) else 0
        )
    else:
        sub_worlds_n = 6 * min(160, eff_dev_wpc * 5)
        full_dev_n = 6 * eff_dev_wpc * 5
        episodes_per_job = full_dev_n + 2 * sub_worlds_n + (4 * full_dev_n) + (
            (6 * eff_conf_wpc * 5) if (phase == "replication" or confirmation_seed is not None) else 0
        )

    updates_est: list[int] = []
    per_job_train_seconds: list[float] = []
    per_job_eval_seconds: list[float] = []
    max_spu = 0.0
    for j in phase_jobs:
        cond_key = f"{j['arm']}-{j['dataset']}"
        p_res = profile_results[cond_key]
        spu = float(p_res["seconds_per_update"])
        max_spu = max(max_spu, spu)
        reg_u = expected_updates_for_job(j, cpu_smoke=cpu_smoke, updates_override=effective_updates_override)
        est_u = int(p_res.get("estimated_full_updates", reg_u)) if cpu_smoke else max(reg_u, int(p_res.get("estimated_full_updates", reg_u)))
        updates_est.append(est_u)
        per_job_train_seconds.append(est_u * spu)
        if p_res.get("eval_seconds_per_episode") is not None:
            per_job_eval_seconds.append(float(p_res["eval_seconds_per_episode"]) * episodes_per_job)
        else:
            per_job_eval_seconds.append(float(eval_seconds_per_job))

    lane_count = 1 if cpu_smoke else min(2, len(gpu_devices)) if gpu_devices else 2
    forecast = forecast_campaign_runtime(
        phase_jobs=phase_jobs,
        seconds_per_update=max_spu,
        updates_per_job=updates_est,
        canary_seconds=canary_wall_seconds,
        profile_seconds=profile_wall,
        lanes=lane_count,
        eval_seconds_per_job=eval_seconds_per_job,
        per_job_train_seconds=per_job_train_seconds,
        per_job_eval_seconds=per_job_eval_seconds,
        work_margin=work_margin,
        safety_reserve_seconds=safety_reserve_seconds,
        phase_budget_seconds=phase_budget_seconds,
    )
    remaining_s = deadline - time.monotonic()
    if float(forecast["total_forecast_seconds"]) > remaining_s:
        reason = (
            f"RESOURCE_INFEASIBLE: time admission failed: forecast {forecast['total_forecast_seconds']:.1f}s > remaining {remaining_s:.1f}s"
        )
        state_log["state"] = "NO_GO"
        state_log["reason"] = reason
        state_log["forecast"] = forecast
        _save_state_machine(work_root, state_log)
        _write_sidecars_and_export(
            work_root=work_root,
            output_dir=output_dir,
            repo_root=repo,
            phase=phase,
            status="NO_GO",
            source_commit=source_commit,
            run_records=[],
            paired_comparisons=[],
            condition_pairings=[],
            confirmation_pairing=None,
            advancement=None,
            executed_scenarios=set(),
            diagnostics_payload={
                "schema": "elendril-v8-diagnostics-v1",
                "phase": phase,
                "status": "NO_GO",
                "reason": reason,
                "reason_code": "RESOURCE_INFEASIBLE",
                "canaries": canary_results,
                "profiles": profile_results,
                "forecast": forecast,
            },
            split_audit_sha256=split_audit_sha256,
            forecast=forecast,
            notes=[reason],
            status_extra={"reason": reason, "reason_code": "RESOURCE_INFEASIBLE"},
        )
        return {"status": "NO_GO", "reason": reason, "reason_code": "RESOURCE_INFEASIBLE", "forecast": forecast}

    state_log["state"] = "ADMITTED"
    state_log["storage"] = storage_est
    state_log["forecast"] = forecast
    state_log["canaries"] = canary_results
    state_log["profiles"] = profile_results
    _save_state_machine(work_root, state_log)

    # 5. Execute 2-lane queue
    state_log["state"] = "RUNNING"
    _save_state_machine(work_root, state_log)
    completed_records: list[dict[str, object]] = []
    invalid_errors: list[str] = []
    deadline_hit = False
    check_confirmation = phase == "replication"

    if job_runner is not None:
        # Custom / deterministic runner: validate with the exact same shared validator!
        for job in phase_jobs:
            if deadline - time.monotonic() <= min(DEFAULT_EXPORT_RESERVE_SECONDS, minutes * 30.0) and completed_records:
                deadline_hit = True
                break
            try:
                job_runner(job, work_root)
                exp_u = expected_updates_for_job(job, cpu_smoke=cpu_smoke, updates_override=effective_updates_override)
                rec = _verify_job_result(
                    job,
                    work_root,
                    source_commit=source_commit,
                    expected_updates=exp_u,
                    expected_split_audit_sha256=split_audit_sha256,
                    check_confirmation=check_confirmation,
                    cpu_smoke=cpu_smoke,
                )
                completed_records.append(rec)
            except Exception as exc:
                invalid_errors.append(f"{job['job_id']}: {exc}")
                break
    else:
        pending = list(phase_jobs)
        active: dict[int, tuple[dict[str, object], subprocess.Popen]] = {}
        extra_worker_args: list[str] = (
            ["--cpu-canary", "--updates-override", str(effective_updates_override), "--dev-worlds", str(eff_dev_wpc)]
            if cpu_smoke
            else []
        )
        if confirmation_seed is not None:
            extra_worker_args.extend(["--confirmation-seed", str(confirmation_seed)])

        while pending or active:
            rem = deadline - time.monotonic()
            if rem <= DEFAULT_EXPORT_RESERVE_SECONDS and active:
                deadline_hit = True
                pending.clear()
                for _, (_, proc) in list(active.items()):
                    proc.terminate()
                for lane, (job, proc) in list(active.items()):
                    proc.wait(timeout=30)
                    fh = getattr(proc, "_v8_log_fh", None)
                    if fh is not None:
                        fh.close()
                    if proc.returncode == 0:
                        try:
                            exp_u = expected_updates_for_job(
                                job, cpu_smoke=cpu_smoke, updates_override=effective_updates_override
                            )
                            completed_records.append(
                                _verify_job_result(
                                    job,
                                    work_root,
                                    source_commit=source_commit,
                                    expected_updates=exp_u,
                                    expected_split_audit_sha256=split_audit_sha256,
                                    check_confirmation=check_confirmation,
                                    cpu_smoke=cpu_smoke,
                                )
                            )
                        except Exception as exc:
                            invalid_errors.append(f"{job['job_id']}: {exc}")
                    del active[lane]
                break

            for lane in range(lane_count):
                if lane not in active and pending and (deadline - time.monotonic() > DEFAULT_EXPORT_RESERVE_SECONDS):
                    job = pending.pop(0)
                    gpu_id = gpu_devices[lane % len(gpu_devices)] if not cpu_smoke else 0
                    proc = _run_worker_subprocess(
                        "train",
                        job_id=str(job["job_id"]),
                        source_commit=source_commit,
                        work_dir=work_root,
                        gpu_lane=gpu_id,
                        device="cpu" if cpu_smoke else "cuda:0",
                        cwd=repo,
                        extra_args=extra_worker_args,
                    )
                    active[lane] = (job, proc)

            finished_lanes = [lane for lane, (_, proc) in active.items() if proc.poll() is not None]
            for lane in finished_lanes:
                job, proc = active.pop(lane)
                fh = getattr(proc, "_v8_log_fh", None)
                if fh is not None:
                    fh.close()
                if proc.returncode != 0:
                    invalid_errors.append(f"{job['job_id']} exited with code {proc.returncode}")
                    pending.clear()
                    continue
                try:
                    exp_u = expected_updates_for_job(
                        job, cpu_smoke=cpu_smoke, updates_override=effective_updates_override
                    )
                    rec = _verify_job_result(
                        job,
                        work_root,
                        source_commit=source_commit,
                        expected_updates=exp_u,
                        expected_split_audit_sha256=split_audit_sha256,
                        check_confirmation=check_confirmation,
                        cpu_smoke=cpu_smoke,
                    )
                    completed_records.append(rec)
                except Exception as exc:
                    invalid_errors.append(f"{job['job_id']}: {exc}")
                    pending.clear()
            if active:
                time.sleep(0.25)

    # 6. Determine strict completion status
    if invalid_errors:
        final_status = "INVALID_RESULTS"
        state_log["state"] = "INVALID_RESULTS"
        state_log["errors"] = invalid_errors
    elif deadline_hit or len(completed_records) < len(phase_jobs):
        final_status = "PARTIAL"
        state_log["state"] = "EXECUTED"
    else:
        final_status = "COMPLETE"
        state_log["state"] = "EXECUTED"
    _save_state_machine(work_root, state_log)

    # 7. R5 — Condition + Seed Pairing & Confirmation Isolation
    paired_comparisons: list[dict[str, object]] = []
    condition_pairings: list[dict[str, object]] = []
    confirmation_pairing: dict[str, object] | None = None
    advancement: dict[str, object] | None = None

    if final_status == "COMPLETE" and completed_records:
        by_job_id: dict[str, dict[str, object]] = {}
        seen_keys: set[tuple[str, int, int]] = set()
        for rec in completed_records:
            jid = str(rec.get("job_id", ""))
            cond_id = str(rec.get("condition_id") or f"{rec.get('arm')}-{rec.get('dataset')}")
            seed_key = (cond_id, int(rec["init_seed"]), int(rec["order_seed"]))
            if jid in by_job_id or seed_key in seen_keys:
                invalid_errors.append(f"duplicate completed job/seed entry: {jid} {seed_key}")
            seen_keys.add(seed_key)
            by_job_id[jid] = rec

        expected_job_ids = {str(j["job_id"]) for j in phase_jobs}
        if set(by_job_id) != expected_job_ids:
            invalid_errors.append(
                f"completed job set mismatch: missing={sorted(expected_job_ids - set(by_job_id))}, extra={sorted(set(by_job_id) - expected_job_ids)}"
            )

        if not invalid_errors:
            comparison_pairs = _resolve_condition_comparison_pairs(
                phase,
                phase_jobs,
                winner_arm=winner_arm,
                comparator_arm=comparator_arm,
            )
            try:
                for cand_cond, comp_cond in comparison_pairs:
                    exp_seeds = sorted(
                        {
                            (int(j["init_seed"]), int(j["order_seed"]))
                            for j in phase_jobs
                            if f"{j['arm']}-{j['dataset']}" == cand_cond
                        }
                    )
                    pairing = pair_condition_runs(
                        completed_records,
                        candidate_condition=cand_cond,
                        comparator_condition=comp_cond,
                        expected_seed_pairs=exp_seeds,
                    )
                    condition_pairings.append(pairing)

                    for pair in pairing["pairs"]:
                        ra = by_job_id[str(pair["candidate_job_id"])]
                        rb = by_job_id[str(pair["comparator_job_id"])]
                        eps_a = _load_episodes_for_pairwise(Path(str(ra["episodes_file"]))) if ra.get("episodes_file") else []
                        eps_b = _load_episodes_for_pairwise(Path(str(rb["episodes_file"]))) if rb.get("episodes_file") else []
                        if eps_a and eps_b and {e.world_id for e in eps_a} == {e.world_id for e in eps_b}:
                            comp = paired_world_comparison(
                                eps_a,
                                eps_b,
                                arm_a=cand_cond,
                                arm_b=comp_cond,
                                bootstrap_samples=400,
                            )
                            comp["init_seed"] = pair["init_seed"]
                            comp["order_seed"] = pair["order_seed"]
                            comp["candidate_job_id"] = pair["candidate_job_id"]
                            comp["comparator_job_id"] = pair["comparator_job_id"]
                            paired_comparisons.append(comp)

                if phase == "replication" and condition_pairings:
                    primary = condition_pairings[0]
                    conf_pairs = [
                        _extract_confirmation_pair_metrics(
                            by_job_id[str(pair["candidate_job_id"])],
                            by_job_id[str(pair["comparator_job_id"])],
                            init_seed=int(pair["init_seed"]),
                            order_seed=int(pair["order_seed"]),
                        )
                        for pair in primary["pairs"]
                    ]
                    confirmation_pairing = {
                        "schema": "elendril-v8-confirmation-pairing-v1",
                        "phase": phase,
                        "split": "confirmation",
                        "split_name": "confirmation",
                        "is_confirmation": True,
                        "candidate_condition": primary["candidate_condition"],
                        "comparator_condition": primary["comparator_condition"],
                        "source_commit": primary["source_commit"],
                        "split_audit_sha256": primary["split_audit_sha256"],
                        "seed_count": len(conf_pairs),
                        "pairs": conf_pairs,
                        "mean_paired_gain": sum(float(p["paired_gain"]) for p in conf_pairs) / len(conf_pairs),
                        "positive_seed_differences": sum(1 for p in conf_pairs if float(p["paired_gain"]) > 0.0),
                    }
                    advancement = evaluate_advancement_gate(confirmation_pairing, confirmation=True)
                elif condition_pairings:
                    advancement = evaluate_advancement_gate(condition_pairings[0], confirmation=False)
            except Exception as exc:
                invalid_errors.append(f"pairwise comparison error: {exc}")

        if invalid_errors:
            final_status = "INVALID_RESULTS"
            state_log["state"] = "INVALID_RESULTS"
            state_log["errors"] = invalid_errors
            _save_state_machine(work_root, state_log)

    executed_scenarios: set[str] = set()
    if completed_records:
        executed_scenarios.update(EXPECTED_COHORTS)
        executed_scenarios.update({"F01", "F02", "F04", "G05", "G06", "G07", "A17", "A18"})
        if any(isinstance(r.get("extensions"), dict) and "F03" in r["extensions"] for r in completed_records):  # type: ignore[operator]
            executed_scenarios.add("F03")
        if any(isinstance(r.get("extensions"), dict) and "A20" in r["extensions"] for r in completed_records):  # type: ignore[operator]
            executed_scenarios.add("A20")
        if phase == "replication" and any(r.get("confirmation_summary") for r in completed_records):
            executed_scenarios.add("G08")

    per_job_diag = [
        {
            "job_id": r.get("job_id"),
            "condition_id": r.get("condition_id"),
            "completed_updates": r.get("completed_updates"),
            "checkpoint_sha256": r.get("checkpoint_sha256"),
            "final_probe": r.get("final_probe"),
            "rescore_summary": r.get("rescore_summary"),
            "milestones": sorted((r.get("milestones") or {}).keys()) if isinstance(r.get("milestones"), dict) else [],
            "pointer_interventions": r.get("pointer_interventions"),
            "extensions": r.get("extensions"),
        }
        for r in completed_records
    ]
    diagnostics_payload = {
        "schema": "elendril-v8-diagnostics-v1",
        "phase": phase,
        "status": final_status,
        "source_commit": source_commit,
        "split_audit_sha256": split_audit_sha256,
        "storage": storage_est,
        "canaries": canary_results,
        "profiles": profile_results,
        "forecast": forecast,
        "validation_errors": invalid_errors,
        "per_job_diagnostics": per_job_diag,
    }

    # 8. R3 — Write sidecars, export, and verify archives
    export_res = _write_sidecars_and_export(
        work_root=work_root,
        output_dir=output_dir,
        repo_root=repo,
        phase=phase,
        status=final_status,
        source_commit=source_commit,
        run_records=completed_records,
        paired_comparisons=paired_comparisons,
        condition_pairings=condition_pairings,
        confirmation_pairing=confirmation_pairing,
        advancement=advancement,
        executed_scenarios=executed_scenarios,
        diagnostics_payload=diagnostics_payload,
        split_audit_sha256=split_audit_sha256,
        forecast=forecast,
        notes=invalid_errors + (["Graceful deadline preemption triggered."] if deadline_hit else []),
        status_extra={
            "completed_jobs": [r["job_id"] for r in completed_records],
            "expected_jobs": [j["job_id"] for j in phase_jobs],
            "errors": invalid_errors,
        },
    )
    if final_status == "COMPLETE":
        state_log["state"] = "COMPLETE"
    _save_state_machine(work_root, state_log)
    return {
        "status": final_status,
        "state": state_log["state"],
        "split_audit_sha256": split_audit_sha256,
        "completed_jobs": len(completed_records),
        "expected_jobs": len(phase_jobs),
        "errors": invalid_errors,
        "condition_pairings": condition_pairings,
        "confirmation_pairing": confirmation_pairing,
        "advancement": advancement,
        "export": export_res,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ELENDRIL V8 pure campaign orchestrator")
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minutes", type=float, default=330.0)
    parser.add_argument("--owner-enabled", action="store_true")
    parser.add_argument("--winner-arm", default=None)
    parser.add_argument("--comparator-arm", default=None)
    parser.add_argument("--confirmation-seed", type=int, default=None)
    parser.add_argument("--cpu-smoke", action="store_true")
    args = parser.parse_args(argv)

    res = run_campaign(
        phase=args.phase,
        source_commit=args.source,
        work_root=args.work_root,
        output_dir=args.output,
        minutes=args.minutes,
        owner_enabled=args.owner_enabled,
        winner_arm=args.winner_arm,
        comparator_arm=args.comparator_arm,
        confirmation_seed=args.confirmation_seed,
        cpu_smoke=args.cpu_smoke,
    )
    print(json.dumps(res, indent=2))
    return 0 if res["status"] in ("COMPLETE", "NOT_RUN", "PARTIAL", "NO_GO") else 1


__all__ = [
    "DEFAULT_SAFETY_RESERVE_SECONDS",
    "DEFAULT_WORK_MARGIN",
    "estimate_campaign_storage_bytes",
    "expected_updates_for_job",
    "forecast_campaign_runtime",
    "main",
    "run_campaign",
    "validate_job_completion_artifacts",
]

if __name__ == "__main__":
    raise SystemExit(main())
