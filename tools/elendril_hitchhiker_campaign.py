"""Source-pinned 2-GPU Hitchhiker experiment campaign coordinator and dry-run validator."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any
from uuid import uuid4
import zipfile

from anra_v5.elendril.codec import canonical
from anra_v5.elendril.hitchhiker_data import (
    V6Dataset,
    arm_comparison_table,
    assert_all_stage_disjoint,
    build_counterfactual_suite,
    build_health_probes,
    build_structural_holdout_suite,
    build_training_stream,
)
from anra_v5.elendril.hitchhiker_evaluation import (
    evaluate_formation_gate,
    evaluate_screening_gate,
    independent_rescore_hitchhiker,
)
from anra_v5.elendril.hitchhiker_protocol import (
    AUDITED_V717_COMMIT,
    FORMATION_GATE,
    HISTORICAL_S20_LOWLR_MANIFESTS,
    PROTOCOL_SHA,
    SEEDS,
    STAGE2_ARMS,
    resolve_scale_profile,
    stage_jobs,
    validate_stage_selection,
)
from anra_v5.elendril.hitchhiker_training import resolve_exposure_schedule
from anra_v5.elendril.v6_data import assert_disjoint as assert_v6_disjoint, make_dataset, read_dataset, write_dataset
from tools.elendril_cloud import admit_devices, check_source, check_torch, device_env, inventory
from tools.elendril_hitchhiker_export import finalize_hitchhiker, handoff_hitchhiker_archive, restore_hitchhiker_archive
from tools.elendril_hitchhiker_runner import resolve_executing_source_commit, verify_and_rescore_stage0_package
from tools.elendril_hitchhiker_worker import (
    hitchhiker_case_from_record,
    load_suite_bundle,
    serialize_suite_bundle,
)
from tools.elendril_v6_export import _write_derived, immutable_json
from v5_training.checkpoint_lock import checkpoint_writer


REPO_ROOT = Path(__file__).resolve().parents[1]
PHASE_TO_STAGE = {
    "stage0": "stage0_checkpoint_audit",
    "stage1": "stage1_s3_screen",
    "stage2": "stage2_followups",
    "stage3": "stage3_scale_confirmation",
}
PROD_STORAGE_BYTES = {
    "stage0": 2 << 30,
    "stage1": 14 << 30,
    "stage2": 14 << 30,
    "stage3": 16 << 30,
}


def storage_admission(output: Path, required_bytes: int, *, minimum_free_gib_floor: int = 1 << 30) -> dict[str, Any]:
    """Verify real writable output directory and free disk headroom before execution."""
    output.mkdir(parents=True, exist_ok=True)
    probe = output.resolve(strict=True)
    free = shutil.disk_usage(probe).free
    headroom = max(required_bytes, minimum_free_gib_floor)
    if free < headroom:
        raise ValueError(
            f"NO_GO storage admission: {free} bytes free at {probe}, require >= {headroom} bytes"
        )
    return {
        "status": "ADMITTED",
        "path": str(probe),
        "free_bytes": free,
        "required_bytes": headroom,
    }


def _env_for_gpu(gpu: Any | None, *, cpu_canary: bool = False) -> dict[str, str]:
    if cpu_canary or gpu is None:
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        env.update(PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        return env
    return device_env(gpu)


def _run_worker(cmd: list[str], *, env: dict[str, str], log_path: Path | None = None) -> dict[str, Any]:
    proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            f"$ {' '.join(cmd)}\n\n[stdout]\n{proc.stdout}\n\n[stderr]\n{proc.stderr}\n",
            encoding="utf-8",
        )
    if proc.returncode not in (0, 2):
        raise RuntimeError(f"worker failed ({proc.returncode}): {proc.stderr.strip() or proc.stdout.strip()}")
    return {"returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}


def prepare_hitchhiker_data(
    output: Path,
    *,
    train_bytes: int = 2_200_000,
    val_worlds: int = 256,
    conf_worlds: int = 512,
    struct_worlds: int = 64,
    probe_worlds: int = 64,
    orig_val_bytes: int = 60_000,
) -> dict[str, Any]:
    """Generate, custody-verify, and atomically write immutable Hitchhiker evaluation & train inputs."""
    prepared = output / "prepared"
    manifest_path = prepared / "prepared_manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            existing.get("protocol_sha256") == PROTOCOL_SHA
            and existing.get("train_bytes") == train_bytes
            and existing.get("val_worlds") == val_worlds
            and existing.get("conf_worlds") == conf_worlds
            and existing.get("struct_worlds") == struct_worlds
            and existing.get("probe_worlds", probe_worlds) == probe_worlds
        ):
            # Re-verify serialized suite integrity before reusing across sessions.
            v_suite = load_suite_bundle(prepared / "validation_suite.json")
            c_suite = load_suite_bundle(prepared / "confirmation_suite.json")
            s_suite = load_suite_bundle(prepared / "structural_suite.json")
            v_probes = load_suite_bundle(prepared / "validation_probes.json")
            read_dataset(prepared / "train_base.json")
            v_orig = read_dataset(prepared / "validation_orig.json")
            if "val_suite_cohort_counts" not in existing:
                existing = {
                    **existing,
                    "probe_worlds": probe_worlds,
                    "val_suite_cohort_counts": {k: len(v) for k, v in v_suite["cohorts"].items()},
                    "val_suite_cases": sum(len(v) for v in v_suite["cohorts"].values()),
                    "conf_suite_cohort_counts": {k: len(v) for k, v in c_suite["cohorts"].items()},
                    "conf_suite_cases": sum(len(v) for v in c_suite["cohorts"].values()),
                    "struct_suite_cohort_counts": {k: len(v) for k, v in s_suite["cohorts"].items()},
                    "struct_suite_cases": sum(len(v) for v in s_suite["cohorts"].values()),
                    "val_probes_counts": {k: len(v) for k, v in v_probes.items()},
                    "val_probes_cases": sum(len(v) for v in v_probes.values()),
                    "val_orig_cases": len(v_orig.cases),
                    "val_equivariance_cases": len(v_suite["cohorts"].get("factor-entity-permutation", ())),
                    "conf_equivariance_cases": len(c_suite["cohorts"].get("factor-entity-permutation", ())),
                }
            return existing

    prepared.mkdir(parents=True, exist_ok=True)
    train_base = make_dataset("train", target_bytes=train_bytes, seed=606)
    val_orig = make_dataset("validation", target_bytes=orig_val_bytes, seed=707)
    val_suite = build_counterfactual_suite(seed=17071, worlds_per_hop=val_worlds, split="validation")
    conf_suite = build_counterfactual_suite(seed=17072, worlds_per_hop=conf_worlds, split="confirmation")
    struct_suite = build_structural_holdout_suite(seed=37073, worlds_per_topology=struct_worlds)
    probes = build_health_probes(seed=27173, worlds_per_probe=probe_worlds, split="validation")

    val_bundle = V6Dataset(
        split="validation",
        cases=(*(c for v in val_suite["cohorts"].values() for c in v), *(c for p in probes.values() for c in p)),
        examples=(),
        samples=(),
        manifest={"version": "hitchhiker-val-bundle-v1"},
    )
    conf_bundle = V6Dataset(
        split="confirmation",
        cases=tuple(c for v in conf_suite["cohorts"].values() for c in v),
        examples=(),
        samples=(),
        manifest={"version": "hitchhiker-conf-bundle-v1"},
    )
    custody = assert_all_stage_disjoint(
        train_base,
        val_bundle,
        conf_bundle,
        structural_cohorts=struct_suite,
    )
    assert_v6_disjoint(train_base, val_orig)

    write_dataset(train_base, prepared / "train_base.json")
    write_dataset(val_orig, prepared / "validation_orig.json")
    _write_derived(prepared / "validation_suite.json", canonical(serialize_suite_bundle(val_suite)) + b"\n")
    _write_derived(prepared / "confirmation_suite.json", canonical(serialize_suite_bundle(conf_suite)) + b"\n")
    _write_derived(prepared / "structural_suite.json", canonical(serialize_suite_bundle(struct_suite)) + b"\n")
    _write_derived(prepared / "validation_probes.json", canonical(serialize_suite_bundle(probes)) + b"\n")
    _write_derived(prepared / "split_custody.json", canonical(custody) + b"\n")

    # Save baseline exposure comparison across registered arms for auditability.
    sample_streams = {
        arm: build_training_stream(
            arm,
            base_data=train_base,
            total_updates=4,
            batch_size=4,
            data_seed=606,
            order_seed=7,
            heldout_suites=(val_bundle, conf_bundle, struct_suite, val_orig),
        )
        for arm in ("B0", "A", "B", "AB", "C", "C+penalty", "AB+C", "S20-contrast", "D-external")
    }
    comp_table = arm_comparison_table(sample_streams)
    _write_derived(prepared / "arm_comparison_table.json", canonical(comp_table) + b"\n")

    manifest = {
        "schema": "elendril-hitchhiker-prepared-v1",
        "protocol_sha256": PROTOCOL_SHA,
        "train_bytes": train_bytes,
        "train_data_sha256": train_base.identity,
        "train_samples": len(train_base.samples),
        "val_worlds": val_worlds,
        "conf_worlds": conf_worlds,
        "struct_worlds": struct_worlds,
        "probe_worlds": probe_worlds,
        "val_suite_cohort_counts": {k: len(v) for k, v in val_suite["cohorts"].items()},
        "val_suite_cases": sum(len(v) for v in val_suite["cohorts"].values()),
        "conf_suite_cohort_counts": {k: len(v) for k, v in conf_suite["cohorts"].items()},
        "conf_suite_cases": sum(len(v) for v in conf_suite["cohorts"].values()),
        "struct_suite_cohort_counts": {k: len(v) for k, v in struct_suite["cohorts"].items()},
        "struct_suite_cases": sum(len(v) for v in struct_suite["cohorts"].values()),
        "val_probes_counts": {k: len(v) for k, v in probes.items()},
        "val_probes_cases": sum(len(v) for v in probes.values()),
        "val_orig_cases": len(val_orig.cases),
        "val_equivariance_cases": len(val_suite["cohorts"].get("factor-entity-permutation", ())),
        "conf_equivariance_cases": len(conf_suite["cohorts"].get("factor-entity-permutation", ())),
        "custody_status": custody["status"],
        "split_claims": custody["split_claims"],
    }
    _write_derived(manifest_path, canonical(manifest) + b"\n")
    return manifest


def _resolve_stage_jobs(
    phase: str,
    *,
    selection_receipt: dict[str, Any] | None,
    winning_arm: str | None,
    admitted_followups: list[str] | None,
) -> list[dict[str, Any]]:
    stage = PHASE_TO_STAGE[phase]
    if phase == "stage0":
        return [{"job_id": f"HH-stage0-S20-lowlr-s{s}", "stage": stage, "seed": s, "profile": "S20"} for s in SEEDS]
    if phase == "stage1":
        return stage_jobs(stage)
    if phase == "stage2":
        admitted = validate_stage_selection(stage, selection_receipt, winning_arm=winning_arm)
        if admitted_followups is not None:
            requested = list(admitted_followups)
            if not requested or len(set(requested)) != len(requested):
                raise ValueError("admitted_followups override must be non-empty and duplicate-free")
            unknown = [arm for arm in requested if arm not in STAGE2_ARMS]
            if unknown:
                raise ValueError(f"unknown Stage 2 followup arm(s): {unknown}")
            unadmitted = [arm for arm in requested if arm not in admitted["admitted_followups"]]
            if unadmitted:
                raise ValueError(
                    f"requested Stage 2 arm(s) {unadmitted} are not admitted by selection_receipt "
                    f"({admitted['admitted_followups']})"
                )
            selected = requested
        else:
            selected = list(admitted["admitted_followups"])
        return [j for j in stage_jobs(stage) if j["arm"] in selected]
    admitted = validate_stage_selection(stage, selection_receipt, winning_arm=winning_arm)
    return stage_jobs(stage, winning_arm=admitted["winning_arm"])


def _estimate_job_eval_seconds(
    job: dict[str, Any],
    m: dict[str, Any],
    prepared_manifest: dict[str, Any] | None,
) -> float:
    comps = m.get("eval_components")
    if not isinstance(comps, dict):
        return (m.get("eval_sample_ms", 0.0) / 1000.0) * 6.0

    split = "confirmation" if job["stage"] == "stage3_scale_confirmation" else "validation"
    suite_comp = comps.get("suite") or {}
    cohort_ms_per_case = suite_comp.get("cohort_ms_per_case") or {}
    mean_ms_per_case = (
        sum(cohort_ms_per_case.values()) / len(cohort_ms_per_case)
        if cohort_ms_per_case
        else 0.0
    )

    if prepared_manifest is not None:
        cohort_counts = (
            prepared_manifest.get("conf_suite_cohort_counts")
            if split == "confirmation"
            else prepared_manifest.get("val_suite_cohort_counts")
        ) or {}
    else:
        cohort_counts = suite_comp.get("cohort_full_counts") or {}

    eval_s = 0.0
    for cname, cnt in cohort_counts.items():
        ms_case = cohort_ms_per_case.get(cname, mean_ms_per_case)
        eval_s += (ms_case / 1000.0) * cnt

    orig_comp = comps.get("orig") or {}
    orig_cnt = (
        prepared_manifest.get("val_orig_cases", orig_comp.get("full_cases", 0))
        if prepared_manifest is not None
        else orig_comp.get("full_cases", 0)
    )
    eval_s += (orig_comp.get("ms_per_case", 0.0) / 1000.0) * orig_cnt

    if job["stage"] in ("stage1_s3_screen", "stage2_followups"):
        probes_comp = comps.get("probes") or {}
        probes_cnt = (
            prepared_manifest.get("val_probes_cases", probes_comp.get("full_cases", 0))
            if prepared_manifest is not None
            else probes_comp.get("full_cases", 0)
        )
        eval_s += (probes_comp.get("ms_per_case", 0.0) / 1000.0) * probes_cnt
    else:
        struct_comp = comps.get("structural") or {}
        struct_cnt = (
            prepared_manifest.get("struct_suite_cases", struct_comp.get("full_cases", 0))
            if prepared_manifest is not None
            else struct_comp.get("full_cases", 0)
        )
        eval_s += (struct_comp.get("ms_per_case", 0.0) / 1000.0) * struct_cnt

    eq_comp = comps.get("equivariance") or {}
    if prepared_manifest is not None:
        eq_cnt = (
            prepared_manifest.get("conf_equivariance_cases", eq_comp.get("full_cases", 0))
            if split == "confirmation"
            else prepared_manifest.get("val_equivariance_cases", eq_comp.get("full_cases", 0))
        )
    else:
        eq_cnt = eq_comp.get("full_cases", 0)
    eval_s += (eq_comp.get("ms_per_case", 0.0) / 1000.0) * eq_cnt
    return eval_s


def _forecast_phase_seconds(
    jobs: list[dict[str, Any]],
    profile_receipt: dict[str, Any],
    *,
    train_samples: int,
    prepared_manifest: dict[str, Any] | None = None,
    gpu_workers: int = 2,
    updates_override: int | None = None,
    remaining_budget_s: float | None = None,
    elapsed_prep_and_profile_s: float | None = None,
) -> dict[str, Any]:
    measured = profile_receipt["profiles"]
    stream_probe = profile_receipt.get("stream_probe") or {}
    per_job = []
    total_replay_s = 0.0
    total_ckpt_bytes = 0
    for job in jobs:
        prof = job["profile"]
        m = measured[prof if prof in measured else "S3"]
        updates = updates_override or (
            math.ceil(train_samples / job["batch_size"]) * job["epochs"] if job.get("training_updates") is None else job["training_updates"]
        )
        train_s = (m["ms_per_update"] / 1000.0) * updates
        save_interval = updates if updates_override else max(1, min(100, updates // 4))
        num_saves = (updates // save_interval) + (0 if updates % save_interval == 0 else 1) + 1
        ckpt_save_s = (m.get("checkpoint_write_per_save_ms", m.get("checkpoint_write_ms", 0.0)) / 1000.0) * num_saves
        total_ckpt_bytes += int(m.get("checkpoint_bytes", 0))

        sp = m.get("stream_probe") or stream_probe
        stream_s = 0.0
        if updates_override is None and sp:
            base_ov = float(sp.get("base_stream_overhead_s", 0.0))
            per_upd_s = float(sp.get("ms_per_materialized_update", 0.0)) / 1000.0
            if job["arm"] in ("A", "AB", "AB+C", "S20-contrast"):
                stream_s = base_ov + per_upd_s * updates
            else:
                stream_s = base_ov

        eval_s = _estimate_job_eval_seconds(job, m, prepared_manifest)
        replay_s = min(60.0, 0.25 * eval_s)
        total_replay_s += replay_s
        # Add +15% safety reserve on per-job execution.
        est_sec = (train_s + ckpt_save_s + stream_s + eval_s) * 1.15
        per_job.append(
            {
                "job_id": job["job_id"],
                "profile": prof,
                "updates": updates,
                "train_seconds": train_s,
                "checkpoint_save_seconds": ckpt_save_s,
                "stream_build_seconds": stream_s,
                "eval_seconds": eval_s,
                "replay_seconds": replay_s,
                "estimated_seconds": est_sec,
            }
        )
    # Assign LPT across `gpu_workers` parallel workers.
    lanes = [0.0] * max(1, gpu_workers)
    for item in sorted(per_job, key=lambda r: r["estimated_seconds"], reverse=True):
        idx = min(range(len(lanes)), key=lambda i: lanes[i])
        lanes[idx] += item["estimated_seconds"]
    worker_wall_s = max(lanes) if lanes else 0.0
    export_reserve_s = max(5.0, (total_ckpt_bytes / (100 * 1024 * 1024))) if per_job else 0.0
    projected_wall_s = worker_wall_s + (total_replay_s + export_reserve_s) * 1.15
    return {
        "gpu_workers": gpu_workers,
        "jobs": len(per_job),
        "worker_wall_seconds": worker_wall_s,
        "replay_reserve_seconds": total_replay_s,
        "export_reserve_seconds": export_reserve_s,
        "projected_wall_seconds": projected_wall_s,
        "remaining_budget_seconds": remaining_budget_s,
        "elapsed_prep_and_profile_seconds": elapsed_prep_and_profile_s,
        "lane_seconds": lanes,
        "per_job": per_job,
    }


def _replay_job_eval(
    job_eval_path: Path,
    prepared: Path,
    *,
    expected_source: str,
    job: dict[str, Any],
    expected_split: str,
    run_receipt_path: Path,
) -> dict[str, Any]:
    if not job_eval_path.is_file():
        raise ValueError(f"missing job evaluation file: {job_eval_path}")
    if not run_receipt_path.is_file():
        raise ValueError(f"missing job run_receipt file: {run_receipt_path}")
    doc = json.loads(job_eval_path.read_text(encoding="utf-8"))
    run_receipt = json.loads(run_receipt_path.read_text(encoding="utf-8"))
    if doc.get("protocol_sha256") != PROTOCOL_SHA:
        raise ValueError(f"{job_eval_path} protocol_sha256 mismatch")
    if doc.get("source_commit") != expected_source:
        raise ValueError(f"{job_eval_path} source_commit mismatch")
    if doc.get("job_id") != job["job_id"] or run_receipt.get("job_id") != job["job_id"]:
        raise ValueError(f"{job_eval_path} job_id mismatch for {job['job_id']}")
    if doc.get("arm") != job["arm"]:
        raise ValueError(f"{job_eval_path} arm mismatch: {doc.get('arm')!r} != {job['arm']!r}")
    if doc.get("seed") != job["seed"]:
        raise ValueError(f"{job_eval_path} seed mismatch: {doc.get('seed')!r} != {job['seed']!r}")
    if doc.get("profile") != job["profile"]:
        raise ValueError(f"{job_eval_path} profile mismatch: {doc.get('profile')!r} != {job['profile']!r}")
    split = doc.get("split")
    if split != expected_split:
        raise ValueError(f"{job_eval_path} split mismatch: {split!r} != {expected_split!r}")
    ckpt_sha = doc.get("checkpoint_sha256")
    if not ckpt_sha or run_receipt.get("checkpoint_sha256") != ckpt_sha:
        raise ValueError(f"{job_eval_path} checkpoint_sha256 disagrees with run_receipt")

    suite = load_suite_bundle(prepared / ("confirmation_suite.json" if split == "confirmation" else "validation_suite.json"))
    suite_eval = doc["suite_eval"]
    replayed_cohorts = 0
    for name, cases in suite["cohorts"].items():
        if not cases:
            continue
        independent_rescore_hitchhiker(cases, suite_eval["records"][name])
        replayed_cohorts += 1
    orig_dev = read_dataset(prepared / "validation_orig.json")
    independent_rescore_hitchhiker(orig_dev.cases, doc["original_eval"]["record"])
    if doc.get("probes_eval"):
        probes = load_suite_bundle(prepared / "validation_probes.json")
        for p_name, plist in probes.items():
            if plist:
                independent_rescore_hitchhiker(plist, doc["probes_eval"]["records"][p_name])
    if doc.get("structural_eval"):
        struct = load_suite_bundle(prepared / "structural_suite.json")
        for name, cases in struct["cohorts"].items():
            if cases:
                independent_rescore_hitchhiker(cases, doc["structural_eval"]["records"][name])
    return {"status": "PASS", "replayed_cohorts": replayed_cohorts, "checkpoint_sha256": ckpt_sha}


def _arm_mean_gain(gate_doc: dict[str, Any]) -> float | None:
    if gate_doc.get("mean_primary_gain") is not None:
        return float(gate_doc["mean_primary_gain"])
    mech = gate_doc.get("mechanical_statistics") or {}
    val = mech.get("mean_primary_gain")
    return float(val) if val is not None else None


def _build_stage2_admission(
    arm_gates: dict[str, Any],
    jobs: list[dict[str, Any]],
    evals_by_job_id: dict[str, dict[str, Any]],
    *,
    locked: str | None,
    top_status: str,
) -> tuple[list[str], dict[str, str], dict[str, str]]:
    """Derive evidence-bound Stage 2 follow-up admissions and explicit exclusions from Stage 1 metrics."""
    max_invalid_by_arm: dict[str, float] = {}
    mean_eq_by_arm: dict[str, float] = {}
    for arm in ("B0", "A", "B", "AB", "C"):
        arm_docs = [evals_by_job_id[j["job_id"]] for j in jobs if j["arm"] == arm]
        inv_vals = [
            float(m["invalid_rate"])
            for d in arm_docs
            for m in (d.get("cohort_metrics") or {}).values()
            if m.get("invalid_rate") is not None
        ]
        max_invalid_by_arm[arm] = max(inv_vals) if inv_vals else 0.0
        eq_vals = [
            float((d.get("equivariance_report") or {})["argmax_equivariance_rate"])
            for d in arm_docs
            if (d.get("equivariance_report") or {}).get("argmax_equivariance_rate") is not None
        ]
        if eq_vals:
            mean_eq_by_arm[arm] = sum(eq_vals) / len(eq_vals)

    inv_limit = float(FORMATION_GATE["max_invalid_action_rate"])
    eq_limit = float(FORMATION_GATE["min_renaming_consistency"])
    admitted: list[str] = []
    followup_rationales: dict[str, str] = {}
    excluded_followups: dict[str, str] = {}

    if locked is not None:
        locked_gain = _arm_mean_gain(arm_gates[locked])
        admitted.append("AB+C")
        followup_rationales["AB+C"] = (
            f"Stage 1 arm {locked} achieved PASS_SCREENING (mean_primary_gain={locked_gain:.4f}); "
            "admit AB+C to test fresh+relabeling+coverage interaction."
        )
        admitted.append("D-external")
        followup_rationales["D-external"] = (
            f"Stage 1 arm {locked} passed screening; admit D-external as engineered canonicalizer comparator."
        )
        c_inv = max_invalid_by_arm.get("C", 0.0)
        locked_inv = max_invalid_by_arm.get(locked, 0.0)
        if c_inv > inv_limit or locked_inv > inv_limit:
            admitted.append("C+penalty")
            followup_rationales["C+penalty"] = (
                f"Observed invalid_rate (C={c_inv:.4f}, {locked}={locked_inv:.4f}) exceeds {inv_limit:.4f}; "
                "admit C+penalty to test illegal-action regularization."
            )
        else:
            excluded_followups["C+penalty"] = (
                f"Withheld: Stage 1 max invalid_rate (C={c_inv:.4f}, {locked}={locked_inv:.4f}) <= {inv_limit:.4f}."
            )
        excluded_followups["S20-contrast"] = (
            f"Withheld: Stage 1 {locked} already passed S3 screening and advances to Stage 3 S8/S20 confirmation."
        )
        return admitted, followup_rationales, excluded_followups

    ab_gain = _arm_mean_gain(arm_gates["AB"])
    c_gain = _arm_mean_gain(arm_gates["C"])
    c_inv = max_invalid_by_arm.get("C", 0.0)
    b0_inv = max_invalid_by_arm.get("B0", 0.0)

    excluded_followups["AB+C"] = (
        f"Withheld: no Stage 1 arm achieved PASS_SCREENING (top_status={top_status}, AB_gain={ab_gain})."
    )
    excluded_followups["S20-contrast"] = (
        "Withheld: Stage 1 S3 screening produced no PASS_SCREENING arm; capacity escalation to S20 is not warranted."
    )

    if (c_gain is not None and c_gain > 0.0) or c_inv > inv_limit or b0_inv > inv_limit:
        admitted.append("C+penalty")
        followup_rationales["C+penalty"] = (
            f"Diagnostic trigger: C_mean_primary_gain={c_gain}, max_invalid_rate(C={c_inv:.4f}, B0={b0_inv:.4f}) "
            f"exceeds threshold {inv_limit:.4f}."
        )
    else:
        excluded_followups["C+penalty"] = (
            f"Withheld: C_mean_primary_gain={c_gain} <= 0 and max_invalid_rate(C={c_inv:.4f}, B0={b0_inv:.4f}) <= {inv_limit:.4f}."
        )

    if mean_eq_by_arm and min(mean_eq_by_arm.values()) < eq_limit:
        worst_arm = min(mean_eq_by_arm, key=lambda a: mean_eq_by_arm[a])
        admitted.append("D-external")
        followup_rationales["D-external"] = (
            f"Diagnostic trigger: Stage 1 entity equivariance gap ({worst_arm} mean_argmax_equivariance="
            f"{mean_eq_by_arm[worst_arm]:.4f} < {eq_limit:.4f}); admit engineered slot canonicalizer control."
        )
    else:
        excluded_followups["D-external"] = (
            f"Withheld: Stage 1 equivariance rates ({mean_eq_by_arm}) did not trigger gap < {eq_limit:.4f}."
        )
    return admitted, followup_rationales, excluded_followups


def _evaluate_stage_outcomes(
    phase: str,
    jobs: list[dict[str, Any]],
    output: Path,
    prepared: Path,
    *,
    source: str,
    winning_arm: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    """Independently replay all job evaluations and apply fail-closed Stage 1/2/3 gates."""
    analysis = output / "analysis"
    if phase == "stage0":
        seed_summaries = {}
        replays = {}
        for job in jobs:
            sdir = analysis / "stage0" / f"seed_{job['seed']}"
            summary_file = sdir / "stage0_summary.json"
            doc = json.loads(summary_file.read_text(encoding="utf-8"))
            seed_summaries[job["seed"]] = doc
            if doc.get("status") == "EXECUTED":
                suite = load_suite_bundle(prepared / "validation_suite.json")
                probes = load_suite_bundle(prepared / "validation_probes.json")
                orig = read_dataset(prepared / "validation_orig.json")
                struct = load_suite_bundle(prepared / "structural_suite.json")
                replays[f"seed_{job['seed']}"] = verify_and_rescore_stage0_package(
                    sdir / "stage0_package",
                    suite=suite,
                    probes=probes,
                    original_dev=orig,
                    structural_suite=struct,
                )
        pending = [s for s, d in seed_summaries.items() if d.get("status") == "CHECKPOINT_AUDIT_PENDING"]
        replay = {"status": "PENDING" if pending else "PASS", "seeds": replays, "pending_seeds": pending}
        gate = {"schema": "elendril-hitchhiker-stage0-gate-v1", "status": "PENDING" if pending else "EXECUTED", "pending_seeds": pending}
        _write_derived(analysis / "stage0_gate_receipt.json", canonical(gate) + b"\n")
        return replay, gate, None

    expected_split = "confirmation" if phase == "stage3" else "validation"
    evals_by_job_id: dict[str, dict[str, Any]] = {}
    replay_jobs = {}
    for job in jobs:
        jid = job["job_id"]
        if jid in evals_by_job_id:
            raise ValueError(f"duplicate job_id in stage evaluation: {jid}")
        jdir = analysis / "jobs" / jid
        epath = jdir / "final_eval.json"
        rpath = jdir / "run_receipt.json"
        replay_jobs[jid] = _replay_job_eval(
            epath,
            prepared,
            expected_source=source,
            job=job,
            expected_split=expected_split,
            run_receipt_path=rpath,
        )
        evals_by_job_id[jid] = json.loads(epath.read_text(encoding="utf-8"))

    replay = {"status": "PASS", "jobs": replay_jobs}

    if phase == "stage1":
        b0_reports = {j["seed"]: evals_by_job_id[j["job_id"]]["suite_eval"] for j in jobs if j["arm"] == "B0"}
        arm_gates = {}
        passing = []
        for arm in ("A", "B", "AB", "C"):
            arm_reports = {j["seed"]: evals_by_job_id[j["job_id"]]["suite_eval"] for j in jobs if j["arm"] == arm}
            g = evaluate_screening_gate(arm_reports, b0_reports)
            arm_gates[arm] = g
            if g["status"] == "PASS_SCREENING":
                passing.append((g["mean_primary_gain"], arm, g))
        passing.sort(reverse=True)
        locked = passing[0][1] if passing else None
        top_status = "PASS_SCREENING" if passing else arm_gates["AB"]["status"]
        gate = {"schema": "elendril-hitchhiker-stage1-gate-v1", "arms": arm_gates, "locked_winner": locked, "status": top_status}
        gate_bytes = canonical(gate) + b"\n"
        _write_derived(analysis / "gate_receipt.json", gate_bytes)
        gate_receipt_sha256 = sha256(gate_bytes).hexdigest()

        admitted_followups, followup_rationales, excluded_followups = _build_stage2_admission(
            arm_gates,
            jobs,
            evals_by_job_id,
            locked=locked,
            top_status=top_status,
        )
        selection = {
            "schema": "elendril-hitchhiker-selection-v1",
            "protocol_sha256": PROTOCOL_SHA,
            "source_commit": source,
            "gate_receipt_sha256": gate_receipt_sha256,
            "stage1_completed": True,
            "screening_status": top_status,
            "split": "validation",
            "winning_arm": locked,
            "model_profile": "S8" if locked else None,
            "admitted_followups": admitted_followups,
            "followup_rationales": followup_rationales,
            "excluded_followups": excluded_followups,
            "rationale": (
                f"Stage 1 dev screening results across arms {sorted(arm_gates)}: "
                f"top_status={top_status}, winning_arm={locked}, admitted_followups={admitted_followups}."
            ),
        }
        _write_derived(analysis / "selection_receipt.json", canonical(selection) + b"\n")
        return replay, gate, selection

    if phase == "stage2":
        arms_summary: dict[str, dict[str, Any]] = {}
        for job in jobs:
            d = evals_by_job_id[job["job_id"]]
            arms_summary.setdefault(job["arm"], {})[str(job["seed"])] = {
                "job_id": job["job_id"],
                "profile": job["profile"],
                "cohort_metrics": d["cohort_metrics"],
                "original_kind_accuracies": d["original_kind_accuracies"],
            }
        gate = {
            "schema": "elendril-hitchhiker-stage2-gate-v1",
            "status": "EXECUTED",
            "arms": arms_summary,
        }
        _write_derived(analysis / "stage2_gate_receipt.json", canonical(gate) + b"\n")
        return replay, gate, None

    # Stage 3: apply fail-closed formation gate on every confirmation run independently!
    per_job_gates = {}
    for job in jobs:
        d = evals_by_job_id[job["job_id"]]
        per_job_gates[job["job_id"]] = evaluate_formation_gate(
            d["suite_eval"],
            equivariance_report=d["equivariance_report"],
        )
    treatment = winning_arm or [j["arm"] for j in jobs if j["arm"] != "B0"][0]
    treatment_jobs = [j for j in jobs if j["arm"] == treatment]
    per_scale: dict[str, Any] = {}
    for scale in ("S8", "S20"):
        scale_jobs = [j for j in treatment_jobs if j["profile"] == scale]
        scale_statuses = [per_job_gates[j["job_id"]]["status"] for j in scale_jobs]
        scale_status = (
            "FORMATION_QUALIFIED"
            if scale_statuses and all(s == "FORMATION_QUALIFIED" for s in scale_statuses)
            else scale_statuses[0]
            if len(set(scale_statuses)) == 1 and scale_statuses
            else "NOT_QUALIFIED"
        )
        per_scale[scale] = {
            "status": scale_status,
            "jobs": [j["job_id"] for j in scale_jobs],
            "per_job_statuses": {j["job_id"]: per_job_gates[j["job_id"]]["status"] for j in scale_jobs},
        }
    all_treatment_statuses = [per_job_gates[j["job_id"]]["status"] for j in treatment_jobs]
    overall = (
        "FORMATION_QUALIFIED"
        if per_scale["S8"]["status"] == "FORMATION_QUALIFIED" and per_scale["S20"]["status"] == "FORMATION_QUALIFIED"
        else all_treatment_statuses[0]
        if len(set(all_treatment_statuses)) == 1 and all_treatment_statuses
        else "NOT_QUALIFIED"
    )
    gate = {
        "schema": "elendril-hitchhiker-stage3-gate-v1",
        "treatment_arm": treatment,
        "status": overall,
        "per_scale": per_scale,
        "per_job": per_job_gates,
    }
    _write_derived(analysis / "formation_gate_receipt.json", canonical(gate) + b"\n")
    return replay, gate, None


def _detect_archive_phase(zip_path: Path) -> str:
    """Inspect a Hitchhiker export archive's verified identity to determine its parent phase."""
    zp = Path(zip_path)
    if zp.is_symlink() or not zp.is_file():
        raise ValueError(f"handoff archive must be an existing regular file: {zp}")
    with zipfile.ZipFile(zp, "r") as zf:
        try:
            raw = zf.read("analysis/identity.json")
        except KeyError as exc:
            raise ValueError(f"{zp} is missing analysis/identity.json") from exc
    ident = json.loads(raw.decode("utf-8"))
    if not isinstance(ident, dict) or ident.get("phase") not in PHASE_TO_STAGE:
        raise ValueError(f"{zp} has invalid identity phase: {ident.get('phase') if isinstance(ident, dict) else ident!r}")
    return str(ident["phase"])


def _valid_reusable_profile(
    prof_path: Path,
    *,
    phase: str,
    source: str,
    train_data_sha256: str,
    gpu_uuids: list[str],
) -> dict[str, Any] | None:
    """Reuse an existing profile_receipt.json only when phase, source, data, and GPU lanes match."""
    if prof_path.is_symlink() or not prof_path.is_file():
        return None
    try:
        doc = json.loads(prof_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(doc, dict) or doc.get("status") != "PASS":
        return None
    if (
        doc.get("phase") != phase
        or doc.get("source") != source
        or doc.get("protocol_sha256") != PROTOCOL_SHA
        or doc.get("train_data_sha256") != train_data_sha256
        or doc.get("gpu_uuids") != gpu_uuids
    ):
        return None
    profiles = doc.get("profiles")
    if not isinstance(profiles, dict):
        return None
    for scale in ("S3", "S8", "S20"):
        entry = profiles.get(scale)
        if not isinstance(entry, dict) or not isinstance(entry.get("eval_components"), dict):
            return None
        if scale == "S3" and entry.get("recovery_verified") is not True:
            return None
    return doc


def _combine_lane_profiles(
    lane_docs: list[dict[str, Any]],
    *,
    phase: str,
    source: str,
    train_data_sha256: str,
    gpu_uuids: list[str],
) -> dict[str, Any]:
    """Conservatively merge per-lane GPU profile receipts by taking max latencies across lanes."""
    if not lane_docs:
        raise ValueError("lane_docs must not be empty")
    combined_profiles: dict[str, Any] = {}
    for scale in ("S3", "S8", "S20"):
        entries = [d["profiles"][scale] for d in lane_docs if scale in (d.get("profiles") or {})]
        if not entries:
            continue
        merged = dict(entries[0])
        for key in (
            "wall_seconds",
            "ms_per_update",
            "peak_cuda_mb",
            "checkpoint_bytes",
            "checkpoint_write_ms",
            "checkpoint_write_per_save_ms",
            "eval_sample_ms",
            "projected_full_eval_seconds",
        ):
            vals = [float(e[key]) for e in entries if e.get(key) is not None]
            if vals:
                merged[key] = int(max(vals)) if key == "checkpoint_bytes" else max(vals)
        if any("recovery_verified" in e for e in entries):
            merged["recovery_verified"] = all(bool(e.get("recovery_verified")) for e in entries)

        merged_comps: dict[str, Any] = {}
        comp_keys = {k for e in entries for k in (e.get("eval_components") or {}).keys()}
        for ck in sorted(comp_keys):
            centries = [e["eval_components"][ck] for e in entries if ck in (e.get("eval_components") or {})]
            cmerged = dict(centries[0])
            for num_k in ("ms_per_case", "sample_ms", "projected_s"):
                vals = [float(ce[num_k]) for ce in centries if ce.get(num_k) is not None]
                if vals:
                    cmerged[num_k] = max(vals)
            for map_k in ("cohort_ms_per_case", "cohort_projected_s"):
                if any(isinstance(ce.get(map_k), dict) for ce in centries):
                    mkeys = {mk for ce in centries for mk in (ce.get(map_k) or {}).keys()}
                    cmerged[map_k] = {
                        mk: max(
                            float(ce[map_k][mk])
                            for ce in centries
                            if isinstance(ce.get(map_k), dict) and ce[map_k].get(mk) is not None
                        )
                        for mk in sorted(mkeys)
                    }
            merged_comps[ck] = cmerged
        merged["eval_components"] = merged_comps
        combined_profiles[scale] = merged

    stream_entries = [d.get("stream_probe") for d in lane_docs if isinstance(d.get("stream_probe"), dict)]
    combined_stream: dict[str, Any] = {}
    if stream_entries:
        combined_stream = dict(stream_entries[0])
        for sk in ("base_stream_overhead_s", "ms_per_materialized_update", "materialized_updates_measured"):
            vals = [float(se[sk]) for se in stream_entries if se.get(sk) is not None]
            if vals:
                combined_stream[sk] = int(max(vals)) if sk == "materialized_updates_measured" else max(vals)
    for prof_entry in combined_profiles.values():
        if combined_stream:
            prof_entry["stream_probe"] = combined_stream

    return {
        "schema": "elendril-hitchhiker-profile-v1",
        "status": "PASS",
        "phase": phase,
        "source": source,
        "protocol_sha256": PROTOCOL_SHA,
        "train_data_sha256": train_data_sha256,
        "gpu_uuids": list(gpu_uuids),
        "lane_receipts": [
            {
                "lane_index": d.get("lane_index", idx),
                "device_name": d.get("device_name"),
                "profile_wall_seconds": d.get("profile_wall_seconds"),
            }
            for idx, d in enumerate(lane_docs)
        ],
        "stream_probe": combined_stream,
        "profiles": combined_profiles,
    }


def run(
    *,
    output: Path,
    source: str | None,
    phase: str = "stage1",
    minutes: float = 540.0,
    owner_enabled: bool = False,
    stage0_checkpoint_root: Path | None = None,
    winning_arm: str | None = None,
    admitted_followups: list[str] | None = None,
    selection_receipt_path: Path | None = None,
    resume_archive: Path | None = None,
    handoff_archive: Path | None = None,
    session_deadline_s: float | None = None,
    worlds_per_hop: int | None = None,
    train_bytes: int | None = None,
    updates_override: int | None = None,
    cpu_interrupt_after: int | None = None,
    dry_run: bool = False,
    cpu_canary: bool = False,
) -> dict[str, Any]:
    """Run dry-run preflight, GPU-isolated Hitchhiker campaign phase, replay verification, and export."""
    session_t0 = time.monotonic()
    if not dry_run and not owner_enabled and not cpu_canary:
        raise ValueError("campaign execution requires --owner-enabled (or --dry-run / bounded --cpu-canary)")
    if cpu_interrupt_after is not None and not cpu_canary:
        raise ValueError("cpu_interrupt_after is forbidden outside --cpu-canary tests")
    if resume_archive is not None and handoff_archive is not None:
        raise ValueError("pass at most one of resume_archive (same-phase) or handoff_archive (cross-phase)")
    if session_deadline_s is not None and session_deadline_s <= 0.0:
        raise ValueError("session_deadline_s must be positive when provided")
    if cpu_canary and updates_override is None:
        updates_override = 2

    resolved_source = resolve_executing_source_commit(source)
    if not cpu_canary:
        check_source(REPO_ROOT, resolved_source)

    if resume_archive is not None:
        restore_hitchhiker_archive(resume_archive, output, source=resolved_source, phase=phase)
    elif handoff_archive is not None:
        parent_phase = _detect_archive_phase(handoff_archive)
        handoff_hitchhiker_archive(
            handoff_archive,
            output,
            source=resolved_source,
            from_phase=parent_phase,
            to_phase=phase,
            winning_arm=winning_arm,
        )

    selection = None
    sel_candidate = selection_receipt_path or (output / "analysis" / "selection_receipt.json")
    if sel_candidate.is_file():
        selection = json.loads(sel_candidate.read_text(encoding="utf-8"))

    effective_winning_arm = winning_arm or (
        str(selection["winning_arm"]) if isinstance(selection, dict) and selection.get("winning_arm") else None
    )

    jobs = _resolve_stage_jobs(
        phase,
        selection_receipt=selection,
        winning_arm=effective_winning_arm,
        admitted_followups=admitted_followups,
    )

    t_bytes = train_bytes or (12_000 if cpu_canary else 2_200_000)
    v_worlds = worlds_per_hop or (2 if cpu_canary else 256)
    c_worlds = worlds_per_hop or (2 if cpu_canary else 512)
    s_worlds = min(v_worlds, 4) if cpu_canary else 64
    p_worlds = min(v_worlds, 4) if cpu_canary else 64

    if dry_run:
        storage_bytes = 64 << 20 if cpu_canary else PROD_STORAGE_BYTES[phase]
        storage = storage_admission(output, storage_bytes, minimum_free_gib_floor=64 << 20 if cpu_canary else 1 << 30)
        return {
            "schema": "elendril-hitchhiker-campaign-v1",
            "status": "DRY_RUN_READY",
            "claim_status": "IMPLEMENTED",
            "source": resolved_source,
            "audited_v717_ancestor": AUDITED_V717_COMMIT,
            "protocol_sha256": PROTOCOL_SHA,
            "phase": phase,
            "stage": PHASE_TO_STAGE[phase],
            "jobs": [j["job_id"] for j in jobs],
            "job_count": len(jobs),
            "storage_admission": storage,
            "config": {"train_bytes": t_bytes, "val_worlds": v_worlds, "conf_worlds": c_worlds, "struct_worlds": s_worlds},
        }

    storage = storage_admission(
        output,
        64 << 20 if cpu_canary else PROD_STORAGE_BYTES[phase],
        minimum_free_gib_floor=64 << 20 if cpu_canary else 1 << 30,
    )
    gpus: tuple[Any, ...] = ()
    runtime_check: list[dict[str, Any]] = []
    if not cpu_canary:
        gpus = inventory()
        admit_devices(gpus, 2)
        runtime_check = check_torch(REPO_ROOT, gpus)

    run_id = f"hh-{phase}-{uuid4().hex[:12]}"
    with checkpoint_writer(output, filename=".pending-hitchhiker-campaign-writer"):
        prepared_manifest = prepare_hitchhiker_data(
            output,
            train_bytes=t_bytes,
            val_worlds=v_worlds,
            conf_worlds=c_worlds,
            struct_worlds=s_worlds,
            probe_worlds=p_worlds,
            orig_val_bytes=6_000 if cpu_canary else 60_000,
        )
        prepared_dir = output / "prepared"
        analysis_dir = output / "analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)
        prof_receipt_path = analysis_dir / "profile_receipt.json"
        if phase == "stage0":
            profile_receipt = {"status": "SKIPPED_FOR_STAGE0", "profiles": {}}
            forecast = {
                "gpu_workers": 2,
                "jobs": len(jobs),
                "projected_wall_seconds": 0.0,
                "lane_seconds": [0.0, 0.0],
                "per_job": [],
            }
        else:
            gpu_uuids = [] if cpu_canary else [str(g.uuid) for g in gpus[:2]]
            reused_profile = _valid_reusable_profile(
                prof_receipt_path,
                phase=phase,
                source=resolved_source,
                train_data_sha256=str(prepared_manifest["train_data_sha256"]),
                gpu_uuids=gpu_uuids,
            )
            if reused_profile is not None:
                profile_receipt = reused_profile
            else:
                lane_indices = [0] if cpu_canary else [0, 1]

                def _profile_lane(lane_idx: int) -> dict[str, Any]:
                    lane_prof_path = analysis_dir / f"profile_lane_{lane_idx}.json"
                    profile_cmd = [
                        sys.executable, "-m", "tools.elendril_hitchhiker_worker",
                        "--mode", "profile",
                        "--source", resolved_source,
                        "--prepared", str(prepared_dir),
                        "--receipt", str(lane_prof_path),
                        "--lane-index", str(lane_idx),
                    ]
                    if cpu_canary:
                        profile_cmd.append("--cpu-canary")
                    _run_worker(
                        profile_cmd,
                        env=_env_for_gpu(gpus[lane_idx] if gpus else None, cpu_canary=cpu_canary),
                        log_path=analysis_dir / "logs" / f"profile_lane_{lane_idx}.txt",
                    )
                    return json.loads(lane_prof_path.read_text(encoding="utf-8"))

                if len(lane_indices) == 1:
                    lane_docs = [_profile_lane(0)]
                else:
                    with ThreadPoolExecutor(max_workers=2) as pool:
                        futs = [pool.submit(_profile_lane, idx) for idx in lane_indices]
                        lane_docs = [fut.result() for fut in futs]
                profile_receipt = _combine_lane_profiles(
                    lane_docs,
                    phase=phase,
                    source=resolved_source,
                    train_data_sha256=str(prepared_manifest["train_data_sha256"]),
                    gpu_uuids=gpu_uuids,
                )
                _write_derived(prof_receipt_path, canonical(profile_receipt) + b"\n")

            elapsed_prep_and_profile_s = time.monotonic() - session_t0
            remaining_budget_s = max(0.0, minutes * 60.0 - elapsed_prep_and_profile_s)
            forecast = _forecast_phase_seconds(
                jobs,
                profile_receipt,
                train_samples=prepared_manifest["train_samples"],
                prepared_manifest=prepared_manifest,
                gpu_workers=2,
                updates_override=updates_override if cpu_canary else None,
                remaining_budget_s=remaining_budget_s,
                elapsed_prep_and_profile_s=elapsed_prep_and_profile_s,
            )
            _write_derived(analysis_dir / "forecast_receipt.json", canonical(forecast) + b"\n")
            if forecast["projected_wall_seconds"] > remaining_budget_s * 0.90:
                no_go = {
                    "schema": "elendril-hitchhiker-campaign-v1",
                    "run_id": run_id,
                    "status": "NO_GO",
                    "claim_status": "GPU_QUALIFIED" if not cpu_canary else "IMPLEMENTED",
                    "reason": "projected_wall_seconds exceeds 90% of remaining session budget",
                    "source": resolved_source,
                    "protocol_sha256": PROTOCOL_SHA,
                    "phase": phase,
                    "forecast": forecast,
                    "storage_admission": storage,
                    "runtime": runtime_check,
                }
                _write_derived(analysis_dir / "campaign_status.json", canonical(no_go) + b"\n")
                return no_go

        if session_deadline_s is not None:
            worker_deadline_mono = session_t0 + min(minutes * 60.0, session_deadline_s)
        else:
            reserve_s = min(180.0, minutes * 60.0 * 0.05)
            worker_deadline_mono = session_t0 + max(0.0, minutes * 60.0 - reserve_s)

        worker_receipts: list[dict[str, Any]] = []
        errors: list[str] = []
        interrupted = False

        def _execute_on_lane(lane_idx: int, lane_jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
            nonlocal interrupted
            env = _env_for_gpu(gpus[lane_idx] if gpus else None, cpu_canary=cpu_canary)
            results = []
            for job in lane_jobs:
                if interrupted:
                    break
                jid = job["job_id"]
                jdir = analysis_dir / ("stage0" if phase == "stage0" else f"jobs/{jid}")
                jdir.mkdir(parents=True, exist_ok=True)
                receipt_out = jdir / "worker_receipt.json"
                if phase == "stage0":
                    cmd = [
                        sys.executable, "-m", "tools.elendril_hitchhiker_worker",
                        "--mode", "stage0",
                        "--source", resolved_source,
                        "--prepared", str(prepared_dir),
                        "--output", str(output),
                        "--seed", str(job["seed"]),
                        "--receipt", str(receipt_out),
                    ]
                    if stage0_checkpoint_root is not None:
                        cmd.extend(["--checkpoint-root", str(stage0_checkpoint_root)])
                else:
                    rem_worker_s = max(0.0, worker_deadline_mono - time.monotonic())
                    if rem_worker_s <= 0.0:
                        interrupted = True
                        break
                    job_file = jdir / "job_spec.json"
                    _write_derived(job_file, canonical(job) + b"\n")
                    cmd = [
                        sys.executable, "-m", "tools.elendril_hitchhiker_worker",
                        "--mode", "train",
                        "--source", resolved_source,
                        "--prepared", str(prepared_dir),
                        "--output", str(output),
                        "--job-file", str(job_file),
                        "--receipt", str(receipt_out),
                        "--lane-index", str(lane_idx),
                        "--budget-remaining-seconds", f"{rem_worker_s:.3f}",
                    ]
                    if effective_winning_arm:
                        cmd.extend(["--winning-arm", effective_winning_arm])
                    if sel_candidate.is_file():
                        cmd.extend(["--selection-file", str(sel_candidate)])
                if cpu_canary:
                    cmd.append("--cpu-canary")
                    if updates_override is not None and phase != "stage0":
                        cmd.extend(["--updates-override", str(updates_override)])
                    if cpu_interrupt_after is not None and phase != "stage0":
                        cmd.extend(["--cpu-interrupt-after", str(cpu_interrupt_after)])
                res = _run_worker(cmd, env=env, log_path=analysis_dir / "logs" / f"{jid}.txt")
                rec = json.loads(receipt_out.read_text(encoding="utf-8"))
                results.append(rec)
                if res["returncode"] == 2 or rec.get("status") == "INTERRUPTED":
                    interrupted = True
                    break
            return results

        lanes: list[list[dict[str, Any]]] = [[], []]
        for idx, job in enumerate(jobs):
            lanes[idx % 2].append(job)

        with ThreadPoolExecutor(max_workers=2) as pool:
            futs = [pool.submit(_execute_on_lane, i, lanes[i]) for i in range(2) if lanes[i]]
            for fut in futs:
                try:
                    worker_receipts.extend(fut.result())
                except Exception as exc:
                    errors.append(str(exc))

        if errors or interrupted:
            campaign = {
                "schema": "elendril-hitchhiker-campaign-v1",
                "run_id": run_id,
                "status": "INTERRUPTED" if interrupted and not errors else "FAILED",
                "claim_status": "GPU_QUALIFIED" if not cpu_canary else "IMPLEMENTED",
                "source": resolved_source,
                "audited_v717_ancestor": AUDITED_V717_COMMIT,
                "protocol_sha256": PROTOCOL_SHA,
                "phase": phase,
                "errors": errors,
                "worker_receipts": worker_receipts,
                "forecast": forecast,
                "independent_replay": {"status": "NOT_RUN"},
            }
            delivery = finalize_hitchhiker(
                output,
                source=resolved_source,
                protocol_sha=PROTOCOL_SHA,
                campaign=campaign,
                required_jobs=[] if phase == "stage0" else [j["job_id"] for j in jobs],
                required_checkpoints=[
                    f"checkpoints/{r['job_id']}/objects/{r['checkpoint_sha256']}"
                    for r in worker_receipts
                    if r.get("checkpoint_sha256") and r.get("job_id")
                ],
                complete=False,
            )
            campaign["delivery"] = delivery
            return campaign

        replay, gate, locked_selection = _evaluate_stage_outcomes(
            phase,
            jobs,
            output,
            prepared_dir,
            source=resolved_source,
            winning_arm=effective_winning_arm,
        )
        pending_stage0 = phase == "stage0" and replay.get("status") == "PENDING"
        claim_status = (
            "IMPLEMENTED"
            if cpu_canary
            else "PENDING"
            if pending_stage0
            else "CAPABILITY_QUALIFIED"
            if phase == "stage3" and gate.get("status") == "FORMATION_QUALIFIED"
            else "EXECUTED"
        )
        campaign = {
            "schema": "elendril-hitchhiker-campaign-v1",
            "run_id": run_id,
            "status": "EXECUTED",
            "claim_status": claim_status,
            "source": resolved_source,
            "audited_v717_ancestor": AUDITED_V717_COMMIT,
            "protocol_sha256": PROTOCOL_SHA,
            "phase": phase,
            "storage_admission": storage,
            "runtime": runtime_check,
            "prepared_manifest": prepared_manifest,
            "profile": profile_receipt,
            "forecast": forecast,
            "worker_receipts": worker_receipts,
            "independent_replay": replay,
            "gate": gate,
            "selection_receipt": locked_selection,
        }
        req_jobs = [] if phase == "stage0" else [j["job_id"] for j in jobs]
        req_ckpts = [
            f"checkpoints/{r['job_id']}/objects/{r['checkpoint_sha256']}"
            for r in worker_receipts
            if r.get("checkpoint_sha256") and r.get("job_id")
        ]
        delivery = finalize_hitchhiker(
            output,
            source=resolved_source,
            protocol_sha=PROTOCOL_SHA,
            campaign=campaign,
            required_jobs=req_jobs,
            required_checkpoints=req_ckpts,
            complete=(replay.get("status") == "PASS"),
        )
        campaign["delivery"] = delivery
        campaign["status"] = delivery["status"]
        _write_derived(analysis_dir / "campaign_status.json", canonical(campaign) + b"\n")
        return campaign


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("artifacts/elendril/hitchhiker/campaign"))
    p.add_argument("--source", default=None)
    p.add_argument("--phase", choices=tuple(PHASE_TO_STAGE), default="stage1")
    p.add_argument("--minutes", type=float, default=540.0)
    p.add_argument("--owner-enabled", action="store_true")
    p.add_argument("--stage0-checkpoint-root", type=Path, default=None)
    p.add_argument("--winning-arm", default=None)
    p.add_argument("--admitted-followups", nargs="*", default=None)
    p.add_argument("--selection-receipt", type=Path, default=None)
    p.add_argument("--resume-archive", type=Path, default=None)
    p.add_argument("--handoff-archive", type=Path, default=None)
    p.add_argument("--deadline-seconds", type=float, default=None)
    p.add_argument("--worlds-per-hop", type=int, default=None)
    p.add_argument("--train-bytes", type=int, default=None)
    p.add_argument("--updates-override", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--cpu-canary", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if not args.dry_run and not args.cpu_canary and args.source is None:
        raise ValueError("production GPU execution requires explicit --source 40-char git commit SHA")
    result = run(
        output=args.output,
        source=args.source,
        phase=args.phase,
        minutes=args.minutes,
        owner_enabled=args.owner_enabled,
        stage0_checkpoint_root=args.stage0_checkpoint_root,
        winning_arm=args.winning_arm,
        admitted_followups=args.admitted_followups,
        selection_receipt_path=args.selection_receipt,
        resume_archive=args.resume_archive,
        handoff_archive=args.handoff_archive,
        session_deadline_s=args.deadline_seconds,
        worlds_per_hop=args.worlds_per_hop,
        train_bytes=args.train_bytes,
        updates_override=args.updates_override,
        dry_run=args.dry_run,
        cpu_canary=args.cpu_canary,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "claim_status": result.get("claim_status"),
                "phase": result.get("phase"),
                "source": result.get("source"),
                "gate_status": result.get("gate", {}).get("status"),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if result["status"] in ("DRY_RUN_READY", "COMPLETE", "PARTIAL") else 1


if __name__ == "__main__":
    raise SystemExit(main())
