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
from tools.elendril_hitchhiker_export import finalize_hitchhiker, restore_hitchhiker_archive
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
        ):
            # Re-verify serialized suite integrity before reusing across sessions.
            for fname in (
                "validation_suite.json",
                "confirmation_suite.json",
                "structural_suite.json",
                "validation_probes.json",
            ):
                load_suite_bundle(prepared / fname)
            read_dataset(prepared / "train_base.json")
            read_dataset(prepared / "validation_orig.json")
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
            heldout_suites=(val_bundle, conf_bundle, struct_suite),
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
        selected = admitted["admitted_followups"] if admitted_followups is None else list(admitted_followups)
        return [j for j in stage_jobs(stage) if j["arm"] in selected]
    admitted = validate_stage_selection(stage, selection_receipt, winning_arm=winning_arm)
    return stage_jobs(stage, winning_arm=admitted["winning_arm"])


def _forecast_phase_seconds(
    jobs: list[dict[str, Any]],
    profile_receipt: dict[str, Any],
    *,
    train_samples: int,
    gpu_workers: int = 2,
    updates_override: int | None = None,
) -> dict[str, Any]:
    measured = profile_receipt["profiles"]
    per_job = []
    for job in jobs:
        prof = job["profile"]
        m = measured[prof if prof in measured else "S3"]
        updates = updates_override or (
            math.ceil(train_samples / job["batch_size"]) * job["epochs"] if job.get("training_updates") is None else job["training_updates"]
        )
        # Add measured eval overhead (+ 15% safety margin for checkpoint I/O).
        est_sec = ((m["ms_per_update"] / 1000.0) * updates + (m["eval_sample_ms"] / 1000.0) * 6.0) * 1.15
        per_job.append({"job_id": job["job_id"], "profile": prof, "updates": updates, "estimated_seconds": est_sec})
    # Assign LPT across `gpu_workers` parallel workers.
    lanes = [0.0] * max(1, gpu_workers)
    for item in sorted(per_job, key=lambda r: r["estimated_seconds"], reverse=True):
        idx = min(range(len(lanes)), key=lambda i: lanes[i])
        lanes[idx] += item["estimated_seconds"]
    return {
        "gpu_workers": gpu_workers,
        "jobs": len(per_job),
        "projected_wall_seconds": max(lanes) if lanes else 0.0,
        "lane_seconds": lanes,
        "per_job": per_job,
    }


def _replay_job_eval(job_eval_path: Path, prepared: Path, *, expected_source: str) -> dict[str, Any]:
    doc = json.loads(job_eval_path.read_text(encoding="utf-8"))
    if doc.get("protocol_sha256") != PROTOCOL_SHA:
        raise ValueError(f"{job_eval_path} protocol_sha256 mismatch")
    if doc.get("source_commit") != expected_source:
        raise ValueError(f"{job_eval_path} source_commit mismatch")
    split = doc["split"]
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
    return {"status": "PASS", "replayed_cohorts": replayed_cohorts}


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

    evals_by_arm_seed: dict[str, dict[int, dict[str, Any]]] = {}
    replay_jobs = {}
    for job in jobs:
        jdir = analysis / "jobs" / job["job_id"]
        epath = jdir / "final_eval.json"
        replay_jobs[job["job_id"]] = _replay_job_eval(epath, prepared, expected_source=source)
        doc = json.loads(epath.read_text(encoding="utf-8"))
        evals_by_arm_seed.setdefault(job["arm"], {})[job["seed"]] = doc

    replay = {"status": "PASS", "jobs": replay_jobs}

    if phase == "stage1":
        b0_reports = {s: d["suite_eval"] for s, d in evals_by_arm_seed.get("B0", {}).items()}
        arm_gates = {}
        passing = []
        for arm in ("A", "B", "AB", "C"):
            arm_reports = {s: d["suite_eval"] for s, d in evals_by_arm_seed.get(arm, {}).items()}
            g = evaluate_screening_gate(arm_reports, b0_reports)
            arm_gates[arm] = g
            if g["status"] == "PASS_SCREENING":
                passing.append((g["mean_primary_gain"], arm, g))
        passing.sort(reverse=True)
        locked = passing[0][1] if passing else None
        top_status = "PASS_SCREENING" if passing else arm_gates["AB"]["status"]
        selection = {
            "schema": "elendril-hitchhiker-selection-v1",
            "protocol_sha256": PROTOCOL_SHA,
            "source_commit": source,
            "stage1_completed": True,
            "screening_status": top_status,
            "split": "validation",
            "winning_arm": locked,
            "model_profile": "S8" if locked else None,
            "admitted_followups": [a for a in STAGE2_ARMS] if locked is None else ["AB+C", "D-external"],
            "rationale": f"Stage 1 dev screening results across arms {sorted(arm_gates)}: top_status={top_status}, winning_arm={locked}.",
        }
        gate = {"schema": "elendril-hitchhiker-stage1-gate-v1", "arms": arm_gates, "locked_winner": locked, "status": top_status}
        _write_derived(analysis / "gate_receipt.json", canonical(gate) + b"\n")
        _write_derived(analysis / "selection_receipt.json", canonical(selection) + b"\n")
        return replay, gate, selection

    if phase == "stage2":
        gate = {
            "schema": "elendril-hitchhiker-stage2-gate-v1",
            "status": "EXECUTED",
            "arms": {
                arm: {
                    str(s): {"cohort_metrics": d["cohort_metrics"], "original_kind_accuracies": d["original_kind_accuracies"]}
                    for s, d in seeds.items()
                }
                for arm, seeds in evals_by_arm_seed.items()
            },
        }
        _write_derived(analysis / "stage2_gate_receipt.json", canonical(gate) + b"\n")
        return replay, gate, None

    # Stage 3: apply fail-closed formation gate on every confirmation run independently!
    per_job_gates = {}
    for job in jobs:
        d = evals_by_arm_seed[job["arm"]][job["seed"]]
        per_job_gates[job["job_id"]] = evaluate_formation_gate(
            d["suite_eval"],
            equivariance_report=d["equivariance_report"],
        )
    treatment = winning_arm or [j["arm"] for j in jobs if j["arm"] != "B0"][0]
    statuses = [g["status"] for jid, g in per_job_gates.items() if treatment in jid]
    overall = (
        "FORMATION_QUALIFIED"
        if statuses and all(s == "FORMATION_QUALIFIED" for s in statuses)
        else statuses[0]
        if len(set(statuses)) == 1
        else "NOT_QUALIFIED"
    )
    gate = {
        "schema": "elendril-hitchhiker-stage3-gate-v1",
        "treatment_arm": treatment,
        "status": overall,
        "per_job": per_job_gates,
    }
    _write_derived(analysis / "formation_gate_receipt.json", canonical(gate) + b"\n")
    return replay, gate, None


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
    worlds_per_hop: int | None = None,
    train_bytes: int | None = None,
    updates_override: int | None = None,
    cpu_interrupt_after: int | None = None,
    dry_run: bool = False,
    cpu_canary: bool = False,
) -> dict[str, Any]:
    """Run dry-run preflight, GPU-isolated Hitchhiker campaign phase, replay verification, and export."""
    if not dry_run and not owner_enabled and not cpu_canary:
        raise ValueError("campaign execution requires --owner-enabled (or --dry-run / bounded --cpu-canary)")
    if cpu_interrupt_after is not None and not cpu_canary:
        raise ValueError("cpu_interrupt_after is forbidden outside --cpu-canary tests")
    if cpu_canary and updates_override is None:
        updates_override = 2

    resolved_source = resolve_executing_source_commit(source)
    if not cpu_canary:
        check_source(REPO_ROOT, resolved_source)

    if resume_archive is not None:
        restore_hitchhiker_archive(resume_archive, output, source=resolved_source, phase=phase)

    selection = None
    sel_candidate = selection_receipt_path or (output / "analysis" / "selection_receipt.json")
    if sel_candidate.is_file():
        selection = json.loads(sel_candidate.read_text(encoding="utf-8"))
    elif phase == "stage2" and admitted_followups:
        # Only admit stage2 selection override if Stage 1 receipt or explicit selection exists.
        if (output / "analysis" / "gate_receipt.json").is_file():
            selection = {
                "stage1_completed": True,
                "admitted_followups": list(admitted_followups),
                "rationale": "CLI stage2 followup admission after completed Stage 1 screen.",
            }

    jobs = _resolve_stage_jobs(
        phase,
        selection_receipt=selection,
        winning_arm=winning_arm,
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
            forecast = {"gpu_workers": 2, "jobs": len(jobs), "projected_wall_seconds": 0.0, "lane_seconds": [0.0, 0.0], "per_job": []}
        else:
            profile_cmd = [
                sys.executable, "-m", "tools.elendril_hitchhiker_worker",
                "--mode", "profile",
                "--source", resolved_source,
                "--prepared", str(prepared_dir),
                "--receipt", str(prof_receipt_path),
            ]
            if cpu_canary:
                profile_cmd.append("--cpu-canary")
            _run_worker(
                profile_cmd,
                env=_env_for_gpu(gpus[0] if gpus else None, cpu_canary=cpu_canary),
                log_path=analysis_dir / "logs" / "profile.txt",
            )
            profile_receipt = json.loads(prof_receipt_path.read_text(encoding="utf-8"))
            forecast = _forecast_phase_seconds(
                jobs,
                profile_receipt,
                train_samples=prepared_manifest["train_samples"],
                gpu_workers=2,
                updates_override=updates_override if cpu_canary else None,
            )
            _write_derived(analysis_dir / "forecast_receipt.json", canonical(forecast) + b"\n")
            if forecast["projected_wall_seconds"] > (minutes * 60.0) * 0.90:
                no_go = {
                    "schema": "elendril-hitchhiker-campaign-v1",
                    "run_id": run_id,
                    "status": "NO_GO",
                    "claim_status": "GPU_QUALIFIED" if not cpu_canary else "IMPLEMENTED",
                    "reason": "projected_wall_seconds exceeds 90% of session minutes budget",
                    "source": resolved_source,
                    "protocol_sha256": PROTOCOL_SHA,
                    "phase": phase,
                    "forecast": forecast,
                    "storage_admission": storage,
                    "runtime": runtime_check,
                }
                _write_derived(analysis_dir / "campaign_status.json", canonical(no_go) + b"\n")
                return no_go

        worker_receipts: list[dict[str, Any]] = []
        errors: list[str] = []
        interrupted = False

        def _execute_on_lane(lane_idx: int, lane_jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
            nonlocal interrupted
            env = _env_for_gpu(gpus[lane_idx] if gpus else None, cpu_canary=cpu_canary)
            results = []
            for job in lane_jobs:
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
                    ]
                    if winning_arm:
                        cmd.extend(["--winning-arm", winning_arm])
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
            winning_arm=winning_arm,
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
