"""Pure V8 multi-phase Kaggle 2xT4 campaign orchestrator with admission, 2-lane scheduling, and verified export."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Callable

from anra_v5.elendril.v8.analysis import (
    build_v8_report_payload,
    evaluate_advancement_gate,
    load_v717_archive_baseline,
    paired_world_comparison,
)
from anra_v5.elendril.v8.data import build_d1_dev_suite, build_d1_training_dataset, load_d0_dataset
from anra_v5.elendril.v8.evaluation import EpisodeRecord
from anra_v5.elendril.v8.contracts import EvaluationOutcome, PolicyProposal
from anra_v5.elendril.v8.holdout import audit_split_disjointness
from anra_v5.elendril.v8.protocol import PHASES, jobs as protocol_jobs
from anra_v5.elendril.v8.teacher import audit_teacher_samples, compute_shortcut_baselines

from .elendril_v8_export import export_v8_archives, verify_v8_export
from .elendril_v8_worker import run_canary, run_job_worker, run_profile

DEFAULT_SAFETY_RESERVE_SECONDS = 600.0
DEFAULT_EXPORT_RESERVE_SECONDS = 180.0


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
    safety_reserve_seconds: float = DEFAULT_SAFETY_RESERVE_SECONDS,
) -> dict[str, float]:
    """2-lane runtime forecast with startup, canary/profile, eval, and explicit 600s safety reserve."""

    total_train_s = sum(u * seconds_per_update for u in updates_per_job)
    total_eval_s = len(phase_jobs) * eval_seconds_per_job
    parallel_work_s = (total_train_s + total_eval_s) / max(lanes, 1)
    total_forecast_s = parallel_work_s + canary_seconds + profile_seconds + safety_reserve_seconds
    return {
        "lanes": float(lanes),
        "seconds_per_update": float(seconds_per_update),
        "total_train_seconds_serial": float(total_train_s),
        "parallel_work_seconds": float(parallel_work_s),
        "canary_seconds": float(canary_seconds),
        "profile_seconds": float(profile_seconds),
        "safety_reserve_seconds": float(safety_reserve_seconds),
        "total_forecast_seconds": float(total_forecast_s),
    }


def _save_state_machine(work_root: Path, state: dict[str, object]) -> None:
    work_root.mkdir(parents=True, exist_ok=True)
    (work_root / "CAMPAIGN_STATE.json").write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run_worker_subprocess(
    mode: str,
    *,
    job_id: str | None = None,
    arm: str | None = None,
    source_commit: str,
    work_dir: Path,
    gpu_lane: int,
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
        "cuda:0",
    ]
    if job_id:
        cmd.extend(["--job-id", job_id])
    if arm:
        cmd.extend(["--arm", arm])
    if extra_args:
        cmd.extend(extra_args)
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_lane)
    log_dir = work_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_name = f"{mode}_{job_id or arm or 'worker'}.log"
    fh = (log_dir / log_name).open("w", encoding="utf-8")
    proc = subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT)
    proc._v8_log_fh = fh  # type: ignore[attr-defined]
    return proc


def _verify_job_result(job: dict[str, object], job_json_path: Path, *, source_commit: str) -> dict[str, object]:
    if not job_json_path.exists():
        raise ValueError(f"missing worker report for {job['job_id']}: {job_json_path}")
    data = json.loads(job_json_path.read_text(encoding="utf-8"))
    if data.get("job_id") != job["job_id"]:
        raise ValueError(f"worker report job_id mismatch: {data.get('job_id')!r} != {job['job_id']!r}")
    if data.get("source_commit") != source_commit:
        raise ValueError(
            f"worker report wrong source_commit: {data.get('source_commit')!r} != {source_commit!r}"
        )
    if int(data.get("completed_updates", -1)) != int(data.get("target_updates", -2)) or int(data.get("completed_updates", 0)) <= 0:
        raise ValueError(f"incomplete update count for {job['job_id']}")
    return data


def _load_episodes_for_pairwise(ep_file: Path) -> list[EpisodeRecord]:
    out: list[EpisodeRecord] = []
    if not ep_file.exists():
        return out
    for line in ep_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("mode") != "raw":
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
    metrics, md = build_v8_report_payload(
        phase="archive-diagnostic",
        status="COMPLETE",
        source_commit=source_commit,
        run_records=[],
        paired_comparisons=[],
        advancement=None,
        executed_scenarios=set(),
        archive_baseline=archive_info,
        notes=[
            "Phase 0 archive diagnostic verified locally without synthetic weights.",
            f"Parent V7.17 checkpoint status: {archive_info.get('weights_status')}.",
            f"D0 identity verified: {d0.identity} ({len(d0.samples)} samples).",
        ],
    )
    diag_path = work_root / "DIAGNOSTICS.json"
    metrics_path = work_root / "METRICS.json"
    results_path = work_root / "RESULTS.md"
    diag_path.write_text(json.dumps(diag, indent=2) + "\n", encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    results_path.write_text(md, encoding="utf-8")

    export_res = export_v8_archives(
        work_root,
        output_dir,
        source_commit=source_commit,
        phase="archive-diagnostic",
        status_payload={"status": "COMPLETE", "phase": "archive-diagnostic"},
        analysis_files={
            "DIAGNOSTICS.json": diag_path,
            "METRICS.json": metrics_path,
            "RESULTS.md": results_path,
        },
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
    free_bytes_override: int | None = None,
    job_runner: Callable[[dict[str, object], Path], dict[str, object]] | None = None,
    canary_runner: Callable[[], dict[str, object]] | None = None,
    profile_runner: Callable[[dict[str, object]], dict[str, object]] | None = None,
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
        (output_dir / "FINAL_STATUS.json").write_text(json.dumps(status_payload, indent=2) + "\n", encoding="utf-8")
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
    state_log: dict[str, object] = {
        "phase": phase,
        "source_commit": source_commit,
        "state": "PREPARED",
        "jobs": [j["job_id"] for j in phase_jobs],
    }
    _save_state_machine(work_root, state_log)

    # 1. Storage Admission
    storage_est = estimate_campaign_storage_bytes(phase_jobs, gpu_mode=(not cpu_smoke))
    disk_free = int(free_bytes_override) if free_bytes_override is not None else shutil.disk_usage(work_root).free
    if disk_free < storage_est["total_required_bytes"]:
        reason = (
            f"storage admission failed: free={disk_free} < required={storage_est['total_required_bytes']} bytes"
        )
        state_log["state"] = "NO_GO"
        state_log["reason"] = reason
        _save_state_machine(work_root, state_log)
        metrics, md = build_v8_report_payload(
            phase=phase,
            status="NO_GO",
            source_commit=source_commit,
            run_records=[],
            paired_comparisons=[],
            advancement=None,
            executed_scenarios=set(),
            notes=[reason],
        )
        m_path = work_root / "METRICS.json"
        r_path = work_root / "RESULTS.md"
        m_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        r_path.write_text(md, encoding="utf-8")
        export_v8_archives(
            work_root,
            output_dir,
            source_commit=source_commit,
            phase=phase,
            status_payload={"status": "NO_GO", "reason": reason},
            analysis_files={"METRICS.json": m_path, "RESULTS.md": r_path},
        )
        return {"status": "NO_GO", "reason": reason, "storage": storage_est, "disk_free": disk_free}

    # 2. Stage A (canary) + Stage B (profile + time forecast)
    t_campaign_start = time.monotonic()
    deadline = t_campaign_start + minutes * 60.0
    typed_arm = next((str(j["arm"]) for j in phase_jobs if j["arm"] != "B0"), "F")
    if canary_runner is not None:
        canary_res = canary_runner()
    else:
        canary_res = run_canary(
            arm=typed_arm,
            source_commit=source_commit,
            work_dir=work_root / "preflight",
            device="cpu" if cpu_smoke else "cuda:0",
            cpu_canary=cpu_smoke,
        )
    if not canary_res.get("passed"):
        reason = f"Stage A canary check failed: {canary_res}"
        state_log["state"] = "NO_GO"
        state_log["reason"] = reason
        _save_state_machine(work_root, state_log)
        m_path = work_root / "METRICS.json"
        r_path = work_root / "RESULTS.md"
        metrics, md = build_v8_report_payload(
            phase=phase, status="NO_GO", source_commit=source_commit, run_records=[], paired_comparisons=[], advancement=None, executed_scenarios=set(), notes=[reason]
        )
        m_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        r_path.write_text(md, encoding="utf-8")
        export_v8_archives(work_root, output_dir, source_commit=source_commit, phase=phase, status_payload={"status": "NO_GO", "reason": reason}, analysis_files={"METRICS.json": m_path, "RESULTS.md": r_path})
        return {"status": "NO_GO", "reason": reason, "canary": canary_res}

    first_job = phase_jobs[0]
    t_prof0 = time.perf_counter()
    if profile_runner is not None:
        prof_res = profile_runner(first_job)
    else:
        prof_res = run_profile(
            job_id=str(first_job["job_id"]),
            source_commit=source_commit,
            work_dir=work_root / "preflight",
            device="cpu" if cpu_smoke else "cuda:0",
            cpu_canary=cpu_smoke,
        )
    profile_wall = time.perf_counter() - t_prof0
    if not prof_res.get("resume_parity_at_4_verified"):
        raise RuntimeError("Stage B profile failed save@2 -> resume@4 parity check")

    updates_est = [int(prof_res.get("estimated_full_updates", 400)) for _ in phase_jobs]
    forecast = forecast_campaign_runtime(
        phase_jobs=phase_jobs,
        seconds_per_update=float(prof_res["seconds_per_update"]),
        updates_per_job=updates_est,
        canary_seconds=float(canary_res.get("wall_seconds", 0.0)),
        profile_seconds=profile_wall,
    )
    remaining_s = deadline - time.monotonic()
    if forecast["total_forecast_seconds"] > remaining_s:
        reason = (
            f"time admission failed: forecast {forecast['total_forecast_seconds']:.1f}s > remaining {remaining_s:.1f}s"
        )
        state_log["state"] = "NO_GO"
        state_log["reason"] = reason
        _save_state_machine(work_root, state_log)
        m_path = work_root / "METRICS.json"
        r_path = work_root / "RESULTS.md"
        metrics, md = build_v8_report_payload(
            phase=phase, status="NO_GO", source_commit=source_commit, run_records=[], paired_comparisons=[], advancement=None, executed_scenarios=set(), notes=[reason]
        )
        m_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        r_path.write_text(md, encoding="utf-8")
        export_v8_archives(work_root, output_dir, source_commit=source_commit, phase=phase, status_payload={"status": "NO_GO", "reason": reason}, analysis_files={"METRICS.json": m_path, "RESULTS.md": r_path})
        return {"status": "NO_GO", "reason": reason, "forecast": forecast}

    state_log["state"] = "ADMITTED"
    state_log["storage"] = storage_est
    state_log["forecast"] = forecast
    _save_state_machine(work_root, state_log)

    # 3. Execute 2-lane queue
    state_log["state"] = "RUNNING"
    _save_state_machine(work_root, state_log)
    completed_records: list[dict[str, object]] = []
    invalid_errors: list[str] = []
    deadline_hit = False

    if job_runner is not None:
        # Custom / deterministic test runner
        for job in phase_jobs:
            if deadline - time.monotonic() <= min(DEFAULT_EXPORT_RESERVE_SECONDS, minutes * 30.0) and completed_records:
                deadline_hit = True
                break
            try:
                job_runner(job, work_root)
                rec = _verify_job_result(job, work_root / "jobs" / f"{job['job_id']}.json", source_commit=source_commit)
                completed_records.append(rec)
            except Exception as exc:
                invalid_errors.append(f"{job['job_id']}: {exc}")
                break
    else:
        pending = list(phase_jobs)
        active: dict[int, tuple[dict[str, object], subprocess.Popen]] = {}
        lane_count = 1 if cpu_smoke else 2
        extra_worker_args: list[str] = ["--cpu-canary", "--updates-override", "2", "--dev-worlds", "2"] if cpu_smoke else []
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
                    jpath = work_root / "jobs" / f"{job['job_id']}.json"
                    if proc.returncode == 0 and jpath.exists():
                        try:
                            completed_records.append(_verify_job_result(job, jpath, source_commit=source_commit))
                        except Exception:
                            pass
                    del active[lane]
                break

            for lane in range(lane_count):
                if lane not in active and pending and (deadline - time.monotonic() > DEFAULT_EXPORT_RESERVE_SECONDS):
                    job = pending.pop(0)
                    proc = _run_worker_subprocess(
                        "train",
                        job_id=str(job["job_id"]),
                        source_commit=source_commit,
                        work_dir=work_root,
                        gpu_lane=lane,
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
                    rec = _verify_job_result(job, work_root / "jobs" / f"{job['job_id']}.json", source_commit=source_commit)
                    completed_records.append(rec)
                except Exception as exc:
                    invalid_errors.append(f"{job['job_id']}: {exc}")
                    pending.clear()
            if active:
                time.sleep(0.25)

    # 4. Determine strict completion status
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

    # 5. Pairwise comparison + advancement gate
    paired_comparisons: list[dict[str, object]] = []
    seed_pairs: list[dict[str, object]] = []
    arms_present = sorted({str(r["arm"]) for r in completed_records})
    if len(arms_present) == 2 and final_status == "COMPLETE":
        a_name, b_name = arms_present[1], arms_present[0]
        if a_name in ("F", "P") and b_name in ("J", "F-cap"):
            pass
        elif b_name in ("F", "P") and a_name in ("J", "F-cap"):
            a_name, b_name = b_name, a_name
        a_recs = [r for r in completed_records if r["arm"] == a_name]
        b_recs = [r for r in completed_records if r["arm"] == b_name]
        for ra, rb in zip(a_recs, b_recs):
            eps_a = _load_episodes_for_pairwise(Path(str(ra["episodes_file"]))) if ra.get("episodes_file") else []
            eps_b = _load_episodes_for_pairwise(Path(str(rb["episodes_file"]))) if rb.get("episodes_file") else []
            if eps_a and eps_b and {e.world_id for e in eps_a} == {e.world_id for e in eps_b}:
                comp = paired_world_comparison(eps_a, eps_b, arm_a=a_name, arm_b=b_name, bootstrap_samples=400)
                paired_comparisons.append(comp)
            gain = float(ra.get("raw_overall_accuracy") or 0.0) - float(rb.get("raw_overall_accuracy") or 0.0)
            seed_pairs.append(
                {
                    "paired_gain": gain,
                    "candidate_raw_r09": float(ra.get("raw_r09_accuracy") or 0.0),
                    "candidate_relevant_cf_both_correct": float(ra.get("relevant_cf_both_correct") or 0.0),
                    "candidate_irrelevant_cf_stability": float(ra.get("irrelevant_cf_stability") or 0.0),
                }
            )
    advancement = evaluate_advancement_gate(seed_pairs, confirmation=(phase == "replication")) if seed_pairs else None

    executed_scenarios = {"R09", "R10", "E13", "E14", "E15", "A19"} if completed_records else set()
    metrics, md = build_v8_report_payload(
        phase=phase,
        status=final_status,
        source_commit=source_commit,
        run_records=completed_records,
        paired_comparisons=paired_comparisons,
        advancement=advancement,
        executed_scenarios=executed_scenarios,
        archive_baseline=load_v717_archive_baseline(repo),
        notes=invalid_errors + (["Graceful deadline preemption triggered."] if deadline_hit else []),
    )
    m_path = work_root / "METRICS.json"
    r_path = work_root / "RESULTS.md"
    m_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    r_path.write_text(md, encoding="utf-8")

    # 6. Export & Verify
    export_res = export_v8_archives(
        work_root,
        output_dir,
        source_commit=source_commit,
        phase=phase,
        status_payload={
            "status": final_status,
            "phase": phase,
            "completed_jobs": [r["job_id"] for r in completed_records],
            "expected_jobs": [j["job_id"] for j in phase_jobs],
            "errors": invalid_errors,
        },
        analysis_files={"METRICS.json": m_path, "RESULTS.md": r_path},
    )
    verify_v8_export(output_dir, expected_source_commit=source_commit)
    if final_status == "COMPLETE":
        state_log["state"] = "COMPLETE"
    _save_state_machine(work_root, state_log)
    return {
        "status": final_status,
        "state": state_log["state"],
        "completed_jobs": len(completed_records),
        "expected_jobs": len(phase_jobs),
        "errors": invalid_errors,
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
    "estimate_campaign_storage_bytes",
    "forecast_campaign_runtime",
    "run_campaign",
]

if __name__ == "__main__":
    raise SystemExit(main())
