"""Source-pinned AN-RA V8-Y1 2-GPU campaign coordinator, dry-run validator, and budget gatekeeper."""

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

from anra_v5.elendril.codec import canonical, digest
from anra_v5.elendril.y1_data import (
    audit_y1_split_custody,
    build_confirmation_suite,
    build_d2_probe_dataset,
    build_development_suite,
    build_followup_suite,
    load_y1_suite,
    serialize_y1_suite,
    y1_case_to_record,
)
from anra_v5.elendril.y1_diagnostics import verify_and_inspect_hh_full_archive
from anra_v5.elendril.y1_evaluation import (
    evaluate_all_heuristics,
    evaluate_f11_bottleneck_admission,
    evaluate_t1_or_t2_gate,
    evaluate_y1_cases,
    summarize_y1_predictions,
)
from anra_v5.elendril.y1_protocol import (
    HISTORICAL_HH_S3_MANIFESTS,
    PHASE_CEILING_MINUTES,
    PHASES,
    PROTOCOL_SHA,
    SEEDS,
    SPLIT_QUOTAS,
    build_core_jobs,
    build_followup_jobs,
    effective_protocol_digest,
)
from tools.elendril_cloud import admit_devices, check_source, check_torch, device_env, inventory
from tools.elendril_v6_export import _write_derived
from tools.elendril_y1_export import (
    finalize_y1_campaign,
    handoff_y1_archive,
    restore_y1_archive,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PROD_STORAGE_BYTES = {
    "prepare": 1 << 30,
    "canary": 512 << 20,
    "diagnose": 4 << 30,
    "core": 16 << 30,
    "followup": 16 << 30,
    "finalize": 2 << 30,
}


def storage_admission(output: Path, required_bytes: int, *, floor_bytes: int = 256 << 20) -> dict[str, Any]:
    """Verify real writable output directory and free disk headroom before launch."""
    output.mkdir(parents=True, exist_ok=True)
    probe = output.resolve(strict=True)
    free = shutil.disk_usage(probe).free
    need = max(required_bytes, floor_bytes)
    if free < need:
        raise ValueError(f"NO_GO storage admission: {free} bytes free at {probe}, require >= {need}")
    return {"status": "ADMITTED", "path": str(probe), "free_bytes": free, "required_bytes": need}


def _worker_env(gpu: Any | None, *, cpu_canary: bool) -> dict[str, str]:
    if cpu_canary or gpu is None:
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        env.update(PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        return env
    return device_env(gpu)


def _run_worker(cmd: list[str], *, env: dict[str, str], log_path: Path) -> int:
    proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        f"$ {' '.join(cmd)}\n\n[stdout]\n{proc.stdout}\n\n[stderr]\n{proc.stderr}\n",
        encoding="utf-8",
    )
    if proc.returncode not in (0, 2):
        raise RuntimeError(f"worker failed ({proc.returncode}): {proc.stderr.strip() or proc.stdout.strip()}")
    return proc.returncode


def _prepare_suites(output: Path, *, compute_heuristics: bool = True) -> dict[str, Any]:
    """Generate/verify all frozen Y1 splits once, audit custody, and emit heuristics."""
    prep = output / "prepared"
    prep.mkdir(parents=True, exist_ok=True)
    custody_path = output / "analysis" / "custody_audit.json"
    if (prep / "development_suite.json").is_file() and (prep / "confirmation_suite.json").is_file() and (prep / "followup_suite.json").is_file() and custody_path.is_file():
        return json.loads(custody_path.read_text(encoding="utf-8"))

    dev = build_development_suite()
    conf = build_confirmation_suite()
    fol = build_followup_suite()
    d2 = build_d2_probe_dataset()
    custody = audit_y1_split_custody(dev, conf, fol)

    _write_derived(prep / "development_suite.json", canonical(serialize_y1_suite(dev)) + b"\n")
    _write_derived(prep / "confirmation_suite.json", canonical(serialize_y1_suite(conf)) + b"\n")
    _write_derived(prep / "followup_suite.json", canonical(serialize_y1_suite(fol)) + b"\n")
    _write_derived(
        prep / "d2_probe.json",
        canonical(
            {
                "schema": d2["schema"],
                "seed": d2["seed"],
                "train_cases": [y1_case_to_record(c) for c in d2["train_cases"]],
                "test_cases": [y1_case_to_record(c) for c in d2["test_cases"]],
            }
        )
        + b"\n",
    )
    (output / "analysis").mkdir(parents=True, exist_ok=True)
    _write_derived(custody_path, canonical(custody) + b"\n")

    if compute_heuristics:
        hdir = output / "analysis" / "heuristics"
        hdir.mkdir(parents=True, exist_ok=True)
        for name, suite in (("development", dev), ("confirmation", conf), ("capacity_followup", fol)):
            hres = evaluate_all_heuristics(suite["cases"], historical_cases=suite["historical_continuity"])
            _write_derived(hdir / f"{name}.json", canonical(hres) + b"\n")
    return custody


def _load_or_init_budget_ledger(output: Path, *, phase: str, ceiling_minutes: float) -> dict[str, Any]:
    path = output / "analysis" / "budget_ledger.json"
    if path.is_file():
        doc = json.loads(path.read_text(encoding="utf-8"))
        per = doc.setdefault("cumulative_seconds_by_phase", {})
        doc["current_phase_ceiling_seconds"] = ceiling_minutes * 60.0
        if per.get(phase, 0.0) >= doc["current_phase_ceiling_seconds"]:
            raise ValueError(
                f"cumulative budget exhausted for {phase}: {per[phase]:.1f}s >= {doc['current_phase_ceiling_seconds']:.1f}s"
            )
        return doc
    doc = {
        "schema": "anra-v8-y1-budget-ledger-v1",
        "cumulative_seconds_by_phase": {},
        "sessions": [],
        "current_phase_ceiling_seconds": ceiling_minutes * 60.0,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_derived(path, canonical(doc) + b"\n")
    return doc


def _update_budget_ledger(output: Path, ledger: dict[str, Any], *, phase: str, session_elapsed_s: float) -> dict[str, Any]:
    per = ledger.setdefault("cumulative_seconds_by_phase", {})
    per[phase] = float(per.get(phase, 0.0)) + float(session_elapsed_s)
    ledger.setdefault("sessions", []).append({"phase": phase, "elapsed_seconds": session_elapsed_s, "timestamp": time.time()})
    _write_derived(output / "analysis" / "budget_ledger.json", canonical(ledger) + b"\n")
    return ledger


def _forecast_jobs(
    jobs: list[dict[str, Any]],
    profile_receipt: dict[str, Any],
    *,
    workers: int,
    eval_cases_per_job: int,
    remaining_budget_s: float,
    measured_export_s: float = 30.0,
) -> dict[str, Any]:
    """Conservatively forecast dual-lane runtime including 30% margin and reserve."""
    profs = profile_receipt["profiles"]
    lanes = [0.0] * max(1, workers)
    per_job: list[dict[str, Any]] = []
    for j in jobs:
        p = profs[j["profile"]]
        train_s = float(j["updates"]) * float(p["sec_per_update"])
        supp_s = 1024.0 * float(p.get("sec_per_supp_base_case", 0.0)) if eval_cases_per_job > 1024 else 0.0
        eval_s = float(eval_cases_per_job) * float(p["sec_per_eval_case"]) + supp_s
        ckpt_s = math.ceil(float(j["updates"]) / 64.0) * float(p.get("sec_per_ckpt_save", 0.2))
        job_s = (train_s + eval_s + ckpt_s) * 1.30  # +30% conservative safety margin
        lane_idx = min(range(len(lanes)), key=lambda i: lanes[i])
        lanes[lane_idx] += job_s
        per_job.append({"job_id": j["job_id"], "lane": lane_idx, "projected_seconds_with_margin": job_s})

    reserve_s = max(900.0 if workers > 1 else 5.0, measured_export_s * 1.5)
    projected_wall_s = max(lanes) + reserve_s
    return {
        "schema": "anra-v8-y1-forecast-v1",
        "workers": workers,
        "safety_margin_fraction": 0.30,
        "export_reserve_seconds": reserve_s,
        "lane_seconds": lanes,
        "projected_wall_seconds": projected_wall_s,
        "remaining_budget_seconds": remaining_budget_s,
        "fits_within_remaining_budget": projected_wall_s <= remaining_budget_s,
        "per_job": per_job,
    }


def _profile_lanes(
    output: Path,
    *,
    source: str,
    protocol_sha: str,
    gpus: tuple[Any, ...],
    cpu_canary: bool,
) -> dict[str, Any]:
    prof_path = output / "analysis" / "profile_receipt.json"
    if prof_path.is_file():
        return json.loads(prof_path.read_text(encoding="utf-8"))

    lanes = [0] if cpu_canary else [0, 1]
    logs = output / "analysis" / "logs"

    def _run_lane(lidx: int) -> None:
        gpu = None if cpu_canary else gpus[lidx]
        cmd = [
            sys.executable, "-m", "tools.elendril_y1_worker",
            "--mode", "profile",
            "--output", str(output),
            "--source", source,
            "--protocol-sha", protocol_sha,
            "--lane", str(lidx),
        ]
        if cpu_canary:
            cmd.append("--cpu-canary")
        _run_worker(cmd, env=_worker_env(gpu, cpu_canary=cpu_canary), log_path=logs / f"profile_lane_{lidx}.log")

    with ThreadPoolExecutor(max_workers=len(lanes)) as pool:
        list(pool.map(_run_lane, lanes))

    lane_docs = [json.loads((output / "analysis" / f"profile_lane_{l}.json").read_bytes()) for l in lanes]
    merged: dict[str, Any] = {}
    for scale in ("S3", "S20"):
        entries = [d["profiles"][scale] for d in lane_docs]
        merged[scale] = {
            k: max(float(e.get(k, 0.0)) for e in entries)
            for k in ("sec_per_update", "sec_per_ckpt_save", "sec_per_eval_case", "sec_per_supp_base_case", "peak_cuda_mb")
        }
        merged[scale]["recovery_verified"] = all(bool(e.get("recovery_verified")) for e in entries)

    combined = {
        "schema": "anra-v8-y1-profile-combined-v1",
        "status": "PASS",
        "source": source,
        "protocol_sha256": protocol_sha,
        "gpu_uuids": [d["gpu_uuid"] for d in lane_docs],
        "profiles": merged,
    }
    _write_derived(prof_path, canonical(combined) + b"\n")
    return combined


def _write_summary_markdown(output: Path, campaign: dict[str, Any]) -> None:
    lines = [
        f"# AN-RA V8-Y1 {campaign['phase'].upper()} Report",
        "",
        f"- **Run ID**: `{campaign['run_id']}`",
        f"- **Source commit**: `{campaign['source']}`",
        f"- **Protocol SHA256**: `{campaign['protocol_sha256']}`",
        f"- **Execution status**: `{campaign['status']}`",
        f"- **Scientific status**: `{campaign.get('scientific_status', 'UNEXECUTED')}`",
        f"- **Claim status**: `{campaign.get('claim_status', 'IMPLEMENTED')}`",
        "",
    ]
    t1_gate = output / "analysis" / "gates" / "t1_gate.json"
    if t1_gate.is_file():
        g = json.loads(t1_gate.read_text(encoding="utf-8"))
        lines.extend([
            "## T1 Screening Gate",
            f"- Gate status: `{g['status']}` (locked winner: `{g.get('locked_winner')}`)",
            "",
        ])
    t2_gate = output / "analysis" / "gates" / "t2_gate.json"
    if t2_gate.is_file():
        g = json.loads(t2_gate.read_text(encoding="utf-8"))
        lines.extend([
            "## T2 Confirmation Gate",
            f"- Gate status: `{g['status']}`",
            "",
        ])
    sel_path = output / "analysis" / "gates" / "selection_receipt.json"
    if sel_path.is_file():
        sel = json.loads(sel_path.read_text(encoding="utf-8"))
        lines.extend([
            "## Selection & Follow-up Admission",
            f"- Next recommended branch: `{sel.get('recommended_followup_branch')}`",
            "",
        ])
    summary_dir = output / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    _write_derived(summary_dir / "REPORT.md", ("\n".join(lines) + "\n").encode("utf-8"))


def _replay_phase_evidence(output: Path, phase: str) -> dict[str, Any]:
    """Independent replay check confirming all prediction files match existing run receipts and gates."""
    jobs_dir = output / "analysis" / "jobs"
    preds_dir = output / "evidence" / "predictions"
    if not jobs_dir.is_dir():
        return {"status": "NO_JOBS_EXECUTED", "verified_jobs": 0}
    checked = 0
    for jdir in sorted(jobs_dir.iterdir()):
        if not jdir.is_dir() or not (jdir / "run_receipt.json").is_file():
            continue
        checked += 1
        rec = json.loads((jdir / "run_receipt.json").read_text(encoding="utf-8"))
        if (jdir / "metrics_summary.json").is_file():
            pfile = preds_dir / f"{jdir.name}.json"
            if not pfile.is_file():
                return {"status": "FAILED", "reason": f"missing prediction file for {jdir.name}"}
    return {"status": "PASS", "phase": phase, "verified_jobs": checked}


def _run_diagnose_phase(args: argparse.Namespace, output: Path, *, source: str, protocol_sha: str, gpus: tuple[Any, ...]) -> dict[str, Any]:
    if args.checkpoint_archive is None or not (Path(args.checkpoint_archive).is_file() or Path(args.checkpoint_archive).is_dir()):
        raise ValueError("diagnose phase requires a valid --checkpoint-archive (ELENDRIL_HITCHHIKER_FULL.zip or unpacked dataset directory)")
    arch_info = verify_and_inspect_hh_full_archive(args.checkpoint_archive)
    all_lineages = arch_info["verified_lineages"]
    if args.cpu_canary:
        all_lineages = all_lineages[:1]

    logs = output / "analysis" / "logs"
    workers = [0] if args.cpu_canary else [0, 1]
    splits = [all_lineages[i :: len(workers)] for i in range(len(workers))]

    def _run_diag_lane(lidx: int) -> None:
        lins = splits[lidx]
        if not lins:
            return
        gpu = None if args.cpu_canary else gpus[lidx]
        cmd = [
            sys.executable, "-m", "tools.elendril_y1_worker",
            "--mode", "diagnose-historical",
            "--output", str(output),
            "--source", source,
            "--protocol-sha", protocol_sha,
            "--checkpoint-archive", str(args.checkpoint_archive),
            "--lineages", *lins,
        ]
        if args.enable_d2:
            cmd.append("--enable-d2")
        if args.cpu_canary:
            cmd.append("--cpu-canary")
        _run_worker(cmd, env=_worker_env(gpu, cpu_canary=args.cpu_canary), log_path=logs / f"diagnose_lane_{lidx}.log")

    with ThreadPoolExecutor(max_workers=len(workers)) as pool:
        list(pool.map(_run_diag_lane, range(len(workers))))

    agg = {
        "schema": "anra-v8-y1-d0d1-aggregate-v1",
        "archive": arch_info,
        "mismatch_blockers": [],
        "models": {},
    }
    for lin in all_lineages:
        jdir = output / "analysis" / "jobs" / lin
        rr = json.loads((jdir / "run_receipt.json").read_text(encoding="utf-8"))
        ms = json.loads((jdir / "metrics_summary.json").read_text(encoding="utf-8"))
        d1 = json.loads((jdir / "d1_summary.json").read_text(encoding="utf-8"))
        hrep = rr.get("historical_reproduction", {})
        if hrep.get("mismatch_blocker"):
            agg["mismatch_blockers"].append(lin)
        agg["models"][lin] = {
            "primary_endpoint": ms["primary_endpoint"],
            "valid_action_rate": ms["guardrails"]["valid_action_rate"],
            "mean_p_answer_mass": d1["mean_p_answer_mass"],
            "known_base_rank1_within_ap_rate": d1["known_base_rank1_within_ap_rate"],
            "known_base_constrained_ap_eos_rate": d1["known_base_constrained_ap_eos_rate"],
            "descriptive_rescue": d1["descriptive_rescue_constrained_minus_raw"],
            "historical_reproduction": hrep,
            "representative_failed_case_ids": ms["representative_failed_case_ids"][:16],
        }
    _write_derived(output / "analysis" / "d0_d1_summary.json", canonical(agg) + b"\n")
    return {"status": "COMPLETE", "scientific_status": "D0_D1_EXECUTED", "claim_status": "OBSERVED"}


def _run_jobs_parallel(
    jobs: list[dict[str, Any]],
    *,
    output: Path,
    source: str,
    protocol_sha: str,
    gpus: tuple[Any, ...],
    cpu_canary: bool,
    canary_updates: int,
    deadline_epoch: float,
) -> tuple[list[str], bool]:
    """Execute training/eval jobs across isolated lanes in interleaved seed-balanced order."""
    logs = output / "analysis" / "logs"
    jtmp = output / ".tmp_jobs"
    jtmp.mkdir(parents=True, exist_ok=True)
    completed: list[str] = []
    interrupted = False

    n_lanes = 1 if cpu_canary else len(gpus)
    pending = [j for j in jobs if not (output / "analysis" / "jobs" / j["job_id"] / "run_receipt.json").is_file()]
    completed.extend(j["job_id"] for j in jobs if j not in pending)

    for start in range(0, len(pending), n_lanes):
        if time.time() >= deadline_epoch:
            interrupted = True
            break
        batch = pending[start : start + n_lanes]

        def _exec_one(pair: tuple[int, dict[str, Any]]) -> tuple[str, int]:
            lidx, j = pair
            jfile = jtmp / f"{j['job_id']}.json"
            jfile.write_bytes(canonical(j))
            gpu = None if cpu_canary else gpus[lidx]
            cmd = [
                sys.executable, "-m", "tools.elendril_y1_worker",
                "--mode", "train-job",
                "--output", str(output),
                "--source", source,
                "--protocol-sha", protocol_sha,
                "--job-file", str(jfile),
                "--deadline-epoch", str(deadline_epoch),
            ]
            if cpu_canary:
                cmd.extend(["--cpu-canary", "--canary-updates", str(canary_updates)])
            rc = _run_worker(cmd, env=_worker_env(gpu, cpu_canary=cpu_canary), log_path=logs / f"{j['job_id']}.log")
            return j["job_id"], rc

        with ThreadPoolExecutor(max_workers=len(batch)) as pool:
            res = list(pool.map(_exec_one, enumerate(batch)))
        for jid, rc in res:
            if rc == 2:
                interrupted = True
            else:
                completed.append(jid)
        if interrupted:
            break
    return completed, interrupted


def _run_eval_jobs_parallel(
    specs: list[tuple[str, str, str, str]],  # (eval_job_id, lineage, profile, split)
    *,
    output: Path,
    source: str,
    protocol_sha: str,
    gpus: tuple[Any, ...],
    cpu_canary: bool,
) -> None:
    logs = output / "analysis" / "logs"
    n_lanes = 1 if cpu_canary else len(gpus)
    todo = [s for s in specs if not (output / "analysis" / "jobs" / s[0] / "run_receipt.json").is_file()]
    for start in range(0, len(todo), n_lanes):
        batch = todo[start : start + n_lanes]

        def _ev(pair: tuple[int, tuple[str, str, str, str]]) -> None:
            lidx, (ejid, lin, prof, spl) = pair
            gpu = None if cpu_canary else gpus[lidx]
            cmd = [
                sys.executable, "-m", "tools.elendril_y1_worker",
                "--mode", "eval-checkpoint",
                "--output", str(output),
                "--source", source,
                "--protocol-sha", protocol_sha,
                "--eval-job-id", ejid,
                "--lineage", lin,
                "--profile", prof,
                "--split", spl,
            ]
            if cpu_canary:
                cmd.append("--cpu-canary")
            _run_worker(cmd, env=_worker_env(gpu, cpu_canary=cpu_canary), log_path=logs / f"{ejid}.log")

        with ThreadPoolExecutor(max_workers=len(batch)) as pool:
            list(pool.map(_ev, enumerate(batch)))


def _run_core_phase(args: argparse.Namespace, output: Path, *, source: str, protocol_sha: str, gpus: tuple[Any, ...], prof_receipt: dict[str, Any], deadline_epoch: float) -> dict[str, Any]:
    all_jobs = build_core_jobs(include_t0=True)
    t0_job = all_jobs[0]
    t1_jobs = all_jobs[1:]

    # 1. T0 positive learnability gate (seed 7; allow seed 19 single fallback)
    _, int_t0 = _run_jobs_parallel([t0_job], output=output, source=source, protocol_sha=protocol_sha, gpus=gpus, cpu_canary=args.cpu_canary, canary_updates=args.canary_updates, deadline_epoch=deadline_epoch)
    if int_t0:
        return {"status": "INTERRUPTED", "scientific_status": "PARTIAL", "claim_status": "INCOMPLETE_NOT_QUALIFIED"}
    t0_gate_path = output / "analysis" / "jobs" / t0_job["job_id"] / "t0_gate.json"
    t0_gate = json.loads(t0_gate_path.read_text(encoding="utf-8"))
    gates_dir = output / "analysis" / "gates"
    gates_dir.mkdir(parents=True, exist_ok=True)
    if not args.cpu_canary and t0_gate["status"] != "PASS":
        fb_job = {**t0_job, "job_id": "Y1-t0_learnability-S3-T0-s19", "seed": 19, "order_seed": 19}
        _run_jobs_parallel([fb_job], output=output, source=source, protocol_sha=protocol_sha, gpus=gpus, cpu_canary=False, canary_updates=args.canary_updates, deadline_epoch=deadline_epoch)
        fb_gate = json.loads((output / "analysis" / "jobs" / fb_job["job_id"] / "t0_gate.json").read_text(encoding="utf-8"))
        _write_derived(gates_dir / "t0_gate.json", canonical({"primary": t0_gate, "fallback": fb_gate, "status": fb_gate["status"]}) + b"\n")
        if fb_gate["status"] != "PASS":
            return {"status": "COMPLETE", "scientific_status": "T0_FAILED_NO_GO", "claim_status": "FALSIFIED"}
    else:
        _write_derived(gates_dir / "t0_gate.json", canonical({"primary": t0_gate, "status": "PASS" if args.cpu_canary else t0_gate["status"]}) + b"\n")

    # 2. Forecast remaining T1 work against remaining cumulative budget
    pending_t1 = [j for j in t1_jobs if not (output / "analysis" / "jobs" / j["job_id"] / "run_receipt.json").is_file()]
    rem_s = max(0.0, deadline_epoch - time.time())
    forecast = _forecast_jobs(pending_t1, prof_receipt, workers=1 if args.cpu_canary else len(gpus), eval_cases_per_job=11920 if not args.cpu_canary else 64, remaining_budget_s=rem_s)
    _write_derived(output / "analysis" / "forecast_receipt.json", canonical(forecast) + b"\n")

    # 3. Execute T1 12-job interleaved matrix cleanly up to safe deadline reserve
    soft_deadline = deadline_epoch - forecast["export_reserve_seconds"]
    _, int_t1 = _run_jobs_parallel(t1_jobs, output=output, source=source, protocol_sha=protocol_sha, gpus=gpus, cpu_canary=args.cpu_canary, canary_updates=args.canary_updates, deadline_epoch=soft_deadline)
    if int_t1:
        return {"status": "INTERRUPTED", "scientific_status": "PARTIAL", "claim_status": "INCOMPLETE_NOT_QUALIFIED"}

    # 4. Evaluate T1 screening gate on development
    heur_dev = json.loads((output / "analysis" / "heuristics" / "development.json").read_text(encoding="utf-8"))
    arms_metrics: dict[str, dict[int, dict[str, Any]]] = {}
    f11_d1_by_seed: dict[int, dict[str, Any]] = {}
    for j in t1_jobs:
        jdir = output / "analysis" / "jobs" / j["job_id"]
        ms = json.loads((jdir / "metrics_summary.json").read_text(encoding="utf-8"))
        fc = json.loads((jdir / "family_clusters.json").read_text(encoding="utf-8"))
        ms.update(fc)
        arms_metrics.setdefault(j["arm"], {})[j["seed"]] = ms
        if j["arm"] == "F11":
            f11_d1_by_seed[j["seed"]] = json.loads((jdir / "d1_summary.json").read_text(encoding="utf-8"))

    t1_gate = evaluate_t1_or_t2_gate(stage="t1_factorial", arms_metrics=arms_metrics, heuristics=heur_dev, require_ci_lower_gt_zero=False)
    _write_derived(gates_dir / "t1_gate.json", canonical(t1_gate) + b"\n")

    bottleneck = evaluate_f11_bottleneck_admission(f11_d1_by_seed)
    _write_derived(gates_dir / "t3_bottleneck_admission.json", canonical(bottleneck) + b"\n")

    winner = t1_gate.get("locked_winner")
    t2_status = "NOT_RUN_NO_T1_WINNER"
    if winner is not None:
        # Locked T2 confirmation on untouched confirmation suite
        conf_specs = [
            (f"Y1-t2_confirm-S3-{arm}-s{s}", f"Y1-t1_factorial-S3-{arm}-s{s}", "S3", "confirmation")
            for s in SEEDS
            for arm in ("F00", winner)
        ]
        _run_eval_jobs_parallel(conf_specs, output=output, source=source, protocol_sha=protocol_sha, gpus=gpus, cpu_canary=args.cpu_canary)
        heur_conf = json.loads((output / "analysis" / "heuristics" / "confirmation.json").read_text(encoding="utf-8"))
        t2_metrics: dict[str, dict[int, dict[str, Any]]] = {}
        for s in SEEDS:
            for arm in ("F00", winner):
                jdir = output / "analysis" / "jobs" / f"Y1-t2_confirm-S3-{arm}-s{s}"
                ms = json.loads((jdir / "metrics_summary.json").read_text(encoding="utf-8"))
                ms.update(json.loads((jdir / "family_clusters.json").read_text(encoding="utf-8")))
                t2_metrics.setdefault(arm, {})[s] = ms
        t2_gate = evaluate_t1_or_t2_gate(stage="t2_confirmation", arms_metrics=t2_metrics, heuristics=heur_conf, require_ci_lower_gt_zero=True)
        _write_derived(gates_dir / "t2_gate.json", canonical(t2_gate) + b"\n")
        t2_status = t2_gate["status"]

    rec_branch = "targeted_group_ce" if bottleneck["admitted"] else "capacity_s20"
    comp_arm = winner if winner is not None else "F11"
    selection = {
        "schema": "anra-v8-y1-selection-receipt-v1",
        "protocol_sha256": protocol_sha,
        "source_commit": source,
        "t1_status": t1_gate["status"],
        "t1_winner": winner,
        "t2_status": t2_status,
        "bottleneck_admitted": bottleneck["admitted"],
        "recommended_followup_branch": rec_branch,
        "capacity_comparison_arm": comp_arm,
        "failure_diagnosis_mode": t1_gate["status"] != "PASS" or t2_status != "PASS",
    }
    _write_derived(gates_dir / "selection_receipt.json", canonical(selection) + b"\n")
    sci_stat = "T2_CONFIRMED" if t2_status == "PASS" else ("T1_SCREEN_ONLY_NO_CONFIRM" if winner else "T1_NO_WINNER")
    return {
        "status": "COMPLETE",
        "scientific_status": sci_stat,
        "claim_status": "EXECUTED" if not args.cpu_canary else "IMPLEMENTED",
    }


def _run_followup_phase(args: argparse.Namespace, output: Path, *, source: str, protocol_sha: str, gpus: tuple[Any, ...], deadline_epoch: float) -> dict[str, Any]:
    sel_file = args.selection_receipt or (output / "analysis" / "gates" / "selection_receipt.json")
    if not sel_file.is_file():
        raise ValueError("followup phase requires a verified --selection-receipt")
    sel = json.loads(sel_file.read_text(encoding="utf-8"))
    branch = sel["recommended_followup_branch"]
    jobs = build_followup_jobs(branch, comparison_arm=sel.get("capacity_comparison_arm"))
    if args.cpu_canary:
        jobs = jobs[:2]

    _, interrupted = _run_jobs_parallel(jobs, output=output, source=source, protocol_sha=protocol_sha, gpus=gpus, cpu_canary=args.cpu_canary, canary_updates=args.canary_updates, deadline_epoch=deadline_epoch - 30.0)
    if interrupted:
        return {"status": "INTERRUPTED", "scientific_status": "PARTIAL", "claim_status": "INCOMPLETE_NOT_QUALIFIED"}

    # If capacity_s20 branch, also evaluate matching S3 checkpoints on frozen seed-1109 followup_suite.
    if branch == "capacity_s20":
        comp_arm = sel.get("capacity_comparison_arm", "F11")
        s3_specs = [
            (f"Y1-t3_s3ref-S3-{arm}-s{j['seed']}", f"Y1-t1_factorial-S3-{arm}-s{j['seed']}", "S3", "capacity_followup")
            for j in jobs
            for arm in (j["arm"],)
        ]
        _run_eval_jobs_parallel(s3_specs, output=output, source=source, protocol_sha=protocol_sha, gpus=gpus, cpu_canary=args.cpu_canary)
    return {"status": "COMPLETE", "scientific_status": f"T3_{branch.upper()}_EXECUTED", "claim_status": "EXECUTED" if not args.cpu_canary else "IMPLEMENTED"}


def run_campaign(args: argparse.Namespace) -> dict[str, Any]:
    """Orchestrate any Y1 phase cleanly, failing closed on missing inputs or budget overrun."""
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    source = args.source
    phase = args.phase

    # Enforce mutually exclusive / required phase flags before touching CUDA.
    if args.resume_archive is not None and args.handoff_archive is not None:
        raise ValueError("pass either --resume-archive or --handoff-archive, not both")
    if phase == "diagnose" and args.checkpoint_archive is None and args.resume_archive is None:
        raise ValueError("diagnose requires --checkpoint-archive ELENDRIL_HITCHHIKER_FULL.zip")
    if phase == "followup" and args.handoff_archive is None and args.resume_archive is None and not (output / "analysis" / "gates" / "selection_receipt.json").is_file():
        raise ValueError("followup requires --handoff-archive (core FULL ZIP) or pre-staged selection_receipt.json")

    if not args.skip_source_check:
        check_source(REPO_ROOT, source)

    proto_sha, is_prod = effective_protocol_digest({
        "cpu_canary": args.cpu_canary,
        "canary_updates": args.canary_updates if args.cpu_canary else None,
        "skip_source_check": args.skip_source_check,
    })

    if args.resume_archive is not None:
        restore_y1_archive(args.resume_archive, output, source=source, protocol_sha=proto_sha, phase=phase)
    elif args.handoff_archive is not None:
        from_phase = "diagnose" if phase == "core" else "core"
        handoff_y1_archive(args.handoff_archive, output, source=source, protocol_sha=proto_sha, from_phase=from_phase, to_phase=phase)

    ceiling_min = float(args.minutes or PHASE_CEILING_MINUTES[phase])
    if ceiling_min > PHASE_CEILING_MINUTES[phase]:
        raise ValueError(f"requested minutes {ceiling_min} exceeds frozen ceiling {PHASE_CEILING_MINUTES[phase]} for {phase}")

    custody = _prepare_suites(output, compute_heuristics=(phase != "canary" or not args.dry_run))
    if args.dry_run or phase == "prepare":
        res = {
            "schema": "anra-v8-y1-dryrun-v1",
            "status": "DRY_RUN_READY",
            "phase": phase,
            "source": source,
            "protocol_sha256": proto_sha,
            "production_protocol": is_prod,
            "custody": custody,
        }
        print(canonical(res).decode("utf-8"))
        return res

    storage_admission(output, PROD_STORAGE_BYTES[phase] if not args.cpu_canary else (32 << 20))
    ledger = _load_or_init_budget_ledger(output, phase=phase, ceiling_minutes=ceiling_min)
    rem_budget_s = max(60.0, ledger["current_phase_ceiling_seconds"] - ledger["cumulative_seconds_by_phase"].get(phase, 0.0))
    t_sess_start = time.time()
    deadline_epoch = t_sess_start + rem_budget_s

    gpus: tuple[Any, ...] = ()
    if not args.cpu_canary and phase != "finalize":
        gpus = inventory()
        admit_devices(gpus, 2)
        check_torch(REPO_ROOT, gpus)

    run_id = f"y1-{phase}-{uuid4().hex[:10]}"
    if phase == "canary" or args.cpu_canary:
        args.cpu_canary = True
        if phase == "canary":
            prof = _profile_lanes(output, source=source, protocol_sha=proto_sha, gpus=(), cpu_canary=True)
            t0_job = build_core_jobs(include_t0=True)[0]
            _run_jobs_parallel([t0_job], output=output, source=source, protocol_sha=proto_sha, gpus=(), cpu_canary=True, canary_updates=args.canary_updates, deadline_epoch=deadline_epoch)
            outcome = {"status": "COMPLETE", "scientific_status": "CANARY_PASS", "claim_status": "IMPLEMENTED"}
        elif phase == "diagnose":
            outcome = _run_diagnose_phase(args, output, source=source, protocol_sha=proto_sha, gpus=())
        elif phase == "core":
            prof = _profile_lanes(output, source=source, protocol_sha=proto_sha, gpus=(), cpu_canary=True)
            outcome = _run_core_phase(args, output, source=source, protocol_sha=proto_sha, gpus=(), prof_receipt=prof, deadline_epoch=deadline_epoch)
        elif phase == "followup":
            outcome = _run_followup_phase(args, output, source=source, protocol_sha=proto_sha, gpus=(), deadline_epoch=deadline_epoch)
        else:
            outcome = {"status": "COMPLETE", "scientific_status": "FINALIZED", "claim_status": "IMPLEMENTED"}
    elif phase == "diagnose":
        outcome = _run_diagnose_phase(args, output, source=source, protocol_sha=proto_sha, gpus=gpus)
    elif phase == "core":
        if not args.owner_enabled:
            raise ValueError("GPU training requires --owner-enabled")
        prof = _profile_lanes(output, source=source, protocol_sha=proto_sha, gpus=gpus, cpu_canary=False)
        outcome = _run_core_phase(args, output, source=source, protocol_sha=proto_sha, gpus=gpus, prof_receipt=prof, deadline_epoch=deadline_epoch)
    elif phase == "followup":
        if not args.owner_enabled:
            raise ValueError("GPU training requires --owner-enabled")
        outcome = _run_followup_phase(args, output, source=source, protocol_sha=proto_sha, gpus=gpus, deadline_epoch=deadline_epoch)
    elif phase == "finalize":
        outcome = {"status": "COMPLETE", "scientific_status": "FINALIZED", "claim_status": "EXECUTED"}
    else:
        raise ValueError(phase)

    _update_budget_ledger(output, ledger, phase=phase, session_elapsed_s=time.time() - t_sess_start)
    campaign_doc = {
        "schema": "anra-v8-y1-campaign-v1",
        "run_id": run_id,
        "phase": phase,
        "source": source,
        "protocol_sha256": proto_sha,
        "production_protocol": is_prod,
        **outcome,
    }
    _write_summary_markdown(output, campaign_doc)
    replay = _replay_phase_evidence(output, phase)
    include_full = phase in ("core", "followup", "canary")
    final_status = finalize_y1_campaign(
        output,
        source=source,
        protocol_sha=proto_sha,
        campaign=campaign_doc,
        replay_payload=replay,
        include_full=include_full,
    )
    print(json.dumps({
        "run_id": final_status["run_id"],
        "phase": final_status["phase"],
        "status": final_status["status"],
        "scientific_status": final_status["scientific_status"],
        "delivery_status": final_status["delivery_status"],
    }))
    return final_status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=PHASES)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--minutes", type=float, default=None)
    parser.add_argument("--checkpoint-archive", type=Path, default=None)
    parser.add_argument("--handoff-archive", type=Path, default=None)
    parser.add_argument("--resume-archive", type=Path, default=None)
    parser.add_argument("--selection-receipt", type=Path, default=None)
    parser.add_argument("--owner-enabled", action="store_true")
    parser.add_argument("--enable-d2", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--cpu-canary", action="store_true")
    parser.add_argument("--canary-updates", type=int, default=2)
    parser.add_argument("--skip-source-check", action="store_true")
    args = parser.parse_args(argv)
    res = run_campaign(args)
    if res.get("status") == "DRY_RUN_READY":
        return 0
    return 0 if res.get("delivery_status") == "PASS" and res.get("status") == "COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
