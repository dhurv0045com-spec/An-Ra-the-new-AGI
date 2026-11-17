"""ELENDRIL V8-X1 session campaign orchestrator using Planthir IndependentWorkerScheduler."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Mapping, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from anra_v5.elendril.codec import digest
from anra_v5.elendril.training import runtime_identity
from anra_v5.elendril.v8x.analysis import (
    build_x1_report_markdown,
    cluster_bootstrap_paired_world_diff,
    evaluate_x1_development_selection,
)
from anra_v5.elendril.v8x.evaluation import run_h01_frozen_intervention
from anra_v5.elendril.v8x.protocol import (
    X1_MILESTONES,
    X1_SESSION_CONFIG,
    X1_SESSIONS,
    session_jobs,
    verify_against_design_registry,
)
from anra_v5.planthir.agent_tools import init_spec_template, validate_spec_document
from anra_v5.planthir.scheduler import IndependentWorkerScheduler, WorkerJobSpec

from .elendril_v8x_export import export_x1_archives
from .elendril_v8x_worker import prepare_splits, profile_x1_condition, run_arm_canary

DISK_RESERVE_BYTES = 2 * 1024 * 1024 * 1024  # 2 GiB mandatory reserve (Section 10.3)
EXPORT_RESERVE_SECONDS = int(X1_SESSION_CONFIG["minimum_export_reserve_minutes"]) * 60


def _atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _validate_commit_sha(source: str) -> str:
    if len(source) != 40 or any(c not in "0123456789abcdef" for c in source):
        raise ValueError(f"--source must be a 40-char lowercase hex SHA, got {source!r}")
    return source


def _parse_devices(devices_arg: str, *, cpu_fixtures: bool) -> list[str]:
    if cpu_fixtures:
        return ["cpu:0", "cpu:1"]
    parts = [p.strip() for p in devices_arg.split(",") if p.strip()]
    return [f"cuda:{int(p)}" if p.isdigit() else p for p in parts]


def prepare_session(
    *,
    session: str,
    source_commit: str,
    work_dir: Path,
    cpu_fixtures: bool = False,
) -> dict[str, Any]:
    """Subcommand `prepare`: write identity, environment, registry, and verified splits (no training)."""

    source_commit = _validate_commit_sha(source_commit)
    if session not in X1_SESSIONS:
        raise ValueError(f"unknown X1 session {session!r}")
    jobs = list(session_jobs(session))
    if len(jobs) > 6:
        raise ValueError(f"session {session} exceeds max 6 jobs per default session ({len(jobs)})")

    splits_manifest = prepare_splits(work_dir=work_dir, cpu_fixtures=cpu_fixtures)
    ident = {
        "schema": "elendril-x1-session-identity-v1",
        "session": session,
        "source_commit": source_commit,
        "cpu_fixtures": bool(cpu_fixtures),
        "data_manifest_sha256": splits_manifest["manifest_sha256"],
        "registry_sha256": digest(jobs),
    }
    env_doc = {
        "schema": "elendril-x1-environment-v1",
        "runtime": runtime_identity(),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "cuda_devices": [
            {"index": i, "name": torch.cuda.get_device_name(i)}
            for i in range(torch.cuda.device_count() if torch.cuda.is_available() else 0)
        ],
        "free_disk_bytes": int(shutil.disk_usage(work_dir).free),
    }
    reg_doc = {
        "schema": "elendril-x1-session-registry-v1",
        "session": session,
        "expected_milestones": list(X1_MILESTONES),
        "jobs": jobs,
    }
    _atomic_write_json(work_dir / "identity.json", ident)
    _atomic_write_json(work_dir / "environment.json", env_doc)
    _atomic_write_json(work_dir / "registry.json", reg_doc)
    return {
        "status": "PREPARED",
        "session": session,
        "source_commit": source_commit,
        "job_count": len(jobs),
        "data_manifest_sha256": splits_manifest["manifest_sha256"],
    }


def validate_session(
    *,
    session: str,
    source_commit: str,
    work_dir: Path,
    cpu_fixtures: bool = False,
) -> dict[str, Any]:
    """Subcommand `validate`: check design registry, splits, and bounded CPU Planthir fixtures (never full S20)."""

    source_commit = _validate_commit_sha(source_commit)
    id_path = work_dir / "identity.json"
    reg_path = work_dir / "registry.json"
    man_path = work_dir / "data" / "manifest.json"
    if not (id_path.is_file() and reg_path.is_file() and man_path.is_file()):
        prep = prepare_session(session=session, source_commit=source_commit, work_dir=work_dir, cpu_fixtures=cpu_fixtures)
    else:
        ident = json.loads(id_path.read_text(encoding="utf-8"))
        if ident.get("source_commit") != source_commit or ident.get("session") != session:
            raise ValueError("work_dir identity.json mismatch against --source / --session")
        prep = {"status": "EXISTING_IDENTITY_VERIFIED"}

    design_check = verify_against_design_registry()
    tpl = init_spec_template(backend_id="x1_s20", canary=True)
    tpl["source_commit"] = source_commit
    spec_val = validate_spec_document(tpl, canary_override=True)
    if not spec_val["valid"]:
        raise ValueError(f"Planthir x1_s20 template validation failed: {spec_val['errors']}")

    reg = json.loads(reg_path.read_text(encoding="utf-8"))
    canary_receipt = None
    if cpu_fixtures:
        j0 = reg["jobs"][0]
        canary_receipt = run_arm_canary(
            arm=str(j0["architecture"]),
            objective=str(j0["objective"]),
            device="cpu",
            work_dir=work_dir / "validate_canary",
            source_commit=source_commit,
            cpu_fixtures=True,
        )
        if not canary_receipt["passed"]:
            raise RuntimeError(f"CPU fixture canary failed: {canary_receipt}")

    return {
        "status": "VALIDATED",
        "session": session,
        "source_commit": source_commit,
        "cpu_fixtures": bool(cpu_fixtures),
        "design_registry_verified": bool(design_check.get("verified")),
        "planthir_spec_valid": True,
        "prepare": prep,
        "canary": canary_receipt,
    }


def admit_session(
    *,
    session: str,
    source_commit: str,
    work_dir: Path,
    devices_arg: str = "0,1",
    max_minutes: int = 330,
    cpu_fixtures: bool = False,
) -> dict[str, Any]:
    """Subcommand `admit`: fail-closed 2xT4 hardware, disk, Stage A canary, Stage B profile & time admission."""

    source_commit = _validate_commit_sha(source_commit)
    if not (work_dir / "identity.json").is_file():
        prepare_session(session=session, source_commit=source_commit, work_dir=work_dir, cpu_fixtures=cpu_fixtures)

    ident = json.loads((work_dir / "identity.json").read_text(encoding="utf-8"))
    if ident.get("source_commit") != source_commit or ident.get("session") != session:
        raise ValueError("work_dir identity.json mismatch during admit")
    reg = json.loads((work_dir / "registry.json").read_text(encoding="utf-8"))
    jobs: list[dict[str, Any]] = reg["jobs"]
    slots = _parse_devices(devices_arg, cpu_fixtures=cpu_fixtures)

    # Hardware & dual-GPU admission (Section 10.2: missing 2nd GPU does not silently run 1-GPU protocol)
    if not cpu_fixtures:
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2 or len(slots) < 2:
            doc = {
                "schema": "elendril-x1-admission-v1",
                "status": "NO_GO",
                "reason_code": "GPU_ADMISSION_FAILED",
                "reason": (
                    f"ELENDRIL V8-X1 requires 2 CUDA GPUs; detected cuda_available={torch.cuda.is_available()}, "
                    f"device_count={torch.cuda.device_count() if torch.cuda.is_available() else 0}, requested_slots={slots}"
                ),
                "session": session,
                "source_commit": source_commit,
            }
            _atomic_write_json(work_dir / "admission.json", doc)
            return doc

    free_bytes = int(shutil.disk_usage(work_dir).free)
    if not cpu_fixtures and free_bytes < DISK_RESERVE_BYTES + 4 * 1024 * 1024 * 1024:
        doc = {
            "schema": "elendril-x1-admission-v1",
            "status": "NO_GO",
            "reason_code": "DISK_ADMISSION_FAILED",
            "reason": f"insufficient disk space ({free_bytes} bytes free; need >= 6 GiB before profiling)",
            "session": session,
            "source_commit": source_commit,
        }
        _atomic_write_json(work_dir / "admission.json", doc)
        return doc

    # Unique (arm, objective) pairs in session
    conds: dict[tuple[str, str], dict[str, Any]] = {}
    for j in jobs:
        key = (str(j["architecture"]), str(j["objective"]))
        conds.setdefault(key, j)

    canaries: dict[str, Any] = {}
    profiles: dict[str, Any] = {}
    for idx, ((arm, obj), rep_job) in enumerate(sorted(conds.items())):
        dev = "cpu" if cpu_fixtures else slots[idx % len(slots)]
        ck = f"{arm}:{obj}"
        c_res = run_arm_canary(
            arm=arm,
            objective=obj,
            device=dev,
            work_dir=work_dir / "admission_canaries",
            source_commit=source_commit,
            cpu_fixtures=cpu_fixtures,
        )
        canaries[ck] = c_res
        if not c_res["passed"]:
            doc = {
                "schema": "elendril-x1-admission-v1",
                "status": "NO_GO",
                "reason_code": "CANARY_FAILED",
                "reason": f"canary failed for {ck}",
                "canaries": canaries,
                "session": session,
                "source_commit": source_commit,
            }
            _atomic_write_json(work_dir / "admission.json", doc)
            return doc

        p_res = profile_x1_condition(
            job=rep_job,
            device=dev,
            work_dir=work_dir,
            source_commit=source_commit,
            cpu_fixtures=cpu_fixtures,
        )
        profiles[ck] = p_res

    # Forecast wall time & disk across 2 parallel lanes
    usable_seconds = max(60.0, float(max_minutes) * 60.0 - EXPORT_RESERVE_SECONDS)
    total_serial_seconds = 0.0
    estimated_disk_bytes = 0
    for j in jobs:
        ck = f"{j['architecture']}:{j['objective']}"
        pr = profiles[ck]
        updates = 4 if cpu_fixtures else int(j["updates"])
        eval_eps = 12 if cpu_fixtures else (4 * 576 + 2304 + 640)
        job_s = updates * float(pr["p95_step_seconds"]) + eval_eps * float(pr["eval_seconds_per_episode"])
        total_serial_seconds += job_s
        estimated_disk_bytes += 2 * int(pr["measured_full_checkpoint_bytes"]) + 5 * int(pr["measured_milestone_bytes"])

    lanes = max(1, min(len(slots), len(jobs)))
    forecast_seconds = (total_serial_seconds / lanes) * 1.25  # 25% safety margin
    disk_needed_total = estimated_disk_bytes * 2 + (0 if cpu_fixtures else DISK_RESERVE_BYTES)

    if not cpu_fixtures and forecast_seconds > usable_seconds:
        status = "NO_GO"
        reason_code = "TIME_ADMISSION_FAILED"
        reason = f"forecast {forecast_seconds:.1f}s > usable budget {usable_seconds:.1f}s (lanes={lanes})"
    elif not cpu_fixtures and free_bytes < disk_needed_total:
        status = "NO_GO"
        reason_code = "DISK_ADMISSION_FAILED"
        reason = f"free disk {free_bytes} < estimated required {disk_needed_total} (including 2 GiB reserve)"
    else:
        status = "ADMITTED"
        reason_code = "ADMITTED"
        reason = None

    doc = {
        "schema": "elendril-x1-admission-v1",
        "status": status,
        "reason_code": reason_code,
        "reason": reason,
        "session": session,
        "source_commit": source_commit,
        "cpu_fixtures": bool(cpu_fixtures),
        "lanes": lanes,
        "usable_seconds": usable_seconds,
        "forecast_seconds": float(forecast_seconds),
        "estimated_job_disk_bytes": int(estimated_disk_bytes),
        "free_disk_bytes": int(free_bytes),
        "canaries": canaries,
        "profiles": profiles,
    }
    _atomic_write_json(work_dir / "admission.json", doc)
    return doc


def _analyze_session_outcomes(
    *,
    session: str,
    source_commit: str,
    execution_status: str,
    work_dir: Path,
    jobs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Assemble session comparisons, bootstrap CIs, diagnosis (observed vs supported), and RESULTS.md."""

    analysis_dir = work_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    charts_dir = work_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    job_summaries: list[dict[str, Any]] = []
    recs_by_job: dict[str, list[dict[str, Any]]] = {}
    for j in jobs:
        jid = str(j["job_id"])
        jpath = work_dir / "jobs" / jid / "job.json"
        if not jpath.is_file():
            continue
        jdoc = json.loads(jpath.read_text(encoding="utf-8"))
        ms_dirs = sorted((work_dir / "jobs" / jid / "milestones").glob("u*"))
        final_sum: dict[str, Any] = {}
        final_recs: list[dict[str, Any]] = []
        if ms_dirs:
            last_ms = ms_dirs[-1]
            spath = last_ms / "summary.json"
            epath = last_ms / "episodes.jsonl"
            if spath.is_file():
                final_sum = json.loads(spath.read_text(encoding="utf-8"))
            if epath.is_file():
                final_recs = [json.loads(ln) for ln in epath.read_text(encoding="utf-8").splitlines() if ln.strip()]
        recs_by_job[jid] = final_recs
        job_summaries.append(
            {
                "job_id": jid,
                "architecture": jdoc["architecture"],
                "objective": jdoc["objective"],
                "init_seed": jdoc["init_seed"],
                "completed_updates": jdoc["completed_updates"],
                "status": jdoc["status"],
                "summary": final_sum,
            }
        )

    # Paired bootstrap & seed-level contrast between the two session arms when complete
    by_arch: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in job_summaries:
        key = f"{row['architecture']}:{row['objective']}"
        by_arch[key][int(row["init_seed"])] = row

    cond_keys = sorted(by_arch)
    comparisons: list[dict[str, Any]] = []
    dev_gate: dict[str, Any] | None = None
    if len(cond_keys) >= 2:
        c_key, t_key = cond_keys[0], cond_keys[1]
        shared_seeds = sorted(set(by_arch[c_key]) & set(by_arch[t_key]))
        seed_pairs: list[dict[str, Any]] = []
        ctrl_all: list[dict[str, Any]] = []
        trt_all: list[dict[str, Any]] = []
        for s in shared_seeds:
            cj = by_arch[c_key][s]
            tj = by_arch[t_key][s]
            c_m12 = float(cj["summary"].get("macro_12_family_accuracy", 0.0))
            t_m12 = float(tj["summary"].get("macro_12_family_accuracy", 0.0))
            seed_pairs.append({"init_seed": s, "order_seed": s, "ctrl_macro_12": c_m12, "trt_macro_12": t_m12})
            ctrl_all.extend(recs_by_job.get(cj["job_id"], []))
            trt_all.extend(recs_by_job.get(tj["job_id"], []))
        if ctrl_all and trt_all:
            replicates = 200 if len(ctrl_all) < 500 else 10_000
            boot = cluster_bootstrap_paired_world_diff(ctrl_all, trt_all, replicates=replicates)
            comparisons.append(
                {
                    "control": c_key,
                    "treatment": t_key,
                    "shared_seeds": shared_seeds,
                    "cluster_bootstrap": boot,
                }
            )
            dev_gate = evaluate_x1_development_selection(seed_pairs)

    comp_doc = {
        "schema": "elendril-x1-comparisons-v1",
        "session": session,
        "comparisons": comparisons,
        "development_gate": dev_gate,
        "h01_status": run_h01_frozen_intervention(),
    }
    diag_doc = {
        "schema": "elendril-x1-diagnosis-v1",
        "session": session,
        "observed_jobs": len(job_summaries),
        "supported_claims": [c for c in comparisons if c["cluster_bootstrap"].get("excludes_zero")],
        "unresolved": [] if comparisons else ["incomplete pair comparison"],
    }
    _atomic_write_json(analysis_dir / "comparisons.json", comp_doc)
    _atomic_write_json(analysis_dir / "diagnosis.json", diag_doc)
    _atomic_write_json(charts_dir / "summary_table.json", {"jobs": job_summaries})

    sci_status = "HYPOTHESIS_SUPPORTED_DEV" if (dev_gate and dev_gate.get("qualified")) else "INCONCLUSIVE_OR_PENDING_CONF"
    md = build_x1_report_markdown(
        session=session,
        source_commit=source_commit,
        execution_status=execution_status,
        delivery_status="EXPORTED_VERIFIED_PENDING_RECEIPT",
        scientific_status=sci_status,
        cause_status="EVALUATED" if comparisons else "UNRESOLVED",
        job_summaries=job_summaries,
        notes=[f"H01 status: {comp_doc['h01_status']['status']}"],
    )
    (analysis_dir / "RESULTS.md").write_text(md, encoding="utf-8")
    return {"comparisons": comp_doc, "diagnosis": diag_doc}


def run_or_resume_session(
    *,
    session: str,
    source_commit: str,
    work_dir: Path,
    output_dir: Path | None = None,
    devices_arg: str = "0,1",
    max_minutes: int = 330,
    owner_authorized: bool = False,
    resume: bool = False,
    cpu_fixtures: bool = False,
    updates_override: int | None = None,
) -> dict[str, Any]:
    """Subcommands `run` / `resume`: execute session via Planthir IndependentWorkerScheduler."""

    source_commit = _validate_commit_sha(source_commit)
    out_dir = output_dir or (work_dir / "exports")
    if not (work_dir / "identity.json").is_file():
        prepare_session(session=session, source_commit=source_commit, work_dir=work_dir, cpu_fixtures=cpu_fixtures)

    reg = json.loads((work_dir / "registry.json").read_text(encoding="utf-8"))
    jobs: list[dict[str, Any]] = reg["jobs"]
    expected_ids = [str(j["job_id"]) for j in jobs]

    if not owner_authorized:
        receipt = export_x1_archives(
            work_dir=work_dir,
            output_dir=out_dir,
            session=session,
            status="NOT_LAUNCHED",
            source_commit=source_commit,
            completed_jobs=[],
            expected_jobs=expected_ids,
            missing_jobs=expected_ids,
            reason="NOT_LAUNCHED: --owner-authorized switch is false",
        )
        return {"status": "NOT_LAUNCHED", "reason": "owner switch is false", "export": receipt}

    adm_path = work_dir / "admission.json"
    if not adm_path.is_file():
        receipt = export_x1_archives(
            work_dir=work_dir,
            output_dir=out_dir,
            session=session,
            status="NO_GO",
            source_commit=source_commit,
            completed_jobs=[],
            expected_jobs=expected_ids,
            missing_jobs=expected_ids,
            reason="ADMISSION_REQUIRED: run admit before run/resume",
        )
        return {"status": "NO_GO", "reason": "ADMISSION_REQUIRED", "export": receipt}

    adm = json.loads(adm_path.read_text(encoding="utf-8"))
    if (
        adm.get("status") != "ADMITTED"
        or adm.get("source_commit") != source_commit
        or adm.get("session") != session
    ):
        reason = f"ADMISSION_INVALID: status={adm.get('status')!r}, reason={adm.get('reason')}"
        receipt = export_x1_archives(
            work_dir=work_dir,
            output_dir=out_dir,
            session=session,
            status="NO_GO",
            source_commit=source_commit,
            completed_jobs=[],
            expected_jobs=expected_ids,
            missing_jobs=expected_ids,
            reason=reason,
        )
        return {"status": "NO_GO", "reason": reason, "export": receipt}

    slots = _parse_devices(devices_arg, cpu_fixtures=cpu_fixtures)
    usable_sec = max(60.0, float(max_minutes) * 60.0 - EXPORT_RESERVE_SECONDS)
    deadline_epoch = time.time() + usable_sec

    pending_jobs: list[dict[str, Any]] = []
    for j in jobs:
        jid = str(j["job_id"])
        jpath = work_dir / "jobs" / jid / "job.json"
        if resume and jpath.is_file():
            jdoc = json.loads(jpath.read_text(encoding="utf-8"))
            if jdoc.get("status") == "COMPLETE":
                continue
        pending_jobs.append(j)

    repo_root = str(Path(__file__).resolve().parents[1])
    sched = IndependentWorkerScheduler(
        coordinator_root=work_dir / "scheduler",
        max_parallel=max(1, len(slots)),
        allow_shared_cuda_device=cpu_fixtures,
    )

    # Execute pending jobs in waves of len(slots) across independent GPU/CPU workers
    wave_size = max(1, len(slots))
    for w_start in range(0, len(pending_jobs), wave_size):
        if time.time() >= deadline_epoch:
            break
        wave = pending_jobs[w_start : w_start + wave_size]
        worker_specs: list[WorkerJobSpec] = []
        for idx, j in enumerate(wave):
            jid = str(j["job_id"])
            slot_dev = slots[idx % len(slots)]
            # Note: IndependentWorkerScheduler sets CUDA_VISIBLE_DEVICES to the assigned GPU index,
            # so inside that worker subprocess the visible GPU ordinal is always cuda:0.
            child_device = "cuda:0" if slot_dev.startswith("cuda") else "cpu"
            cmd = [
                sys.executable,
                "-m",
                "tools.elendril_v8x_worker",
                "run-job",
                "--job-id",
                jid,
                "--work-dir",
                str(work_dir),
                "--source",
                source_commit,
                "--device",
                child_device,
                "--deadline-epoch",
                f"{deadline_epoch:.3f}",
            ]
            if resume:
                cmd.append("--resume")
            if cpu_fixtures:
                cmd.append("--cpu-fixtures")
            if updates_override is not None:
                cmd.extend(["--updates-override", str(int(updates_override))])
            worker_specs.append(
                WorkerJobSpec(
                    worker_id=jid,
                    work_root=work_dir / "worker_slots" / jid,
                    command=tuple(cmd),
                    device=slot_dev,
                    env={"PYTHONPATH": repo_root},
                )
            )
        sched.run(worker_specs)

    completed_jobs: list[str] = []
    failed_jobs: list[str] = []
    missing_jobs: list[str] = []
    partial_jobs: list[str] = []
    for jid in expected_ids:
        jpath = work_dir / "jobs" / jid / "job.json"
        if not jpath.is_file():
            missing_jobs.append(jid)
            continue
        st = json.loads(jpath.read_text(encoding="utf-8")).get("status")
        if st == "COMPLETE":
            completed_jobs.append(jid)
        elif st == "PARTIAL":
            partial_jobs.append(jid)
        else:
            failed_jobs.append(jid)

    if len(completed_jobs) == len(expected_ids) and not failed_jobs:
        final_status = "COMPLETE"
    elif completed_jobs or partial_jobs:
        final_status = "PARTIAL"
    else:
        final_status = "NO_GO"

    _analyze_session_outcomes(
        session=session,
        source_commit=source_commit,
        execution_status=final_status,
        work_dir=work_dir,
        jobs=jobs,
    )

    export_receipt = export_x1_archives(
        work_dir=work_dir,
        output_dir=out_dir,
        session=session,
        status=final_status,
        source_commit=source_commit,
        completed_jobs=completed_jobs,
        expected_jobs=expected_ids,
        failed_jobs=failed_jobs,
        missing_jobs=[*partial_jobs, *missing_jobs],
    )
    return {
        "status": final_status,
        "session": session,
        "source_commit": source_commit,
        "completed_jobs": completed_jobs,
        "partial_jobs": partial_jobs,
        "failed_jobs": failed_jobs,
        "missing_jobs": missing_jobs,
        "export": export_receipt,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ELENDRIL V8-X1 Campaign Orchestrator")
    sub = parser.add_subparsers(dest="command", required=True)

    for cname in ("prepare", "validate", "admit", "run", "resume"):
        sp = sub.add_parser(cname, help=f"{cname.capitalize()} an ELENDRIL V8-X1 session.")
        sp.add_argument("--source", required=True, help="40-char lowercase hex commit SHA")
        sp.add_argument("--session", required=True, choices=tuple(X1_SESSIONS))
        sp.add_argument("--work-dir", type=Path, required=True)
        sp.add_argument("--cpu-fixtures", action="store_true")
        if cname in ("admit", "run", "resume"):
            sp.add_argument("--devices", default="0,1")
            sp.add_argument("--max-minutes", type=int, default=int(X1_SESSION_CONFIG["session_max_minutes"]))
        if cname in ("run", "resume"):
            sp.add_argument("--output-dir", type=Path, default=None)
            sp.add_argument("--owner-authorized", action="store_true")
            sp.add_argument("--updates-override", type=int, default=None)

    args = parser.parse_args(argv)
    if args.command == "prepare":
        res = prepare_session(
            session=args.session,
            source_commit=args.source,
            work_dir=args.work_dir,
            cpu_fixtures=bool(args.cpu_fixtures),
        )
        code = 0
    elif args.command == "validate":
        res = validate_session(
            session=args.session,
            source_commit=args.source,
            work_dir=args.work_dir,
            cpu_fixtures=bool(args.cpu_fixtures),
        )
        code = 0
    elif args.command == "admit":
        res = admit_session(
            session=args.session,
            source_commit=args.source,
            work_dir=args.work_dir,
            devices_arg=args.devices,
            max_minutes=int(args.max_minutes),
            cpu_fixtures=bool(args.cpu_fixtures),
        )
        code = 0 if res["status"] == "ADMITTED" else 1
    else:
        res = run_or_resume_session(
            session=args.session,
            source_commit=args.source,
            work_dir=args.work_dir,
            output_dir=args.output_dir,
            devices_arg=args.devices,
            max_minutes=int(args.max_minutes),
            owner_authorized=bool(args.owner_authorized),
            resume=(args.command == "resume"),
            cpu_fixtures=bool(args.cpu_fixtures),
            updates_override=args.updates_override,
        )
        code = 0 if res["status"] in ("COMPLETE", "PARTIAL") else 1

    print(json.dumps(res, indent=2, sort_keys=True))
    return code


__all__ = [
    "admit_session",
    "prepare_session",
    "run_or_resume_session",
    "validate_session",
]

if __name__ == "__main__":
    raise SystemExit(main())
