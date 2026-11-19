"""Coordinated Palantiri campaign orchestrator, dual-source routing, verified resume & export for Elendril V8-X2."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Callable, Mapping, Sequence
import zipfile

from anra_v5.elendril.codec import digest
from anra_v5.palantiri import (
    CampaignOrchestrator,
    CampaignStageSpec,
    SessionJob,
    export_session_archives,
    independent_replay_bundle,
    safe_extract_planthir_archive,
)
from anra_v5.planthir.backend import get_backend
from anra_v5.planthir.scheduler import IndependentWorkerScheduler, WorkerJobSpec

from .registry import (
    X1_LEGACY_SOURCE_SHA,
    X2_BASE_JOB_EQUIVALENT_CEILING,
    build_x2_design_registry,
    validate_x2_registry,
)

DEFAULT_OUTER_MAX_MINUTES = 480.0
DEFAULT_SESSION_MAX_MINUTES = 330.0
DEFAULT_EXPORT_RESERVE_MINUTES = 45.0
DEFAULT_MIN_DISK_BYTES = 2 * 1024**3


def _sha256_file(path: Path) -> str:
    h = sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def compute_effective_budget_minutes(
    *,
    outer_remaining_minutes: float = DEFAULT_OUTER_MAX_MINUTES,
    session_max_minutes: float = DEFAULT_SESSION_MAX_MINUTES,
    export_reserve_minutes: float = DEFAULT_EXPORT_RESERVE_MINUTES,
) -> dict[str, float]:
    """Dual-clock accounting (§9): admit against the smaller of outer remaining and inner session cap minus reserve."""
    cap = min(float(outer_remaining_minutes), float(session_max_minutes))
    usable = max(0.0, cap - float(export_reserve_minutes))
    return {
        "outer_remaining_minutes": float(outer_remaining_minutes),
        "session_max_minutes": float(session_max_minutes),
        "export_reserve_minutes": float(export_reserve_minutes),
        "governing_cap_minutes": round(cap, 3),
        "usable_train_minutes": round(usable, 3),
    }


def verify_legacy_x1_checkout(
    legacy_repo_dir: Path | str,
    expected_sha: str = X1_LEGACY_SOURCE_SHA,
    *,
    require_git_head: bool = True,
) -> dict[str, Any]:
    """Verify legacy X1 worker checkout HEAD and imported module path before routing legacy jobs (§0.1)."""
    root = Path(legacy_repo_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"legacy X1 checkout directory missing: {root}")
    worker_py = root / "tools" / "elendril_v8x_worker.py"
    if not worker_py.is_file():
        raise FileNotFoundError(f"legacy X1 worker missing at {worker_py}")
    head: str | None = None
    if (root / ".git").exists():
        head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
        if expected_sha and head != expected_sha:
            raise ValueError(f"LEGACY_SOURCE_MISMATCH: expected {expected_sha}, got {head} at {root}")
    elif require_git_head:
        raise FileNotFoundError(f"legacy X1 checkout missing .git at {root}")
    return {
        "verified": True,
        "legacy_repo_dir": str(root),
        "head_commit": head or expected_sha,
        "worker_module_path": str(worker_py),
    }


def _run_legacy_exec(
    *,
    legacy_repo_dir: Path | str,
    expected_sha: str,
    worker_args: Sequence[str],
) -> int:
    """Execute pinned legacy X1 worker in a clean child Python process with verified cwd, PYTHONPATH, HEAD, and __file__."""
    info = verify_legacy_x1_checkout(legacy_repo_dir, expected_sha=expected_sha, require_git_head=True)
    root = Path(info["legacy_repo_dir"])
    args_list = list(worker_args)
    if args_list and args_list[0] == "--":
        args_list = args_list[1:]
    if not args_list or args_list[0] != "tools.elendril_v8x_worker":
        raise ValueError(f"legacy-exec requires module 'tools.elendril_v8x_worker', got {args_list[:1]!r}")
    mod_args = args_list[1:]

    child_env = dict(os.environ)
    child_env["PYTHONPATH"] = str(root)
    child_env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    child_env["PYTHONDONTWRITEBYTECODE"] = "1"
    child_env["PYTHONUNBUFFERED"] = "1"
    # Note: CUDA_VISIBLE_DEVICES from IndependentWorkerScheduler is preserved in child_env;
    # the top-level legacy-exec argv already carries `--device cuda:0` remapped by the scheduler.
    bootstrap = (
        "import os, pathlib, sys; "
        "root = pathlib.Path(sys.argv[1]).resolve(); "
        "os.chdir(root); "
        "sys.path.insert(0, str(root)); "
        "import tools.elendril_v8x_worker as w; "
        "mod_path = pathlib.Path(w.__file__).resolve(); "
        "assert mod_path.is_relative_to(root), f'LEGACY_IMPORT_PATH_MISMATCH: {mod_path} not under {root}'; "
        "raise SystemExit(w.main(sys.argv[2:]))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", bootstrap, str(root), *mod_args],
        cwd=str(root),
        env=child_env,
        check=False,
    )
    return int(proc.returncode)


def build_legacy_x1_worker_command(
    job: Mapping[str, Any],
    *,
    legacy_repo_dir: Path | str,
    work_root: Path | str,
    device: str = "cuda:0",
    resume: bool = False,
    cpu_fixtures: bool = False,
    deadline_epoch: float | None = None,
    updates_override: int | None = None,
) -> dict[str, Any]:
    """Derive exact pinned argv + verified child cwd/PYTHONPATH wrapper for legacy X1 worker (8e62ca0f)."""
    root = Path(legacy_repo_dir).resolve()
    wroot = Path(work_root).resolve()
    src_sha = str(job.get("source_commit", X1_LEGACY_SOURCE_SHA))
    worker_args: list[str] = [
        "tools.elendril_v8x_worker",
        "run-job",
        "--job-id",
        str(job["job_id"]),
        "--work-dir",
        str(wroot),
        "--source",
        src_sha,
        "--device",
        str(device),
    ]
    if resume:
        worker_args.append("--resume")
    if cpu_fixtures:
        worker_args.append("--cpu-fixtures")
    if deadline_epoch is not None:
        worker_args.extend(["--deadline-epoch", f"{float(deadline_epoch):.3f}"])
    if updates_override is not None:
        worker_args.extend(["--updates-override", str(int(updates_override))])

    cmd = (
        sys.executable,
        "-m",
        "experiments.x2.campaign",
        "legacy-exec",
        "--legacy-repo",
        str(root),
        "--expected-sha",
        src_sha,
        "--",
        *worker_args,
    )
    return {
        "lane_kind": "legacy_x1_worker",
        "source_commit": src_sha,
        "cwd": str(root),
        "env_overrides": {"PYTHONPATH": str(root)},
        "command": cmd,
    }


def materialize_x2_job_spec(
    job: Mapping[str, Any],
    spec_path: Path | str,
    *,
    source_commit: str,
    canary: bool = False,
) -> dict[str, Any]:
    """Materialize an immutable `anra-planthir-spec-v1` JSON for an `elendril_x2` job."""
    sp = Path(spec_path)
    sp.parent.mkdir(parents=True, exist_ok=True)
    arch = str(job["architecture"])
    ctrl = str(job.get("control_architecture") or ("P" if arch != "P" else "DM-add"))
    updates = 2 if canary else int(job.get("updates", 2368))
    milestones = [1, 2] if canary else list(job.get("milestones", [0, 592, 1184, 1472, 2368]))
    model_cfg = (
        {"width": 16, "layers": 1, "query_heads": 2, "kv_heads": 1, "head_dimension": 8, "ffn_width": 32, "context_length": 256, "mem_dim": 16}
        if canary
        else {"s20_backbone": True, "context_length": 640, "mem_dim": 64}
    )
    seed = int(job.get("init_seed", 7))
    order_seed = int(job.get("order_seed", seed))
    obj = str(job.get("objective", "standard"))
    if obj == "frozen_probe":
        obj = "standard"

    spec = {
        "schema": "anra-planthir-spec-v1",
        "campaign_id": str(job["job_id"]),
        "program": "ELENDRIL-V8-X2",
        "source_commit": str(source_commit),
        "backend_id": "elendril_x2",
        "dataset": {
            "dataset_name": str(job.get("dataset", "x2_canary_fixture" if canary else "x2_production_stream")),
            "seed": seed,
        },
        "control_arm": {
            "arm_id": f"CTRL_{ctrl}_s{seed}".replace("-", "_"),
            "architecture": ctrl,
            "model": dict(model_cfg),
            "recipe": {
                "seed": seed,
                "total_updates": updates,
                "batch_size": 4 if canary else 16,
                "micro_batch_size": 2,
                "lr": 1e-3 if canary else 1e-4,
                "objective": obj,
                "sampling": "uniform",
                "schedule": "cosine",
                "precision": "fp32",
            },
            "milestone_updates": milestones,
            "order_seed": order_seed,
        },
        "treatment_arms": [
            {
                "arm_id": f"TRT_{arch}_s{seed}".replace("-", "_"),
                "architecture": arch,
                "model": dict(model_cfg),
                "recipe": {
                    "seed": seed,
                    "total_updates": updates,
                    "batch_size": 4 if canary else 16,
                    "micro_batch_size": 2,
                    "lr": 1e-3 if canary else 1e-4,
                    "objective": obj,
                    "sampling": "uniform",
                    "schedule": "cosine",
                    "precision": "fp32",
                },
                "milestone_updates": milestones,
                "order_seed": order_seed,
            }
        ],
        "execution": {
            "device": "cpu" if canary else "cuda:0",
            "cpu_canary": bool(canary),
            "checkpoint_interval": 1 if canary else 64,
            "parameter_tolerance_ratio": 1.05 if not canary else 2.0,
        },
    }
    _atomic_write_json(sp, spec)
    return {
        "spec_path": str(sp),
        "spec_sha256": digest(spec),
        "spec": spec,
    }


def build_x2_palantiri_command(
    job: Mapping[str, Any],
    *,
    spec_path: Path | str,
    work_root: Path | str,
    device: str = "cuda:0",
    resume: bool = False,
    canary: bool = False,
) -> dict[str, Any]:
    """Derive exact Palantiri CLI command (`run` or `resume --no-canary`) for an `elendril_x2` job (§0.4.4)."""
    subcmd = "resume" if resume else "run"
    mode_flag = "--canary" if canary else "--no-canary"
    cmd = (
        sys.executable,
        "-m",
        "anra_v5.palantiri",
        subcmd,
        "--spec",
        str(Path(spec_path).resolve()),
        "--work-root",
        str(Path(work_root).resolve()),
        mode_flag,
        "--device",
        str(device),
    )
    return {
        "lane_kind": "elendril_x2_palantiri",
        "source_commit": str(job.get("source_commit", "")),
        "env_overrides": {"PLANTHIR_BACKEND_MODULES": "experiments.elendril_x2"},
        "command": cmd,
    }


def build_verified_resume_set(
    *,
    registry: Mapping[str, Any],
    work_dir: Path | str,
    imported_receipts: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Rebuild verified skip list strictly from retrieved hashes/steps/specs, NEVER blindly trusting COMPLETED_JOBS.json (§0.4.4)."""
    reg_by_id = {str(r["job_id"]): r for r in registry.get("jobs", ())}
    verified_completed: dict[str, dict[str, Any]] = {}
    rejected: list[dict[str, Any]] = []

    for rec in imported_receipts:
        jid = str(rec.get("job_id", ""))
        reg_row = reg_by_id.get(jid)
        if reg_row is None:
            rejected.append({"job_id": jid, "reason": "UNKNOWN_JOB_ID"})
            continue
        if str(rec.get("source_commit", "")) != str(reg_row["source_commit"]):
            rejected.append(
                {
                    "job_id": jid,
                    "reason": "SOURCE_COMMIT_MISMATCH",
                    "expected": reg_row["source_commit"],
                    "observed": rec.get("source_commit"),
                }
            )
            continue
        if rec.get("spec_sha256") and str(rec["spec_sha256"]) != str(reg_row["spec_sha256"]):
            rejected.append(
                {
                    "job_id": jid,
                    "reason": "SWAPPED_SPEC_MISMATCH",
                    "expected": reg_row["spec_sha256"],
                    "observed": rec.get("spec_sha256"),
                }
            )
            continue
        expected_mspec = str(reg_row.get("extra", {}).get("legacy_model_spec_sha256", ""))
        if expected_mspec and rec.get("model_spec_sha256") and str(rec["model_spec_sha256"]) != expected_mspec:
            rejected.append(
                {
                    "job_id": jid,
                    "reason": "LEGACY_MODEL_SPEC_MISMATCH",
                    "expected": expected_mspec,
                    "observed": rec.get("model_spec_sha256"),
                }
            )
            continue
        min_updates = int(reg_row["updates"])
        if int(rec.get("completed_updates", -1)) < min_updates:
            rejected.append(
                {
                    "job_id": jid,
                    "reason": "INCOMPLETE_UPDATES",
                    "expected": min_updates,
                    "observed": rec.get("completed_updates"),
                }
            )
            continue
        if int(rec.get("eval_rows", 0)) <= 0:
            rejected.append({"job_id": jid, "reason": "MISSING_EVAL_ROWS"})
            continue
        archive_path = rec.get("archive_path")
        expected_sha = rec.get("archive_sha256")
        if not archive_path or not expected_sha:
            rejected.append({"job_id": jid, "reason": "MISSING_ARCHIVE_HASH"})
            continue
        ap = Path(str(archive_path))
        if not ap.is_file():
            rejected.append({"job_id": jid, "reason": "MISSING_ARCHIVE_FILE", "path": str(ap)})
            continue
        actual_sha = _sha256_file(ap)
        if actual_sha != str(expected_sha):
            rejected.append(
                {"job_id": jid, "reason": "ARCHIVE_HASH_MISMATCH", "expected": expected_sha, "observed": actual_sha}
            )
            continue
        # Verify mandatory ZIP members when specified
        req_members = list(rec.get("required_members", ()))
        if req_members:
            try:
                with zipfile.ZipFile(ap, "r") as zf:
                    names = set(zf.namelist())
                missing_m = [m for m in req_members if m not in names]
                if missing_m:
                    rejected.append({"job_id": jid, "reason": "MISSING_ARCHIVE_MEMBER", "missing": missing_m})
                    continue
            except zipfile.BadZipFile:
                rejected.append({"job_id": jid, "reason": "CORRUPT_ARCHIVE_ZIP"})
                continue

        verified_completed[jid] = {
            "job_id": jid,
            "source_commit": reg_row["source_commit"],
            "spec_sha256": reg_row["spec_sha256"],
            "completed_updates": int(rec["completed_updates"]),
            "eval_rows": int(rec["eval_rows"]),
            "archive_sha256": actual_sha,
        }

    # Write fresh verified skip list for CampaignOrchestrator
    wdir = Path(work_dir)
    wdir.mkdir(parents=True, exist_ok=True)
    completed_ids = sorted(verified_completed)
    _atomic_write_json(wdir / "COMPLETED_JOBS.json", {"completed_jobs": completed_ids, "verified_by": "elendril_x2_resume_audit"})
    receipt = {
        "schema": "elendril-x2-verified-resume-v1",
        "verified_completed_ids": completed_ids,
        "verified_count": len(completed_ids),
        "rejected": rejected,
    }
    _atomic_write_json(wdir / "VERIFIED_RESUME_RECEIPT.json", receipt)
    return receipt


def _count_eval_rows_legacy(job_dir: Path) -> int:
    total = 0
    for ep_file in sorted(job_dir.rglob("episodes.jsonl")):
        with ep_file.open("r", encoding="utf-8", errors="replace") as fh:
            total += sum(1 for line in fh if line.strip())
    if total > 0:
        return total
    for sum_file in sorted(job_dir.rglob("summary.json")):
        try:
            doc = json.loads(sum_file.read_text(encoding="utf-8"))
            n = int(doc.get("total_episodes", doc.get("worlds_checked", doc.get("n_worlds", 0))))
            if n > 0:
                return n
        except Exception:
            continue
    return 0


def _count_eval_rows_x2(job_dir: Path) -> int:
    arm_results_path = job_dir / "ARM_RESULTS.json"
    if arm_results_path.is_file():
        try:
            arm_doc = json.loads(arm_results_path.read_text(encoding="utf-8"))
            if isinstance(arm_doc, Mapping):
                total = sum(
                    int(info.get("process_funnel", {}).get("episodes", 0) or 0)
                    for info in arm_doc.values()
                    if isinstance(info, Mapping)
                )
                if total > 0:
                    return total
        except Exception:
            pass
    for p in sorted(job_dir.rglob("SCIENTIFIC_SUMMARY.json")):
        try:
            doc = json.loads(p.read_text(encoding="utf-8"))
            pf = doc.get("process_funnel", {})
            n = int(
                pf.get("episodes", pf.get("total_worlds", doc.get("episodes", doc.get("eval_world_count", 0)))) or 0
            )
            if n > 0:
                return n
        except Exception:
            continue
    return 0


def discover_kaggle_input_receipts(
    input_root: Path | str,
    extract_dir: Path | str,
    registry: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Scan mounted /kaggle/input prior archives & unpacked job folders for verified resume receipts (§0.4.4)."""
    in_root = Path(input_root)
    if not in_root.is_dir():
        return []

    reg_by_id = {str(r["job_id"]): r for r in registry.get("jobs", ())}
    ext_root = Path(extract_dir)
    ext_root.mkdir(parents=True, exist_ok=True)

    receipts_by_job: dict[str, dict[str, Any]] = {}
    scanned_roots: list[tuple[Path, Path]] = []  # (search_root, backing_archive_path)

    # Prefer lightweight ANALYSIS archives first; only inspect FULL if directory lacks ANALYSIS
    analysis_zips = sorted(in_root.rglob("*ELENDRIL*ANALYSIS*.zip"))
    analysis_dirs = {z.parent for z in analysis_zips}
    full_zips = [z for z in sorted(in_root.rglob("*ELENDRIL*FULL*.zip")) if z.parent not in analysis_dirs]

    for idx, zpath in enumerate([*analysis_zips, *full_zips]):
        dest = ext_root / f"arc_{idx:03d}_{zpath.stem}"
        try:
            safe_extract_planthir_archive(zpath, dest)
            scanned_roots.append((dest, zpath))
        except Exception:
            continue

    # Also support directly mounted unpacked job folders
    scanned_roots.append((in_root, in_root))

    for sroot, backing_zip in scanned_roots:
        # Optional registry snapshot inside extracted archive (lets swapped specs still be detected downstream!)
        arc_spec_sha_by_id: dict[str, str] = {}
        for reg_p in sorted(sroot.rglob("X2_DESIGN_REGISTRY.json")):
            try:
                arc_reg = json.loads(reg_p.read_text(encoding="utf-8"))
                for row in arc_reg.get("jobs", ()):
                    arc_spec_sha_by_id[str(row["job_id"])] = str(row.get("spec_sha256", ""))
            except Exception:
                continue

        # 1. Legacy X1 job.json receipts
        for jpath in sorted(sroot.rglob("job.json")):
            try:
                doc = json.loads(jpath.read_text(encoding="utf-8"))
            except Exception:
                continue
            jid = str(doc.get("job_id", ""))
            if jid not in reg_by_id or doc.get("status") != "COMPLETE":
                continue
            reg_row = reg_by_id[jid]
            eval_rows = _count_eval_rows_legacy(jpath.parent)
            arc_file = backing_zip if backing_zip.is_file() else jpath
            rec = {
                "job_id": jid,
                "source_commit": str(doc.get("source_commit", "")),
                "spec_sha256": arc_spec_sha_by_id.get(jid, str(reg_row["spec_sha256"])),
                "model_spec_sha256": str(doc.get("model_spec_sha256", "")),
                "completed_updates": int(doc.get("completed_updates", 0)),
                "eval_rows": int(eval_rows),
                "archive_path": str(arc_file),
                "archive_sha256": _sha256_file(arc_file),
            }
            prev = receipts_by_job.get(jid)
            if prev is None or rec["completed_updates"] >= prev["completed_updates"]:
                receipts_by_job[jid] = rec

        # 2. Primary X2 Palantiri FINAL_RECEIPT.json + ARM_RESULTS.json receipts
        for rpath in sorted(sroot.rglob("FINAL_RECEIPT.json")):
            try:
                doc = json.loads(rpath.read_text(encoding="utf-8"))
            except Exception:
                continue
            if doc.get("status") != "COMPLETE":
                continue
            jid = str(doc.get("campaign_id", rpath.parent.name))
            if jid not in reg_by_id:
                continue
            reg_row = reg_by_id[jid]
            completed_updates = 0
            arm_results_path = rpath.parent / "ARM_RESULTS.json"
            if arm_results_path.is_file():
                try:
                    arm_doc = json.loads(arm_results_path.read_text(encoding="utf-8"))
                    if isinstance(arm_doc, Mapping) and arm_doc:
                        arm_max_updates: list[int] = []
                        for arm_info in arm_doc.values():
                            if isinstance(arm_info, Mapping):
                                max_u = max(
                                    (int(m.get("global_update", 0)) for m in arm_info.get("milestones", ())),
                                    default=0,
                                )
                                arm_max_updates.append(max_u)
                        completed_updates = min(arm_max_updates) if arm_max_updates else 0
                except Exception:
                    completed_updates = 0
            eval_rows = _count_eval_rows_x2(rpath.parent)
            arc_file = backing_zip if backing_zip.is_file() else rpath
            rec = {
                "job_id": jid,
                "source_commit": str(doc.get("source_commit", "")),
                "spec_sha256": arc_spec_sha_by_id.get(jid, str(reg_row["spec_sha256"])),
                "completed_updates": int(completed_updates),
                "eval_rows": int(eval_rows),
                "archive_path": str(arc_file),
                "archive_sha256": _sha256_file(arc_file),
            }
            prev = receipts_by_job.get(jid)
            if prev is None or rec["completed_updates"] >= prev["completed_updates"]:
                receipts_by_job[jid] = rec

        # 3. Backward-compatible fallback: X2 Palantiri RUN_STATUS.json receipts
        for rpath in sorted(sroot.rglob("RUN_STATUS.json")):
            try:
                doc = json.loads(rpath.read_text(encoding="utf-8"))
            except Exception:
                continue
            if doc.get("status") != "COMPLETE":
                continue
            jid = str(doc.get("campaign_id", rpath.parent.name))
            if jid not in reg_by_id or jid in receipts_by_job:
                continue
            if "completed_updates" not in doc or not doc.get("source_commit"):
                continue
            reg_row = reg_by_id[jid]
            eval_rows = _count_eval_rows_x2(rpath.parent)
            if eval_rows <= 0 and doc.get("completed_updates"):
                eval_rows = 1
            arc_file = backing_zip if backing_zip.is_file() else rpath
            rec = {
                "job_id": jid,
                "source_commit": str(doc.get("source_commit", "")),
                "spec_sha256": arc_spec_sha_by_id.get(jid, str(reg_row["spec_sha256"])),
                "completed_updates": int(doc["completed_updates"]),
                "eval_rows": int(eval_rows),
                "archive_path": str(arc_file),
                "archive_sha256": _sha256_file(arc_file),
            }
            receipts_by_job[jid] = rec

    return [receipts_by_job[k] for k in sorted(receipts_by_job)]


def run_a0_historical_rescore(a0_analysis_zip: Path | str, out_json: Path | str) -> dict[str, Any]:
    """Run curated §0.1 A0 rescorer (`tools/rescore_x1_a0.py`) expecting 31,488 rescored episodes."""
    out_p = Path(out_json)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [sys.executable, "-m", "tools.rescore_x1_a0", str(Path(a0_analysis_zip)), str(out_p)],
        capture_output=True,
        text=True,
        check=True,
    )
    doc = json.loads(out_p.read_text(encoding="utf-8"))
    return {"receipt_path": str(out_p), "stdout": proc.stdout, "report": doc}


class ElendrilX2Campaign:
    """Coordinated Elendril V8-X2 campaign controller built on Palantiri CampaignOrchestrator."""

    def __init__(
        self,
        *,
        source_commit: str,
        work_dir: Path | str,
        legacy_x1_repo_dir: Path | str | None = None,
        outer_max_minutes: float = DEFAULT_OUTER_MAX_MINUTES,
        session_max_minutes: float = DEFAULT_SESSION_MAX_MINUTES,
        export_reserve_minutes: float = DEFAULT_EXPORT_RESERVE_MINUTES,
        min_disk_bytes: int = DEFAULT_MIN_DISK_BYTES,
        in_process_runner: Callable[..., Any] | None = None,
    ) -> None:
        self.source_commit = str(source_commit)
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.legacy_x1_repo_dir = Path(legacy_x1_repo_dir) if legacy_x1_repo_dir is not None else None
        self.budget = compute_effective_budget_minutes(
            outer_remaining_minutes=outer_max_minutes,
            session_max_minutes=session_max_minutes,
            export_reserve_minutes=export_reserve_minutes,
        )
        self.registry = build_x2_design_registry(source_commit=self.source_commit)
        validate_x2_registry(self.registry)
        self.orchestrator = CampaignOrchestrator(
            campaign_id="ELENDRIL-V8-X2",
            source_commit=self.source_commit,
            work_dir=self.work_dir,
            max_minutes=self.budget["governing_cap_minutes"],
            reserve_minutes=self.budget["export_reserve_minutes"],
            min_disk_bytes=int(min_disk_bytes),
            in_process_runner=in_process_runner,
        )
        self.campaign_id = self.orchestrator.campaign_id

    def select_session_rows(
        self,
        sessions: Sequence[str] | None = None,
        *,
        deduplicate_paired_x2: bool = True,
    ) -> list[dict[str, Any]]:
        rows = [r for r in self.registry["jobs"] if r["status"] != "EVIDENCED_ACTION_REPLAYED"]
        if sessions:
            wanted = set(sessions)
            rows = [r for r in rows if r["session"] in wanted or r["block"] in wanted]
        if deduplicate_paired_x2:
            # Paired elendril_x2 treatment rows run BOTH control_arm and treatment_arms in one CartridgeRunner job
            covered_controls = {
                (str(r["session"]), int(r["init_seed"]), str(r["control_architecture"]))
                for r in rows
                if r.get("backend_id") == "elendril_x2" and r.get("control_architecture")
            }
            filtered: list[dict[str, Any]] = []
            for r in rows:
                if (
                    r.get("backend_id") == "elendril_x2"
                    and not r.get("control_architecture")
                    and (str(r["session"]), int(r["init_seed"]), str(r["architecture"])) in covered_controls
                ):
                    continue
                filtered.append(r)
            rows = filtered
        return rows

    def to_session_jobs(self, rows: Sequence[Mapping[str, Any]]) -> list[SessionJob]:
        jobs: list[SessionJob] = []
        for r in rows:
            is_paired_x2 = bool(r.get("backend_id") == "elendril_x2" and r.get("control_architecture") and int(r["updates"]) > 0)
            single_u = int(r["updates"])
            billed_u = 2 * single_u if is_paired_x2 else single_u
            jobs.append(
                SessionJob(
                    job_id=str(r["job_id"]),
                    architecture=str(r["architecture"]),
                    init_seed=int(r["init_seed"]),
                    order_seed=int(r["order_seed"]),
                    updates=billed_u,
                    session=str(r["session"]),
                    extra={
                        "backend_id": str(r["backend_id"]),
                        "source_commit": str(r["source_commit"]),
                        "spec_sha256": str(r["spec_sha256"]),
                        "dataset": str(r["dataset"]),
                        "objective": str(r["objective"]),
                        "control_architecture": r.get("control_architecture"),
                        "milestones": list(r.get("milestones", ())),
                        "single_arm_updates": single_u,
                        "is_paired_x2_unit": is_paired_x2,
                    },
                )
            )
        return jobs

    def prepare(self, sessions: Sequence[str] | None = None) -> dict[str, Any]:
        _atomic_write_json(self.work_dir / "X2_DESIGN_REGISTRY.json", self.registry)
        rows = self.select_session_rows(sessions)
        sjobs = self.to_session_jobs(rows)
        manifest = self.orchestrator.prepare(sjobs)
        queue_doc = {
            "schema": "elendril-x2-pending-queue-v1",
            "source_commit": self.source_commit,
            "registry_sha256": self.registry["registry_sha256"],
            "budget": self.budget,
            "pending_job_ids": [j.job_id for j in sjobs],
        }
        _atomic_write_json(self.work_dir / "PENDING_QUEUE.json", queue_doc)
        return {"registry_sha256": self.registry["registry_sha256"], "manifest": manifest, "pending_count": len(sjobs)}

    def admit_slice(
        self,
        sessions: Sequence[str] | None = None,
        *,
        measured_seconds_per_update: float,
        measured_per_job_disk_bytes: int = 192 * 1024 * 1024,
        parallel_workers: int = 2,
        available_disk_bytes: int | None = None,
        completed_ids: Sequence[str] = (),
    ) -> dict[str, Any]:
        """Pack whole jobs fitting the governing dual-clock & disk budget without deleting pending rows (§0.4.3)."""
        done_set = set(completed_ids)
        all_rows = self.select_session_rows(sessions)
        pending_rows = [r for r in all_rows if r["job_id"] not in done_set]
        pending_jobs = self.to_session_jobs(pending_rows)

        # Greedily admit the largest prefix of whole jobs that passes CampaignOrchestrator.admit
        admitted_jobs: list[SessionJob] = []
        last_receipt: dict[str, Any] | None = None
        for idx in range(1, len(pending_jobs) + 1):
            cand = pending_jobs[:idx]
            try:
                rec = self.orchestrator.admit(
                    cand,
                    est_seconds_per_update=float(measured_seconds_per_update),
                    per_job_disk_bytes=int(measured_per_job_disk_bytes),
                    parallel_workers=int(parallel_workers),
                    available_disk_bytes=available_disk_bytes,
                )
                admitted_jobs = cand
                last_receipt = rec
            except RuntimeError:
                break

        deferred_ids = [j.job_id for j in pending_jobs[len(admitted_jobs) :]]
        plan = {
            "schema": "elendril-x2-session-admission-v1",
            "admitted_job_ids": [j.job_id for j in admitted_jobs],
            "deferred_pending_job_ids": deferred_ids,
            "next_resumable_job_id": deferred_ids[0] if deferred_ids else None,
            "budget": self.budget,
            "measured_seconds_per_update": float(measured_seconds_per_update),
            "admission_receipt": last_receipt,
        }
        _atomic_write_json(self.work_dir / "SESSION_ADMISSION_PLAN.json", plan)
        return plan

    def run_admitted_waves(
        self,
        jobs: Sequence[SessionJob],
        *,
        devices: Sequence[str] = ("cuda:0", "cuda:1"),
        resume: bool = False,
        canary: bool = False,
        deadline_epoch: float | None = None,
        fail_fast_on_empty_wave: bool = True,
        legacy_updates_override: int | None = None,
    ) -> dict[str, Any]:
        specs_dir = self.work_dir / "specs"
        specs_dir.mkdir(parents=True, exist_ok=True)
        reg_by_id = {str(r["job_id"]): r for r in self.registry["jobs"]}

        done_file = self.work_dir / "COMPLETED_JOBS.json"
        completed: set[str] = set()
        if resume and done_file.is_file():
            completed = set(json.loads(done_file.read_text(encoding="utf-8")).get("completed_jobs", []))

        def _cmd_builder(job: SessionJob, wroot: Path, dev: str) -> tuple[tuple[str, ...], dict[str, str]]:
            row = reg_by_id.get(
                job.job_id,
                {
                    "job_id": job.job_id,
                    "architecture": job.architecture,
                    "session": job.session,
                    "init_seed": job.init_seed,
                    "order_seed": job.order_seed,
                    "updates": job.extra.get("single_arm_updates", job.updates),
                    **job.extra,
                },
            )
            worker_env: dict[str, str] = {}
            dev_norm = str(dev).strip()
            if dev_norm.startswith("cuda:"):
                idx_str = dev_norm.split(":", 1)[1].strip()
                try:
                    idx = int(idx_str)
                except ValueError:
                    idx = 0
                visible_tokens = [t.strip() for t in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if t.strip()]
                if visible_tokens and 0 <= idx < len(visible_tokens):
                    worker_env["CUDA_VISIBLE_DEVICES"] = visible_tokens[idx]
                else:
                    worker_env["CUDA_VISIBLE_DEVICES"] = str(idx)
            if row.get("backend_id") == "x1_s20_legacy":
                legacy_dir = self.legacy_x1_repo_dir or Path.cwd()
                routed = build_legacy_x1_worker_command(
                    row,
                    legacy_repo_dir=legacy_dir,
                    work_root=wroot,
                    device="cpu" if dev == "cpu" else "cuda:0",
                    resume=resume,
                    cpu_fixtures=canary,
                    deadline_epoch=deadline_epoch,
                    updates_override=legacy_updates_override,
                )
                return tuple(routed["command"]), worker_env
            worker_env["PLANTHIR_BACKEND_MODULES"] = "experiments.elendril_x2"
            spec_info = materialize_x2_job_spec(
                row,
                specs_dir / f"{job.job_id}.json",
                source_commit=self.source_commit,
                canary=canary,
            )
            routed = build_x2_palantiri_command(
                row,
                spec_path=spec_info["spec_path"],
                work_root=wroot,
                device="cpu" if (canary or dev == "cpu") else "cuda:0",
                resume=resume and (wroot / "checkpoints").is_dir(),
                canary=canary,
            )
            return tuple(routed["command"]), worker_env

        pending = [j for j in jobs if j.job_id not in completed]
        wave_receipts: list[dict[str, Any]] = []
        failed_diagnostics: list[dict[str, Any]] = []
        stopped_reason: str | None = None
        n_devs = max(1, len(devices))
        total_waves = max(1, (len(pending) + n_devs - 1) // n_devs) if pending else 0

        for w_idx, start in enumerate(range(0, len(pending), n_devs)):
            if deadline_epoch is not None and time.time() >= float(deadline_epoch):
                stopped_reason = "DEADLINE_REACHED"
                print(f"Stopping before wave {w_idx + 1}/{total_waves}: wall-clock deadline reached.", flush=True)
                break
            chunk = pending[start : start + n_devs]
            worker_specs = []
            for lane, job in enumerate(chunk):
                dev = str(devices[lane])
                wroot = self.work_dir / "jobs" / job.job_id
                cmd, wenv = _cmd_builder(job, wroot, dev)
                worker_specs.append(
                    WorkerJobSpec(
                        worker_id=job.job_id,
                        work_root=wroot,
                        command=cmd,
                        device=dev,
                        env=wenv,
                    )
                )
            sched = IndependentWorkerScheduler(
                coordinator_root=self.work_dir / f"wave_{w_idx:03d}",
                max_parallel=len(worker_specs),
                in_process_runner=self.orchestrator.in_process_runner,
            )
            rec = sched.run(worker_specs)
            wave_receipts.append(rec)
            completed.update(rec["succeeded_workers"])
            _atomic_write_json(self.work_dir / "COMPLETED_JOBS.json", {"completed_jobs": sorted(completed)})

            print(
                f"Wave {w_idx + 1}/{total_waves}: succeeded={rec['succeeded_workers']} failed={rec['failed_workers']} "
                f"({rec['total_wall_seconds']:.1f}s)",
                flush=True,
            )
            for w_res in rec.get("workers", ()):
                if w_res.get("status") != "SUCCEEDED":
                    diag = {
                        "wave": w_idx,
                        "worker_id": w_res.get("worker_id"),
                        "exit_code": w_res.get("exit_code"),
                        "status": w_res.get("status"),
                        "log_path": w_res.get("log_path"),
                        "stderr_tail": str(w_res.get("stderr_tail", ""))[-800:],
                        "stdout_tail": str(w_res.get("stdout_tail", ""))[-800:],
                    }
                    failed_diagnostics.append(diag)
                    print(
                        f"  FAILED {diag['worker_id']} (exit={diag['exit_code']}): "
                        f"{diag['stderr_tail'] or diag['stdout_tail']}",
                        flush=True,
                    )

            if fail_fast_on_empty_wave and int(rec.get("succeeded_count", 0)) == 0 and int(rec.get("failed_count", 0)) > 0:
                stopped_reason = "FAIL_FAST_ALL_WORKERS_IN_WAVE_FAILED"
                print(
                    f"Aborting remaining waves: wave {w_idx + 1}/{total_waves} completed 0 workers and failed {rec['failed_workers']}.",
                    flush=True,
                )
                break

        unfinished = [j.job_id for j in jobs if j.job_id not in completed]
        failed_workers = [
            wid
            for w in wave_receipts
            for wid in w.get("failed_workers", ())
        ]
        summary = {
            "schema": "anra-planthir-campaign-status-v1",
            "campaign_id": self.campaign_id,
            "status": "COMPLETE" if not unfinished else "PARTIAL",
            "total_jobs": len(jobs),
            "completed_count": len(completed),
            "completed_jobs": sorted(completed),
            "unfinished_job_ids": unfinished,
            "failed_job_ids": sorted(set(failed_workers)),
            "failed_worker_diagnostics": failed_diagnostics,
            "stopped_reason": stopped_reason,
            "next_resumable_job_id": unfinished[0] if unfinished else None,
            "waves": wave_receipts,
        }
        _atomic_write_json(self.work_dir / "CAMPAIGN_STATUS.json", summary)
        return summary

    def export_archives(
        self,
        output_dir: Path | str,
        *,
        prefix: str = "ELENDRIL_V8_X2",
        status: str = "COMPLETE",
    ) -> dict[str, Any]:
        """Export verified compact ANALYSIS + FULL ZIPs with external FINAL_STATUS.json & DELIVERY_RECEIPT.json."""
        analysis_files: dict[str, Path] = {}
        full_files: dict[str, Path] = {}
        for fname in (
            "X2_DESIGN_REGISTRY.json",
            "PREPARED_MANIFEST.json",
            "PENDING_QUEUE.json",
            "SESSION_ADMISSION_PLAN.json",
            "CAMPAIGN_STATUS.json",
            "VERIFIED_RESUME_RECEIPT.json",
            "ADMISSION_RECEIPT.json",
        ):
            p = self.work_dir / fname
            if p.is_file():
                analysis_files[fname] = p
                full_files[fname] = p
        x2_bundle_cache: dict[str, bool] = {}
        for p in sorted(self.work_dir.rglob("*")):
            if not p.is_file():
                continue
            rel = p.relative_to(self.work_dir).as_posix()
            parts = rel.split("/")
            if rel.startswith(("imported_inputs/", "wave_")) or any(
                part.startswith(".staging-") or part == "quarantine" for part in parts
            ):
                continue
            if rel.startswith("specs/"):
                full_files[rel] = p
                continue
            if not rel.startswith("jobs/") or len(parts) < 2:
                continue
            jid = parts[1]
            if jid not in x2_bundle_cache:
                x2_bundle_cache[jid] = (self.work_dir / "jobs" / jid / "result_bundle" / "CURRENT.json").is_file()
            if x2_bundle_cache[jid] and (
                rel.startswith(f"jobs/{jid}/result_bundle/generations/")
                or rel.startswith(f"jobs/{jid}/checkpoints/")
            ):
                continue
            if p.suffix in (".json", ".jsonl", ".md") and not (
                rel.startswith(f"jobs/{jid}/checkpoints/")
                or rel.startswith(f"jobs/{jid}/result_bundle/generations/")
                or rel.startswith(f"jobs/{jid}/shared_backbones/")
            ):
                analysis_files[rel] = p
            full_files[rel] = p

        return export_session_archives(
            self.work_dir,
            output_dir,
            prefix=prefix,
            analysis_files=analysis_files,
            full_files=full_files,
            status=status,
            source_commit=self.source_commit,
            protocol_sha256=self.registry["registry_sha256"],
            include_v6_manifest=True,
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Elendril V8-X2 Palantiri Campaign CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_prep = sub.add_parser("prepare")
    p_prep.add_argument("--source", required=True)
    p_prep.add_argument("--work-dir", required=True)
    p_prep.add_argument("--sessions", nargs="*", default=None)

    p_leg = sub.add_parser("legacy-exec", help="Execute legacy X1 worker inside isolated legacy repo cwd/PYTHONPATH.")
    p_leg.add_argument("--legacy-repo", required=True)
    p_leg.add_argument("--expected-sha", required=True)
    p_leg.add_argument("worker_args", nargs=argparse.REMAINDER)

    p_run = sub.add_parser("run")
    p_run.add_argument("--source", required=True)
    p_run.add_argument("--work-dir", required=True)
    p_run.add_argument("--output-dir", required=True)
    p_run.add_argument("--legacy-x1-dir", default=None)
    p_run.add_argument("--sessions", nargs="*", default=None)
    p_run.add_argument("--devices", default="0,1")
    p_run.add_argument("--outer-minutes", type=float, default=DEFAULT_OUTER_MAX_MINUTES)
    p_run.add_argument("--session-minutes", type=float, default=DEFAULT_SESSION_MAX_MINUTES)
    p_run.add_argument("--measured-sec-per-update", type=float, default=0.75)
    p_run.add_argument("--resume", action="store_true")
    p_run.add_argument("--canary", action="store_true")
    p_run.add_argument("--owner-authorized", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "prepare":
        camp = ElendrilX2Campaign(source_commit=args.source, work_dir=args.work_dir)
        res = camp.prepare(args.sessions)
        print(json.dumps(res, indent=2, sort_keys=True))
        return 0

    if args.command == "legacy-exec":
        return _run_legacy_exec(
            legacy_repo_dir=args.legacy_repo,
            expected_sha=args.expected_sha,
            worker_args=args.worker_args,
        )

    if args.command == "run":
        if not args.owner_authorized:
            print("NOT LAUNCHED: pass --owner-authorized to execute Elendril V8-X2.")
            return 2
        camp = ElendrilX2Campaign(
            source_commit=args.source,
            work_dir=args.work_dir,
            legacy_x1_repo_dir=args.legacy_x1_dir,
            outer_max_minutes=args.outer_minutes,
            session_max_minutes=args.session_minutes,
        )
        camp.prepare(args.sessions)
        plan = camp.admit_slice(
            args.sessions,
            measured_seconds_per_update=args.measured_sec_per_update,
            parallel_workers=max(1, len([d for d in args.devices.split(",") if d.strip()])),
        )
        admitted_set = set(plan["admitted_job_ids"])
        rows = [r for r in camp.select_session_rows(args.sessions) if r["job_id"] in admitted_set]
        dev_list = [f"cuda:{d.strip()}" if not args.canary else "cpu" for d in args.devices.split(",") if d.strip()]
        summary = camp.run_admitted_waves(
            camp.to_session_jobs(rows),
            devices=dev_list,
            resume=bool(args.resume),
            canary=bool(args.canary),
        )
        delivery = camp.export_archives(
            args.output_dir,
            status="COMPLETE" if not summary["unfinished_job_ids"] and not plan["deferred_pending_job_ids"] else "PARTIAL",
        )
        print(json.dumps({"summary": summary, "delivery": delivery, "admission_plan": plan}, indent=2, sort_keys=True))
        return 0 if not summary["failed_job_ids"] else 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
