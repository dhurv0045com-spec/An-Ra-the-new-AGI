"""Coordinated Palantiri campaign orchestrator, dual-source routing, verified resume & export for Elendril V8-X2."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Callable, Collection, Mapping, Sequence
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
    X1_LEGACY_ACCEPTED_IMPORT_SHAS,
    X1_LEGACY_COMPAT_DISCLOSURE,
    X1_LEGACY_COMPAT_SHA,
    X1_LEGACY_ORIGINAL_SHA,
    X1_LEGACY_SOURCE_SHA,
    X2_BASE_JOB_EQUIVALENT_CEILING,
    _infer_execution_route,
    _legacy_scientific_identity,
    build_x2_design_registry,
    build_x2_obligation_ledger,
    evaluate_confirmation_trigger,
    evaluate_x2_completion_audit,
    is_paired_cartridge_row,
    validate_x2_registry,
)

DEFAULT_OUTER_MAX_MINUTES = 480.0
DEFAULT_SESSION_MAX_MINUTES = 330.0
DEFAULT_EXPORT_RESERVE_MINUTES = 45.0
DEFAULT_MIN_DISK_BYTES = 2 * 1024**3
X3_FROZEN_PARENT_IDS: tuple[str, ...] = tuple(
    f"X1-{sess}-S20-{arch}-standard-s{seed}"
    for sess, arch in (("A0", "P"), ("A0", "R"), ("A1", "F-cap"), ("A2", "R-noedge"))
    for seed in (7, 19, 41)
)


def _all_registry_rows(registry: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        *list(registry.get("jobs", ())),
        *list(registry.get("separate_ledgers", {}).get("t14_routing_prep", {}).get("jobs", ())),
        *list(registry.get("separate_ledgers", {}).get("t13_transfer_parents", {}).get("jobs", ())),
    ]


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


def _dataset_split_for_name(ds_name: str) -> str:
    low = str(ds_name).lower()
    if "confirmation" in low:
        return "confirmation"
    if "t14" in low or "frozen" in low:
        return "dev"
    return "train"


def materialize_x2_job_spec(
    job: Mapping[str, Any],
    spec_path: Path | str,
    *,
    source_commit: str,
    canary: bool = False,
) -> dict[str, Any]:
    """Materialize an immutable `anra-planthir-spec-v1` JSON for a paired `elendril_x2` scratch job."""
    sp = Path(spec_path)
    sp.parent.mkdir(parents=True, exist_ok=True)
    arch = str(job["architecture"])
    obj = str(job.get("objective", "standard"))
    explicit_route = job.get("execution_route")
    inferred_route = str(
        explicit_route
        or _infer_execution_route(
            backend_id=str(job.get("backend_id", "elendril_x2")),
            block=str(job.get("block", "")),
            objective=obj,
            resource_class=str(job.get("resource_class", "")),
            parent_job_id=job.get("parent_job_id"),
            status=str(job.get("status", "PENDING")),
        )
    )
    if obj == "frozen_probe" or inferred_route == "x2_frozen_probe" or str(job.get("resource_class", "")) == "frozen_inference":
        raise ValueError("FROZEN_PROBE_MUST_USE_FROZEN_ROUTE")
    if inferred_route != "x2_scratch_train":
        raise ValueError(f"NON_SCRATCH_ROUTE_MUST_USE_LINEAGE_RUNNER:{inferred_route}")
    raw_ctrl = job.get("control_architecture")
    if not raw_ctrl:
        raise ValueError(f"PAIRED_CARTRIDGE_REQUIRES_REGISTERED_CONTROL_ARM:{job.get('job_id')}")
    ctrl = str(raw_ctrl)
    updates = 2 if canary else int(job.get("updates", 2368))
    milestones = [1, 2] if canary else list(job.get("milestones", [0, 592, 1184, 1472, 2368]))
    model_cfg = (
        {"width": 16, "layers": 1, "query_heads": 2, "kv_heads": 1, "head_dimension": 8, "ffn_width": 32, "context_length": 640, "mem_dim": 16}
        if canary
        else {"s20_backbone": True, "context_length": 640, "mem_dim": 64}
    )
    seed = int(job.get("init_seed", 7))
    order_seed = int(job.get("order_seed", seed))
    ds_name = str(job.get("dataset", "x2_canary_fixture" if canary else "x2_production_stream"))

    spec = {
        "schema": "anra-planthir-spec-v1",
        "campaign_id": str(job["job_id"]),
        "program": "ELENDRIL-V8-X2",
        "source_commit": str(source_commit),
        "backend_id": "elendril_x2",
        "dataset": {
            "dataset_name": ds_name,
            "split": _dataset_split_for_name(ds_name),
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


def _ledger_parent_worker_map(registry: Mapping[str, Any]) -> dict[str, str]:
    ledger = build_x2_obligation_ledger(registry)
    return {str(e["logical_id"]): str(e["worker_unit_id"]) for e in ledger.get("entries", ())}


def resolve_parent_checkpoint_path(
    registry: Mapping[str, Any],
    parent_job_id: str | None,
    *,
    work_dir: Path | str,
    input_roots: Sequence[Path | str] = (),
) -> Path | None:
    """Resolve an immutable parent checkpoint dir from work_dir/jobs or mounted input dirs."""
    if not parent_job_id:
        return None
    wdir = Path(work_dir)
    pmap = _ledger_parent_worker_map(registry)
    candidates_ids = [str(parent_job_id)]
    wuid = pmap.get(str(parent_job_id))
    if wuid and wuid not in candidates_ids:
        candidates_ids.append(wuid)

    search_bases = [wdir / "jobs", wdir / "parent_checkpoints", wdir / "imported_inputs", *[Path(p) for p in input_roots]]

    def _has_ckpt(d: Path) -> bool:
        if not d.is_dir():
            return False
        if (d / "LATEST").is_file():
            return True
        ck = d / "checkpoints"
        if ck.is_dir() and any(p.is_file() for p in ck.rglob("LATEST")):
            return True
        if any(p.is_file() for p in d.rglob("LATEST")):
            return True
        if any(p.is_file() for p in d.glob("*.bin")):
            return True
        return False

    for cid in candidates_ids:
        for base in search_bases:
            direct = base / cid
            if _has_ckpt(direct):
                return direct
            if base.is_dir():
                for sub in sorted(base.rglob(cid)):
                    if _has_ckpt(sub):
                        return sub
    return None


def compute_protected_parent_ids(
    registry: Mapping[str, Any],
    completed_ids: Collection[str] = (),
) -> dict[str, tuple[str, ...]]:
    """Return protected parent job/worker IDs needed by incomplete descendants or frozen X3 evaluation."""
    done = set(str(x) for x in completed_ids)
    wmap = _ledger_parent_worker_map(registry)
    protected: set[str] = set()
    retain_opt: set[str] = set()

    for r in _all_registry_rows(registry):
        lid = str(r["job_id"])
        pid = r.get("parent_job_id")
        if pid and lid not in done:
            pid_s = str(pid)
            protected.add(pid_s)
            if pid_s in wmap:
                protected.add(wmap[pid_s])
            if str(r.get("execution_route", "")) == "x2_horizon_continuation" or str(r.get("block", "")) == "V_HORIZON_4736":
                retain_opt.add(pid_s)
                if pid_s in wmap:
                    retain_opt.add(wmap[pid_s])

    for x3_pid in X3_FROZEN_PARENT_IDS:
        protected.add(x3_pid)
        if x3_pid in wmap:
            protected.add(wmap[x3_pid])

    return {
        "protected_job_ids": tuple(sorted(protected)),
        "retain_optimizer_for_ids": tuple(sorted(retain_opt)),
    }


def build_lineage_runner_command(
    job: Mapping[str, Any],
    *,
    work_root: Path | str,
    source_commit: str,
    parent_checkpoint: Path | str | None = None,
    missing_original_receipt: Path | str | None = None,
    reproduced_checkpoint: Path | str | None = None,
    device: str = "cpu",
    canary: bool = False,
) -> dict[str, Any]:
    """Build exact explicit CLI command for `frozen-probe`, `task-adapt`, `horizon-continue`, or `scratch-single`."""
    route = str(
        job.get("execution_route")
        or _infer_execution_route(
            backend_id=str(job.get("backend_id", "elendril_x2")),
            block=str(job.get("block", "")),
            objective=str(job.get("objective", "standard")),
            resource_class=str(job.get("resource_class", "full_job")),
            parent_job_id=job.get("parent_job_id"),
            status=str(job.get("status", "PENDING")),
        )
    )
    wroot = str(Path(work_root).resolve())
    arch = str(job["architecture"])
    seed = int(job.get("init_seed", 7))
    oseed = int(job.get("order_seed", seed))
    updates = 2 if canary else int(job.get("updates", 2368))
    milestones = "1,2" if canary else ",".join(str(int(m)) for m in job.get("milestones", (0, 592, 1184, 1472, 2368)))

    if route == "x2_frozen_probe":
        if not parent_checkpoint and not (missing_original_receipt and reproduced_checkpoint):
            raise FileNotFoundError(f"MISSING_PARENT_CHECKPOINT: frozen probe {job['job_id']} requires parent={job.get('parent_job_id')}")
        cmd: list[str] = [
            sys.executable,
            "-m",
            "experiments.x2.lineage_runner",
            "frozen-probe",
            "--job-id",
            str(job["job_id"]),
            "--architecture",
            arch,
            "--work-root",
            wroot,
            "--source",
            str(source_commit),
            "--seed",
            str(seed),
            "--dataset",
            str(job.get("dataset", "x1-a0-frozen-dev2304-depth640-v1")),
            "--device",
            str(device),
        ]
        if job.get("parent_job_id"):
            cmd.extend(["--parent-job-id", str(job["parent_job_id"])])
        if parent_checkpoint:
            cmd.extend(["--parent-checkpoint", str(Path(parent_checkpoint).resolve())])
        if missing_original_receipt:
            cmd.extend(["--missing-original-receipt", str(Path(missing_original_receipt).resolve())])
        if reproduced_checkpoint:
            cmd.extend(["--reproduced-checkpoint", str(Path(reproduced_checkpoint).resolve())])
    elif route == "x2_task_adaptation":
        if not parent_checkpoint:
            raise FileNotFoundError(f"MISSING_PARENT_CHECKPOINT: task adaptation {job['job_id']} requires parent={job.get('parent_job_id')}")
        cmd = [
            sys.executable,
            "-m",
            "experiments.x2.lineage_runner",
            "task-adapt",
            "--job-id",
            str(job["job_id"]),
            "--architecture",
            arch,
            "--parent-checkpoint",
            str(Path(parent_checkpoint).resolve()),
            "--dataset",
            str(job["dataset"]),
            "--objective",
            str(job.get("objective", "T13_ADAPT")),
            "--work-root",
            wroot,
            "--source",
            str(source_commit),
            "--seed",
            str(seed),
            "--order-seed",
            str(oseed),
            "--updates",
            str(updates),
            "--milestones",
            milestones,
            "--device",
            str(device),
        ]
        if job.get("parent_job_id"):
            cmd.extend(["--parent-job-id", str(job["parent_job_id"])])
    elif route == "x2_horizon_continuation":
        if not parent_checkpoint:
            raise FileNotFoundError(f"MISSING_PARENT_CHECKPOINT: horizon job {job['job_id']} requires parent={job.get('parent_job_id')}")
        ms_list = [int(m) for m in job.get("milestones", (2368, 3552, 4736))]
        p_u = 2 if canary else int(ms_list[0] if ms_list else 2368)
        m_u = 3 if canary else int(ms_list[1] if len(ms_list) > 1 else 3552)
        e_u = 4 if canary else int(ms_list[-1] if len(ms_list) > 1 else p_u + updates)
        cmd = [
            sys.executable,
            "-m",
            "experiments.x2.lineage_runner",
            "horizon-continue",
            "--job-id",
            str(job["job_id"]),
            "--architecture",
            arch,
            "--parent-checkpoint",
            str(Path(parent_checkpoint).resolve()),
            "--dataset",
            str(job.get("dataset", "x2-d1h-selective-edit-v1")),
            "--objective",
            str(job.get("objective", "standard")),
            "--work-root",
            wroot,
            "--source",
            str(source_commit),
            "--seed",
            str(seed),
            "--order-seed",
            str(oseed),
            "--parent-updates",
            str(p_u),
            "--updates",
            str(updates),
            "--mid-milestone",
            str(m_u),
            "--end-milestone",
            str(e_u),
            "--device",
            str(device),
        ]
        if job.get("parent_job_id"):
            cmd.extend(["--parent-job-id", str(job["parent_job_id"])])
    elif route == "x2_scratch_train":
        cmd = [
            sys.executable,
            "-m",
            "experiments.x2.lineage_runner",
            "scratch-single",
            "--job-id",
            str(job["job_id"]),
            "--architecture",
            arch,
            "--dataset",
            str(job["dataset"]),
            "--objective",
            str(job.get("objective", "standard")),
            "--work-root",
            wroot,
            "--source",
            str(source_commit),
            "--seed",
            str(seed),
            "--order-seed",
            str(oseed),
            "--updates",
            str(updates),
            "--milestones",
            milestones,
            "--device",
            str(device),
        ]
    else:
        raise ValueError(f"UNSUPPORTED_LINEAGE_RUNNER_ROUTE:{route}")

    if canary:
        cmd.append("--canary")
    return {
        "lane_kind": f"elendril_x2_lineage_{route}",
        "source_commit": str(source_commit),
        "env_overrides": {"PLANTHIR_BACKEND_MODULES": "experiments.elendril_x2"},
        "command": tuple(cmd),
    }


def build_verified_resume_set(
    *,
    registry: Mapping[str, Any],
    work_dir: Path | str,
    imported_receipts: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Rebuild verified skip list strictly from retrieved hashes/steps/specs, NEVER blindly trusting COMPLETED_JOBS.json (§0.4.4)."""
    reg_by_id = {str(r["job_id"]): r for r in _all_registry_rows(registry)}
    verified_completed: dict[str, dict[str, Any]] = {}
    compatibility_mapped_imports: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for rec in imported_receipts:
        jid = str(rec.get("job_id", ""))
        reg_row = reg_by_id.get(jid)
        if reg_row is None:
            rejected.append({"job_id": jid, "reason": "UNKNOWN_JOB_ID"})
            continue
        if rec.get("job_id_dir_mismatch"):
            rejected.append({"job_id": jid, "reason": "JOB_ID_PATH_MISMATCH"})
            continue
        rec_src = str(rec.get("source_commit", ""))
        exp_src = str(reg_row["source_commit"])
        is_legacy_compat = bool(
            reg_row.get("backend_id") == "x1_s20_legacy"
            and rec_src in X1_LEGACY_ACCEPTED_IMPORT_SHAS
            and exp_src in X1_LEGACY_ACCEPTED_IMPORT_SHAS
            and rec_src != exp_src
        )
        if rec_src != exp_src and not is_legacy_compat:
            rejected.append(
                {
                    "job_id": jid,
                    "reason": "SOURCE_COMMIT_MISMATCH",
                    "expected": exp_src,
                    "observed": rec.get("source_commit"),
                }
            )
            continue

        expected_spec = (
            str(reg_row.get("extra", {}).get("original_legacy_spec_sha256") or reg_row["spec_sha256"])
            if is_legacy_compat and rec_src == X1_LEGACY_ORIGINAL_SHA
            else str(reg_row["spec_sha256"])
        )
        if rec.get("spec_sha256") and str(rec["spec_sha256"]) != expected_spec:
            rejected.append(
                {
                    "job_id": jid,
                    "reason": "SWAPPED_SPEC_MISMATCH",
                    "expected": expected_spec,
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
        expected_sci_id = str(reg_row.get("extra", {}).get("legacy_scientific_identity_sha256", ""))
        if expected_sci_id and rec.get("legacy_scientific_identity_sha256") and str(rec["legacy_scientific_identity_sha256"]) != expected_sci_id:
            rejected.append(
                {
                    "job_id": jid,
                    "reason": "LEGACY_SCIENTIFIC_IDENTITY_MISMATCH",
                    "expected": expected_sci_id,
                    "observed": rec.get("legacy_scientific_identity_sha256"),
                }
            )
            continue

        route = str(reg_row.get("execution_route") or "")
        min_updates = int(reg_row["updates"])
        if route == "x2_frozen_probe" or min_updates == 0:
            opt_steps = rec.get("optimizer_steps_executed", -1)
            if int(opt_steps) != 0:
                rejected.append(
                    {
                        "job_id": jid,
                        "reason": "FROZEN_PROBE_EXECUTED_OPTIMIZER_STEPS",
                        "expected": 0,
                        "observed": opt_steps,
                    }
                )
                continue
        else:
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
            "source_commit": exp_src,
            "original_source_commit": rec_src,
            "imported_via_compatibility_map": is_legacy_compat,
            "spec_sha256": expected_spec,
            "completed_updates": int(rec.get("completed_updates", 0)),
            "optimizer_steps_executed": int(rec.get("optimizer_steps_executed", rec.get("completed_updates", 0))),
            "eval_rows": int(rec["eval_rows"]),
            "archive_sha256": actual_sha,
            "evidence_source": rec.get("evidence_source"),
            "extracted_job_dir": rec.get("extracted_job_dir"),
        }
        if is_legacy_compat:
            compatibility_mapped_imports.append(
                {
                    "job_id": jid,
                    "original_source_commit": rec_src,
                    "repaired_source_commit": exp_src,
                    "original_legacy_spec_sha256": expected_spec,
                    "scientific_identity_sha256": expected_sci_id,
                }
            )

    # Write fresh verified skip list for CampaignOrchestrator
    wdir = Path(work_dir)
    wdir.mkdir(parents=True, exist_ok=True)
    completed_ids = sorted(verified_completed)
    _atomic_write_json(wdir / "COMPLETED_JOBS.json", {"completed_jobs": completed_ids, "verified_by": "elendril_x2_resume_audit"})
    receipt = {
        "schema": "elendril-x2-verified-resume-v1",
        "verified_completed_ids": completed_ids,
        "verified_count": len(completed_ids),
        "verified_receipts_by_id": verified_completed,
        "compatibility_mapped_imports": compatibility_mapped_imports,
        "legacy_compatibility_disclosure": dict(X1_LEGACY_COMPAT_DISCLOSURE),
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


def _parse_int_milestones(raw: Any) -> list[int]:
    out: list[int] = []
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return out
    for item in raw:
        s = str(item).strip()
        if s.startswith("u"):
            s = s[1:]
        try:
            out.append(int(s))
        except ValueError:
            return []
    return out


def discover_kaggle_input_receipts(
    input_root: Path | str,
    extract_dir: Path | str,
    registry: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Scan mounted /kaggle/input prior archives & unpacked job folders for verified resume receipts (§0.4.4)."""
    in_root = Path(input_root)
    if not in_root.is_dir():
        return []

    reg_by_id = {str(r["job_id"]): r for r in _all_registry_rows(registry)}
    ext_root = Path(extract_dir)
    ext_root.mkdir(parents=True, exist_ok=True)

    receipts_by_job: dict[str, dict[str, Any]] = {}
    scanned_roots: list[tuple[Path, Path, str | None]] = []  # (search_root, backing_archive_path, sidecar_expected_sha)

    # Prefer lightweight ANALYSIS archives first; only inspect FULL if directory lacks ANALYSIS
    analysis_zips = sorted(in_root.rglob("*ELENDRIL*ANALYSIS*.zip"))
    analysis_dirs = {z.parent for z in analysis_zips}
    full_zips = [z for z in sorted(in_root.rglob("*ELENDRIL*FULL*.zip")) if z.parent not in analysis_dirs]
    parent_zips = sorted(in_root.rglob("*ELENDRIL*PARENT_CHECKPOINTS*.zip"))

    for idx, zpath in enumerate([*analysis_zips, *full_zips, *parent_zips]):
        sidecar_sha: str | None = None
        sidecar = zpath.with_name(zpath.name + ".sha256")
        if sidecar.is_file():
            tokens = sidecar.read_text(encoding="utf-8", errors="replace").strip().split()
            if tokens:
                sidecar_sha = tokens[0].lower()
        dest = ext_root / f"arc_{idx:03d}_{zpath.stem}"
        try:
            safe_extract_planthir_archive(zpath, dest)
            scanned_roots.append((dest, zpath, sidecar_sha))
        except Exception:
            continue

    scanned_roots.append((in_root, in_root, None))

    for sroot, backing_zip, sidecar_sha in scanned_roots:
        arc_spec_sha_by_id: dict[str, str] = {}
        for reg_p in sorted(sroot.rglob("X2_DESIGN_REGISTRY.json")):
            try:
                arc_reg = json.loads(reg_p.read_text(encoding="utf-8"))
                for row in _all_registry_rows(arc_reg):
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
            rec_sha = sidecar_sha if sidecar_sha is not None else _sha256_file(arc_file)

            has_full_sci = all(k in doc for k in ("architecture", "dataset", "objective", "init_seed", "order_seed"))
            sci_id = None
            if has_full_sci:
                ms_list = _parse_int_milestones(doc.get("milestones_recorded", reg_row["milestones"]))
                if not ms_list:
                    ms_list = list(reg_row["milestones"])
                sci_id = _legacy_scientific_identity(
                    job_id=jid,
                    architecture=str(doc["architecture"]),
                    dataset=str(doc["dataset"]),
                    objective=str(doc["objective"]),
                    init_seed=int(doc["init_seed"]),
                    order_seed=int(doc["order_seed"]),
                    updates=int(doc.get("target_updates", doc.get("completed_updates", 0))),
                    milestones=ms_list,
                    legacy_model_spec_sha256=str(doc.get("model_spec_sha256", "")),
                )

            rec: dict[str, Any] = {
                "job_id": jid,
                "job_id_dir_mismatch": jpath.parent.name != jid,
                "source_commit": str(doc.get("source_commit", "")),
                "model_spec_sha256": str(doc.get("model_spec_sha256", "")),
                "completed_updates": int(doc.get("completed_updates", 0)),
                "eval_rows": int(eval_rows),
                "archive_path": str(arc_file),
                "archive_sha256": rec_sha,
                "extracted_job_dir": str(jpath.parent),
            }
            if jid in arc_spec_sha_by_id:
                rec["spec_sha256"] = arc_spec_sha_by_id[jid]
            if sci_id is not None:
                rec["legacy_scientific_identity_sha256"] = sci_id
            prev = receipts_by_job.get(jid)
            if prev is None or rec["completed_updates"] >= prev["completed_updates"]:
                receipts_by_job[jid] = rec

        # 2. Primary X2 Palantiri FINAL_RECEIPT.json (+ FROZEN_PROBE_RECEIPT / ARM_TO_LOGICAL_ROW_MAP)
        for rpath in sorted(sroot.rglob("FINAL_RECEIPT.json")):
            try:
                doc = json.loads(rpath.read_text(encoding="utf-8"))
            except Exception:
                continue
            if doc.get("status") != "COMPLETE":
                continue
            jdir = rpath.parent
            jid = str(doc.get("campaign_id", jdir.name))
            arc_file = backing_zip if backing_zip.is_file() else rpath
            rec_sha = sidecar_sha if sidecar_sha is not None else _sha256_file(arc_file)

            frozen_p = jdir / "FROZEN_PROBE_RECEIPT.json"
            if frozen_p.is_file() and jid in reg_by_id:
                try:
                    fdoc = json.loads(frozen_p.read_text(encoding="utf-8"))
                    frec = {
                        "job_id": jid,
                        "source_commit": str(doc.get("source_commit", "")),
                        "spec_sha256": arc_spec_sha_by_id.get(jid, str(reg_by_id[jid]["spec_sha256"])),
                        "completed_updates": 0,
                        "optimizer_steps_executed": int(fdoc.get("optimizer_steps_executed", -1)),
                        "eval_rows": int(fdoc.get("eval_rows", 0)),
                        "evidence_source": fdoc.get("evidence_source"),
                        "archive_path": str(arc_file),
                        "archive_sha256": rec_sha,
                        "extracted_job_dir": str(jdir),
                    }
                    receipts_by_job[jid] = frec
                    continue
                except Exception:
                    pass

            arm_map_p = jdir / "ARM_TO_LOGICAL_ROW_MAP.json"
            arm_results_path = jdir / "ARM_RESULTS.json"
            if arm_map_p.is_file() and arm_results_path.is_file():
                try:
                    mdoc = json.loads(arm_map_p.read_text(encoding="utf-8"))
                    arm_doc = json.loads(arm_results_path.read_text(encoding="utf-8"))
                    mappings = list(mdoc.get("mappings", ()))
                    if mappings and all(
                        m.get("logical_id") in reg_by_id
                        and m.get("arm_id") in arm_doc
                        and int(arm_doc[m["arm_id"]].get("process_funnel", {}).get("episodes", 0) or 0) > 0
                        for m in mappings
                    ):
                        for m in mappings:
                            lid = str(m["logical_id"])
                            ainfo = arm_doc[m["arm_id"]]
                            max_u = max((int(ms.get("global_update", 0)) for ms in ainfo.get("milestones", ())), default=0)
                            ep_c = int(ainfo.get("process_funnel", {}).get("episodes", 0) or 0)
                            receipts_by_job[lid] = {
                                "job_id": lid,
                                "source_commit": str(doc.get("source_commit", "")),
                                "spec_sha256": arc_spec_sha_by_id.get(lid, str(reg_by_id[lid]["spec_sha256"])),
                                "completed_updates": int(max_u),
                                "eval_rows": int(ep_c),
                                "archive_path": str(arc_file),
                                "archive_sha256": rec_sha,
                                "extracted_job_dir": str(jdir),
                            }
                        continue
                except Exception:
                    pass

            if jid not in reg_by_id:
                continue
            reg_row = reg_by_id[jid]
            completed_updates = 0
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
            if completed_updates <= 0 and "completed_updates" in doc:
                completed_updates = int(doc.get("completed_updates", 0))
            eval_rows = _count_eval_rows_x2(jdir) or int(doc.get("eval_rows", 0))
            rec = {
                "job_id": jid,
                "source_commit": str(doc.get("source_commit", "")),
                "spec_sha256": arc_spec_sha_by_id.get(jid, str(reg_row["spec_sha256"])),
                "completed_updates": int(completed_updates),
                "eval_rows": int(eval_rows),
                "archive_path": str(arc_file),
                "archive_sha256": rec_sha,
                "extracted_job_dir": str(jdir),
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
            rec_sha = sidecar_sha if sidecar_sha is not None else _sha256_file(arc_file)
            rec = {
                "job_id": jid,
                "source_commit": str(doc.get("source_commit", "")),
                "spec_sha256": arc_spec_sha_by_id.get(jid, str(reg_row["spec_sha256"])),
                "completed_updates": int(doc["completed_updates"]),
                "eval_rows": int(eval_rows),
                "archive_path": str(arc_file),
                "archive_sha256": rec_sha,
                "extracted_job_dir": str(rpath.parent),
            }
            receipts_by_job[jid] = rec

    return [receipts_by_job[k] for k in sorted(receipts_by_job)]


def prune_completed_job_heavy_artifacts(
    job_work_dir: Path | str,
    *,
    protected_job_ids: Collection[str] = (),
    retain_optimizer_for_ids: Collection[str] = (),
) -> dict[str, Any]:
    """Prune heavy weight/optimizer binaries for VERIFIED COMPLETE jobs unless protected for descendants/X3."""
    jroot = Path(job_work_dir)
    if not jroot.is_dir():
        return {"pruned": False, "protected": False, "files_removed": 0, "bytes_freed": 0}
    if jroot.name in set(protected_job_ids) or jroot.name in set(retain_optimizer_for_ids):
        return {
            "pruned": False,
            "protected": True,
            "reason": "PROTECTED_PARENT_CHECKPOINT",
            "files_removed": 0,
            "bytes_freed": 0,
        }

    x2_complete = False
    final_rec = jroot / "FINAL_RECEIPT.json"
    if final_rec.is_file():
        try:
            doc = json.loads(final_rec.read_text(encoding="utf-8"))
            x2_complete = doc.get("status") == "COMPLETE"
        except Exception:
            x2_complete = False

    x1_complete = False
    jobs_sub = jroot / "jobs"
    if jobs_sub.is_dir():
        for jpath in jobs_sub.glob("*/job.json"):
            try:
                doc = json.loads(jpath.read_text(encoding="utf-8"))
                if doc.get("status") == "COMPLETE":
                    x1_complete = True
                    break
            except Exception:
                continue

    if not x2_complete and not x1_complete:
        return {"pruned": False, "files_removed": 0, "bytes_freed": 0}

    files_removed = 0
    bytes_freed = 0

    def _rm_file(path: Path) -> None:
        nonlocal files_removed, bytes_freed
        if path.is_file():
            try:
                bytes_freed += path.stat().st_size
                path.unlink()
                files_removed += 1
            except OSError:
                pass

    def _rm_tree(dpath: Path) -> None:
        nonlocal files_removed, bytes_freed
        if dpath.is_dir():
            for child in dpath.rglob("*"):
                if child.is_file():
                    try:
                        bytes_freed += child.stat().st_size
                        files_removed += 1
                    except OSError:
                        pass
            shutil.rmtree(dpath, ignore_errors=True)

    if x2_complete:
        _rm_tree(jroot / "checkpoints")
        _rm_tree(jroot / "result_bundle" / "generations")
        _rm_file(jroot / "result_bundle" / "checkpoint_full.zip")
        _rm_file(jroot / "result_bundle" / "checkpoint_full.zip.sha256")
    if x1_complete:
        _rm_tree(jroot / "checkpoints")
        _rm_tree(jroot / "shared_backbones")
        if jobs_sub.is_dir():
            for bin_path in list(jobs_sub.rglob("*.bin")):
                _rm_file(bin_path)

    return {"pruned": True, "files_removed": files_removed, "bytes_freed": bytes_freed}


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
        include_extra_ledgers: bool = False,
    ) -> list[dict[str, Any]]:
        wanted = set(sessions) if sessions else None
        extra_rows = [
            *self.registry.get("separate_ledgers", {}).get("t14_routing_prep", {}).get("jobs", ()),
            *self.registry.get("separate_ledgers", {}).get("t13_transfer_parents", {}).get("jobs", ()),
        ]
        wants_extra = bool(
            include_extra_ledgers
            or (wanted and any(r["session"] in wanted or r["block"] in wanted for r in extra_rows))
        )
        base_rows = [*self.registry["jobs"], *(extra_rows if wants_extra else ())]
        rows = [r for r in base_rows if r["status"] != "EVIDENCED_ACTION_REPLAYED"]
        if wanted:
            rows = [r for r in rows if r["session"] in wanted or r["block"] in wanted]
        if deduplicate_paired_x2:
            # ONLY IIIb_NEW_MECHANISMS & CONF_SOURCE paired CartridgeRunner jobs bundle treatment + control in 1 worker
            covered_controls = {
                (str(r["session"]), int(r["init_seed"]), str(r["control_architecture"]))
                for r in rows
                if is_paired_cartridge_row(r)
            }
            filtered: list[dict[str, Any]] = []
            for r in rows:
                if (
                    r.get("backend_id") == "elendril_x2"
                    and r.get("execution_route") == "x2_scratch_train"
                    and r.get("block") in ("IIIb_NEW_MECHANISMS", "CONF_SOURCE")
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
            is_paired_x2 = bool(is_paired_cartridge_row(r) and int(r["updates"]) > 0)
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
                        "execution_route": str(r.get("execution_route", "")),
                        "block": str(r.get("block", "")),
                        "parent_job_id": r.get("parent_job_id"),
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

    def prepare(self, sessions: Sequence[str] | None = None, *, include_extra_ledgers: bool = False) -> dict[str, Any]:
        _atomic_write_json(self.work_dir / "X2_DESIGN_REGISTRY.json", self.registry)
        ledger = build_x2_obligation_ledger(self.registry)
        _atomic_write_json(self.work_dir / "X2_OBLIGATION_LEDGER.json", ledger)
        rows = self.select_session_rows(sessions, include_extra_ledgers=include_extra_ledgers)
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
        include_extra_ledgers: bool = False,
    ) -> dict[str, Any]:
        """Pack whole jobs fitting the governing dual-clock & disk budget without deleting pending rows (§0.4.3)."""
        done_set = set(completed_ids)
        all_rows = self.select_session_rows(sessions, include_extra_ledgers=include_extra_ledgers)
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
        prune_completed_heavy_artifacts: bool = False,
        on_wave_complete: Callable[[int, Mapping[str, Any]], None] | None = None,
        frozen_probe_overrides: Mapping[str, Mapping[str, Any]] | None = None,
        input_roots: Sequence[Path | str] = (),
    ) -> dict[str, Any]:
        specs_dir = self.work_dir / "specs"
        specs_dir.mkdir(parents=True, exist_ok=True)
        all_rows = _all_registry_rows(self.registry)
        reg_by_id = {str(r["job_id"]): r for r in all_rows}
        ledger = build_x2_obligation_ledger(self.registry)
        worker_to_entries: dict[str, list[Mapping[str, Any]]] = {}
        for ent in ledger.get("entries", ()):
            worker_to_entries.setdefault(str(ent["worker_unit_id"]), []).append(ent)

        done_file = self.work_dir / "COMPLETED_JOBS.json"
        completed: set[str] = set()
        if resume and done_file.is_file():
            completed = set(json.loads(done_file.read_text(encoding="utf-8")).get("completed_jobs", []))

        def _maybe_write_arm_map(wid: str) -> None:
            jdir = self.work_dir / "jobs" / wid
            arm_p = jdir / "ARM_RESULTS.json"
            map_p = jdir / "ARM_TO_LOGICAL_ROW_MAP.json"
            entries = worker_to_entries.get(wid, ())
            row = reg_by_id.get(wid)
            if not row or not is_paired_cartridge_row(row) or not arm_p.is_file():
                return
            try:
                arm_doc = json.loads(arm_p.read_text(encoding="utf-8"))
            except Exception:
                return
            seed = int(row["init_seed"])
            ctrl_arch = str(row["control_architecture"])
            trt_arch = str(row["architecture"])
            ctrl_arm_id = f"CTRL_{ctrl_arch}_s{seed}".replace("-", "_")
            trt_arm_id = f"TRT_{trt_arch}_s{seed}".replace("-", "_")
            mappings = []
            for ent in entries:
                lid = str(ent["logical_id"])
                aid = trt_arm_id if ent.get("arm_role") == "treatment" else ctrl_arm_id
                if aid not in arm_doc:
                    for cand_id in sorted(arm_doc):
                        if (ent.get("arm_role") == "treatment" and cand_id.startswith("TRT")) or (
                            ent.get("arm_role") == "control" and cand_id.startswith("CTRL")
                        ):
                            aid = cand_id
                            break
                mappings.append({"logical_id": lid, "arm_id": aid, "arm_role": ent.get("arm_role")})
            _atomic_write_json(map_p, {"schema": "elendril-x2-arm-logical-map-v1", "worker_unit_id": wid, "mappings": mappings})

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
            route = str(
                row.get("execution_route")
                or _infer_execution_route(
                    backend_id=str(row.get("backend_id", "elendril_x2")),
                    block=str(row.get("block", "")),
                    objective=str(row.get("objective", "standard")),
                    resource_class=str(row.get("resource_class", "full_job")),
                    parent_job_id=row.get("parent_job_id"),
                    status=str(row.get("status", "PENDING")),
                )
            )
            target_dev = "cpu" if (canary or dev == "cpu") else "cuda:0"
            if route == "legacy_x1_train" or row.get("backend_id") == "x1_s20_legacy":
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
            if is_paired_cartridge_row(row):
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
                    device=target_dev,
                    resume=resume and (wroot / "checkpoints").is_dir(),
                    canary=canary,
                )
                return tuple(routed["command"]), worker_env

            f_ovr = dict((frozen_probe_overrides or {}).get(job.job_id, ()))
            parent_ckpt = f_ovr.get("parent_checkpoint")
            if not parent_ckpt and row.get("parent_job_id"):
                parent_ckpt = resolve_parent_checkpoint_path(
                    self.registry, str(row["parent_job_id"]), work_dir=self.work_dir, input_roots=input_roots
                )
            routed = build_lineage_runner_command(
                row,
                work_root=wroot,
                source_commit=self.source_commit,
                parent_checkpoint=parent_ckpt,
                missing_original_receipt=f_ovr.get("missing_original_receipt"),
                reproduced_checkpoint=f_ovr.get("reproduced_checkpoint"),
                device=target_dev,
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
            for wid in rec["succeeded_workers"]:
                _maybe_write_arm_map(wid)
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

            if prune_completed_heavy_artifacts and rec.get("succeeded_workers"):
                prot = compute_protected_parent_ids(self.registry, completed_ids=completed)
                wave_freed = 0
                for wid in rec["succeeded_workers"]:
                    p_info = prune_completed_job_heavy_artifacts(
                        self.work_dir / "jobs" / wid,
                        protected_job_ids=prot["protected_job_ids"],
                        retain_optimizer_for_ids=prot["retain_optimizer_for_ids"],
                    )
                    wave_freed += int(p_info.get("bytes_freed", 0))
                if wave_freed > 0:
                    print(
                        f"Pruned completed weights for wave {w_idx + 1}: freed {wave_freed / (1024 ** 2):.1f} MB",
                        flush=True,
                    )

            if on_wave_complete is not None:
                on_wave_complete(w_idx, rec)

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

    def export_parent_checkpoints(
        self,
        output_dir: Path | str,
        *,
        prefix: str = "ELENDRIL_V8_X2_PARENT_CHECKPOINTS",
        protected_job_ids: Collection[str] | None = None,
    ) -> dict[str, Any]:
        """Export retained parent checkpoints to a compact verified archive + SHA-256 sidecar for next session/X3."""
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        prot = set(
            protected_job_ids
            if protected_job_ids is not None
            else compute_protected_parent_ids(self.registry)["protected_job_ids"]
        )
        zip_path = out_dir / f"{prefix}.zip"
        tmp_zip = out_dir / f".{prefix}.zip.tmp"
        exported: list[str] = []
        file_count = 0
        with zipfile.ZipFile(tmp_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            reg_p = self.work_dir / "X2_DESIGN_REGISTRY.json"
            if reg_p.is_file():
                zf.write(reg_p, "X2_DESIGN_REGISTRY.json")
                file_count += 1
            jobs_root = self.work_dir / "jobs"
            seen_arcs: set[str] = set()
            for pid in sorted(prot):
                cand_dir = jobs_root / pid
                has_local_ckpt = cand_dir.is_dir() and (
                    (cand_dir / "LATEST").is_file()
                    or ((cand_dir / "checkpoints").is_dir() and any((cand_dir / "checkpoints").rglob("LATEST")))
                    or any(cand_dir.rglob("LATEST"))
                    or any(cand_dir.glob("*.bin"))
                )
                pdir = cand_dir if has_local_ckpt else resolve_parent_checkpoint_path(
                    self.registry, pid, work_dir=self.work_dir
                )
                if pdir is None or not pdir.is_dir():
                    continue
                has_ckpt = False
                for fp in sorted(pdir.rglob("*")):
                    if not fp.is_file():
                        continue
                    sub_rel = fp.relative_to(pdir).as_posix()
                    is_ckpt_member = (
                        "/checkpoints/" in f"/{sub_rel}"
                        or fp.suffix in (".bin", ".pt")
                        or fp.name == "LATEST"
                    )
                    if is_ckpt_member or fp.name in ("FINAL_RECEIPT.json", "ARM_RESULTS.json", "ARM_TO_LOGICAL_ROW_MAP.json", "job.json"):
                        arc_rel = f"jobs/{pid}/{sub_rel}"
                        if arc_rel not in seen_arcs:
                            zf.write(fp, arc_rel)
                            seen_arcs.add(arc_rel)
                            file_count += 1
                        if is_ckpt_member:
                            has_ckpt = True
                if has_ckpt:
                    exported.append(pid)
        os.replace(tmp_zip, zip_path)
        sha = _sha256_file(zip_path)
        (out_dir / f"{prefix}.zip.sha256").write_text(f"{sha}  {zip_path.name}\n", encoding="utf-8")

        with zipfile.ZipFile(zip_path, "r") as zf:
            bad = zf.testzip()
            if bad is not None:
                raise RuntimeError(f"PARENT_CHECKPOINT_ZIP_CORRUPT:{bad}")
            names = set(zf.namelist())
        return {
            "archive_path": str(zip_path),
            "sha256": sha,
            "exported_parent_ids": exported,
            "file_count": file_count,
            "verified_readable": True,
            "member_count": len(names),
        }

    def build_global_completion_audit(
        self,
        output_dir: Path | str | None = None,
        *,
        confirmation_decision: Mapping[str, Any] | None = None,
        case_recomputation_receipts: Mapping[str, Mapping[str, Any]] | None = None,
        session_queue_empty: bool | None = None,
    ) -> dict[str, Any]:
        """Reconcile all 192 logical rows globally across sessions and emit X2_COMPLETION_AUDIT.json & X2_REPORT.md."""
        ledger = build_x2_obligation_ledger(self.registry)
        all_rows = _all_registry_rows(self.registry)
        reg_by_id = {str(r["job_id"]): r for r in all_rows}
        completed_receipts: dict[str, dict[str, Any]] = {}
        recomp: dict[str, dict[str, Any]] = {str(k): dict(v) for k, v in (case_recomputation_receipts or {}).items()}

        res_p = self.work_dir / "VERIFIED_RESUME_RECEIPT.json"
        if res_p.is_file():
            try:
                rdoc = json.loads(res_p.read_text(encoding="utf-8"))
                for lid, info in rdoc.get("verified_receipts_by_id", {}).items():
                    if isinstance(info, Mapping):
                        completed_receipts[str(lid)] = {
                            "status": "COMPLETE",
                            "eval_rows": int(info.get("eval_rows", 0)),
                            "completed_updates": int(info.get("completed_updates", 0)),
                            "source": "imported_verified_resume",
                        }
                        if str(lid) not in recomp and int(info.get("eval_rows", 0)) > 0:
                            recomp[str(lid)] = {"verified": True, "source": "verified_resume_archive"}
            except Exception:
                pass

        jobs_dir = self.work_dir / "jobs"
        if jobs_dir.is_dir():
            for fpath in sorted(jobs_dir.glob("*/FINAL_RECEIPT.json")):
                jdir = fpath.parent
                wid = jdir.name
                try:
                    fdoc = json.loads(fpath.read_text(encoding="utf-8"))
                except Exception:
                    continue
                if fdoc.get("status") != "COMPLETE":
                    continue
                map_p = jdir / "ARM_TO_LOGICAL_ROW_MAP.json"
                if map_p.is_file():
                    try:
                        mdoc = json.loads(map_p.read_text(encoding="utf-8"))
                        arm_doc = json.loads((jdir / "ARM_RESULTS.json").read_text(encoding="utf-8"))
                        for m in mdoc.get("mappings", ()):
                            lid = str(m["logical_id"])
                            aid = str(m["arm_id"])
                            if lid in reg_by_id and aid in arm_doc:
                                ep_n = int(arm_doc[aid].get("process_funnel", {}).get("episodes", 0) or 0)
                                completed_receipts[lid] = {"status": "COMPLETE", "eval_rows": ep_n}
                                if lid not in recomp and ep_n > 0:
                                    recomp[lid] = {"verified": True, "checked_rows": ep_n, "source": "local_arm_results"}
                        continue
                    except Exception:
                        pass
                if wid in reg_by_id:
                    ep_n = _count_eval_rows_x2(jdir) or int(fdoc.get("eval_rows", 0))
                    completed_receipts[wid] = {"status": "COMPLETE", "eval_rows": ep_n}
                    if wid not in recomp and ep_n > 0:
                        recomp[wid] = {"verified": True, "checked_rows": ep_n, "source": "local_final_receipt"}

            for jpath in sorted(jobs_dir.glob("*/*/*/job.json")) + sorted(jobs_dir.glob("*/job.json")):
                try:
                    jdoc = json.loads(jpath.read_text(encoding="utf-8"))
                except Exception:
                    continue
                lid = str(jdoc.get("job_id", ""))
                if lid in reg_by_id and jdoc.get("status") == "COMPLETE":
                    ep_n = _count_eval_rows_legacy(jpath.parent)
                    completed_receipts[lid] = {"status": "COMPLETE", "eval_rows": ep_n}
                    if lid not in recomp and ep_n > 0:
                        recomp[lid] = {"verified": True, "checked_rows": ep_n, "source": "local_legacy_job"}

        if session_queue_empty is None:
            cs_p = self.work_dir / "CAMPAIGN_STATUS.json"
            if cs_p.is_file():
                try:
                    cs = json.loads(cs_p.read_text(encoding="utf-8"))
                    session_queue_empty = not bool(cs.get("unfinished_job_ids"))
                except Exception:
                    session_queue_empty = None

        audit = evaluate_x2_completion_audit(
            ledger,
            completed_receipts,
            confirmation_decision=confirmation_decision,
            case_recomputation_receipts=recomp,
            session_queue_empty=session_queue_empty,
        )
        lines = [
            "# ELENDRIL V8-X2 Global Completion Audit",
            "",
            f"- **Global Status:** `{audit['global_x2_status']}` (`x2_complete={audit['x2_complete']}`, `x3_unlocked={audit['x3_unlocked']}`)",
            f"- **Session Queue Empty:** `{audit['session_queue_empty']}`",
            f"- **Totals:** eligible={audit['totals']['eligible']}, completed={audit['totals']['completed']}, "
            f"negative={audit['totals']['negative']}, invalid={audit['totals']['invalid']}, "
            f"pending={audit['totals']['pending']}, not_triggered={audit['totals']['not_triggered']}",
            f"- **Legacy Compatibility Map:** original `{X1_LEGACY_ORIGINAL_SHA[:12]}` -> repaired `{X1_LEGACY_COMPAT_SHA[:12]}`",
            "",
            "## Counts by Block",
        ]
        for bname, bc in sorted(audit["counts_by_block"].items()):
            lines.append(
                f"- `{bname}`: eligible={bc['eligible']}, completed={bc['completed']}, "
                f"negative={bc['negative']}, invalid={bc['invalid']}, pending={bc['pending']}, not_triggered={bc['not_triggered']}"
            )
        if audit["unresolved_mandatory_ids"]:
            lines.extend(["", f"## Unresolved Mandatory IDs ({len(audit['unresolved_mandatory_ids'])})", ", ".join(audit["unresolved_mandatory_ids"][:20])])
        if audit["unresolved_conditional_ids"]:
            lines.extend(["", f"## Unresolved Conditional IDs ({len(audit['unresolved_conditional_ids'])})", ", ".join(audit["unresolved_conditional_ids"][:20])])
        md = "\n".join(lines) + "\n"

        _atomic_write_json(self.work_dir / "X2_COMPLETION_AUDIT.json", audit)
        (self.work_dir / "X2_REPORT.md").write_text(md, encoding="utf-8")
        if output_dir is not None:
            odir = Path(output_dir)
            odir.mkdir(parents=True, exist_ok=True)
            _atomic_write_json(odir / "X2_COMPLETION_AUDIT.json", audit)
            (odir / "X2_REPORT.md").write_text(md, encoding="utf-8")
        return audit

    def export_archives(
        self,
        output_dir: Path | str,
        *,
        prefix: str = "ELENDRIL_V8_X2",
        status: str = "COMPLETE",
        results_only: bool = False,
    ) -> dict[str, Any]:
        """Export verified complete ANALYSIS ZIP (and FULL ZIP unless results_only=True) with SHA-256 receipts."""
        analysis_files: dict[str, Path] = {}
        full_files: dict[str, Path] = {}
        for fname in (
            "X2_DESIGN_REGISTRY.json",
            "X2_OBLIGATION_LEDGER.json",
            "X2_COMPLETION_AUDIT.json",
            "X2_REPORT.md",
            "PREPARED_MANIFEST.json",
            "PENDING_QUEUE.json",
            "SESSION_ADMISSION_PLAN.json",
            "CAMPAIGN_STATUS.json",
            "VERIFIED_RESUME_RECEIPT.json",
            "ADMISSION_RECEIPT.json",
            "PREFLIGHT_REPLAY_RECEIPT.json",
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
                if results_only:
                    analysis_files[rel] = p
                else:
                    full_files[rel] = p
                continue
            if not rel.startswith("jobs/") or len(parts) < 2:
                continue
            jid = parts[1]
            if results_only:
                if (
                    rel.startswith(f"jobs/{jid}/checkpoints/")
                    or rel.startswith(f"jobs/{jid}/result_bundle/generations/")
                    or rel.startswith(f"jobs/{jid}/shared_backbones/")
                    or p.suffix == ".bin"
                    or p.name in ("checkpoint_full.zip", "checkpoint_full.zip.sha256")
                ):
                    continue
                if p.suffix in (".json", ".jsonl", ".md", ".npz") or rel in (
                    f"jobs/{jid}/result_bundle/result.zip",
                    f"jobs/{jid}/result_bundle/result.zip.sha256",
                ):
                    analysis_files[rel] = p
                continue
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
            full_files=None if results_only else full_files,
            status=status,
            source_commit=self.source_commit,
            protocol_sha256=self.registry["registry_sha256"],
            include_v6_manifest=True,
            write_full_zip=not results_only,
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
