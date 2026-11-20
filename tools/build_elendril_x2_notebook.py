"""Generate and verify the owner-armed Kaggle 2xT4 notebook for ELENDRIL V8-X2 (with gated X3 follow-up)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.x2.registry import X1_LEGACY_ORIGINAL_SHA, X1_LEGACY_SOURCE_SHA

REPO_HTTP_URL = "https://github.com/dhurv0045com-spec/An-Ra-the-new-AGI.git"
DEFAULT_NOTEBOOK_PATH = Path("notebooks/ELENDRIL_V8_X2_KAGGLE_T4X2.ipynb")
DEFAULT_X2_SESSIONS = (
    "A1",
    "A2",
    "A3",
    "D0",
    "D1",
    "D2",
    "U0",
    "U1",
    "IIIb-DM",
    "IIIb-P2",
    "IIIb-R2",
    "IIIb-U-P",
    "IIIb-PS",
    "BIV-T13",
    "BIV-T14",
    "BV-H4736",
)
FULL_X2_SESSIONS = (
    *DEFAULT_X2_SESSIONS,
    "BII-FROZEN",
    "EXTRA-T14-PS",
    "EXTRA-T13-HIST_A",
    "EXTRA-T13-HIST_APRIME",
    "EXTRA-T13-HIST_DISJOINT_PRETRAIN",
    "EXTRA-T13-HIST_FRESH_CTRL",
    "CONF-SRC",
    "CONF-T13",
    "CONF-T14",
)


def _cell_code(lines: list[str], cell_id: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {},
        "outputs": [],
        "source": [line + ("\n" if not line.endswith("\n") else "") for line in lines],
    }


def _cell_md(lines: list[str], cell_id: str) -> dict[str, object]:
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {},
        "source": [line + ("\n" if not line.endswith("\n") else "") for line in lines],
    }


def build_notebook_dict(
    *,
    source_commit: str,
    legacy_x1_commit: str = X1_LEGACY_SOURCE_SHA,
    legacy_x1_original_commit: str = X1_LEGACY_ORIGINAL_SHA,
    repo_url: str = REPO_HTTP_URL,
    owner_authorized_default: bool = True,
) -> dict[str, object]:
    """Build deterministic nbformat v4.5 JSON for ELENDRIL V8-X2 -> gated X3 on Kaggle 2xT4."""
    for label, sha in (
        ("source_commit", source_commit),
        ("legacy_x1_commit", legacy_x1_commit),
        ("legacy_x1_original_commit", legacy_x1_original_commit),
    ):
        if len(sha) != 40 or any(c not in "0123456789abcdef" for c in sha):
            raise ValueError(f"{label} must be a 40-character lowercase hex SHA, got {sha!r}")

    auth_literal = "True" if owner_authorized_default else "False"
    sessions_literal = json.dumps(list(FULL_X2_SESSIONS))
    cells = [
        _cell_md(
            [
                "# ELENDRIL V8-X2 (Repaired Continuation) & Gated X3 — Kaggle 2xT4",
                "",
                "Zero-edit owner-armed Kaggle 2xT4 notebook pinned to immutable commits:",
                f"- **X2/X3 Runtime (`SOURCE_SHA`):** `{source_commit}` (`PLANTHIR_BACKEND_MODULES=experiments.elendril_x2`)",
                f"- **Repaired Legacy X1 Worker (`LEGACY_X1_SHA`):** `{legacy_x1_commit}` (compatibility-mapped from Run 1 `{legacy_x1_original_commit[:12]}`)",
                "- Select **Accelerator: GPU T4 x2**, enable **Internet**, attach Run 1 `ELENDRIL_V8_X2_ANALYSIS.zip` (optional, resumes at `X1-A3-S20-B0-standard-s7`), and click **Run All**.",
                "- **Combined 10h 2xT4 session (`600m` session / `660m` outer / `45m` reserve):** runs preflight (`doctor` + dataset dispatch + X2 canary/replay + repaired `B0`/`J` smoke check), admits whole jobs, uses descendant-safe pruning + exports retained parent checkpoints (`ELENDRIL_V8_X2_PARENT_CHECKPOINTS.zip`), computes the 192-row `X2_COMPLETION_AUDIT.json`, and keeps **X3 strictly gated (`0.0` GPU seconds)** until `X2_COMPLETE` passes.",
            ],
            "x2-title",
        ),
        _cell_code(
            [
                "# Owner-armed zero-edit defaults: select 2xT4 GPU and click Run All.",
                f"OWNER_AUTHORIZED = {auth_literal}",
                f"SESSIONS = {sessions_literal}",
                "OUTER_MAX_MINUTES = 660",
                "SESSION_MAX_MINUTES = 600",
                "EXPORT_RESERVE_MINUTES = 45",
                "MEASURED_SEC_PER_UPDATE = 0.65",
                "PER_JOB_DISK_MB = 32",
                "PRUNE_COMPLETED_WEIGHTS = True",
                "RESULTS_ONLY_EXPORT = True",
                'DEVICES = "0,1"',
                "AUTO_RESUME_FROM_KAGGLE_INPUT = True",
                f'SOURCE_SHA = "{source_commit}"',
                f'LEGACY_X1_SHA = "{legacy_x1_commit}"',
                f'LEGACY_X1_ORIGINAL_SHA = "{legacy_x1_original_commit}"',
                f'REPO_URL = "{repo_url}"',
            ],
            "x2-config",
        ),
        _cell_code(
            [
                "import os, shutil, subprocess, sys",
                "from pathlib import Path",
                "",
                'os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"',
                'os.environ["PYTHONDONTWRITEBYTECODE"] = "1"',
                'os.environ["PYTHONUNBUFFERED"] = "1"',
                'os.environ["OMP_NUM_THREADS"] = "2"',
                'os.environ["MKL_NUM_THREADS"] = "2"',
                'os.environ["PLANTHIR_BACKEND_MODULES"] = "experiments.elendril_x2"',
                "",
                'WORK_ROOT = Path("/tmp/elendril_x2")',
                'REPO_X2_DIR = WORK_ROOT / "repo_x2"',
                'REPO_X1_DIR = WORK_ROOT / "repo_x1_legacy"',
                'OUTPUT_DIR = Path("/kaggle/working")',
                "",
                "def _checkout_pinned(dest: Path, sha: str) -> None:",
                "    if dest.exists():",
                "        shutil.rmtree(dest)",
                "    dest.mkdir(parents=True, exist_ok=True)",
                '    subprocess.run(["git", "-C", str(dest), "init", "-q"], check=True)',
                '    subprocess.run(["git", "-C", str(dest), "remote", "add", "origin", REPO_URL], check=True)',
                "    try:",
                '        subprocess.run(["git", "-C", str(dest), "fetch", "--depth=1", "origin", sha], check=True)',
                "    except subprocess.CalledProcessError:",
                '        subprocess.run(["git", "-C", str(dest), "fetch", "origin", sha], check=True)',
                '    subprocess.run(["git", "-C", str(dest), "checkout", "--detach", sha], check=True)',
                '    head = subprocess.check_output(["git", "-C", str(dest), "rev-parse", "HEAD"], text=True).strip()',
                '    assert head == sha, f"Checkout mismatch: expected {sha}, got {head}"',
                "",
                "if not OWNER_AUTHORIZED:",
                '    print("NOT LAUNCHED: set OWNER_AUTHORIZED = True in the configuration cell to execute X2.")',
                "else:",
                "    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)",
                "    WORK_ROOT.mkdir(parents=True, exist_ok=True)",
                "    _checkout_pinned(REPO_X2_DIR, SOURCE_SHA)",
                "    _checkout_pinned(REPO_X1_DIR, LEGACY_X1_SHA)",
                "    os.chdir(REPO_X2_DIR)",
                "    if str(REPO_X2_DIR) not in sys.path:",
                "        sys.path.insert(0, str(REPO_X2_DIR))",
                '    prev_pp = os.environ.get("PYTHONPATH", "")',
                "    parts = [p for p in prev_pp.split(os.pathsep) if p] if prev_pp else []",
                "    if str(REPO_X2_DIR) not in parts:",
                '        os.environ["PYTHONPATH"] = os.pathsep.join([str(REPO_X2_DIR), *parts]) if parts else str(REPO_X2_DIR)',
                '    print("Checked out X2/X3 runtime:", SOURCE_SHA)',
                '    print("Checked out repaired legacy X1 worker:", LEGACY_X1_SHA)',
            ],
            "x2-checkout",
        ),
        _cell_code(
            [
                "import json, shutil, subprocess, sys, torch",
                "from pathlib import Path",
                "from experiments.elendril_x2 import REGISTERED_X2_DATASETS, x2_dataset_factory",
                "from experiments.x2.tasks import verify_x2_dataset_dispatch_preflight",
                "",
                "if not OWNER_AUTHORIZED:",
                '    print("NOT LAUNCHED: owner switch is false")',
                "else:",
                "    gpu_count = torch.cuda.device_count()",
                '    print("Torch:", torch.__version__, "CUDA GPUs:", gpu_count)',
                "    for idx in range(gpu_count):",
                '        print(f"  cuda:{idx} -> {torch.cuda.get_device_name(idx)}")',
                '    assert torch.cuda.is_available() and gpu_count >= 2, f"2xT4 CUDA GPUs required, found {gpu_count}"',
                "",
                "    # 1. Palantiri environment doctor + full dataset dispatch preflight",
                '    subprocess.run([sys.executable, "-m", "anra_v5.palantiri", "doctor"], check=True)',
                '    prod_ds = [k for k in sorted(REGISTERED_X2_DATASETS) if k not in ("grad_check", "resume_parity", "trunc_ckpt", "x2_ds", "x2_canary_fixture")]',
                "    ds_preflight = verify_x2_dataset_dispatch_preflight(x2_dataset_factory, prod_ds, seed=7)",
                '    print("Dataset dispatch preflight PASSED:", ds_preflight["datasets_checked"], "distinct datasets verified.")',
                "",
                "    # 2. Mandatory X2 2-device canary + independent replay-bundle gate",
                '    canary_dir = WORK_ROOT / "preflight_canary"',
                "    canary_dir.mkdir(parents=True, exist_ok=True)",
                '    canary_spec = canary_dir / "x2_canary_spec.json"',
                '    subprocess.run([sys.executable, "-m", "anra_v5.palantiri", "init", "--backend", "elendril_x2", "--out", str(canary_spec)], check=True)',
                '    doc = json.loads(canary_spec.read_text(encoding="utf-8"))',
                '    doc["source_commit"] = SOURCE_SHA',
                '    doc["execution"]["device"] = "cuda:0"',
                '    canary_spec.write_text(json.dumps(doc, indent=2) + "\\n", encoding="utf-8")',
                '    subprocess.run([sys.executable, "-m", "anra_v5.palantiri", "validate", "--spec", str(canary_spec), "--canary"], check=True)',
                '    subprocess.run([sys.executable, "-m", "anra_v5.palantiri", "run", "--spec", str(canary_spec), "--work-root", str(canary_dir / "run"), "--canary", "--device", "cuda:0"], check=True)',
                '    subprocess.run([sys.executable, "-m", "anra_v5.palantiri", "verify-bundle", "--bundle-dir", str(canary_dir / "run" / "result_bundle"), "--expected-source-commit", SOURCE_SHA], check=True)',
                '    subprocess.run([sys.executable, "-m", "anra_v5.palantiri", "replay-bundle", "--bundle-dir", str(canary_dir / "run" / "result_bundle"), "--expected-source-commit", SOURCE_SHA, "--device", "cuda:0", "--output", str(canary_dir / "REPLAY_RECEIPT.json")], check=True)',
                "",
                "    # 3. Repaired legacy X1 B0 & J subprocess smoke check against pinned REPO_X1_DIR",
                '    b0_smoke = WORK_ROOT / "b0_j_smoke"',
                "    shutil.rmtree(b0_smoke, ignore_errors=True)",
                "    b0_smoke.mkdir(parents=True, exist_ok=True)",
                '    for smoke_jid, smoke_dev in (("X1-A3-S20-B0-standard-s7", "cuda:0"), ("X1-A3-S20-J-standard-s7", "cuda:1")):',
                '        sdir = b0_smoke / smoke_jid',
                "        subprocess.run([",
                '            sys.executable, "-m", "experiments.x2.campaign", "legacy-exec",',
                '            "--legacy-repo", str(REPO_X1_DIR), "--expected-sha", LEGACY_X1_SHA, "--",',
                '            "tools.elendril_v8x_worker", "prepare-splits", "--work-dir", str(sdir), "--cpu-fixtures",',
                "        ], check=True)",
                "        subprocess.run([",
                '            sys.executable, "-m", "experiments.x2.campaign", "legacy-exec",',
                '            "--legacy-repo", str(REPO_X1_DIR), "--expected-sha", LEGACY_X1_SHA, "--",',
                '            "tools.elendril_v8x_worker", "run-job", "--job-id", smoke_jid, "--work-dir", str(sdir),',
                '            "--source", LEGACY_X1_SHA, "--device", smoke_dev, "--cpu-fixtures", "--updates-override", "1",',
                "        ], check=True)",
                '    shutil.rmtree(b0_smoke, ignore_errors=True)',
                '    camp_work = WORK_ROOT / "campaign_work"',
                "    camp_work.mkdir(parents=True, exist_ok=True)",
                '    shutil.copy2(canary_dir / "REPLAY_RECEIPT.json", camp_work / "PREFLIGHT_REPLAY_RECEIPT.json")',
                '    shutil.rmtree(canary_dir / "run", ignore_errors=True)',
                '    print("PREFLIGHT CANARY, REPLAY-BUNDLE & REPAIRED B0/J SMOKE GATE PASSED.")',
            ],
            "x2-gpu-canary-gate",
        ),
        _cell_code(
            [
                "import json, time",
                "from pathlib import Path",
                "from experiments.x2.campaign import (",
                "    ElendrilX2Campaign,",
                "    X3_FROZEN_PARENT_IDS,",
                "    build_verified_resume_set,",
                "    discover_kaggle_input_receipts,",
                "    resolve_parent_checkpoint_path,",
                "    run_a0_historical_rescore,",
                ")",
                "from experiments.x3.runner import run_gated_x3_campaign",
                "",
                "if not OWNER_AUTHORIZED:",
                '    print("NOT LAUNCHED: owner switch is false")',
                "else:",
                '    camp_work = WORK_ROOT / "campaign_work"',
                "    camp = ElendrilX2Campaign(",
                "        source_commit=SOURCE_SHA,",
                "        work_dir=camp_work,",
                "        legacy_x1_repo_dir=REPO_X1_DIR,",
                "        outer_max_minutes=OUTER_MAX_MINUTES,",
                "        session_max_minutes=SESSION_MAX_MINUTES,",
                "        export_reserve_minutes=EXPORT_RESERVE_MINUTES,",
                "    )",
                "    camp.prepare(SESSIONS, include_extra_ledgers=True)",
                "",
                "    imported_receipts = []",
                '    k_in = Path("/kaggle/input")',
                "    if AUTO_RESUME_FROM_KAGGLE_INPUT and k_in.is_dir():",
                '        for a0_zip in sorted(k_in.rglob("*A0*ANALYSIS*.zip")):',
                '            print("Cross-checking mounted historical A0 ZIP:", a0_zip)',
                '            run_a0_historical_rescore(a0_zip, camp_work / "a0_rescore.json")',
                '        imported_receipts = discover_kaggle_input_receipts(k_in, camp_work / "imported_inputs", camp.registry)',
                '        print("Discovered mounted resume receipts:", len(imported_receipts))',
                "    resume_rec = build_verified_resume_set(registry=camp.registry, work_dir=camp_work, imported_receipts=imported_receipts)",
                '    print("Verified completed jobs imported:", resume_rec["verified_count"], "| compat_mapped:", len(resume_rec["compatibility_mapped_imports"]))',
                "",
                "    plan = camp.admit_slice(",
                "        SESSIONS,",
                "        measured_seconds_per_update=MEASURED_SEC_PER_UPDATE,",
                "        measured_per_job_disk_bytes=PER_JOB_DISK_MB * 1024 * 1024,",
                "        parallel_workers=2,",
                '        completed_ids=resume_rec["verified_completed_ids"],',
                "        include_extra_ledgers=True,",
                "    )",
                '    print("Admitted jobs for this session:", len(plan["admitted_job_ids"]), "->", plan["admitted_job_ids"][:10], "...")',
                '    admitted_set = set(plan["admitted_job_ids"])',
                '    rows = [r for r in camp.select_session_rows(SESSIONS, include_extra_ledgers=True) if r["job_id"] in admitted_set]',
                '    dev_list = [f"cuda:{d.strip()}" for d in DEVICES.split(",") if d.strip()]',
                '    deadline_epoch = time.time() + camp.budget["usable_train_minutes"] * 60.0',
                "    def _on_wave_complete(wave_idx: int, wave_rec: dict) -> None:",
                '        if wave_rec.get("succeeded_count", 0) > 0:',
                "            camp.export_archives(OUTPUT_DIR, prefix='ELENDRIL_V8_X2', status='PARTIAL', results_only=RESULTS_ONLY_EXPORT)",
                "    summary = camp.run_admitted_waves(",
                "        camp.to_session_jobs(rows),",
                "        devices=dev_list,",
                '        resume=bool(resume_rec["verified_completed_ids"]),',
                "        canary=False,",
                "        deadline_epoch=deadline_epoch,",
                "        fail_fast_on_empty_wave=True,",
                "        prune_completed_heavy_artifacts=PRUNE_COMPLETED_WEIGHTS,",
                "        on_wave_complete=_on_wave_complete,",
                '        input_roots=[k_in] if k_in.is_dir() else [],',
                "    )",
                "    x2_audit = camp.build_global_completion_audit(OUTPUT_DIR)",
                '    print("GLOBAL X2 AUDIT:", json.dumps({"status": x2_audit["global_x2_status"], "x2_complete": x2_audit["x2_complete"], "totals": x2_audit["totals"]}, indent=2))',
                "    parent_arc = camp.export_parent_checkpoints(OUTPUT_DIR)",
                '    print("Exported protected parent checkpoints:", parent_arc["exported_parent_ids"])',
                '    final_status = "COMPLETE" if x2_audit["x2_complete"] else "PARTIAL"',
                "    delivery = camp.export_archives(OUTPUT_DIR, prefix='ELENDRIL_V8_X2', status=final_status, results_only=RESULTS_ONLY_EXPORT)",
                '    print("X2 DELIVERY RECEIPT:", json.dumps(delivery["final_status"], indent=2, sort_keys=True))',
                "",
                "    # Strictly gated X3 execution: runs ONLY if global X2_COMPLETE == True; otherwise 0.0 GPU seconds spent",
                "    x3_ckpts = {}",
                '    if x2_audit["x2_complete"]:',
                "        for x3_pid in X3_FROZEN_PARENT_IDS:",
                '            p_ckpt = resolve_parent_checkpoint_path(camp.registry, x3_pid, work_dir=camp_work, input_roots=[k_in] if k_in.is_dir() else [])',
                "            if p_ckpt is not None:",
                '                parts = x3_pid.split("-")',
                '                arch = "F-cap" if "F-cap" in x3_pid else ("R-noedge" if "R-noedge" in x3_pid else parts[3])',
                '                seed_v = int(parts[-1].lstrip("s"))',
                "                x3_ckpts.setdefault(arch, {})[seed_v] = str(p_ckpt)",
                "    x3_res = run_gated_x3_campaign(",
                "        x2_completion_audit=x2_audit,",
                "        output_dir=OUTPUT_DIR,",
                "        source_commit=SOURCE_SHA,",
                "        checkpoint_map=x3_ckpts,",
                '        device="cuda:0",',
                "        run_confirmation=True,",
                "    )",
                '    print("X3 GATE / STATUS:", json.dumps({"status": x3_res.get("status"), "x3_executed": x3_res.get("x3_executed"), "gpu_seconds_spent": x3_res.get("gpu_seconds_spent")}, indent=2))',
                '    if len(summary["completed_jobs"]) == 0 and summary["failed_job_ids"]:',
                '        raise RuntimeError(f"Campaign had failed jobs with 0 completions: {summary[\'failed_job_ids\']}")',
            ],
            "x2-run-and-export",
        ),
        _cell_code(
            [
                "from IPython.display import FileLink, display",
                "from pathlib import Path",
                "",
                "if not OWNER_AUTHORIZED:",
                '    print("NOT LAUNCHED: owner switch is false")',
                "else:",
                '    out = Path("/kaggle/working")',
                '    targets = sorted(out.glob("ELENDRIL_V8_*.zip")) + sorted(out.glob("*.sha256")) + [',
                '        out / "X2_COMPLETION_AUDIT.json",',
                '        out / "X2_REPORT.md",',
                '        out / "X3_GATE_STATUS.json",',
                '        out / "X3_METRICS.json",',
                '        out / "X3_REPORT.md",',
                '        out / "FINAL_STATUS.json",',
                '        out / "DELIVERY_RECEIPT.json",',
                "    ]",
                "    for p in targets:",
                "        if p.is_file():",
                '            print(f"{p.name} ({p.stat().st_size:,} bytes)")',
                "            display(FileLink(str(p)))",
            ],
            "x2-links",
        ),
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "elendril_x2_source_commit": source_commit,
            "elendril_x1_legacy_commit": legacy_x1_commit,
            "elendril_x1_legacy_original_commit": legacy_x1_original_commit,
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def render_notebook_bytes(
    *,
    source_commit: str,
    legacy_x1_commit: str = X1_LEGACY_SOURCE_SHA,
    legacy_x1_original_commit: str = X1_LEGACY_ORIGINAL_SHA,
    repo_url: str = REPO_HTTP_URL,
    owner_authorized_default: bool = True,
) -> bytes:
    doc = build_notebook_dict(
        source_commit=source_commit,
        legacy_x1_commit=legacy_x1_commit,
        legacy_x1_original_commit=legacy_x1_original_commit,
        repo_url=repo_url,
        owner_authorized_default=owner_authorized_default,
    )
    return (json.dumps(doc, indent=2, sort_keys=True) + "\n").encode("utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build or check the pinned ELENDRIL V8-X2/X3 Kaggle 2xT4 notebook.")
    parser.add_argument("--source", required=True, help="40-char immutable X2/X3 runtime commit SHA (Commit A)")
    parser.add_argument("--legacy-x1", default=X1_LEGACY_SOURCE_SHA, help="40-char repaired legacy X1 worker commit SHA")
    parser.add_argument("--legacy-x1-original", default=X1_LEGACY_ORIGINAL_SHA, help="40-char original Run 1 X1 commit SHA")
    parser.add_argument("--output", default=str(DEFAULT_NOTEBOOK_PATH))
    parser.add_argument("--disabled", action="store_true", help="Emit OWNER_AUTHORIZED = False instead of True")
    parser.add_argument("--check", action="store_true", help="Verify existing file matches generated bytes exactly")
    args = parser.parse_args(argv)

    raw = render_notebook_bytes(
        source_commit=args.source,
        legacy_x1_commit=args.legacy_x1,
        legacy_x1_original_commit=args.legacy_x1_original,
        owner_authorized_default=not args.disabled,
    )
    out = Path(args.output)
    if args.check:
        if not out.is_file():
            raise SystemExit(f"Notebook missing: {out}")
        if out.read_bytes() != raw:
            raise SystemExit(f"Notebook drift detected at {out}")
        print(f"OK: {out} matches exact deterministic bytes for {args.source}")
        return 0
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(raw)
    print(f"Wrote {out} ({len(raw)} bytes) pinned to {args.source}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
