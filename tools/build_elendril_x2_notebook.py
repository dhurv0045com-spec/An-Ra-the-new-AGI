"""Generate and verify the owner-armed Kaggle 2xT4 notebook for ELENDRIL V8-X2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from experiments.x2.registry import X1_LEGACY_SOURCE_SHA

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
    repo_url: str = REPO_HTTP_URL,
    owner_authorized_default: bool = True,
) -> dict[str, object]:
    """Build deterministic nbformat v4.5 JSON for ELENDRIL V8-X2 on Kaggle 2xT4."""
    for label, sha in (("source_commit", source_commit), ("legacy_x1_commit", legacy_x1_commit)):
        if len(sha) != 40 or any(c not in "0123456789abcdef" for c in sha):
            raise ValueError(f"{label} must be a 40-character lowercase hex SHA, got {sha!r}")

    auth_literal = "True" if owner_authorized_default else "False"
    sessions_literal = json.dumps(list(DEFAULT_X2_SESSIONS))
    cells = [
        _cell_md(
            [
                "# ELENDRIL V8-X2 — Palantiri 2xT4 Coordinated Research Campaign",
                "",
                "Zero-edit owner-armed Kaggle 2xT4 notebook pinned to immutable commits:",
                f"- **X2 Runtime (`SOURCE_SHA`):** `{source_commit}` (`PLANTHIR_BACKEND_MODULES=experiments.elendril_x2`)",
                f"- **Legacy X1 Compatibility Lane (`LEGACY_X1_SHA`):** `{legacy_x1_commit}`",
                "- Select **Accelerator: GPU T4 x2**, enable **Internet**, and click **Run All** (no code edits required).",
                "- Enforces mandatory 2-GPU canary + independent `replay-bundle` verification gate before launching waves.",
                "- Automatically discovers mounted `/kaggle/input/**` prior receipts, runs within dual-clock budget (`480m` outer / `330m` session + `45m` reserve), and mirrors verified ZIPs + SHA-256 receipts to `/kaggle/working`.",
            ],
            "x2-title",
        ),
        _cell_code(
            [
                "# Owner-armed zero-edit defaults: select 2xT4 GPU and click Run All.",
                f"OWNER_AUTHORIZED = {auth_literal}",
                f"SESSIONS = {sessions_literal}",
                "OUTER_MAX_MINUTES = 480",
                "SESSION_MAX_MINUTES = 330",
                "EXPORT_RESERVE_MINUTES = 45",
                'DEVICES = "0,1"',
                "AUTO_RESUME_FROM_KAGGLE_INPUT = True",
                f'SOURCE_SHA = "{source_commit}"',
                f'LEGACY_X1_SHA = "{legacy_x1_commit}"',
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
                '    print("Checked out X2 runtime:", SOURCE_SHA)',
                '    print("Checked out legacy X1 worker:", LEGACY_X1_SHA)',
            ],
            "x2-checkout",
        ),
        _cell_code(
            [
                "import json, subprocess, sys, torch",
                "from pathlib import Path",
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
                "    # Mandatory pre-flight gate: Palantiri doctor + 2-device X2 canary + independent replay-bundle",
                '    subprocess.run([sys.executable, "-m", "anra_v5.palantiri", "doctor"], check=True)',
                '    canary_dir = WORK_ROOT / "preflight_canary"',
                '    canary_dir.mkdir(parents=True, exist_ok=True)',
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
                '    print("PREFLIGHT CANARY + INDEPENDENT REPLAY PASSED:", canary_dir / "REPLAY_RECEIPT.json")',
            ],
            "x2-gpu-canary-gate",
        ),
        _cell_code(
            [
                "import json, shutil, time",
                "from pathlib import Path",
                "from experiments.x2.campaign import ElendrilX2Campaign, build_verified_resume_set, run_a0_historical_rescore",
                "",
                "if not OWNER_AUTHORIZED:",
                '    print("NOT LAUNCHED: owner switch is false")',
                "else:",
                "    t_start = time.monotonic()",
                '    camp_work = WORK_ROOT / "campaign_work"',
                "    camp = ElendrilX2Campaign(",
                "        source_commit=SOURCE_SHA,",
                "        work_dir=camp_work,",
                "        legacy_x1_repo_dir=REPO_X1_DIR,",
                "        outer_max_minutes=OUTER_MAX_MINUTES,",
                "        session_max_minutes=SESSION_MAX_MINUTES,",
                "        export_reserve_minutes=EXPORT_RESERVE_MINUTES,",
                "    )",
                "    camp.prepare(SESSIONS)",
                "",
                "    # Auto-discover mounted /kaggle/input prior receipts and historical A0 ZIP",
                "    imported_receipts = []",
                '    k_in = Path("/kaggle/input")',
                "    if AUTO_RESUME_FROM_KAGGLE_INPUT and k_in.is_dir():",
                '        for a0_zip in sorted(k_in.rglob("*A0*ANALYSIS*.zip")):',
                '            print("Cross-checking mounted historical A0 ZIP:", a0_zip)',
                '            run_a0_historical_rescore(a0_zip, camp_work / "a0_rescore.json")',
                '        for rec_file in sorted(k_in.rglob("VERIFIED_RESUME_RECEIPT.json")):',
                '            print("Inspecting prior receipt:", rec_file)',
                "    resume_rec = build_verified_resume_set(registry=camp.registry, work_dir=camp_work, imported_receipts=imported_receipts)",
                '    print("Verified completed jobs imported:", resume_rec["verified_count"])',
                "",
                "    # Admit whole jobs fitting the governing dual-clock & disk budget and run on 2xT4",
                "    plan = camp.admit_slice(",
                "        SESSIONS,",
                "        measured_seconds_per_update=0.35,",
                "        parallel_workers=2,",
                '        completed_ids=resume_rec["verified_completed_ids"],',
                "    )",
                '    admitted_set = set(plan["admitted_job_ids"])',
                '    rows = [r for r in camp.select_session_rows(SESSIONS) if r["job_id"] in admitted_set]',
                '    dev_list = [f"cuda:{d.strip()}" for d in DEVICES.split(",") if d.strip()]',
                "    summary = camp.run_admitted_waves(",
                "        camp.to_session_jobs(rows),",
                "        devices=dev_list,",
                '        resume=bool(resume_rec["verified_completed_ids"]),',
                "        canary=False,",
                "    )",
                '    final_status = "COMPLETE" if (not summary["unfinished_job_ids"] and not plan["deferred_pending_job_ids"]) else "PARTIAL"',
                "    delivery = camp.export_archives(OUTPUT_DIR, prefix='ELENDRIL_V8_X2', status=final_status)",
                '    print("DELIVERY RECEIPT:", json.dumps(delivery["final_status"], indent=2, sort_keys=True))',
                '    print("NEXT RESUMABLE JOB ID:", summary.get("next_resumable_job_id") or plan.get("next_resumable_job_id"))',
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
                '    for p in sorted(out.glob("ELENDRIL_V8_X2*.zip")) + sorted(out.glob("*.sha256")) + [out / "FINAL_STATUS.json", out / "DELIVERY_RECEIPT.json"]:',
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
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def render_notebook_bytes(
    *,
    source_commit: str,
    legacy_x1_commit: str = X1_LEGACY_SOURCE_SHA,
    repo_url: str = REPO_HTTP_URL,
    owner_authorized_default: bool = True,
) -> bytes:
    doc = build_notebook_dict(
        source_commit=source_commit,
        legacy_x1_commit=legacy_x1_commit,
        repo_url=repo_url,
        owner_authorized_default=owner_authorized_default,
    )
    return (json.dumps(doc, indent=2, sort_keys=True) + "\n").encode("utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build or check the pinned ELENDRIL V8-X2 Kaggle 2xT4 notebook.")
    parser.add_argument("--source", required=True, help="40-char immutable X2 runtime commit SHA (Commit A)")
    parser.add_argument("--legacy-x1", default=X1_LEGACY_SOURCE_SHA, help="40-char legacy X1 worker commit SHA")
    parser.add_argument("--output", default=str(DEFAULT_NOTEBOOK_PATH))
    parser.add_argument("--disabled", action="store_true", help="Emit OWNER_AUTHORIZED = False instead of True")
    parser.add_argument("--check", action="store_true", help="Verify existing file matches generated bytes exactly")
    args = parser.parse_args(argv)

    raw = render_notebook_bytes(
        source_commit=args.source,
        legacy_x1_commit=args.legacy_x1,
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
