"""Deterministic Kaggle 2xT4 / T4 notebook generator for V8-X3 Beyond Prediction (§14)."""

from __future__ import annotations

import argparse
import ast
from hashlib import sha256
import json
from pathlib import Path

REPO_HTTP_URL = "https://github.com/dhurv0045com-spec/An-Ra-the-new-AGI.git"
DEFAULT_X3_NOTEBOOK_PATH = Path("notebooks/ELENDRIL_V8_X3_KAGGLE_T4X2.ipynb")


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


def build_x3_notebook_dict(
    *,
    source_commit: str,
    repo_url: str = REPO_HTTP_URL,
    owner_authorized_default: bool = True,
) -> dict[str, object]:
    """Build deterministic nbformat v4.5 JSON for V8-X3 on Kaggle 2xT4 / T4."""
    if len(source_commit) != 40 or any(c not in "0123456789abcdef" for c in source_commit):
        raise ValueError(f"source_commit must be a 40-char lowercase hex SHA, got {source_commit!r}")
    auth_literal = "True" if owner_authorized_default else "False"
    cells = [
        _cell_md(
            [
                "# ELENDRIL V8-X3 — Beyond Prediction: Cognitive Intelligence Program (Kaggle 2xT4)",
                "",
                "Zero-edit owner-armed Kaggle GPU notebook pinned to immutable runtime commit:",
                f"- **Pinned X3 Source (`SOURCE_SHA`):** `{source_commit}` (`PLANTHIR_BACKEND_MODULES=experiments.x3.module`)",
                "- **Accelerator:** Select **GPU T4 x2** (auto-uses `cuda:0, cuda:1`; also runs safely on single **GPU T4**), enable **Internet**, and click **Run All**.",
                "- **Stages:** Preflight (`doctor`, `calibrate`, CLI `replay-bundle`, 9-arch `u0000` smoke, honest FP16 AMP S20/S50 GPU profiling) -> One-Go Stage 0–3 S20 (45 jobs x 1,024 FP16 AMP updates) -> Evidence-Gated Stage 4 S50 `50.55M` (12 jobs, unlocked ONLY if S20 passes) -> Stage 5 Complete-Cohort Sealed Confirmation + OOD Transfer + 5 Learning Modes -> Verified Archives (`_ANALYSIS.zip`, `_FULL.zip`, `_SUMMARY.zip`, `_EVIDENCE.zip`).",
            ],
            "x3-title",
        ),
        _cell_code(
            [
                "# Owner-armed zero-edit defaults: ONE-GO full completion inside a single 11h Kaggle T4x2 session.",
                f"OWNER_AUTHORIZED = {auth_literal}",
                "ONE_GO = True",
                "SESSION_MINUTES = 660",
                "RESERVE_MINUTES = 25",
                "PER_JOB_DISK_MB = 95",
                "USE_ALL_VISIBLE_GPUS = True",
                "AUTO_RESUME_FROM_KAGGLE_INPUT = True",
                f'SOURCE_SHA = "{source_commit}"',
                f'REPO_URL = "{repo_url}"',
            ],
            "x3-config",
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
                'os.environ["PLANTHIR_BACKEND_MODULES"] = "experiments.x3.module"',
                "",
                'WORK_ROOT = Path("/tmp/elendril_x3")',
                'REPO_DIR = WORK_ROOT / "repo"',
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
                '    assert head == sha, f"Checkout mismatch: {sha} != {head}"',
                "",
                "if not OWNER_AUTHORIZED:",
                '    print("NOT LAUNCHED: set OWNER_AUTHORIZED = True in the configuration cell.")',
                "else:",
                "    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)",
                "    WORK_ROOT.mkdir(parents=True, exist_ok=True)",
                "    _checkout_pinned(REPO_DIR, SOURCE_SHA)",
                "    os.chdir(REPO_DIR)",
                "    if str(REPO_DIR) not in sys.path:",
                "        sys.path.insert(0, str(REPO_DIR))",
                '    prev_pp = os.environ.get("PYTHONPATH", "")',
                "    parts = [p for p in prev_pp.split(os.pathsep) if p] if prev_pp else []",
                "    if str(REPO_DIR) not in parts:",
                '        os.environ["PYTHONPATH"] = os.pathsep.join([str(REPO_DIR), *parts]) if parts else str(REPO_DIR)',
                '    print("Checked out pinned V8-X3 runtime:", SOURCE_SHA)',
            ],
            "x3-checkout",
        ),
        _cell_code(
                [
                "import gc, json, shutil, subprocess, sys, time, torch",
                "from pathlib import Path",
                "from experiments.x3.models import X3CognitiveModel, build_x3_backbone_spec",
                "from experiments.x3.d3_objectives import build_d3_hybrid_batches, compute_d3_objective_step, prepack_d3_batches",
                "from experiments.x3.cognitive_battery import build_x3_cognitive_battery",
                "from experiments.x3.module import X3_DEFAULT_UPDATES",
                "",
                "if not OWNER_AUTHORIZED:",
                '    print("NOT LAUNCHED: owner switch false")',
                "else:",
                "    gpu_count = torch.cuda.device_count()",
                '    print("Torch:", torch.__version__, "CUDA GPUs:", gpu_count)',
                "    for idx in range(gpu_count):",
                '        print(f"  cuda:{idx} -> {torch.cuda.get_device_name(idx)}")',
                '    assert torch.cuda.is_available() and gpu_count >= 1, f"CUDA GPU required, found {gpu_count}"',
                '    ACTIVE_DEVICES = tuple(f"cuda:{i}" for i in range(gpu_count)) if (USE_ALL_VISIBLE_GPUS and gpu_count >= 2) else ("cuda:0",)',
                "",
                "    # 1. Palantiri platform self-checks",
                '    subprocess.run([sys.executable, "-m", "anra_v5.palantiri", "doctor"], check=True)',
                '    subprocess.run([sys.executable, "-m", "anra_v5.palantiri", "calibrate"], check=True)',
                "",
                "    # 2. All 9-architecture u0000 init+train+eval+restore smoke test",
                '    smk_dir = WORK_ROOT / "preflight_u0000"',
                '    subprocess.run([sys.executable, "-m", "experiments.x3.module", "u0000-smoke", "--work-dir", str(smk_dir), "--source", SOURCE_SHA], check=True)',
                "    shutil.rmtree(smk_dir, ignore_errors=True)",
                "",
                "    # 3. Full fresh-subprocess Plugin CLI canary + independent replay-bundle gate on CUDA",
                '    can_dir = WORK_ROOT / "preflight_canary"',
                "    can_dir.mkdir(parents=True, exist_ok=True)",
                '    spec_p = can_dir / "x3_canary_spec.json"',
                '    subprocess.run([sys.executable, "-m", "anra_v5.palantiri", "init", "--backend", "elendril_x3", "--out", str(spec_p)], check=True)',
                '    doc = json.loads(spec_p.read_text(encoding="utf-8"))',
                '    doc["source_commit"] = SOURCE_SHA',
                '    spec_p.write_text(json.dumps(doc, indent=2) + "\\n", encoding="utf-8")',
                '    subprocess.run([sys.executable, "-m", "anra_v5.palantiri", "validate", "--spec", str(spec_p), "--canary"], check=True)',
                '    subprocess.run([sys.executable, "-m", "anra_v5.palantiri", "run", "--spec", str(spec_p), "--work-root", str(can_dir / "run"), "--canary", "--device", "cuda:0"], check=True)',
                '    subprocess.run([sys.executable, "-m", "anra_v5.palantiri", "verify-bundle", "--bundle-dir", str(can_dir / "run" / "result_bundle"), "--expected-source-commit", SOURCE_SHA], check=True)',
                '    subprocess.run([sys.executable, "-m", "anra_v5.palantiri", "replay-bundle", "--bundle-dir", str(can_dir / "run" / "result_bundle"), "--expected-source-commit", SOURCE_SHA, "--device", "cuda:0", "--output", str(can_dir / "REPLAY_RECEIPT.json")], check=True)',
                "    shutil.rmtree(can_dir, ignore_errors=True)",
                "",
                "    # 4. Honest Dual-Backbone FP16 AMP GPU Timing Profile (S20 + S50 separately at production batch_size=16 with cuda.synchronize)",
                '    prof_bat = build_x3_cognitive_battery(split="train", per_family=4)["cases"]',
                "    prof_raw = build_d3_hybrid_batches(prof_bat, seed=7, batch_size=16)",
                "    TIMINGS = {}",
                '    for sc_name in ("S20", "S50"):',
                '        dev = torch.device("cuda:0")',
                "        sp = build_x3_backbone_spec(sc_name)",
                '        pm = X3CognitiveModel("P-R-Iter", sp, seed=7).to(dev)',
                "        popt = torch.optim.AdamW(pm.parameters(), lr=1e-4)",
                '        pscaler = torch.amp.GradScaler("cuda", enabled=True)',
                '        pbatches = prepack_d3_batches(prof_raw, max_length=pm.max_ctx, device=dev, objective="D3-Hybrid")',
                '        compute_d3_objective_step(pm, popt, pbatches[0], objective="D3-Hybrid", use_amp=True, scaler=pscaler)',
                "        torch.cuda.synchronize(dev)",
                "        t0 = time.perf_counter()",
                "        for u_i in range(3):",
                '            compute_d3_objective_step(pm, popt, pbatches[(u_i + 1) % len(pbatches)], objective="D3-Hybrid", use_amp=True, scaler=pscaler)',
                "        torch.cuda.synchronize(dev)",
                "        sec_u = (time.perf_counter() - t0) / 3.0",
                "        vram_mb = torch.cuda.max_memory_allocated(dev) / (1024 ** 2)",
                '        TIMINGS[sc_name] = {"sec_per_update": round(sec_u, 4), "peak_vram_mb": round(vram_mb, 1)}',
                "        del pm, popt, pscaler, pbatches",
                "        gc.collect()",
                "        torch.cuda.empty_cache()",
                "        torch.cuda.reset_peak_memory_stats(dev)",
                '    MEASURED_SEC_PER_UPDATE = max(0.05, float(TIMINGS["S20"]["sec_per_update"]))',
                "    n_lanes = max(1, len(ACTIVE_DEVICES))",
                '    est_s20_min = (45 * X3_DEFAULT_UPDATES * TIMINGS["S20"]["sec_per_update"]) / (60.0 * n_lanes)',
                '    est_s50_min = (12 * X3_DEFAULT_UPDATES * TIMINGS["S50"]["sec_per_update"]) / (60.0 * n_lanes)',
                '    TIMINGS["one_go_estimate_minutes"] = {"s20_45_jobs": round(est_s20_min, 1), "s50_12_jobs": round(est_s50_min, 1), "total_if_s50_unlocked": round(est_s20_min + est_s50_min, 1)}',
                '    print("PREFLIGHT & DUAL-BACKBONE FP16 AMP PROFILE PASSED:", json.dumps({"devices": ACTIVE_DEVICES, "updates_per_job": X3_DEFAULT_UPDATES, "timings": TIMINGS}, indent=2))',
            ],
            "x3-preflight",
        ),
        _cell_code(
            [
                "import hashlib, json, zipfile",
                "from pathlib import Path",
                "from anra_v5.palantiri import safe_extract_planthir_archive",
                "from experiments.x3.module import run_x3_campaign",
                "",
                "def _discover_x3_resume_receipts(input_root: Path, stage_dir: Path) -> list[dict]:",
                "    if not input_root.is_dir():",
                "        return []",
                "    stage_dir.mkdir(parents=True, exist_ok=True)",
                "    by_id: dict[str, dict] = {}",
                '    zips = sorted(input_root.rglob("*ELENDRIL*X3*ANALYSIS*.zip"))',
                "    for idx, zp in enumerate(zips):",
                '        dest = stage_dir / f"a{idx:03d}"',
                "        try:",
                "            safe_extract_planthir_archive(zp, dest)",
                "            z_sha = hashlib.sha256(zp.read_bytes()).hexdigest()",
                "        except Exception:",
                "            continue",
                "        for rp in sorted(dest.rglob('JOB_RECEIPT.json')):",
                "            try:",
                "                doc = json.loads(rp.read_text(encoding='utf-8'))",
                "            except Exception:",
                "                continue",
                "            jid = str(doc.get('job_id', ''))",
                "            if doc.get('status') == 'COMPLETE' and jid:",
                "                doc.setdefault('archive_path', str(zp))",
                "                doc.setdefault('archive_sha256', z_sha)",
                "                by_id[jid] = doc",
                "    # Scan auto-unpacked /kaggle/input folders outside extracted stage_dir",
                "    for rp in sorted(input_root.rglob('JOB_RECEIPT.json')):",
                "        try:",
                "            doc = json.loads(rp.read_text(encoding='utf-8'))",
                "        except Exception:",
                "            continue",
                "        jid = str(doc.get('job_id', ''))",
                "        if doc.get('status') == 'COMPLETE' and jid and jid not in by_id:",
                "            by_id[jid] = doc",
                "    return [by_id[k] for k in sorted(by_id)]",
                "",
                "if not OWNER_AUTHORIZED:",
                '    print("NOT LAUNCHED: owner switch false")',
                "else:",
                '    camp_work = WORK_ROOT / "campaign_work"',
                '    camp_export = WORK_ROOT / "campaign_exports"',
                '    resumed = _discover_x3_resume_receipts(Path("/kaggle/input"), WORK_ROOT / "imported_x3") if AUTO_RESUME_FROM_KAGGLE_INPUT else []',
                '    print("Discovered prior X3 completed receipts:", len(resumed))',
                "    res = run_x3_campaign(",
                "        work_dir=camp_work,",
                "        export_dir=camp_export,",
                "        source_commit=SOURCE_SHA,",
                "        devices=ACTIVE_DEVICES,",
                "        session_minutes=float(SESSION_MINUTES),",
                "        reserve_minutes=float(RESERVE_MINUTES),",
                "        measured_seconds_per_update=float(MEASURED_SEC_PER_UPDATE),",
                "        per_job_disk_bytes=int(PER_JOB_DISK_MB) * 1024 * 1024,",
                "        canary=False,",
                "        resume_receipts=resumed,",
                "        mirror_dir=OUTPUT_DIR,",
                "    )",
                '    print("X3 CAMPAIGN COMPLETE:", json.dumps({k: res.get(k) for k in ("status", "completed_count", "total_jobs", "next_resumable_job_id", "s50_scaling_gate", "confirmation_lock", "analysis_archive_verified")}, indent=2))',
            ],
            "x3-campaign",
        ),
        _cell_code(
            [
                "from IPython.display import FileLink, display",
                "from pathlib import Path",
                "",
                "if not OWNER_AUTHORIZED:",
                '    print("NOT LAUNCHED: owner switch false")',
                "else:",
                '    out = Path("/kaggle/working")',
                '    for p in sorted(out.glob("ELENDRIL_V8_X3*")) + [out / "X3_CAMPAIGN_SUMMARY.json", out / "RESULTS.md"]:',
                "        if p.is_file():",
                '            print(f"{p.name} ({p.stat().st_size:,} bytes)")',
                "            display(FileLink(str(p)))",
            ],
            "x3-links",
        ),
    ]

    # Validate every code cell syntactically with ast.parse before returning!
    for cell in cells:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "elendril_x3_source_commit": source_commit,
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def render_x3_notebook_bytes(
    *,
    source_commit: str,
    repo_url: str = REPO_HTTP_URL,
    owner_authorized_default: bool = True,
) -> bytes:
    doc = build_x3_notebook_dict(
        source_commit=source_commit,
        repo_url=repo_url,
        owner_authorized_default=owner_authorized_default,
    )
    return (json.dumps(doc, indent=2, sort_keys=True) + "\n").encode("utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build or check pinned V8-X3 Kaggle 2xT4 notebook.")
    parser.add_argument("--source", required=True, help="40-char immutable CODE_SHA commit")
    parser.add_argument("--output", default=str(DEFAULT_X3_NOTEBOOK_PATH))
    parser.add_argument("--disabled", action="store_true")
    parser.add_argument("--check", "--check-notebook", dest="check", action="store_true")
    args = parser.parse_args(argv)

    raw = render_x3_notebook_bytes(source_commit=args.source, owner_authorized_default=not args.disabled)
    out = Path(args.output)
    if args.check:
        if not out.is_file() or out.read_bytes() != raw:
            raise SystemExit(f"Notebook check mismatch at {out} (match=false)")
        print(f"OK: match=true sha256={sha256(raw).hexdigest()} bytes={len(raw)}")
        return 0
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(raw)
    print(f"Wrote {out} ({len(raw)} bytes, sha256={sha256(raw).hexdigest()}) pinned to {args.source}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
