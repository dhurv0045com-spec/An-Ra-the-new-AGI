"""Deterministic Kaggle 2xT4 / T4 notebook generator for AN-RA V8-X4 (~20M S20 Only)."""

from __future__ import annotations

import argparse
import ast
from hashlib import sha256
import json
from pathlib import Path

REPO_HTTP_URL = "https://github.com/dhurv0045com-spec/An-Ra-the-new-AGI.git"
DEFAULT_X4_NOTEBOOK_PATH = Path("notebooks/ELENDRIL_V8_X4_KAGGLE_T4X2.ipynb")


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


def build_x4_notebook_dict(
    *,
    source_commit: str,
    repo_url: str = REPO_HTTP_URL,
    owner_authorized_default: bool = True,
    reserve_minutes: int = 45,
) -> dict[str, object]:
    """Build deterministic nbformat v4.5 JSON for V8-X4 (~20M S20 only, 4 resumable cycles, zero S50)."""
    if len(source_commit) != 40 or any(c not in "0123456789abcdef" for c in source_commit):
        raise ValueError(f"source_commit must be a 40-char lowercase hex SHA, got {source_commit!r}")
    if int(reserve_minutes) < 45:
        reserve_minutes = 45
    auth_literal = "True" if owner_authorized_default else "False"
    cells = [
        _cell_md(
            [
                "# ELENDRIL V8-X4 — Fundamental Reconstruction & Cognitive Mechanism Discovery (Kaggle 2xT4, ~20M S20)",
                "",
                "Zero-edit one-click Kaggle 2xT4 / T4 notebook executing the reconstructed **AN-RA V8-X4** program (`S20` ~20M regime ONLY — no S50 jobs):",
                f"- **Pinned X4 Source (`SOURCE_SHA`):** `{source_commit}` (`PLANTHIR_BACKEND_MODULES=experiments.x4.module`)",
                "- **Accelerator:** Select **GPU T4 x2** (auto-uses `cuda:0, cuda:1`; also runs on single **GPU T4**), enable **Internet**, and click **Run All** (`OWNER_AUTHORIZED = True`).",
                "- **4 Resumable Cycles:** Cycle 1 Foundational Integrity Gate (3.1–3.5) + Faithful Foundations (`Base`/`P-Ref`/`R-Ref`/`R-NoEdge`) -> Cycle 2 Matched-Exposure `D0–D4-ECCO` & Curriculum -> Cycle 3 `P1-Alpha`/`R1-Alpha`/`U2-Fix`/`U4-Fix`/`P1-OpBank` + Controls -> Cycle 4 Independent Confirmation, 22-Family L0–L5 Profile, X01–X12 Interventions & Capability Registry.",
            ],
            "x4-title",
        ),
        _cell_code(
            [
                "# Default owner switch is ON (`True`) for automatic one-click Kaggle execution.",
                f"OWNER_AUTHORIZED = {auth_literal}",
                "RUN_CYCLES = ('C1', 'C2', 'C3', 'C4')",
                "SESSION_MINUTES = 660",
                f"RESERVE_MINUTES = {int(reserve_minutes)}",
                "PER_JOB_DISK_MB = 95",
                "USE_ALL_VISIBLE_GPUS = True",
                "AUTO_RESUME_FROM_KAGGLE_INPUT = True",
                f'SOURCE_SHA = "{source_commit}"',
                f'REPO_URL = "{repo_url}"',
            ],
            "x4-config",
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
                'os.environ["PLANTHIR_BACKEND_MODULES"] = "experiments.x4.module"',
                "",
                'WORK_ROOT = Path("/tmp/elendril_x4")',
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
                '    print("Checked out pinned V8-X4 runtime:", SOURCE_SHA)',
            ],
            "x4-checkout",
        ),
        _cell_code(
            [
                "import gc, json, shutil, subprocess, sys, time, torch",
                "from pathlib import Path",
                "from experiments.x4.battery import PublicOracleX4Decoder, build_x4_battery, evaluate_x4_cases",
                "from experiments.x4.integrity import run_foundational_integrity_gate",
                "from experiments.x4.models import X4CognitiveModel, build_x4_backbone_spec",
                "from experiments.x4.objectives import build_x4_matched_batches, compute_x4_objective_step, prepack_x4_batches",
                "from experiments.x4.module import X4_DEFAULT_UPDATES, build_x4_cycle_ledger",
                "",
                "if not OWNER_AUTHORIZED:",
                '    print("NOT LAUNCHED: owner switch false")',
                "else:",
                "    gpu_count = torch.cuda.device_count()",
                '    assert torch.cuda.is_available() and gpu_count >= 1, f"CUDA GPU required, found {gpu_count}"',
                '    ACTIVE_DEVICES = tuple(f"cuda:{i}" for i in range(gpu_count)) if (USE_ALL_VISIBLE_GPUS and gpu_count >= 2) else ("cuda:0",)',
                "",
                "    # 1. Cycle 1 Foundational Integrity Gate (§3.1-§3.5) + 22-family Oracle Preflight",
                '    gate_rep = run_foundational_integrity_gate(WORK_ROOT / "preflight_gate")',
                '    assert gate_rep["all_passed"], f"Integrity gate failed: {gate_rep}"',
                '    ledger_preview = build_x4_cycle_ledger(source_commit=SOURCE_SHA, cycles=RUN_CYCLES)',
                '    assert ledger_preview["s50_jobs"] == 0',
                "",
                "    # 2. Palantiri Level-7 platform checks & shield",
                '    subprocess.run([sys.executable, "-m", "anra_v5.palantiri", "doctor"], check=True)',
                '    subprocess.run([sys.executable, "-m", "anra_v5.palantiri", "calibrate"], check=True)',
                '    subprocess.run([sys.executable, "-m", "anra_v5.palantiri", "shield"], check=True)',
                "",
                "    # 3. Fresh-subprocess Plugin CLI canary + independent replay-bundle on CUDA",
                '    can_dir = WORK_ROOT / "preflight_canary"',
                "    can_dir.mkdir(parents=True, exist_ok=True)",
                '    spec_p = can_dir / "x4_canary_spec.json"',
                '    subprocess.run([sys.executable, "-m", "anra_v5.palantiri", "init", "--backend", "elendril_x4", "--out", str(spec_p)], check=True)',
                '    doc = json.loads(spec_p.read_text(encoding="utf-8"))',
                '    doc["source_commit"] = SOURCE_SHA',
                '    spec_p.write_text(json.dumps(doc, indent=2) + "\\n", encoding="utf-8")',
                '    subprocess.run([sys.executable, "-m", "anra_v5.palantiri", "validate", "--spec", str(spec_p), "--canary"], check=True)',
                '    subprocess.run([sys.executable, "-m", "anra_v5.palantiri", "run", "--spec", str(spec_p), "--work-root", str(can_dir / "run"), "--canary", "--device", "cuda:0"], check=True)',
                '    subprocess.run([sys.executable, "-m", "anra_v5.palantiri", "verify-bundle", "--bundle-dir", str(can_dir / "run" / "result_bundle"), "--expected-source-commit", SOURCE_SHA], check=True)',
                '    subprocess.run([sys.executable, "-m", "anra_v5.palantiri", "replay-bundle", "--bundle-dir", str(can_dir / "run" / "result_bundle"), "--expected-source-commit", SOURCE_SHA, "--device", "cuda:0", "--output", str(can_dir / "REPLAY_RECEIPT.json")], check=True)',
                "    shutil.rmtree(can_dir, ignore_errors=True)",
                "",
                "    # 4. S20-Only (~20M) FP16 AMP training & batched byte-generation GPU timing profile",
                '    dev = torch.device("cuda:0")',
                '    prof_cases = build_x4_battery(split="train", per_family=2)["cases"][:32]',
                "    prof_matched = build_x4_matched_batches(prof_cases, seed=7, batch_size=16)",
                '    sp = build_x4_backbone_spec("S20")',
                '    pm = X4CognitiveModel("ANRA-X4-Int", sp, seed=7).to(dev)',
                "    popt = torch.optim.AdamW(pm.parameters(), lr=1e-4)",
                '    pscaler = torch.amp.GradScaler("cuda", enabled=True)',
                "    pbatches = prepack_x4_batches(prof_matched, max_length=pm.max_ctx, device=dev)",
                '    compute_x4_objective_step(pm, popt, pbatches[0], objective="D4-ECCO", use_amp=True, scaler=pscaler)',
                "    torch.cuda.synchronize(dev)",
                "    t0 = time.perf_counter()",
                "    for ui in range(3):",
                '        compute_x4_objective_step(pm, popt, pbatches[ui % len(pbatches)], objective="D4-ECCO", use_amp=True, scaler=pscaler)',
                "    torch.cuda.synchronize(dev)",
                "    sec_u = (time.perf_counter() - t0) / 3.0",
                "    t_dec = time.perf_counter()",
                "    pm.eval()",
                "    _ = pm.generate_bytes_batch([c['prompt'] for c in prof_cases[:4]])",
                "    torch.cuda.synchronize(dev)",
                "    batched_dec_s = (time.perf_counter() - t_dec) / 4.0",
                "    vram_mb = torch.cuda.max_memory_allocated(dev) / (1024 ** 2)",
                "    del pm, popt, pscaler, pbatches",
                "    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(dev)",
                "    MEASURED_SEC_PER_UPDATE = max(0.05, float(sec_u))",
                '    print("X4 PREFLIGHT & S20 PROFILE PASSED:", json.dumps({"devices": ACTIVE_DEVICES, "s20_jobs": ledger_preview["s20_jobs"], "s50_jobs": ledger_preview["s50_jobs"], "sec_per_update": round(sec_u, 4), "batched_decode_s_per_case": round(batched_dec_s, 4), "peak_vram_mb": round(vram_mb, 1)}, indent=2))',
            ],
            "x4-preflight",
        ),
        _cell_code(
            [
                "import hashlib, json",
                "from pathlib import Path",
                "from anra_v5.palantiri import dump_json_safe, mirror_to_working_dir, safe_extract_planthir_archive",
                "from experiments.x4.module import run_x4_campaign",
                "",
                "def _discover_x4_resume_receipts(input_root: Path, stage_dir: Path) -> list[dict]:",
                "    if not input_root.is_dir():",
                "        return []",
                "    stage_dir.mkdir(parents=True, exist_ok=True)",
                "    by_id: dict[str, dict] = {}",
                '    for idx, zp in enumerate(sorted(input_root.rglob("*ELENDRIL*X4*ANALYSIS*.zip"))):',
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
                "    return [by_id[k] for k in sorted(by_id)]",
                "",
                "if not OWNER_AUTHORIZED:",
                '    print("NOT LAUNCHED: owner switch false")',
                "else:",
                '    camp_work = WORK_ROOT / "campaign_work"',
                '    camp_export = WORK_ROOT / "campaign_exports"',
                "    camp_work.mkdir(parents=True, exist_ok=True)",
                "    camp_export.mkdir(parents=True, exist_ok=True)",
                '    resumed = _discover_x4_resume_receipts(Path("/kaggle/input"), WORK_ROOT / "imported_x4") if AUTO_RESUME_FROM_KAGGLE_INPUT else []',
                '    print("Discovered prior X4 completed receipts:", len(resumed))',
                "    try:",
                "        res = run_x4_campaign(",
                "            work_dir=camp_work,",
                "            export_dir=camp_export,",
                "            source_commit=SOURCE_SHA,",
                "            devices=ACTIVE_DEVICES,",
                "            cycles=RUN_CYCLES,",
                "            session_minutes=float(SESSION_MINUTES),",
                "            reserve_minutes=float(RESERVE_MINUTES),",
                "            measured_seconds_per_update=float(MEASURED_SEC_PER_UPDATE),",
                "            per_job_disk_bytes=int(PER_JOB_DISK_MB) * 1024 * 1024,",
                "            canary=False,",
                "            resume_receipts=resumed,",
                "            mirror_dir=OUTPUT_DIR,",
                "        )",
                '        print("X4 CAMPAIGN STATUS:", dump_json_safe({k: res.get(k) for k in ("status", "completed_count", "total_jobs", "s50_jobs", "next_resumable_job_id", "confirmation_lock", "analysis_archive_verified")}))',
                "    except BaseException as _exc:",
                '        (camp_work / "CRASH_RECEIPT.json").write_text(',
                '            dump_json_safe({"status": "INTERRUPTED" if isinstance(_exc, KeyboardInterrupt) else "FAILED", "error": f"{type(_exc).__name__}: {_exc}"}),',
                '            encoding="utf-8",',
                "        )",
                "        raise",
                "    finally:",
                "        if OUTPUT_DIR.is_dir():",
                '            salvage = [p for p in sorted(camp_export.glob("*.zip")) + sorted(camp_export.glob("*.sha256")) + sorted(camp_export.glob("*.json")) + sorted(camp_work.glob("*.json")) if p.is_file()]',
                '            salvage += [p for p in (camp_work / "RESULTS.md", camp_work / "X4_CAMPAIGN_SUMMARY.json", camp_work / "CRASH_RECEIPT.json") if p.is_file()]',
                "            if salvage:",
                "                mirror_to_working_dir(list(dict.fromkeys(salvage)), OUTPUT_DIR)",
            ],
            "x4-campaign",
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
                '    for p in sorted(out.glob("ELENDRIL_V8_X4*")) + [out / "X4_CAMPAIGN_SUMMARY.json", out / "RESULTS.md"]:',
                "        if p.is_file():",
                '            print(f"{p.name} ({p.stat().st_size:,} bytes)")',
                "            display(FileLink(str(p)))",
            ],
            "x4-links",
        ),
    ]

    for cell in cells:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "elendril_x4_source_commit": source_commit,
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def render_x4_notebook_bytes(
    *,
    source_commit: str,
    repo_url: str = REPO_HTTP_URL,
    owner_authorized_default: bool = True,
    reserve_minutes: int = 45,
) -> bytes:
    doc = build_x4_notebook_dict(
        source_commit=source_commit,
        repo_url=repo_url,
        owner_authorized_default=owner_authorized_default,
        reserve_minutes=reserve_minutes,
    )
    return (json.dumps(doc, indent=2, sort_keys=True) + "\n").encode("utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build or check pinned V8-X4 Kaggle 2xT4 notebook.")
    parser.add_argument("--source", required=True, help="40-char immutable CODE_SHA commit")
    parser.add_argument("--output", default=str(DEFAULT_X4_NOTEBOOK_PATH))
    parser.add_argument("--disabled", action="store_true", help="Explicit flag emitting OWNER_AUTHORIZED = False")
    parser.add_argument("--reserve-minutes", type=int, default=45)
    parser.add_argument("--check", "--check-notebook", dest="check", action="store_true")
    args = parser.parse_args(argv)

    raw = render_x4_notebook_bytes(
        source_commit=args.source,
        owner_authorized_default=not bool(args.disabled),
        reserve_minutes=int(args.reserve_minutes),
    )
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
