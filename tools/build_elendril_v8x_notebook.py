"""Generate and verify the owner-armed Kaggle 2xT4 notebook for ELENDRIL V8-X1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

REPO_HTTP_URL = "https://github.com/dhurv0045com-spec/An-Ra-the-new-AGI.git"
DEFAULT_NOTEBOOK_PATH = Path("notebooks/ELENDRIL_V8_X1_KAGGLE_T4X2.ipynb")
DEFAULT_CORE_SESSIONS = ("A0", "A1", "A2", "A3", "D0", "D1", "D2")


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
    repo_url: str = REPO_HTTP_URL,
    owner_authorized_default: bool = True,
) -> dict[str, object]:
    """Build deterministic nbformat v4.5 JSON for ELENDRIL V8-X1 on Kaggle 2xT4."""

    if len(source_commit) != 40 or any(c not in "0123456789abcdef" for c in source_commit):
        raise ValueError("source_commit must be a 40-character lowercase hex SHA")

    auth_literal = "True" if owner_authorized_default else "False"
    cells = [
        _cell_md(
            [
                "# ELENDRIL V8-X1 — Automatic 2xT4 Full Screen",
                "",
                "Owner-armed Kaggle 2xT4 notebook pinned to immutable runtime commit:",
                f"- `SOURCE_SHA = \"{source_commit}\"`",
                "- Automatically runs all core sessions `A0 -> A1 -> A2 -> A3 -> D0 -> D1 -> D2` within available GPU wall-clock & disk budget.",
                "- Select **Accelerator: GPU T4 x2**, enable **Internet**, and click **Run All** (no code edits required).",
            ],
            "x1-title",
        ),
        _cell_code(
            [
                "# Owner-armed defaults: select 2xT4 GPU and click Run All with zero edits.",
                f"OWNER_AUTHORIZED = {auth_literal}",
                'SESSIONS = ["A0", "A1", "A2", "A3", "D0", "D1", "D2"]  # Optional extensions: "U0", "U1"',
                "MINUTES = 480",
                'DEVICES = "0,1"',
                "RESUME = False",
                f'SOURCE_SHA = "{source_commit}"',
                f'REPO_URL = "{repo_url}"',
            ],
            "x1-config",
        ),
        _cell_code(
            [
                "import os, shutil, subprocess, sys, tarfile",
                "from pathlib import Path",
                "",
                'os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"',
                'os.environ["PYTHONDONTWRITEBYTECODE"] = "1"',
                'os.environ["PYTHONUNBUFFERED"] = "1"',
                'os.environ["OMP_NUM_THREADS"] = "2"',
                'os.environ["MKL_NUM_THREADS"] = "2"',
                "",
                'WORK_ROOT = Path("/tmp/elendril_x1")',
                'REPO_DIR = WORK_ROOT / "repo"',
                'OUTPUT_DIR = Path("/kaggle/working")',
                "",
                "if not OWNER_AUTHORIZED:",
                '    print("NOT LAUNCHED: set OWNER_AUTHORIZED = True in the configuration cell to execute X1.")',
                "else:",
                "    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)",
                "    WORK_ROOT.mkdir(parents=True, exist_ok=True)",
                "    if REPO_DIR.exists():",
                "        shutil.rmtree(REPO_DIR)",
                "    REPO_DIR.mkdir(parents=True, exist_ok=True)",
                '    local_tar = Path(f"/kaggle/input/elendril-x1-source/{SOURCE_SHA}.tar.gz")',
                "    if local_tar.exists():",
                '        tar_path = WORK_ROOT / f"{SOURCE_SHA}.tar.gz"',
                "        shutil.copy2(local_tar, tar_path)",
                '        with tarfile.open(tar_path, "r:gz") as tf:',
                "            tf.extractall(REPO_DIR)",
                "    else:",
                '        subprocess.run(["git", "-C", str(REPO_DIR), "init", "-q"], check=True)',
                '        subprocess.run(["git", "-C", str(REPO_DIR), "remote", "add", "origin", REPO_URL], check=True)',
                "        try:",
                '            subprocess.run(["git", "-C", str(REPO_DIR), "fetch", "--depth=1", "origin", SOURCE_SHA], check=True)',
                "        except subprocess.CalledProcessError:",
                '            subprocess.run(["git", "-C", str(REPO_DIR), "fetch", "origin", SOURCE_SHA], check=True)',
                '        subprocess.run(["git", "-C", str(REPO_DIR), "checkout", "--detach", SOURCE_SHA], check=True)',
                "    os.chdir(REPO_DIR)",
                "    if str(REPO_DIR) not in sys.path:",
                "        sys.path.insert(0, str(REPO_DIR))",
                '    prev_pp = os.environ.get("PYTHONPATH", "")',
                "    pp_parts = [p for p in prev_pp.split(os.pathsep) if p] if prev_pp else []",
                "    if str(REPO_DIR) not in pp_parts:",
                '        os.environ["PYTHONPATH"] = os.pathsep.join([str(REPO_DIR), *pp_parts]) if pp_parts else str(REPO_DIR)',
                '    print("Checked out immutable X1 runtime:", SOURCE_SHA)',
            ],
            "x1-checkout",
        ),
        _cell_code(
            [
                "import torch",
                "if not OWNER_AUTHORIZED:",
                '    print("NOT LAUNCHED: set OWNER_AUTHORIZED = True")',
                "else:",
                "    gpu_count = torch.cuda.device_count()",
                '    print("Torch:", torch.__version__, "CUDA GPUs:", gpu_count)',
                "    for idx in range(gpu_count):",
                '        print(f"  cuda:{idx} -> {torch.cuda.get_device_name(idx)}")',
                '    assert torch.cuda.is_available() and gpu_count >= 2, f"2xT4 CUDA GPUs required, found {gpu_count}"',
            ],
            "x1-gpu-check",
        ),
        _cell_code(
            [
                "import json, os, shutil, subprocess, sys, time",
                "from pathlib import Path",
                "",
                "if not OWNER_AUTHORIZED:",
                '    print("NOT LAUNCHED: set OWNER_AUTHORIZED = True")',
                "else:",
                "    if str(REPO_DIR) not in sys.path:",
                "        sys.path.insert(0, str(REPO_DIR))",
                "    from tools.elendril_v8x_export import verify_x1_export",
                "",
                "    env = dict(os.environ)",
                '    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"',
                '    env["PYTHONDONTWRITEBYTECODE"] = "1"',
                '    env["PYTHONUNBUFFERED"] = "1"',
                '    env["OMP_NUM_THREADS"] = "2"',
                '    env["MKL_NUM_THREADS"] = "2"',
                "",
                "    notebook_t0 = time.time()",
                "    GLOBAL_DEADLINE = notebook_t0 + max(60, (MINUTES - 45)) * 60.0",
                '    shared_data = WORK_ROOT / "shared_data"',
                "    completed_sessions, partial_sessions, deferred_sessions, failed_sessions = [], [], [], []",
                "",
                "    for idx, sess in enumerate(SESSIONS):",
                '        prior = OUTPUT_DIR / f"FINAL_STATUS_{sess}.json"',
                "        if prior.is_file() and json.loads(prior.read_text(encoding='utf-8')).get('status') == 'COMPLETE':",
                '            print(f"[{sess}] Skipping: already COMPLETE in {prior}")',
                "            completed_sessions.append(sess)",
                "            continue",
                "",
                "        rem_minutes = int((GLOBAL_DEADLINE - time.time()) // 60) + 45",
                "        out_free_gb = shutil.disk_usage(OUTPUT_DIR).free / (1024 ** 3)",
                "        if rem_minutes < 60 or out_free_gb < 4.0:",
                '            print(f"[{sess}] Stopping cleanly before session: rem_minutes={rem_minutes}m, output_free={out_free_gb:.2f} GiB")',
                "            deferred_sessions.extend(SESSIONS[idx:])",
                "            break",
                "",
                '        print(f"=== ELENDRIL V8-X1 Session {sess} ({idx + 1}/{len(SESSIONS)}) | rem={rem_minutes}m | out_free={out_free_gb:.2f} GiB ===")',
                '        sess_dir = WORK_ROOT / f"session_{sess}"',
                '        common = ["--source", SOURCE_SHA, "--session", sess, "--work-dir", str(sess_dir)]',
                "",
                '        if (shared_data / "manifest.json").is_file() and not (sess_dir / "data" / "manifest.json").exists():',
                '            (sess_dir / "data").mkdir(parents=True, exist_ok=True)',
                '            shutil.copy2(shared_data / "manifest.json", sess_dir / "data" / "manifest.json")',
                '            if (shared_data / "eval_splits.json").is_file():',
                '                shutil.copy2(shared_data / "eval_splits.json", sess_dir / "data" / "eval_splits.json")',
                "",
                '        subprocess.run([sys.executable, "-m", "tools.elendril_v8x_campaign", "prepare", *common], env=env, cwd=str(REPO_DIR), check=True)',
                '        if not (shared_data / "manifest.json").is_file():',
                '            shutil.copytree(sess_dir / "data", shared_data, dirs_exist_ok=True)',
                "",
                '        subprocess.run([sys.executable, "-m", "tools.elendril_v8x_campaign", "validate", *common, "--cpu-fixtures"], env=env, cwd=str(REPO_DIR), check=True)',
                "        adm_proc = subprocess.run(",
                '            [sys.executable, "-m", "tools.elendril_v8x_campaign", "admit", *common, "--devices", DEVICES, "--max-minutes", str(rem_minutes)],',
                "            env=env, cwd=str(REPO_DIR), check=False,",
                "        )",
                "        if adm_proc.returncode != 0:",
                '            adm_path = sess_dir / "admission.json"',
                '            adm_doc = json.loads(adm_path.read_text(encoding="utf-8")) if adm_path.is_file() else {}',
                '            print(f"[{sess}] Admission stopped: {adm_doc.get(\'reason_code\')} - {adm_doc.get(\'reason\')}")',
                '            if (completed_sessions or partial_sessions) and adm_doc.get("reason_code") in ("TIME_ADMISSION_FAILED", "DISK_ADMISSION_FAILED"):',
                "                deferred_sessions.extend(SESSIONS[idx:])",
                "                break",
                '            raise RuntimeError(f"Admission failed on {sess}: {adm_doc}")',
                "",
                "        run_cmd = [",
                '            sys.executable, "-m", "tools.elendril_v8x_campaign",',
                '            "resume" if RESUME else "run",',
                "            *common,",
                '            "--devices", DEVICES,',
                '            "--max-minutes", str(rem_minutes),',
                '            "--output-dir", str(OUTPUT_DIR),',
                '            "--owner-authorized",',
                "        ]",
                "        subprocess.run(run_cmd, env=env, cwd=str(REPO_DIR), check=False)",
                "        receipt = verify_x1_export(OUTPUT_DIR, session=sess, expected_source_commit=SOURCE_SHA)",
                '        st_path = OUTPUT_DIR / "FINAL_STATUS.json"',
                "        if st_path.is_file():",
                '            (OUTPUT_DIR / f"FINAL_STATUS_{sess}.json").write_bytes(st_path.read_bytes())',
                '        if receipt.get("status") == "COMPLETE":',
                "            completed_sessions.append(sess)",
                '        elif receipt.get("status") == "PARTIAL":',
                "            partial_sessions.append(sess)",
                "            deferred_sessions.extend(SESSIONS[idx + 1:])",
                "            break",
                "        else:",
                '            failed_sessions.append({"session": sess, "receipt": receipt})',
                "            break",
                "",
                "        # Free /tmp uncompressed checkpoints after verified archive creation in /kaggle/working",
                '        for sub in ("checkpoints", "jobs", "shared_backbones", "worker_slots"):',
                "            shutil.rmtree(sess_dir / sub, ignore_errors=True)",
                "",
                "    program_status = {",
                '        "schema": "elendril-x1-program-status-v1",',
                '        "source_commit": SOURCE_SHA,',
                '        "requested_sessions": list(SESSIONS),',
                '        "completed_sessions": completed_sessions,',
                '        "partial_sessions": partial_sessions,',
                '        "deferred_sessions": deferred_sessions,',
                '        "failed_sessions": failed_sessions,',
                '        "elapsed_seconds": round(time.time() - notebook_t0, 2),',
                "    }",
                '    (OUTPUT_DIR / "PROGRAM_STATUS.json").write_text(json.dumps(program_status, indent=2, sort_keys=True) + "\\n", encoding="utf-8")',
                '    print("Program summary:", json.dumps(program_status, indent=2))',
            ],
            "x1-run",
        ),
        _cell_code(
            [
                "import json, sys",
                "if not OWNER_AUTHORIZED:",
                '    print("NOT LAUNCHED: set OWNER_AUTHORIZED = True")',
                "else:",
                "    if str(REPO_DIR) not in sys.path:",
                "        sys.path.insert(0, str(REPO_DIR))",
                "    from IPython.display import FileLink, display",
                "    from tools.elendril_v8x_export import analysis_zip_name, full_zip_name, verify_x1_export",
                "",
                '    prog = json.loads((OUTPUT_DIR / "PROGRAM_STATUS.json").read_text(encoding="utf-8"))',
                '    ran = [*prog.get("completed_sessions", []), *prog.get("partial_sessions", [])]',
                "    verified_receipts = {}",
                "    for sess in ran:",
                "        receipt = verify_x1_export(OUTPUT_DIR, session=sess, expected_source_commit=SOURCE_SHA)",
                "        verified_receipts[sess] = receipt",
                '        for fname in (f"FINAL_STATUS_{sess}.json", analysis_zip_name(sess), full_zip_name(sess)):',
                "            if (OUTPUT_DIR / fname).exists():",
                "                display(FileLink(str(OUTPUT_DIR / fname)))",
                '    if (OUTPUT_DIR / "PROGRAM_STATUS.json").exists():',
                '        display(FileLink(str(OUTPUT_DIR / "PROGRAM_STATUS.json")))',
                '    print("Verified session exports:", json.dumps(verified_receipts, indent=2))',
                '    if not ran or prog.get("failed_sessions"):',
                '        raise RuntimeError(f"ELENDRIL V8-X1 run did not succeed cleanly: {prog}")',
            ],
            "x1-verify-export",
        ),
    ]

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "elendril_v8x_source_commit": source_commit,
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write_notebook(
    output_path: Path,
    *,
    source_commit: str,
    repo_url: str = REPO_HTTP_URL,
    owner_authorized_default: bool = True,
) -> Path:
    nb = build_notebook_dict(
        source_commit=source_commit,
        repo_url=repo_url,
        owner_authorized_default=owner_authorized_default,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(nb, indent=2) + "\n")
    return output_path


def verify_notebook_file(notebook_path: Path, *, expected_source_commit: str | None = None) -> dict[str, object]:
    """Validate owner-armed defaults, pinned commit, AST compilation of all code cells, and multi-session exports."""

    nb = json.loads(notebook_path.read_text(encoding="utf-8"))
    meta_sha = nb.get("metadata", {}).get("elendril_v8x_source_commit", "")
    cells = nb.get("cells", [])
    code_sources = ["".join(c.get("source", [])) for c in cells if c.get("cell_type") == "code"]
    for idx, src in enumerate(code_sources):
        compile(src, f"<cell-{idx}>", "exec")

    all_src = "\n".join(code_sources)
    for needle in (
        "OWNER_AUTHORIZED = True",
        'SESSIONS = ["A0", "A1", "A2", "A3", "D0", "D1", "D2"]',
        "MINUTES = 480",
        "NOT LAUNCHED:",
        "tools.elendril_v8x_campaign",
        "verify_x1_export",
        "PROGRAM_STATUS.json",
    ):
        if needle not in all_src:
            raise ValueError(f"X1 notebook missing required token: {needle!r}")

    verify_cells = ["".join(c.get("source", [])) for c in cells if c.get("id") == "x1-verify-export"]
    if len(verify_cells) != 1:
        raise ValueError("notebook must contain exactly one x1-verify-export cell")
    vsrc = verify_cells[0]
    if vsrc.find("if not OWNER_AUTHORIZED:") > vsrc.find("from tools.elendril_v8x_export"):
        raise ValueError("x1-verify-export must guard tools.elendril_v8x_export import behind OWNER_AUTHORIZED")

    if len(meta_sha) != 40 or f'SOURCE_SHA = "{meta_sha}"' not in all_src:
        raise ValueError("notebook lacks valid pinned 40-hex SOURCE_SHA")
    if expected_source_commit is not None and meta_sha != expected_source_commit:
        raise ValueError(f"notebook source commit {meta_sha} != {expected_source_commit}")
    return {"verified": True, "source_commit": meta_sha, "cells": len(cells), "code_cells": len(code_sources)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build or verify ELENDRIL V8-X1 Kaggle 2xT4 notebook")
    parser.add_argument("--source", default=None, help="40-hex runtime Git commit SHA")
    parser.add_argument("--output", type=Path, default=DEFAULT_NOTEBOOK_PATH)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    source = args.source
    if source is None:
        source = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()

    if args.check:
        res = verify_notebook_file(args.output, expected_source_commit=source)
        print(json.dumps(res, indent=2, sort_keys=True))
        return 0

    out = write_notebook(args.output, source_commit=source)
    res = verify_notebook_file(out, expected_source_commit=source)
    print(json.dumps({"written": str(out), **res}, indent=2, sort_keys=True))
    return 0


__all__ = ["build_notebook_dict", "verify_notebook_file", "write_notebook"]

if __name__ == "__main__":
    raise SystemExit(main())
