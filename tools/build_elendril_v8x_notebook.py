"""Generate and verify the deterministic Kaggle 2xT4 notebook for ELENDRIL V8-X1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

REPO_HTTP_URL = "https://github.com/dhurv0045com-spec/An-Ra-the-new-AGI.git"
DEFAULT_NOTEBOOK_PATH = Path("notebooks/ELENDRIL_V8_X1_KAGGLE_T4X2.ipynb")


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
    owner_authorized_default: bool = False,
) -> dict[str, object]:
    """Build deterministic nbformat v4.5 JSON for ELENDRIL V8-X1 on Kaggle 2xT4."""

    if len(source_commit) != 40 or any(c not in "0123456789abcdef" for c in source_commit):
        raise ValueError("source_commit must be a 40-character lowercase hex SHA")

    auth_literal = "True" if owner_authorized_default else "False"
    cells = [
        _cell_md(
            [
                "# ELENDRIL V8-X1 — Relational Evidence Reader & D2 Paired Screen (Kaggle 2xT4)",
                "",
                "Execution notebook pinned to immutable runtime commit:",
                f"- `SOURCE_SHA = \"{source_commit}\"`",
                "- Sessions: `A0` (P vs R), `A1` (F vs F-cap), `A2` (R-noedge vs P-capR), `A3` (B0 vs J), `D0` (PAIR-CE), `D1` (PAIR-EQ), `D2` (PAIR-EQ-CF)",
                "- Change **only `OWNER_AUTHORIZED = True`** when launching an authorized 2xT4 cloud run.",
            ],
            "x1-title",
        ),
        _cell_code(
            [
                "# Owner-controlled switches (keep OWNER_AUTHORIZED = False during dry-run/CI; set True to run 2xT4)",
                f"OWNER_AUTHORIZED = {auth_literal}",
                "SESSION = \"A0\"  # A0 | A1 | A2 | A3 | D0 | D1 | D2 | U0 | U1",
                "MINUTES = 330",
                "DEVICES = \"0,1\"",
                "RESUME = False",
                "OPTIONAL_INPUT_DIR = \"\"",
                f"SOURCE_SHA = \"{source_commit}\"",
                f"REPO_URL = \"{repo_url}\"",
            ],
            "x1-config",
        ),
        _cell_code(
            [
                "import os, shutil, subprocess, sys, tarfile",
                "from pathlib import Path",
                "",
                "os.environ[\"CUBLAS_WORKSPACE_CONFIG\"] = \":4096:8\"",
                "os.environ[\"PYTHONDONTWRITEBYTECODE\"] = \"1\"",
                "os.environ[\"PYTHONUNBUFFERED\"] = \"1\"",
                "os.environ[\"OMP_NUM_THREADS\"] = \"2\"",
                "os.environ[\"MKL_NUM_THREADS\"] = \"2\"",
                "",
                "WORK_ROOT = Path(\"/tmp/elendril_x1\")",
                "REPO_DIR = WORK_ROOT / \"repo\"",
                "RUN_DIR = WORK_ROOT / f\"session_{SESSION}\"",
                "OUTPUT_DIR = Path(\"/kaggle/working\")",
                "",
                "if not OWNER_AUTHORIZED:",
                "    print(\"NOT LAUNCHED: set OWNER_AUTHORIZED = True in the configuration cell to execute X1.\")",
                "else:",
                "    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)",
                "    WORK_ROOT.mkdir(parents=True, exist_ok=True)",
                "    if REPO_DIR.exists():",
                "        shutil.rmtree(REPO_DIR)",
                "    REPO_DIR.mkdir(parents=True, exist_ok=True)",
                "    local_tar = Path(f\"/kaggle/input/elendril-x1-source/{SOURCE_SHA}.tar.gz\")",
                "    if local_tar.exists():",
                "        tar_path = WORK_ROOT / f\"{SOURCE_SHA}.tar.gz\"",
                "        shutil.copy2(local_tar, tar_path)",
                "        with tarfile.open(tar_path, \"r:gz\") as tf:",
                "            tf.extractall(REPO_DIR)",
                "    else:",
                "        subprocess.run([\"git\", \"-C\", str(REPO_DIR), \"init\", \"-q\"], check=True)",
                "        subprocess.run([\"git\", \"-C\", str(REPO_DIR), \"remote\", \"add\", \"origin\", REPO_URL], check=True)",
                "        try:",
                "            subprocess.run([\"git\", \"-C\", str(REPO_DIR), \"fetch\", \"--depth=1\", \"origin\", SOURCE_SHA], check=True)",
                "        except subprocess.CalledProcessError:",
                "            subprocess.run([\"git\", \"-C\", str(REPO_DIR), \"fetch\", \"origin\", SOURCE_SHA], check=True)",
                "        subprocess.run([\"git\", \"-C\", str(REPO_DIR), \"checkout\", \"--detach\", SOURCE_SHA], check=True)",
                "    os.chdir(REPO_DIR)",
                "    if str(REPO_DIR) not in sys.path:",
                "        sys.path.insert(0, str(REPO_DIR))",
                "    prev_pp = os.environ.get(\"PYTHONPATH\", \"\")",
                "    pp_parts = [p for p in prev_pp.split(os.pathsep) if p] if prev_pp else []",
                "    if str(REPO_DIR) not in pp_parts:",
                "        os.environ[\"PYTHONPATH\"] = os.pathsep.join([str(REPO_DIR), *pp_parts]) if pp_parts else str(REPO_DIR)",
                "    print(\"Checked out immutable X1 runtime:\", SOURCE_SHA)",
            ],
            "x1-checkout",
        ),
        _cell_code(
            [
                "import torch",
                "if not OWNER_AUTHORIZED:",
                "    print(\"NOT LAUNCHED: set OWNER_AUTHORIZED = True\")",
                "else:",
                "    gpu_count = torch.cuda.device_count()",
                "    print(\"Torch:\", torch.__version__, \"CUDA GPUs:\", gpu_count)",
                "    for idx in range(gpu_count):",
                "        print(f\"  cuda:{idx} -> {torch.cuda.get_device_name(idx)}\")",
                "    assert torch.cuda.is_available() and gpu_count >= 2, f\"2xT4 CUDA GPUs required, found {gpu_count}\"",
            ],
            "x1-gpu-check",
        ),
        _cell_code(
            [
                "import os, subprocess, sys",
                "if not OWNER_AUTHORIZED:",
                "    print(\"NOT LAUNCHED: set OWNER_AUTHORIZED = True\")",
                "else:",
                "    env = dict(os.environ)",
                "    env[\"CUBLAS_WORKSPACE_CONFIG\"] = \":4096:8\"",
                "    env[\"PYTHONDONTWRITEBYTECODE\"] = \"1\"",
                "    env[\"PYTHONUNBUFFERED\"] = \"1\"",
                "    env[\"OMP_NUM_THREADS\"] = \"2\"",
                "    env[\"MKL_NUM_THREADS\"] = \"2\"",
                "    common = [\"--source\", SOURCE_SHA, \"--session\", SESSION, \"--work-dir\", str(RUN_DIR)]",
                "    for sub_args in (",
                "        [\"prepare\", *common],",
                "        [\"validate\", *common, \"--cpu-fixtures\"],",
                "        [\"admit\", *common, \"--devices\", DEVICES, \"--max-minutes\", str(MINUTES)],",
                "    ):",
                "        subprocess.run([sys.executable, \"-m\", \"tools.elendril_v8x_campaign\", *sub_args], env=env, cwd=str(REPO_DIR), check=True)",
                "    run_cmd = [",
                "        sys.executable, \"-m\", \"tools.elendril_v8x_campaign\",",
                "        \"resume\" if RESUME else \"run\",",
                "        *common,",
                "        \"--devices\", DEVICES,",
                "        \"--max-minutes\", str(MINUTES),",
                "        \"--output-dir\", str(OUTPUT_DIR),",
                "        \"--owner-authorized\",",
                "    ]",
                "    subprocess.run(run_cmd, env=env, cwd=str(REPO_DIR), check=True)",
            ],
            "x1-run",
        ),
        _cell_code(
            [
                "import json, sys",
                "if not OWNER_AUTHORIZED:",
                "    print(\"NOT LAUNCHED: set OWNER_AUTHORIZED = True\")",
                "else:",
                "    if str(REPO_DIR) not in sys.path:",
                "        sys.path.insert(0, str(REPO_DIR))",
                "    from IPython.display import FileLink, display",
                "    from tools.elendril_v8x_export import analysis_zip_name, full_zip_name, verify_x1_export",
                "",
                "    receipt = verify_x1_export(OUTPUT_DIR, session=SESSION, expected_source_commit=SOURCE_SHA)",
                "    print(\"Verified X1 exports:\", json.dumps(receipt, indent=2))",
                "    for fname in (\"FINAL_STATUS.json\", analysis_zip_name(SESSION), full_zip_name(SESSION)):",
                "        if (OUTPUT_DIR / fname).exists():",
                "            display(FileLink(str(OUTPUT_DIR / fname)))",
                "    if receipt.get(\"status\") not in (\"COMPLETE\", \"PARTIAL\"):",
                "        raise RuntimeError(f\"ELENDRIL V8-X1 did not complete training/evaluation: {receipt}\")",
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
    owner_authorized_default: bool = False,
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
    """Validate disabled default, pinned commit, AST compilation of all code cells, and X1 archive names."""

    nb = json.loads(notebook_path.read_text(encoding="utf-8"))
    meta_sha = nb.get("metadata", {}).get("elendril_v8x_source_commit", "")
    cells = nb.get("cells", [])
    code_sources = ["".join(c.get("source", [])) for c in cells if c.get("cell_type") == "code"]
    for idx, src in enumerate(code_sources):
        compile(src, f"<cell-{idx}>", "exec")

    all_src = "\n".join(code_sources)
    for needle in (
        "OWNER_AUTHORIZED = False",
        'SESSION = "A0"',
        "MINUTES = 330",
        "NOT LAUNCHED:",
        "tools.elendril_v8x_campaign",
        "verify_x1_export",
        "FINAL_STATUS.json",
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
