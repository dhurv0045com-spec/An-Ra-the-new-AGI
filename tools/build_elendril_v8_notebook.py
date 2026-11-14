"""Generate and verify the Kaggle 2xT4 notebook pinned to an immutable runtime Git commit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

REPO_HTTP_URL = "https://github.com/dhurv0045com-spec/An-Ra-the-new-AGI.git"
DEFAULT_NOTEBOOK_PATH = Path("notebooks/ELENDRIL_V8_KAGGLE_T4X2.ipynb")


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


def build_notebook_dict(*, source_commit: str, repo_url: str = REPO_HTTP_URL) -> dict[str, object]:
    """Build deterministic nbformat v4.5 JSON for ELENDRIL V8 Kaggle 2xT4 execution."""

    if len(source_commit) != 40 or any(c not in "0123456789abcdef" for c in source_commit):
        raise ValueError("source_commit must be a 40-character lowercase hex SHA")

    cells = [
        _cell_md(
            [
                "# ELENDRIL V8 — Typed Action / Fact-Pointer Screen (Kaggle 2xT4)",
                "",
                "Execution notebook pinned to immutable runtime commit:",
                f"- `SOURCE_SHA = \"{source_commit}\"`",
                "- Owner-armed defaults (`RUN_STUDY = True`) for immediate Kaggle 2xT4 execution (set `RUN_STUDY = False` for inert dry-run).",
            ],
            "v8-title",
        ),
        _cell_code(
            [
                "# Owner-controlled switches (RUN_STUDY=True is owner-armed for Kaggle 2xT4; set False for inert dry-run)",
                "RUN_STUDY = True",
                "PHASE = \"data-control\"  # archive-diagnostic | data-control | head-screen | reader-screen | replication",
                "MINUTES = 330",
                "WINNER_ARM = None",
                "COMPARATOR_ARM = None",
                "CONFIRMATION_SEED = None",
                f"SOURCE_SHA = \"{source_commit}\"",
                f"REPO_URL = \"{repo_url}\"",
            ],
            "v8-config",
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
                "WORK_ROOT = Path(\"/kaggle/working/elendril_v8\")",
                "REPO_DIR = WORK_ROOT / \"repo\"",
                "OUTPUT_DIR = Path(\"/kaggle/working\")",
                "",
                "if not RUN_STUDY:",
                "    print(\"NOT LAUNCHED: owner switch is false\")",
                "else:",
                "    WORK_ROOT.mkdir(parents=True, exist_ok=True)",
                "    if REPO_DIR.exists():",
                "        shutil.rmtree(REPO_DIR)",
                "    REPO_DIR.mkdir(parents=True, exist_ok=True)",
                "    local_tar = Path(f\"/kaggle/input/elendril-v8-source/{SOURCE_SHA}.tar.gz\")",
                "    if local_tar.exists():",
                "        tar_path = WORK_ROOT / f\"{SOURCE_SHA}.tar.gz\"",
                "        shutil.copy2(local_tar, tar_path)",
                "        with tarfile.open(tar_path, \"r:gz\") as tf:",
                "            tf.extractall(REPO_DIR)",
                "    else:",
                "        subprocess.run([\"git\", \"-C\", str(REPO_DIR), \"init\", \"-q\"], check=True)",
                "        remotes = subprocess.run(",
                "            [\"git\", \"-C\", str(REPO_DIR), \"remote\"],",
                "            check=True,",
                "            capture_output=True,",
                "            text=True,",
                "        ).stdout.split()",
                "        if \"origin\" in remotes:",
                "            subprocess.run([\"git\", \"-C\", str(REPO_DIR), \"remote\", \"set-url\", \"origin\", REPO_URL], check=True)",
                "        else:",
                "            subprocess.run([\"git\", \"-C\", str(REPO_DIR), \"remote\", \"add\", \"origin\", REPO_URL], check=True)",
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
                "    print(\"Checked out immutable runtime:\", SOURCE_SHA)",
            ],
            "v8-checkout",
        ),
        _cell_code(
            [
                "import torch",
                "if RUN_STUDY:",
                "    gpu_count = torch.cuda.device_count()",
                "    print(\"Torch:\", torch.__version__, \"CUDA GPUs:\", gpu_count)",
                "    for idx in range(gpu_count):",
                "        print(f\"  cuda:{idx} -> {torch.cuda.get_device_name(idx)}\")",
                "    assert torch.cuda.is_available(), \"Kaggle GPU accelerator must be enabled\"",
                "    if gpu_count < 2:",
                "        print(f\"WARNING: expected 2xT4 GPUs, found {gpu_count} CUDA device(s); continuing on available GPU(s)\")",
            ],
            "v8-gpu-check",
        ),
        _cell_code(
            [
                "import os, subprocess, sys",
                "if not RUN_STUDY:",
                "    print(\"NOT LAUNCHED: owner switch is false\")",
                "else:",
                "    cmd = [",
                "        sys.executable, \"-m\", \"tools.elendril_v8_campaign\",",
                "        \"--phase\", PHASE,",
                "        \"--source\", SOURCE_SHA,",
                "        \"--work-root\", str(WORK_ROOT / \"runs\"),",
                "        \"--output\", str(OUTPUT_DIR),",
                "        \"--minutes\", str(MINUTES),",
                "        \"--owner-enabled\",",
                "    ]",
                "    if WINNER_ARM:",
                "        cmd.extend([\"--winner-arm\", str(WINNER_ARM)])",
                "    if COMPARATOR_ARM:",
                "        cmd.extend([\"--comparator-arm\", str(COMPARATOR_ARM)])",
                "    if CONFIRMATION_SEED is not None:",
                "        cmd.extend([\"--confirmation-seed\", str(CONFIRMATION_SEED)])",
                "    env = dict(os.environ)",
                "    env[\"CUBLAS_WORKSPACE_CONFIG\"] = \":4096:8\"",
                "    env[\"PYTHONDONTWRITEBYTECODE\"] = \"1\"",
                "    env[\"PYTHONUNBUFFERED\"] = \"1\"",
                "    env[\"OMP_NUM_THREADS\"] = \"2\"",
                "    env[\"MKL_NUM_THREADS\"] = \"2\"",
                "    prev_pp = env.get(\"PYTHONPATH\", \"\")",
                "    pp_parts = [p for p in prev_pp.split(os.pathsep) if p] if prev_pp else []",
                "    if str(REPO_DIR) not in pp_parts:",
                "        env[\"PYTHONPATH\"] = os.pathsep.join([str(REPO_DIR), *pp_parts]) if pp_parts else str(REPO_DIR)",
                "    subprocess.run(cmd, env=env, cwd=str(REPO_DIR), check=True)",
            ],
            "v8-run",
        ),
        _cell_code(
            [
                "import json, sys",
                "",
                "if not RUN_STUDY:",
                "    print(\"NOT LAUNCHED: owner switch is false\")",
                "else:",
                "    if str(REPO_DIR) not in sys.path:",
                "        sys.path.insert(0, str(REPO_DIR))",
                "    from IPython.display import FileLink, display",
                "    from tools.elendril_v8_export import verify_v8_export",
                "",
                "    receipt = verify_v8_export(OUTPUT_DIR, expected_source_commit=SOURCE_SHA)",
                "    print(\"Verified exports:\", json.dumps(receipt, indent=2))",
                "    for fname in (\"FINAL_STATUS.json\", \"ELENDRIL_V8_ANALYSIS.zip\", \"ELENDRIL_V8_FULL.zip\"):",
                "        if (OUTPUT_DIR / fname).exists():",
                "            display(FileLink(str(OUTPUT_DIR / fname)))",
            ],
            "v8-verify-export",
        ),
    ]

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "elendril_v8_source_commit": source_commit,
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write_notebook(output_path: Path, *, source_commit: str, repo_url: str = REPO_HTTP_URL) -> Path:
    nb = build_notebook_dict(source_commit=source_commit, repo_url=repo_url)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(nb, indent=2) + "\n")
    return output_path


def verify_notebook_file(notebook_path: Path, *, expected_source_commit: str | None = None) -> dict[str, object]:
    """Validate owner-armed defaults, pinned commit, no launch-when-false bug, and V8 archive names."""

    nb = json.loads(notebook_path.read_text(encoding="utf-8"))
    meta_sha = nb.get("metadata", {}).get("elendril_v8_source_commit", "")
    cells = nb.get("cells", [])
    all_src = "\n".join("".join(cell.get("source", [])) for cell in cells)

    for needle in (
        "RUN_STUDY = True",
        'PHASE = "data-control"',
        "MINUTES = 330",
        "CUBLAS_WORKSPACE_CONFIG",
        "PYTHONUNBUFFERED",
        "NOT LAUNCHED: owner switch is false",
        "tools.elendril_v8_campaign",
        "ELENDRIL_V8_ANALYSIS.zip",
        "ELENDRIL_V8_FULL.zip",
        "FINAL_STATUS.json",
    ):
        if needle not in all_src:
            raise ValueError(f"notebook missing required token: {needle!r}")

    verify_cells = ["".join(cell.get("source", [])) for cell in cells if cell.get("id") == "v8-verify-export"]
    if len(verify_cells) != 1:
        raise ValueError("notebook must contain exactly one v8-verify-export cell")
    verify_src = verify_cells[0]
    guard_pos = verify_src.find("if not RUN_STUDY:")
    import_pos = verify_src.find("from tools.elendril_v8_export import verify_v8_export")
    if guard_pos < 0 or import_pos < 0 or import_pos < guard_pos:
        raise ValueError("v8-verify-export must guard tools.elendril_v8_export import after 'if not RUN_STUDY:'")

    if len(meta_sha) != 40 or f'SOURCE_SHA = "{meta_sha}"' not in all_src:
        raise ValueError("notebook lacks valid pinned 40-hex SOURCE_SHA")
    if expected_source_commit is not None and meta_sha != expected_source_commit:
        raise ValueError(f"notebook source commit {meta_sha} != {expected_source_commit}")
    return {"verified": True, "source_commit": meta_sha, "cells": len(cells)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build or verify ELENDRIL V8 Kaggle notebook")
    parser.add_argument("--source", default=None, help="40-hex runtime Git commit SHA")
    parser.add_argument("--output", type=Path, default=DEFAULT_NOTEBOOK_PATH)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    source = args.source
    if source is None:
        source = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()

    if args.check:
        res = verify_notebook_file(args.output, expected_source_commit=args.source)
        print(json.dumps(res, indent=2))
        return 0

    out = write_notebook(args.output, source_commit=source)
    res = verify_notebook_file(out, expected_source_commit=source)
    print(json.dumps({"written": str(out), **res}, indent=2))
    return 0


__all__ = ["build_notebook_dict", "verify_notebook_file", "write_notebook"]

if __name__ == "__main__":
    raise SystemExit(main())
