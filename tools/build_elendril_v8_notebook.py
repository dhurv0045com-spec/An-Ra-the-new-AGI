"""Generate and verify the Kaggle 2xT4 notebook pinned to an immutable runtime Git commit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

REPO_HTTP_URL = "https://github.com/AnkitkumarSoni/An-Ra-the-new-AGI"
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
                "- Defaults are inert (`RUN_STUDY = False`) until explicitly enabled by the owner.",
            ],
            "v8-title",
        ),
        _cell_code(
            [
                "# Owner-controlled switches (keep RUN_STUDY=False until ready to run on Kaggle 2xT4)",
                "RUN_STUDY = False",
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
                "import os, shutil, subprocess, tarfile, urllib.request",
                "from pathlib import Path",
                "",
                "WORK_ROOT = Path(\"/kaggle/working/elendril_v8\")",
                "REPO_DIR = WORK_ROOT / \"repo\"",
                "OUTPUT_DIR = Path(\"/kaggle/working\")",
                "WORK_ROOT.mkdir(parents=True, exist_ok=True)",
                "",
                "if not RUN_STUDY:",
                "    print(\"NOT LAUNCHED: owner switch is false\")",
                "else:",
                "    if REPO_DIR.exists():",
                "        shutil.rmtree(REPO_DIR)",
                "    REPO_DIR.mkdir(parents=True, exist_ok=True)",
                "    local_tar = Path(f\"/kaggle/input/elendril-v8-source/{SOURCE_SHA}.tar.gz\")",
                "    tar_path = WORK_ROOT / f\"{SOURCE_SHA}.tar.gz\"",
                "    if local_tar.exists():",
                "        shutil.copy2(local_tar, tar_path)",
                "    else:",
                "        subprocess.run([\"git\", \"clone\", \"--no-checkout\", REPO_URL, str(REPO_DIR)], check=True)",
                "        subprocess.run([\"git\", \"-C\", str(REPO_DIR), \"fetch\", \"origin\", SOURCE_SHA], check=True)",
                "        subprocess.run([\"git\", \"-C\", str(REPO_DIR), \"checkout\", \"--detach\", SOURCE_SHA], check=True)",
                "    if tar_path.exists():",
                "        with tarfile.open(tar_path, \"r:gz\") as tf:",
                "            tf.extractall(REPO_DIR)",
                "    os.chdir(REPO_DIR)",
                "    print(\"Checked out immutable runtime:\", SOURCE_SHA)",
            ],
            "v8-checkout",
        ),
        _cell_code(
            [
                "import torch",
                "if RUN_STUDY:",
                "    print(\"Torch:\", torch.__version__, \"CUDA GPUs:\", torch.cuda.device_count())",
                "    for idx in range(torch.cuda.device_count()):",
                "        print(f\"  cuda:{idx} -> {torch.cuda.get_device_name(idx)}\")",
                "    assert torch.cuda.is_available(), \"Kaggle GPU accelerator must be enabled\"",
            ],
            "v8-gpu-check",
        ),
        _cell_code(
            [
                "import sys, subprocess",
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
                "    subprocess.run(cmd, check=True)",
            ],
            "v8-run",
        ),
        _cell_code(
            [
                "import json",
                "from IPython.display import FileLink, display",
                "from tools.elendril_v8_export import verify_v8_export",
                "",
                "if not RUN_STUDY:",
                "    print(\"NOT LAUNCHED: owner switch is false\")",
                "else:",
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
    output_path.write_text(json.dumps(nb, indent=2) + "\n", encoding="utf-8")
    return output_path


def verify_notebook_file(notebook_path: Path, *, expected_source_commit: str | None = None) -> dict[str, object]:
    """Validate inert defaults, pinned commit, no launch-when-false bug, and V8 archive names."""

    nb = json.loads(notebook_path.read_text(encoding="utf-8"))
    meta_sha = nb.get("metadata", {}).get("elendril_v8_source_commit", "")
    all_src = "\n".join("".join(cell.get("source", [])) for cell in nb.get("cells", []))

    for needle in (
        "RUN_STUDY = False",
        'PHASE = "data-control"',
        "MINUTES = 330",
        "NOT LAUNCHED: owner switch is false",
        "tools.elendril_v8_campaign",
        "ELENDRIL_V8_ANALYSIS.zip",
        "ELENDRIL_V8_FULL.zip",
        "FINAL_STATUS.json",
    ):
        if needle not in all_src:
            raise ValueError(f"notebook missing required token: {needle!r}")

    if len(meta_sha) != 40 or f'SOURCE_SHA = "{meta_sha}"' not in all_src:
        raise ValueError("notebook lacks valid pinned 40-hex SOURCE_SHA")
    if expected_source_commit is not None and meta_sha != expected_source_commit:
        raise ValueError(f"notebook source commit {meta_sha} != {expected_source_commit}")
    return {"verified": True, "source_commit": meta_sha, "cells": len(nb.get("cells", []))}


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
