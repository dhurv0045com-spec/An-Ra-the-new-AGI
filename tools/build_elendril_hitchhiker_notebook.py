"""Build the source-pinned Kaggle T4x2 Hitchhiker experiment notebook."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess
import sys

from tools.build_elendril_v715_notebook import cell


REPOSITORY_URL = "https://github.com/dhurv0045com-spec/An-Ra-the-new-AGI.git"


NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "ELENDRIL_HITCHHIKER_KAGGLE_T4X2.ipynb"


def build(source: str) -> dict:
    """Build the reproducible source-pinned Kaggle T4x2 Hitchhiker campaign notebook."""
    if re.fullmatch(r"[0-9a-f]{40}", source) is None:
        raise ValueError("notebook builder requires an explicit 40-char git commit SHA")

    header = cell(
        "markdown",
        f"""# Elendril Hitchhiker -- Kaggle T4x2 Source-Pinned Experiment

- Exact pinned commit: `{source}`
- Phases supported:
  - `stage0`: Historical `V7.17` `S20-low-lr` frozen checkpoint audit (`--stage0-checkpoint-root`, if mounted).
  - `stage1`: 15-run `S3` dev screening (`B0, A, B, AB, C` x seeds `7, 19, 41`), locking dev `selection_receipt.json`.
  - `stage2`: Conditional follow-ups (`AB+C`, `C+penalty`, `D-external`, `S20-contrast`).
  - `stage3`: Locked-winner `S8`/`S20` scale & formation confirmation on independent `confirmation` + structural holdouts.
- Output directory: `/kaggle/working/elendril_hitchhiker_run`
- Verified deliverable bundles:
  - `exports/ELENDRIL_HITCHHIKER_ANALYSIS.zip`
  - `exports/ELENDRIL_HITCHHIKER_FULL.zip`
  - `exports/FINAL_STATUS.json`

Archive integrity proves byte fidelity, not scientific capability. Only `gate.status == 'FORMATION_QUALIFIED'` on `confirmation` qualifies formation.""",
    )

    setup = cell(
        "code",
        f"""import json, os, shutil, subprocess, tarfile
from pathlib import Path

OWNER_ENABLED = True
PHASE = "stage1"  # 'stage0', 'stage1', 'stage2', or 'stage3'
MINUTES_BUDGET = 540
SOURCE_COMMIT = "{source}"
REPO_URL = "{REPOSITORY_URL}"

STAGE0_CHECKPOINT_ROOT = None  # e.g. Path("/kaggle/input/elendril-v717-s20-checkpoints/checkpoints")
RESUME_ARCHIVE = None          # e.g. Path("/kaggle/input/elendril-hitchhiker-stage1/ELENDRIL_HITCHHIKER_FULL.zip")
WINNING_ARM = None             # required when running PHASE='stage3' directly from a restored Stage 1 bundle
ADMITTED_FOLLOWUPS = []        # optional subset for PHASE='stage2', e.g. ["AB+C", "C+penalty"]

WORK_ROOT = Path("/kaggle/working/elendril_hitchhiker_run")
CODE_ROOT = Path("/kaggle/working/anra_src")
WORK_ROOT.mkdir(parents=True, exist_ok=True)
if CODE_ROOT.exists():
    shutil.rmtree(CODE_ROOT)
CODE_ROOT.mkdir(parents=True, exist_ok=True)

tar_path = Path("/kaggle/working/pinned_repo.tar")
subprocess.run(["git", "clone", "--no-checkout", REPO_URL, str(CODE_ROOT / ".git_mirror")], check=True)
subprocess.run(["git", "-C", str(CODE_ROOT / ".git_mirror"), "fetch", "--all", "--tags"], check=True)
resolved = subprocess.check_output(["git", "-C", str(CODE_ROOT / ".git_mirror"), "rev-parse", SOURCE_COMMIT], text=True).strip()
assert resolved == SOURCE_COMMIT, f"commit mismatch: {{resolved}} != {{SOURCE_COMMIT}}"
subprocess.run(["git", "-C", str(CODE_ROOT / ".git_mirror"), "archive", "--format=tar", f"--output={{tar_path}}", SOURCE_COMMIT], check=True)
with tarfile.open(tar_path) as tf:
    tf.extractall(CODE_ROOT)
os.chdir(CODE_ROOT)
print("Verified clean source extract at commit", SOURCE_COMMIT)
""",
    )

    run_cell = cell(
        "code",
        f"""cmd = [
    "python", "-m", "tools.elendril_hitchhiker_campaign",
    "--output", str(WORK_ROOT),
    "--source", SOURCE_COMMIT,
    "--phase", PHASE,
    "--minutes", str(MINUTES_BUDGET),
]
if OWNER_ENABLED:
    cmd.append("--owner-enabled")
if STAGE0_CHECKPOINT_ROOT is not None:
    cmd.extend(["--stage0-checkpoint-root", str(STAGE0_CHECKPOINT_ROOT)])
if RESUME_ARCHIVE is not None:
    cmd.extend(["--resume-archive", str(RESUME_ARCHIVE)])
if WINNING_ARM is not None:
    cmd.extend(["--winning-arm", str(WINNING_ARM)])
if ADMITTED_FOLLOWUPS:
    cmd.extend(["--admitted-followups", *ADMITTED_FOLLOWUPS])

print("Launching:", " ".join(cmd))
subprocess.run(cmd, cwd=CODE_ROOT, check=True)
final_status = json.loads((WORK_ROOT / "exports" / "FINAL_STATUS.json").read_text())
print(json.dumps({{
    "status": final_status["status"],
    "claim_status": final_status.get("claim_status"),
    "phase": final_status.get("phase"),
    "replay": final_status.get("independent_replay", {{}}).get("status"),
    "errors": final_status.get("errors"),
}}, indent=2))
""",
    )

    links = cell(
        "code",
        """from IPython.display import FileLink, display

for name in ("ELENDRIL_HITCHHIKER_ANALYSIS.zip", "ELENDRIL_HITCHHIKER_FULL.zip", "FINAL_STATUS.json"):
    target = WORK_ROOT / "exports" / name
    if target.exists():
        display(FileLink(str(target)))
""",
    )

    return {
        "cells": [header, setup, run_cell, links],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11.0"},
            "source_reference": source,
            "hitchhiker_notebook": True,
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def _default_repo_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
    ).strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=None, help="Pinned 40-char git commit SHA.")
    parser.add_argument("--output", type=Path, default=NOTEBOOK)
    args = parser.parse_args(argv)
    source = args.source or _default_repo_head()
    nb = build(source)
    raw = (json.dumps(nb, indent=2) + "\n").encode("utf-8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(raw)
    print(json.dumps({"notebook": str(args.output), "source": source, "sha256": sha256(raw).hexdigest()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
