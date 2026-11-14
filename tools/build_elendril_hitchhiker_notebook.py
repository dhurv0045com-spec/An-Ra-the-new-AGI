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
- Phases & readiness:
  - `stage0`: Historical `V7.17` `S20-low-lr` frozen checkpoint audit (**requires mounted `V7.17` `S20-low-lr` weights** via `STAGE0_CHECKPOINT_ROOT`).
  - `stage1`: **Stage 1 ready after GPU preflight** -- 15-run `S3` dev screening (`B0, A, B, AB, C` x seeds `7, 19, 41`), locking dev `selection_receipt.json`.
  - `stage2`: **Requires completed Stage 1 handoff archive & `selection_receipt.json`** (`HANDOFF_ARCHIVE`) -- conditional follow-ups (`AB+C`, `C+penalty`, `D-external`, `S20-contrast`).
  - `stage3`: **Requires completed Stage 1 handoff archive & `selection_receipt.json`** (`HANDOFF_ARCHIVE`, with `WINNING_ARM` optional when `selection_receipt.json` supplies it) -- locked-winner `S8`/`S20` scale & formation confirmation on independent `confirmation` + structural holdouts.
- Output directory: `/kaggle/working/elendril_hitchhiker_<PHASE>`
- Verified deliverable bundles:
  - `exports/ELENDRIL_HITCHHIKER_ANALYSIS.zip`
  - `exports/ELENDRIL_HITCHHIKER_FULL.zip`
  - `exports/FINAL_STATUS.json`

Archive integrity proves byte fidelity, not scientific capability. Only `gate.status == 'FORMATION_QUALIFIED'` on `confirmation` qualifies formation.""",
    )

    setup = cell(
        "code",
        f"""import json, os, shutil, subprocess, sys
from pathlib import Path

OWNER_ENABLED = True
PHASE = "stage1"  # 'stage0', 'stage1', 'stage2', or 'stage3'
MINUTES_BUDGET = 480
SOURCE_COMMIT = "{source}"
REPO_URL = "{REPOSITORY_URL}"

STAGE0_CHECKPOINT_ROOT = None  # required for stage0: mounted V7.17 S20-low-lr weights
RESUME_ARCHIVE = None          # optional same-phase ELENDRIL_HITCHHIKER_FULL.zip for interrupted recovery
HANDOFF_ARCHIVE = None         # required for stage2/stage3 from completed Stage 1 (or Stage 2) ELENDRIL_HITCHHIKER_FULL.zip
WINNING_ARM = None             # optional for stage3 when selection_receipt.json supplies winning_arm
ADMITTED_FOLLOWUPS = []        # optional narrowing subset for stage2, e.g. ["AB+C", "D-external"]

assert PHASE in {{"stage0", "stage1", "stage2", "stage3"}}, f"Unsupported PHASE: {{PHASE!r}}"
WORK_ROOT = Path(f"/kaggle/working/elendril_hitchhiker_{{PHASE}}")
CODE_ROOT = Path("/kaggle/working/anra_src")
WORK_ROOT.mkdir(parents=True, exist_ok=True)
if CODE_ROOT.exists():
    shutil.rmtree(CODE_ROOT)

subprocess.run(["git", "clone", "--no-checkout", REPO_URL, str(CODE_ROOT)], check=True)
subprocess.run(["git", "-C", str(CODE_ROOT), "fetch", "--all", "--tags"], check=True)
resolved = subprocess.check_output(["git", "-C", str(CODE_ROOT), "rev-parse", SOURCE_COMMIT], text=True).strip()
assert resolved == SOURCE_COMMIT, f"commit mismatch: {{resolved}} != {{SOURCE_COMMIT}}"
subprocess.run(["git", "-C", str(CODE_ROOT), "checkout", "--detach", SOURCE_COMMIT], check=True)
os.chdir(CODE_ROOT)
print("Verified clean source checkout at commit", SOURCE_COMMIT, "Python:", sys.executable)
""",
    )

    run_cell = cell(
        "code",
        f"""cmd = [
    sys.executable, "-m", "tools.elendril_hitchhiker_campaign",
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
if HANDOFF_ARCHIVE is not None:
    cmd.extend(["--handoff-archive", str(HANDOFF_ARCHIVE)])
if WINNING_ARM:
    cmd.extend(["--winning-arm", str(WINNING_ARM)])
if ADMITTED_FOLLOWUPS:
    cmd.extend(["--admitted-followups", *ADMITTED_FOLLOWUPS])

env = {{**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}}
print("Launching:", " ".join(cmd))
proc = subprocess.run(cmd, cwd=CODE_ROOT, env=env)

status_file = WORK_ROOT / "exports" / "FINAL_STATUS.json"
if not status_file.exists():
    status_file = WORK_ROOT / "analysis" / "campaign_status.json"
if status_file.exists():
    final_status = json.loads(status_file.read_text(encoding="utf-8"))
    print(json.dumps({{
        "returncode": proc.returncode,
        "status": final_status.get("status"),
        "claim_status": final_status.get("claim_status"),
        "phase": final_status.get("phase"),
        "replay": final_status.get("independent_replay", {{}}).get("status"),
        "errors": final_status.get("errors"),
    }}, indent=2))

try:
    from IPython.display import FileLink, display
    for name in ("ELENDRIL_HITCHHIKER_ANALYSIS.zip", "ELENDRIL_HITCHHIKER_FULL.zip", "FINAL_STATUS.json"):
        target = WORK_ROOT / "exports" / name
        if target.exists():
            display(FileLink(str(target)))
except ImportError:
    pass

if proc.returncode != 0:
    raise RuntimeError(f"Hitchhiker campaign exited with code {{proc.returncode}}; inspect printed status and exports above.")
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
