"""Build the source-pinned AN-RA V8-Y1 Kaggle T4x2 campaign notebook."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from typing import Any


REPOSITORY_URL = "https://github.com/dhurv0045com-spec/An-Ra-the-new-AGI.git"
DEFAULT_NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "ELENDRIL_V8_Y1_KAGGLE_T4X2.ipynb"
EXPECTED_HH_ARCHIVE_SHA256 = "43a0c074cdcc3e12d8d00e0509663a2d47702586414317818ca0876579e487e2"
EXPECTED_HH_MANIFEST_SHA256 = "33f88e2e6473c245a7a1d6636db84ddd7ce316cd1fda7fc3eeb337d5618ad68d"


def _cell(kind: str, source: str) -> dict[str, Any]:
    lines = source.splitlines(keepends=True)
    base: dict[str, Any] = {"cell_type": kind, "metadata": {}, "source": lines}
    if kind == "code":
        base["execution_count"] = None
        base["outputs"] = []
    return base


def build_y1_notebook(source: str) -> dict[str, Any]:
    """Construct the source-pinned 4-cell Kaggle T4x2 Y1 notebook."""
    if len(source) != 40 or any(c not in "0123456789abcdef" for c in source):
        raise ValueError(f"source must be a 40-char lowercase git SHA: {source!r}")

    header = _cell(
        "markdown",
        f"""# AN-RA V8-Y1 -- Kaggle T4x2 Source-Pinned Experiment

- **Exact pinned commit**: `{source}`
- **Default phase policy (`PHASE = "auto"`)**:
  - Automatically checks `/kaggle/input/**` for the mounted verified historical archive `ELENDRIL_HITCHHIKER_FULL.zip` (`sha256: {EXPECTED_HH_ARCHIVE_SHA256[:16]}...`) and runs `diagnose` (`D0`/`D1` across all 15 Hitchhiker S3 checkpoints, 90m ceiling).
  - Fails clearly if `PHASE` is `"auto"` or `"diagnose"` and no verified historical ZIP is mounted (never silently switches to a training run).
  - Subsequent sessions:
    - Set `PHASE = "core"` and pass `HANDOFF_ARCHIVE = "/kaggle/input/.../ELENDRIL_V8_Y1_EVIDENCE.zip"` (or leave `HANDOFF_ARCHIVE = None` if reusing prepared inputs cleanly; 360m cumulative ceiling across resumes).
    - Set `PHASE = "followup"` and pass `HANDOFF_ARCHIVE = "/kaggle/input/.../ELENDRIL_V8_Y1_FULL.zip"` (+ `SELECTION_RECEIPT` if stored separately; 240m cumulative ceiling).
    - For interrupted recovery within any phase, set `RESUME_ARCHIVE = "/kaggle/input/.../ELENDRIL_V8_Y1_FULL.zip"`.
- **Output directory**: `/kaggle/working/elendril_y1_<PHASE>`
- **Exports emitted**:
  - `exports/ELENDRIL_V8_Y1_SUMMARY.zip` (< 5 MiB)
  - `exports/ELENDRIL_V8_Y1_EVIDENCE.zip` (deduplicated prompts + all traces + compressed `.npz` FP32 logits)
  - `exports/ELENDRIL_V8_Y1_FULL.zip` (verified resumable checkpoints when applicable)
  - `exports/FINAL_STATUS.json` (written beside archives; never inside them)

Archive integrity certifies byte fidelity, not scientific capability.""",
    )

    setup = _cell(
        "code",
        f"""from hashlib import sha256
import json, os, shutil, subprocess, sys
from pathlib import Path

OWNER_ENABLED = True
PHASE = "auto"  # 'auto' (defaults to 'diagnose' when HH ZIP mounted), 'diagnose', 'core', 'followup', 'finalize'
MINUTES_BUDGET = None  # None uses frozen phase ceiling: diagnose=90, core=360, followup=240
ENABLE_D2 = False      # Optional separately budgeted layer-probe + patching follow-up in diagnose

CHECKPOINT_ARCHIVE = None  # Optional explicit path to mounted ELENDRIL_HITCHHIKER_FULL.zip
HANDOFF_ARCHIVE = None     # Required for followup (core FULL ZIP); optional compact handoff for core
RESUME_ARCHIVE = None      # Optional same-phase ELENDRIL_V8_Y1_FULL.zip for interrupted recovery
SELECTION_RECEIPT = None   # Optional explicit path to selection_receipt.json for followup

SOURCE_COMMIT = "{source}"
REPO_URL = "{REPOSITORY_URL}"
EXPECTED_HH_SHA = "{EXPECTED_HH_ARCHIVE_SHA256}"
EXPECTED_HH_MANIFEST_SHA = "{EXPECTED_HH_MANIFEST_SHA256}"

def _file_sha256(path: Path) -> str:
    h = sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def _discover_mounted_hh_zip() -> Path | None:
    search_roots = [Path("/kaggle/input")]
    for root in search_roots:
        if not root.exists():
            continue
        for cand in sorted(root.rglob("ELENDRIL_HITCHHIKER_FULL.zip")):
            if cand.is_file() and _file_sha256(cand) == EXPECTED_HH_SHA:
                return cand
        for mpath in sorted(root.rglob("BUNDLE_MANIFEST.json")):
            if mpath.is_file() and _file_sha256(mpath) == EXPECTED_HH_MANIFEST_SHA:
                return mpath.parent
    return None

def _discover_mounted_y1_handoff(for_phase: str) -> Path | None:
    root = Path("/kaggle/input")
    if not root.exists():
        return None
    target_zip = "ELENDRIL_V8_Y1_EVIDENCE.zip" if for_phase == "core" else "ELENDRIL_V8_Y1_FULL.zip"
    for cand in sorted(root.rglob(target_zip)):
        if cand.is_file():
            return cand
    expected_from = "diagnose" if for_phase == "core" else "core"
    for ipath in sorted(root.rglob("identity.json")):
        if ipath.parent.name == "analysis" and (ipath.parent.parent / "BUNDLE_MANIFEST.json").is_file():
            try:
                ident = json.loads(ipath.read_text(encoding="utf-8"))
                if ident.get("phase") == expected_from:
                    return ipath.parent.parent
            except Exception:
                pass
    return None

assert PHASE in {{"auto", "diagnose", "core", "followup", "finalize"}}, f"Unsupported PHASE: {{PHASE!r}}"
EFFECTIVE_PHASE = PHASE
if EFFECTIVE_PHASE == "auto":
    if CHECKPOINT_ARCHIVE is not None:
        EFFECTIVE_PHASE = "diagnose"
    else:
        found = _discover_mounted_hh_zip()
        if found is None:
            raise FileNotFoundError(
                "PHASE='auto' defaults to 'diagnose' and requires mounted ELENDRIL_HITCHHIKER_FULL.zip "
                f"(sha256={{EXPECTED_HH_SHA}}) or its auto-unpacked Kaggle Dataset under /kaggle/input/. "
                "Mount the archive or set PHASE='core' explicitly."
            )
        CHECKPOINT_ARCHIVE = str(found)
        EFFECTIVE_PHASE = "diagnose"
elif EFFECTIVE_PHASE == "diagnose" and CHECKPOINT_ARCHIVE is None and RESUME_ARCHIVE is None:
    found = _discover_mounted_hh_zip()
    if found is None:
        raise FileNotFoundError(
            "PHASE='diagnose' requires CHECKPOINT_ARCHIVE pointing to verified ELENDRIL_HITCHHIKER_FULL.zip."
        )
    CHECKPOINT_ARCHIVE = str(found)
elif EFFECTIVE_PHASE in ("core", "followup") and HANDOFF_ARCHIVE is None and RESUME_ARCHIVE is None:
    found_h = _discover_mounted_y1_handoff(EFFECTIVE_PHASE)
    if found_h is not None:
        HANDOFF_ARCHIVE = str(found_h)

WORK_ROOT = Path(f"/kaggle/working/elendril_y1_{{EFFECTIVE_PHASE}}")
CODE_ROOT = Path("/kaggle/working/anra_y1_src")
WORK_ROOT.mkdir(parents=True, exist_ok=True)
if CODE_ROOT.exists():
    shutil.rmtree(CODE_ROOT)

subprocess.run(["git", "clone", "--no-checkout", REPO_URL, str(CODE_ROOT)], check=True)
subprocess.run(["git", "-C", str(CODE_ROOT), "fetch", "--all", "--tags"], check=True)
resolved = subprocess.check_output(["git", "-C", str(CODE_ROOT), "rev-parse", SOURCE_COMMIT], text=True).strip()
assert resolved == SOURCE_COMMIT, f"commit mismatch: {{resolved}} != {{SOURCE_COMMIT}}"
subprocess.run(["git", "-C", str(CODE_ROOT), "checkout", "--detach", SOURCE_COMMIT], check=True)
os.chdir(CODE_ROOT)
print(
    f"Verified clean source checkout at {{SOURCE_COMMIT}} | phase={{EFFECTIVE_PHASE}} | "
    f"checkpoint_archive={{CHECKPOINT_ARCHIVE}} | work_root={{WORK_ROOT}}"
)
""",
    )

    run_cell = _cell(
        "code",
        """cmd = [
    sys.executable, "-m", "tools.elendril_y1_campaign",
    "--phase", EFFECTIVE_PHASE,
    "--source", SOURCE_COMMIT,
    "--output", str(WORK_ROOT),
]
if OWNER_ENABLED:
    cmd.append("--owner-enabled")
if MINUTES_BUDGET is not None:
    cmd.extend(["--minutes", str(MINUTES_BUDGET)])
if CHECKPOINT_ARCHIVE is not None:
    cmd.extend(["--checkpoint-archive", str(CHECKPOINT_ARCHIVE)])
if HANDOFF_ARCHIVE is not None:
    cmd.extend(["--handoff-archive", str(HANDOFF_ARCHIVE)])
if RESUME_ARCHIVE is not None:
    cmd.extend(["--resume-archive", str(RESUME_ARCHIVE)])
if SELECTION_RECEIPT is not None:
    cmd.extend(["--selection-receipt", str(SELECTION_RECEIPT)])
if ENABLE_D2:
    cmd.append("--enable-d2")

env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
print("Launching:", " ".join(cmd))
proc = subprocess.run(cmd, cwd=CODE_ROOT, env=env)

status_file = WORK_ROOT / "exports" / "FINAL_STATUS.json"
if status_file.exists():
    doc = json.loads(status_file.read_text(encoding="utf-8"))
    print(json.dumps({
        "returncode": proc.returncode,
        "status": doc.get("status"),
        "scientific_status": doc.get("scientific_status"),
        "delivery_status": doc.get("delivery_status"),
        "errors": doc.get("errors"),
    }, indent=2))

for name in ("ELENDRIL_V8_Y1_SUMMARY.zip", "ELENDRIL_V8_Y1_EVIDENCE.zip", "ELENDRIL_V8_Y1_FULL.zip", "FINAL_STATUS.json"):
    target = WORK_ROOT / "exports" / name
    if target.exists():
        print("ARTIFACT:", str(target))
        try:
            from IPython.display import FileLink, display
            display(FileLink(str(target)))
        except ImportError:
            pass

if proc.returncode != 0:
    raise RuntimeError(f"V8-Y1 campaign exited with code {proc.returncode}; inspect status and artifact links above.")
""",
    )

    links = _cell(
        "code",
        """for name in ("ELENDRIL_V8_Y1_SUMMARY.zip", "ELENDRIL_V8_Y1_EVIDENCE.zip", "ELENDRIL_V8_Y1_FULL.zip", "FINAL_STATUS.json"):
    target = WORK_ROOT / "exports" / name
    if target.exists():
        print(str(target))
        try:
            from IPython.display import FileLink, display
            display(FileLink(str(target)))
        except ImportError:
            pass
""",
    )

    return {
        "cells": [header, setup, run_cell, links],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11.0"},
            "source_reference": source,
            "v8_y1_notebook": True,
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_NOTEBOOK)
    args = parser.parse_args(argv)
    source = args.source or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True).strip()
    nb = build_y1_notebook(source)
    raw = (json.dumps(nb, indent=2) + "\n").encode("utf-8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(raw)
    print(json.dumps({"notebook": str(args.output), "source": source, "sha256": sha256(raw).hexdigest()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
