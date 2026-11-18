"""Build the source-pinned Avendi V8-Y2 Kaggle 2xT4 notebook (notebooks/AVENDI_V8_Y2_T4X2.ipynb)."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from typing import Any

from anra_v5.elendril.avendi.protocol import (
    DESIGN_SHA,
    HISTORICAL_HH_FULL_ARCHIVE_SHA256,
    Y1_BASELINE_LINEAGES,
    Y1_EVIDENCE_ARCHIVE_SHA256,
    Y1_SUMMARY_ARCHIVE_SHA256,
)


REPOSITORY_URL = "https://github.com/dhurv0045com-spec/An-Ra-the-new-AGI.git"
DEFAULT_NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "AVENDI_V8_Y2_T4X2.ipynb"


def _cell(kind: str, source: str) -> dict[str, Any]:
    lines = source.splitlines(keepends=True)
    base: dict[str, Any] = {"cell_type": kind, "metadata": {}, "source": lines}
    if kind == "code":
        base["execution_count"] = None
        base["outputs"] = []
    return base


def build_avendi_notebook(source: str) -> dict[str, Any]:
    """Construct the source-pinned Avendi V8-Y2 Kaggle 2xT4 notebook."""
    if len(source) != 40 or any(c not in "0123456789abcdef" for c in source):
        raise ValueError(f"source must be a 40-char lowercase git SHA: {source!r}")

    lineages_json = json.dumps({k: v["checkpoint_sha256"] for k, v in Y1_BASELINE_LINEAGES.items()})

    cell_header = _cell(
        "markdown",
        f"""# Avendi V8-Y2 — Kaggle 2×T4 Source-Pinned Campaign Notebook

- **Pinned implementation commit (`SOURCE_COMMIT`)**: `{source}`
- **Design registry digest (`DESIGN_SHA`)**: `{DESIGN_SHA}`
- **Default phase**: `PHASE = "frozen"` (120 wall-minute ceiling; executes P0 CPU instrument qualification + P1 Y1 checkpoint restoration & parity join + P2 frozen crossover diagnosis + stage routing receipt).
- **Missing-checkpoint resilience**: If `ELENDRIL_V8_Y1_FULL.zip` is not attached, the notebook still completes P0 CPU qualification and emits a valid `frozen_receipt.json` with an explicit missing-checkpoint manifest and deterministic retraining recipe.
- **Top-level `/kaggle/working/` exports**: All exported `.zip` bundles and `FINAL_STATUS.json` are written to `/kaggle/working/avendi_v8_y2_<PHASE>/exports/` AND mirrored directly to `/kaggle/working/` top level for one-click download.""",
    )

    cell_setup_and_discover = _cell(
        "code",
        f"""from hashlib import sha256
import json, os, shutil, subprocess, sys
from pathlib import Path

OWNER_ENABLED = True
PHASE = "frozen"  # 'fixture', 'frozen' (default), 'acquisition', 'transition', 'conditional', 'correction', 'all'
MINUTES_BUDGET = 120  # frozen=120, acquisition=240, transition=240
ADMISSION_RECEIPT = None  # Required when PHASE in ('acquisition', 'transition', 'conditional', 'correction')

SOURCE_COMMIT = "{source}"
DESIGN_SHA = "{DESIGN_SHA}"
REPO_URL = "{REPOSITORY_URL}"
EXPECTED_Y1_EVIDENCE_SHA = "{Y1_EVIDENCE_ARCHIVE_SHA256}"
EXPECTED_Y1_SUMMARY_SHA = "{Y1_SUMMARY_ARCHIVE_SHA256}"
EXPECTED_HH_FULL_SHA = "{HISTORICAL_HH_FULL_ARCHIVE_SHA256}"
EXPECTED_Y1_LINEAGES = {lineages_json}

def _sha256(path: Path) -> str:
    h = sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def discover_mounted_inputs() -> dict:
    root = Path("/kaggle/input")
    found = {{"y1_checkpoint_root": None, "y1_evidence": None, "hh_full": None}}
    if not root.exists():
        return found
    for p in sorted(root.rglob("*.zip")):
        if not p.is_file():
            continue
        if p.name == "ELENDRIL_V8_Y1_FULL.zip":
            found["y1_checkpoint_root"] = str(p)
        elif p.name == "ELENDRIL_V8_Y1_EVIDENCE.zip" and _sha256(p) == EXPECTED_Y1_EVIDENCE_SHA:
            found["y1_evidence"] = str(p)
        elif p.name == "ELENDRIL_HITCHHIKER_FULL.zip" and _sha256(p) == EXPECTED_HH_FULL_SHA:
            found["hh_full"] = str(p)
    if found["y1_checkpoint_root"] is None:
        for d in sorted(root.rglob("checkpoints")):
            if d.is_dir() and any((d / jid).is_dir() for jid in EXPECTED_Y1_LINEAGES):
                found["y1_checkpoint_root"] = str(d)
                break
    if found["y1_evidence"] is None:
        for ipath in sorted(root.rglob("identity.json")):
            if ipath.parent.name == "analysis" and (ipath.parent.parent / "prepared" / "development_suite.json").is_file():
                found["y1_evidence"] = str(ipath.parent.parent)
                break
    return found

DISCOVERED = discover_mounted_inputs()
WORK_ROOT = Path(f"/kaggle/working/avendi_v8_y2_{{PHASE}}")
CODE_ROOT = Path("/kaggle/working/anra_avendi_src")
WORK_ROOT.mkdir(parents=True, exist_ok=True)
if CODE_ROOT.exists():
    shutil.rmtree(CODE_ROOT)

subprocess.run(["git", "clone", "--no-checkout", REPO_URL, str(CODE_ROOT)], check=True)
subprocess.run(["git", "-C", str(CODE_ROOT), "fetch", "--all", "--tags"], check=True)
resolved = subprocess.check_output(["git", "-C", str(CODE_ROOT), "rev-parse", SOURCE_COMMIT], text=True).strip()
assert resolved == SOURCE_COMMIT, f"commit mismatch: {{resolved}} != {{SOURCE_COMMIT}}"
subprocess.run(["git", "-C", str(CODE_ROOT), "checkout", "--detach", SOURCE_COMMIT], check=True)
os.chdir(CODE_ROOT)
print(json.dumps({{"source": SOURCE_COMMIT, "design_sha256": DESIGN_SHA, "phase": PHASE, "discovered": DISCOVERED}}, indent=2))
""",
    )

    cell_p0_and_campaign = _cell(
        "code",
        """# Step 1: Run P0 CPU instrument qualification first
p0_cmd = [
    sys.executable, "-m", "tools.avendi_campaign",
    "--phase", "fixture",
    "--source", SOURCE_COMMIT,
    "--output", str(WORK_ROOT / "p0_preflight"),
]
subprocess.run(p0_cmd, cwd=CODE_ROOT, check=True)

# Step 2: Execute requested campaign phase (default: 'frozen')
cmd = [
    sys.executable, "-m", "tools.avendi_campaign",
    "--phase", PHASE,
    "--source", SOURCE_COMMIT,
    "--output", str(WORK_ROOT),
    "--minutes", str(MINUTES_BUDGET),
]
if OWNER_ENABLED:
    cmd.append("--owner-enabled")
if DISCOVERED.get("y1_checkpoint_root"):
    cmd.extend(["--checkpoint-root", str(DISCOVERED["y1_checkpoint_root"])])
if DISCOVERED.get("y1_evidence"):
    cmd.extend(["--evidence", str(DISCOVERED["y1_evidence"])])
if ADMISSION_RECEIPT is not None:
    cmd.extend(["--admission", str(ADMISSION_RECEIPT)])

print("Launching:", " ".join(cmd))
proc = subprocess.run(cmd, cwd=CODE_ROOT)

# Step 3: Run offline analyzer to emit RESULTS.md & METRICS.json
ev_zip = WORK_ROOT / "exports" / "AVENDI_V8_Y2_EVIDENCE.zip"
if ev_zip.is_file():
    subprocess.run(
        [sys.executable, "-m", "tools.analyze_avendi", "--evidence", str(ev_zip), "--output", str(WORK_ROOT / "report")],
        cwd=CODE_ROOT,
        check=False,
    )
""",
    )

    cell_export_links = _cell(
        "code",
        """# Mirror exports directly into /kaggle/working/ root so they appear at the very top of Kaggle's Output sidebar
top_root = Path("/kaggle/working")
for p in sorted((WORK_ROOT / "exports").glob("*")):
    if p.is_file():
        dest = top_root / p.name
        if dest.resolve() != p.resolve():
            shutil.copy2(p, dest)
        print(f"EXPORT: {dest} ({dest.stat().st_size / (1024 * 1024):.2f} MB)")
        try:
            from IPython.display import FileLink, display
            display(FileLink(p.name))
        except ImportError:
            pass

for rfile in (WORK_ROOT / "report" / "RESULTS.md", WORK_ROOT / "report" / "METRICS.json"):
    if rfile.is_file():
        dest = top_root / f"AVENDI_V8_Y2_{rfile.name}"
        shutil.copy2(rfile, dest)
        print(f"REPORT: {dest}")
""",
    )

    return {
        "cells": [cell_header, cell_setup_and_discover, cell_p0_and_campaign, cell_export_links],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11.0"},
            "source_reference": source,
            "design_sha256": DESIGN_SHA,
            "avendi_v8_y2_notebook": True,
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_NOTEBOOK)
    args = parser.parse_args(argv)
    source = args.source or subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
    ).strip()
    nb = build_avendi_notebook(source)
    raw = (json.dumps(nb, indent=2) + "\n").encode("utf-8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(raw)
    print(json.dumps({"notebook": str(args.output), "source": source, "design_sha256": DESIGN_SHA, "sha256": sha256(raw).hexdigest()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
