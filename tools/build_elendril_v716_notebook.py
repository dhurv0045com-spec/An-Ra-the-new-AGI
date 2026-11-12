"""Create the source-pinned Kaggle T4x2 notebook for Elendril V7.16.

Safe default: RUN_STUDY is False, so importing or running the notebook does
nothing until the owner flips it. Every phase is custody-gated: the parent
archive (or a previously verified extracted parent) must verify before any
data generation, and the execution binding is written once per output
directory. A rejected gate or an exhausted budget preserves receipts and
never silently shortens the registered matrix.
"""
import argparse
import json
from pathlib import Path
import re

DEST = Path(__file__).resolve().parents[1] / "notebooks" / "ELENDRIL_V716_HYPOTHESIS_KAGGLE_T4X2.ipynb"
PARENT_SOURCE = "6104ffd7134cb4cf4b3f340bd060d24a1e2dd200"  # V7.1 core source pin

ARMS = ("H1-COADAPT-MASK", "H2-ARK014-INVARIANCE", "H2B-NORM", "H3-CONTRAST-MEM",
        "H3B-CONTRASTIVE", "H4-OUTCOME-DELIB", "H5-POINTER-READ", "H6-ARK017-CONTINUAL")


def cell(kind, source):
    c = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        c.update({"execution_count": None, "outputs": []})
    return c


def build(source):
    if not isinstance(source, str) or not re.fullmatch("[0-9a-f]{40}", source):
        raise ValueError("immutable 40-hex source SHA required")
    arms = ", ".join(f"`{arm}`" for arm in ARMS)
    intro = f"""# ELENDRIL V7.16 — Causal Falsification & Mechanism Suite (Kaggle T4×2)

**Run All does nothing until you set `RUN_STUDY = True`.** The suite executes
the preregistered falsification arms H1–H6 against the blocked cross-entropy
baseline with the exposure ledger locked (4 epochs / 2,368 updates /
75,776 supervised tokens at S3; unpadded input positions recorded per epoch).
Claim ceiling: CONTROLLED_DEVELOPMENT_MECHANISM_ONLY. A failed hypothesis is a
valid result; nothing is tuned post hoc and no promotion happens without the
preregistered gates.

Phases (one per Kaggle session):

- **mechanism** — all seven S3 mechanism arms (three seeds each): {arms}
  (H6 is the continual phase's native arm). Custody: the recorded V7.15 parent
  FULL ZIP (`PARENT_ARCHIVE`, byte-verified, never extracted) or a previously
  verified extracted V7.1 core output (`PARENT_OUTPUT`) must pass the
  pre-execution gate before any data is generated.
- **continual** — the nine registered v7.15 continual control arms (b-only,
  replay, fresh-b) plus the native ark017-sparse replay arm. Requires the
  genuine V7.1 core FULL ZIP via `PARENT_ARCHIVE` (verified then extracted) or
  an existing verified extraction via `PARENT_OUTPUT`.
- **promotion** — one winning mechanism arm (`PROMOTION_ARM`) rerun at S8,
  admitted only after the preregistered gate: aggregate confirmation ≥ 279/336,
  every seed ≥ 79/112, stale-memory ≥ 28/48 and row-reversal consistency ≥ 0.5
  from `PROMOTION_EVIDENCE`. Also requires `MECHANISM_OUTPUT` (the completed
  mechanism-phase output directory, attachable as a Kaggle input), the recorded
  V7.15 S8 ANALYSIS zip (`S8_ANALYSIS_ZIP`) and parent custody as above.

Optional narrowing: `ARM_SUBSET` (mechanism), `SEED_SUBSET` (7/19/41) and
`STAGE_SUBSET` (continual). Never attach an analysis-only ZIP in place of the
parent FULL archive. Each output directory receives a write-once execution
binding; rerunning a phase requires a fresh `OUTPUT`. Failure, NO_GO and
DEFERRED preserve receipts — the registered matrix is not silently shortened.
CUDA/PyTorch is native Kaggle; do not pip-install a replacement.
Archives: ELENDRIL_V716_ANALYSIS.zip and ELENDRIL_V716_FULL.zip. The repository
results folder stores only curated Markdown/JSON; retain the original large
Kaggle ZIP files externally.
Pinned source: {source}
"""
    config = f"""from pathlib import Path
SOURCE_SHA = "{source}"
RUN_STUDY = False
PHASE = "mechanism"  # mechanism / continual / promotion
MINUTES = 330
ARM_SUBSET = None     # mechanism: e.g. "H4-OUTCOME-DELIB"
SEED_SUBSET = None    # 7 / 19 / 41 (mechanism and promotion slices)
STAGE_SUBSET = None   # continual: "ark017-sparse" / "b-only" / "replay" / "fresh-b"
PROMOTION_ARM = None  # promotion: one registered arm, e.g. "H2-ARK014-INVARIANCE"
MECHANISM_OUTPUT = None       # promotion: completed mechanism-phase output directory
PROMOTION_EVIDENCE = None     # promotion: diagnostic evidence JSON (stale-memory + row-reversal)
S8_ANALYSIS_ZIP = None        # promotion: recorded V7.15 S8 ANALYSIS zip (custody bytes)
PARENT_ARCHIVE = None  # V7.15 parent FULL ZIP (mechanism/promotion custody) or V7.1 core FULL ZIP (continual)
PARENT_OUTPUT = None   # Optional existing verified extracted parent directory.
OUTPUT = Path("/kaggle/working") / ("elendril-v716-" + SOURCE_SHA[:12] + "-" + PHASE)
"""
    launch = '''import os, re, subprocess, sys
from pathlib import Path
from uuid import uuid4
from IPython.display import FileLink, display

REPOSITORY = "https://github.com/dhurv0045com-spec/An-Ra-the-new-AGI.git"
REPO = Path("/kaggle/working") / ("elendril-v716-source-" + SOURCE_SHA)
failed = False
if RUN_STUDY:
    try:
        if not re.fullmatch("[0-9a-f]{40}", SOURCE_SHA):
            raise ValueError("invalid source SHA")
        if PHASE not in ("mechanism", "continual", "promotion"):
            raise ValueError("unknown V7.16 phase")
        if type(MINUTES) not in (int, float) or not 0 < MINUTES <= 330:
            raise ValueError("invalid owner session minutes")
        if PARENT_ARCHIVE is not None and PARENT_OUTPUT is not None:
            raise ValueError("select parent archive or extracted directory, not both")
        if ARM_SUBSET is not None and PHASE != "mechanism":
            raise ValueError("ARM_SUBSET belongs to the mechanism phase")
        if STAGE_SUBSET is not None and PHASE != "continual":
            raise ValueError("STAGE_SUBSET belongs to the continual phase")
        if SEED_SUBSET is not None and SEED_SUBSET not in (7, 19, 41):
            raise ValueError("unknown registered seed")
        if PHASE == "promotion":
            if PROMOTION_ARM is None or not isinstance(PROMOTION_ARM, str):
                raise ValueError("promotion requires PROMOTION_ARM")
            if SEED_SUBSET is not None:
                raise ValueError("promotion runs all three seeds; no seed narrowing")
            for name, value in (("MECHANISM_OUTPUT", MECHANISM_OUTPUT),
                                ("PROMOTION_EVIDENCE", PROMOTION_EVIDENCE),
                                ("S8_ANALYSIS_ZIP", S8_ANALYSIS_ZIP)):
                if value is None or not Path(value).exists():
                    raise ValueError("promotion requires an existing " + name)
        if PARENT_ARCHIVE is None and PARENT_OUTPUT is None:
            raise ValueError("custody requires PARENT_ARCHIVE or PARENT_OUTPUT")
        if not REPO.exists():
            stage = REPO.parent / (".v716-source-" + uuid4().hex)
            stage.mkdir(parents=True)
            subprocess.run(["git", "init", "-q"], cwd=stage, check=True, timeout=30)
            subprocess.run(["git", "remote", "add", "origin", REPOSITORY], cwd=stage, check=True, timeout=30)
            subprocess.run(["git", "fetch", "--depth=1", "origin", SOURCE_SHA], cwd=stage, check=True, timeout=240)
            subprocess.run(["git", "checkout", "--detach", SOURCE_SHA], cwd=stage, check=True, timeout=60)
            assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=stage, text=True).strip() == SOURCE_SHA
            stage.rename(REPO)
        if REPO.is_symlink() or not REPO.is_dir():
            raise ValueError("source checkout must be a real directory")
        if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip() != SOURCE_SHA:
            raise ValueError("source pin mismatch")
        if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all"], cwd=REPO, text=True).strip():
            raise ValueError("source checkout is dirty")
        env = os.environ.copy()
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        parent = PARENT_OUTPUT
        if PARENT_ARCHIVE is not None and PHASE == "continual":
            if parent is not None:
                raise ValueError("continual parent already supplied")
            parent = Path("/kaggle/working") / ("elendril-v716-parent-continual-" + SOURCE_SHA[:12])
            if parent.exists() or parent.is_symlink():
                raise ValueError("parent extraction destination exists; use PARENT_OUTPUT after verification")
            restore = ("from tools.elendril_v71_export import restore_archive; import sys; "
                       "restore_archive(sys.argv[1],sys.argv[2],source=sys.argv[3],phase='continual',parent=True)")
            subprocess.run([sys.executable, "-c", restore, str(PARENT_ARCHIVE), str(parent),
                            "6104ffd7134cb4cf4b3f340bd060d24a1e2dd200"], cwd=REPO, env=env, check=True)
        cmd = [sys.executable, "-m", "tools.elendril_v716_campaign", "--output", str(OUTPUT),
               "--source", SOURCE_SHA, "--phase", PHASE, "--minutes", str(MINUTES),
               "--owner-enabled"]
        if PARENT_ARCHIVE is not None:
            cmd += ["--parent-archive", str(PARENT_ARCHIVE)]
        if parent is not None:
            cmd += ["--parent-output", str(parent)]
        if ARM_SUBSET is not None:
            cmd += ["--arm-subset", str(ARM_SUBSET)]
        if SEED_SUBSET is not None:
            cmd += ["--seed-subset", str(SEED_SUBSET)]
        if STAGE_SUBSET is not None:
            cmd += ["--stage-subset", str(STAGE_SUBSET)]
        if PHASE == "promotion":
            cmd += ["--promotion-arm", PROMOTION_ARM,
                    "--mechanism-output", str(MECHANISM_OUTPUT),
                    "--promotion-evidence", str(PROMOTION_EVIDENCE),
                    "--s8-analysis-zip", str(S8_ANALYSIS_ZIP)]
        subprocess.run(cmd, cwd=REPO, env=env, check=True)
        print("Phase finished. Archive verification describes bytes, not scientific success: "
              "causal claims remain CAUSE_UNRESOLVED until the preregistered gates pass.")
    except Exception as exc:
        failed = True
        print("V7.16 phase incomplete:", type(exc).__name__, str(exc))
        print("Raw output and partial receipts are retained when available.")
else:
    print("Experiment disabled by owner.")
names = ["ELENDRIL_V716_ANALYSIS.zip", "ELENDRIL_V716_FULL.zip"]
for name in names:
    file = OUTPUT / "exports" / name
    print(name, "exists:", file.is_file())
    if file.is_file():
        display(FileLink(str(file)))
receipt = OUTPUT / "analysis" / "DELIVERY_RECEIPT.json"
if receipt.is_file():
    display(FileLink(str(receipt)))
if failed:
    raise RuntimeError("V7.16 Kaggle phase was not complete; inspect saved evidence.")
'''
    return {"cells": [cell("markdown", intro), cell("code", config), cell("code", launch)],
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"}, "source_reference": source,
                         "elendril_v716_status": "OWNER_GPU_QUALIFICATION_PENDING"},
            "nbformat": 4, "nbformat_minor": 4}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--source-sha", required=True)
    p.add_argument("--output", type=Path, default=DEST)
    args = p.parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(build(args.source_sha), indent=2, ensure_ascii=False) + "\n"
    args.output.write_bytes(payload.encode("utf-8"))
    print(f"published {args.output} pinned to {args.source_sha}")


if __name__ == "__main__":
    main()
