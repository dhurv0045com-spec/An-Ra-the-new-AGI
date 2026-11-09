"""Create the source-pinned Kaggle T4x2 release notebook for Elendril V7.15."""
import argparse
import json
from pathlib import Path
import re

DEST = Path(__file__).resolve().parents[1] / "notebooks" / "ELENDRIL_V715_KAGGLE_T4X2.ipynb"


def cell(kind, source):
    c = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        c.update({"execution_count": None, "outputs": []})
    return c


def build(source):
    if not isinstance(source, str) or not re.fullmatch("[0-9a-f]{40}", source):
        raise ValueError("immutable 40-hex source SHA required")
    intro = f"""# ELENDRIL V7.15 — Kaggle T4×2

**Run All** begins the 12-job S8/S20 scale study by default, after existing
reproducibility and capacity checks. All three seeds and both registered
uniform/blocked methods are preserved. No external input is required.

Optional phases: diagnostics (all nine frozen S3 V7.1 checkpoints; 1,120
development cases and both decoders); continual (nine A-to-B learning
comparators); historic-frozen (original V7 checkpoints). Diagnostics and
continual require the genuine V7.1 core FULL ZIP, and historic-frozen requires
the distinct original V7 FULL ZIP. Set PARENT_ARCHIVE appropriately. Never
attach an analysis-only ZIP in place of weights.

This notebook can only run ONE phase in each Kaggle session. Other phases
require separate owner-run sessions and measured admission. The 330-minute
maximum is not a guaranteed runtime. Failure and NO_GO preserve receipts;
the registered matrix is not silently shortened to fit.
CUDA/PyTorch is native Kaggle; do not pip-install a replacement.
Pinned source: {source}

Scale/continual/historic-frozen use inherited V7.1 report format and archive
names ELENDRIL_V71_ANALYSIS.zip and ELENDRIL_V71_FULL.zip, with their
actual inherited protocol identity. New frozen diagnostics use
ELENDRIL_V715_ANALYSIS.zip. The repository results folder stores only curated
Markdown/JSON; retain original large Kaggle ZIP files externally.
"""
    config = f"""from pathlib import Path
SOURCE_SHA = "{source}"
RUN_STUDY = True
PHASE = "scale"  # scale / diagnostics / continual / historic-frozen
MINUTES = 330
PARENT_ARCHIVE = None  # Set Kaggle input path to matching original FULL ZIP for non-scale phases.
PARENT_OUTPUT = None   # Optional existing verified extracted parent for continual/historic-frozen only.
RESTORE_ARCHIVE = None # Optional prior FULL ZIP for same-phase recovery (inherited training phases).
OUTPUT = Path("/kaggle/working") / ("elendril-v715-" + SOURCE_SHA[:12] + "-" + PHASE)
"""
    launch = '''import os, re, subprocess, sys
from pathlib import Path
from uuid import uuid4
from IPython.display import FileLink, display

REPOSITORY = "https://github.com/dhurv0045com-spec/An-Ra-the-new-AGI.git"
REPO = Path("/kaggle/working") / ("elendril-v715-source-" + SOURCE_SHA)
failed = False
if RUN_STUDY:
    try:
        if not re.fullmatch("[0-9a-f]{40}", SOURCE_SHA):
            raise ValueError("invalid source SHA")
        if PHASE not in ("scale", "diagnostics", "continual", "historic-frozen"):
            raise ValueError("unknown V7.15 phase")
        if type(MINUTES) not in (int, float) or not 0 < MINUTES <= 330:
            raise ValueError("invalid owner session minutes")
        if PARENT_ARCHIVE is not None and PARENT_OUTPUT is not None:
            raise ValueError("select parent archive or extracted directory, not both")
        if PHASE == "diagnostics" and (PARENT_ARCHIVE is None or PARENT_OUTPUT is not None or RESTORE_ARCHIVE is not None):
            raise ValueError("diagnostics needs V7.1 core FULL ZIP through PARENT_ARCHIVE")
        if PHASE == "scale" and (PARENT_ARCHIVE is not None or PARENT_OUTPUT is not None):
            raise ValueError("scale uses fresh weights and no parent")
        if PHASE in ("continual", "historic-frozen") and PARENT_ARCHIVE is None and PARENT_OUTPUT is None:
            raise ValueError("phase requires its original FULL archive or previously extracted parent")
        if not REPO.exists():
            stage = REPO.parent / (".v715-source-" + uuid4().hex)
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
        if PHASE == "diagnostics":
            cmd = [sys.executable, "-m", "tools.elendril_v715_diagnostic_campaign",
                   "--output", str(OUTPUT), "--source", SOURCE_SHA,
                   "--parent-archive", str(PARENT_ARCHIVE), "--owner-enabled",
                   "--minutes", str(MINUTES)]
        else:
            old_phase = "frozen" if PHASE == "historic-frozen" else PHASE
            if RESTORE_ARCHIVE is not None:
                if OUTPUT.exists() or OUTPUT.is_symlink():
                    raise ValueError("restore requires a new destination")
                restore = ("from tools.elendril_v71_export import restore_archive; import sys; "
                           "restore_archive(sys.argv[1],sys.argv[2],source=sys.argv[3],phase=sys.argv[4],parent=False)")
                subprocess.run([sys.executable, "-c", restore, str(RESTORE_ARCHIVE),
                                str(OUTPUT), SOURCE_SHA, old_phase], cwd=REPO, env=env, check=True)
            parent = PARENT_OUTPUT
            if PARENT_ARCHIVE is not None:
                parent = Path("/kaggle/working") / ("elendril-v715-parent-" + PHASE + "-" + SOURCE_SHA[:12])
                if parent.exists() or parent.is_symlink():
                    raise ValueError("parent extraction destination exists; use PARENT_OUTPUT after verification")
                restore = ("from tools.elendril_v71_export import restore_archive; import sys; "
                           "restore_archive(sys.argv[1],sys.argv[2],source=sys.argv[3],phase=sys.argv[4],parent=True)")
                subprocess.run([sys.executable, "-c", restore, str(PARENT_ARCHIVE),
                                str(parent), SOURCE_SHA, old_phase], cwd=REPO, env=env, check=True)
            cmd = [sys.executable, "-m", "tools.elendril_v71_campaign", "--output", str(OUTPUT),
                   "--source", SOURCE_SHA, "--phase", old_phase,
                   "--minutes", str(MINUTES), "--owner-enabled"]
            if parent is not None:
                cmd += ["--parent-output", str(parent)]
        subprocess.run(cmd, cwd=REPO, env=env, check=True)
        print("Phase finished. Treat model performance and engineering delivery separately.")
    except Exception as exc:
        failed = True
        print("V7.15 phase incomplete:", type(exc).__name__, str(exc))
        print("Raw output and partial receipts are retained when available.")
else:
    print("Experiment disabled by owner.")
names = (["ELENDRIL_V715_ANALYSIS.zip"] if PHASE == "diagnostics" else
         ["ELENDRIL_V71_ANALYSIS.zip", "ELENDRIL_V71_FULL.zip"])
for name in names:
    file = OUTPUT / "exports" / name
    print(name, "exists:", file.is_file())
    if file.is_file():
        display(FileLink(str(file)))
final = OUTPUT / "exports" / "FINAL_STATUS.json"
if final.is_file():
    display(FileLink(str(final)))
if failed:
    raise RuntimeError("V7.15 Kaggle phase was not complete; inspect saved evidence.")
'''
    return {"cells": [cell("markdown", intro), cell("code", config), cell("code", launch)],
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"}, "source_reference": source,
                         "elendril_v715_status": "OWNER_GPU_QUALIFICATION_PENDING"},
            "nbformat": 4, "nbformat_minor": 4}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--source-sha", required=True)
    p.add_argument("--output", type=Path, default=DEST)
    args = p.parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(build(args.source_sha), indent=2, ensure_ascii=False) + "\n")
    print(f"published {args.output} pinned to {args.source_sha}")


if __name__ == "__main__":
    main()
