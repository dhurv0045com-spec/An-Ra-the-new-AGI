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

**IMPORTANT: Set `RUN_STUDY = True` in the configuration cell to actually run.**
With the default `False`, Kaggle now stops with a clear `V16 NOT STARTED`
error instead of falsely displaying a successful experiment. The suite executes
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
RUN_STUDY = False  # REQUIRED: change to True to launch; False reports V16 NOT STARTED (not success)
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
if type(RUN_STUDY) is not bool:
    raise TypeError("RUN_STUDY must be an explicit Python bool: True to launch or False to block")
if RUN_STUDY:
    print(f"V16 OWNER LAUNCH REQUESTED: phase={PHASE}, arm={ARM_SUBSET}, seed={SEED_SUBSET}, budget={MINUTES}m", flush=True)
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
            raise ValueError("V16 NO-GO: attach the genuine V7.15 parent FULL ZIP as a Kaggle input and set PARENT_ARCHIVE, or provide a verified PARENT_OUTPUT")
        if PARENT_ARCHIVE is not None and not Path(PARENT_ARCHIVE).is_file():
            raise FileNotFoundError("V16 NO-GO: PARENT_ARCHIVE is not an existing file: " + str(PARENT_ARCHIVE))
        if PARENT_OUTPUT is not None and not Path(PARENT_OUTPUT).is_dir():
            raise FileNotFoundError("V16 NO-GO: PARENT_OUTPUT is not an existing directory: " + str(PARENT_OUTPUT))
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
        # The parent archive is verified before extraction (continual) or by
        # the campaign gate (mechanism/promotion); once an extracted parent
        # directory exists it is the custody input, never both.
        if PARENT_ARCHIVE is not None and parent is None:
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
    print("V16 NOT STARTED: RUN_STUDY=False. Set RUN_STUDY=True in the first code cell, attach the verified parent FULL ZIP, and select Kaggle GPU T4x2.", flush=True)
    raise RuntimeError("V16 NOT STARTED: notebook deliberately disabled; zero optimizer updates ran. A disabled run must not appear as Kaggle SUCCESS.")
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


OWNER_LAUNCH_DEST = (Path(__file__).resolve().parents[1] / "notebooks" /
                     "ELENDRIL_V716_OWNER_LAUNCH_KAGGLE_T4X2.ipynb")
OWNER_FIRST_RUN_PREFLIGHT = "\"\"\"Owner-authorized V16 launch: diagnose attached Kaggle inputs before GPU training.\n\nAll input files are read-only. The diagnostic shows exactly which datasets\nand ZIP archives are present and admits only the historical FULL ZIP SHA-256.\nA NO-GO is a custody failure, not a neural-network or GPU learning result.\n\"\"\"\nfrom hashlib import sha256\nfrom pathlib import Path\nimport json\nimport subprocess\n\nPINNED_PARENT_SHA256 = \"1826c85b5a858d2a600661f7e5916f0fa812c55a8d53c37013c695df28a15d19\"\nPARENT_CORE_RUN_ID = \"20c80f66e76e4b589aed2053afbddffc\"\nPARENT_CORE_SOURCE_SHA = \"6104ffd7134cb4cf4b3f340bd060d24a1e2dd200\"\nPARENT_CORE_BYTES = 1958029910\nINPUT_DIAGNOSTIC = Path(\"/kaggle/working/V16_INPUT_DIAGNOSTIC.json\")\n\n\ndef _file_sha256(path):\n    digest = sha256()\n    with Path(path).open(\"rb\") as handle:\n        while True:\n            block = handle.read(8 * 1024 * 1024)\n            if not block:\n                break\n            digest.update(block)\n    return digest.hexdigest()\n\n\ndef _find_parent_archive(root, expected_sha256, diagnostic_path=None):\n    \"\"\"Inventory and hash Kaggle ZIPs, save a report, refuse unverified inputs.\"\"\"\n    root = Path(root)\n    root_exists = root.is_dir()\n    report = {\n        \"schema\": \"v16-kaggle-input-diagnostic-v1\",\n        \"expected_parent_sha256\": expected_sha256,\n        \"expected_parent_run_id\": PARENT_CORE_RUN_ID,\n        \"expected_parent_source\": PARENT_CORE_SOURCE_SHA,\n        \"expected_parent_bytes\": PARENT_CORE_BYTES,\n        \"input_root\": str(root),\n        \"attached_datasets\": [],\n        \"file_count\": 0,\n        \"file_examples\": [],\n        \"zip_archives\": [],\n        \"status\": \"INSPECTING\",\n    }\n\n    def publish():\n        if diagnostic_path is not None:\n            output = Path(diagnostic_path)\n            output.parent.mkdir(parents=True, exist_ok=True)\n            output.write_text(json.dumps(report, indent=2, sort_keys=True) + \"\\n\",\n                              encoding=\"utf-8\")\n            print(\"V16 input diagnostic saved:\", output, flush=True)\n\n    if not root_exists:\n        report[\"status\"] = \"MISSING_INPUT_ROOT\"\n        publish()\n        raise RuntimeError(\"V16 INPUT NO-GO: /kaggle/input directory is missing. \"\n                           \"Attach the genuine V7.1 CORE FULL ZIP via Kaggle Add Input.\")\n\n    report[\"attached_datasets\"] = sorted(\n        entry.name for entry in root.iterdir() if entry.is_dir() and not entry.is_symlink())\n    candidates = []\n    for path in root.rglob(\"*\"):\n        if not path.is_file() or path.is_symlink():\n            continue\n        report[\"file_count\"] += 1\n        if len(report[\"file_examples\"]) < 25:\n            report[\"file_examples\"].append(str(path.relative_to(root)))\n        if path.suffix.lower() == \".zip\":\n            candidates.append(path)\n    candidates.sort(key=lambda path: str(path))\n    print(\"V16 REQUIRED original V7.1 CORE FULL ZIP:\", PARENT_CORE_RUN_ID,\n          \"| expected bytes:\", PARENT_CORE_BYTES, \"| SHA-256:\", expected_sha256,\n          flush=True)\n    print(\"V16 attached Kaggle datasets:\", report[\"attached_datasets\"], flush=True)\n    print(\"V16 input files:\", report[\"file_count\"],\n          \"| ZIP candidates:\", len(candidates), flush=True)\n    for name in report[\"file_examples\"][:15]:\n        print(\"V16 input example:\", name, flush=True)\n\n    if not candidates:\n        report[\"status\"] = \"NO_ZIP_ARCHIVES\"\n        publish()\n        raise RuntimeError(\n            \"V16 INPUT NO-GO: no ZIP files found under /kaggle/input. \"\n            \"Attached extracted V7.15 scale-S8 files or a Kaggle notebook output \"\n            \"are NOT proof of the required historical V7.1 CORE FULL ZIP. \"\n            \"Use Add Input to attach the original V7.1 CORE FULL ZIP, \"\n            \"run 20c80f66e76e4b589aed2053afbddffc \"\n            \"(1,958,029,910 bytes). \"\n            \"See V16_INPUT_DIAGNOSTIC.json for the actual input inventory.\")\n\n    matching = None\n    for path in candidates:\n        digest = _file_sha256(path)\n        same = digest == expected_sha256\n        record = {\n            \"path\": str(path),\n            \"sha256\": digest,\n            \"byte_size\": path.stat().st_size,\n            \"parent_full_match\": same,\n            \"expected_core_byte_size_match\": path.stat().st_size == PARENT_CORE_BYTES,\n            \"looks_like_wrong_s8_stage\": \"scale-s8\" in str(path).lower(),\n        }\n        report[\"zip_archives\"].append(record)\n        print(\"V16 ZIP:\", path, \"| bytes:\", record[\"byte_size\"],\n              \"| SHA-256:\", digest, \"| MATCH:\", same, flush=True)\n        if same:\n            matching = path\n    if matching is None:\n        report[\"status\"] = \"ZIP_PRESENT_BUT_HASH_MISMATCH\"\n        report[\"wrong_v715_scale_s8_detected\"] = any(\n            row[\"looks_like_wrong_s8_stage\"] for row in report[\"zip_archives\"])\n        if report[\"wrong_v715_scale_s8_detected\"]:\n            print(\"V16 WRONG INPUT: attached V7.15 scale-S8 outputs, \"\n                  \"not the V7.1 CORE FULL archive.\", flush=True)\n        publish()\n        raise RuntimeError(\n            \"V16 INPUT NO-GO: none of the \" + str(len(candidates)) +\n            \" attached ZIP files matches the historical V7.1 CORE FULL \"\n            \"SHA-256 \" + expected_sha256 + \". An ANALYSIS ZIP, an extracted \"\n            \"dataset, or another V7.15 stage is not a substitute. \"\n            \"Find the earlier V7.1 CORE Kaggle output, not the V7.15 scale-S8 \"\n            \"notebook output. Required ELENDRIL_V71_FULL.zip is exactly \"\n            \"1,958,029,910 bytes. See V16_INPUT_DIAGNOSTIC.json for observed \"\n            \"SHA-256 values.\")\n    report[\"status\"] = \"VERIFIED_PARENT_FULL\"\n    report[\"selected_parent_archive\"] = str(matching)\n    publish()\n    return matching\n\n\ndef _auto_fetch_original_core(expected_sha256):\n    \"\"\"Try Kaggle's native notebook-output API; no owner Add Input required.\n\n    The original V7.1 CORE notebook was observed at this exact Kaggle handle.\n    Download one specific historical FULL ZIP, not latest unrelated output.\n    Fail closed if it is unavailable, private, missing or hash-mismatched.\n    \"\"\"\n    handle = \"ankitrajm7c/elendril-v71-kaggle-t4x2\"\n    relative = \"elendril-v71-6104ffd7134c-core/exports/ELENDRIL_V71_FULL.zip\"\n    try:\n        import kagglehub\n    except ImportError as exc:\n        raise RuntimeError(\n            \"V16 AUTO-FETCH NO-GO: kagglehub is unavailable in this Kaggle runtime. \"\n            \"No dataset has been substituted.\") from exc\n    print(\"V16 attempting automatic read-only attachment from Kaggle notebook:\",\n          handle, \"path:\", relative, flush=True)\n    try:\n        downloaded = kagglehub.notebook_output_download(handle, path=relative)\n    except Exception as exc:\n        raise RuntimeError(\n            \"V16 AUTO-FETCH NO-GO: Kaggle could not grant access to the recorded \"\n            \"V7.1 CORE notebook output. This may be a version or access restriction. \"\n            \"Original data has NOT been replaced.\") from exc\n    target = Path(downloaded)\n    if target.is_dir():\n        candidates = sorted(target.rglob(\"ELENDRIL_V71_FULL.zip\"))\n        if len(candidates) != 1:\n            raise RuntimeError(\"V16 AUTO-FETCH NO-GO: downloaded parent file is ambiguous\")\n        target = candidates[0]\n    if not target.is_file() or target.is_symlink():\n        raise RuntimeError(\"V16 AUTO-FETCH NO-GO: Kaggle returned no regular parent FULL ZIP\")\n    if target.stat().st_size != PARENT_CORE_BYTES:\n        raise RuntimeError(\n            \"V16 AUTO-FETCH NO-GO: remote FULL archive size mismatched the \"\n            \"recorded V7.1 CORE parent; got \" + str(target.stat().st_size))\n    actual = _file_sha256(target)\n    if actual != expected_sha256:\n        raise RuntimeError(\"V16 AUTO-FETCH NO-GO: remote parent SHA-256 mismatch: \" + actual)\n    print(\"V16 AUTO-FETCH VERIFIED V7.1 CORE FULL:\", target, flush=True)\n    return target\n\n\n# EXECUTE OWNER PREFLIGHT (no GPU training begins before this passes)\nif RUN_STUDY is not True:\n    raise RuntimeError(\"OWNER LAUNCH CONFIG ERROR: the owner launch notebook requires RUN_STUDY=True\")\nif PHASE != \"mechanism\" or ARM_SUBSET != \"H2B-NORM\" or SEED_SUBSET != 7:\n    raise RuntimeError(\"OWNER LAUNCH CONFIG ERROR: expected H2B-NORM seed-7 screening\")\nif PARENT_OUTPUT is not None:\n    raise RuntimeError(\"OWNER LAUNCH CONFIG ERROR: this launch profile requires the recorded parent FULL ZIP\")\nif PARENT_ARCHIVE is None:\n    try:\n        PARENT_ARCHIVE = _find_parent_archive(\n            \"/kaggle/input\", PINNED_PARENT_SHA256, diagnostic_path=INPUT_DIAGNOSTIC)\n    except RuntimeError as input_error:\n        print(\"V16 original parent not mounted; trying automatic KaggleHub attachment.\",\n              flush=True)\n        try:\n            PARENT_ARCHIVE = _auto_fetch_original_core(PINNED_PARENT_SHA256)\n        except RuntimeError as fetch_error:\n            raise RuntimeError(\n                \"V16 PARENT UNAVAILABLE: neither existing read-only Kaggle inputs \"\n                \"nor the automatic original V7.1 CORE notebook-output fetch \"\n                \"provided the verified FULL archive. Attached-input diagnosis: \" +\n                str(input_error) + \". Automatic-fetch diagnosis: \" +\n                str(fetch_error)) from fetch_error\n        if INPUT_DIAGNOSTIC.is_file():\n            report = json.loads(INPUT_DIAGNOSTIC.read_text(encoding=\"utf-8\"))\n            report.update(status=\"AUTO_FETCHED_PARENT_VERIFIED\",\n                          selected_parent_archive=str(PARENT_ARCHIVE),\n                          auto_fetch_source=\"ankitrajm7c/elendril-v71-kaggle-t4x2\")\n            INPUT_DIAGNOSTIC.write_text(\n                json.dumps(report, indent=2, sort_keys=True) + \"\\n\",\n                encoding=\"utf-8\")\nelse:\n    PARENT_ARCHIVE = Path(PARENT_ARCHIVE)\n    if not PARENT_ARCHIVE.is_file() or PARENT_ARCHIVE.is_symlink():\n        raise RuntimeError(\"V16 INPUT NO-GO: configured PARENT_ARCHIVE does not exist as a normal file\")\n    if _file_sha256(PARENT_ARCHIVE) != PINNED_PARENT_SHA256:\n        raise RuntimeError(\"V16 INPUT NO-GO: configured parent ZIP SHA-256 does not match historical record\")\nprint(\"V16 VERIFIED parent FULL ZIP:\", PARENT_ARCHIVE, flush=True)\n\ntry:\n    gpu = subprocess.run(\n        [\"nvidia-smi\", \"--query-gpu=name,memory.free\",\n         \"--format=csv,noheader,nounits\"],\n        capture_output=True, text=True, timeout=30, check=True)\nexcept (OSError, subprocess.SubprocessError) as exc:\n    raise RuntimeError(\"V16 HARDWARE NO-GO: choose GPU T4 x2 in Kaggle settings.\") from exc\ngpu_lines = [line.strip() for line in gpu.stdout.splitlines() if line.strip()]\nif len(gpu_lines) != 2 or any(\"T4\" not in line.upper() for line in gpu_lines):\n    raise RuntimeError(\"V16 HARDWARE NO-GO: require two visible NVIDIA T4 GPUs; \"\n                       \"observed: \" + repr(gpu_lines))\nprint(\"V16 INPUT + VISIBLE GPU PRECHECK PASSED\", flush=True)\nprint(\"Starting registered H2B-NORM / seed 7 screening with pinned V16 code\", flush=True)\n"
OWNER_FIRST_RUN_INTRO = "# AN-RA V16 — OWNER LAUNCH • H2B-NORM / seed 7\n\n**Use this notebook, not `ELENDRIL_V716_HYPOTHESIS_KAGGLE_T4X2.ipynb`.** This is the separate, owner-authorized notebook with `RUN_STUDY=True`. Import it into Kaggle and select **Run All** manually when ready.\n\n## Parent archive is detected or fetched automatically\n\n1. Choose **GPU T4 ×2** in Kaggle and enable Internet.\n2. **No manual Add Input is required if Kaggle permits automatic access.** The notebook first checks any attached input, then uses KaggleHub to retrieve the recorded original V7.1 CORE FULL ZIP from `ankitrajm7c/elendril-v71-kaggle-t4x2` if missing. Your previous V7.15 **scale-S8** notebook output is NOT the parent, even if the ZIP has the same filename. Find the earlier **V7.1 CORE** Kaggle run (run ID `20c80f66e76e4b589aed2053afbddffc`, originally launched with `ELENDRIL_V71_KAGGLE_T4X2.ipynb`, phase `core`). Its `ELENDRIL_V71_FULL.zip` must be **1,958,029,910 bytes**. The S8 FULL ZIP you attached was **2,933,151,710 bytes**.\n3. Run this notebook. Its third cell lists attached dataset names and file examples, computes SHA-256 for each attached ZIP, and writes `/kaggle/working/V16_INPUT_DIAGNOSTIC.json`.\n4. **Only** the exact parent FULL ZIP with SHA-256 `1826c85b5a858d2a600661f7e5916f0fa812c55a8d53c37013c695df28a15d19` passes. If none is mounted, it automatically tries the original V7.1 CORE notebook output; if Kaggle denies the fetch or returns incorrect bytes, it stops **before training** with a concrete NO-GO. It never accepts an S8 archive as a substitute. Download or inspect the diagnostic report to see which input was attached.\n5. If the exact ZIP is present and the T4 checks pass, the pinned V16 code runs the registered **H2B-NORM / seed 7 screening slice** (not the full 21-job campaign). Preserve final `ELENDRIL_V716_ANALYSIS.zip` and `ELENDRIL_V716_FULL.zip` outputs.\n\nA successful source or hardware preflight is **not** scientific success. No assistant has launched your paid GPU run. Historical research source pin: `682c40e1b525424d805e6fb44f0f29afee6c756c`.\n"


def build_owner_launch(source):
    """Produce an explicit owner-run screening artifact; master stays off."""
    notebook = build(source)
    config = "".join(notebook["cells"][1]["source"])
    expected = "RUN_STUDY = False  # REQUIRED: change to True to launch; False reports V16 NOT STARTED (not success)"
    if expected not in config:
        raise ValueError("protected master owner gate changed; refuse launch generation")
    config = config.replace(expected,
        "RUN_STUDY = True   # OWNER-AUTHORIZED: manually launching this notebook starts Kaggle screening")
    config = config.replace('ARM_SUBSET = None     # mechanism: e.g. "H4-OUTCOME-DELIB"',
                            'ARM_SUBSET = "H2B-NORM"  # initial screening arm')
    config = config.replace("SEED_SUBSET = None    # 7 / 19 / 41 (mechanism and promotion slices)",
                            "SEED_SUBSET = 7  # initial screening seed")
    config = config.replace("PARENT_ARCHIVE = None  # V7.15 parent FULL ZIP (mechanism/promotion custody) or V7.1 core FULL ZIP (continual)",
                            "PARENT_ARCHIVE = None  # auto-detected by SHA-256 from /kaggle/input in next cell")
    notebook["cells"][0] = cell("markdown", OWNER_FIRST_RUN_INTRO.replace("682c40e1b525424d805e6fb44f0f29afee6c756c", source))
    notebook["cells"][1] = cell("code", config)
    notebook["cells"].insert(2, cell("code", OWNER_FIRST_RUN_PREFLIGHT))
    notebook["metadata"].update(owner_run_launch_profile="EXPLICIT_OWNER_LAUNCH_H2B_NORM_S7",
                                owner_enabled=True, elendril_v716_status="OWNER_T4_GPU_RUN_UNVERIFIED",
                                parent_custody_origin="V7.1 CORE", v16_input_diagnostic="v1")
    return notebook



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
