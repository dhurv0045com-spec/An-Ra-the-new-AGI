"""Generate the two owner-run notebooks for an immutable ELENDRIL source commit."""

import argparse
import json
from pathlib import Path
import re


FILES = {"colab": "ELENDRIL_SCREEN_COLAB_T4.ipynb", "kaggle": "ELENDRIL_CONFIRM_KAGGLE_T4X2.ipynb"}


def cells(platform: str, source: str):
    colab = platform == "colab"
    heading = "ELENDRIL screening · 1×T4" if colab else "ELENDRIL development replication · 2×T4"
    platforms = (
        """In Colab choose **Runtime → Change runtime type → T4 GPU**. Set `RUN_CANARIES = True` to mount Drive and run the bounded canary. The canary copies a real checkpoint to Drive, reads it back, and starts a fresh learner process from that copy. The 105-minute training and 15-minute export caps are targets, and actual Colab access and session lifetime vary. On resume reuse the same Run ID, exact code SHA and compatible CUDA image.

[Official Colab FAQ](https://research.google.com/colaboratory/faq.html)"""
        if colab else
        """On Kaggle choose **Settings → Accelerator → GPU T4 x2** and enable Internet for the pinned checkout, or attach an authorized Git checkout and set its path. Use **Save Version → Save & Run All** after reviewing the interactive canary and intentionally setting both switches in the notebook. This runs a new session, including fresh canaries. Confirm your actual account GPU allowance and session time before opting in. Attach completed Colab screen outputs as read-only input and list their run roots. Attach earlier Kaggle outputs in `PRIOR_OUTPUTS` when resuming. Retain all prior version references if they hold different committed rungs.

The 315-minute training and 30-minute export caps are targets. Two subprocess lanes see distinct physical T4 UUIDs; each uses its own `cuda:0` within its visibility mask. VRAM remains per device. `/kaggle/working` is session storage until you save and independently retrieve a published version. The runtime copy/restore canary does not establish off-session persistence. Download the saved version via Output or a supported KaggleHub `notebook_output_download` flow and validate the bundle and generation markers. A killed session may lose everything after the latest independently retrieved output. Never embed credentials.

[Official Kaggle notebooks](https://www.kaggle.com/docs/notebooks) · [Kaggle runtime images](https://github.com/Kaggle/docker-python) · [KaggleHub output retrieval](https://github.com/Kaggle/kagglehub)"""
    )
    plans = ("16 arms, seeds 7 and 19, 1,536 updates per arm, rungs 384 / 768 / 1,536. "
             "The expanded profile compares dense capacity, contextual pointer, hybrid one-read, fixed two-read, fixed three-read, shared recurrent, augmentation, and canonicalization."
             if colab else
             "45 arms, seeds 7, 19, 41, 61 and 79, 6,144 updates per arm, rungs 1,024 / 3,072 / 6,144. "
             "The larger cloud profile repeats the screen arms on a structurally disjoint development partition and includes dense-intermediate recurrence for a sparse/dense comparison. It requires a complete matched screening result, independent rescore, a prespecified effect gate and explicit owner review.")
    yield "markdown", f"""# {heading}

**Questions.** Which representation change improves unfamiliar symbols and punctuation? Can weight-shared recurrence extend depth while retaining the base join? Do sparse intermediates preserve behavior and reduce measured cost? These are public development studies with matched seeded controls.

{plans}

**Immutable source:** `{source}`. Default settings run no training, no network fetch and no persistent writes. The bounded canary and expensive campaign are separate switches. Every committed rung includes raw predictions and a verified checkpoint identity. A measured forecast admits only complete comparison rounds that fit the remaining cap. If all arms finish earlier the notebook exits.

{platforms}

Use the platform's native CUDA-enabled PyTorch 2.8+ (2.x) with Python 3.11 or newer. Preflight binds the exact runtime and driver. The operator does not replace platform packages. Use FP32 on T4 with deterministic settings; the live canary, rather than an assumed AMP configuration, decides hardware eligibility. [Official PyTorch installation](https://pytorch.org/get-started/locally/) · [PyTorch reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html)"""
    yield "code", f"""# First owner setting: enable only the bounded canary.
from pathlib import Path
RUN_CANARIES = False
RUN_ID = "elendril-v3-{platform}-001"
SOURCE = "{source}"
PLATFORM = "{platform}"
REPOSITORY = "https://github.com/dhurv0045com-spec/An-Ra-the-new-AGI.git"
REPO = Path({('/content/elendril-v3-source' if colab else '/tmp/elendril-v3-source')!r})
WORKSPACE_ROOT = Path({('/content/elendril-v3-scratch' if colab else '/tmp/elendril-v3-scratch')!r})
DESTINATION_ROOT = Path({('/content/drive/MyDrive/ELENDRIL' if colab else '/kaggle/working/ELENDRIL')!r})
PRIOR_OUTPUTS = ()  # Existing complete or partial published run roots, read-only.
SCREEN_OUTPUTS = ()  # Kaggle only. Complete verified Colab screen run roots.
# Enter current available limits shown by the platform before campaign authorization.
SESSION_MINUTES_AVAILABLE = 0
ACCELERATOR_WALL_MINUTES_AVAILABLE = 0
STATUS = {{"status": "DISABLED", "reason": "Run the owner-controlled canary to begin."}}
"""
    yield "code", """# Preflight. This cell does nothing with default settings.
if RUN_CANARIES is True:
    import hashlib, importlib.util, json, os, re, shutil, subprocess, sys, tempfile, time

    START = time.monotonic()
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", RUN_ID):
        raise ValueError("Run ID must be a safe single path component.")
    if not re.fullmatch(r"[0-9a-f]{40}", SOURCE):
        raise ValueError("A full immutable Git commit is required.")
    if not REPO.exists():
        REPO.mkdir(parents=True)
        for args in (["git", "init", "-q"],
                     ["git", "fetch", "--depth=1", REPOSITORY, SOURCE],
                     ["git", "checkout", "-q", "--detach", SOURCE]):
            subprocess.run(args, cwd=REPO, check=True, timeout=240)
    if (subprocess.check_output(["git","rev-parse","HEAD"], cwd=REPO, text=True).strip() != SOURCE
            or subprocess.check_output(["git","status","--porcelain","--untracked-files=all"],
                                       cwd=REPO, text=True).strip()):
        raise ValueError("Source checkout must be clean and match the exact pinned commit.")
    sys.path.insert(0, str(REPO))
    sys.dont_write_bytecode = True
    from tools import elendril_cloud as operator
    if Path(operator.__file__).resolve() != REPO.resolve() / "tools" / "elendril_cloud.py":
        raise ValueError("Imported the wrong operator checkout.")
    operator.check_source(REPO, SOURCE)
    PLAN = operator.PLANS[PLATFORM]
    GPUS = operator.inventory()
    operator.admit_devices(GPUS, PLAN.device_count)
    ENVIRONMENT = operator.check_torch(REPO, GPUS)
    RUNTIME_SHA = hashlib.sha256(operator.blob([
        {"runtime": row["runtime"], "driver": row["physical"]["driver"]} for row in ENVIRONMENT
    ])).hexdigest()
    WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
    WORKSPACE = Path(tempfile.mkdtemp(prefix=RUN_ID+"-", dir=WORKSPACE_ROOT))
    if shutil.disk_usage(WORKSPACE).free < 2 * operator.GIB:
        raise RuntimeError("NO-GO: scratch disk requires at least 2 GiB free.")
    if importlib.util.find_spec("matplotlib") is None:
        raise RuntimeError("NO-GO: native runtime must provide matplotlib for the result bundle.")
    print(json.dumps(ENVIRONMENT, indent=2))
    STATUS = {"status": "GO-CANARY", "reason": "Source, runtime and device preflight passed."}
else:
    print(STATUS)
"""
    yield "code", """# Scientific audit. Hashes are checked against the frozen reference in the pinned source.
if STATUS["status"] == "GO-CANARY":
    AUDIT = WORKSPACE / "fixture-audit.json"
    FIXTURES = operator.scientific_audit(REPO, PLAN, AUDIT)
    FIXTURE_SHA = operator.fingerprint(AUDIT)
    print({key: FIXTURES[key] for key in ("surface","training_rows","development_rows",
                                         "training_clusters","development_clusters")})
"""
    persistence = (
        """        from google.colab import drive
        drive.mount("/content/drive")
        if not Path("/content/drive/MyDrive").is_dir():
            raise RuntimeError("NO-GO: Drive did not mount with MyDrive available.")"""
        if colab else """        if not Path("/kaggle/working").is_dir():
            raise RuntimeError("NO-GO: Kaggle working output is unavailable.")"""
    )
    yield "code", f"""# Real CUDA update, scorer replay, checkpoint publication and fresh-process restore.
if STATUS["status"] == "GO-CANARY":
    from concurrent.futures import ThreadPoolExecutor
    try:
{persistence}
        OUTPUT = DESTINATION_ROOT / operator.safe_id(RUN_ID)
        OUTPUT.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(OUTPUT).free < operator.GIB:
            raise RuntimeError("NO-GO: destination requires at least 1 GiB free for admission.")
        with ThreadPoolExecutor(max_workers=PLAN.device_count) as pool:
            CANARIES = list(pool.map(
                lambda gpu: operator.canary(REPO, SOURCE, PLAN, gpu, WORKSPACE, OUTPUT), GPUS))
        CANARY_SUMMARY = WORKSPACE / "canary-session.json"
        CANARY_SUMMARY.write_bytes(operator.blob({{"runtime_sha256": RUNTIME_SHA,
                                                    "physical_lanes": CANARIES}}))
        operator.verified_copy(CANARY_SUMMARY, OUTPUT / ("canary-session-" + WORKSPACE.name + ".json"))
        STATUS = {{"status": "CANARIES-PASSED", "reason": "FP32 CUDA update and exact same-device restore verified."}}
        print(json.dumps(CANARIES, indent=2))
    except Exception as exc:
        STATUS = {{"status": "NO-GO", "reason": str(exc)}}
        print(STATUS)
"""
    yield "code", """# Deliberate second owner action. Review the canary output before editing.
RUN_CAMPAIGN = False
OWNER_REVIEWED_SCREEN = False  # Kaggle only. Review actual screen evidence before enabling.
if RUN_CAMPAIGN is True and STATUS["status"] != "CANARIES-PASSED":
    print("NO-GO: this session did not pass its T4 canaries.")
"""
    yield "code", """# Full staged comparison with caps, fixed physical lanes and resume from committed markers.
if RUN_CAMPAIGN is True and STATUS["status"] == "CANARIES-PASSED":
    try:
        if (type(SESSION_MINUTES_AVAILABLE) is not int or type(ACCELERATOR_WALL_MINUTES_AVAILABLE) is not int
                or min(SESSION_MINUTES_AVAILABLE, ACCELERATOR_WALL_MINUTES_AVAILABLE) <= 0):
            raise ValueError("NO-GO: enter your current session and accelerator-wall-minute allowances.")
        SCREENING = None
        if PLATFORM == "kaggle":
            if OWNER_REVIEWED_SCREEN is not True:
                raise ValueError("NO-GO: inspect and explicitly approve the screening evidence.")
            SCREENING = operator.eligible_screen(REPO, SOURCE, tuple(map(Path, SCREEN_OUTPUTS)))
            print(json.dumps(SCREENING["gates"], indent=2))
            if not SCREENING["eligible"]:
                raise ValueError("NO-GO: the completed screen failed every frozen effect gate.")
        budget = min(PLAN.train_seconds + PLAN.export_seconds,
                     SESSION_MINUTES_AVAILABLE * 60, ACCELERATOR_WALL_MINUTES_AVAILABLE * 60)
        remaining = int(budget - (time.monotonic() - START) - PLAN.export_seconds)
        print(f"Plan has {len(PLAN.arms())} arms; reserved export margin {PLAN.export_seconds/60:.0f} min.")
        STATUS = operator.run_campaign(REPO, SOURCE, PLAN, GPUS, CANARIES, WORKSPACE, OUTPUT,
            fixture_sha=FIXTURE_SHA, enable=RUN_CAMPAIGN, prior=tuple(map(Path, PRIOR_OUTPUTS)),
            screening=SCREENING, owner_reviewed_screen=OWNER_REVIEWED_SCREEN,
            budget_seconds=remaining, runtime_sha=RUNTIME_SHA)
    except Exception as exc:
        STATUS = {"status": "NO-GO", "reason": str(exc)}
    print(json.dumps(STATUS, indent=2))
elif STATUS["status"] == "CANARIES-PASSED":
    print("GO FOR OWNER-RUN CAMPAIGN ONLY AFTER REVIEW. Campaign switch remains false.")
"""
    yield "code", """# Export actual evidence, including missing/partial arms, source and checkpoint digests.
if RUN_CANARIES is True and "OUTPUT" in globals() and "AUDIT" in globals() and AUDIT.exists():
    REPORT = operator.export_run(PLAN, SOURCE, OUTPUT, WORKSPACE, audit=AUDIT,
        environment=ENVIRONMENT, launch=STATUS, prior=tuple(map(Path, PRIOR_OUTPUTS)))
    print("Compact result ZIP:", REPORT)
    print("Resume payloads and committed rung markers:", OUTPUT / "arms")
else:
    print("No training evidence was created by this notebook invocation.")
"""
    yield "markdown", """## Owner decision and resume

- **GO for an owner-run canary** only if you selected the specified T4 runtime, reviewed the budget and understand the persistence destination. This notebook has no remotely observed CUDA result until you return actual receipts.
- **GO for campaign** only after every lane passes a new-process continuation check, the fixture audit matches its frozen hashes and the forecast can admit the next complete rung. Kaggle additionally requires a qualified and owner-reviewed Colab screen.
- **NO-GO** on a source, runtime, GPU count, readback, fixture, learning or budget mismatch. Retrieve failed logs and return the compact bundle. Do not override the guards.
- Resume from verified `COMMITTED.json` boundaries using the same run identity and compatible pinned runtime. Incomplete staging data never constitutes a checkpoint. Published resume outputs and compact results are separate. The bundle contains hashes and raw receipts, while resume needs the full `arms/` tree. Keep every published version used by a resumed campaign.
- Results remain public development evidence. The independent replication generator does not authorize sealed or confirmatory promotion. Human review and independent held-out evidence are required for any promotion.
"""


def build(platform: str, source: str):
    if not re.fullmatch(r"[0-9a-f]{40}", source):
        raise ValueError("notebook source must be a full lowercase commit")
    records = []
    for index, (kind, value) in enumerate(cells(platform, source)):
        cell = {"id": f"v3-{platform}-{index}", "cell_type": kind, "metadata": {},
                "source": value.splitlines(keepends=True)}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        records.append(cell)
    metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "accelerator": "GPU", "elendril": {"source": source, "platform": platform}}
    if platform == "colab":
        metadata["colab"] = {"gpuType": "T4", "provenance": []}
    else:
        metadata["kaggle"] = {"accelerator": "nvidiaTeslaT4", "isGpuEnabled": True}
    return {"cells": records, "metadata": metadata, "nbformat": 4, "nbformat_minor": 5}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("--output", type=Path, default=Path("notebooks"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for platform, filename in FILES.items():
        payload = json.dumps(build(platform, args.source), indent=1, ensure_ascii=False)
        (args.output / filename).write_text(payload + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
