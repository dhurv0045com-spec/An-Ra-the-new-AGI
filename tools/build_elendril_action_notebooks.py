"""Reproducible, OFF-by-default cognitive-action notebooks for an immutable source."""

import argparse
import json
from pathlib import Path
import re


FILES = {"colab": "ELENDRIL_ACTION_COLAB_T4.ipynb", "kaggle": "ELENDRIL_ACTION_KAGGLE_T4X2.ipynb"}


def cells(platform, source):
    from tools.elendril_action_cloud import PLANS, fixture
    if platform not in FILES or re.fullmatch(r"[0-9a-f]{40}", source) is None:
        raise ValueError("supported platform and full immutable source commit required")
    colab = platform == "colab"
    plan = PLANS[platform]
    train, evaluation = fixture(platform)
    yield "markdown", f"""# ELENDRIL V5 cognitive-action {'screen · 1×T4' if colab else 'confirmation · 2×T4'}

This is a new action-learning program, separate from the archived V3 graph experiments.
It tests a public-state decoder's complete ANSWER / QUERY / ABSTAIN episodes with
independent truth, disjoint initial and post-query prompts and charged requests.
The development task is synthetic; success does not establish metacognition or transfer.

**Source:** `{source}`. Three matched arms are CE, action-token-weighted CE and
an initial uncertain-world curriculum (missing/conflicted worlds, including honest
post-query answers). Seeds are {plan.seeds}. Each arm runs
{plan.updates} updates, batch16, FP32, the same dense 128-wide/four-layer core,
224 training examples from160 worlds and80 new evaluation worlds. Full epochs
contain14 updates. The screen and confirmation use different world seeds
(1729/2718) and validation/test prompt partitions. Every arm uses the same total
row multiset; only its first-epoch order or loss weighting changes.

Training cap is {plan.train_seconds//60} minutes plus {plan.export_seconds//60}
minutes reserved for export. These are **maximum budgets, not claimed runtimes**.
Measured all-arm canaries must forecast the whole plan within the cap. A faster
completion exits immediately. Kaggle runs independent jobs on distinct physical
UUIDs; its two T4 memories are not pooled.

**Two-session admission.** First enable only canaries, inspect five-world neural
formation, exact same-device restart and copied-output fresh-process checks, and
save the durable challenge and checkpoint objects. End that VM session. In a new
VM, reattach the saved output, set the recovery mappings and acknowledge the fresh
session. Recovery actually resumes updates7→14 and compares every model and AdamW
tensor/counter to the expected generation. Different physical UUIDs require
rtol1e-5/atol1e-6 numerical equivalence and are labeled accordingly. A new notebook
kernel on the same VM is insufficient. Source/runtime identities must agree.
Campaign execution requires current-lane canaries and successful owner-reattached
fresh-session recovery for **every** GPU. Kaggle also requires all six valid Colab
results and explicit owner review. No automatic model promotion exists.

{'Select a T4 runtime and authorize Drive mounting only when ready. Drive can time out; immutable objects are copied, read back and hash-verified before a mirror pointer advances. Keep the full directory and verify it from a fresh VM.' if colab else 'Select GPU T4 x2 and enable Internet for the source fetch. Use Save Version with saved outputs; attach the saved output in a fresh VM before qualification. /kaggle/working alone is not an off-session durability claim. Confirm your actual account quota before enabling any work.'}

Use native CUDA PyTorch2.8+ (<3), Python3.11+ and matplotlib. No dependency installer
replaces the platform CUDA stack; the exact native versions are recorded and bound
into recovery. Free device types, availability, quotas and lifetimes vary.

[Colab FAQ](https://research.google.com/colaboratory/faq.html) ·
[Kaggle notebooks](https://www.kaggle.com/docs/notebooks) ·
[Kaggle runtime images](https://github.com/Kaggle/docker-python) ·
[KaggleHub saved-output retrieval](https://github.com/Kaggle/kagglehub) ·
[PyTorch CUDA](https://docs.pytorch.org/docs/2.14/notes/cuda.html) ·
[PyTorch safe loading](https://docs.pytorch.org/docs/2.14/notes/serialization.html)
"""
    yield "code", f"""# Owner switches. Default execution performs no network, writes or training.
from pathlib import Path
RUN_CANARIES = False
RUN_RECOVERY = False
RUN_CAMPAIGN = False
FRESH_SESSION_ACK = False
OWNER_REVIEWED_COLAB = False
PLATFORM = {platform!r}
SOURCE = {source!r}
EXPECTED_TRAIN = {train.identity!r}
EXPECTED_EVALUATION = {evaluation.identity!r}
RUN_ID = "elendril-action-v5-{platform}-001"
REPOSITORY = "https://github.com/dhurv0045com-spec/An-Ra-the-new-AGI.git"
REPO = Path({('/content/elendril-action-source' if colab else '/tmp/elendril-action-source')!r})
SCRATCH = Path({('/content/elendril-action-scratch' if colab else '/tmp/elendril-action-scratch')!r})
DURABLE_BASE = Path({('/content/drive/MyDrive/ELENDRIL_ACTION' if colab else '/kaggle/working/ELENDRIL_ACTION')!r})
# Reattach a complete earlier canary directory, then set one mapping per current GPU.
# A mirror contains immutable lineage/<sha>/manifest.json and MIRROR_HEAD.
RECOVERY_INPUTS = ()  # tuple of dict(challenge="...json", mirror="...canary-mirror...")
SCREEN_CAMPAIGN = None  # Kaggle only: attached complete Colab campaign.json path.
STATUS = {{"status": "DISABLED", "reason": "Owner has enabled no work."}}
CANARIES, RECOVERY_RECEIPTS = [], []
if any(type(flag) is not bool for flag in (RUN_CANARIES, RUN_RECOVERY, RUN_CAMPAIGN,
                                         FRESH_SESSION_ACK, OWNER_REVIEWED_COLAB)):
    raise TypeError("Owner switches must be actual bool values")
"""
    mounting = """    from google.colab import drive
    drive.mount("/content/drive")
    if not Path("/content/drive/MyDrive").is_dir():
        raise RuntimeError("NO-GO: Drive mount unavailable")
""" if colab else """    if not Path("/kaggle/working").is_dir():
        raise RuntimeError("NO-GO: saved-output working directory unavailable")
"""
    yield "code", f"""# Pinned source, native dependency/device qualification and persistent output.
if RUN_CANARIES or RUN_RECOVERY or RUN_CAMPAIGN:
    import importlib.util, json, os, shutil, subprocess, sys, time, uuid
    STARTED = time.monotonic()
    os.environ.update(PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                      CUBLAS_WORKSPACE_CONFIG=":4096:8")
    if not REPO.exists():
        REPO.mkdir(parents=True)
        subprocess.run(["git", "init"], cwd=REPO, check=True, timeout=30)
        subprocess.run(["git", "remote", "add", "origin", REPOSITORY], cwd=REPO, check=True, timeout=30)
        subprocess.run(["git", "fetch", "--depth=1", "origin", SOURCE], cwd=REPO, check=True, timeout=180)
        subprocess.run(["git", "checkout", "--detach", SOURCE], cwd=REPO, check=True, timeout=30)
    sys.path.insert(0, str(REPO))
    from tools import elendril_action_cloud as operator
    if Path(operator.__file__).resolve() != (REPO / "tools" / "elendril_action_cloud.py").resolve():
        raise RuntimeError("NO-GO: cached operator did not come from pinned source; restart kernel")
    if Path(sys.modules["tools.elendril_cloud"].__file__).resolve() != (REPO / "tools" / "elendril_cloud.py").resolve():
        raise RuntimeError("NO-GO: cached hardware helpers came from another source; restart kernel")
    operator.check_source(REPO, SOURCE)
    DEVICES = operator.inventory()
    operator.admit_devices(DEVICES, operator.PLANS[PLATFORM].device_count)
    ENVIRONMENTS = operator.check_torch(REPO, DEVICES)
    if importlib.util.find_spec("matplotlib") is None:
        raise RuntimeError("NO-GO: native image needs matplotlib for result charts")
    if operator.fixture(PLATFORM)[0].identity != EXPECTED_TRAIN or operator.fixture(PLATFORM)[1].identity != EXPECTED_EVALUATION:
        raise RuntimeError("NO-GO: frozen action fixture drift")
{mounting}    operator.safe_id(RUN_ID)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    WORKSPACE = SCRATCH / (RUN_ID + "-" + uuid.uuid4().hex)
    WORKSPACE.mkdir()
    DESTINATION = DURABLE_BASE / RUN_ID
    DESTINATION.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(WORKSPACE).free < 4 * operator.GIB or shutil.disk_usage(DESTINATION).free < 4 * operator.GIB:
        raise RuntimeError("NO-GO: require at least4 GiB scratch and durable free space")
    STATUS = {{"status": "PREFLIGHT_PASS", "environment": ENVIRONMENTS}}
print(STATUS)
"""
    yield "code", """# Owner-enabled CUDA formation, all-arm timings and exact/cold-process recovery.
if RUN_CANARIES:
    CANARIES = operator.run_canaries(REPO, PLATFORM, SOURCE, WORKSPACE, DESTINATION / "qualification")
    print({"status": "CANARIES_PASS", "fresh_session": "still requires reattached prior VM evidence",
           "challenges": [row["results"][0]["challenge"] for row in CANARIES]})
"""
    yield "code", """# Fresh VM reattachment. No persistent pointer moves until payload verification.
if RUN_RECOVERY:
    if not FRESH_SESSION_ACK or len(RECOVERY_INPUTS) != len(DEVICES):
        raise RuntimeError("NO-GO: owner must acknowledge fresh VM and supply a saved challenge/mirror for every GPU")
    for lane, (gpu, prior) in enumerate(zip(DEVICES, RECOVERY_INPUTS)):
        receipt = DESTINATION / ("recovery-lane-" + str(lane) + "-" + uuid.uuid4().hex + ".json")
        operator.command([sys.executable, "-m", "tools.elendril_action_cloud", "recover",
            "--owner-enabled", "--off-session-ack", "--source", SOURCE, "--platform", PLATFORM,
            "--output", str(WORKSPACE / ("reattach-" + str(lane))), "--receipt", str(receipt),
            "--challenge", str(prior["challenge"]), "--mirror", str(prior["mirror"])],
            cwd=REPO, env=operator.device_env(gpu), timeout=240,
            log=DESTINATION / ("recovery-worker-" + uuid.uuid4().hex + ".log"))
        RECOVERY_RECEIPTS.append(receipt)
    operator.recovery_gate(RECOVERY_RECEIPTS, source=SOURCE, environments=ENVIRONMENTS)
    print({"status": "FRESH_SESSION_RECOVERY_PASS", "receipts": [str(p) for p in RECOVERY_RECEIPTS]})
"""
    yield "code", """# Full comparison only after current canaries, fresh-session recovery and owner launch.
if RUN_CAMPAIGN:
    if not RUN_CANARIES or not RUN_RECOVERY:
        raise RuntimeError("NO-GO: campaign requires current-session canaries and reattached fresh-session recovery")
    CAMPAIGN = operator.run_campaign(REPO, PLATFORM, SOURCE, WORKSPACE, DESTINATION,
        CANARIES, run_id=RUN_ID, recovery_receipts=RECOVERY_RECEIPTS,
        screen=SCREEN_CAMPAIGN, owner_reviewed=OWNER_REVIEWED_COLAB,
        elapsed=time.monotonic()-STARTED)
    import matplotlib.pyplot as plt
    points = {arm: [] for arm in operator.ARMS}
    for item in CAMPAIGN["results"]:
        result = json.loads((DESTINATION / item["file"]).read_bytes())
        score = result["evaluation"]["totals"]["all"]
        points[result["arm"]].append(score["utility"] / score["episodes"])
    fig, ax = plt.subplots(figsize=(7,4))
    for index, (arm, values) in enumerate(points.items()):
        ax.scatter([index] * len(values), values, label=arm)
        ax.plot([index-.15,index+.15], [sum(values)/len(values)]*2, color="black")
    ax.set_xticks(range(len(points)), list(points)); ax.set_ylabel("Correctness minus 0.1 × requests / episode")
    ax.set_title("Synthetic development action utility; individual seeds")
    fig.tight_layout(); fig.savefig(DESTINATION / "action-utility.png", dpi=160)
    plt.show()
    print({"status": "COMPLETE", "directory": str(DESTINATION), "promotion": "NONE"})
else:
    print({"status": "CAMPAIGN_DISABLED", "action": "Review and preserve qualification evidence."})
"""
    yield "markdown", """## Preserve and review

Keep the complete durable directory, including immutable checkpoint objects,
update logs, raw episode predictions, failures, challenges and recovery receipts.
For Kaggle explicitly save output, then reattach or retrieve that saved version;
for Colab keep the Drive directory and verify it in a fresh VM. Mounted-storage
readback and a fresh subprocess alone do not prove off-session persistence.
Never delete the last accepted generation when repairing a failure. A failed
forecast or missing recovery proof is NO-GO, not permission to lower a gate.
Return `campaign.json`, every referenced result, logs, charts, recovery proofs and
checkpoint lineage to the repository owner. No cloud outcome has been observed
until these hardware runs actually complete. Budgets do not override account limits.
"""


def notebook(platform, source):
    return {"nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"display_name": "Python3", "language": "python", "name": "python3"}},
            "cells": [{"cell_type": kind, "id": f"action-{index}", "metadata": {},
                       "source": text.splitlines(keepends=True),
                       **({"execution_count": None, "outputs": []} if kind == "code" else {})}
                      for index, (kind, text) in enumerate(cells(platform, source))]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", type=Path, default=Path("notebooks"))
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    for platform, name in FILES.items():
        (args.output / name).write_text(json.dumps(notebook(platform, args.source), indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
