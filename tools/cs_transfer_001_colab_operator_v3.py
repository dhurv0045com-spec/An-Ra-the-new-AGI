#!/usr/bin/env python3
"""Single-shot fail-closed Colab operator for CS-TRANSFER-001 Amendment 2.

This operator exists because the original notebook exposed failures one gate at
a time. v3 performs the entire qualification chain before the expensive GPU
campaign: exact checkout, static validation, full direct-surface qualification,
real production-tokenizer dry-run prepare on local disk, CPU production preflight,
Drive compatibility check, Drive prepare, CUDA preflight, then all eight arms.
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

SCIENCE = "f8582808b6e2be0753cb2689d9d6d4aeb4d57aeb"
REPO_URL = "https://github.com/dhurv0045com-spec/An-Ra-the-new-AGI.git"
REPO = Path("/content/An-Ra-the-new-AGI-cs-transfer-001-a2")
DRIVE_ROOT = Path("/content/drive/MyDrive/CYMEK/CS_TRANSFER_001")
DRY_ROOT = Path("/content/CS_TRANSFER_001_A2_DRYRUN")
RUNNER = "anra_v5.cs_transfer_001_run_v4"

EXPECTED_BLOBS = {
    "anra_v5/cs_transfer_001_data_v2.py": "d8caeb47dfd0b925ab58c7af5c9f420bedeb0a48",
    "anra_v5/cs_transfer_001_run_v4.py": "5a01171e8fa4948873b9dd701a6a664c83e58686",
    "experiments/CS_TRANSFER_001/AMENDMENT_2.json": "32a75f00d2364b434a615aa78cdafedbbde58d39",
    "experiments/CS_TRANSFER_001/AMENDMENT_1.json": "55c1f5dff370c06f5d6b9f31bf92222e2c12a983",
    "experiments/CS_TRANSFER_001/PREREGISTRATION.json": "6f2fb6210adf3dd5fb5dfeac40260fdd12562214",
    "tests/test_cs_transfer_001_v4.py": "fa13ab40fb1d26cb6e47237806a5ac10e9671ee0",
    "tools/validate_cs_transfer_001_v4.py": "4502b3498494ff6a21f192545a3bb86c3fdcf743",
    "anra_v5/cs_transfer_001_model.py": "35806598fc9119cd76a88ea265d38a716437062c",
    "anra_v5/cs_transfer_001_run_v2.py": "5f372b21358e1eb6fe8a8cae2b16b9c5aada6020",
    "v5_training/production_backend.py": "72646373c1890e19deaeef634b3f4a0dbdf99632",
    "v5_training/checkpoint.py": "6bd1d04dad43e402b9acff9128d1ad56528a37d0",
    "v5_training/step.py": "bf1d0411be249d2e6e4f8634381124e4f6c7e92f",
}


def run(cmd, *, cwd=None, env=None, capture=False):
    print("\n$", " ".join(map(str, cmd)), flush=True)
    if capture:
        p = subprocess.run(cmd, cwd=cwd, env=env, text=True, capture_output=True)
        print(p.stdout, end="")
        if p.stderr.strip():
            print(p.stderr, file=sys.stderr)
    else:
        p = subprocess.run(cmd, cwd=cwd, env=env)
    if p.returncode:
        raise RuntimeError(f"FAIL_CLOSED rc={p.returncode}: {' '.join(map(str, cmd))}")
    return p


def git(*args):
    return subprocess.check_output(["git", "-C", str(REPO), *args], text=True).strip()


def checkout_and_qualify() -> None:
    if REPO.exists():
        run(["git", "-C", str(REPO), "fetch", "origin", SCIENCE, "--depth", "1"])
    else:
        run(["git", "clone", "--filter=blob:none", "--no-checkout", REPO_URL, str(REPO)])
        run(["git", "-C", str(REPO), "fetch", "origin", SCIENCE, "--depth", "1"])
    run(["git", "-C", str(REPO), "checkout", "--detach", SCIENCE])
    if git("rev-parse", "HEAD") != SCIENCE:
        raise RuntimeError("FAIL_CLOSED: wrong scientific executable")
    for path, expected in EXPECTED_BLOBS.items():
        got = git("rev-parse", f"HEAD:{path}")
        if got != expected:
            raise RuntimeError(f"FAIL_CLOSED blob drift {path}: {got} != {expected}")
    print("SCIENTIFIC EXECUTABLE: PASS", SCIENCE)
    print("CRITICAL BLOBS: PASS", len(EXPECTED_BLOBS))

    run([sys.executable, "-m", "pip", "install", "-q", "tokenizers", "pytest"])
    run([sys.executable, "-m", "tools.validate_cs_transfer_001_v4"], cwd=REPO)
    for test_file in (
        "tests/test_cs_transfer_001_v4.py",
        "tests/test_v5_production_backend.py",
        "tests/test_v5_checkpoint_adapter.py",
    ):
        run([sys.executable, "-m", "pytest", test_file, "-q", "--tb=short"], cwd=REPO)
    if git("status", "--porcelain"):
        raise RuntimeError("FAIL_CLOSED: scientific checkout dirty after qualification")
    print("STATIC/CPU QUALIFICATION: PASS")


def runner(root: Path, args: list[str], *, capture=False):
    env = os.environ.copy()
    env["CS_TRANSFER_001_ROOT"] = str(root)
    return run([sys.executable, "-m", RUNNER, *args], cwd=REPO, env=env, capture=capture)


def load_json(path: Path):
    return json.loads(path.read_text())


def validate_data_receipt(root: Path) -> None:
    path = root / "receipts" / "DATA.json"
    if not path.is_file():
        raise RuntimeError(f"FAIL_CLOSED: DATA receipt missing at {path}")
    d = load_json(path)
    m = d["shared_surface"]
    if m.get("surface_revision") != "A2_DIRECT_TOKEN_COMMON_SPACE":
        raise RuntimeError("FAIL_CLOSED: wrong data surface revision")
    if m["contamination"]["clean"] is not True:
        raise RuntimeError("FAIL_CLOSED: contamination")
    if float(m["predictive_shortcut_max"]) >= 0.35:
        raise RuntimeError("FAIL_CLOSED: shortcut threshold")
    for split, fams in m["acceptance"].items():
        for family, row in fams.items():
            if float(row["acceptance_rate"]) != 1.0:
                raise RuntimeError(f"FAIL_CLOSED: direct-token acceptance {split}/{family}={row}")
    for split, stats in m["token_stats"].items():
        if stats["all_lt_4096"] is not True or int(stats["maximum_content_id"]) >= 4096:
            raise RuntimeError(f"FAIL_CLOSED: token range {split}: {stats}")
    print("DATA RECEIPT: PASS", m["surface_revision"], "shortcut_max=", m["predictive_shortcut_max"])


def exhaustive_preflight_before_drive() -> None:
    if DRY_ROOT.exists():
        shutil.rmtree(DRY_ROOT)
    DRY_ROOT.mkdir(parents=True)
    runner(DRY_ROOT, ["--mode", "protocol"])
    runner(DRY_ROOT, ["--mode", "prepare"])
    validate_data_receipt(DRY_ROOT)
    # This goes through the real model/backend/optimizer on CPU and catches
    # orchestration/data/model integration errors before Drive or GPU science.
    runner(DRY_ROOT, ["--mode", "preflight"])
    runner(DRY_ROOT, ["--mode", "scan"])
    print("LOCAL FULL PREPARE + CPU PREFLIGHT: PASS")


def ensure_drive_compatible() -> None:
    if not Path("/content/drive/MyDrive").exists():
        raise RuntimeError("FAIL_CLOSED: Google Drive is not mounted")
    DRIVE_ROOT.mkdir(parents=True, exist_ok=True)
    # Failed old prepares may leave only an empty folder. That is safe.
    incompatible = []
    old_data = DRIVE_ROOT / "receipts" / "DATA.json"
    if old_data.exists():
        try:
            data = load_json(old_data)
            if data.get("shared_surface", {}).get("surface_revision") != "A2_DIRECT_TOKEN_COMMON_SPACE":
                incompatible.append(str(old_data))
        except Exception:
            incompatible.append(str(old_data))
    if (DRIVE_ROOT / "runs").exists() and any((DRIVE_ROOT / "runs").rglob("LATEST")) and not old_data.exists():
        incompatible.append("runs/*/LATEST without Amendment-2 DATA receipt")
    if incompatible:
        raise RuntimeError(
            "FAIL_CLOSED: incompatible prior scientific state exists. Do not delete it manually. "
            + repr(incompatible)
        )


def drive_prepare_and_cuda_preflight() -> None:
    data_receipt = DRIVE_ROOT / "receipts" / "DATA.json"
    if not data_receipt.exists():
        runner(DRIVE_ROOT, ["--mode", "prepare"])
    validate_data_receipt(DRIVE_ROOT)

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("FAIL_CLOSED: select T4 GPU runtime")
    print("GPU:", torch.cuda.get_device_name(0))

    runs_root = DRIVE_ROOT / "runs"
    def heads():
        if not runs_root.exists():
            return {}
        return {str(p.relative_to(DRIVE_ROOT)): p.read_text().strip()
                for p in sorted(runs_root.glob("pair_*/*/state/*/LATEST"))}
    before = heads()
    runner(DRIVE_ROOT, ["--mode", "preflight", "--cuda"])
    after = heads()
    if after != before:
        raise RuntimeError("FAIL_CLOSED: CUDA preflight changed scientific checkpoint heads")
    gc.collect(); torch.cuda.empty_cache()
    runner(DRIVE_ROOT, ["--mode", "scan"])
    print("DRIVE PREPARE + CUDA PREFLIGHT: PASS")


def execute_campaign() -> None:
    for pair in range(4):
        for arm in ("PHYS_4096", "PHYS_24576"):
            print("\n" + "=" * 76)
            print(f"PAIR {pair} / {arm}")
            print("=" * 76)
            runner(DRIVE_ROOT, ["--mode", "run-arm", "--pair-index", str(pair), "--arm", arm, "--cuda"])
            runner(DRIVE_ROOT, ["--mode", "scan"])
    runner(DRIVE_ROOT, ["--mode", "development"])

    final_path = DRIVE_ROOT / "receipts" / "FINAL_RESULT.json"
    sealed_marker = DRIVE_ROOT / "SEALED_CONSUMPTION.json"
    if final_path.exists():
        print("FINAL_RESULT already exists; sealed evaluation will not repeat.")
    elif sealed_marker.exists():
        raise RuntimeError("FAIL_CLOSED: sealed marker exists without FINAL_RESULT; do not delete it")
    else:
        runner(DRIVE_ROOT, ["--mode", "finalize", "--cuda"])
    if not final_path.exists():
        raise RuntimeError("FAIL_CLOSED: final result missing")
    print(json.dumps(load_json(final_path), indent=2)[:20000])


def package() -> None:
    out = Path("/content/CS_TRANSFER_001_A2_RESULTS.zip")
    if out.exists():
        out.unlink()
    included = []
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(DRIVE_ROOT.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(DRIVE_ROOT)
            if "state" in set(rel.parts) or "data" in set(rel.parts):
                continue
            z.write(path, arcname=str(rel))
            included.append(str(rel))
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    print("RESULT BUNDLE:", out)
    print("FILES:", len(included))
    print("SHA256:", sha)
    print("SIZE MiB:", round(out.stat().st_size / 2**20, 2))


def main() -> int:
    checkout_and_qualify()
    exhaustive_preflight_before_drive()
    ensure_drive_compatible()
    drive_prepare_and_cuda_preflight()
    execute_campaign()
    package()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
