#!/usr/bin/env python3
"""One-click, resumable Colab operator for CS-TRANSFER-001 Amendment 2.

Engineering-only operator revision. Scientific execution remains pinned to
f8582808b6e2be0753cb2689d9d6d4aeb4d57aeb and runner
anra_v5.cs_transfer_001_run_v4.

Goals:
- never reuse the failed pre-A2 Drive root;
- qualify the exact science end-to-end before starting any arm;
- preserve/resume valid A2 checkpoints rather than deleting them;
- fail closed on protocol/data/checkpoint drift;
- retry a failed arm invocation once only after safe cache cleanup, relying on
  the runner's exact-resume transaction semantics;
- write useful failure diagnostics to both /content and Drive.
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
import time
import traceback
import zipfile

SCIENCE = "f8582808b6e2be0753cb2689d9d6d4aeb4d57aeb"
REPO_URL = "https://github.com/dhurv0045com-spec/An-Ra-the-new-AGI.git"
REPO = Path("/content/An-Ra-the-new-AGI-cs-transfer-a2-science")
LEGACY_DRIVE_ROOT = Path("/content/drive/MyDrive/CYMEK/CS_TRANSFER_001")
DRIVE_ROOT = Path("/content/drive/MyDrive/CYMEK/CS_TRANSFER_001_A2")
DRY_ROOT = Path("/content/CS_TRANSFER_001_A2_DRYRUN")
FAILURE_FILE = Path("/content/CS_TRANSFER_001_A2_FAILURE.txt")
RUNNER = "anra_v5.cs_transfer_001_run_v4"
SURFACE_REVISION = "A2_DIRECT_TOKEN_COMMON_SPACE"
ARMS = ("PHYS_4096", "PHYS_24576")
PAIR_COUNT = 4

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


def run(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None,
        capture: bool = False) -> subprocess.CompletedProcess[str]:
    print("\n$", " ".join(map(str, cmd)), flush=True)
    p = subprocess.run(cmd, cwd=cwd, env=env, text=True,
                       capture_output=capture)
    if capture:
        if p.stdout:
            print(p.stdout, end="")
        if p.stderr:
            print(p.stderr, file=sys.stderr, end="")
    if p.returncode:
        raise RuntimeError(f"FAIL_CLOSED rc={p.returncode}: {' '.join(map(str, cmd))}")
    return p


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(REPO), *args], text=True).strip()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def phase(name: str) -> None:
    print("\n" + "=" * 80)
    print(name)
    print("=" * 80, flush=True)


def environment_gate() -> None:
    phase("0 / ENVIRONMENT GATE")
    if sys.version_info < (3, 10):
        raise RuntimeError(f"FAIL_CLOSED: Python >=3.10 required, got {sys.version}")
    run(["git", "--version"])
    run([sys.executable, "-c", "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"])
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("FAIL_CLOSED HARDWARE: select a Colab T4 GPU runtime before running")
    print("ENVIRONMENT: PASS")


def checkout_exact_science() -> None:
    phase("1 / EXACT SCIENCE CHECKOUT")
    if REPO.exists():
        shutil.rmtree(REPO)
    run(["git", "clone", "--filter=blob:none", "--no-checkout", REPO_URL, str(REPO)])
    run(["git", "-C", str(REPO), "fetch", "origin", SCIENCE, "--depth", "1"])
    run(["git", "-C", str(REPO), "checkout", "--detach", SCIENCE])
    if git("rev-parse", "HEAD") != SCIENCE:
        raise RuntimeError("FAIL_CLOSED: wrong scientific executable checkout")
    for path, expected in EXPECTED_BLOBS.items():
        got = git("rev-parse", f"HEAD:{path}")
        if got != expected:
            raise RuntimeError(f"FAIL_CLOSED blob drift {path}: {got} != {expected}")
    print("SCIENTIFIC EXECUTABLE: PASS", SCIENCE)
    print("CRITICAL BLOBS: PASS", len(EXPECTED_BLOBS))


def qualification_gate() -> None:
    phase("2 / STATIC + CPU QUALIFICATION")
    run([sys.executable, "-m", "pip", "install", "-q", "tokenizers", "pytest"])
    run([sys.executable, "-m", "compileall", "-q", "anra_v5", "v5_model", "v5_training", "v5_objectives", "tools"], cwd=REPO)
    run([sys.executable, "-m", "tools.validate_cs_transfer_001_v4"], cwd=REPO)
    for test_file in (
        "tests/test_cs_transfer_001_v4.py",
        "tests/test_v5_model.py",
        "tests/test_v5_production_backend.py",
        "tests/test_v5_checkpoint_adapter.py",
    ):
        run([sys.executable, "-m", "pytest", test_file, "-q", "--tb=short"], cwd=REPO)
    tracked = git("status", "--porcelain", "--untracked-files=no")
    if tracked:
        raise RuntimeError("FAIL_CLOSED: tracked scientific checkout changed during qualification\n" + tracked)
    print("STATIC/CPU QUALIFICATION: PASS")


def runner(root: Path, args: list[str], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["CS_TRANSFER_001_ROOT"] = str(root)
    return run([sys.executable, "-m", RUNNER, *args], cwd=REPO, env=env, capture=capture)


def validate_data_receipt(root: Path) -> dict:
    path = root / "receipts" / "DATA.json"
    if not path.is_file():
        raise RuntimeError(f"FAIL_CLOSED: DATA receipt missing at {path}")
    d = load_json(path)
    m = d.get("shared_surface", {})
    if m.get("surface_revision") != SURFACE_REVISION:
        raise RuntimeError(f"FAIL_CLOSED: wrong data surface revision at {path}")
    if m.get("contamination", {}).get("clean") is not True:
        raise RuntimeError("FAIL_CLOSED: data contamination gate failed")
    if float(m.get("predictive_shortcut_max", 1.0)) >= 0.35:
        raise RuntimeError("FAIL_CLOSED: shortcut threshold failed")
    acceptance = m.get("acceptance", {})
    for split in ("training", "development", "sealed"):
        if split not in acceptance:
            raise RuntimeError(f"FAIL_CLOSED: missing acceptance split {split}")
        for family, row in acceptance[split].items():
            if float(row.get("acceptance_rate", -1)) != 1.0:
                raise RuntimeError(f"FAIL_CLOSED: direct-token acceptance {split}/{family}={row}")
    for split, stats in m.get("token_stats", {}).items():
        if stats.get("all_lt_4096") is not True or int(stats.get("maximum_content_id", 999999)) >= 4096:
            raise RuntimeError(f"FAIL_CLOSED: token range {split}: {stats}")
    print("DATA RECEIPT: PASS", path)
    return d


def exhaustive_local_dry_run() -> None:
    phase("3 / FULL LOCAL END-TO-END DRY RUN")
    if DRY_ROOT.exists():
        shutil.rmtree(DRY_ROOT)
    DRY_ROOT.mkdir(parents=True, exist_ok=True)
    runner(DRY_ROOT, ["--mode", "protocol"])
    runner(DRY_ROOT, ["--mode", "prepare"])
    validate_data_receipt(DRY_ROOT)
    runner(DRY_ROOT, ["--mode", "preflight"])
    runner(DRY_ROOT, ["--mode", "scan"])
    print("LOCAL PREPARE + CPU PRODUCTION PREFLIGHT: PASS")


def ensure_drive() -> None:
    phase("4 / DRIVE STATE GATE")
    mydrive = Path("/content/drive/MyDrive")
    if not mydrive.exists():
        raise RuntimeError("FAIL_CLOSED: Google Drive is not mounted")
    DRIVE_ROOT.mkdir(parents=True, exist_ok=True)
    print("A2 DRIVE ROOT:", DRIVE_ROOT)
    if LEGACY_DRIVE_ROOT.exists():
        print("LEGACY ROOT PRESERVED (not touched):", LEGACY_DRIVE_ROOT)

    data = DRIVE_ROOT / "receipts" / "DATA.json"
    final = DRIVE_ROOT / "receipts" / "FINAL_RESULT.json"
    sealed = DRIVE_ROOT / "SEALED_CONSUMPTION.json"
    if data.exists():
        validate_data_receipt(DRIVE_ROOT)
    elif (DRIVE_ROOT / "runs").exists() and any((DRIVE_ROOT / "runs").rglob("LATEST")):
        raise RuntimeError("FAIL_CLOSED: A2 checkpoints exist without an A2 DATA receipt")
    if sealed.exists() and not final.exists():
        raise RuntimeError("FAIL_CLOSED: sealed marker exists without FINAL_RESULT; preserve state and investigate")
    print("DRIVE STATE: PASS")


def drive_prepare_and_cuda_preflight() -> None:
    phase("5 / DRIVE PREPARE + CUDA PREFLIGHT")
    data = DRIVE_ROOT / "receipts" / "DATA.json"
    if not data.exists():
        runner(DRIVE_ROOT, ["--mode", "prepare"])
    validate_data_receipt(DRIVE_ROOT)

    import torch
    runs_root = DRIVE_ROOT / "runs"

    def heads() -> dict[str, str]:
        if not runs_root.exists():
            return {}
        out: dict[str, str] = {}
        for p in sorted(runs_root.rglob("LATEST")):
            if "state" not in p.parts:
                continue
            out[str(p.relative_to(DRIVE_ROOT))] = p.read_text(encoding="ascii").strip()
        return out

    before = heads()
    runner(DRIVE_ROOT, ["--mode", "preflight", "--cuda"])
    after = heads()
    if after != before:
        raise RuntimeError("FAIL_CLOSED: CUDA preflight changed scientific checkpoint heads")
    gc.collect()
    torch.cuda.empty_cache()
    runner(DRIVE_ROOT, ["--mode", "scan"])
    print("DRIVE PREPARE + CUDA PREFLIGHT: PASS")


def arm_complete(pair: int, arm: str) -> bool:
    receipt = DRIVE_ROOT / "runs" / f"pair_{pair}" / arm / "receipts" / "ARM_RESULT.json"
    if not receipt.exists():
        return False
    try:
        d = load_json(receipt)
    except Exception:
        return False
    return d.get("status") == "COMPLETE" and int(d.get("global_update", -1)) == 480


def run_arm_resumable(pair: int, arm: str) -> None:
    if arm_complete(pair, arm):
        print(f"PAIR {pair} / {arm}: already COMPLETE; skip")
        return
    import torch
    last_exc: Exception | None = None
    for attempt in (1, 2):
        try:
            print(f"PAIR {pair} / {arm}: invocation attempt {attempt}/2")
            runner(DRIVE_ROOT, ["--mode", "run-arm", "--pair-index", str(pair), "--arm", arm, "--cuda"])
            runner(DRIVE_ROOT, ["--mode", "scan"])
            if not arm_complete(pair, arm):
                raise RuntimeError(f"FAIL_CLOSED: arm returned without complete endpoint receipt: pair={pair} arm={arm}")
            return
        except Exception as exc:
            last_exc = exc
            if attempt == 2:
                break
            print("First invocation failed. Clearing only transient CUDA/Python caches; scientific state is preserved.")
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(2)
    assert last_exc is not None
    raise last_exc


def execute_campaign() -> None:
    phase("6 / EIGHT-ARM FIXED-ENDPOINT CAMPAIGN")
    for pair in range(PAIR_COUNT):
        for arm in ARMS:
            print("\n" + "-" * 80)
            print(f"PAIR {pair} / {arm}")
            print("-" * 80)
            run_arm_resumable(pair, arm)
    print("ALL EIGHT ARMS: COMPLETE")


def finalize() -> None:
    phase("7 / DEVELOPMENT FREEZE + ONE-SHOT SEALED FINALIZE")
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
        raise RuntimeError("FAIL_CLOSED: final result missing after finalize")
    result = load_json(final_path)
    print(json.dumps(result, indent=2)[:30000])


def package_results() -> Path:
    phase("8 / PACKAGE RESULTS")
    out = Path("/content/CS_TRANSFER_001_A2_RESULTS.zip")
    if out.exists():
        out.unlink()
    included: list[str] = []
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(DRIVE_ROOT.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(DRIVE_ROOT)
            if "state" in rel.parts or "data" in rel.parts:
                continue
            z.write(path, arcname=str(rel))
            included.append(str(rel))
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    print("RESULT BUNDLE:", out)
    print("FILES:", len(included))
    print("SHA256:", sha)
    print("SIZE MiB:", round(out.stat().st_size / 2**20, 2))
    return out


def write_failure(exc: BaseException) -> None:
    text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    FAILURE_FILE.write_text(text, encoding="utf-8")
    payload = {
        "schema": "anra-cs-transfer-001-operator-failure/v1",
        "science_commit": SCIENCE,
        "drive_root": str(DRIVE_ROOT),
        "exception_type": type(exc).__name__,
        "message": str(exc),
        "traceback": text,
        "unix_time": time.time(),
    }
    try:
        if Path("/content/drive/MyDrive").exists():
            atomic_json(DRIVE_ROOT / "operator_logs" / "last_failure.json", payload)
    except Exception:
        pass
    print("\nFAILURE DIAGNOSTIC:", FAILURE_FILE, file=sys.stderr)


def main() -> int:
    if FAILURE_FILE.exists():
        FAILURE_FILE.unlink()
    try:
        environment_gate()
        checkout_exact_science()
        qualification_gate()
        exhaustive_local_dry_run()
        ensure_drive()
        drive_prepare_and_cuda_preflight()
        execute_campaign()
        finalize()
        package_results()
        phase("COMPLETE")
        print("CS-TRANSFER-001 A2 completed successfully.")
        print("Scientific Drive state:", DRIVE_ROOT)
        return 0
    except BaseException as exc:
        write_failure(exc)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
