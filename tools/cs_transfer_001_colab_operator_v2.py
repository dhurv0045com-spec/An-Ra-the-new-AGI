#!/usr/bin/env python3
"""Fail-closed Colab operator for CS-TRANSFER-001.

Operator-only wrapper. Scientific executable is always detached at the frozen
commit below. The moving branch is used only to obtain this operator and the
pre-execution qualification-fixture repair.
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import zipfile

SCIENCE = "a916d1c8d2637abb86d16b1c78e418c95461f3c7"
REPAIR_COMMIT = "acd1d51421466eafe208c71a4f7be70c229d2a53"
REPAIR_TEST_BLOB = "1a17fe9d29c48273d8c92e6320edf577bff6f19b"
ORIGINAL_TEST_BLOB = "ebd5d25962efa6a0d9a3ddb6d67d49843b76b6d6"
REPO_URL = "https://github.com/dhurv0045com-spec/An-Ra-the-new-AGI.git"
REPO = pathlib.Path("/content/An-Ra-the-new-AGI-cs-transfer-001-science")
ROOT = pathlib.Path("/content/drive/MyDrive/CYMEK/CS_TRANSFER_001")

EXPECTED_BLOBS = {
    "anra_v5/cs_transfer_001_run_v3.py": "3ecdcf317a3e3fe22398d0302ccd4193e09f110c",
    "anra_v5/cs_transfer_001_run_v2.py": "5f372b21358e1eb6fe8a8cae2b16b9c5aada6020",
    "anra_v5/cs_transfer_001_run.py": "ee60921f2bf9b4465b16d4637338f1dcec9c4865",
    "anra_v5/cs_transfer_001_data.py": "b53b7d1ab9661853fbce5a5860882f6e0cdea48c",
    "anra_v5/cs_transfer_001_model.py": "35806598fc9119cd76a88ea265d38a716437062c",
    "experiments/CS_TRANSFER_001/PREREGISTRATION.json": "6f2fb6210adf3dd5fb5dfeac40260fdd12562214",
    "experiments/CS_TRANSFER_001/AMENDMENT_1.json": "55c1f5dff370c06f5d6b9f31bf92222e2c12a983",
    "tools/validate_cs_transfer_001.py": "26339b9ef5d4599acb92327ea170bad64e45ae99",
    "tests/test_cs_transfer_001.py": ORIGINAL_TEST_BLOB,
    "tests/test_cs_transfer_001_runner.py": "e7962a2a1ea4010817f8bc9d19bb449cd3bf565d",
    "tests/test_cs_transfer_001_protocol.py": "f0fd8833b2b624eecde4b2f2e891b8300a1c9e53",
    "v5_model/core.py": "7cf64b6f557a0556c074e5f61adfc86702f4c725",
    "v5_training/production_backend.py": "72646373c1890e19deaeef634b3f4a0dbdf99632",
    "v5_training/checkpoint.py": "6bd1d04dad43e402b9acff9128d1ad56528a37d0",
    "v5_training/step.py": "bf1d0411be249d2e6e4f8634381124e4f6c7e92f",
    "v5_data/corpus_loading.py": "2c5e64656d9241d2c916549c9fb938e1df848ee0",
}


def run(cmd, *, cwd=None, capture=False, env=None):
    print("\n$", " ".join(map(str, cmd)), flush=True)
    if capture:
        p = subprocess.run(cmd, cwd=cwd, env=env, text=True, capture_output=True)
        print(p.stdout, end="")
        if p.stderr.strip():
            print(p.stderr, file=sys.stderr)
    else:
        p = subprocess.run(cmd, cwd=cwd, env=env)
    if p.returncode:
        raise RuntimeError(f"FAIL_CLOSED: command returned {p.returncode}: {' '.join(map(str, cmd))}")
    return p


def git(*args, text=True):
    return subprocess.check_output(["git", "-C", str(REPO), *args], text=text).strip()


def exact_checkout() -> None:
    if REPO.exists():
        run(["git", "-C", str(REPO), "fetch", "origin", SCIENCE, "--depth", "1"])
    else:
        run(["git", "clone", "--filter=blob:none", "--no-checkout", REPO_URL, str(REPO)])
        run(["git", "-C", str(REPO), "fetch", "origin", SCIENCE, "--depth", "1"])
    run(["git", "-C", str(REPO), "checkout", "--detach", SCIENCE])
    assert git("rev-parse", "HEAD") == SCIENCE
    print("SCIENTIFIC EXECUTABLE:", SCIENCE)
    for path, expected in EXPECTED_BLOBS.items():
        got = git("rev-parse", f"HEAD:{path}")
        if got != expected:
            raise RuntimeError(f"FAIL_CLOSED blob drift {path}: {got} != {expected}")
    print("CRITICAL BLOBS: PASS", len(EXPECTED_BLOBS))


def cpu_qualification() -> None:
    run([sys.executable, "-m", "pip", "install", "-q", "tokenizers", "pytest"])
    # Correct module invocation; file-path invocation breaks package imports in clean Colab.
    run([sys.executable, "-m", "tools.validate_cs_transfer_001"], cwd=REPO)
    print("VALIDATOR: PASS")

    test_rel = "tests/test_cs_transfer_001.py"
    test_path = REPO / test_rel
    run(["git", "-C", str(REPO), "fetch", "origin", REPAIR_COMMIT, "--depth", "1"])
    repair_blob = git("rev-parse", f"{REPAIR_COMMIT}:{test_rel}")
    if repair_blob != REPAIR_TEST_BLOB:
        raise RuntimeError("FAIL_CLOSED: qualification-repair blob mismatch")
    original = subprocess.check_output(["git", "-C", str(REPO), "show", f"HEAD:{test_rel}"])
    repaired = subprocess.check_output(["git", "-C", str(REPO), "show", f"{REPAIR_COMMIT}:{test_rel}"])
    try:
        test_path.write_bytes(repaired)
        if git("hash-object", str(test_path)) != REPAIR_TEST_BLOB:
            raise RuntimeError("FAIL_CLOSED: repaired test bytes mismatch")
        run([sys.executable, "-m", "pytest", test_rel, "-q", "--tb=short"], cwd=REPO)
    finally:
        test_path.write_bytes(original)
    if git("hash-object", str(test_path)) != ORIGINAL_TEST_BLOB:
        raise RuntimeError("FAIL_CLOSED: frozen qualification test not restored")
    if git("status", "--porcelain") != "":
        raise RuntimeError("FAIL_CLOSED: scientific worktree dirty after qualification repair")

    for t in (
        "tests/test_cs_transfer_001_runner.py",
        "tests/test_cs_transfer_001_protocol.py",
        "tests/test_v5_production_backend.py",
        "tests/test_v5_checkpoint_adapter.py",
    ):
        run([sys.executable, "-m", "pytest", t, "-q", "--tb=short"], cwd=REPO)
    if git("rev-parse", "HEAD") != SCIENCE or git("status", "--porcelain") != "":
        raise RuntimeError("FAIL_CLOSED: frozen science identity changed during qualification")
    print("CPU QUALIFICATION: PASS")


def runner(args, *, capture=False):
    env = os.environ.copy()
    env["CS_TRANSFER_001_ROOT"] = str(ROOT)
    cmd = [sys.executable, "-m", "anra_v5.cs_transfer_001_run_v3", *args]
    return run(cmd, cwd=REPO, capture=capture, env=env)


def load_json(path: pathlib.Path):
    return json.loads(path.read_text())


def prepare_and_gate() -> None:
    if not pathlib.Path("/content/drive/MyDrive").exists():
        raise RuntimeError("FAIL_CLOSED: Google Drive is not mounted at /content/drive")
    ROOT.mkdir(parents=True, exist_ok=True)
    runner(["--mode", "protocol"])
    data_receipt = ROOT / "receipts" / "DATA.json"
    if not data_receipt.exists():
        runner(["--mode", "prepare"])
    d = load_json(data_receipt)
    surface = d["shared_surface"]
    if surface["contamination"]["clean"] is not True:
        raise RuntimeError("FAIL_CLOSED DATA: contamination screen failed")
    if float(surface["predictive_shortcut_max"]) >= 0.35:
        raise RuntimeError("FAIL_CLOSED DATA: predictive shortcut threshold failed")
    if d["maximum_allowed_content_id"] != 4095 or d["identical_token_sequences_required"] is not True:
        raise RuntimeError("FAIL_CLOSED DATA: common-token contract failed")
    for split, fams in surface["acceptance"].items():
        for fam, row in fams.items():
            if float(row["acceptance_rate"]) < 0.15:
                raise RuntimeError(f"FAIL_CLOSED DATA acceptance {split}/{fam}: {row}")
    for split, stats in surface["token_stats"].items():
        if stats["all_lt_4096"] is not True or int(stats["maximum_content_id"]) >= 4096:
            raise RuntimeError(f"FAIL_CLOSED DATA token range {split}: {stats}")
    print("DATA GATES: PASS")


def cuda_preflight() -> None:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("FAIL_CLOSED: select a T4 GPU runtime")
    print("GPU:", torch.cuda.get_device_name(0))
    runs_root = ROOT / "runs"
    def heads():
        if not runs_root.exists():
            return {}
        return {str(p.relative_to(ROOT)): p.read_text().strip()
                for p in sorted(runs_root.glob("pair_*/*/state/*/LATEST"))}
    before = heads()
    runner(["--mode", "preflight", "--cuda"])
    after = heads()
    if after != before:
        raise RuntimeError("FAIL_CLOSED: preflight changed scientific checkpoint heads")
    gc.collect(); torch.cuda.empty_cache()
    print("CUDA PREFLIGHT: PASS")


def execute_campaign() -> None:
    runner(["--mode", "scan"])
    for pair in range(4):
        for arm in ("PHYS_4096", "PHYS_24576"):
            print("\n" + "=" * 72)
            print(f"PAIR {pair} / {arm}")
            print("=" * 72)
            runner(["--mode", "run-arm", "--pair-index", str(pair), "--arm", arm, "--cuda"])
            runner(["--mode", "scan"])
    runner(["--mode", "development"])

    final_path = ROOT / "receipts" / "FINAL_RESULT.json"
    sealed_marker = ROOT / "SEALED_CONSUMPTION.json"
    if final_path.exists():
        print("FINAL_RESULT already exists; sealed evaluation will not be repeated.")
    elif sealed_marker.exists():
        raise RuntimeError("FAIL_CLOSED: sealed marker exists without FINAL_RESULT; do not delete it")
    else:
        runner(["--mode", "finalize", "--cuda"])
    if not final_path.exists():
        raise RuntimeError("FAIL_CLOSED: finalizer returned without FINAL_RESULT")
    print(json.dumps(load_json(final_path), indent=2)[:20000])


def package() -> None:
    out = pathlib.Path("/content/CS_TRANSFER_001_RESULTS.zip")
    if out.exists():
        out.unlink()
    included = []
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(ROOT.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(ROOT)
            parts = set(rel.parts)
            if "state" in parts or "data" in parts:
                continue
            z.write(path, arcname=str(rel))
            included.append(str(rel))
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    print("RESULT BUNDLE:", out)
    print("FILES:", len(included))
    print("SHA256:", sha)
    print("SIZE MiB:", round(out.stat().st_size / 2**20, 2))


def main() -> int:
    exact_checkout()
    cpu_qualification()
    prepare_and_gate()
    cuda_preflight()
    execute_campaign()
    package()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
