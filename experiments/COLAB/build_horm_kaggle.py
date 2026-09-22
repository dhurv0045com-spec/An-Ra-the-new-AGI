"""Build the Kaggle T4x2 operator notebook for the HORM verification session.

The notebook is an operator only. Its HORM runner and test suite remain CPU
bound as specified by their existing protocols; the two Kaggle T4 devices are
checked and recorded, never silently substituted into CPU experiments.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path
import textwrap


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "notebooks" / "HORM-KAGGLE-T4X2.ipynb"

MARKDOWN = """# HORM verification session — Kaggle T4 ×2 (`cymek-beta`)

In Kaggle, select **Settings → Accelerator → GPU T4 x2** and turn **Internet on**. Run the cells in order.

This notebook moves the existing `HORM-colab-gpu.ipynb` engineering and verification workflow to Kaggle. The existing HORM protocols explicitly require CPU execution; the two T4s are checked and recorded but are not used to change those experiments. Pytest is also pinned to CPU to keep the run consistent with its receipts.

The main suite takes time. Each stage writes a log and an atomic status record under `/kaggle/working/HORM_KAGGLE_SESSION`. Completed stages are skipped when you rerun the cell. Failed stages keep their logs and can be retried. You can run the final packaging cell after interrupting a long stage to save the partial logs and status for recovery.

The source is pinned to the `cymek-beta` commit recorded below. The notebook never hard-resets an existing checkout. A different checkout is an error, so prior outputs are preserved. Results and logs are written outside the source tree. HORM-003/004 reruns are off by default because their prospective results already exist; enable those switches only when you intentionally want separately stored replications.

At the end, download the generated ZIP from the Kaggle output panel or use **Save Version** with notebook outputs enabled. To continue in a fresh Kaggle session, attach that ZIP as a Kaggle input and set `RECOVERY_BUNDLE` in Cell 1 to its path.
"""

BOOTSTRAP = r'''# CELL 1 — pin Cymek Beta, confirm Kaggle T4 x2, prepare resumable output
from pathlib import Path
import hashlib, json, os, platform, shutil, subprocess, sys

REMOTE = "https://github.com/dhurv0045com-spec/An-Ra-the-new-AGI.git"
BRANCH = "cymek-beta"
PINNED_COMMIT = "7b09985bba5692caeea46cc3187e8d16f1673cc4"
WORK = Path("/kaggle/working/HORM_KAGGLE_SESSION")
# Keep the full checkout out of Kaggle's saved Output bundle. The source is
# disposable and can be recloned after a runtime reset; run evidence lives in
# /kaggle/working and is what the user saves/downloads.
TEMP_ROOT = Path("/kaggle/temp")
REPO_DIR = TEMP_ROOT / ("cymek-beta-source-" + PINNED_COMMIT[:12])
RUN_DIR = WORK / "run" / PINNED_COMMIT
RECOVERY_BUNDLE = None  # e.g. "/kaggle/input/horm-session-recovery/HORM-....zip"
RUN_HORM003_REPLICATION = False
RUN_HORM004_REPLICATION = False

def _run(command, **kwargs):
    print("+", " ".join(map(str, command)), flush=True)
    return subprocess.run(list(map(str, command)), check=True, **kwargs)

WORK.mkdir(parents=True, exist_ok=True)
RUN_DIR.mkdir(parents=True, exist_ok=True)
TEMP_ROOT.mkdir(parents=True, exist_ok=True)

# Kaggle hardware preflight without importing torch or allocating GPU memory.
smi = subprocess.run(
    ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
    capture_output=True, text=True)
if smi.returncode:
    raise RuntimeError("nvidia-smi failed; select Kaggle GPU T4 x2. " + smi.stderr[-2000:])
GPU_ROWS = [line.strip() for line in smi.stdout.splitlines() if line.strip()]
if len(GPU_ROWS) != 2 or any("T4" not in row.upper() for row in GPU_ROWS):
    raise RuntimeError("Expected exactly two Kaggle T4 GPUs; found: " + repr(GPU_ROWS))
GPU_RECEIPT = {
    "requested": "Kaggle T4 x2", "observed": GPU_ROWS,
    "scientific_workloads_use_cuda": False,
    "note": "The pinned HORM runs and verification suite are CPU-only by protocol."}
(RUN_DIR / "kaggle_environment.json").write_text(
    json.dumps({"gpu": GPU_RECEIPT, "python": sys.version,
                "platform": platform.platform()}, indent=2) + "\n", encoding="utf-8")
print("GPU preflight: PASS", GPU_ROWS)

if not (REPO_DIR / ".git").exists():
    if REPO_DIR.exists():
        raise RuntimeError("Source path exists without .git; refusing to overwrite: " + str(REPO_DIR))
    _run(["git", "clone", "--depth", "1", "--no-checkout", "--branch", BRANCH, REMOTE, REPO_DIR])
current = subprocess.check_output(["git", "-C", str(REPO_DIR), "rev-parse", "HEAD"], text=True).strip()
if current != PINNED_COMMIT:
    dirty = subprocess.check_output(["git", "-C", str(REPO_DIR), "status", "--porcelain"], text=True).strip()
    if dirty:
        raise RuntimeError("Pinned source directory has local changes; preserving it. Use a fresh Kaggle session/path.")
    _run(["git", "-C", str(REPO_DIR), "checkout", "--detach", PINNED_COMMIT])
HEAD_SHA = subprocess.check_output(["git", "-C", str(REPO_DIR), "rev-parse", "HEAD"], text=True).strip()
if HEAD_SHA != PINNED_COMMIT:
    raise RuntimeError("Source pin mismatch: " + HEAD_SHA)
os.chdir(REPO_DIR)

for module, package in (("pytest", "pytest"), ("tokenizers", "tokenizers"),
                        ("numpy", "numpy")):
    if __import__("importlib").util.find_spec(module) is None:
        _run([sys.executable, "-m", "pip", "install", "-q", package])

# Optional verified recovery: extract only manifest-listed run files after
# checking every SHA and rejecting unsafe ZIP paths. Never overwrite a file.
if RECOVERY_BUNDLE:
    import zipfile
    bundle = Path(RECOVERY_BUNDLE)
    if not bundle.is_file():
        raise FileNotFoundError("Recovery bundle not found: " + str(bundle))
    with zipfile.ZipFile(bundle) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise RuntimeError("Recovery bundle contains duplicate ZIP member names")
        manifest = json.loads(archive.read("bundle_manifest.json"))
        if manifest.get("head_sha") != PINNED_COMMIT:
            raise RuntimeError("Recovery bundle belongs to a different source commit")
        listed_files = manifest.get("files", {})
        expected_members = {"bundle_manifest.json"} | {"run/" + Path(rel).as_posix()
                                                       for rel in listed_files}
        if set(names) != expected_members:
            raise RuntimeError("Recovery bundle members differ from its manifest")
        for rel, expected in listed_files.items():
            path = Path(rel)
            if path.is_absolute() or ".." in path.parts or "\\" in rel:
                raise RuntimeError("Unsafe recovery member: " + rel)
            member = "run/" + path.as_posix()
            payload = archive.read(member)
            if hashlib.sha256(payload).hexdigest() != expected:
                raise RuntimeError("Recovery member hash mismatch: " + rel)
            dest = RUN_DIR / path
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                if hashlib.sha256(dest.read_bytes()).hexdigest() != expected:
                    raise RuntimeError("Recovery would overwrite different local evidence: " + str(dest))
            else:
                dest.write_bytes(payload)
print("Pinned source:", HEAD_SHA)
print("Run state:", RUN_DIR)
print("CPU execution is enforced per-stage; GPU identity is recorded only.")
'''

RUNNER = r'''# CELL 2 — resumable suite stages and optional, separately stored HORM reruns
import hashlib, json, os, subprocess, sys, time
from pathlib import Path
from queue import Empty, Queue
from threading import Thread

LOG_DIR = RUN_DIR / "logs"
STATE_DIR = RUN_DIR / "stages"
SCIENCE_DIR = RUN_DIR / "science"
for directory in (LOG_DIR, STATE_DIR, SCIENCE_DIR):
    directory.mkdir(parents=True, exist_ok=True)

CPU_ENV = dict(os.environ)
CPU_ENV.update({"PYTHONPATH": str(REPO_DIR), "CUDA_VISIBLE_DEVICES": "",
                "ANRA_TEST_DEVICE": "cpu", "OMP_NUM_THREADS": "2",
                "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
                "PYTHONUNBUFFERED": "1"})
PYTEST = [sys.executable, "-m", "pytest", "-q", "-rf", "-p", "no:cacheprovider"]

def _atomic_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)

def run_stage(name, command, threads="2"):
    command = [str(x) for x in command]
    identity = hashlib.sha256(json.dumps({"head": HEAD_SHA, "command": command,
                                         "threads": threads, "cpu_only": True},
                                        sort_keys=True).encode()).hexdigest()
    marker = STATE_DIR / (name + ".json")
    prior = json.loads(marker.read_text("utf-8")) if marker.exists() else {}
    if prior.get("identity_sha256") == identity and prior.get("status") == "passed":
        print("SKIP already passed:", name, "log:", prior.get("log"))
        return prior
    if prior and prior.get("identity_sha256") not in (None, identity):
        raise RuntimeError("Stage identity changed; use a new RUN_DIR: " + name)
    old_attempts = [int(path.stem.rsplit("attempt", 1)[-1])
                    for path in LOG_DIR.glob(name + ".attempt*.log")
                    if path.stem.rsplit("attempt", 1)[-1].isdigit()]
    attempt = max([int(prior.get("attempt", 0)), *old_attempts], default=0) + 1
    log = LOG_DIR / (name + ".attempt%02d.log" % attempt)
    env = dict(CPU_ENV, OMP_NUM_THREADS=threads, MKL_NUM_THREADS=threads)
    started = time.time()
    print("\n===", name, "attempt", attempt, "===\n", " ".join(command), flush=True)
    with log.open("w", encoding="utf-8", errors="replace") as handle:
        proc = subprocess.Popen(command, cwd=REPO_DIR, env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, errors="replace", bufsize=1,
                                start_new_session=True)
        output_queue = Queue()
        def _drain_output():
            for output_line in proc.stdout:
                output_queue.put(output_line)
            output_queue.put(None)
        reader = Thread(target=_drain_output, daemon=True)
        reader.start()
        try:
            stream_closed = False
            while not stream_closed:
                try:
                    line = output_queue.get(timeout=30)
                except Empty:
                    size = log.stat().st_size if log.exists() else 0
                    print("Still running:", name, "elapsed_s=", int(time.time()-started),
                          "saved_log_bytes=", size, flush=True)
                    continue
                if line is None:
                    stream_closed = True
                    continue
                handle.write(line)
                handle.flush()
                sys.stdout.write(line)
                sys.stdout.flush()
            rc = proc.wait()
            reader.join(timeout=2)
        except KeyboardInterrupt:
            import signal
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=10)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()
            result = {"name": name, "status": "interrupted", "returncode": proc.returncode,
                      "command": command, "identity_sha256": identity,
                      "head_sha": HEAD_SHA, "attempt": attempt,
                      "elapsed_seconds": round(time.time()-started, 3),
                      "log": str(log), "cuda_visible_devices": ""}
            _atomic_json(marker, result)
            print("Interrupted; subprocess stopped. Log preserved:", log)
            raise
    result = {"name": name, "status": "passed" if rc == 0 else "failed",
              "returncode": rc, "command": command, "identity_sha256": identity,
              "head_sha": HEAD_SHA, "attempt": attempt, "elapsed_seconds": round(time.time()-started, 3),
              "log": str(log), "cuda_visible_devices": ""}
    _atomic_json(marker, result)
    if rc:
        tail = log.read_text("utf-8", errors="replace").splitlines()[-60:]
        print("\n--- failure tail; full log:", log, "---")
        print("\n".join(tail))
        raise RuntimeError("Stage %s failed (exit %s). Rerun Cell 2 after reviewing the log." % (name, rc))
    return result

# Quick state/integration gate.
run_stage("hormonal-focused", PYTEST + ["tests/test_hormonal_state.py",
    "tests/test_hormonal_integration.py", "tests/test_hormonal_session.py"])

# Optional re-execution writes to a fresh session directory and never uses
# --force. Existing committed historical results are not modified.
for enabled, flag, name in ((RUN_HORM003_REPLICATION, "--horm003", "horm003-replication"),
                            (RUN_HORM004_REPLICATION, "--horm004", "horm004-replication")):
    if enabled:
        result_dir = SCIENCE_DIR / name
        result_file = result_dir / ("RESULT_horm003_prospective.json" if flag == "--horm003"
                                    else "RESULT_horm004_prospective.json")
        if result_file.exists():
            prior_result = json.loads(result_file.read_text("utf-8"))
            expected_schema = ("anra-horm-003-prospective/v1" if flag == "--horm003"
                               else "anra-horm-004-prospective/v1")
            if prior_result.get("schema") != expected_schema or not prior_result.get("sha256"):
                raise RuntimeError("Existing optional result is malformed; preserving it: " + str(result_file))
            print("Preserving separately stored result; skipping:", result_file,
                  "verdict:", prior_result.get("verdict"))
        else:
            result_dir.mkdir(parents=True, exist_ok=True)
            run_stage(name, [sys.executable, "experiments/HORM-001/run_horm002_ab.py",
                             flag, "--output", result_dir])
    else:
        print("Optional protocol rerun disabled:", name)

# Run each test module in a fresh process. This bounds peak memory and gives
# every completed module a durable resume marker after session interruption.
bulk_files = sorted(path.relative_to(REPO_DIR).as_posix()
    for path in (REPO_DIR / "tests").rglob("test_*.py")
    if path.name != "test_production_entry.py"
    and path.name != "test_v5_cyr_gpu014_r1c_e2e_preflight.py")
if not bulk_files:
    raise RuntimeError("No bulk test modules discovered")
for index, test_file in enumerate(bulk_files):
    name = "bulk-%03d" % index
    run_stage(name, PYTEST + [test_file,
        "--junitxml=" + str(LOG_DIR / (name + ".xml"))])
run_stage("r1c", PYTEST + ["tests/test_v5_cyr_gpu014_r1c_e2e_preflight.py",
    "--junitxml=" + str(LOG_DIR / "r1c.xml")], threads="4")
print("Core stages complete:", len(bulk_files), "resumable test modules. Production-entry stages are in Cell 3.")
'''

PE_STAGE = r'''# CELL 3 — production-entry tests, one test process at a time
import ast, json

SOURCE = REPO_DIR / "tests/test_production_entry.py"
tree = ast.parse(SOURCE.read_text("utf-8"))
all_nodes = sorted(node.name for node in ast.walk(tree)
                   if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"))
all_nodes = [name for name in all_nodes if name != "test_exact_head_test_receipt"]
heavy_suffixes = {
    "test_mid_campaign_resume_matches_uninterrupted",
    "test_full_update_uses_four_microsteps_one_step",
    "test_bucket_shapes_exact_and_certified",
    "test_milestone_protection_across_sessions",
    "test_rotation_keeps_milestones_and_head",
    "test_lr_no_rewarm_after_resume",
    "test_frozen_mixture_end_to_end",
    "test_cognition_mixture_resume_equality",
    "test_compressed_e2e_fresh_recovery_milestone_stop_resume_partial_complete",
    "test_multi_session_soak_state_machine",
    "test_session_timebox_resumable_then_completes",
}
heavy = [name for name in all_nodes if name in heavy_suffixes]
light = [name for name in all_nodes if name not in heavy_suffixes]
if len(all_nodes) != 51 or len(heavy) != 11 or len(light) != 40:
    raise RuntimeError("Pinned production-entry test inventory drifted: total/light/heavy="
                       + repr((len(all_nodes), len(light), len(heavy))))

def available_ram_gib():
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / (1024 * 1024)
    except OSError:
        pass
    return 0.0

RAM_GIB = available_ram_gib()
print("Available RAM GiB:", round(RAM_GIB, 2))
for group, nodes in (("pe-light", light),):
    for index, name in enumerate(nodes):
        run_stage(group + "-%02d" % index,
                  PYTEST + ["tests/test_production_entry.py::" + name,
                            "--junitxml=" + str(LOG_DIR / (group + "-%02d.xml" % index))])

if RAM_GIB >= 14.0:
    for index, name in enumerate(heavy):
        run_stage("pe-heavy-%02d" % index,
                  PYTEST + ["tests/test_production_entry.py::" + name,
                            "--junitxml=" + str(LOG_DIR / ("pe-heavy-%02d.xml" % index))])
    HEAVY_STATUS = "executed"
    _atomic_json(STATE_DIR / "pe-heavy-exclusion.json", {
        "status": "executed", "available_ram_gib": RAM_GIB,
        "required_ram_gib": 14.0, "tests_excluded": 0})
else:
    HEAVY_STATUS = "excluded-low-ram"
    _atomic_json(STATE_DIR / "pe-heavy-exclusion.json", {
        "status": HEAVY_STATUS, "available_ram_gib": RAM_GIB,
        "required_ram_gib": 14.0, "tests_excluded": len(heavy),
        "note": "Excluded, not passed. Can rerun this cell on a host meeting the gate."})
    print("RAM gate: excluded", len(heavy), "large-campaign tests; they are not counted as passed.")
print("Production-entry light stages complete; heavy status:", HEAVY_STATUS)
'''

RECEIPT_PACKAGE = r'''# CELL 4 — receipt, safe recovery bundle, and partial-run export
import datetime, glob, hashlib, json, os, shutil, zipfile
from pathlib import Path

def junit_counts(paths):
    total = {"passed": 0, "failed": 0, "skipped": 0}
    for path in paths:
        try:
            root = __import__("xml.etree.ElementTree", fromlist=["parse"]).parse(path).getroot()
        except Exception:
            continue
        suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
        if root.tag != "testsuite":
            suites = [node for node in suites if node is not root]
        for suite in suites:
            tests = int(suite.get("tests", 0)); failures = int(suite.get("failures", 0))
            errors = int(suite.get("errors", 0)); skipped = int(suite.get("skipped", 0))
            total["failed"] += failures + errors
            total["skipped"] += skipped
            total["passed"] += tests - failures - errors - skipped
    return total

results = []
for name, pattern in (("bulk", "bulk-*.xml"), ("r1c", "r1c.xml"),
                      ("pe-light", "pe-light-*.xml"), ("pe-heavy", "pe-heavy-*.xml")):
    counts = junit_counts(sorted(LOG_DIR.glob(pattern)))
    excluded = name == "pe-heavy" and not list(LOG_DIR.glob(pattern))
    if excluded:
        counts = {"passed": 0, "failed": 0, "skipped": 11}
    results.append({"command": "Kaggle stage " + name, "device": "cpu",
                    "files": ["tests/"], "provenance": (
                        "RAM_EXCLUDED; not run" if excluded else
                        "Kaggle T4x2 host; tests CPU-only; counts from saved JUnit XML"),
                    **counts})

# Do not mint a success-looking closure receipt for an interrupted run.
light_complete = all(json.loads((STATE_DIR / ("pe-light-%02d.json" % i)).read_text("utf-8")).get("status") == "passed"
                     for i in range(40) if (STATE_DIR / ("pe-light-%02d.json" % i)).exists())
light_complete = light_complete and all((STATE_DIR / ("pe-light-%02d.json" % i)).exists() for i in range(40))
heavy_exclusion_path = STATE_DIR / "pe-heavy-exclusion.json"
heavy_excluded = (heavy_exclusion_path.exists()
    and json.loads(heavy_exclusion_path.read_text("utf-8")).get("status") == "excluded-low-ram")
heavy_complete = heavy_excluded or all(
    (STATE_DIR / ("pe-heavy-%02d.json" % i)).exists()
    and json.loads((STATE_DIR / ("pe-heavy-%02d.json" % i)).read_text("utf-8")).get("status") == "passed"
    for i in range(11))
focused_complete = ((STATE_DIR / "hormonal-focused.json").exists()
    and json.loads((STATE_DIR / "hormonal-focused.json").read_text("utf-8")).get("status") == "passed")
bulk_files = sorted(path.relative_to(REPO_DIR).as_posix()
    for path in (REPO_DIR / "tests").rglob("test_*.py")
    if path.name != "test_production_entry.py"
    and path.name != "test_v5_cyr_gpu014_r1c_e2e_preflight.py")
bulk_complete = bool(bulk_files) and all(
    (STATE_DIR / ("bulk-%03d.json" % i)).exists()
    and json.loads((STATE_DIR / ("bulk-%03d.json" % i)).read_text("utf-8")).get("status") == "passed"
    for i in range(len(bulk_files)))
r1c_complete = ((STATE_DIR / "r1c.json").exists()
    and json.loads((STATE_DIR / "r1c.json").read_text("utf-8")).get("status") == "passed")
suite_complete = focused_complete and bulk_complete and r1c_complete and light_complete and heavy_complete
final_status = "partial"

if suite_complete:
    receipt_cmd = [sys.executable, "-c", "from v5_training.test_receipt import build_receipt,write_receipt; "
     "import json,os; from pathlib import Path; "
     "r=json.loads(os.environ['HORM_RECEIPT_RESULTS']); "
     "write_receipt('artifacts/v5/cymek_500m_closure_test_receipt.json', "
     "build_receipt(tested_commit_sha=os.environ['HORM_HEAD'], results=r, "
     "environment={'note':'Kaggle T4x2 host; CPU-only suite; no scientific GPU or TPU training.'}))"]
    receipt_env = dict(CPU_ENV, HORM_RECEIPT_RESULTS=json.dumps(results), HORM_HEAD=HEAD_SHA)
    receipt_log = LOG_DIR / "receipt-generation.log"
    with receipt_log.open("w", encoding="utf-8") as handle:
        proc = subprocess.run(receipt_cmd, cwd=REPO_DIR, env=receipt_env,
                              stdout=handle, stderr=subprocess.STDOUT, text=True)
    if proc.returncode:
        final_status = "validation_failed"
        _atomic_json(STATE_DIR / "receipt-generation.json", {"status": "failed", "returncode": proc.returncode,
                      "log": str(receipt_log), "head_sha": HEAD_SHA})
        print("Receipt generation failed; partial bundle will still be produced:", receipt_log)
    else:
        shutil.copy2(REPO_DIR / "artifacts/v5/cymek_500m_closure_test_receipt.json",
                     RUN_DIR / "cymek_500m_closure_test_receipt.json")
        _atomic_json(STATE_DIR / "receipt-generation.json", {"status": "passed", "head_sha": HEAD_SHA,
                      "log": str(receipt_log)})
        try:
            run_stage("receipt-contract", PYTEST + [
                "tests/test_production_entry.py::test_exact_head_test_receipt"])
            run_stage("import-boundaries", [sys.executable, "-m", "v5_contracts.import_boundaries"])
            final_status = "complete"
        except Exception as exc:
            final_status = "validation_failed"
            _atomic_json(STATE_DIR / "package-validation-error.json", {
                "status": "failed", "error": repr(exc), "head_sha": HEAD_SHA})
            print("Post-suite validation failed; package the logs for review.")
else:
    print("Suite incomplete. No closure receipt generated; packaging partial logs and stage states.")

def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

files = {}
for path in sorted(RUN_DIR.rglob("*")):
    if path.is_file() and path.name != "bundle_manifest.json":
        files[path.relative_to(RUN_DIR).as_posix()] = sha256_file(path)
manifest = {"schema": "anra-horm-kaggle-session/v1", "head_sha": HEAD_SHA,
            "gpu": GPU_RECEIPT, "files": files,
            "counts": results,
            "status": final_status,
            "heavy_tests": "excluded_low_ram" if heavy_excluded else "executed_or_pending"}
manifest_path = RUN_DIR / "bundle_manifest.json"
manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
bundle = WORK / ("HORM_CYMEK_BETA_KAGGLE_T4X2_" + stamp + ".zip")
with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    archive.write(manifest_path, "bundle_manifest.json")
    for rel in files:
        archive.write(RUN_DIR / Path(rel), "run/" + rel)
print("Bundle:", bundle)
print("Bundle status:", manifest["status"])
print("To continue later: Save Version with outputs enabled, attach this ZIP as an input, set RECOVERY_BUNDLE in Cell 1, and rerun cells.")
'''


def cell(cell_type: str, source: str) -> dict:
    return {"cell_type": cell_type, "metadata": {},
            "source": source.splitlines(keepends=True),
            **({"outputs": [], "execution_count": None} if cell_type == "code" else {})}


notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "kaggle": {"accelerator": "GPU", "gpuType": "T4 x2"},
    },
    "cells": [cell("markdown", MARKDOWN), cell("code", BOOTSTRAP),
              cell("code", RUNNER), cell("code", PE_STAGE),
              cell("code", RECEIPT_PACKAGE)],
}

for index, item in enumerate(notebook["cells"]):
    if item["cell_type"] == "code":
        ast.parse("".join(item["source"]), filename=f"notebook-cell-{index}")

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"Wrote {OUT} with {len(notebook['cells'])} cells; code cells parse successfully.")
