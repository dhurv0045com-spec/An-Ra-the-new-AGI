"""Pinned cloud campaign runner. Importing this module does no work."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from hashlib import file_digest, sha256
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
from threading import Event
from uuid import uuid4
import zipfile


METHODS = ("capacity-matched", "context-only", "associative", "two-read",
           "recurrent-read", "recurrent-aug", "recurrent-canon", "three-read", "recurrent-dense")
GIB = 1 << 30
RESUME_COMPONENTS = ("model.bin", "optimizer.bin", "rng.bin", "scheduler.json",
                     "cursor.json", "ledger.json")


@dataclass(frozen=True)
class Plan:
    name: str
    surface: str
    profile: str
    methods: tuple[str, ...]
    seeds: tuple[int, ...]
    rungs: tuple[int, ...]
    device_count: int
    train_seconds: int
    export_seconds: int

    def arms(self):
        return tuple((method, seed) for seed in self.seeds for method in self.methods)


PLANS = {
    "colab": Plan("colab", "graph-v3", "expanded", METHODS[:-1], (7, 19), (384, 768, 1536), 1, 6300, 900),
    "kaggle": Plan("kaggle", "graph-v3-replication", "cloud", METHODS, (7, 19, 41, 61, 79),
                   (1024, 3072, 6144), 2, 18900, 1800),
}


@dataclass(frozen=True)
class GPU:
    index: int
    name: str
    uuid: str
    total_mib: int
    free_mib: int
    driver: str


def fingerprint(path: Path) -> str:
    with path.open("rb") as handle:
        return file_digest(handle, "sha256").hexdigest()


def blob(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"


def safe_id(value: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[a-zA-Z0-9_-]{1,100}", value) is None:
        raise ValueError("unsafe run or lineage name")
    return value


def sha_id(value: str, *, length: int = 64) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{" + str(length) + r"}", value) is None:
        raise ValueError("expected an immutable lowercase Git or SHA-256 identity")
    return value


def command(args, *, cwd=None, env=None, timeout=120, log=None):
    if log is None:
        result = subprocess.run(args, cwd=cwd, env=env, timeout=timeout, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        if result.returncode:
            raise RuntimeError(f"command failed ({result.returncode}): {result.stdout[-2000:]}")
        return result.stdout.strip()
    with log.open("xb") as handle:
        return subprocess.run(args, cwd=cwd, env=env, timeout=timeout, stdout=handle,
                              stderr=subprocess.STDOUT, check=True)


def check_source(repo: Path, source: str) -> None:
    sha_id(source, length=40)
    if command(["git", "rev-parse", "HEAD"], cwd=repo) != source:
        raise ValueError("checkout does not match the notebook source pin")
    if command(["git", "status", "--porcelain", "--untracked-files=all"], cwd=repo):
        raise ValueError("notebook requires a clean source checkout")


def gpu_rows(raw: str, visible: str | None = None) -> tuple[GPU, ...]:
    rows = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        index, name, uuid, total, free, driver = (cell.strip() for cell in line.split(","))
        if not re.fullmatch(r"GPU-[a-fA-F0-9-]+", uuid):
            raise ValueError("nvidia-smi did not provide a physical GPU UUID")
        rows.append(GPU(int(index), name, uuid, int(total), int(free), driver))
    if visible is not None:
        lookup = {key: row for row in rows for key in (str(row.index), row.uuid)}
        keys = [key.strip() for key in visible.split(",") if key.strip()]
        if any(key not in lookup for key in keys):
            raise ValueError("CUDA visibility cannot be mapped to full physical UUIDs")
        rows = [lookup[key] for key in keys]
    if len({row.uuid for row in rows}) != len(rows):
        raise ValueError("CUDA lanes resolve to duplicate physical GPU UUIDs")
    return tuple(rows)


def inventory() -> tuple[GPU, ...]:
    raw = command(["nvidia-smi", "--query-gpu=index,name,uuid,memory.total,memory.free,driver_version",
                   "--format=csv,noheader,nounits"], timeout=25)
    return gpu_rows(raw, os.environ.get("CUDA_VISIBLE_DEVICES"))


def admit_devices(rows: tuple[GPU, ...], expected: int) -> None:
    if len(rows) != expected or any("T4" not in row.name or row.total_mib < 14000 for row in rows):
        raise ValueError(f"NO-GO: require exactly {expected} visible full NVIDIA T4 device(s)")
    if any(row.free_mib < 8000 for row in rows):
        raise ValueError("NO-GO: require at least 8,000 MiB free on every T4")


def device_env(gpu: GPU | None = None) -> dict:
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env.update(PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1",
               MKL_NUM_THREADS="1", CUBLAS_WORKSPACE_CONFIG=":4096:8")
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu.uuid
    return env


def check_torch(repo: Path, gpus: tuple[GPU, ...]) -> list[dict]:
    probe = (
        "import json,sys,torch;"
        "assert sys.version_info >= (3, 11);"
        "assert torch.cuda.is_available();"
        "print(json.dumps(dict(python=sys.version.split()[0],torch=str(torch.__version__),"
        "cuda=torch.version.cuda,count=torch.cuda.device_count(),"
        "names=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],"
        "capability=torch.cuda.get_device_capability(0))))"
    )
    combined = json.loads(command([sys.executable, "-c", probe], cwd=repo, env=device_env()))
    if combined["count"] != len(gpus) or any("T4" not in name for name in combined["names"]):
        raise ValueError("Torch and nvidia-smi disagree on visible T4 count")
    results = []
    for gpu in gpus:
        row = json.loads(command([sys.executable, "-c", probe], cwd=repo, env=device_env(gpu)))
        match = re.match(r"^(\d+)\.(\d+)\.", row["torch"])
        if (match is None or not (2, 8) <= tuple(map(int, match.groups())) < (3, 0)
                or row["count"] != 1 or row["capability"] != [7, 5] or not row["cuda"]):
            raise ValueError("require CUDA-enabled PyTorch 2.8+ (2.x) and one T4 in each lane")
        results.append({"physical": asdict(gpu), "runtime": row})
    return results


def cli(repo: Path, arguments, *, gpu=None, timeout=120, log=None):
    return command([sys.executable, "-m", "anra_v5.elendril", *arguments], cwd=repo,
                   env=device_env(gpu), timeout=timeout, log=log)


def scientific_audit(repo: Path, plan: Plan, output: Path) -> dict:
    cli(repo, ["audit-v3", "--surface", plan.surface, "--output", str(output)])
    evidence = json.loads(output.read_bytes())
    if (evidence["schema"] != "elendril-graph-v3-audit/v1" or evidence["surface"] != plan.surface
            or evidence["training_rows"] != 256):
        raise ValueError("fixture audit does not match the registered campaign")
    expected = json.loads((repo/"tools"/"elendril_cloud_fixtures.json").read_bytes())[plan.surface]
    if sha256(blob(evidence)).hexdigest() != expected["audit_sha256"]:
        raise ValueError("fixture audit drifted from the immutable preregistered audit")
    return evidence


def config(source: str, plan: Plan, fixture_sha: str, *, screen_sha: str | None = None,
           runtime_sha: str | None = None) -> dict:
    sha_id(source, length=40)
    sha_id(fixture_sha)
    if screen_sha is not None:
        sha_id(screen_sha)
    if runtime_sha is not None:
        sha_id(runtime_sha)
    return {"schema": "elendril-cloud-v3", "source": source, "plan": asdict(plan),
            "fixture_sha256": fixture_sha, "screen_sha256": screen_sha, "runtime_sha256": runtime_sha}


def read_checkpoint(path: Path, identity: str) -> dict:
    manifest = path / "manifest.json"
    if path.is_symlink() or not manifest.is_file() or manifest.is_symlink() or fingerprint(manifest) != sha_id(identity):
        raise ValueError("checkpoint manifest integrity failure")
    data = json.loads(manifest.read_bytes())
    records = data["components"]
    names = set()
    for record in records:
        name = record["name"]
        if re.fullmatch(r"[a-zA-Z0-9_-]+\.[a-zA-Z0-9]+", name) is None or name in names:
            raise ValueError("unsafe or duplicate checkpoint component name")
        names.add(name)
        item = path / name
        if (item.is_symlink() or not item.is_file() or item.stat().st_size != record["byte_size"]
                or fingerprint(item) != sha_id(record["sha256"])):
            raise ValueError(f"checkpoint component integrity failure: {name}")
    if names != set(RESUME_COMPONENTS) | {"training_state.json"}:
        raise ValueError("checkpoint component inventory mismatch")
    if {p.name for p in path.iterdir()} != names | {"manifest.json"}:
        raise ValueError("checkpoint contains unregistered files")
    return {row["name"]: row["sha256"] for row in records}


def verified_copy(src: Path, dst: Path) -> None:
    if src.is_symlink() or dst.exists():
        raise ValueError("cannot copy linked inputs or replace an existing file")
    with src.open("rb") as before, dst.open("xb") as after:
        shutil.copyfileobj(before, after)
        after.flush()
        os.fsync(after.fileno())
    if fingerprint(src) != fingerprint(dst):
        raise ValueError("destination failed read-after-write verification")


def register_campaign(root: Path, specification: dict) -> str:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "campaign.json"
    data = blob(specification)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError("campaign configuration mismatch")
    else:
        with path.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    if path.read_bytes() != data:
        raise ValueError("campaign configuration failed readback")
    return sha256(data).hexdigest()


def committed_boundary(path: Path, campaign: str) -> tuple[dict, Path, Path] | None:
    marker = path / "COMMITTED.json"
    if not marker.exists():
        return None
    if marker.is_symlink():
        raise ValueError("linked commit marker")
    record = json.loads(marker.read_bytes())
    if record["campaign_sha256"] != sha_id(campaign):
        raise ValueError("checkpoint belongs to a different campaign")
    payload = path / safe_id(record["payload"])
    if payload.is_symlink():
        raise ValueError("linked transaction payload")
    receipt = payload / "receipt.json"
    if receipt.is_symlink() or fingerprint(receipt) != sha_id(record["receipt_sha256"]):
        raise ValueError("receipt integrity failure")
    checkpoint = payload / "checkpoint"
    read_checkpoint(checkpoint, record["checkpoint_sha256"])
    item = json.loads(receipt.read_bytes())
    if (item["final_checkpoint_sha256"] != record["checkpoint_sha256"]
            or item["lineage"] != record["lineage"]
            or item["evaluations"][-1]["update"] != record["boundary"]):
        raise ValueError("receipt and checkpoint marker disagree")
    manifest = json.loads((checkpoint/"manifest.json").read_bytes())
    if (manifest["state"]["global_update"] != record["boundary"]
            or manifest["state"]["lineage_id"] != record["lineage"]):
        raise ValueError("checkpoint training state disagrees with the boundary")
    return record, receipt, checkpoint


def publish_boundary(path: Path, campaign: str, receipt: Path, checkpoint: Path) -> dict:
    path.mkdir(parents=True, exist_ok=True)
    existing = committed_boundary(path, campaign)
    source = json.loads(receipt.read_bytes())
    if existing is not None:
        if existing[0]["receipt_sha256"] != fingerprint(receipt):
            raise ValueError("refusing to replace a committed campaign boundary")
        return existing[0]
    digest = sha_id(source["final_checkpoint_sha256"])
    read_checkpoint(checkpoint, digest)
    payload = path / ("tx-" + uuid4().hex)
    obj = payload / "checkpoint"
    obj.mkdir(parents=True)
    for file in checkpoint.iterdir():
        verified_copy(file, obj / file.name)
    verified_copy(receipt, payload / "receipt.json")
    read_checkpoint(obj, digest)
    marker = {"schema": "elendril-rung-v1", "campaign_sha256": sha_id(campaign),
              "payload": payload.name, "lineage": safe_id(source["lineage"]),
              "boundary": source["evaluations"][-1]["update"], "checkpoint_sha256": digest,
              "receipt_sha256": fingerprint(receipt)}
    with (path / "COMMITTED.json").open("xb") as handle:
        handle.write(blob(marker))
        handle.flush()
        os.fsync(handle.fileno())
    reloaded = committed_boundary(path, campaign)
    if reloaded is None or reloaded[0] != marker:
        raise ValueError("committed marker failed verified readback")
    return marker


def materialize(boundary, root: Path) -> None:
    marker, _, source = boundary
    lineage = root / safe_id(marker["lineage"])
    identity = sha_id(marker["checkpoint_sha256"])
    checkpoint = lineage / "objects" / identity
    checkpoint.mkdir(parents=True, exist_ok=False)
    for file in source.iterdir():
        verified_copy(file, checkpoint / file.name)
    read_checkpoint(checkpoint, identity)
    (lineage / "LATEST").write_bytes((identity + "\n").encode())


def arm_args(plan: Plan, method: str, seed: int, total: int, boundary: int,
             scratch: Path, output: Path, *, resume: bool = False) -> list[str]:
    if method not in plan.methods:
        raise ValueError("method is outside the frozen campaign")
    args = ["pilot", "--surface", plan.surface, "--profile", plan.profile, "--arm", "binding",
            "--method", method, "--seed", str(seed), "--steps", str(total), "--until", str(boundary),
            "--device", "cuda:0", "--root", str(scratch), "--output", str(output)]
    if resume:
        args.append("--resume")
    return args


def pilot(repo: Path, plan: Plan, gpu: GPU, method: str, seed: int, total: int,
          boundary: int, scratch: Path, output: Path, *, timeout: int = 240,
          resume: bool = False) -> tuple[dict, Path]:
    cli(repo, arm_args(plan, method, seed, total, boundary, scratch, output, resume=resume),
        gpu=gpu, timeout=timeout, log=output.with_suffix(".log"))
    cli(repo, ["rescore", "--receipt", str(output)], timeout=min(timeout, 120))
    receipt = json.loads(output.read_bytes())
    if (receipt["surface"] != plan.surface or receipt["profile"] != plan.profile
            or receipt["method"] != method or receipt["seed"] != seed
            or receipt["evaluations"][-1]["update"] != boundary):
        raise ValueError("pilot receipt differs from the registered rung")
    expected = json.loads((repo/"tools"/"elendril_cloud_fixtures.json").read_bytes())[plan.surface]
    transform = {"recurrent-aug": "augment", "recurrent-canon": "canonical"}.get(method, "raw")
    if (receipt["training_sha256"] != expected["training"][transform]
            or receipt["development_sha256"] != expected["base"]):
        raise ValueError("pilot fixture hash differs from the preregistered audit")
    obj = scratch / safe_id(receipt["lineage"]) / "objects" / sha_id(receipt["final_checkpoint_sha256"])
    read_checkpoint(obj, receipt["final_checkpoint_sha256"])
    return receipt, obj


def canary(repo: Path, source: str, plan: Plan, gpu: GPU, workspace: Path,
           destination: Path) -> dict:
    canary_id = sha256(blob(["canary", source, plan.name, gpu.uuid])).hexdigest()
    canary_dir = destination / ("canary-" + uuid4().hex)
    canary_dir.mkdir(parents=True)
    with tempfile.TemporaryDirectory(dir=workspace) as name:
        root = Path(name)
        try:
            a_path, b_path, c_path, d_path = (root / f"{key}.json" for key in ("direct", "first", "resumed", "forecast"))
            direct, direct_obj = pilot(repo, plan, gpu, "recurrent-read", 7, 32, 2, root/"a", a_path)
            first, first_obj = pilot(repo, plan, gpu, "recurrent-read", 7, 32, 1, root/"b", b_path)
            marker = publish_boundary(canary_dir/"first", canary_id, b_path, first_obj)
            restored = committed_boundary(canary_dir/"first", canary_id)
            materialize(restored, root/"restored")
            resumed, resumed_obj = pilot(repo, plan, gpu, "recurrent-read", 7, 32, 2,
                                         root/"restored", c_path, resume=True)
            before = read_checkpoint(first_obj, first["final_checkpoint_sha256"])
            expected = read_checkpoint(direct_obj, direct["final_checkpoint_sha256"])
            actual = read_checkpoint(resumed_obj, resumed["final_checkpoint_sha256"])
            if (first["updates"] + resumed["updates"] != direct["updates"]
                    or any(expected[k] != actual[k] for k in RESUME_COMPONENTS)
                    or direct["evaluations"][-1]["predictions_sha256"] != resumed["evaluations"][-1]["predictions_sha256"]
                    or before["model.bin"] == actual["model.bin"]
                    or before["optimizer.bin"] == actual["optimizer.bin"]):
                raise ValueError("same-device exact continuation or genuine update check failed")
            publish_boundary(canary_dir/"resumed", canary_id, c_path, resumed_obj)
            forecast, _ = pilot(repo, plan, gpu, "recurrent-read", 7, 32, 16, root/"forecast", d_path)
            forecasts = [forecast]
            extra = None
            if "recurrent-dense" in plan.methods:
                extra = root/"forecast-dense.json"
                dense, _ = pilot(repo, plan, gpu, "recurrent-dense", 7, 32, 16,
                                 root/"forecast-dense", extra)
                forecasts.append(dense)
            trained = [row["evaluations"][-1]["segment_training_seconds"] for row in forecasts]
            if any(len(row["updates"]) != 16 for row in forecasts) or min(trained) <= 0:
                raise ValueError("forecast canary has incomplete timing")
            result = {"schema": "elendril-t4-canary-v3", "status": "PASS", "source": source,
                      "gpu": asdict(gpu), "same_device_parity": "byte-exact FP32 state and predictions",
                      "first_checkpoint": marker["checkpoint_sha256"], "model_updated": True,
                      "optimizer_updated": True, "training_seconds_per_step": max(trained) / 16,
                      "nontraining_seconds": max(0., *(row["wall_seconds"] - elapsed
                                                       for row, elapsed in zip(forecasts, trained))),
                      "peak_allocated_bytes": max(row["accelerator"]["peak_allocated_bytes"] for row in forecasts),
                      "peak_reserved_bytes": max(row["accelerator"]["peak_reserved_bytes"] for row in forecasts),
                      "checkpoint_bytes": sum(file.stat().st_size for file in direct_obj.iterdir()),
                      "forecast_receipt_sha256": fingerprint(d_path), "resumed_components": actual}
            for path in (a_path, b_path, c_path, d_path, *((extra,) if extra else ())):
                verified_copy(path, canary_dir/path.name)
        except Exception:
            for path in root.glob("*.log"):
                verified_copy(path, canary_dir/path.name)
            raise
    result_path = canary_dir / "CANARY.json"
    result_path.write_bytes(blob(result))
    if json.loads(result_path.read_bytes()) != result:
        raise ValueError("canary result failed readback")
    return result



def rung_path(root: Path, method: str, seed: int, boundary: int) -> Path:
    return root / "arms" / safe_id(f"{method}-{seed}") / f"rung-{boundary:05d}"


def lookup(roots, method: str, seed: int, boundary: int, campaign: str):
    found = []
    for root in roots:
        item = committed_boundary(rung_path(root, method, seed, boundary), campaign)
        if item is not None:
            found.append(item)
    if len({(item[0]["checkpoint_sha256"], item[0]["receipt_sha256"]) for item in found}) > 1:
        raise ValueError("conflicting historical campaign boundaries")
    return found[0] if found else None


def require_prefix(plan: Plan, roots, campaign: str):
    starts = {}
    for method, seed in plan.arms():
        previous = None
        missing = False
        for boundary in plan.rungs:
            item = lookup(roots, method, seed, boundary, campaign)
            if item is None:
                missing = True
                continue
            row = json.loads(item[1].read_bytes())
            if missing or row["start_checkpoint_sha256"] != (previous[0]["checkpoint_sha256"] if previous else None):
                raise ValueError("broken or noncontiguous checkpoint lineage")
            previous = item
        starts[(method, seed)] = previous
    return starts


def rate(metrics, key: str = "query_swap_both_correct") -> float:
    hit, size = metrics[key]
    if type(hit) is not int or type(size) is not int or not 0 <= hit <= size or size == 0:
        raise ValueError("invalid metric numerator or denominator")
    return hit / size


def eligible_screen(repo: Path, source: str, roots: tuple[Path, ...]) -> dict:
    if not roots:
        raise ValueError("NO-GO: attach the complete screening output before confirmation")
    expected = json.loads((roots[0] / "campaign.json").read_bytes())
    expected_source = config(source, PLANS["colab"], expected["fixture_sha256"],
                             runtime_sha=sha_id(expected["runtime_sha256"]))
    if blob(expected) != blob(expected_source):
        raise ValueError("screen provenance differs from the frozen Colab suite or source")
    campaign = sha256(blob(expected_source)).hexdigest()
    for root in roots:
        if (root / "campaign.json").read_bytes() != blob(expected_source):
            raise ValueError("screen roots carry different campaign identities")
    with tempfile.TemporaryDirectory() as name:
        audit_path = Path(name) / "audit.json"
        scientific_audit(repo, PLANS["colab"], audit_path)
        if fingerprint(audit_path) != expected["fixture_sha256"]:
            raise ValueError("screen fixture audit differs from the pinned generator")
    starts = require_prefix(PLANS["colab"], roots, campaign)
    rows = {}
    for (method, seed), item in starts.items():
        if item is None or item[0]["boundary"] != PLANS["colab"].rungs[-1]:
            raise ValueError("NO-GO: incomplete matched screen")
        cli(repo, ["rescore", "--receipt", str(item[1])])
        receipt = json.loads(item[1].read_bytes())
        if (receipt["source_commit"] != source or receipt["surface"] != "graph-v3"
                or receipt["method"] != method or receipt["seed"] != seed
                or receipt["status"] != "completed"):
            raise ValueError("screen receipt provenance mismatch")
        rows[(method, seed)] = (receipt, item[0]["receipt_sha256"])

    def passed(method, control, panel):
        values = []
        for seed in PLANS["colab"].seeds:
            a, b = (rows[(arm, seed)][0] for arm in (method, control))
            delta = rate(a["panels"][panel]["metrics"]) - rate(b["panels"][panel]["metrics"])
            base = rate(a["evaluations"][-1]["metrics"]) - rate(b["evaluations"][-1]["metrics"])
            values.append({"seed": seed, "stress_delta": delta, "base_delta": base})
        return {"method": method, "control": control, "panel": panel, "by_seed": values,
                "passed": all(x["stress_delta"] >= 0.10 - 1e-12 and x["base_delta"] >= -0.05 - 1e-12
                              for x in values)}

    gates = [passed(method, "recurrent-read", "alias-combined")
             for method in ("recurrent-aug", "recurrent-canon")]
    gates.append(passed("recurrent-read", "two-read", "depth-four"))
    return {"campaign": campaign, "gates": gates, "eligible": any(g["passed"] for g in gates),
            "receipts": {f"{m}-{s}": pair[1] for (m, s), pair in rows.items()}}


def estimate_remaining(plan: Plan, starts, canaries: list[dict], boundary: int) -> float:
    lanes = [0.] * plan.device_count
    for ordinal, arm in enumerate(plan.arms()):
        lane = ordinal % plan.device_count
        previous = starts[arm]
        current = previous[0]["boundary"] if previous else 0
        if current < boundary:
            sample = canaries[lane]
            lanes[lane] += (boundary - current) * sample["training_seconds_per_step"] * 1.6
            lanes[lane] += sample["nontraining_seconds"] * 2 + 20
    return max(lanes)


def capacity_check(plan: Plan, destination: Path, scratch: Path, canaries: list[dict]) -> None:
    max_checkpoint = max(row["checkpoint_bytes"] for row in canaries)
    all_bytes = int(1.2 * max_checkpoint) * len(plan.arms()) * len(plan.rungs) + GIB
    if plan.name == "kaggle" and all_bytes > 18 * GIB:
        raise ValueError("NO-GO: projected retained checkpoints exceed Kaggle output allocation")
    if shutil.disk_usage(destination).free < all_bytes:
        raise ValueError("NO-GO: insufficient campaign destination capacity")
    if shutil.disk_usage(scratch).free < max(2 * GIB, max_checkpoint * 12 * plan.device_count):
        raise ValueError("NO-GO: insufficient local scratch capacity")
    for row in canaries:
        if row["peak_reserved_bytes"] > (row["gpu"]["free_mib"] - 2048) * (1 << 20):
            raise ValueError("NO-GO: forecast does not leave a 2 GiB GPU margin")


def archive_failure(root: Path, method: str, seed: int, boundary: int, stage: Path, exc: Exception) -> None:
    path = root / "failures" / f"{method}-{seed}-{boundary}-{uuid4().hex[:10]}"
    path.mkdir(parents=True)
    (path/"error.json").write_bytes(blob({"method": method, "seed": seed, "boundary": boundary,
                                         "error_type": type(exc).__name__, "error": str(exc)[-1500:]}))
    for item in (stage/"receipt.log", stage/"receipt.json"):
        if item.is_file():
            verified_copy(item, path/item.name)


def run_campaign(repo: Path, source: str, plan: Plan, gpus: tuple[GPU, ...], canaries: list[dict],
                 workspace: Path, output: Path, *, fixture_sha: str, enable: bool,
                 prior: tuple[Path, ...] = (), screening: dict | None = None,
                 owner_reviewed_screen: bool = False, budget_seconds: int | None = None,
                 runtime_sha: str | None = None) -> dict:
    if enable is not True:
        return {"status": "DISABLED", "reason": "owner campaign switch remains false"}
    check_source(repo, source)
    if plan.name == "kaggle" and (not owner_reviewed_screen or not screening or not screening.get("eligible")):
        raise ValueError("NO-GO: a qualified screen and explicit owner review are required")
    admit_devices(inventory(), plan.device_count)
    if (len(gpus) != plan.device_count or len(canaries) != plan.device_count
            or any(row["source"] != source or row["status"] != "PASS"
                   or row["gpu"]["uuid"] != gpu.uuid for gpu, row in zip(gpus, canaries))):
        raise ValueError("NO-GO: each physical GPU requires a passing current-session canary")
    screen_sha = sha256(blob(screening)).hexdigest() if screening else None
    spec = config(source, plan, fixture_sha, screen_sha=screen_sha, runtime_sha=sha_id(runtime_sha))
    for root in prior:
        if (root/"campaign.json").read_bytes() != blob(spec):
            raise ValueError("attached resume output has a different campaign configuration")
    campaign = register_campaign(output, spec)
    if screening and not (output/"screen-decision.json").exists():
        (output/"screen-decision.json").write_bytes(blob(screening))
    capacity_check(plan, output, workspace, canaries)
    budget = plan.train_seconds if budget_seconds is None else min(plan.train_seconds, budget_seconds)
    if budget < 60:
        raise ValueError("NO-GO: less than one minute remains before the export margin")
    deadline = time.monotonic() + budget
    roots = (output, *prior)
    starts = require_prefix(plan, roots, campaign)
    stop = Event()
    completed_rounds = []
    interruptions = []
    for boundary in plan.rungs:
        estimate = estimate_remaining(plan, starts, canaries, boundary)
        print(f"Rung {boundary} | conservative ETA {estimate / 60:.1f} min | remaining {(deadline-time.monotonic()) / 60:.1f} min", flush=True)
        if estimate > deadline - time.monotonic():
            interruptions.append({"boundary": boundary, "reason": "next complete comparison rung exceeds remaining budget",
                                  "forecast_seconds": round(estimate, 2)})
            break

        def lane_worker(lane: int, arms):
            gpu = gpus[lane]
            for method, seed in arms:
                previous = starts[(method, seed)]
                prior_update = previous[0]["boundary"] if previous else 0
                if prior_update >= boundary:
                    continue
                if stop.is_set():
                    return
                with tempfile.TemporaryDirectory(dir=workspace) as temp:
                    stage = Path(temp)
                    try:
                        current = {item.uuid: item for item in inventory()}.get(gpu.uuid)
                        if current is None or current.free_mib < 6000:
                            raise RuntimeError("T4 disappeared or has insufficient VRAM")
                        if shutil.disk_usage(workspace).free < 2 * GIB or shutil.disk_usage(output).free < GIB:
                            raise RuntimeError("disk safety margin violated")
                        remaining = int(deadline - time.monotonic())
                        if remaining <= 0:
                            raise TimeoutError("campaign deadline reached before worker admission")
                        if previous:
                            materialize(previous, stage/"scratch")
                        result, obj = pilot(repo, plan, gpu, method, seed, plan.rungs[-1], boundary,
                                            stage/"scratch", stage/"receipt.json",
                                            resume=previous is not None, timeout=remaining)
                        if (result["source_commit"] != source or result["start_update"] != prior_update
                                or result["start_checkpoint_sha256"] != (
                                    previous[0]["checkpoint_sha256"] if previous else None)):
                            raise ValueError("pilot source or checkpoint parent mismatch")
                        path = rung_path(output, method, seed, boundary)
                        publish_boundary(path, campaign, stage/"receipt.json", obj)
                        starts[(method, seed)] = committed_boundary(path, campaign)
                        final = result["evaluations"][-1]["metrics"]["query_swap_both_correct"]
                        print(f"{gpu.uuid[-8:]} | {method} seed {seed} | rung {boundary} | both-correct {final[0]}/{final[1]}", flush=True)
                    except Exception as exc:
                        stop.set()
                        archive_failure(output, method, seed, boundary, stage, exc)
                        raise

        arms = plan.arms()
        with ThreadPoolExecutor(max_workers=plan.device_count) as executor:
            futures = [executor.submit(lane_worker, lane, arms[lane::plan.device_count])
                       for lane in range(plan.device_count)]
            for future in futures:
                try:
                    future.result()
                except Exception as exc:
                    interruptions.append({"boundary": boundary, "reason": type(exc).__name__,
                                          "detail": str(exc)[-1000:]})
        if stop.is_set():
            break
        completed_rounds.append(boundary)
    return {"status": "COMPLETE" if not interruptions and len(completed_rounds) == len(plan.rungs) else "PARTIAL",
            "campaign": campaign, "completed_rounds": completed_rounds, "interruptions": interruptions,
            "peak_forecast_reserved_bytes": max(row["peak_reserved_bytes"] for row in canaries)}



def paired_cluster_ci(a: dict, b: dict, *, samples: int = 2000) -> list[float] | None:
    """Bootstrap the exact same world clusters in a paired arm contrast."""
    from tools.compare_elendril_pairs import paired_cluster_interval

    left, right = a.get("cluster_query_pairs", {}), b.get("cluster_query_pairs", {})
    if not left or left.keys() != right.keys() or len(left) < 2:
        return None
    convert = lambda row: {"cluster_query_pairs": row["cluster_query_pairs"],
                           "query_swap_both_correct": row["both_correct"]}
    result = paired_cluster_interval({7: convert(b)}, {7: convert(a)}, draws=samples)
    return result["interval"]


def report_records(plan: Plan, roots, campaign: str):
    starts = require_prefix(plan, roots, campaign)
    summaries = []
    receipts = {}
    for method, seed in plan.arms():
        curve = {}
        elapsed = 0.
        occupied = 0.
        peak = 0
        forwards = 0
        positions = 0
        last = None
        for boundary in plan.rungs:
            entry = lookup(roots, method, seed, boundary, campaign)
            if entry is None:
                break
            marker, receipt, checkpoint = entry
            row = json.loads(receipt.read_bytes())
            if row["method"] != method or row["seed"] != seed or row["surface"] != plan.surface:
                raise ValueError("report arm identity mismatch")
            receipts[(method, seed, boundary)] = (receipt, checkpoint/"manifest.json", marker)
            for item in [*row["evaluations"], *row["panels"].values()]:
                inference = item["inference"]
                forwards += inference["forward_calls"]
                positions += inference["evaluated_positions"]
            for item in row["evaluations"]:
                curve[item["update"]] = {"update": item["update"],
                    "training_seconds": elapsed + item["segment_training_seconds"],
                    "both_correct": item["metrics"]["query_swap_both_correct"],
                    "exact_eos": item["metrics"]["exact_eos"]}
            elapsed += row["evaluations"][-1]["segment_training_seconds"]
            occupied += row["wall_seconds"]
            peak = max(peak, row["accelerator"]["peak_reserved_bytes"])
            last = row
        status = "MISSING" if last is None else ("COMPLETE" if starts[(method, seed)][0]["boundary"] == plan.rungs[-1] else "PARTIAL")
        summaries.append({"method": method, "seed": seed, "status": status, "curve": list(curve.values()),
                          "latest_training_tokens": last["training_tokens"] if last else 0,
                          "latest_checkpoint": last["final_checkpoint_sha256"] if last else None,
                          "parameters": last["model_parameters"] if last else None,
                          "inference_forward_calls": forwards, "inference_positions": positions,
                          "in_process_lane_hours": occupied / 3600, "peak_reserved_bytes": peak,
                          "panels": {name: {"both_correct": item["metrics"]["query_swap_both_correct"],
                                           "exact_eos": item["metrics"]["exact_eos"],
                                           "cluster_query_pairs": item["metrics"].get("cluster_query_pairs", {})}
                                     for name, item in last["panels"].items()} if last else {}})
    comparison = []
    complete = {(row["method"], row["seed"]): row for row in summaries if row["status"] == "COMPLETE"}
    for method, control, panel in (("recurrent-aug", "recurrent-read", "alias-combined"),
                                   ("recurrent-canon", "recurrent-read", "alias-combined"),
                                   ("recurrent-read", "two-read", "depth-four")):
        for seed in plan.seeds:
            a, b = (complete.get((arm, seed)) for arm in (method, control))
            if a and b and panel in a["panels"] and panel in b["panels"]:
                comparison.append({"method": method, "control": control, "seed": seed,
                    "panel": panel, "paired_cluster_bootstrap_95pct": paired_cluster_ci(
                        a["panels"][panel], b["panels"][panel])})
    return summaries, receipts, comparison


def make_charts(summaries: list[dict], directory: Path) -> list[Path]:
    if not any(arm["curve"] for arm in summaries):
        return []
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    colors = dict(zip(METHODS, plt.get_cmap("tab10").colors))
    styles = ("-", "--", "-.", ":", (0, (5, 1)))
    seed_index = {seed: i for i, seed in enumerate(sorted({row["seed"] for row in summaries}))}
    for row in summaries:
        if not row["curve"]:
            continue
        for ax, key, label in zip(axes, ("update", "training_seconds"), ("Optimizer updates", "Training seconds")):
            xs = [item[key] for item in row["curve"]]
            ys = [rate({"query_swap_both_correct": item["both_correct"]}) for item in row["curve"]]
            ax.plot(xs, ys, color=colors[row["method"]], linestyle=styles[seed_index[row["seed"]]],
                    marker=".", alpha=.75, label=row["method"])
            ax.set(xlabel=label, ylabel="Both answers exact + EOS", ylim=(-.04, 1.05))
            ax.grid(alpha=.2)
    handles, labels = axes[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    axes[0].legend(unique.values(), unique.keys(), fontsize=8, frameon=False)
    figure.suptitle("Frozen development comparison · one trace per seed")
    path = directory/"learning-curves.png"
    figure.savefig(path, dpi=140)
    plt.close(figure)
    paths = [path]
    panels = ("alias-symbol", "alias-separator", "alias-combined", "depth-three", "depth-four",
              "rewire", "absent", "ambiguous")
    if any(row["panels"] for row in summaries):
        figure, ax = plt.subplots(figsize=(12, 5), layout="constrained")
        for index, method in enumerate(METHODS):
            for row in summaries:
                if row["method"] != method:
                    continue
                for j, name in enumerate(panels):
                    pair = row["panels"].get(name, {}).get("both_correct")
                    if pair and pair[1]:
                        ax.scatter(j+(index-(len(METHODS)-1)/2)*.08, pair[0]/pair[1], color=colors[method], s=17)
        ax.set(xticks=range(len(panels)), xticklabels=panels, ylabel="Both answers exact + EOS",
               ylim=(-.04, 1.05), title="Terminal robustness · points are seeded fits on shared worlds")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=.2)
        path = directory/"robustness.png"
        figure.savefig(path, dpi=140)
        plt.close(figure)
        paths.append(path)
    return paths


def export_run(plan: Plan, source: str, output: Path, workspace: Path, *,
               audit: Path, environment: list[dict], launch: dict, prior: tuple[Path, ...] = ()) -> Path:
    roots = (output, *prior)
    records = {}
    comparisons = []
    if (output/"campaign.json").exists():
        campaign = fingerprint(output/"campaign.json")
        summaries, records, comparisons = report_records(plan, roots, campaign)
    else:
        summaries = []
    with tempfile.TemporaryDirectory(dir=workspace) as name:
        stage = Path(name)
        summary = {"schema": "elendril-cloud-report-v3", "notebook_version": f"v3-{plan.name}-1",
                   "source": source, "plan": asdict(plan), "environment": environment, "launch": launch,
                   "arms": summaries, "paired_cluster_intervals": comparisons,
                   "in_process_lane_hours": sum(x["in_process_lane_hours"] for x in summaries),
                   "training_tokens": sum(x["latest_training_tokens"] for x in summaries),
                   "inference_forward_calls": sum(x["inference_forward_calls"] for x in summaries),
                   "inference_positions": sum(x["inference_positions"] for x in summaries),
                   "scope": "Public development. Cluster intervals are paired within each seed and do not certify independent confirmation."}
        (stage/"summary.json").write_bytes(blob(summary))
        files = [(stage/"summary.json", "summary.json"), (audit, "fixture-audit.json")]
        files.extend((path, path.name) for path in make_charts(summaries, stage))
        for (method, seed, boundary), (receipt, manifest, marker) in sorted(records.items()):
            prefix = f"arms/{method}-{seed}/rung-{boundary}"
            files.extend(((receipt, prefix+"/receipt.json"), (manifest, prefix+"/checkpoint-manifest.json")))
            marker_path = stage/f"{method}-{seed}-{boundary}-marker.json"
            marker_path.write_bytes(blob(marker))
            files.append((marker_path, prefix+"/publication.json"))
        for item in ("campaign.json", "screen-decision.json", "CANARY_SESSION.json"):
            path = output/item
            if path.exists():
                files.append((path, item))
        for root_index, root in enumerate(roots):
            for pattern in ("canary-*/CANARY.json", "canary-*/*.json", "failures/*/*"):
                for item in sorted(root.glob(pattern)):
                    if item.is_file():
                        arcname = f"session-{root_index}/" + item.relative_to(root).as_posix()
                        if not any(existing == arcname for _, existing in files):
                            files.append((item, arcname))
        manifest = {arcname: fingerprint(file) for file, arcname in files}
        (stage/"manifest.json").write_bytes(blob(manifest))
        archive = stage/"bundle.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
            for file, arcname in [*files, (stage/"manifest.json", "manifest.json")]:
                handle.write(file, arcname)
        with zipfile.ZipFile(archive) as handle:
            if handle.testzip() is not None:
                raise ValueError("export ZIP CRC check failed")
            if any(sha256(handle.read(name)).hexdigest() != digest for name, digest in manifest.items()):
                raise ValueError("export ZIP digest mismatch")
        destination = output / ("elendril-report-" + uuid4().hex[:12] + ".zip")
        verified_copy(archive, destination)
    return destination
