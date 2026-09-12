from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import os
import platform
import tempfile
import uuid
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

REPO = Path(__file__).resolve().parents[2]
RESULTS_DIR = Path(os.environ.get("ARKENSTONE_RESULTS_ROOT", "artifacts/arkenstone/discovery_v6"))

MASTER_PLAN_SHA = "f46bae74c783821452947f3927acddbc14cc6dbf"
CANONICAL_T2_SHA = "0dd9305697045b0fbf4e7f268b46a4d7276e4794af5d78b60e999df914ae4236"


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_json(value: Any) -> str:
    return sha_bytes(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8"))


def file_sha256(path: Path) -> str:
    return sha_bytes(path.read_bytes())


def git_head() -> str:
    return subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()


def current_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not detected. In Colab select a T4 GPU runtime.")
    return torch.device("cuda")


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def load_ark11():
    path = REPO / "experiments" / "ARK-011" / "run_ark011.py"
    spec = importlib.util.spec_from_file_location("arkenstone_ark11_runtime", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load ARK-011 runtime from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def bind_ark11_runtime(ark11, device: torch.device, head: str, *, ctx=None) -> None:
    ark11.DEVICE = device
    ark11.RUNNER_HEAD = head
    ark11.RUNNER_SOURCE_SHA256 = file_sha256(Path(ark11.__file__))
    ark11.SESSION_START = time.time()
    if ctx is not None:
        ark11.RESULTS_DIR = ctx.output_dir / "ARK-011"
        original_loss = ark11.loss_and_positions

        def budgeted_loss(*args, **kwargs):
            ensure_budget(ctx)
            return original_loss(*args, **kwargs)

        ark11.loss_and_positions = budgeted_loss


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def cpu_tree(obj: Any) -> Any:
    if torch.is_tensor(obj):
        return obj.detach().cpu().clone()
    if isinstance(obj, dict):
        return {k: cpu_tree(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [cpu_tree(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(cpu_tree(v) for v in obj)
    return copy.deepcopy(obj)


def parameter_sha(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        h.update(name.encode("utf-8"))
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def flat_params(model: torch.nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def order_sha256(indices: list[list[int]]) -> str:
    return sha_bytes(json.dumps(indices, separators=(",", ":")).encode("utf-8"))


def generate_indices(seed: int, n_batches: int, batch_size: int, pool_size: int) -> list[list[int]]:
    g = torch.Generator().manual_seed(seed)
    if n_batches < 0 or batch_size <= 0 or pool_size <= 0:
        raise ValueError("require nonnegative batches and positive batch/pool sizes")
    return torch.randint(0, pool_size, (n_batches, batch_size), generator=g).tolist()


def detect_sustained(
    evals: list[tuple[int, float]], bar: float, consecutive: int = 3, *, below: bool = False
) -> tuple[int | None, int | None]:
    if consecutive < 1 or not math.isfinite(bar):
        raise ValueError("require positive consecutive count and finite threshold")
    streak = 0
    onset = None
    for step, value in evals:
        hit = value < bar if below else value >= bar
        if hit:
            if streak == 0:
                onset = step
            streak += 1
            if streak >= consecutive:
                return onset, step
        else:
            streak = 0
            onset = None
    return None, None


def trajectory_metrics(trajectory: list[dict], key: str, *, bar: float = 0.90) -> dict:
    if not trajectory:
        return {"status": "EMPTY"}
    vals = [float(x[key]) for x in trajectory]
    step_key = "step" if "step" in trajectory[0] else "relative_step"
    steps = [int(x[step_key]) for x in trajectory]
    if any(not math.isfinite(v) or not 0.0 <= v <= 1.0 for v in vals):
        raise ValueError("exact-accuracy trajectory must contain finite values in [0, 1]")
    if any(b <= a for a, b in zip(steps, steps[1:])):
        raise ValueError("trajectory steps must be strictly increasing")
    onset, confirm = detect_sustained([(int(x[step_key]), float(x[key])) for x in trajectory], bar, 3)
    # A recurrent drop is measured only after sustained recovery, not the
    # initial below-threshold state from which a recovery arm begins.
    drop_onset, drop_confirm = detect_sustained(
        [(int(x[step_key]), float(x[key])) for x in trajectory
         if confirm is not None and int(x[step_key]) > confirm], bar, 3, below=True
    )
    return {
        "AREA": sum(vals) / len(vals),
        "FINAL": vals[-1],
        "PEAK": max(vals),
        "RET90": sum(v >= 0.90 for v in vals) / len(vals),
        "RET50": sum(v >= 0.50 for v in vals) / len(vals),
        "G90_ONSET": onset,
        "G90_CONFIRM": confirm,
        "DROP90_ONSET": drop_onset,
        "DROP90_CONFIRM": drop_confirm,
    }


class BudgetExhausted(RuntimeError):
    """The allotted wall budget expired; unfinished arms are not results."""


@dataclass
class RunContext:
    device: torch.device
    head: str
    started: float
    budget_minutes: float
    output_dir: Path | None = None
    _monotonic_start: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not math.isfinite(self.budget_minutes) or self.budget_minutes <= 0:
            raise ValueError("budget_minutes must be finite and positive")
        if not math.isfinite(self.started):
            raise ValueError("started must be a finite Unix timestamp")
        self._monotonic_start = time.monotonic() - max(0.0, time.time() - self.started)
        self.output_dir = Path(self.output_dir) if self.output_dir is not None else (
            RESULTS_DIR / (time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + uuid.uuid4().hex[:12])
        )
        self.output_dir = self.output_dir.resolve()
        # Claim a fresh directory. Existing runs, even empty ones, are immutable.
        self.output_dir.mkdir(parents=True, exist_ok=False)

    @property
    def minutes_used(self) -> float:
        return (time.monotonic() - self._monotonic_start) / 60.0

    @property
    def minutes_left(self) -> float:
        return self.budget_minutes - self.minutes_used


def ensure_budget(ctx: RunContext) -> None:
    if ctx.minutes_left <= 0:
        raise BudgetExhausted("campaign wall budget exhausted")


def atomic_write(path: Path, data: bytes) -> None:
    """Publish whole bytes on the same filesystem; retain the old file on error."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def device_name(device: torch.device) -> str:
    if device.type != "cuda":
        return platform.processor() or "CPU"
    try:
        return torch.cuda.get_device_name(device)
    except Exception as exc:
        # A CUDA initialization failure must still be recordable in a receipt.
        return "unavailable: " + type(exc).__name__


def verify_receipt(payload: dict) -> bool:
    body = dict(payload)
    actual = body.pop("receipt_sha256", None)
    return isinstance(actual, str) and actual == sha_json(body)


class ReceiptWriter:
    def __init__(self, ctx: RunContext, *, experiment_id: str, plan_sha: str, runner_path: Path,
                 extra_plan_shas: dict[str, str] | None = None) -> None:
        self.ctx = ctx
        self.experiment_id = experiment_id
        self.plan_sha = plan_sha
        self.runner_path = runner_path
        self.extra_plan_shas = dict(extra_plan_shas or {})
        # Hash the shared scientific implementation as well as the leaf runner.
        paths = [runner_path, Path(__file__), REPO / "experiments/ARK-011/run_ark011.py",
                 REPO / "experiments/ARK-001/run_ark001.py", REPO / "experiments/lib/ark_tasks.py",
                 REPO / "experiments/ARK-012/run_ark012.py", REPO / "experiments/ARK-013/run_ark013.py",
                 REPO / "experiments/COLAB/run_discovery_v6.py",
                 REPO / "experiments/COLAB/MASTER_DISCOVERY_V6_PLAN.md",
                 REPO / "experiments/ARK-012/PLAN.md", REPO / "experiments/ARK-013/PLAN.md",
                 REPO / "experiments/ARK-013/PREEXECUTION_ADDENDUM.md",
                 REPO / "experiments/ARK-011/PLAN.md",
                 REPO / "experiments/ARK-011/PREEXECUTION_ADDENDUM.md",
                 REPO / "experiments/ARK-002B/TASK_MANIFEST.json"]
        self.sources = {p.resolve().relative_to(REPO).as_posix(): file_sha256(p) for p in paths
                        if p.resolve().is_relative_to(REPO)}
        self.runner_sha = file_sha256(runner_path)
        for relative, digest in self.sources.items():
            source = REPO / relative
            snapshot = ctx.output_dir / "sources" / relative
            if snapshot.exists():
                if file_sha256(snapshot) != digest:
                    raise RuntimeError(f"source changed within run: {relative}")
            else:
                data = source.read_bytes()
                if sha_bytes(data) != digest:
                    raise RuntimeError(f"source changed during snapshot: {relative}")
                atomic_write(snapshot, data)

    def save(self, filename: str, payload: dict) -> Path:
        if Path(filename).name != filename or not filename.endswith(".json"):
            raise ValueError("receipt filename must be a plain .json basename")
        out = dict(payload)
        out["experiment_id"] = self.experiment_id
        out["plan_commit_sha"] = self.plan_sha
        out["master_plan_commit_sha"] = MASTER_PLAN_SHA
        out.update(self.extra_plan_shas)
        out["runner_commit_sha"] = self.ctx.head
        out["source_identity_note"] = "Commit identifies checkout base; source_sha256 binds working-tree contents."
        out["runner_source_sha256"] = self.runner_sha
        out["source_sha256"] = self.sources
        out["run_id"] = self.ctx.output_dir.name
        out["device"] = str(self.ctx.device)
        out["device_name"] = device_name(self.ctx.device)
        out["torch"] = torch.__version__
        out["python"] = platform.python_version()
        out["budget_minutes"] = self.ctx.budget_minutes
        out["campaign_minutes_used"] = self.ctx.minutes_used
        body = dict(out)
        body.pop("receipt_sha256", None)
        out["receipt_sha256"] = sha_json(body)
        encoded = (json.dumps(out, indent=2, default=str, allow_nan=False) + "\n").encode("utf-8")
        path = self.ctx.output_dir / filename
        # Preserve every completed revision before advancing the latest view.
        revision = self.ctx.output_dir / "revisions" / (path.stem + "-" + uuid.uuid4().hex + ".json")
        atomic_write(revision, encoded)
        atomic_write(path, encoded)
        print("saved:", path, flush=True)
        return path


def package_all(download: bool = True, *, ctx: RunContext) -> Path:
    """Package only this run, never stale sibling or legacy Colab results."""
    zip_path = ctx.output_dir / "ARKENSTONE_DISCOVERY_V6_RESULTS.zip"
    fd, temporary = tempfile.mkstemp(suffix=".zip.tmp", dir=ctx.output_dir)
    os.close(fd)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            paths = set(ctx.output_dir.rglob("*.json"))
            paths.update(p for p in (ctx.output_dir / "sources").rglob("*") if p.is_file())
            for path in sorted(paths):
                zf.write(path, path.relative_to(ctx.output_dir).as_posix())
        os.replace(temporary, zip_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    print("RESULT ZIP:", zip_path, flush=True)
    if download:
        try:
            from google.colab import files
            files.download(str(zip_path))
        except Exception as exc:
            print("auto-download skipped:", repr(exc), flush=True)
    return zip_path


def import_experiment_module(number: int):
    name = f"run_ark{number:03d}"
    path = REPO / "experiments" / f"ARK-{number:03d}" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"arkenstone_{name}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load experiment module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
