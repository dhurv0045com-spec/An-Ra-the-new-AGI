from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

REPO = Path(__file__).resolve().parents[2]
RESULTS_DIR = Path("/content/arkenstone_discovery_v7_results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MASTER_PLAN_SHA = "cd058b9067c3fe2e001c70969a1966928d74c7df"
ARK015_PLAN_SHA = "da56dfbbafc6951c8d1c17c86410187cd7472e74"
ARK016_PLAN_SHA = "459b8cebb080a7e325378f6396a37963d8fc4ac1"
CANONICAL_T2_SHA = "0dd9305697045b0fbf4e7f268b46a4d7276e4794af5d78b60e999df914ae4236"
ARK014_BINDING_MANIFEST_SHA = "fbc8605dc1cfc19ac8692d2b2c6338fcb2af3e02b24907f3b5cca3571947fee3"


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
        raise RuntimeError("CUDA GPU not detected. In Colab choose Runtime -> Change runtime type -> T4 GPU.")
    return torch.device("cuda")


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


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
    return [torch.randint(0, pool_size, (batch_size,), generator=g).tolist() for _ in range(n_batches)]


def detect_sustained(
    evals: list[tuple[int, float]], bar: float, consecutive: int = 3, *, below: bool = False
) -> tuple[int | None, int | None]:
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


def model_state_equal(a: dict, b: dict) -> bool:
    return a.keys() == b.keys() and all(torch.equal(a[k].detach().cpu(), b[k].detach().cpu()) for k in a)


@dataclass
class RunContext:
    device: torch.device
    head: str
    started: float
    budget_minutes: float

    @property
    def minutes_used(self) -> float:
        return (time.time() - self.started) / 60.0

    @property
    def minutes_left(self) -> float:
        return self.budget_minutes - self.minutes_used


class ReceiptWriter:
    def __init__(
        self,
        ctx: RunContext,
        *,
        experiment_id: str,
        plan_sha: str,
        runner_path: Path,
        extra_plan_shas: dict[str, str] | None = None,
    ) -> None:
        self.ctx = ctx
        self.experiment_id = experiment_id
        self.plan_sha = plan_sha
        self.runner_path = runner_path
        self.extra_plan_shas = dict(extra_plan_shas or {})

    def save(self, filename: str, payload: dict) -> Path:
        out = dict(payload)
        out.setdefault("experiment_id", self.experiment_id)
        out["plan_commit_sha"] = self.plan_sha
        out["master_plan_commit_sha"] = MASTER_PLAN_SHA
        out.update(self.extra_plan_shas)
        out["runner_commit_sha"] = self.ctx.head
        out["runner_source_sha256"] = file_sha256(self.runner_path)
        out["device"] = str(self.ctx.device)
        out["torch"] = torch.__version__
        out["campaign_minutes_used"] = self.ctx.minutes_used
        body = dict(out)
        body.pop("receipt_sha256", None)
        out["receipt_sha256"] = sha_json(body)
        path = RESULTS_DIR / filename
        path.write_text(json.dumps(out, indent=2, default=str) + "\n", encoding="utf-8")
        print("saved:", path, flush=True)
        return path


def import_experiment_module(number: int):
    name = f"run_ark{number:03d}"
    path = REPO / "experiments" / f"ARK-{number:03d}" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"arkenstone_v7_{name}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import experiment module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_ark11():
    path = REPO / "experiments" / "ARK-011" / "run_ark011.py"
    spec = importlib.util.spec_from_file_location("arkenstone_v7_ark11_runtime", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load ARK-011 runtime from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def bind_ark11_runtime(ark11, device: torch.device, head: str) -> None:
    ark11.DEVICE = device
    ark11.RUNNER_HEAD = head
    ark11.RUNNER_SOURCE_SHA256 = file_sha256(Path(ark11.__file__))
    ark11.SESSION_START = time.time()


def load_ark14():
    return import_experiment_module(14)


def gradient_norm_and_clip(model: torch.nn.Module, max_norm: float = 1.0) -> float:
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    return float(norm.detach().item() if torch.is_tensor(norm) else norm)


def optimizer_step_with_delta(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    cap_norm: float | None = None,
) -> dict[str, float | bool | None]:
    """Apply optimizer.step(), optionally globally rescaling only the applied parameter delta.

    Optimizer moments remain exactly as produced by the underlying AdamW step. This is
    intentional: ARK-016 separates applied movement from nominal HIGH-LR optimizer state.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("model has no trainable parameters")
    before = [p.detach().clone() for p in params]
    optimizer.step()

    raw_sq = torch.zeros((), device=params[0].device, dtype=torch.float32)
    for p, b in zip(params, before):
        d = p.detach() - b
        raw_sq = raw_sq + d.float().pow(2).sum()
    raw_norm = float(raw_sq.sqrt().item())

    cap_fired = False
    applied_norm = raw_norm
    if cap_norm is not None and raw_norm > float(cap_norm):
        scale = float(cap_norm) / max(raw_norm, 1e-30)
        with torch.no_grad():
            for p, b in zip(params, before):
                p.copy_(b + (p - b) * scale)
        applied_norm = float(cap_norm)
        cap_fired = True

    return {
        "raw_delta_norm": raw_norm,
        "applied_delta_norm": applied_norm,
        "cap_norm": None if cap_norm is None else float(cap_norm),
        "cap_fired": cap_fired,
    }


def package_all(download: bool = True) -> Path:
    zip_path = RESULTS_DIR / "ARKENSTONE_DISCOVERY_V7_RESULTS.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(RESULTS_DIR.rglob("*.json")):
            zf.write(path, path.relative_to(RESULTS_DIR).as_posix())
    print("RESULT ZIP:", zip_path, flush=True)
    if download:
        try:
            from google.colab import files
            files.download(str(zip_path))
        except Exception as exc:
            print("auto-download skipped:", repr(exc), flush=True)
    return zip_path
