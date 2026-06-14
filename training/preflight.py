"""Hardware-aware launch preflight with honest T4 runtime classes."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import shutil
import tempfile
import time

import torch

from anra.anra_paths import (
    DATASET_CANONICAL,
    PROFILE_DIR,
    STATE_DIR,
    TOKEN_INVENTORY_MANIFEST,
    V3_TOKENIZER_FILE,
)
from runtime.training_readiness import assess_training_readiness


RUNTIME_CLASSES = ("t4_full_25m", "t4_frontier_smoke", "t4_3b_preflight")


@dataclass(frozen=True)
class HardwareProfile:
    gpu_name: str
    cuda_available: bool
    vram_bytes: int
    ram_bytes: int
    disk_free_bytes: int
    bf16_supported: bool


@dataclass(frozen=True)
class LaunchDecision:
    allowed: bool
    model_size: str
    runtime_class: str
    blockers: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    estimated_runtime_hours: float | None = None
    recommended_profile: str = ""
    checks: dict[str, object] = field(default_factory=dict)
    generated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def detect_hardware() -> HardwareProfile:
    import psutil

    cuda = torch.cuda.is_available()
    return HardwareProfile(
        gpu_name=torch.cuda.get_device_name(0) if cuda else "none",
        cuda_available=cuda,
        vram_bytes=torch.cuda.get_device_properties(0).total_memory if cuda else 0,
        ram_bytes=int(psutil.virtual_memory().total),
        disk_free_bytes=int(shutil.disk_usage(Path.cwd()).free),
        bf16_supported=bool(cuda and torch.cuda.is_bf16_supported()),
    )


def run_preflight(
    model_size: str,
    *,
    runtime_class: str | None = None,
    hardware: HardwareProfile | None = None,
) -> LaunchDecision:
    profile = model_size.lower()
    runtime_class = runtime_class or (
        "t4_full_25m" if profile == "25m" else
        "t4_frontier_smoke" if profile in {"frontier", "904m", "1b"} else
        "t4_3b_preflight"
    )
    if runtime_class not in RUNTIME_CLASSES:
        raise ValueError(f"Unknown runtime class: {runtime_class}")
    hw = hardware or detect_hardware()
    readiness = assess_training_readiness()
    blockers = list(readiness.blockers)
    warnings = list(readiness.warnings)
    if not hw.cuda_available:
        blockers.append("CUDA GPU unavailable")
    is_t4 = "T4" in hw.gpu_name.upper()
    if profile in {"frontier", "904m", "1b"} and runtime_class != "t4_frontier_smoke":
        blockers.append("T4 permits frontier smoke/adapter work only without a measured full-campaign profile")
    if profile == "3b":
        blockers.append("3B full allocation/training is prohibited on T4 preflight")
    if profile in {"frontier", "904m", "1b"} and not TOKEN_INVENTORY_MANIFEST.exists():
        warnings.append("licensed token inventory missing; smoke work is allowed but a full campaign is not")
    if runtime_class == "t4_full_25m" and profile != "25m":
        blockers.append("t4_full_25m runtime class only supports the 25m profile")
    consent_path = STATE_DIR / "cognition" / "consent.json"
    checks = {
        "hardware": asdict(hw),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "tokenizer_exists": V3_TOKENIZER_FILE.exists(),
        "dataset_exists": DATASET_CANONICAL.exists(),
        "token_inventory_exists": TOKEN_INVENTORY_MANIFEST.exists(),
        "measured_profile_exists": (PROFILE_DIR / f"memory_profile_{profile}.json").exists(),
        "cognitive_consent_declared": consent_path.exists(),
        "owner_data_authorized": _owner_data_authorized(consent_path),
        "optimizer_backends": {
            name: __import__("importlib").util.find_spec(name) is not None
            for name in ("bitsandbytes", "galore_torch", "muon")
        },
        "drive_integrity": _write_read_test(Path(os.environ.get("ANRA_DRIVE_DIR", tempfile.gettempdir()))),
        "emergency_save": _emergency_save_test(),
        "seed_reproducibility": _seed_test(),
        "t4_detected": is_t4,
        "precision": "bf16" if hw.bf16_supported else "fp16",
    }
    supported_mode = (
        profile == "25m"
        or (
            profile in {"frontier", "904m", "1b"}
            and runtime_class == "t4_frontier_smoke"
        )
    )
    return LaunchDecision(
        allowed=not blockers and supported_mode,
        model_size=profile,
        runtime_class=runtime_class,
        blockers=tuple(dict.fromkeys(blockers)),
        warnings=tuple(warnings),
        estimated_runtime_hours=(
            6.0 if profile == "25m" and is_t4
            else 0.5 if runtime_class == "t4_frontier_smoke" and is_t4
            else None
        ),
        recommended_profile=runtime_class,
        checks=checks,
    )


def _write_read_test(root: Path) -> bool:
    try:
        root.mkdir(parents=True, exist_ok=True)
        path = root / f".anra-preflight-{os.getpid()}"
        payload = os.urandom(32)
        path.write_bytes(payload)
        valid = hashlib.sha256(path.read_bytes()).digest() == hashlib.sha256(payload).digest()
        path.unlink(missing_ok=True)
        return valid
    except OSError:
        return False


def _emergency_save_test() -> bool:
    try:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "emergency.pt"
            torch.save({"step": 1}, target)
            return torch.load(target, map_location="cpu", weights_only=False)["step"] == 1
    except Exception:
        return False


def _seed_test() -> bool:
    random.seed(1301)
    left = [random.random() for _ in range(4)]
    random.seed(1301)
    return left == [random.random() for _ in range(4)]


def _owner_data_authorized(consent_path: Path) -> bool:
    try:
        payload = json.loads(consent_path.read_text(encoding="utf-8"))
        return bool(payload.get("training_use", False))
    except (OSError, json.JSONDecodeError, TypeError):
        return False
