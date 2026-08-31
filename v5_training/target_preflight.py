"""Executable target TPU/XLA preflight.

The preflight intentionally fails closed when XLA is unavailable.  A passing
receipt means only that the target runtime can execute a tiny BF16 collective
and expose rank/RNG identities; it is not training or throughput evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


SCHEMA = "anra-v5-target-preflight/v1"


@dataclass(frozen=True, slots=True)
class PreflightConfig:
    expected_world_size: int = 1
    seed: int = 47_101
    matrix_size: int = 128

    def assert_valid(self) -> None:
        if self.expected_world_size <= 0 or self.seed < 0 or self.matrix_size <= 0:
            raise ValueError("invalid target preflight configuration")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_hash(tensor: Any) -> str:
    return _sha256(tensor.detach().cpu().contiguous().numpy().tobytes())


def run_preflight(config: PreflightConfig) -> dict[str, object]:
    config.assert_valid()
    missing_dependencies: list[str] = []
    try:
        import torch
    except ImportError as exc:
        missing_dependencies.append("torch")
        return {
            "schema": SCHEMA,
            "status": "BLOCKED_TORCH_XLA",
            "reason": str(exc),
            "missing_dependencies": missing_dependencies,
            "config": asdict(config),
            "implementation_sha256": _sha256_file(Path(__file__)),
        }
    try:
        import torch_xla.core.xla_model as xm
    except ImportError as exc:
        missing_dependencies.append("torch_xla")
        return {
            "schema": SCHEMA,
            "status": "BLOCKED_TORCH_XLA",
            "reason": str(exc),
            "missing_dependencies": missing_dependencies,
            "config": asdict(config),
            "implementation_sha256": _sha256_file(Path(__file__)),
        }

    try:
        device = xm.xla_device()
        world_size = int(xm.xrt_world_size())
        ordinal = int(xm.get_ordinal())
    except Exception as exc:
        return {
            "schema": SCHEMA,
            "status": "FAIL",
            "scope": "target TPU/XLA runtime preflight; no model training",
            "phase": "device_initialization",
            "error_type": type(exc).__name__,
            "reason": str(exc),
            "config": asdict(config),
            "implementation_sha256": _sha256_file(Path(__file__)),
        }
    checks: dict[str, bool] = {
        "xla_device": str(device).startswith("xla"),
        "world_size_matches": world_size == config.expected_world_size,
        "ordinal_in_range": 0 <= ordinal < world_size,
    }
    torch.manual_seed(config.seed + ordinal)
    try:
        started = time.perf_counter()
        left = torch.randn((config.matrix_size, config.matrix_size), device=device, dtype=torch.bfloat16)
        right = torch.randn((config.matrix_size, config.matrix_size), device=device, dtype=torch.bfloat16)
        product = left @ right
        checks["bf16_matmul_finite"] = bool(torch.isfinite(product.float()).all().item())
        collective = torch.tensor([float(ordinal + 1)], device=device, dtype=torch.float32)
        reduced = xm.all_reduce(xm.REDUCE_SUM, collective)
        # Probe the actual XLA generator twice; CPU RNG state is not a reliable
        # proxy for device-side randomness on TPU/XLA.
        rng_probe_1 = torch.rand((16,), device=device, dtype=torch.float32)
        rng_probe_2 = torch.rand((16,), device=device, dtype=torch.float32)
        xm.mark_step()
        checks["all_reduce_world_sum"] = abs(float(reduced.cpu().item()) - world_size * (world_size + 1) / 2) < 1e-5
        checks["mark_step_completed"] = True
        rng_hash_1 = _tensor_hash(rng_probe_1)
        rng_hash_2 = _tensor_hash(rng_probe_2)
        checks["device_rng_progresses"] = rng_hash_1 != rng_hash_2
        elapsed = time.perf_counter() - started
    except Exception as exc:
        return {
            "schema": SCHEMA,
            "status": "FAIL",
            "scope": "target TPU/XLA runtime preflight; no model training",
            "phase": "device_smoke",
            "error_type": type(exc).__name__,
            "reason": str(exc),
            "device": str(device),
            "world_size": world_size,
            "ordinal": ordinal,
            "config": asdict(config),
            "implementation_sha256": _sha256_file(Path(__file__)),
        }
    return {
        "schema": SCHEMA,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "scope": "target TPU/XLA runtime preflight; no model training",
        "implementation_sha256": _sha256_file(Path(__file__)),
        "torch_version": torch.__version__,
        "torch_xla_version": getattr(__import__("torch_xla"), "__version__", "unknown"),
        "platform": platform.platform(),
        "device": str(device),
        "world_size": world_size,
        "ordinal": ordinal,
        "config": asdict(config),
        "device_rng_probe_sha256_1": rng_hash_1,
        "device_rng_probe_sha256_2": rng_hash_2,
        "elapsed_seconds": elapsed,
        "checks": checks,
        "limitations": [
            "This proves only tiny XLA device/collective/BF16/RNG plumbing, not P35 execution or training quality.",
            "A target checkpoint canary must still bind real model, optimizer, cursor, and remote durability state.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-world-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=47_101)
    parser.add_argument("--matrix-size", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_preflight(PreflightConfig(args.expected_world_size, args.seed, args.matrix_size))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "status": result["status"]}, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
