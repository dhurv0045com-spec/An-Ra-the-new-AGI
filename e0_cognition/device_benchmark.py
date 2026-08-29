"""Bounded local device and E0 harness benchmark.

This is an evidence probe, not a model trainer.  It measures the executable
E0 workload on CPU and, when an optional PyTorch CUDA runtime is present, a
small matched matrix-multiply smoke test on CPU and CUDA.  The probe keeps
allocations small enough for laptop GPUs and never mutates checkpoints.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

from .baselines import evaluate_all_baselines
from .contracts import Split
from .evaluation_generators import build_evaluation_suite


def _e0_probe() -> dict[str, Any]:
    start = time.perf_counter()
    suite = build_evaluation_suite(Split.DEVELOPMENT, seed=271828, groups_per_family=16)
    generated = time.perf_counter()
    evaluate_all_baselines(suite)
    finished = time.perf_counter()
    elapsed = finished - start
    return {
        "cases": len(suite.cases),
        "pairs": len(suite.pairs),
        "generation_seconds": generated - start,
        "baseline_seconds": finished - generated,
        "total_seconds": elapsed,
        "cases_per_second": len(suite.cases) / elapsed if elapsed else 0.0,
    }


def _torch_probe(*, size: int, warmup: int, repeats: int) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        return {"available": False, "reason": f"torch unavailable: {exc}"}

    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    results: dict[str, Any] = {
        "available": True,
        "version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "devices": {},
    }
    for device in devices:
        left = torch.randn((size, size), device=device, dtype=torch.float32)
        right = torch.randn((size, size), device=device, dtype=torch.float32)
        for _ in range(warmup):
            torch.mm(left, right)
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(repeats):
            torch.mm(left, right)
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        operations = 2 * (size**3) * repeats
        result: dict[str, Any] = {
            "size": size,
            "warmup": warmup,
            "repeats": repeats,
            "seconds": elapsed,
            "gflop_per_second": operations / elapsed / 1e9 if elapsed else 0.0,
        }
        if device == "cuda":
            result["device_name"] = torch.cuda.get_device_name(0)
            result["peak_memory_bytes"] = torch.cuda.max_memory_allocated(0)
        results["devices"][device] = result
    return results


def collect_probe(*, size: int = 1024, warmup: int = 5, repeats: int = 20) -> dict[str, Any]:
    """Collect a bounded device snapshot and E0 CPU workload measurement."""

    return {
        "schema": "esoes-local-device-probe/v1",
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_processors": os.cpu_count(),
        "e0_cpu": _e0_probe(),
        "torch": _torch_probe(size=size, warmup=warmup, repeats=repeats),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.size <= 0 or args.warmup < 0 or args.repeats <= 0:
        parser.error("size and repeats must be positive; warmup cannot be negative")
    probe = collect_probe(size=args.size, warmup=args.warmup, repeats=args.repeats)
    encoded = json.dumps(probe, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
