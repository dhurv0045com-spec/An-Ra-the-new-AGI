"""Bounded CUDA benchmark for E2 attention topology and context hypotheses.

This is deliberately an attention-kernel probe, not a model benchmark and not
training. It measures matched causal SDPA forward and forward/backward work on
the available device, records peak allocated memory, and verifies that native
GQA agrees with explicitly repeated K/V heads on a small deterministic case.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class AttentionCase:
    name: str
    query_heads: int
    kv_heads: int
    context_length: int
    head_dimension: int = 64
    qk_norm: bool = True
    implementation: str = "native"

    def assert_valid(self) -> None:
        if not self.name or min(
            self.query_heads, self.kv_heads, self.context_length, self.head_dimension
        ) <= 0:
            raise ValueError("attention benchmark dimensions must be positive")
        if self.query_heads % self.kv_heads:
            raise ValueError("query heads must be divisible by KV heads")
        if self.implementation not in {"native", "repeat-kv"}:
            raise ValueError("unknown attention implementation")
        if self.implementation == "repeat-kv" and self.query_heads == self.kv_heads:
            raise ValueError("repeat-kv is meaningful only for grouped-query attention")


def default_cases() -> tuple[AttentionCase, ...]:
    return (
        AttentionCase("mha-qk-2k", 6, 6, 2_048, qk_norm=True),
        AttentionCase("gqa-qk-2k", 6, 2, 2_048, qk_norm=True),
        AttentionCase(
            "gqa-repeat-kv-qk-2k", 6, 2, 2_048, qk_norm=True, implementation="repeat-kv"
        ),
        AttentionCase("gqa-no-qk-2k", 6, 2, 2_048, qk_norm=False),
        AttentionCase("gqa-qk-4k", 6, 2, 4_096, qk_norm=True),
    )


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)
    return ordered[max(0, index)]


def _qk_normalize(tensor: Any, *, epsilon: float = 1e-6) -> Any:
    import torch

    scale = torch.rsqrt(tensor.float().square().mean(dim=-1, keepdim=True) + epsilon)
    return tensor * scale.to(dtype=tensor.dtype)


def _attention(case: AttentionCase, query: Any, key: Any, value: Any) -> Any:
    import torch.nn.functional as functional

    if case.qk_norm:
        query = _qk_normalize(query)
        key = _qk_normalize(key)
    native_gqa = case.query_heads != case.kv_heads
    if native_gqa and case.implementation == "repeat-kv":
        groups = case.query_heads // case.kv_heads
        key = key.repeat_interleave(groups, dim=1)
        value = value.repeat_interleave(groups, dim=1)
        native_gqa = False
    return functional.scaled_dot_product_attention(
        query,
        key,
        value,
        is_causal=True,
        enable_gqa=native_gqa,
    )


def _native_gqa_backend_support(torch: Any, *, dtype: Any) -> dict[str, dict[str, str | bool]]:
    import torch.nn.functional as functional
    from torch.nn.attention import SDPBackend, sdpa_kernel

    query = torch.randn((1, 6, 128, 64), device="cuda", dtype=dtype)
    key = torch.randn((1, 2, 128, 64), device="cuda", dtype=dtype)
    value = torch.randn((1, 2, 128, 64), device="cuda", dtype=dtype)
    results: dict[str, dict[str, str | bool]] = {}
    for name, backend in (
        ("flash", SDPBackend.FLASH_ATTENTION),
        ("efficient", SDPBackend.EFFICIENT_ATTENTION),
        ("math", SDPBackend.MATH),
    ):
        try:
            with sdpa_kernel(backend):
                output = functional.scaled_dot_product_attention(
                    query, key, value, is_causal=True, enable_gqa=True
                )
                torch.cuda.synchronize()
            results[name] = {"supported": True, "finite": bool(torch.isfinite(output).all().item())}
        except Exception as exc:  # backend support is runtime-specific evidence
            results[name] = {
                "supported": False,
                "error": f"{type(exc).__name__}: {exc}"[:500],
            }
    return results


def _timed_cuda(operation: Any, *, warmup: int, repeats: int, torch: Any) -> list[float]:
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()
    durations: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        operation()
        end.record()
        end.synchronize()
        durations.append(float(start.elapsed_time(end)))
    return durations


def _summary(milliseconds: list[float], *, tokens: int) -> dict[str, float]:
    median_ms = statistics.median(milliseconds)
    return {
        "median_ms": median_ms,
        "p95_ms": _percentile(milliseconds, 0.95),
        "minimum_ms": min(milliseconds),
        "tokens_per_second_at_batch_one": tokens / (median_ms / 1_000.0),
    }


def _gqa_equivalence(torch: Any, *, dtype: Any) -> dict[str, Any]:
    import torch.nn.functional as functional

    generator = torch.Generator(device="cuda").manual_seed(81_923)
    query = torch.randn((1, 6, 128, 64), generator=generator, device="cuda", dtype=dtype)
    key = torch.randn((1, 2, 128, 64), generator=generator, device="cuda", dtype=dtype)
    value = torch.randn((1, 2, 128, 64), generator=generator, device="cuda", dtype=dtype)
    native = functional.scaled_dot_product_attention(
        query, key, value, is_causal=True, enable_gqa=True
    )
    repeated = functional.scaled_dot_product_attention(
        query,
        key.repeat_interleave(3, dim=1),
        value.repeat_interleave(3, dim=1),
        is_causal=True,
    )
    difference = (native.float() - repeated.float()).abs()
    maximum = float(difference.max().item())
    finite = bool(torch.isfinite(native).all().item())
    return {
        "finite": finite,
        "maximum_absolute_error": maximum,
        "mean_absolute_error": float(difference.mean().item()),
        "tolerance": 0.02,
        "pass": finite and maximum <= 0.02,
    }


def benchmark(
    *, warmup: int = 5, repeats: int = 20, seed: int = 31_001
) -> dict[str, Any]:
    if warmup < 0 or repeats <= 0:
        raise ValueError("warmup must be nonnegative and repeats must be positive")
    try:
        import torch
    except ImportError as exc:
        return {
            "schema": "esoes-e2-device-benchmark/v1",
            "status": "BLOCKED_TORCH",
            "reason": str(exc),
        }
    if not torch.cuda.is_available():
        return {
            "schema": "esoes-e2-device-benchmark/v1",
            "status": "BLOCKED_CUDA",
            "torch_version": torch.__version__,
        }

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    torch.manual_seed(seed)
    rows: list[dict[str, Any]] = []
    for case in default_cases():
        case.assert_valid()
        shape_q = (1, case.query_heads, case.context_length, case.head_dimension)
        shape_kv = (1, case.kv_heads, case.context_length, case.head_dimension)
        query = torch.randn(shape_q, device="cuda", dtype=dtype, requires_grad=True)
        key = torch.randn(shape_kv, device="cuda", dtype=dtype, requires_grad=True)
        value = torch.randn(shape_kv, device="cuda", dtype=dtype, requires_grad=True)

        def forward() -> None:
            with torch.inference_mode():
                output = _attention(case, query, key, value)
                if not torch.isfinite(output).all():
                    raise RuntimeError(f"non-finite forward output in {case.name}")

        def train_step() -> None:
            for tensor in (query, key, value):
                tensor.grad = None
            output = _attention(case, query, key, value)
            output.float().square().mean().backward()

        torch.cuda.reset_peak_memory_stats()
        forward_ms = _timed_cuda(forward, warmup=warmup, repeats=repeats, torch=torch)
        forward_peak = int(torch.cuda.max_memory_allocated())
        torch.cuda.reset_peak_memory_stats()
        training_ms = _timed_cuda(train_step, warmup=warmup, repeats=repeats, torch=torch)
        training_peak = int(torch.cuda.max_memory_allocated())
        rows.append(
            {
                **asdict(case),
                "dtype": str(dtype).removeprefix("torch."),
                "forward": _summary(forward_ms, tokens=case.context_length),
                "forward_peak_allocated_bytes": forward_peak,
                "forward_backward": _summary(training_ms, tokens=case.context_length),
                "forward_backward_peak_allocated_bytes": training_peak,
            }
        )
        del query, key, value
        torch.cuda.empty_cache()

    equivalence = _gqa_equivalence(torch, dtype=dtype)
    return {
        "schema": "esoes-e2-device-benchmark/v1",
        "status": "PASS" if equivalence["pass"] else "FAIL",
        "scope": "isolated causal SDPA plus optional RMS QK normalization; not a full model",
        "implementation_sha256": _sha256_file(Path(__file__)),
        "seed": seed,
        "warmup": warmup,
        "repeats": repeats,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(0),
        "device_total_memory_bytes": int(torch.cuda.get_device_properties(0).total_memory),
        "bf16_supported": bool(torch.cuda.is_bf16_supported()),
        "gqa_equivalence": equivalence,
        "native_gqa_backend_support": _native_gqa_backend_support(torch, dtype=dtype),
        "cases": rows,
        "limitations": [
            "The probe excludes projections, FFN, optimizer, collectives, and input pipeline.",
            "Kernel speed and memory do not establish cognition quality.",
            "Laptop GPU results are not TPU throughput evidence.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=31_001)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = benchmark(warmup=args.warmup, repeats=args.repeats, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(args.output)}, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
