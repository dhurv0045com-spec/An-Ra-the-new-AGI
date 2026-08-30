"""Certify the executable RoPE path against an independent float64 oracle.

The probe extracts the actual RotaryEmbedding module from the exact P35 model
constructor, then checks FP32 and BF16 rotations through native 4k context for
reference agreement, norm preservation, and relative-shift invariance. It does
not perform attention, backward, optimizer updates, or model training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .block_benchmark import _build_model, shape_arms


DTYPES = ("float32", "bfloat16")
SAMPLE_POSITIONS = (0, 1, 31, 127, 511, 2047, 4095)
CONFORMANCE_LIMITS = {
    # The executable table computes phases in float32 before storing sin/cos;
    # 5e-5 is a conservative bound for accumulated phase-rounding at position
    # 4095, not a fit to any single seed. BF16 receives its representation bound.
    "float32": {
        "reference_relative_rms": 5e-5,
        "reference_cosine": 0.999999,
        "norm_relative_error": 2e-6,
        "shift_relative_rms_error": 5e-5,
    },
    "bfloat16": {
        "reference_relative_rms": 0.01,
        "reference_cosine": 0.9999,
        "norm_relative_error": 0.01,
        "shift_relative_rms_error": 0.02,
    },
}
SHIFT_GROUPS = {
    1: ((0, 1), (127, 128), (2047, 2048), (4094, 4095)),
    31: ((0, 31), (100, 131), (2000, 2031), (4064, 4095)),
    127: ((0, 127), (512, 639), (2048, 2175), (3968, 4095)),
    2048: ((0, 2048), (2047, 4095)),
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class RopeConfig:
    device: str
    context_length: int
    seeds: tuple[int, ...]
    heads: int = 6
    head_dimension: int = 64
    base: float = 10_000.0

    def assert_valid(self) -> None:
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("device must be cpu or cuda")
        if self.context_length != 4096:
            raise ValueError("the registered RoPE canary is native 4096 context")
        if len(self.seeds) < 3 or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("RoPE conformance requires at least three distinct seeds")
        if self.heads <= 0 or self.head_dimension <= 0 or self.head_dimension % 2:
            raise ValueError("invalid RoPE head geometry")
        if self.base <= 1 or any(seed < 0 for seed in self.seeds):
            raise ValueError("invalid RoPE base or seed")
        referenced = [position for pairs in SHIFT_GROUPS.values() for pair in pairs for position in pair]
        if max((*SAMPLE_POSITIONS, *referenced)) >= self.context_length:
            raise ValueError("registered positions exceed context")


def _reference_rotate(base_vectors: Any, positions: Any, *, base: float, torch: Any) -> Any:
    """Independent float64 formula; shape [heads, positions, dimension]."""
    dimension = base_vectors.shape[-1]
    inverse = 1.0 / (
        base
        ** (
            torch.arange(0, dimension, 2, dtype=torch.float64)
            / dimension
        )
    )
    angles = positions.to(dtype=torch.float64)[:, None] * inverse[None, :]
    cosine, sine = angles.cos()[None, :, :], angles.sin()[None, :, :]
    even = base_vectors[:, None, 0::2]
    odd = base_vectors[:, None, 1::2]
    return torch.stack(
        (even * cosine - odd * sine, even * sine + odd * cosine), dim=-1
    ).flatten(-2)


def _relative_rms(reference: Any, candidate: Any, torch: Any) -> float:
    difference = candidate.double() - reference.double()
    denominator = torch.sqrt(reference.double().square().mean()).clamp_min(1e-20)
    return float((torch.sqrt(difference.square().mean()) / denominator).item())


def _cosine(reference: Any, candidate: Any, torch: Any) -> float:
    left, right = reference.double().reshape(-1), candidate.double().reshape(-1)
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    value = float((torch.dot(left, right) / denominator.clamp_min(1e-20)).item())
    return max(-1.0, min(1.0, value))


def _extract_rope(*, config: RopeConfig, dtype: Any, torch: Any) -> Any:
    arm = shape_arms()[1]
    model = _build_model(
        torch, arm, maximum_sequence_length=config.context_length
    ).to(device=torch.device(config.device), dtype=dtype)
    rope = model.blocks[0].attention.rope
    del model
    if config.device == "cuda":
        torch.cuda.empty_cache()
    return rope


def _shift_error(rotated_q: Any, rotated_k: Any, reference_q: Any, reference_k: Any, torch: Any) -> float:
    observed_differences: list[Any] = []
    reference_values: list[Any] = []
    for pairs in SHIFT_GROUPS.values():
        observed = [
            (rotated_q[:, left, :] * rotated_k[:, right, :]).sum(dim=-1).double()
            for left, right in pairs
        ]
        expected = [
            (reference_q[:, left, :] * reference_k[:, right, :]).sum(dim=-1)
            for left, right in pairs
        ]
        anchor = expected[0]
        for value in observed:
            observed_differences.append(value - anchor)
        reference_values.extend(expected)
    differences = torch.cat(observed_differences)
    references = torch.cat(reference_values)
    denominator = torch.sqrt(references.square().mean()).clamp_min(1e-20)
    return float((torch.sqrt(differences.square().mean()) / denominator).item())


def _one_case(
    *, config: RopeConfig, seed: int, dtype_name: str, rope: Any, torch: Any
) -> dict[str, Any]:
    if dtype_name not in DTYPES:
        raise ValueError("unregistered RoPE dtype")
    dtype = torch.float32 if dtype_name == "float32" else torch.bfloat16
    generator = torch.Generator(device="cpu").manual_seed(seed)
    base_q = torch.randn(
        (config.heads, config.head_dimension), generator=generator, dtype=torch.float64
    )
    base_k = torch.randn(
        (config.heads, config.head_dimension), generator=generator, dtype=torch.float64
    )
    positions = torch.arange(config.context_length, dtype=torch.float64)
    reference_q = _reference_rotate(base_q, positions, base=config.base, torch=torch)
    reference_k = _reference_rotate(base_k, positions, base=config.base, torch=torch)
    repeated_q = base_q[:, None, :].expand(-1, config.context_length, -1)
    repeated_k = base_k[:, None, :].expand(-1, config.context_length, -1)
    device = torch.device(config.device)
    implementation_q = rope(repeated_q[None, ...].to(device=device, dtype=dtype))[0]
    implementation_k = rope(repeated_k[None, ...].to(device=device, dtype=dtype))[0]
    implementation_q = implementation_q.detach().double().cpu()
    implementation_k = implementation_k.detach().double().cpu()
    sample_indices = torch.tensor(SAMPLE_POSITIONS, dtype=torch.long)
    sampled_reference = torch.cat(
        (reference_q[:, sample_indices, :], reference_k[:, sample_indices, :]), dim=0
    )
    sampled_implementation = torch.cat(
        (implementation_q[:, sample_indices, :], implementation_k[:, sample_indices, :]),
        dim=0,
    )
    reference_relative_rms = _relative_rms(
        sampled_reference, sampled_implementation, torch
    )
    reference_cosine = _cosine(sampled_reference, sampled_implementation, torch)
    input_norms = torch.cat(
        (
            base_q.norm(dim=-1)[:, None].expand(-1, len(SAMPLE_POSITIONS)),
            base_k.norm(dim=-1)[:, None].expand(-1, len(SAMPLE_POSITIONS)),
        ),
        dim=0,
    )
    output_norms = sampled_implementation.norm(dim=-1)
    norm_relative_error = float(
        ((output_norms - input_norms).abs() / input_norms.clamp_min(1e-20)).max().item()
    )
    shift_relative_rms_error = _shift_error(
        implementation_q, implementation_k, reference_q, reference_k, torch
    )
    finite = bool(
        torch.isfinite(sampled_implementation).all().item()
        and all(
            math.isfinite(value)
            for value in (
                reference_relative_rms,
                reference_cosine,
                norm_relative_error,
                shift_relative_rms_error,
            )
        )
    )
    limits = CONFORMANCE_LIMITS[dtype_name]
    return {
        "seed": seed,
        "dtype": dtype_name,
        "reference_relative_rms_error": reference_relative_rms,
        "reference_cosine": reference_cosine,
        "maximum_norm_relative_error": norm_relative_error,
        "relative_shift_relative_rms_error": shift_relative_rms_error,
        "sample_positions": list(SAMPLE_POSITIONS),
        "checks": {
            "finite": finite,
            "reference_relative_rms_within_limit": reference_relative_rms
            <= limits["reference_relative_rms"],
            "reference_cosine_above_minimum": reference_cosine
            >= limits["reference_cosine"],
            "norm_preservation_within_limit": norm_relative_error
            <= limits["norm_relative_error"],
            "relative_shift_invariance_within_limit": shift_relative_rms_error
            <= limits["shift_relative_rms_error"],
        },
    }


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = (
        "reference_relative_rms_error",
        "reference_cosine",
        "maximum_norm_relative_error",
        "relative_shift_relative_rms_error",
    )
    aggregates: list[dict[str, Any]] = []
    for dtype_name in DTYPES:
        selected = [row for row in rows if row["dtype"] == dtype_name]
        if not selected:
            raise ValueError("missing RoPE dtype rows")
        aggregates.append(
            {
                "dtype": dtype_name,
                "seeds": [row["seed"] for row in selected],
                **{
                    metric: {
                        "median": statistics.median(float(row[metric]) for row in selected),
                        "range": [
                            min(float(row[metric]) for row in selected),
                            max(float(row[metric]) for row in selected),
                        ],
                    }
                    for metric in metrics
                },
            }
        )
    return aggregates


def classify(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_dtype = {
        dtype_name: {
            "cases": len(selected := [row for row in rows if row["dtype"] == dtype_name]),
            "passes_all_preregistered_limits": bool(selected)
            and all(all(row["checks"].values()) for row in selected),
        }
        for dtype_name in DTYPES
    }
    supported = sum(
        bool(value["passes_all_preregistered_limits"]) for value in by_dtype.values()
    )
    verdict = (
        "SUPPORTED_LOCAL_ROPE_CONFORMANCE"
        if supported == len(by_dtype)
        else "CONTRADICTED_LOCAL_ROPE_CONFORMANCE"
        if supported == 0
        else "MIXED_LOCAL_ROPE_CONFORMANCE"
    )
    return {"verdict": verdict, "by_dtype": by_dtype}


def benchmark(config: RopeConfig) -> dict[str, Any]:
    config.assert_valid()
    try:
        import torch
    except ImportError as exc:
        return {"schema": "esoes-e2-rope-conformance/v1", "status": "BLOCKED_TORCH", "reason": str(exc)}
    if config.device == "cuda" and not torch.cuda.is_available():
        return {
            "schema": "esoes-e2-rope-conformance/v1",
            "status": "BLOCKED_CUDA",
            "torch_version": torch.__version__,
        }
    ropes = {
        "float32": _extract_rope(config=config, dtype=torch.float32, torch=torch),
        "bfloat16": _extract_rope(config=config, dtype=torch.bfloat16, torch=torch),
    }
    rows = [
        _one_case(
            config=config,
            seed=seed,
            dtype_name=dtype_name,
            rope=ropes[dtype_name],
            torch=torch,
        )
        for dtype_name in DTYPES
        for seed in config.seeds
    ]
    rows.sort(key=lambda row: (DTYPES.index(row["dtype"]), row["seed"]))
    aggregate = _aggregate(rows)
    classification = classify(rows)
    return {
        "schema": "esoes-e2-rope-conformance/v1",
        "status": "PASS" if all(all(row["checks"].values()) for row in rows) else "FAIL",
        "scope": "actual P35 RoPE versus independent float64 oracle at native 4k; no training",
        "implementation_sha256": _sha256_file(Path(__file__)),
        "model_implementation_sha256": _sha256_file(Path(__file__).with_name("block_benchmark.py")),
        "config": {
            "device": config.device,
            "context_length": config.context_length,
            "seeds": list(config.seeds),
            "heads": config.heads,
            "head_dimension": config.head_dimension,
            "base": config.base,
            "dtypes": list(DTYPES),
        },
        "conformance_limits": CONFORMANCE_LIMITS,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(0) if config.device == "cuda" else platform.processor(),
        "shift_groups": {str(delta): [list(pair) for pair in pairs] for delta, pairs in SHIFT_GROUPS.items()},
        "rows": rows,
        "aggregate": aggregate,
        "classification": classification,
        "limitations": [
            "Conformance proves the implementation's geometry, not that base 10,000 is optimal.",
            "The probe does not measure attention, learning, extrapolation, or cognition.",
            "Only native positions 0..4095 are claimed; no context extrapolation claim is made.",
            "Target TPU/XLA must reproduce the receipt before freeze.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--context-length", type=int, default=4096)
    parser.add_argument("--seed", action="append", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = benchmark(
        RopeConfig(
            device=args.device,
            context_length=args.context_length,
            seeds=tuple(args.seed),
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": result["status"],
                "verdict": result.get("classification", {}).get("verdict"),
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
