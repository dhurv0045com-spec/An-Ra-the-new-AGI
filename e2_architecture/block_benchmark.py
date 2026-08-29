"""Bounded full-stack execution benchmark for the E2 shape arms.

The probe instantiates the exact P35 deep/middle/wide parameter contracts and
measures embedding, pre-RMSNorm blocks, affine QK norm, RoPE, causal MHA,
SwiGLU, tied output projection, cross-entropy, and backward. It performs no
optimizer update and therefore does not train a model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .plan import StaticArm, build_plan


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)
    return ordered[max(0, index)]


def shape_arms() -> tuple[StaticArm, ...]:
    arms = tuple(arm for arm in build_plan().arms if arm.group == "shape")
    if tuple(arm.name for arm in arms) != ("deep-narrow", "middle", "wide-shallow"):
        raise RuntimeError("E2 shape plan drifted")
    return arms


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    device: str
    sequence_lengths: tuple[int, ...]
    batch_size: int
    warmup: int
    repeats: int
    seed: int

    def assert_valid(self) -> None:
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("device must be cpu or cuda")
        if not self.sequence_lengths or any(length <= 0 for length in self.sequence_lengths):
            raise ValueError("sequence lengths must be positive")
        if len(set(self.sequence_lengths)) != len(self.sequence_lengths):
            raise ValueError("sequence lengths must be unique")
        if self.batch_size <= 0 or self.warmup < 0 or self.repeats <= 0:
            raise ValueError("invalid benchmark repetitions or batch size")
        if self.seed < 0:
            raise ValueError("seed must be nonnegative")


def _build_model(torch: Any, arm: StaticArm, *, maximum_sequence_length: int) -> Any:
    import torch.nn as nn
    import torch.nn.functional as functional

    specification = arm.model

    class RMSNorm(nn.Module):
        def __init__(self, width: int, epsilon: float) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(width))
            self.epsilon = epsilon

        def forward(self, tensor: Any) -> Any:
            scale = torch.rsqrt(
                tensor.float().square().mean(dim=-1, keepdim=True) + self.epsilon
            )
            return tensor * scale.to(dtype=tensor.dtype) * self.weight

    class RotaryEmbedding(nn.Module):
        def __init__(self, dimension: int, length: int, base: float) -> None:
            super().__init__()
            inverse = 1.0 / (
                base ** (torch.arange(0, dimension, 2, dtype=torch.float32) / dimension)
            )
            positions = torch.arange(length, dtype=torch.float32)
            angles = torch.outer(positions, inverse)
            self.register_buffer("cosine", angles.cos(), persistent=False)
            self.register_buffer("sine", angles.sin(), persistent=False)

        def forward(self, tensor: Any) -> Any:
            length = tensor.shape[-2]
            cosine = self.cosine[:length].to(dtype=tensor.dtype)[None, None, :, :]
            sine = self.sine[:length].to(dtype=tensor.dtype)[None, None, :, :]
            even, odd = tensor[..., 0::2], tensor[..., 1::2]
            rotated_even = even * cosine - odd * sine
            rotated_odd = even * sine + odd * cosine
            return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2)

    class Attention(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            width = specification.width
            kv_width = specification.kv_heads * specification.head_dimension
            self.query = nn.Linear(width, width, bias=False)
            self.key = nn.Linear(width, kv_width, bias=False)
            self.value = nn.Linear(width, kv_width, bias=False)
            self.output = nn.Linear(width, width, bias=False)
            self.query_scale = nn.Parameter(
                torch.ones(specification.query_heads, specification.head_dimension)
            )
            self.key_scale = nn.Parameter(
                torch.ones(specification.kv_heads, specification.head_dimension)
            )
            self.rope = RotaryEmbedding(
                specification.head_dimension,
                maximum_sequence_length,
                specification.rope_base,
            )

        @staticmethod
        def normalize(tensor: Any, scale: Any) -> Any:
            inverse = torch.rsqrt(tensor.float().square().mean(dim=-1, keepdim=True) + 1e-6)
            return tensor * inverse.to(dtype=tensor.dtype) * scale[None, :, None, :]

        def forward(self, hidden: Any) -> Any:
            batch, length, _ = hidden.shape
            query = self.query(hidden).view(
                batch, length, specification.query_heads, specification.head_dimension
            ).transpose(1, 2)
            key = self.key(hidden).view(
                batch, length, specification.kv_heads, specification.head_dimension
            ).transpose(1, 2)
            value = self.value(hidden).view(
                batch, length, specification.kv_heads, specification.head_dimension
            ).transpose(1, 2)
            query = self.rope(self.normalize(query, self.query_scale))
            key = self.rope(self.normalize(key, self.key_scale))
            attended = functional.scaled_dot_product_attention(
                query, key, value, is_causal=True, enable_gqa=False
            )
            attended = attended.transpose(1, 2).contiguous().view(
                batch, length, specification.width
            )
            return self.output(attended)

    class FeedForward(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gate = nn.Linear(specification.width, specification.ffn_width, bias=False)
            self.up = nn.Linear(specification.width, specification.ffn_width, bias=False)
            self.down = nn.Linear(specification.ffn_width, specification.width, bias=False)

        def forward(self, hidden: Any) -> Any:
            return self.down(functional.silu(self.gate(hidden)) * self.up(hidden))

    class Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attention_norm = RMSNorm(specification.width, specification.norm_epsilon)
            self.attention = Attention()
            self.ffn_norm = RMSNorm(specification.width, specification.norm_epsilon)
            self.ffn = FeedForward()

        def forward(self, hidden: Any) -> Any:
            hidden = hidden + self.attention(self.attention_norm(hidden))
            return hidden + self.ffn(self.ffn_norm(hidden))

    class ProbeModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = nn.Embedding(specification.vocabulary_size, specification.width)
            self.blocks = nn.ModuleList(Block() for _ in range(specification.layers))
            self.final_norm = RMSNorm(specification.width, specification.norm_epsilon)

        def forward(self, token_ids: Any) -> Any:
            hidden = self.embedding(token_ids)
            for block in self.blocks:
                hidden = block(hidden)
            hidden = self.final_norm(hidden)
            return functional.linear(hidden, self.embedding.weight)

    return ProbeModel()


def _timed(operation: Any, *, config: BenchmarkConfig, torch: Any) -> list[float]:
    for _ in range(config.warmup):
        operation()
    if config.device == "cuda":
        torch.cuda.synchronize()
    durations: list[float] = []
    for _ in range(config.repeats):
        if config.device == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            operation()
            end.record()
            end.synchronize()
            durations.append(float(start.elapsed_time(end)))
        else:
            started = time.perf_counter()
            operation()
            durations.append((time.perf_counter() - started) * 1_000.0)
    return durations


def _summary(milliseconds: list[float], *, tokens: int) -> dict[str, float]:
    median = statistics.median(milliseconds)
    return {
        "minimum_ms": min(milliseconds),
        "median_ms": median,
        "p95_ms": _percentile(milliseconds, 0.95),
        "tokens_per_second": tokens / (median / 1_000.0),
    }


def benchmark(config: BenchmarkConfig) -> dict[str, Any]:
    config.assert_valid()
    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as exc:
        return {"schema": "esoes-e2-full-stack-benchmark/v1", "status": "BLOCKED_TORCH", "reason": str(exc)}
    if config.device == "cuda" and not torch.cuda.is_available():
        return {
            "schema": "esoes-e2-full-stack-benchmark/v1",
            "status": "BLOCKED_CUDA",
            "torch_version": torch.__version__,
        }

    device = torch.device(config.device)
    dtype = (
        torch.bfloat16
        if config.device == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float32
    )
    torch.manual_seed(config.seed)
    if config.device == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    rows: list[dict[str, Any]] = []
    all_pass = True
    tasks = [
        (arm, sequence_length)
        for arm in shape_arms()
        for sequence_length in config.sequence_lengths
    ]
    random.Random(config.seed).shuffle(tasks)
    execution_order = [f"{arm.name}:{sequence_length}" for arm, sequence_length in tasks]
    arm_order = {arm.name: index for index, arm in enumerate(shape_arms())}
    for arm, sequence_length in tasks:
        torch.manual_seed(config.seed)
        model = _build_model(
            torch, arm, maximum_sequence_length=max(config.sequence_lengths)
        ).to(device=device, dtype=dtype)
        expected_parameters = arm.model.parameter_receipt().total
        actual_parameters = sum(parameter.numel() for parameter in model.parameters())
        if actual_parameters != expected_parameters:
            raise RuntimeError(
                f"{arm.name} parameter mismatch: {actual_parameters} != {expected_parameters}"
            )
        token_ids = torch.randint(
            0,
            arm.model.vocabulary_size,
            (config.batch_size, sequence_length),
            device=device,
        )
        targets = torch.randint(
            0,
            arm.model.vocabulary_size,
            (config.batch_size, sequence_length),
            device=device,
        )

        def forward_only() -> None:
            with torch.inference_mode():
                model(token_ids)

        def forward_backward() -> None:
            model.zero_grad(set_to_none=True)
            logits = model(token_ids)
            loss = functional.cross_entropy(
                logits.float().view(-1, arm.model.vocabulary_size), targets.view(-1)
            )
            loss.backward()

        model.zero_grad(set_to_none=True)
        logits = model(token_ids)
        loss = functional.cross_entropy(
            logits.float().view(-1, arm.model.vocabulary_size), targets.view(-1)
        )
        loss.backward()
        finite_loss = bool(torch.isfinite(loss).item())
        gradient_tensors = [
            parameter.grad for parameter in model.parameters() if parameter.grad is not None
        ]
        gradients_finite = bool(
            gradient_tensors
            and all(torch.isfinite(gradient).all().item() for gradient in gradient_tensors)
        )
        correctness = {
            "parameter_count_exact": actual_parameters == expected_parameters,
            "finite_loss": finite_loss,
            "all_gradients_finite": gradients_finite,
            "parameter_tensors_with_gradient": len(gradient_tensors),
        }
        all_pass = all_pass and all(
            value for key, value in correctness.items() if key != "parameter_tensors_with_gradient"
        )
        del logits, loss, gradient_tensors
        model.zero_grad(set_to_none=True)
        if config.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        forward_times = _timed(forward_only, config=config, torch=torch)
        forward_peak = (
            int(torch.cuda.max_memory_allocated()) if config.device == "cuda" else None
        )
        if config.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        backward_times = _timed(forward_backward, config=config, torch=torch)
        backward_peak = (
            int(torch.cuda.max_memory_allocated()) if config.device == "cuda" else None
        )
        rows.append(
            {
                "arm": arm.name,
                "model_sha256": arm.model.sha256(),
                "layers": arm.model.layers,
                "width": arm.model.width,
                "ffn_width": arm.model.ffn_width,
                "query_heads": arm.model.query_heads,
                "parameters": actual_parameters,
                "sequence_length": sequence_length,
                "batch_size": config.batch_size,
                "tokens": config.batch_size * sequence_length,
                "dtype": str(dtype).removeprefix("torch."),
                "correctness": correctness,
                "forward": _summary(
                    forward_times, tokens=config.batch_size * sequence_length
                ),
                "forward_peak_allocated_bytes": forward_peak,
                "forward_backward": _summary(
                    backward_times, tokens=config.batch_size * sequence_length
                ),
                "forward_backward_peak_allocated_bytes": backward_peak,
            }
        )
        del model, token_ids, targets
        if config.device == "cuda":
            torch.cuda.empty_cache()

    rows.sort(key=lambda row: (arm_order[row["arm"]], row["sequence_length"]))

    return {
        "schema": "esoes-e2-full-stack-benchmark/v1",
        "status": "PASS" if all_pass else "FAIL",
        "scope": "exact P35 shape stacks; forward/backward only; no optimizer update",
        "implementation_sha256": _sha256_file(Path(__file__)),
        "static_plan_sha256": _sha256_file(Path(__file__).with_name("plan.py")),
        "config": {
            "device": config.device,
            "sequence_lengths": list(config.sequence_lengths),
            "batch_size": config.batch_size,
            "warmup": config.warmup,
            "repeats": config.repeats,
            "seed": config.seed,
        },
        "execution_order": execution_order,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(0) if config.device == "cuda" else platform.processor(),
        "rows": rows,
        "limitations": [
            "Synthetic random tokens measure execution, not learning or cognition.",
            "No optimizer step, data pipeline, checkpoint, compiler, or distributed collective is included.",
            "The eager laptop result is not TPU/XLA throughput evidence.",
            "Batch one does not select the final global or microbatch size.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--sequence-length", action="append", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=32_001)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = BenchmarkConfig(
        device=args.device,
        sequence_lengths=tuple(args.sequence_length),
        batch_size=args.batch_size,
        warmup=args.warmup,
        repeats=args.repeats,
        seed=args.seed,
    )
    result = benchmark(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(args.output)}, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
