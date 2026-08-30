"""Compare exact-stack BF16 forward/backward numerics with FP32.

Each case uses identical scaled-residual weights, token IDs, and targets. The
probe compares logits, cross-entropy, and representative parameter gradients
after one backward pass. It performs no optimizer update and does not train.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .block_benchmark import _build_model, shape_arms
from .signal_benchmark import _initialize


LOSS_RELATIVE_ERROR_LIMIT = 0.005
LOGIT_COSINE_MINIMUM = 0.995
LOGIT_RELATIVE_RMS_ERROR_LIMIT = 0.10
GRADIENT_COSINE_MINIMUM = 0.98
GRADIENT_RELATIVE_RMS_ERROR_LIMIT = 0.20


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class PrecisionConfig:
    device: str
    sequence_length: int
    batch_size: int
    seeds: tuple[int, ...]

    def assert_valid(self) -> None:
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("device must be cpu or cuda")
        if self.sequence_length <= 0 or self.batch_size <= 0:
            raise ValueError("sequence length and batch size must be positive")
        if len(self.seeds) < 3 or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("precision parity requires at least three distinct seeds")
        if any(seed < 0 for seed in self.seeds):
            raise ValueError("seeds must be nonnegative")


def _rms(tensor: Any, torch: Any) -> float:
    return float(torch.sqrt(tensor.detach().float().square().mean()).item())


def _relative_rms_error(reference: Any, candidate: Any, torch: Any) -> float:
    difference = candidate.float() - reference.float()
    denominator = torch.sqrt(reference.float().square().mean()).clamp_min(1e-12)
    return float((torch.sqrt(difference.square().mean()) / denominator).item())


def _cosine(reference: Any, candidate: Any, torch: Any) -> float:
    # Accumulate in float64 so a near-perfect high-dimensional cosine does not
    # round above one in float32 and then lose information through clamping.
    left = reference.double().reshape(-1)
    right = candidate.double().reshape(-1)
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    value = float((torch.dot(left, right) / denominator.clamp_min(1e-20)).item())
    return max(-1.0, min(1.0, value))


def _gradient_names(layers: int) -> tuple[str, ...]:
    return (
        "blocks.0.attention.query.weight",
        f"blocks.{layers // 2}.attention.output.weight",
        f"blocks.{layers - 1}.ffn.down.weight",
        "final_norm.weight",
    )


def _evaluate(
    model: Any,
    *,
    token_ids: Any,
    targets: Any,
    gradient_names: tuple[str, ...],
    torch: Any,
) -> dict[str, Any]:
    import torch.nn.functional as functional

    model.zero_grad(set_to_none=True)
    logits = model(token_ids)
    loss = functional.cross_entropy(
        logits.float().view(-1, logits.shape[-1]), targets.view(-1)
    )
    loss.backward()
    named_parameters = dict(model.named_parameters())
    gradients: dict[str, Any] = {}
    for name in gradient_names:
        parameter = named_parameters[name]
        if parameter.grad is None:
            raise RuntimeError(f"missing parity gradient: {name}")
        gradients[name] = parameter.grad.detach().float().cpu().clone()
    finite = bool(
        torch.isfinite(logits).all().item()
        and torch.isfinite(loss).item()
        and all(torch.isfinite(gradient).all().item() for gradient in gradients.values())
    )
    result = {
        "logits": logits.detach().float().cpu(),
        "loss": float(loss.item()),
        "gradients": gradients,
        "finite": finite,
    }
    del logits, loss
    model.zero_grad(set_to_none=True)
    return result


def _one_case(*, arm: Any, seed: int, config: PrecisionConfig, torch: Any) -> dict[str, Any]:
    device = torch.device(config.device)
    torch.manual_seed(seed)
    if config.device == "cuda":
        torch.cuda.manual_seed_all(seed)
    fp32_model = _build_model(
        torch, arm, maximum_sequence_length=config.sequence_length
    ).to(device=device, dtype=torch.float32)
    torch.manual_seed(seed + 1_000_003)
    _initialize(
        fp32_model,
        policy="scaled-residual-0.02",
        layers=arm.model.layers,
        torch=torch,
    )
    state = {
        name: tensor.detach().cpu().clone() for name, tensor in fp32_model.state_dict().items()
    }
    generator = torch.Generator(device=config.device).manual_seed(seed + 2_000_003)
    token_ids = torch.randint(
        0,
        arm.model.vocabulary_size,
        (config.batch_size, config.sequence_length),
        generator=generator,
        device=device,
    )
    targets = torch.randint(
        0,
        arm.model.vocabulary_size,
        (config.batch_size, config.sequence_length),
        generator=generator,
        device=device,
    )
    gradient_names = _gradient_names(arm.model.layers)
    fp32 = _evaluate(
        fp32_model,
        token_ids=token_ids,
        targets=targets,
        gradient_names=gradient_names,
        torch=torch,
    )
    actual_parameters = sum(parameter.numel() for parameter in fp32_model.parameters())
    del fp32_model
    if config.device == "cuda":
        torch.cuda.empty_cache()

    bf16_model = _build_model(
        torch, arm, maximum_sequence_length=config.sequence_length
    ).to(device=device, dtype=torch.bfloat16)
    bf16_model.load_state_dict(state, strict=True)
    bf16 = _evaluate(
        bf16_model,
        token_ids=token_ids,
        targets=targets,
        gradient_names=gradient_names,
        torch=torch,
    )
    del bf16_model, state, token_ids, targets
    if config.device == "cuda":
        torch.cuda.empty_cache()

    logit_cosine = _cosine(fp32["logits"], bf16["logits"], torch)
    logit_relative_rms_error = _relative_rms_error(
        fp32["logits"], bf16["logits"], torch
    )
    logit_maximum_absolute_error = float(
        (bf16["logits"] - fp32["logits"]).abs().max().item()
    )
    top1_agreement = float(
        (bf16["logits"].argmax(dim=-1) == fp32["logits"].argmax(dim=-1))
        .float()
        .mean()
        .item()
    )
    loss_absolute_error = abs(bf16["loss"] - fp32["loss"])
    loss_relative_error = loss_absolute_error / max(abs(fp32["loss"]), 1e-12)
    gradient_rows: list[dict[str, Any]] = []
    for name in gradient_names:
        reference = fp32["gradients"][name]
        candidate = bf16["gradients"][name]
        gradient_rows.append(
            {
                "name": name,
                "fp32_rms": _rms(reference, torch),
                "bf16_rms": _rms(candidate, torch),
                "cosine": _cosine(reference, candidate, torch),
                "relative_rms_error": _relative_rms_error(reference, candidate, torch),
                "maximum_absolute_error": float((candidate - reference).abs().max().item()),
            }
        )
    minimum_gradient_cosine = min(row["cosine"] for row in gradient_rows)
    maximum_gradient_relative_rms_error = max(
        row["relative_rms_error"] for row in gradient_rows
    )
    finite = fp32["finite"] and bf16["finite"] and all(
        math.isfinite(float(value))
        for value in (
            logit_cosine,
            logit_relative_rms_error,
            loss_relative_error,
            minimum_gradient_cosine,
            maximum_gradient_relative_rms_error,
        )
    )
    fp32_loss = fp32["loss"]
    bf16_loss = bf16["loss"]
    del fp32, bf16
    return {
        "arm": arm.name,
        "model_sha256": arm.model.sha256(),
        "layers": arm.model.layers,
        "width": arm.model.width,
        "parameters": actual_parameters,
        "seed": seed,
        "sequence_length": config.sequence_length,
        "fp32_loss": fp32_loss,
        "bf16_loss": bf16_loss,
        "loss_absolute_error": loss_absolute_error,
        "loss_relative_error": loss_relative_error,
        "logit_cosine": logit_cosine,
        "logit_relative_rms_error": logit_relative_rms_error,
        "logit_maximum_absolute_error": logit_maximum_absolute_error,
        "top1_agreement": top1_agreement,
        "minimum_gradient_cosine": minimum_gradient_cosine,
        "maximum_gradient_relative_rms_error": maximum_gradient_relative_rms_error,
        "gradients": gradient_rows,
        "checks": {
            "parameter_count_exact": actual_parameters == arm.model.parameter_receipt().total,
            "finite": finite,
            "loss_relative_error_within_limit": loss_relative_error
            <= LOSS_RELATIVE_ERROR_LIMIT,
            "logit_cosine_above_minimum": logit_cosine >= LOGIT_COSINE_MINIMUM,
            "logit_relative_rms_error_within_limit": logit_relative_rms_error
            <= LOGIT_RELATIVE_RMS_ERROR_LIMIT,
            "gradient_cosine_above_minimum": minimum_gradient_cosine
            >= GRADIENT_COSINE_MINIMUM,
            "gradient_relative_rms_error_within_limit": maximum_gradient_relative_rms_error
            <= GRADIENT_RELATIVE_RMS_ERROR_LIMIT,
        },
    }


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = (
        "loss_relative_error",
        "logit_cosine",
        "logit_relative_rms_error",
        "logit_maximum_absolute_error",
        "top1_agreement",
        "minimum_gradient_cosine",
        "maximum_gradient_relative_rms_error",
    )
    result: list[dict[str, Any]] = []
    for arm in ("deep-narrow", "middle", "wide-shallow"):
        selected = [row for row in rows if row["arm"] == arm]
        if not selected:
            raise ValueError("missing precision-parity shape")
        result.append(
            {
                "arm": arm,
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
    return result


def classify(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_arm: dict[str, Any] = {}
    for arm in ("deep-narrow", "middle", "wide-shallow"):
        selected = [row for row in rows if row["arm"] == arm]
        passes = bool(selected) and all(all(row["checks"].values()) for row in selected)
        by_arm[arm] = {
            "cases": len(selected),
            "passes_all_preregistered_limits": passes,
        }
    supported = sum(
        bool(row["passes_all_preregistered_limits"]) for row in by_arm.values()
    )
    verdict = (
        "SUPPORTED_LOCAL_BF16_PARITY"
        if supported == len(by_arm)
        else "CONTRADICTED_LOCAL_BF16_PARITY"
        if supported == 0
        else "MIXED_LOCAL_BF16_PARITY"
    )
    return {"verdict": verdict, "by_arm": by_arm}


def benchmark(config: PrecisionConfig) -> dict[str, Any]:
    config.assert_valid()
    try:
        import torch
    except ImportError as exc:
        return {"schema": "esoes-e2-precision-parity/v1", "status": "BLOCKED_TORCH", "reason": str(exc)}
    if config.device == "cuda" and not torch.cuda.is_available():
        return {
            "schema": "esoes-e2-precision-parity/v1",
            "status": "BLOCKED_CUDA",
            "torch_version": torch.__version__,
        }
    cases = [(arm, seed) for arm in shape_arms() for seed in config.seeds]
    random.Random(sum(config.seeds)).shuffle(cases)
    rows = [_one_case(arm=arm, seed=seed, config=config, torch=torch) for arm, seed in cases]
    arm_order = {name: index for index, name in enumerate(("deep-narrow", "middle", "wide-shallow"))}
    rows.sort(key=lambda row: (arm_order[row["arm"]], row["seed"]))
    aggregate = _aggregate(rows)
    classification = classify(rows)
    all_finite_and_exact = all(
        row["checks"]["finite"] and row["checks"]["parameter_count_exact"] for row in rows
    )
    return {
        "schema": "esoes-e2-precision-parity/v1",
        "status": "PASS" if all_finite_and_exact else "FAIL",
        "scope": "exact P35 FP32/BF16 forward/backward parity; no optimizer update",
        "implementation_sha256": _sha256_file(Path(__file__)),
        "model_implementation_sha256": _sha256_file(Path(__file__).with_name("block_benchmark.py")),
        "initialization_implementation_sha256": _sha256_file(Path(__file__).with_name("signal_benchmark.py")),
        "config": {
            "device": config.device,
            "sequence_length": config.sequence_length,
            "batch_size": config.batch_size,
            "seeds": list(config.seeds),
            "reference_dtype": "float32",
            "candidate_dtype": "bfloat16",
        },
        "thresholds": {
            "loss_relative_error_limit": LOSS_RELATIVE_ERROR_LIMIT,
            "logit_cosine_minimum": LOGIT_COSINE_MINIMUM,
            "logit_relative_rms_error_limit": LOGIT_RELATIVE_RMS_ERROR_LIMIT,
            "gradient_cosine_minimum": GRADIENT_COSINE_MINIMUM,
            "gradient_relative_rms_error_limit": GRADIENT_RELATIVE_RMS_ERROR_LIMIT,
        },
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(0) if config.device == "cuda" else platform.processor(),
        "rows": rows,
        "aggregate": aggregate,
        "classification": classification,
        "limitations": [
            "Initialization parity does not establish long-run optimizer or loss-scaling stability.",
            "Representative gradients are checked, not every gradient element across the model.",
            "Random tokens do not measure learned cognition or data-dependent numerical tails.",
            "The target TPU/XLA stack must repeat this receipt before freeze.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--sequence-length", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", action="append", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = benchmark(
        PrecisionConfig(
            device=args.device,
            sequence_length=args.sequence_length,
            batch_size=args.batch_size,
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
