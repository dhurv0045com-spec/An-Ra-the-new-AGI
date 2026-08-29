"""Measure activation/gradient propagation under candidate initialization policies.

This probe compares normal(0.02) against the same initialization with attention
output and FFN-down projections scaled by 1/sqrt(2L). It uses exact P35 stacks,
performs one forward/backward per case, and never performs an optimizer update.
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


POLICIES = ("normal-0.02", "scaled-residual-0.02")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class SignalConfig:
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
            raise ValueError("signal propagation requires at least three distinct seeds")
        if any(seed < 0 for seed in self.seeds):
            raise ValueError("seeds must be nonnegative")


def _initialize(model: Any, *, policy: str, layers: int, torch: Any) -> dict[str, float]:
    if policy not in POLICIES:
        raise ValueError("unknown initialization policy")
    residual_std = 0.02 / math.sqrt(2 * layers) if policy == "scaled-residual-0.02" else 0.02
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith("query_scale") or name.endswith("key_scale"):
                parameter.fill_(1.0)
            elif "_norm.weight" in name or name == "final_norm.weight":
                parameter.fill_(1.0)
            elif parameter.ndim >= 2:
                standard_deviation = (
                    residual_std
                    if name.endswith("attention.output.weight")
                    or name.endswith("ffn.down.weight")
                    else 0.02
                )
                torch.nn.init.normal_(parameter, mean=0.0, std=standard_deviation)
            else:
                raise RuntimeError(f"unclassified parameter for initialization: {name}")
    return {"base_std": 0.02, "residual_output_std": residual_std}


def _rms(tensor: Any, torch: Any) -> float:
    return float(torch.sqrt(tensor.detach().float().square().mean()).item())


def _log_slope(values: list[float]) -> float:
    if len(values) < 2 or any(value <= 0 for value in values):
        return float("nan")
    xs = list(range(len(values)))
    ys = [math.log(value) for value in values]
    x_mean = statistics.mean(xs)
    y_mean = statistics.mean(ys)
    denominator = sum((value - x_mean) ** 2 for value in xs)
    return sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / denominator


def _gradient_rms(module: Any, torch: Any) -> tuple[float, int]:
    squared = torch.zeros((), dtype=torch.float64)
    elements = 0
    for parameter in module.parameters():
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach().double().cpu()
        squared += gradient.square().sum()
        elements += gradient.numel()
    return (float(torch.sqrt(squared / elements).item()), elements) if elements else (0.0, 0)


def _one_case(
    *, arm: Any, policy: str, seed: int, config: SignalConfig, torch: Any
) -> dict[str, Any]:
    import torch.nn.functional as functional

    torch.manual_seed(seed)
    if config.device == "cuda":
        torch.cuda.manual_seed_all(seed)
    device = torch.device(config.device)
    dtype = (
        torch.bfloat16
        if config.device == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float32
    )
    model = _build_model(
        torch, arm, maximum_sequence_length=config.sequence_length
    ).to(device=device, dtype=dtype)
    torch.manual_seed(seed + 1_000_003)
    initialization = _initialize(model, policy=policy, layers=arm.model.layers, torch=torch)
    token_ids = torch.randint(
        0,
        arm.model.vocabulary_size,
        (config.batch_size, config.sequence_length),
        device=device,
    )
    targets = torch.randint(
        0,
        arm.model.vocabulary_size,
        (config.batch_size, config.sequence_length),
        device=device,
    )
    embedding_rms = _rms(model.embedding(token_ids), torch)
    block_rms: list[float] = []
    final_rms: list[float] = []

    def block_hook(_module: Any, _inputs: Any, output: Any) -> None:
        block_rms.append(_rms(output, torch))

    def final_hook(_module: Any, _inputs: Any, output: Any) -> None:
        final_rms.append(_rms(output, torch))

    handles = [block.register_forward_hook(block_hook) for block in model.blocks]
    handles.append(model.final_norm.register_forward_hook(final_hook))
    model.zero_grad(set_to_none=True)
    logits = model(token_ids)
    logits_rms = _rms(logits, torch)
    loss = functional.cross_entropy(
        logits.float().view(-1, arm.model.vocabulary_size), targets.view(-1)
    )
    loss.backward()
    for handle in handles:
        handle.remove()

    block_gradient_rms = [_gradient_rms(block, torch)[0] for block in model.blocks]
    global_gradient_rms, gradient_elements = _gradient_rms(model, torch)
    finite = bool(
        torch.isfinite(loss).item()
        and all(math.isfinite(value) for value in block_rms)
        and all(math.isfinite(value) for value in block_gradient_rms)
        and all(torch.isfinite(parameter.grad).all().item() for parameter in model.parameters())
    )
    nonzero_gradients = all(value > 0 for value in block_gradient_rms) and global_gradient_rms > 0
    gradient_min = min(block_gradient_rms)
    gradient_max = max(block_gradient_rms)
    result = {
        "arm": arm.name,
        "model_sha256": arm.model.sha256(),
        "layers": arm.model.layers,
        "width": arm.model.width,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "policy": policy,
        "seed": seed,
        "dtype": str(dtype).removeprefix("torch."),
        "initialization": initialization,
        "loss": float(loss.item()),
        "embedding_rms": embedding_rms,
        "block_output_rms": block_rms,
        "final_norm_output_rms": final_rms[0],
        "logits_rms": logits_rms,
        "final_to_embedding_rms_ratio": block_rms[-1] / embedding_rms,
        "maximum_to_embedding_rms_ratio": max(block_rms) / embedding_rms,
        "activation_log_rms_slope_per_layer": _log_slope(block_rms),
        "block_gradient_rms": block_gradient_rms,
        "block_gradient_max_min_ratio": gradient_max / gradient_min,
        "first_to_last_block_gradient_rms_ratio": block_gradient_rms[0]
        / block_gradient_rms[-1],
        "global_gradient_rms": global_gradient_rms,
        "gradient_elements": gradient_elements,
        "checks": {
            "finite": finite,
            "nonzero_gradients": nonzero_gradients,
            "block_hook_count_exact": len(block_rms) == arm.model.layers,
            "parameter_count_exact": sum(parameter.numel() for parameter in model.parameters())
            == arm.model.parameter_receipt().total,
        },
    }
    del model, token_ids, targets, logits, loss
    if config.device == "cuda":
        torch.cuda.empty_cache()
    return result


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    aggregates: list[dict[str, Any]] = []
    for arm in ("deep-narrow", "middle", "wide-shallow"):
        for policy in POLICIES:
            selected = [row for row in rows if row["arm"] == arm and row["policy"] == policy]
            if not selected:
                raise ValueError("missing signal-propagation arm/policy rows")
            metrics = (
                "loss",
                "final_to_embedding_rms_ratio",
                "maximum_to_embedding_rms_ratio",
                "activation_log_rms_slope_per_layer",
                "block_gradient_max_min_ratio",
                "first_to_last_block_gradient_rms_ratio",
                "global_gradient_rms",
            )
            aggregates.append(
                {
                    "arm": arm,
                    "policy": policy,
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


def classify(aggregates: list[dict[str, Any]]) -> dict[str, Any]:
    by_key = {(row["arm"], row["policy"]): row for row in aggregates}
    shape_results: dict[str, dict[str, float | bool]] = {}
    for arm in ("deep-narrow", "middle", "wide-shallow"):
        unscaled = by_key[(arm, "normal-0.02")]
        scaled = by_key[(arm, "scaled-residual-0.02")]
        unscaled_growth = float(unscaled["final_to_embedding_rms_ratio"]["median"])
        scaled_growth = float(scaled["final_to_embedding_rms_ratio"]["median"])
        unscaled_gradient_spread = float(unscaled["block_gradient_max_min_ratio"]["median"])
        scaled_gradient_spread = float(scaled["block_gradient_max_min_ratio"]["median"])
        shape_results[arm] = {
            "scaled_final_growth_ratio_vs_unscaled": scaled_growth / unscaled_growth,
            "scaled_gradient_spread_ratio_vs_unscaled": scaled_gradient_spread
            / unscaled_gradient_spread,
            "supports_scaling": scaled_growth < unscaled_growth
            and scaled_gradient_spread <= 2 * unscaled_gradient_spread,
        }
    supported = sum(bool(row["supports_scaling"]) for row in shape_results.values())
    verdict = (
        "SUPPORTED_LOCAL_SIGNAL_PROPAGATION"
        if supported == len(shape_results)
        else "CONTRADICTED_LOCAL_SIGNAL_PROPAGATION"
        if supported == 0
        else "MIXED_LOCAL_SIGNAL_PROPAGATION"
    )
    return {"verdict": verdict, "by_shape": shape_results}


def benchmark(config: SignalConfig) -> dict[str, Any]:
    config.assert_valid()
    try:
        import torch
    except ImportError as exc:
        return {"schema": "esoes-e2-signal-propagation/v1", "status": "BLOCKED_TORCH", "reason": str(exc)}
    if config.device == "cuda" and not torch.cuda.is_available():
        return {
            "schema": "esoes-e2-signal-propagation/v1",
            "status": "BLOCKED_CUDA",
            "torch_version": torch.__version__,
        }

    cases = [(arm, policy, seed) for arm in shape_arms() for policy in POLICIES for seed in config.seeds]
    random.Random(sum(config.seeds)).shuffle(cases)
    rows = [
        _one_case(arm=arm, policy=policy, seed=seed, config=config, torch=torch)
        for arm, policy, seed in cases
    ]
    rows.sort(
        key=lambda row: (
            ("deep-narrow", "middle", "wide-shallow").index(row["arm"]),
            POLICIES.index(row["policy"]),
            row["seed"],
        )
    )
    aggregate = _aggregate(rows)
    classification = classify(aggregate)
    all_checks = all(all(row["checks"].values()) for row in rows)
    return {
        "schema": "esoes-e2-signal-propagation/v1",
        "status": "PASS" if all_checks else "FAIL",
        "scope": "initialization signal propagation; one forward/backward; no optimizer update",
        "implementation_sha256": _sha256_file(Path(__file__)),
        "model_implementation_sha256": _sha256_file(Path(__file__).with_name("block_benchmark.py")),
        "config": {
            "device": config.device,
            "sequence_length": config.sequence_length,
            "batch_size": config.batch_size,
            "seeds": list(config.seeds),
        },
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(0) if config.device == "cuda" else platform.processor(),
        "policies": {
            "normal-0.02": "all matrix/embedding weights normal(0,0.02)",
            "scaled-residual-0.02": (
                "same draws, but attention-output and FFN-down std = 0.02/sqrt(2L)"
            ),
        },
        "rows": rows,
        "aggregate": aggregate,
        "classification": classification,
        "limitations": [
            "Random-token initialization statistics do not measure learning or cognition.",
            "One backward pass cannot choose optimizer, LR, batch size, or schedule.",
            "The local CPU/CUDA direction must be checked on the target TPU/XLA constructor.",
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
        SignalConfig(
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
