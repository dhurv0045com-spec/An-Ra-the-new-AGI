"""Probe what QK normalization controls before any model training.

The paired experiment holds hidden states, projection draws, values, targets,
RoPE, and query positions fixed while multiplying Q/K projection weights by
0.25, 1, or 4. It measures attention-logit scale, entropy/concentration, and
proxy backward finiteness with and without per-head RMS QK normalization. No
optimizer update is performed.
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


POLICIES = ("qk-norm", "no-qk-norm")
PROJECTION_SCALES = (0.25, 1.0, 4.0)
QUERY_FRACTIONS = (0.25, 0.5, 0.75, 1.0)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class QKNormConfig:
    device: str
    context_lengths: tuple[int, ...]
    seeds: tuple[int, ...]
    width: int = 384
    query_heads: int = 6
    head_dimension: int = 64

    def assert_valid(self) -> None:
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("device must be cpu or cuda")
        if len(self.seeds) < 3 or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("QK-norm evidence requires at least three distinct seeds")
        if not self.context_lengths or len(set(self.context_lengths)) != len(
            self.context_lengths
        ):
            raise ValueError("context lengths must be nonempty and distinct")
        if any(length < 32 for length in self.context_lengths):
            raise ValueError("context lengths must be at least 32")
        if self.width != self.query_heads * self.head_dimension:
            raise ValueError("width must equal query_heads * head_dimension")
        if any(seed < 0 for seed in self.seeds):
            raise ValueError("seeds must be nonnegative")


def _rms_norm(tensor: Any, torch: Any, *, epsilon: float = 1e-6) -> Any:
    inverse = torch.rsqrt(tensor.float().square().mean(dim=-1, keepdim=True) + epsilon)
    return tensor * inverse.to(dtype=tensor.dtype)


def _rope(tensor: Any, torch: Any, *, positions: Any, base: float = 10_000.0) -> Any:
    dimension = tensor.shape[-1]
    inverse = 1.0 / (
        base
        ** (torch.arange(0, dimension, 2, device=tensor.device, dtype=torch.float32) / dimension)
    )
    angles = positions.float()[:, None] * inverse[None, :]
    cosine = angles.cos().to(dtype=tensor.dtype)[None, :, :]
    sine = angles.sin().to(dtype=tensor.dtype)[None, :, :]
    even, odd = tensor[..., 0::2], tensor[..., 1::2]
    return torch.stack((even * cosine - odd * sine, even * sine + odd * cosine), dim=-1).flatten(-2)


def _rms(tensor: Any, torch: Any) -> float:
    return float(torch.sqrt(tensor.detach().float().square().mean()).item())


def _query_indices(length: int) -> tuple[int, ...]:
    return tuple(sorted({max(1, round((length - 1) * fraction)) for fraction in QUERY_FRACTIONS}))


def _one_case(
    *,
    config: QKNormConfig,
    context_length: int,
    seed: int,
    policy: str,
    projection_scale: float,
    torch: Any,
) -> dict[str, Any]:
    import torch.nn.functional as functional

    if policy not in POLICIES or projection_scale not in PROJECTION_SCALES:
        raise ValueError("unregistered QK-norm case")
    device = torch.device(config.device)
    dtype = (
        torch.bfloat16
        if config.device == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float32
    )
    generator = torch.Generator(device=config.device).manual_seed(seed)
    hidden = torch.randn(
        (context_length, config.width), generator=generator, device=device, dtype=dtype
    )
    hidden = _rms_norm(hidden, torch, epsilon=1e-5)
    base_query_weight = torch.randn(
        (config.width, config.width), generator=generator, device=device, dtype=dtype
    ) * 0.02
    base_key_weight = torch.randn(
        (config.width, config.width), generator=generator, device=device, dtype=dtype
    ) * 0.02
    values = torch.randn(
        (config.query_heads, context_length, config.head_dimension),
        generator=generator,
        device=device,
        dtype=dtype,
    )
    query_indices = _query_indices(context_length)
    targets = torch.randn(
        (config.query_heads, len(query_indices), config.head_dimension),
        generator=generator,
        device=device,
        dtype=torch.float32,
    )
    query_weight = (base_query_weight * projection_scale).detach().requires_grad_(True)
    key_weight = (base_key_weight * projection_scale).detach().requires_grad_(True)
    query = functional.linear(hidden, query_weight).view(
        context_length, config.query_heads, config.head_dimension
    ).transpose(0, 1)
    key = functional.linear(hidden, key_weight).view(
        context_length, config.query_heads, config.head_dimension
    ).transpose(0, 1)
    pre_norm_query_rms = _rms(query, torch)
    pre_norm_key_rms = _rms(key, torch)
    if policy == "qk-norm":
        query = _rms_norm(query, torch)
        key = _rms_norm(key, torch)
    positions = torch.arange(context_length, device=device)
    query = _rope(query, torch, positions=positions)
    key = _rope(key, torch, positions=positions)

    logits_chunks: list[Any] = []
    probability_chunks: list[Any] = []
    entropy: list[float] = []
    normalized_entropy: list[float] = []
    effective_fraction: list[float] = []
    maximum_probability: list[float] = []
    top_ten_mass: list[float] = []
    probability_sum_error: list[float] = []
    outputs: list[Any] = []
    for query_index in query_indices:
        logits = torch.einsum(
            "hd,hkd->hk", query[:, query_index, :], key[:, : query_index + 1, :]
        ) / math.sqrt(config.head_dimension)
        probabilities = torch.softmax(logits.float(), dim=-1)
        log_probabilities = torch.log(probabilities.clamp_min(torch.finfo(torch.float32).tiny))
        head_entropy = -(probabilities * log_probabilities).sum(dim=-1)
        key_count = query_index + 1
        logits_chunks.append(logits.float().reshape(-1))
        probability_chunks.append(probabilities.reshape(-1))
        entropy.extend(float(value) for value in head_entropy.detach().cpu())
        normalized_entropy.extend(
            float(value / math.log(key_count)) for value in head_entropy.detach().cpu()
        )
        effective_fraction.extend(
            float(value.exp() / key_count) for value in head_entropy.detach().cpu()
        )
        maximum_probability.extend(
            float(value) for value in probabilities.max(dim=-1).values.detach().cpu()
        )
        top_ten_mass.extend(
            float(value)
            for value in probabilities.topk(min(10, key_count), dim=-1).values.sum(dim=-1).detach().cpu()
        )
        probability_sum_error.extend(
            float(value)
            for value in (probabilities.sum(dim=-1) - 1).abs().detach().cpu()
        )
        outputs.append(torch.einsum("hk,hkd->hd", probabilities, values[:, :key_count, :].float()))

    output = torch.stack(outputs, dim=1)
    proxy_loss = functional.mse_loss(output, targets)
    proxy_loss.backward()
    all_logits = torch.cat(logits_chunks)
    all_probabilities = torch.cat(probability_chunks)
    finite = bool(
        torch.isfinite(all_logits).all().item()
        and torch.isfinite(all_probabilities).all().item()
        and torch.isfinite(proxy_loss).item()
        and query_weight.grad is not None
        and key_weight.grad is not None
        and torch.isfinite(query_weight.grad).all().item()
        and torch.isfinite(key_weight.grad).all().item()
    )
    return {
        "context_length": context_length,
        "seed": seed,
        "policy": policy,
        "projection_scale": projection_scale,
        "dtype": str(dtype).removeprefix("torch."),
        "query_indices": list(query_indices),
        "pre_norm_query_rms": pre_norm_query_rms,
        "pre_norm_key_rms": pre_norm_key_rms,
        "post_policy_query_rms": _rms(query, torch),
        "post_policy_key_rms": _rms(key, torch),
        "attention_logit_rms": _rms(all_logits, torch),
        "attention_logit_max_abs": float(all_logits.detach().abs().max().item()),
        "normalized_entropy_mean": statistics.mean(normalized_entropy),
        "effective_attended_fraction_mean": statistics.mean(effective_fraction),
        "maximum_probability_mean": statistics.mean(maximum_probability),
        "top_ten_probability_mass_mean": statistics.mean(top_ten_mass),
        "probability_sum_max_error": max(probability_sum_error),
        "proxy_loss": float(proxy_loss.item()),
        "query_weight_gradient_rms": _rms(query_weight.grad, torch),
        "key_weight_gradient_rms": _rms(key_weight.grad, torch),
        "checks": {
            "finite": finite,
            "nonzero_gradients": bool(
                _rms(query_weight.grad, torch) > 0 and _rms(key_weight.grad, torch) > 0
            ),
            "probabilities_sum_to_one": max(probability_sum_error) <= 2e-6,
            "causal_query_count_exact": len(query_indices) == len(QUERY_FRACTIONS),
        },
    }


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = (
        "pre_norm_query_rms",
        "pre_norm_key_rms",
        "post_policy_query_rms",
        "post_policy_key_rms",
        "attention_logit_rms",
        "attention_logit_max_abs",
        "normalized_entropy_mean",
        "effective_attended_fraction_mean",
        "maximum_probability_mean",
        "top_ten_probability_mass_mean",
        "proxy_loss",
        "query_weight_gradient_rms",
        "key_weight_gradient_rms",
    )
    aggregates: list[dict[str, Any]] = []
    contexts = sorted({int(row["context_length"]) for row in rows})
    for context_length in contexts:
        for policy in POLICIES:
            for projection_scale in PROJECTION_SCALES:
                selected = [
                    row
                    for row in rows
                    if row["context_length"] == context_length
                    and row["policy"] == policy
                    and row["projection_scale"] == projection_scale
                ]
                if not selected:
                    raise ValueError("missing QK-norm aggregate case")
                aggregates.append(
                    {
                        "context_length": context_length,
                        "policy": policy,
                        "projection_scale": projection_scale,
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
    by_key = {
        (row["context_length"], row["policy"], row["projection_scale"]): row
        for row in aggregates
    }
    contexts = sorted({int(row["context_length"]) for row in aggregates})
    results: dict[str, Any] = {}
    for context_length in contexts:
        normalized = [
            by_key[(context_length, "qk-norm", scale)] for scale in PROJECTION_SCALES
        ]
        unnormalized = [
            by_key[(context_length, "no-qk-norm", scale)] for scale in PROJECTION_SCALES
        ]
        normalized_logit_rms = [float(row["attention_logit_rms"]["median"]) for row in normalized]
        unnormalized_logit_rms = [
            float(row["attention_logit_rms"]["median"]) for row in unnormalized
        ]
        normalized_entropy = [
            float(row["normalized_entropy_mean"]["median"]) for row in normalized
        ]
        unnormalized_entropy = [
            float(row["normalized_entropy_mean"]["median"]) for row in unnormalized
        ]
        normalized_logit_ratio = max(normalized_logit_rms) / min(normalized_logit_rms)
        unnormalized_logit_ratio = max(unnormalized_logit_rms) / min(unnormalized_logit_rms)
        normalized_entropy_span = max(normalized_entropy) - min(normalized_entropy)
        unnormalized_entropy_span = max(unnormalized_entropy) - min(unnormalized_entropy)
        supports = (
            normalized_logit_ratio <= 1.05
            and normalized_entropy_span <= 0.02
            and unnormalized_logit_ratio >= 100
            and unnormalized_entropy_span >= 0.10
        )
        results[str(context_length)] = {
            "qk_norm_logit_rms_max_min_ratio": normalized_logit_ratio,
            "no_qk_norm_logit_rms_max_min_ratio": unnormalized_logit_ratio,
            "qk_norm_normalized_entropy_span": normalized_entropy_span,
            "no_qk_norm_normalized_entropy_span": unnormalized_entropy_span,
            "supports_scale_control": supports,
        }
    supported = sum(bool(row["supports_scale_control"]) for row in results.values())
    verdict = (
        "SUPPORTED_QK_SCALE_CONTROL"
        if supported == len(results)
        else "CONTRADICTED_QK_SCALE_CONTROL"
        if supported == 0
        else "MIXED_QK_SCALE_CONTROL"
    )
    return {"verdict": verdict, "by_context": results}


def benchmark(config: QKNormConfig) -> dict[str, Any]:
    config.assert_valid()
    try:
        import torch
    except ImportError as exc:
        return {"schema": "esoes-e2-qk-norm-probe/v1", "status": "BLOCKED_TORCH", "reason": str(exc)}
    if config.device == "cuda" and not torch.cuda.is_available():
        return {
            "schema": "esoes-e2-qk-norm-probe/v1",
            "status": "BLOCKED_CUDA",
            "torch_version": torch.__version__,
        }
    cases = [
        (context_length, seed, policy, scale)
        for context_length in config.context_lengths
        for seed in config.seeds
        for policy in POLICIES
        for scale in PROJECTION_SCALES
    ]
    random.Random(sum(config.seeds) + sum(config.context_lengths)).shuffle(cases)
    rows = [
        _one_case(
            config=config,
            context_length=context_length,
            seed=seed,
            policy=policy,
            projection_scale=scale,
            torch=torch,
        )
        for context_length, seed, policy, scale in cases
    ]
    rows.sort(
        key=lambda row: (
            row["context_length"],
            POLICIES.index(row["policy"]),
            PROJECTION_SCALES.index(row["projection_scale"]),
            row["seed"],
        )
    )
    aggregates = _aggregate(rows)
    classification = classify(aggregates)
    all_checks = all(all(row["checks"].values()) for row in rows)
    return {
        "schema": "esoes-e2-qk-norm-probe/v1",
        "status": "PASS" if all_checks else "FAIL",
        "scope": "paired isolated MHA QK scale/entropy/backward probe; no optimizer update",
        "implementation_sha256": _sha256_file(Path(__file__)),
        "config": {
            "device": config.device,
            "context_lengths": list(config.context_lengths),
            "seeds": list(config.seeds),
            "width": config.width,
            "query_heads": config.query_heads,
            "head_dimension": config.head_dimension,
            "projection_scales": list(PROJECTION_SCALES),
            "query_fractions": list(QUERY_FRACTIONS),
        },
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(0) if config.device == "cuda" else platform.processor(),
        "rows": rows,
        "aggregate": aggregates,
        "classification": classification,
        "limitations": [
            "Random hidden states and a proxy backward do not measure learned attention or cognition.",
            "The probe isolates MHA Q/K scale; it does not decide GQA, affine-scale learning, or topology.",
            "Parameter-gradient magnitude is not scale-invariant even when normalized attention is.",
            "The local result must be reproduced on the target TPU/XLA constructor.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--context-length", action="append", type=int, required=True)
    parser.add_argument("--seed", action="append", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = benchmark(
        QKNormConfig(
            device=args.device,
            context_lengths=tuple(args.context_length),
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
