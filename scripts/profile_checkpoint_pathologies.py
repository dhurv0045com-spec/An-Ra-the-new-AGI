# ruff: noqa: E402
"""Profile legacy checkpoint weights for architecture-specific pathologies."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if __name__ == "__main__" and not __package__:
    completed = subprocess.run(
        [sys.executable, "-m", "scripts.profile_checkpoint_pathologies", *sys.argv[1:]],
        cwd=REPO_ROOT,
        check=False,
    )
    raise SystemExit(completed.returncode)

from anra.anra_paths import OUTPUT_V2_DIR, ROOT
from runtime.experience_ledger import content_hash
from runtime.safe_load import safe_torch_load

from scripts.freeze_baseline_hashes import resolve_checkpoint

DEFAULT_REPORT = OUTPUT_V2_DIR / "checkpoint_pathology_profile.json"
DEFAULT_ACTIVATION_PROMPTS = (
    "H: Explain why the sky appears blue in two sentences.\nANRA:",
    "H: Write a Python function that adds two integers.\nANRA:",
    "H: What is 17 multiplied by 23? Show the result.\nANRA:",
    "H: Compare strong consistency with eventual consistency.\nANRA:",
    "H: Follow this instruction exactly: reply with the word amber.\nANRA:",
    "H: Use only this context: the key is cobalt-19. Return the key.\nANRA:",
    "H: Who are you, and what are you designed to do?\nANRA:",
    "H: If all ravens are birds and some birds migrate, what follows?\nANRA:",
)
_ROUTER_KEY = re.compile(r"(?:^|\.)mod_routers\.(\d+)\.(.+)$")
_RIM_ALPHA_KEY = re.compile(r"(?:^|\.)rim_modules\.(\d+)\.raw_alpha$")


def _model_state(blob: object) -> Mapping[str, torch.Tensor]:
    if not isinstance(blob, Mapping):
        raise TypeError("checkpoint payload must be a mapping")
    candidate = blob.get("model", blob.get("model_state_dict", blob.get("model_state")))
    if not isinstance(candidate, Mapping):
        raise KeyError("checkpoint has no model state mapping")
    state = {str(key): value for key, value in candidate.items() if torch.is_tensor(value)}
    if not state:
        raise ValueError("checkpoint model state contains no tensors")
    return state


def _values(tensor: torch.Tensor) -> list[float]:
    return [float(value) for value in tensor.detach().float().reshape(-1).cpu().tolist()]


def _tensor_stats(tensor: torch.Tensor) -> dict[str, object]:
    values = tensor.detach().float().cpu()
    finite = torch.isfinite(values)
    finite_values = values[finite]
    result: dict[str, object] = {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "numel": tensor.numel(),
        "nonfinite": int((~finite).sum().item()),
    }
    if finite_values.numel():
        result.update(
            {
                "min": float(finite_values.min().item()),
                "max": float(finite_values.max().item()),
                "mean": float(finite_values.mean().item()),
                "std": float(finite_values.std(unbiased=False).item()),
                "l2": float(torch.linalg.vector_norm(finite_values).item()),
            }
        )
    return result


class _ActivationAccumulator:
    def __init__(self) -> None:
        self.rows: dict[str, dict[str, float]] = defaultdict(
            lambda: {
                "calls": 0.0,
                "elements": 0.0,
                "nonfinite": 0.0,
                "rms_sum": 0.0,
                "mean_l2_sum": 0.0,
                "max_abs": 0.0,
            }
        )

    def add(self, name: str, tensor: torch.Tensor) -> None:
        detached = tensor.detach().float()
        finite = torch.isfinite(detached)
        safe = torch.where(finite, detached, torch.zeros_like(detached))
        row = self.rows[name]
        row["calls"] += 1
        row["elements"] += detached.numel()
        row["nonfinite"] += int((~finite).sum().item())
        row["rms_sum"] += float(safe.square().mean().sqrt().item())
        row["mean_l2_sum"] += float(safe.norm(dim=-1).mean().item())
        row["max_abs"] = max(row["max_abs"], float(safe.abs().max().item()))

    def report(self) -> dict[str, dict[str, float | int]]:
        output: dict[str, dict[str, float | int]] = {}
        for name, row in sorted(self.rows.items()):
            calls = max(1.0, row["calls"])
            output[name] = {
                "calls": int(row["calls"]),
                "elements": int(row["elements"]),
                "nonfinite": int(row["nonfinite"]),
                "mean_rms": row["rms_sum"] / calls,
                "mean_token_l2": row["mean_l2_sum"] / calls,
                "max_abs": row["max_abs"],
            }
        return output


def _router_scores(router: torch.nn.Module, x: torch.Tensor, ctx: object) -> torch.Tensor:
    scores = router.gate(x).squeeze(-1)
    if ctx is None:
        return scores
    context_values = []
    for value in (ctx.esv_arousal, ctx.token_entropy, ctx.civ_similarity):
        tensor = torch.as_tensor(value, device=x.device, dtype=x.dtype)
        while tensor.ndim < scores.ndim:
            tensor = tensor.unsqueeze(-1)
        context_values.append(tensor.expand_as(scores))
    context_stack = torch.stack(context_values, dim=-1)
    return scores + torch.sum(
        context_stack * router.context_weights.to(dtype=x.dtype), dim=-1
    )


def _normalized_entropy(histogram: torch.Tensor) -> float:
    probabilities = histogram.float() / histogram.sum().clamp_min(1.0)
    nonzero = probabilities[probabilities > 0]
    entropy = -(nonzero * nonzero.log()).sum()
    return float((entropy / math.log(max(2, histogram.numel()))).item())


def profile_checkpoint_activations(
    path: Path,
    *,
    device: str = "cuda",
    prompts: Sequence[str] = DEFAULT_ACTIVATION_PROMPTS,
) -> dict[str, object]:
    """Measure architecture-specific behavior on the real forward path."""
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA activation profiling was requested but CUDA is unavailable")
    if not prompts:
        raise ValueError("activation profiling requires at least one prompt")

    from tokenizer.subword_tokenizer import SubwordTokenizer
    from training.v2_runtime import (
        active_tokenizer_path,
        build_legacy_500m_model,
        load_checkpoint,
    )

    torch_device = torch.device(device)
    tokenizer = SubwordTokenizer.load(active_tokenizer_path())
    model = build_legacy_500m_model(vocab_size=tokenizer.vocab_size).to(torch_device)
    load_state = load_checkpoint(
        model,
        None,
        None,
        None,
        path,
        device=torch_device,
        strict=False,
    )
    if not load_state.get("loaded"):
        raise RuntimeError(f"checkpoint did not load for activation profiling: {path}")
    model.eval()

    modes: dict[str, object] = {}
    for mode in ("diagnostic", "native"):
        accumulator = _ActivationAccumulator()
        handles = []
        router_histograms = {
            key: torch.zeros(16, dtype=torch.int64, device=torch_device)
            for key in model.mod_routers
        }
        router_gate_sums = dict.fromkeys(model.mod_routers, 0.0)
        router_gate_calls = dict.fromkeys(model.mod_routers, 0)

        for index, block in enumerate(model.blocks):
            handles.append(
                block.register_forward_hook(
                    lambda _module, _inputs, output, i=index, acc=accumulator: acc.add(
                        f"block.{i}.residual", output
                    )
                )
            )
            handles.append(
                block.attn.register_forward_hook(
                    lambda _module, _inputs, output, i=index, acc=accumulator: acc.add(
                        f"block.{i}.attention", output
                    )
                )
            )
            handles.append(
                block.mlp.register_forward_hook(
                    lambda _module, _inputs, output, i=index, acc=accumulator: acc.add(
                        f"block.{i}.mlp", output
                    )
                )
            )

        for key, router in model.mod_routers.items():
            def router_hook(
                module: torch.nn.Module,
                inputs: tuple[object, ...],
                _output: torch.Tensor,
                *,
                layer_key: str = key,
                histograms: dict[str, torch.Tensor] = router_histograms,
                gate_sums: dict[str, float] = router_gate_sums,
                gate_calls: dict[str, int] = router_gate_calls,
            ) -> None:
                x = inputs[0]
                ctx = inputs[2] if len(inputs) > 2 else None
                assert isinstance(x, torch.Tensor)
                scores = _router_scores(module, x, ctx)
                capacity = max(0.05, min(1.0, float(module.capacity)))
                k = max(1, min(x.shape[1], int(x.shape[1] * capacity)))
                selected = scores.topk(k, dim=-1).indices
                bins = (selected * 16 // max(1, x.shape[1])).clamp_max(15)
                histograms[layer_key].add_(
                    torch.bincount(bins.reshape(-1), minlength=16)
                )
                probabilities = torch.sigmoid(scores + module.capacity_control)
                gate_sums[layer_key] += float(probabilities.mean().item())
                gate_calls[layer_key] += 1

            handles.append(router.register_forward_hook(router_hook))

        mode_state = model.configure_runtime_mode(mode)
        output_entropies: list[float] = []
        top1_probabilities: list[float] = []
        try:
            with torch.inference_mode():
                for prompt in prompts:
                    token_ids = tokenizer.encode(prompt, add_special_tokens=False)
                    token_ids = [tokenizer.bos_token_id, *token_ids][-model.block_size :]
                    inputs = torch.tensor([token_ids], dtype=torch.long, device=torch_device)
                    logits, _ = model(inputs)
                    probabilities = torch.softmax(logits[:, -1].float(), dim=-1)
                    output_entropies.append(
                        float(
                            -(
                                probabilities * probabilities.clamp_min(1e-12).log()
                            ).sum().item()
                        )
                    )
                    top1_probabilities.append(float(probabilities.max().item()))
        finally:
            model.restore_runtime_mode(mode_state)
            for handle in handles:
                handle.remove()

        activation_rows = accumulator.report()
        router_report = {
            key: {
                "selection_position_entropy": _normalized_entropy(histogram),
                "position_histogram": [int(value) for value in histogram.cpu().tolist()],
                "mean_gate_probability": router_gate_sums[key]
                / max(1, router_gate_calls[key]),
                "calls": router_gate_calls[key],
            }
            for key, histogram in router_histograms.items()
            if router_gate_calls[key]
        }
        modes[mode] = {
            "prompt_count": len(prompts),
            "activations": activation_rows,
            "nonfinite_activations": sum(
                int(row["nonfinite"]) for row in activation_rows.values()
            ),
            "mean_output_entropy": sum(output_entropies) / len(output_entropies),
            "mean_top1_probability": sum(top1_probabilities) / len(top1_probabilities),
            "routers": router_report,
        }

    native = modes["native"]
    diagnostic = modes["diagnostic"]
    assert isinstance(native, dict)
    assert isinstance(diagnostic, dict)
    alerts: list[dict[str, object]] = []
    if int(native["nonfinite_activations"]):
        alerts.append({"severity": "critical", "code": "native_nonfinite_activations"})
    for mode_name, mode_report in modes.items():
        if float(mode_report["mean_top1_probability"]) > 0.95:
            alerts.append(
                {
                    "severity": "critical",
                    "code": "output_distribution_collapse",
                    "mode": mode_name,
                    "mean_top1_probability": mode_report["mean_top1_probability"],
                    "mean_output_entropy": mode_report["mean_output_entropy"],
                }
            )
        residuals = [
            row
            for key, row in mode_report["activations"].items()
            if key.endswith(".residual")
        ]
        if residuals:
            rms_values = [float(row["mean_rms"]) for row in residuals]
            if max(rms_values) / max(1e-8, min(rms_values)) > 8.0:
                alerts.append(
                    {
                        "severity": "high",
                        "code": "residual_amplification",
                        "mode": mode_name,
                        "min_rms": min(rms_values),
                        "max_rms": max(rms_values),
                    }
                )
    for key, router in native["routers"].items():
        if float(router["selection_position_entropy"]) < 0.35:
            alerts.append(
                {
                    "severity": "high",
                    "code": "router_position_collapse",
                    "layer": key,
                    "entropy": router["selection_position_entropy"],
                }
            )
        gate_probability = float(router["mean_gate_probability"])
        if gate_probability < 0.05 or gate_probability > 0.95:
            alerts.append(
                {
                    "severity": "high",
                    "code": "router_gate_strength_collapse",
                    "layer": key,
                    "mean_gate_probability": gate_probability,
                }
            )
    report: dict[str, object] = {
        "schema_version": 1,
        "device": str(torch_device),
        "device_name": (
            torch.cuda.get_device_name(torch_device) if torch_device.type == "cuda" else "CPU"
        ),
        "checkpoint_load": load_state.get("load_report", {}),
        "prompt_sha256": hashlib.sha256(
            json.dumps(list(prompts), ensure_ascii=False).encode("utf-8")
        ).hexdigest(),
        "modes": modes,
        "native_minus_diagnostic_output_entropy": float(native["mean_output_entropy"])
        - float(diagnostic["mean_output_entropy"]),
        "alerts": alerts,
    }
    report["report_hash"] = content_hash(report)
    del model
    if torch_device.type == "cuda":
        torch.cuda.empty_cache()
    return report


def profile_checkpoint(path: Path) -> dict[str, object]:
    blob = safe_torch_load(path, map_location="cpu")
    state = _model_state(blob)
    total_elements = 0
    tensor_entry_bytes = 0
    nonfinite_elements = 0
    unique_storages: dict[tuple[int, int], int] = {}
    for tensor in state.values():
        total_elements += tensor.numel()
        tensor_entry_bytes += tensor.numel() * tensor.element_size()
        nonfinite_elements += int((~torch.isfinite(tensor)).sum().item())
        storage = tensor.untyped_storage()
        unique_storages[(storage.data_ptr(), storage.nbytes())] = storage.nbytes()

    selected: dict[str, dict[str, object]] = {}
    for name in (
        "residual_depth_logits",
        "dstp_temperature_log",
        "layer_temperature_bias",
        "layer_temperature_bias_log",
        "esv_module.state",
        "token_embedding_table.weight",
    ):
        tensor = state.get(name)
        if tensor is not None:
            selected[name] = _tensor_stats(tensor)

    residual_logits = state.get("residual_depth_logits")
    dstp_log = state.get("dstp_temperature_log")
    residual_scales = (
        _values(2.0 * torch.sigmoid(residual_logits)) if residual_logits is not None else []
    )
    dstp_temperatures = _values(dstp_log.exp()) if dstp_log is not None else []
    embedding = state.get("token_embedding_table.weight")
    embedding_rows: dict[str, object] = {}
    if embedding is not None and embedding.ndim == 2:
        row_norms = embedding.detach().float().norm(dim=1).cpu()
        quantiles = torch.quantile(
            row_norms, torch.tensor([0.0, 0.5, 0.9, 0.99, 1.0])
        ).tolist()
        top_count = min(20, row_norms.numel())
        top_values, top_indices = row_norms.topk(top_count)
        embedding_rows = {
            "norm_quantiles": {
                key: float(value)
                for key, value in zip(
                    ("min", "p50", "p90", "p99", "max"), quantiles, strict=True
                )
            },
            "top_norm_rows": [
                {"token_id": int(index), "l2": float(value)}
                for value, index in zip(
                    top_values.tolist(), top_indices.tolist(), strict=True
                )
            ],
        }

    routers: dict[str, dict[str, object]] = {}
    for name, tensor in state.items():
        match = _ROUTER_KEY.search(name)
        if match is None:
            continue
        layer, field = match.groups()
        row = routers.setdefault(layer, {})
        if field in {"capacity_control", "context_weights", "gate.weight"}:
            row[field] = _tensor_stats(tensor)
            if tensor.numel() <= 8:
                row[f"{field}_values"] = _values(tensor)

    rim_strengths: dict[str, float] = {}
    for name, tensor in state.items():
        match = _RIM_ALPHA_KEY.search(name)
        if match is not None:
            rim_strengths[match.group(1)] = float(0.25 * torch.tanh(tensor).item())

    context_rows = [
        state[name]
        for name in state
        if _ROUTER_KEY.search(name) is not None and name.endswith("context_weights")
    ]
    context_weights_all_zero = bool(context_rows) and all(
        torch.count_nonzero(tensor).item() == 0 for tensor in context_rows
    )
    alerts: list[dict[str, object]] = []
    if nonfinite_elements:
        alerts.append(
            {
                "severity": "critical",
                "code": "nonfinite_weights",
                "count": nonfinite_elements,
            }
        )
    if context_weights_all_zero:
        alerts.append(
            {
                "severity": "high",
                "code": "router_context_dormant",
                "detail": "all saved router context weights are exactly zero",
            }
        )
    if residual_scales and not all(0.5 <= value <= 1.5 for value in residual_scales):
        alerts.append(
            {
                "severity": "high",
                "code": "residual_scale_extreme",
                "range": [min(residual_scales), max(residual_scales)],
            }
        )
    if dstp_temperatures and not all(0.5 <= value <= 2.0 for value in dstp_temperatures):
        alerts.append(
            {
                "severity": "high",
                "code": "dstp_temperature_extreme",
                "range": [min(dstp_temperatures), max(dstp_temperatures)],
            }
        )
    if any(not math.isfinite(value) or abs(value) > 0.24 for value in rim_strengths.values()):
        alerts.append(
            {
                "severity": "high",
                "code": "rim_strength_saturated",
            }
        )

    checkpoint_metadata = blob if isinstance(blob, Mapping) else {}
    report: dict[str, object] = {
        "schema_version": 1,
        "checkpoint": str(path),
        "checkpoint_schema_version": checkpoint_metadata.get("checkpoint_schema_version"),
        "global_step": checkpoint_metadata.get("global_step", checkpoint_metadata.get("step")),
        "best_loss": checkpoint_metadata.get("best_loss"),
        "tensor_count": len(state),
        "total_elements": total_elements,
        "tensor_entry_bytes": tensor_entry_bytes,
        "unique_storage_bytes": sum(unique_storages.values()),
        "nonfinite_elements": nonfinite_elements,
        "selected_tensors": selected,
        "residual_scales": residual_scales,
        "dstp_temperatures": dstp_temperatures,
        "embedding_rows": embedding_rows,
        "routers": routers,
        "router_context_weights_all_zero": context_weights_all_zero,
        "rim_strengths": rim_strengths,
        "alerts": alerts,
        "passed_numerical_integrity": nonfinite_elements == 0,
    }
    report["report_hash"] = content_hash(report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--json-out", default=str(DEFAULT_REPORT))
    parser.add_argument("--run-activations", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-prompts", type=int, default=len(DEFAULT_ACTIVATION_PROMPTS))
    args = parser.parse_args()
    report = profile_checkpoint(resolve_checkpoint(args.checkpoint))
    if args.run_activations:
        report["activation_profile"] = profile_checkpoint_activations(
            resolve_checkpoint(args.checkpoint),
            device=args.device,
            prompts=DEFAULT_ACTIVATION_PROMPTS[: max(1, args.max_prompts)],
        )
        unsigned = {key: value for key, value in report.items() if key != "report_hash"}
        report["report_hash"] = content_hash(unsigned)
    output = Path(args.json_out)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(output)
    print(
        json.dumps(
            {
                "checkpoint": report["checkpoint"],
                "tensor_count": report["tensor_count"],
                "total_elements": report["total_elements"],
                "nonfinite_elements": report["nonfinite_elements"],
                "alerts": report["alerts"],
                "report_hash": report["report_hash"],
            },
            indent=2,
        )
    )
    return 0 if report["passed_numerical_integrity"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
