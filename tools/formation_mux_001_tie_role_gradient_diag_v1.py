"""Development-only final-checkpoint gradient-role decomposition.

No sealed rows are loaded. For the tied matrix W we compute the exact input-path
and output-path gradient components by preserving forward values while setting
one autograd scale to zero at a time.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

from anra_v5 import tie_role_model_v1 as fxm
from anra_v5 import tie_role_train_v1 as train
from v5_experiments import tie_role_protocol_v1 as proto
from v5_experiments.formation_mux_surface_v5 import load_public_surface
from v5_model.core import packed_layout
from v5_objectives.causal_lm import causal_lm_loss


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _gradient(model: Any, tokens: Any, segments: Any, eligible: Any, *, input_scale: float, output_scale: float, torch: Any) -> tuple[Any, float]:
    model.zero_grad(set_to_none=True)
    fxm.set_gradient_scales(model, input_scale=input_scale, output_scale=output_scale)
    positions, mask = packed_layout(segments, torch_module=torch)
    logits = model(tokens, positions, mask)
    loss, _ = causal_lm_loss(
        logits,
        tokens,
        segments,
        bos_id=2,
        pad_id=0,
        eligible=eligible,
        torch_module=torch,
    )
    loss.backward()
    grad = model.embedding.weight.grad
    if grad is None:
        raise RuntimeError("tie-role gradient diagnostic found no embedding gradient")
    return grad.detach().float().clone(), float(loss.detach().item())


def decompose_model(model: Any, rows: list[dict[str, Any]], experiment: str, arm: str, *, torch: Any, device: Any) -> dict[str, Any]:
    tokens, segments, eligible, supervised, processed = train.build_batch(
        rows, experiment, arm, torch=torch, device=device
    )
    original = fxm.get_gradient_scales(model)
    g_in, loss_in = _gradient(
        model, tokens, segments, eligible,
        input_scale=1.0, output_scale=0.0, torch=torch,
    )
    g_out, loss_out = _gradient(
        model, tokens, segments, eligible,
        input_scale=0.0, output_scale=1.0, torch=torch,
    )
    fxm.set_gradient_scales(model, input_scale=original[0], output_scale=original[1])
    model.zero_grad(set_to_none=True)

    n_in = float(torch.linalg.vector_norm(g_in).item())
    n_out = float(torch.linalg.vector_norm(g_out).item())
    combined = g_in + g_out
    n_combined = float(torch.linalg.vector_norm(combined).item())
    dot = float((g_in * g_out).sum().item())
    cosine = dot / max(n_in * n_out, 1e-30)
    cancellation = 1.0 - n_combined / max(n_in + n_out, 1e-30)
    return {
        "rows": len(rows),
        "supervised_tokens": int(supervised),
        "processed_tokens": int(processed),
        "loss_input_component_forward": loss_in,
        "loss_output_component_forward": loss_out,
        "input_gradient_norm": n_in,
        "output_gradient_norm": n_out,
        "output_to_input_gradient_norm_ratio": n_out / max(n_in, 1e-30),
        "input_output_gradient_cosine": cosine,
        "combined_gradient_norm": n_combined,
        "gradient_cancellation_fraction": cancellation,
    }


def _mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def run_diagnostic(*, public_path: Path, out: Path, torch: Any, device: Any) -> dict[str, Any]:
    public = load_public_surface(public_path)
    if "sealed" in public.get("splits", {}):
        raise RuntimeError("SEALED_FIREWALL_BREACH: gradient diagnostic received sealed rows")
    identity = [r for r in public["splits"]["development"] if r["family"] == "identity"]
    fixed_rows = identity[: proto.BATCH_ROWS]
    if len(fixed_rows) != proto.BATCH_ROWS:
        raise RuntimeError("tie-role gradient diagnostic development batch incomplete")

    per_arm: dict[str, dict[str, Any]] = {}
    for experiment, arms in ((proto.EXPERIMENT_A, proto.ARMS_A), (proto.EXPERIMENT_B, proto.ARMS_B)):
        for arm in arms:
            key = f"{experiment}/{arm}"
            per_seed: dict[str, Any] = {}
            for index, bundle in enumerate(proto.SEED_BUNDLES, start=1):
                checkpoint = out / experiment / arm / f"S{index}" / "resume.pt"
                if not checkpoint.exists():
                    raise RuntimeError(f"tie-role gradient diagnostic checkpoint missing: {checkpoint}")
                model = train.load_model_for_evaluation(
                    checkpoint,
                    experiment=experiment,
                    arm=arm,
                    seed_bundle=bundle,
                    data_manifest_sha256=str(public["sha256"]),
                    torch=torch,
                    device=device,
                )
                per_seed[str(bundle)] = decompose_model(
                    model, fixed_rows, experiment, arm, torch=torch, device=device
                )
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            ratios = [float(x["output_to_input_gradient_norm_ratio"]) for x in per_seed.values()]
            cosines = [float(x["input_output_gradient_cosine"]) for x in per_seed.values()]
            cancellations = [float(x["gradient_cancellation_fraction"]) for x in per_seed.values()]
            per_arm[key] = {
                "per_seed": per_seed,
                "mean_output_to_input_gradient_norm_ratio": _mean(ratios),
                "mean_input_output_gradient_cosine": _mean(cosines),
                "mean_gradient_cancellation_fraction": _mean(cancellations),
            }

    result = {
        "schema": "anra.tie-role-gradient-diagnostic/v1",
        "extension": proto.EXTENSION,
        "surface": "development identity fixed first 16 rows",
        "sealed_rows_used": False,
        "diagnostic_only": True,
        "per_arm": per_arm,
    }
    _atomic_json(out / "TIE_ROLE_GRADIENT_DIAGNOSTIC.json", result)
    return result


if __name__ == "__main__":
    raise SystemExit("Use through the canonical FORMATION-MUX operator.")
