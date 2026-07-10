# ruff: noqa: E402
"""Profile legacy checkpoint weights for architecture-specific pathologies."""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from collections.abc import Mapping
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


def profile_checkpoint(path: Path) -> dict[str, object]:
    blob = safe_torch_load(path, map_location="cpu")
    state = _model_state(blob)
    total_elements = 0
    nonfinite_elements = 0
    for tensor in state.values():
        total_elements += tensor.numel()
        nonfinite_elements += int((~torch.isfinite(tensor)).sum().item())

    selected: dict[str, dict[str, object]] = {}
    for name in (
        "residual_depth_logits",
        "dstp_temperature_log",
        "layer_temperature_bias",
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
        "nonfinite_elements": nonfinite_elements,
        "selected_tensors": selected,
        "residual_scales": residual_scales,
        "dstp_temperatures": dstp_temperatures,
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
    args = parser.parse_args()
    report = profile_checkpoint(resolve_checkpoint(args.checkpoint))
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
