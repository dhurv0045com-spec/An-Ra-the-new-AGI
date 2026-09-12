"""Operator readiness reporting with separate, non-collapsible dimensions.

``inspect`` never launches training and never silently upgrades a status. A
CPU-ready configuration with no corpus is reported as
``CONFIG_VALID / DATA_NOT_READY / DEVICE_UNVERIFIED`` — never as "ready".
"""
from __future__ import annotations

import json
import math
import os
from typing import Any, Mapping

from bramastra_lab.research.config import ConfigError, BuildConfig

CONFIG_VALID = "CONFIG_VALID"
CONFIG_INVALID = "CONFIG_INVALID"
DATA_NOT_READY = "DATA_NOT_READY"
DATA_READY_REFS_PRESENT = "DATA_READY_REFS_PRESENT"
DEVICE_UNVERIFIED = "DEVICE_UNVERIFIED"
DOES_NOT_FIT = "DOES_NOT_FIT"
LOCAL_SMOKE_PASSED = "LOCAL_SMOKE_PASSED"
TRAINING_READY_FOR_DECLARED_DEVICE = "TRAINING_READY_FOR_DECLARED_DEVICE"

# AdamW keeps parameters, gradients and two moment buffers per parameter.
# This is a declared planning constant for predicted fit, not a measured
# ceiling; activation memory is reported separately as unknown here.
TRAINING_BYTES_PER_PARAMETER = 16


def _device_report(config: BuildConfig) -> dict[str, Any]:
    report: dict[str, Any] = {"status": DEVICE_UNVERIFIED}
    try:
        import torch
    except ImportError:
        report["detail"] = "torch is not installed in this environment"
        return report
    parameters = config.parameter_count()
    parameter_bytes = parameters * 4
    predicted_training_bytes = parameters * TRAINING_BYTES_PER_PARAMETER
    report["torch_version"] = torch.__version__
    report["cuda_available"] = bool(torch.cuda.is_available())
    report["model_parameters"] = parameters
    report["predicted_checkpoint_bytes"] = parameter_bytes
    report["predicted_training_state_bytes"] = predicted_training_bytes
    if torch.cuda.is_available():
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(0)
            report["device_name"] = torch.cuda.get_device_name(0)
            report["device_free_bytes"] = int(free_bytes)
            report["device_total_bytes"] = int(total_bytes)
            if predicted_training_bytes > free_bytes:
                report["status"] = DOES_NOT_FIT
                report["detail"] = (
                    "predicted optimizer+parameter state exceeds reported free device memory; "
                    "activation memory not included, so this is necessary-not-sufficient")
            else:
                report["status"] = "DEVICE_FIT_PREDICTED"
                report["detail"] = "predicted state fits reported free memory; activation memory unverified"
        except Exception as exc:  # device query itself failed
            report["detail"] = f"device query failed: {exc}"
    else:
        report["detail"] = "no CUDA device visible; CPU execution is correctness-only, not a capacity certificate"
    return report


def _data_report(data_manifest_path: str | None) -> dict[str, Any]:
    if not data_manifest_path:
        return {"status": DATA_NOT_READY,
                "detail": "no dataset manifest supplied; provide one from prepare-data"}
    if not os.path.exists(data_manifest_path):
        return {"status": DATA_NOT_READY,
                "detail": f"dataset manifest not found at {data_manifest_path}"}
    try:
        with open(data_manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        return {"status": DATA_NOT_READY, "detail": f"dataset manifest unreadable: {exc}"}
    if not isinstance(manifest, Mapping) or not manifest.get("examples"):
        return {"status": DATA_NOT_READY, "detail": "dataset manifest declares no examples"}
    return {"status": DATA_READY_REFS_PRESENT,
            "detail": "manifest present; availability of referenced local files is re-verified by prepare-data"}


def inspect(raw_config: Mapping[str, Any], *, data_manifest_path: str | None = None,
            check_device: bool = False) -> dict[str, Any]:
    """Produce the readiness report. Idempotent and side-effect free."""
    report: dict[str, Any] = {}
    try:
        config = BuildConfig.from_dict(raw_config)
        report["config_status"] = CONFIG_VALID
        report["detail"] = "configuration parsed and validated"
    except ConfigError as exc:
        report["config_status"] = CONFIG_INVALID
        report["detail"] = str(exc)
        return report
    report["profile"] = config.model.profile
    report["parameter_count"] = config.parameter_count()
    report["base_decoder_parameter_count"] = config.base_decoder_parameter_count()

    data_report = _data_report(data_manifest_path)
    report["data_status"] = data_report["status"]
    if "detail" in data_report:
        report["data_detail"] = data_report["detail"]

    if check_device:
        device_report = _device_report(config)
        report["device_status"] = device_report.pop("status")
        report["device_detail"] = device_report
    else:
        report["device_status"] = DEVICE_UNVERIFIED
        report["device_detail"] = "device check not requested; pass --device"

    smoke_ledger = _read_smoke_ledger()
    report["smoke_ledger"] = smoke_ledger
    report["overall"] = (
        TRAINING_READY_FOR_DECLARED_DEVICE
        if report["config_status"] == CONFIG_VALID
        and report["data_status"] == DATA_READY_REFS_PRESENT
        and report.get("device_status") in {"DEVICE_FIT_PREDICTED"}
        else "NOT_READY_FOR_DECLARED_DEVICE"
    )
    return report


def _read_smoke_ledger() -> dict[str, Any]:
    """Read the authoritative cumulative smoke ledger if one exists nearby."""
    candidates = [
        os.path.join("engineering", "reports", "B2", "SESSION_LEDGER.json"),
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    return json.load(handle)
            except (OSError, json.JSONDecodeError):
                return {"error": "smoke ledger present but unreadable"}
    return {"cpu_learned_smoke_seconds": 0, "cpu_optimizer_updates": 0,
            "gpu_sessions": 0, "gpu_wall_seconds": 0, "gpu_optimizer_updates": 0,
            "note": "no smoke executed"}
