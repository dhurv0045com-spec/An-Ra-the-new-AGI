"""Local-compute execution policy (Mission 27).

Some runs (e.g. instrument-hardening sessions) explicitly prohibit
user-local model/scientific compute. Launchers consult this guard BEFORE
touching checkpoints/CUDA/training so a forbidden run fails closed with a
clear message instead of silently consuming the user's machine.

  TRIQUETRA_NO_LOCAL_MODEL_COMPUTE=1  -> model compute refused
  TRIQUETRA_PHASE=STATIC_DEVELOPMENT  -> informative phase label

This module itself never executes models. The guard is only as strong as
callers: every scientific entry point must call assert_local_compute_allowed()
before loading checkpoints. (Enforced for qualify_checkpoint in this change;
historical scripts predate the policy and are documented as such.)
"""

from __future__ import annotations

import os


class LocalComputeForbidden(RuntimeError):
    pass


def policy_from_env() -> dict:
    return {
        "allow_local_model_compute": os.environ.get("TRIQUETRA_NO_LOCAL_MODEL_COMPUTE", "") != "1",
        "allow_local_training": os.environ.get("TRIQUETRA_NO_LOCAL_TRAINING", "") != "1",
        "phase": os.environ.get("TRIQUETRA_PHASE", "NORMAL"),
    }


def assert_local_compute_allowed(kind: str = "model") -> dict:
    pol = policy_from_env()
    if kind == "model" and not pol["allow_local_model_compute"]:
        raise LocalComputeForbidden(
            "NO_LOCAL_MODEL_COMPUTE: this session prohibits user-local model "
            "compute (TRIQUETRA_NO_LOCAL_MODEL_COMPUTE=1). Mark result "
            "EXECUTION_PENDING_COMPUTE_AUTHORIZATION.")
    if kind == "training" and not pol["allow_local_training"]:
        raise LocalComputeForbidden(
            "NO_LOCAL_TRAINING: training refused by execution policy.")
    return pol


def execution_environment() -> dict:
    """Telemetry recorded in every v2 receipt (versions only, no model run)."""
    try:
        import torch

        torch_version = torch.__version__
        try:
            cuda_available = bool(torch.cuda.is_available())
        except Exception:
            cuda_available = False
    except ImportError:
        torch_version, cuda_available = "unavailable", False
    import platform

    return {"torch_version": torch_version, "cuda_available": cuda_available,
            "platform": platform.platform(), "policy": policy_from_env()}
