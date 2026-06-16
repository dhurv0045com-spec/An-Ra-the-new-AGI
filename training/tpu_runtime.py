from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

import torch

from anra.anra_paths import DRIVE_V2_CHECKPOINTS
from runtime.safe_load import safe_torch_load
from training.v2_runtime import (
    CheckpointCompatibilityError,
    _load_state_with_base_fallback,
    migrate_checkpoint_state,
)


logger = logging.getLogger(__name__)


class TPUUnavailableError(RuntimeError):
    """Raised when the dedicated TPU trainer is started outside PyTorch/XLA."""


def require_torch_xla() -> tuple[Any, Any]:
    """Import PyTorch/XLA lazily so non-TPU tests and --help keep working."""
    try:
        import torch_xla.core.xla_model as xm
        import torch_xla.distributed.parallel_loader as pl
    except Exception as exc:  # pragma: no cover - depends on Colab TPU image.
        raise TPUUnavailableError(
            "PyTorch/XLA is not available. In Colab choose a TPU runtime, then run "
            "`pip install torch torch_xla[tpu] -f https://storage.googleapis.com/libtpu-releases/index.html`."
        ) from exc
    return xm, pl


def optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    """Move restored optimizer state tensors onto the XLA device."""
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


def load_checkpoint_cpu_first(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any,
    checkpoint_path: Path,
    *,
    device: torch.device,
    strict: bool = False,
) -> dict[str, Any]:
    """
    Restore an AN-RA checkpoint for TPU runs without deserializing tensors directly
    onto XLA. This keeps resume compatible with existing CUDA checkpoints.
    """
    state: dict[str, Any] = {
        "loaded": False,
        "global_step": 0,
        "epoch": 0,
        "best_loss": float("inf"),
        "sessions_completed": 0,
        "migration": None,
    }
    if not checkpoint_path.exists():
        return state

    blob = safe_torch_load(checkpoint_path, map_location="cpu")
    model_state = blob.get("model_state_dict", blob.get("model", blob)) if isinstance(blob, dict) else blob
    if not isinstance(model_state, dict):
        raise CheckpointCompatibilityError(f"Checkpoint has no model state: {checkpoint_path}")

    try:
        migrated_state, migration = migrate_checkpoint_state(model_state, model.state_dict())
        _load_state_with_base_fallback(model, migrated_state, strict=strict)
    except RuntimeError as exc:
        raise CheckpointCompatibilityError(
            f"Checkpoint {checkpoint_path} is incompatible with the requested model architecture."
        ) from exc

    if isinstance(blob, dict):
        if optimizer is not None:
            try:
                optimizer.load_state_dict(blob.get("optimizer_state_dict", blob.get("optimizer", {})))
                optimizer_state_to_device(optimizer, device)
            except Exception as exc:
                logger.warning("TPU optimizer state restore skipped from %s: %s", checkpoint_path, exc)
        if scheduler is not None:
            try:
                scheduler.load_state_dict(blob.get("scheduler_state_dict", blob.get("scheduler", {})))
            except Exception as exc:
                logger.warning("TPU scheduler state restore skipped from %s: %s", checkpoint_path, exc)
        state["global_step"] = int(blob.get("global_step", blob.get("step", 0)))
        state["epoch"] = int(blob.get("epoch", 0))
        state["best_loss"] = float(blob.get("best_loss", float("inf")))
        state["sessions_completed"] = int(blob.get("sessions_completed", 0))

    state["loaded"] = True
    state["migration"] = migration
    return state


def xla_save_checkpoint(
    payload: dict[str, Any],
    checkpoint_path: Path,
    *,
    xm: Any,
    mirror_to_drive: bool = True,
) -> None:
    """Save a TPU checkpoint with XLA materialization and an optional Drive mirror."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    xm.mark_step()
    tmp = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    xm.save(payload, str(tmp))
    tmp.replace(checkpoint_path)
    if mirror_to_drive:
        try:
            DRIVE_V2_CHECKPOINTS.mkdir(parents=True, exist_ok=True)
            target = DRIVE_V2_CHECKPOINTS / checkpoint_path.name
            shutil.copy2(checkpoint_path, target)
            print(f"[TPU Drive] checkpoint mirrored: {target}", flush=True)
        except Exception as exc:
            print(f"[TPU Drive] checkpoint mirror failed: {exc}", flush=True)


def restore_checkpoint_from_drive(checkpoint_path: Path) -> bool:
    """Copy the canonical Drive checkpoint into the local runtime if needed."""
    if checkpoint_path.exists():
        return False
    for candidate in (
        DRIVE_V2_CHECKPOINTS / checkpoint_path.name,
        DRIVE_V2_CHECKPOINTS.parent.parent / checkpoint_path.name,
    ):
        if candidate.exists():
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(candidate, checkpoint_path)
            print(f"[TPU Resume] restored {candidate} -> {checkpoint_path}", flush=True)
            return True
    return False
