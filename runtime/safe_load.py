from __future__ import annotations

import importlib
import warnings
from pathlib import Path

import torch


def ensure_torch_serialization_modules() -> None:
    """Bind PyTorch's tensor rebuild module before checkpoint unpickling.

    Some hosted notebook builds expose private torch submodules lazily but do
    not bind ``torch._utils`` when an older/full-resume pickle asks for it.
    Importing and binding the module explicitly preserves PyTorch's standard
    tensor reconstruction path; it does not add custom pickle globals.
    """

    rebuild_module = importlib.import_module("torch._utils")
    if not hasattr(rebuild_module, "_rebuild_tensor_v2"):
        raise RuntimeError(
            "The installed PyTorch build lacks torch._utils._rebuild_tensor_v2; "
            "restart the runtime with an official PyTorch installation."
        )
    if getattr(torch, "_utils", None) is not rebuild_module:
        torch._utils = rebuild_module


def safe_torch_load(
    path: str | Path,
    map_location: object | None = None,
    **kwargs: object,
) -> object:
    ensure_torch_serialization_modules()
    try:
        return torch.load(path, map_location=map_location, weights_only=True, **kwargs)
    except Exception:
        warnings.warn(
            f"Legacy checkpoint {path}: weights_only=False fallback",
            DeprecationWarning,
            stacklevel=2,
        )
        return torch.load(path, map_location=map_location, weights_only=False, **kwargs)
