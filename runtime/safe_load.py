from __future__ import annotations

import warnings
from pathlib import Path

import torch


def safe_torch_load(
    path: str | Path,
    map_location: object | None = None,
    **kwargs: object,
) -> object:
    try:
        return torch.load(path, map_location=map_location, weights_only=True, **kwargs)
    except Exception:
        warnings.warn(
            f"Legacy checkpoint {path}: weights_only=False fallback",
            DeprecationWarning,
            stacklevel=2,
        )
        return torch.load(path, map_location=map_location, weights_only=False, **kwargs)
