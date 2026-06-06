"""Core interfaces, configuration, registries, and model implementations."""

from __future__ import annotations

import sys

from anra.core.config import AnRaConfig, ModelConfig, PathsConfig, TrainingConfig
from anra.core.registry import (
    IDENTITY_REGISTRY,
    INFERENCE_REGISTRY,
    MEMORY_REGISTRY,
    MODEL_REGISTRY,
    OBJECTIVE_REGISTRY,
    TRAINING_REGISTRY,
    Registry,
)


def _anra_brain_import_in_progress() -> bool:
    module = sys.modules.get("anra_brain")
    return module is not None and not hasattr(module, "CausalTransformerV2")


if not _anra_brain_import_in_progress():
    from anra.core import model as _model_module  # triggers registration  # noqa: F401

__all__ = [
    "AnRaConfig",
    "IDENTITY_REGISTRY",
    "INFERENCE_REGISTRY",
    "MEMORY_REGISTRY",
    "MODEL_REGISTRY",
    "ModelConfig",
    "OBJECTIVE_REGISTRY",
    "PathsConfig",
    "Registry",
    "TRAINING_REGISTRY",
    "TrainingConfig",
]
