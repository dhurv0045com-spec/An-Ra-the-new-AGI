"""AN-RA sovereign AGI research platform."""

from __future__ import annotations

__version__ = "0.3.0"

# Core
from anra.core.config import AnRaConfig, ModelConfig, PathsConfig, TrainingConfig
from anra.core.protocols import (
    IdentityModuleProtocol,
    InferenceStrategyProtocol,
    MemoryTierProtocol,
    ModelProtocol,
    TrainerProtocol,
)
from anra.core.registry import (
    IDENTITY_REGISTRY,
    INFERENCE_REGISTRY,
    MEMORY_REGISTRY,
    MODEL_REGISTRY,
    OBJECTIVE_REGISTRY,
    TRAINING_REGISTRY,
    Registry,
)

# Trigger registrations by importing subpackages.
import anra.core.model  # noqa: F401
import anra.identity.hal  # noqa: F401
import anra.memory.router  # noqa: F401

__all__ = [
    "__version__",
    "MODEL_REGISTRY",
    "MEMORY_REGISTRY",
    "TRAINING_REGISTRY",
    "OBJECTIVE_REGISTRY",
    "INFERENCE_REGISTRY",
    "IDENTITY_REGISTRY",
    "Registry",
    "AnRaConfig",
    "ModelConfig",
    "TrainingConfig",
    "PathsConfig",
    "ModelProtocol",
    "MemoryTierProtocol",
    "IdentityModuleProtocol",
    "TrainerProtocol",
    "InferenceStrategyProtocol",
]
