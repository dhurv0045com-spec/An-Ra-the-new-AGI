"""AN-RA sovereign AGI research platform."""

from __future__ import annotations

import sys
from importlib.util import find_spec

__version__ = "0.3.0"

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


def _anra_brain_import_in_progress() -> bool:
    module = sys.modules.get("anra_brain")
    return module is not None and not hasattr(module, "CausalTransformerV2")


def _torch_available() -> bool:
    if "torch" in sys.modules and sys.modules["torch"] is None:
        return False
    return find_spec("torch") is not None


if _torch_available() and not _anra_brain_import_in_progress():
    import anra.core.model  # noqa: F401
import anra.identity.civ  # noqa: E402, F401
import anra.identity.esv  # noqa: E402, F401
import anra.identity.hal  # noqa: E402, F401
import anra.inference  # noqa: E402, F401
import anra.memory  # noqa: E402, F401
import anra.memory.router  # noqa: E402, F401
import anra.serving  # noqa: E402, F401

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
