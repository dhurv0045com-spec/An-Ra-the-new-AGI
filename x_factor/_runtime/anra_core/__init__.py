"""Stable An-Ra Core surface: dense V4 neural model and validated executor.

Sampling and thought policy remain available in ``anra_core.generate`` and
``anra_core.brain`` as reference Connector utilities, but are intentionally not
part of this stable Core namespace.
"""

from .checkpoint import load_core_checkpoint
from .config import CANONICAL_CONFIG, CoreConfig
from .contracts import (
    ArchitectureIdentity,
    CapabilitySet,
    CheckpointIdentity,
    ExecutionProfile,
    PredictionResult,
    RepresentationIdentity,
    RuntimeIdentity,
)
from .errors import (
    CheckpointIncompatibleError,
    ContextOverflowError,
    CoreError,
    RepresentationIncompatibleError,
    ResourceExhaustionError,
    StateIncompatibleError,
    StateReleasedError,
    UnexpectedExecutionFault,
    UnsupportedCapabilityError,
    UnsupportedProfileError,
)
from .executor import CoreExecutor
from .model import AnRaCore
from .state import CoreState
from .tokenizer import V4Tokenizer

__version__ = "0.6.0"
__all__ = [
    "AnRaCore",
    "ArchitectureIdentity",
    "CANONICAL_CONFIG",
    "CapabilitySet",
    "CheckpointIdentity",
    "CheckpointIncompatibleError",
    "ContextOverflowError",
    "CoreConfig",
    "CoreError",
    "CoreExecutor",
    "CoreState",
    "ExecutionProfile",
    "PredictionResult",
    "RepresentationIdentity",
    "RepresentationIncompatibleError",
    "ResourceExhaustionError",
    "RuntimeIdentity",
    "StateIncompatibleError",
    "StateReleasedError",
    "UnexpectedExecutionFault",
    "UnsupportedCapabilityError",
    "UnsupportedProfileError",
    "V4Tokenizer",
    "load_core_checkpoint",
]
