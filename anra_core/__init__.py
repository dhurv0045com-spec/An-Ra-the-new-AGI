"""An-Ra Core: Standalone V4 Neural Model and Core Executor."""

from .brain import Brain, Thought, ThoughtPolicy
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
from .generate import generate
from .model import AnRaCore
from .state import CoreState
from .tokenizer import V4Tokenizer

__version__ = "0.4.0-vnext"
__all__ = [
    "AnRaCore",
    "ArchitectureIdentity",
    "Brain",
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
    "Thought",
    "ThoughtPolicy",
    "UnexpectedExecutionFault",
    "UnsupportedCapabilityError",
    "UnsupportedProfileError",
    "V4Tokenizer",
    "generate",
    "load_core_checkpoint",
]
