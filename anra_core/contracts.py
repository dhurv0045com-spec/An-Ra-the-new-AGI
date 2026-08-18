"""Versioned identities, execution contracts, and capability descriptors for An-Ra Core."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import torch

CORE_RUNTIME_VERSION = "0.5.0"
CORE_API_SCHEMA_VERSION = 1
CORE_STATE_SCHEMA_VERSION = 2


@dataclass(frozen=True, slots=True)
class ArchitectureIdentity:
    """Mathematical architecture identity of the neural substrate."""

    architecture_version: str
    vocab_size: int
    d_model: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    d_ff: int
    block_size: int
    rope_base: float
    sliding_window: int
    full_attention_every: int
    qk_norm: bool
    dense_parameter_count: int
    architecture_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CheckpointIdentity:
    """Identity and provenance of a loaded parameter checkpoint."""

    checkpoint_sha256: str | None
    source_path: str | None
    global_step: int | None
    training_stage: str | None
    source_commit: str | None
    tokenizer_contract_valid: bool
    parameter_sha256: str | None = None
    tokenizer_contract_present: bool = False
    tokenizer_contract_verified: bool = False
    ignored_tensor_names: tuple[str, ...] = ()
    legacy_unverified: bool = False
    artifact_class: str | None = None
    artifact_schema_version: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RepresentationIdentity:
    """Identity and cryptographic contract of the discrete token representation."""

    schema_version: int
    vocab_size: int
    vocabulary_sha256: str
    probe_count: int
    probe_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RuntimeIdentity:
    """Implementation identity of the Core Executor engine."""

    engine_name: str = "anra-core-executor-vnext"
    runtime_version: str = CORE_RUNTIME_VERSION
    api_schema_version: int = CORE_API_SCHEMA_VERSION
    state_schema_version: int = CORE_STATE_SCHEMA_VERSION
    backend_framework: str = "torch"
    torch_version: str = torch.__version__

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CapabilitySet:
    """Explicitly advertised capabilities of the Core Executor."""

    supports_full_forward: bool = True
    supports_incremental_decode: bool = True
    supports_state_fork: bool = True
    supports_state_reset: bool = True
    supports_state_serialization: bool = False
    supports_quantization: bool = False
    supports_multi_device_sharding: bool = False
    supports_homogeneous_batching: bool = True
    supports_ragged_batching: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ExecutionProfile:
    """Execution profile specifying numerical precision and compute backend."""

    profile_id: str
    category: Literal["exact", "optimized", "approximate"]
    device: str
    dtype: str
    deterministic: bool = True
    sliding_window_enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PredictionResult:
    """The raw mathematical output of a forward or incremental decode step."""

    logits: torch.Tensor
    sequence_length: int
    execution_profile_id: str
    metadata: dict[str, Any] = field(default_factory=dict)
