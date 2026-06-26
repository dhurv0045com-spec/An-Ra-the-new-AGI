from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

MetricValue = int | float | str | bool
MetricPayload = Mapping[str, MetricValue]

if TYPE_CHECKING:
    import torch

    TensorBatch = Mapping[str, torch.Tensor]
    KVCache = Sequence[tuple[torch.Tensor, torch.Tensor]]
else:
    TensorBatch = Mapping[str, object]
    KVCache = Sequence[tuple[object, object]]


@dataclass(frozen=True, slots=True)
class ModelOutput:
    """Canonical model output used by trainers and inference engines."""

    logits: torch.Tensor
    loss: torch.Tensor | None = None
    past_key_values: KVCache | None = None


@dataclass(frozen=True, slots=True)
class MemoryRecord:
    """A retrieved or stored memory item."""

    id: str
    text: str
    metadata: Mapping[str, str | int | float | bool]
    score: float
    created_at: float

    @property
    def content(self) -> str:
        """Compatibility alias used by synchronous memory tiers."""
        return self.text


@dataclass(frozen=True, slots=True)
class HealthStatus:
    """Small serializable health report shared by package components."""

    healthy: bool
    message: str
    details: Mapping[str, int | float | str | bool]


@dataclass(frozen=True, slots=True)
class IdentityState:
    """Serializable identity-module state exposed to observability and checkpoints."""

    values: Mapping[str, float]


@runtime_checkable
class ModelProtocol(Protocol):
    """Structural interface for autoregressive PyTorch models."""

    training: bool

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        *,
        use_cache: bool = False,
        past_key_values: KVCache | None = None,
    ) -> ModelOutput:
        """Run a forward pass and optionally return cache state."""

    def generate(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int,
        eos_token_id: int | None = None,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Generate continuations from token IDs."""

    def state_dict(self) -> Mapping[str, torch.Tensor]:
        """Return checkpointable model parameters."""

    def load_state_dict(
        self,
        state_dict: Mapping[str, torch.Tensor],
        strict: bool = True,
    ) -> object:
        """Load checkpointed model parameters."""


@runtime_checkable
class MemoryTierProtocol(Protocol):
    """Structural interface for memory tiers."""

    @property
    def name(self) -> str:
        """Stable tier name used by routers and metrics."""

    async def add(
        self,
        text: str,
        *,
        metadata: Mapping[str, str | int | float | bool] | None = None,
    ) -> str:
        """Store text and return a stable memory ID."""

    async def search(self, query: str, *, limit: int = 10) -> Sequence[MemoryRecord]:
        """Return ranked memories for a query."""

    async def delete(self, memory_id: str) -> bool:
        """Delete a memory by ID."""

    async def close(self) -> None:
        """Release resources held by this tier."""


@runtime_checkable
class IdentityModuleProtocol(Protocol):
    """Structural interface for identity and affect modules."""

    @property
    def name(self) -> str:
        """Stable module name used by registries and metrics."""

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Transform hidden states."""

    def appraise(self, signals: Mapping[str, float]) -> IdentityState:
        """Update identity state from external signals."""

    def snapshot(self) -> IdentityState:
        """Return current serializable state."""

    def restore(self, state: IdentityState) -> None:
        """Restore previously serialized state."""


@runtime_checkable
class TrainerProtocol(Protocol):
    """Structural interface for training algorithms."""

    @property
    def name(self) -> str:
        """Stable trainer name used by registries and metrics."""

    def train_step(self, batch: TensorBatch) -> MetricPayload:
        """Run one optimizer step and return scalar metrics."""

    def evaluate(self, batch: TensorBatch) -> MetricPayload:
        """Evaluate a batch without mutating model parameters."""

    def save_checkpoint(self, path: Path) -> None:
        """Write trainer state to disk."""

    def load_checkpoint(self, path: Path) -> None:
        """Restore trainer state from disk."""


@runtime_checkable
class ObjectiveProtocol(Protocol):
    """Structural interface for composable training objectives."""

    @property
    def name(self) -> str:
        """Stable objective name used by registries and metrics."""

    def __call__(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute a scalar loss tensor."""


@runtime_checkable
class InferenceStrategyProtocol(Protocol):
    """Structural interface for decoding strategies."""

    @property
    def name(self) -> str:
        """Stable strategy name used by registries and metrics."""

    def generate(
        self,
        model: ModelProtocol,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int,
        eos_token_id: int | None = None,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Generate tokens using a concrete decoding strategy."""
