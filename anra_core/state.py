"""Explicit, isolated execution state for incremental autoregressive decode."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

import torch

from .errors import ContextOverflowError, StateReleasedError


@dataclass
class CoreState:
    """Opaque incremental execution state managed by the Core Executor.

    This handle encapsulates intermediate activation history (e.g. KV cache)
    required for accelerated token-by-token decoding without recomputing prefixes.
    It does NOT own sessions, prompts, tools, or user identities.
    """

    architecture_version: str
    checkpoint_id: str
    execution_profile_id: str
    capacity: int = 2048
    current_length: int = 0
    state_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    is_released: bool = False
    _kv_cache: list[tuple[torch.Tensor, torch.Tensor] | None] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self._kv_cache:
            self._kv_cache = [None] * 18

    def assert_active(self) -> None:
        if self.is_released:
            raise StateReleasedError(
                f"State handle {self.state_id} has already been released",
                details={"state_id": self.state_id},
            )

    def check_capacity(self, additional_tokens: int = 1) -> None:
        self.assert_active()
        if self.current_length + additional_tokens > self.capacity:
            raise ContextOverflowError(
                f"Context length ({self.current_length + additional_tokens}) exceeds capacity ({self.capacity})",
                details={
                    "current_length": self.current_length,
                    "additional_tokens": additional_tokens,
                    "capacity": self.capacity,
                },
            )

    def get_layer_kv(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor] | None:
        self.assert_active()
        if 0 <= layer_idx < len(self._kv_cache):
            return self._kv_cache[layer_idx]
        return None

    def set_layer_kv(self, layer_idx: int, kv: tuple[torch.Tensor, torch.Tensor]) -> None:
        self.assert_active()
        self._kv_cache[layer_idx] = kv

    def advance(self, num_tokens: int) -> None:
        self.assert_active()
        self.current_length += num_tokens

    def reset(self) -> None:
        """Reset state to position 0 and clear internal buffers."""
        self.assert_active()
        self._kv_cache = [None] * len(self._kv_cache)
        self.current_length = 0

    def fork(self) -> CoreState:
        """Create an isolated, independent deep copy of this execution state."""
        self.assert_active()
        cloned_cache: list[tuple[torch.Tensor, torch.Tensor] | None] = []
        for item in self._kv_cache:
            if item is None:
                cloned_cache.append(None)
            else:
                k, v = item
                cloned_cache.append((k.clone(), v.clone()))
        return CoreState(
            architecture_version=self.architecture_version,
            checkpoint_id=self.checkpoint_id,
            execution_profile_id=self.execution_profile_id,
            capacity=self.capacity,
            current_length=self.current_length,
            _kv_cache=cloned_cache,
        )

    def release(self) -> None:
        """Explicitly release all tensor buffers."""
        self._kv_cache.clear()
        self.is_released = True

    def descriptor(self) -> dict[str, Any]:
        return {
            "state_id": self.state_id,
            "architecture_version": self.architecture_version,
            "checkpoint_id": self.checkpoint_id,
            "execution_profile_id": self.execution_profile_id,
            "capacity": self.capacity,
            "current_length": self.current_length,
            "is_released": self.is_released,
        }
