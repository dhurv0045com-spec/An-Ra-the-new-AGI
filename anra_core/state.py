"""Opaque, executor-owned incremental state for An-Ra Core.

The public handle exposes identity and lifecycle metadata only. Transformer KV
tensors remain an implementation detail. Buffers are preallocated lazily and a
new logical length is committed only after the complete neural call succeeds.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from typing import Any

import torch

from .errors import ContextOverflowError, StateReleasedError

CORE_STATE_SCHEMA = "anra-core-state/v2"


@dataclass(slots=True)
class CoreState:
    """Opaque handle for one homogeneous batch of incremental sequences.

    Instances are created and mutated by :class:`CoreExecutor`. Direct cache
    access is deliberately private; callers use executor lifecycle methods.
    """

    _owner_id: str
    _architecture_id: str
    _parameter_id: str
    _representation_id: str | None
    _execution_profile_id: str
    _batch_size: int
    _capacity: int
    _n_layers: int
    _n_kv_heads: int
    _head_dim: int
    _state_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    _current_length: int = 0
    _is_released: bool = False
    _buffers: list[tuple[torch.Tensor, torch.Tensor] | None] = field(default_factory=list)
    _history: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self._batch_size <= 0 or self._capacity <= 0:
            raise ValueError("batch size and state capacity must be positive")
        if self._n_layers <= 0 or self._n_kv_heads <= 0 or self._head_dim <= 0:
            raise ValueError("state cache geometry must be positive")
        if not self._buffers:
            self._buffers = [None] * self._n_layers
        if len(self._buffers) != self._n_layers:
            raise ValueError("state cache layer count does not match its geometry")

    @property
    def schema(self) -> str:
        return CORE_STATE_SCHEMA

    @property
    def state_id(self) -> str:
        return self._state_id

    @property
    def architecture_id(self) -> str:
        return self._architecture_id

    @property
    def parameter_id(self) -> str:
        return self._parameter_id

    @property
    def representation_id(self) -> str | None:
        return self._representation_id

    @property
    def execution_profile_id(self) -> str:
        return self._execution_profile_id

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def current_length(self) -> int:
        return self._current_length

    @property
    def is_released(self) -> bool:
        return self._is_released

    def _assert_active(self) -> None:
        if self._is_released:
            raise StateReleasedError(
                f"State handle {self._state_id} has already been released",
                details={"state_id": self._state_id},
            )

    def _check_capacity(self, additional_tokens: int) -> None:
        self._assert_active()
        if additional_tokens <= 0:
            raise ValueError("additional_tokens must be positive")
        requested = self._current_length + additional_tokens
        if requested > self._capacity:
            raise ContextOverflowError(
                f"Context length {requested} exceeds state capacity {self._capacity}",
                details={
                    "current_length": self._current_length,
                    "additional_tokens": additional_tokens,
                    "capacity": self._capacity,
                },
            )

    def _ensure_buffers(self, *, device: torch.device, dtype: torch.dtype) -> None:
        """Allocate stable backing storage once; occupied length stays logical."""
        self._assert_active()
        shape = (
            self._batch_size,
            self._n_kv_heads,
            self._capacity,
            self._head_dim,
        )
        for index, item in enumerate(self._buffers):
            if item is None:
                key = torch.empty(shape, device=device, dtype=dtype)
                self._buffers[index] = (key, torch.empty_like(key))
                continue
            key, value = item
            if key.shape != shape or value.shape != shape:
                raise RuntimeError("state cache storage geometry drifted")
            if key.device != device or value.device != device:
                raise RuntimeError("state cache storage device drifted")
            if key.dtype != dtype or value.dtype != dtype:
                raise RuntimeError("state cache storage dtype drifted")

    def _cache_buffers(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        self._assert_active()
        if any(item is None for item in self._buffers):
            raise RuntimeError("state cache storage has not been allocated")
        return [item for item in self._buffers if item is not None]

    def _commit(self, token_ids: torch.Tensor) -> None:
        """Commit a successful neural call as one atomic logical boundary."""
        self._assert_active()
        if token_ids.ndim != 2 or token_ids.shape[0] != self._batch_size:
            raise RuntimeError("committed token batch does not match state")
        token_copy = token_ids.detach().to(device="cpu", dtype=torch.int64).contiguous()
        if self._history is None:
            self._history = token_copy
        else:
            self._history = torch.cat((self._history, token_copy), dim=1)
        self._current_length += int(token_ids.shape[1])

    def _reset(self) -> None:
        self._assert_active()
        self._current_length = 0
        self._history = None

    def _truncate(self, target_length: int) -> None:
        self._assert_active()
        if not 0 <= target_length <= self._current_length:
            raise ValueError(
                f"target_length ({target_length}) must be in [0, {self._current_length}]"
            )
        self._current_length = target_length
        if target_length == 0:
            self._history = None
        elif self._history is not None:
            self._history = self._history[:, :target_length].clone()

    def _fork(self) -> CoreState:
        self._assert_active()
        clone = CoreState(
            _owner_id=self._owner_id,
            _architecture_id=self._architecture_id,
            _parameter_id=self._parameter_id,
            _representation_id=self._representation_id,
            _execution_profile_id=self._execution_profile_id,
            _batch_size=self._batch_size,
            _capacity=self._capacity,
            _n_layers=self._n_layers,
            _n_kv_heads=self._n_kv_heads,
            _head_dim=self._head_dim,
            _current_length=self._current_length,
            _history=self._history.clone() if self._history is not None else None,
        )
        cloned_buffers: list[tuple[torch.Tensor, torch.Tensor] | None] = []
        for item in self._buffers:
            if item is None:
                cloned_buffers.append(None)
                continue
            key, value = item
            new_key = torch.empty_like(key)
            new_value = torch.empty_like(value)
            if self._current_length:
                new_key[:, :, : self._current_length].copy_(
                    key[:, :, : self._current_length]
                )
                new_value[:, :, : self._current_length].copy_(
                    value[:, :, : self._current_length]
                )
            cloned_buffers.append((new_key, new_value))
        clone._buffers = cloned_buffers
        return clone

    def _release(self) -> None:
        self._buffers.clear()
        self._history = None
        self._is_released = True

    def logical_memory_bytes(self) -> int:
        self._assert_active()
        for item in self._buffers:
            if item is not None:
                element_size = item[0].element_size()
                return (
                    2
                    * self._n_layers
                    * self._batch_size
                    * self._n_kv_heads
                    * self._current_length
                    * self._head_dim
                    * element_size
                )
        return 0

    def reserved_memory_bytes(self) -> int:
        self._assert_active()
        seen: set[tuple[str, int]] = set()
        total = 0
        for item in self._buffers:
            if item is None:
                continue
            for tensor in item:
                storage = tensor.untyped_storage()
                identity = (str(tensor.device), storage.data_ptr())
                if identity not in seen:
                    seen.add(identity)
                    total += storage.nbytes()
        return total

    def memory_bytes(self) -> int:
        """Backward-compatible alias for logical occupied bytes."""
        return self.logical_memory_bytes()

    def prefix_sha256(self) -> str | None:
        self._assert_active()
        if self._history is None:
            return None
        return hashlib.sha256(memoryview(self._history.numpy())).hexdigest()

    def descriptor(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "state_id": self._state_id,
            "architecture_id": self._architecture_id,
            "parameter_id": self._parameter_id,
            "representation_id": self._representation_id,
            "execution_profile_id": self._execution_profile_id,
            "batching_mode": "homogeneous",
            "batch_size": self._batch_size,
            "capacity": self._capacity,
            "current_length": self._current_length,
            "prefix_sha256": self.prefix_sha256() if not self._is_released else None,
            "logical_memory_bytes": (
                self.logical_memory_bytes() if not self._is_released else 0
            ),
            "reserved_memory_bytes": (
                self.reserved_memory_bytes() if not self._is_released else 0
            ),
            "is_released": self._is_released,
        }
