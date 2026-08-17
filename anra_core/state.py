"""Explicit, isolated execution state for incremental autoregressive decode.

Supports multi-batch execution, rollback truncation, zero-copy cloning,
safe byte serialization, and exact memory byte tracking.
"""

from __future__ import annotations

import io
import json
import uuid
from dataclasses import dataclass, field
from typing import Any

import torch

from .errors import ContextOverflowError, StateIncompatibleError, StateReleasedError

STATE_MAGIC_HEADER = b"ANRA_STATE_v1\x00"


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
    batch_size: int = 1
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

    def truncate(self, target_length: int) -> None:
        """Roll back KV cache history to a prefix target length."""
        self.assert_active()
        if target_length < 0 or target_length > self.current_length:
            raise ValueError(
                f"target_length ({target_length}) must be in [0, {self.current_length}]"
            )
        if target_length == 0:
            self.reset()
            return
        for idx, item in enumerate(self._kv_cache):
            if item is not None:
                k, v = item
                self._kv_cache[idx] = (k[:, :, :target_length, :], v[:, :, :target_length, :])
        self.current_length = target_length

    def reset(self) -> None:
        """Reset state to position 0 and clear internal buffers."""
        self.assert_active()
        self._kv_cache = [None] * len(self._kv_cache)
        self.current_length = 0

    def memory_bytes(self) -> int:
        """Calculate the total allocated byte size of the internal KV cache tensors."""
        self.assert_active()
        total = 0
        for item in self._kv_cache:
            if item is not None:
                k, v = item
                total += k.element_size() * k.numel() + v.element_size() * v.numel()
        return total

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
            batch_size=self.batch_size,
            capacity=self.capacity,
            current_length=self.current_length,
            _kv_cache=cloned_cache,
        )

    def serialize(self) -> bytes:
        """Serialize state metadata and tensor buffers safely into a byte stream."""
        self.assert_active()
        meta = {
            "state_id": self.state_id,
            "architecture_version": self.architecture_version,
            "checkpoint_id": self.checkpoint_id,
            "execution_profile_id": self.execution_profile_id,
            "batch_size": self.batch_size,
            "capacity": self.capacity,
            "current_length": self.current_length,
            "layers": len(self._kv_cache),
        }
        meta_bytes = json.dumps(meta, sort_keys=True).encode("utf-8")
        tensor_dict: dict[str, torch.Tensor] = {}
        for idx, item in enumerate(self._kv_cache):
            if item is not None:
                k, v = item
                tensor_dict[f"k_{idx}"] = k.cpu()
                tensor_dict[f"v_{idx}"] = v.cpu()

        buffer = io.BytesIO()
        buffer.write(STATE_MAGIC_HEADER)
        buffer.write(len(meta_bytes).to_bytes(4, byteorder="big"))
        buffer.write(meta_bytes)
        torch.save(tensor_dict, buffer)
        return buffer.getvalue()

    @classmethod
    def deserialize(cls, data: bytes, *, device: str = "cpu") -> CoreState:
        """Safely reconstruct CoreState from serialized bytes."""
        if not data.startswith(STATE_MAGIC_HEADER):
            raise StateIncompatibleError(
                "Invalid state serialization magic header",
                details={"header": data[:16]},
            )
        offset = len(STATE_MAGIC_HEADER)
        meta_len = int.from_bytes(data[offset : offset + 4], byteorder="big")
        offset += 4
        meta_bytes = data[offset : offset + meta_len]
        offset += meta_len
        meta = json.loads(meta_bytes.decode("utf-8"))

        tensor_stream = io.BytesIO(data[offset:])
        tensor_dict = torch.load(tensor_stream, map_location=device, weights_only=True)

        num_layers = int(meta.get("layers", 18))
        reconstructed_cache: list[tuple[torch.Tensor, torch.Tensor] | None] = []
        for idx in range(num_layers):
            k_key, v_key = f"k_{idx}", f"v_{idx}"
            if k_key in tensor_dict and v_key in tensor_dict:
                reconstructed_cache.append((tensor_dict[k_key], tensor_dict[v_key]))
            else:
                reconstructed_cache.append(None)

        return cls(
            architecture_version=str(meta["architecture_version"]),
            checkpoint_id=str(meta["checkpoint_id"]),
            execution_profile_id=str(meta["execution_profile_id"]),
            batch_size=int(meta.get("batch_size", 1)),
            capacity=int(meta["capacity"]),
            current_length=int(meta["current_length"]),
            state_id=str(meta["state_id"]),
            is_released=False,
            _kv_cache=reconstructed_cache,
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
            "batch_size": self.batch_size,
            "capacity": self.capacity,
            "current_length": self.current_length,
            "allocated_memory_bytes": self.memory_bytes() if not self.is_released else 0,
            "is_released": self.is_released,
        }
