"""Allocation-stable exact KV cache for canonical V4 inference.

The previous exact cache appended every token with ``torch.cat``.  That is
functionally correct, but it reallocates and copies the complete key/value
history at every decoding step.  This module keeps the exact same values in a
bounded preallocated tensor and exposes only the occupied view to attention.

This is a runtime efficiency profile, not a checkpoint or architecture change:
it owns no trainable state and can be replaced by the legacy cache at any time.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

EXACT_KV_CACHE_SCHEMA = "anra-exact-kv-cache/v1"


class ExactKVCacheError(RuntimeError):
    """Raised when a request violates the exact-cache geometry contract."""


@dataclass(frozen=True, slots=True)
class ExactKVCacheConfig:
    """Versioned, reversible runtime contract for exact cache storage."""

    profile: str = "preallocated-exact-v1"

    def __post_init__(self) -> None:
        if self.profile != "preallocated-exact-v1":
            raise ValueError(f"unsupported exact KV cache profile: {self.profile}")


class ExactStaticKVCache:
    """Preallocated exact K/V history with bounded recent-token retention.

    Before the cache reaches its capacity, storage addresses remain stable and
    each update writes only the new tokens.  At capacity, the oldest suffix is
    shifted left to preserve the model's existing sliding-window semantics.
    """

    schema = EXACT_KV_CACHE_SCHEMA
    algorithm = "exact-preallocated-contiguous-v1"
    backend = "exact-static"

    def __init__(
        self,
        *,
        num_kv_heads: int,
        max_seq_len: int,
        d_head: int,
        config: ExactKVCacheConfig | None = None,
    ) -> None:
        if num_kv_heads <= 0 or max_seq_len <= 0 or d_head <= 0:
            raise ValueError(
                "KV heads, maximum sequence length, and head dimension must be positive"
            )
        self.num_kv_heads = int(num_kv_heads)
        self.max_seq_len = int(max_seq_len)
        self.d_head = int(d_head)
        self.config = config or ExactKVCacheConfig()
        self.current_len = 0
        self.total_tokens_seen = 0
        self._batch_size: int | None = None
        self._device: torch.device | None = None
        self._dtype: torch.dtype | None = None
        self._key: torch.Tensor | None = None
        self._value: torch.Tensor | None = None
        self._allocation_count = 0
        self._bytes_written = 0
        self._bytes_shifted = 0

    @property
    def position(self) -> int:
        """Absolute sequence position, including tokens evicted by a window."""

        return self.total_tokens_seen

    @property
    def storage_pointers(self) -> tuple[int, int] | None:
        """Diagnostic identity proving storage is reused between updates."""

        if self._key is None or self._value is None:
            return None
        return self._key.data_ptr(), self._value.data_ptr()

    def _ensure_storage(self, tensor: torch.Tensor) -> None:
        batch, heads, _tokens, width = tensor.shape
        if heads != self.num_kv_heads or width != self.d_head:
            raise ExactKVCacheError("KV tensor geometry does not match the exact-cache contract")
        if self._batch_size is not None:
            if (
                batch != self._batch_size
                or tensor.device != self._device
                or tensor.dtype != self._dtype
            ):
                raise ExactKVCacheError(
                    "Exact KV cache cannot change batch, device, or dtype without reset"
                )
            return
        self._batch_size = int(batch)
        self._device = tensor.device
        self._dtype = tensor.dtype
        shape = (batch, self.num_kv_heads, self.max_seq_len, self.d_head)
        self._key = torch.empty(shape, dtype=tensor.dtype, device=tensor.device)
        self._value = torch.empty_like(self._key)
        self._allocation_count += 1

    def _make_room(self, new_tokens: int) -> None:
        overflow = max(0, self.current_len + new_tokens - self.max_seq_len)
        if overflow <= 0:
            return
        assert self._key is not None
        assert self._value is not None
        remaining = max(0, self.current_len - overflow)
        if remaining:
            source = slice(overflow, self.current_len)
            self._key[:, :, :remaining].copy_(self._key[:, :, source].clone())
            self._value[:, :, :remaining].copy_(self._value[:, :, source].clone())
            element_bytes = self._key.element_size()
            self._bytes_shifted += (
                2
                * remaining
                * self._batch_size
                * self.num_kv_heads
                * self.d_head
                * element_bytes
            )
        self.current_len = remaining

    @torch.no_grad()
    def update(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if key.shape != value.shape or key.ndim != 4:
            raise ExactKVCacheError("K and V must have identical [batch, heads, tokens, dim] shape")
        self._ensure_storage(key)
        tokens_seen = int(key.shape[2])
        self.total_tokens_seen += tokens_seen
        if tokens_seen > self.max_seq_len:
            key = key[:, :, -self.max_seq_len :, :]
            value = value[:, :, -self.max_seq_len :, :]
        new_tokens = int(key.shape[2])
        self._make_room(new_tokens)
        start = self.current_len
        end = start + new_tokens
        assert self._key is not None
        assert self._value is not None
        self._key[:, :, start:end].copy_(key.detach())
        self._value[:, :, start:end].copy_(value.detach())
        self.current_len = end
        self._bytes_written += 2 * key.numel() * key.element_size()
        return self._key[:, :, :end], self._value[:, :, :end]

    def reset(self) -> None:
        """Forget request state while retaining the allocation for reuse."""

        self.current_len = 0
        self.total_tokens_seen = 0
        self._bytes_written = 0
        self._bytes_shifted = 0

    def memory_report(self) -> dict[str, object]:
        reserved = 0
        if self._key is not None and self._value is not None:
            reserved = (
                self._key.numel() * self._key.element_size()
                + self._value.numel() * self._value.element_size()
            )
        occupied = 0
        if self._batch_size is not None and self._dtype is not None:
            occupied = (
                2
                * self._batch_size
                * self.num_kv_heads
                * self.current_len
                * self.d_head
                * torch.empty((), dtype=self._dtype).element_size()
            )
        return {
            "schema": self.schema,
            "profile": self.config.profile,
            "backend": self.backend,
            "algorithm": self.algorithm,
            "lossless": True,
            "reserved_bytes": reserved,
            "occupied_bytes": occupied,
            "allocation_count": self._allocation_count,
            "bytes_written": self._bytes_written,
            "bytes_shifted": self._bytes_shifted,
            "tokens_retained": self.current_len,
            "tokens_seen": self.total_tokens_seen,
        }


def analytical_copy_elements(*, tokens: int, elements_per_token: int) -> dict[str, int]:
    """Compare append-copy work before cache capacity is reached.

    The legacy ``cat`` path copies the old history plus the new token for both
    K and V at every step.  The static path writes only each new K/V token.
    """

    if tokens < 0 or elements_per_token <= 0:
        raise ValueError("tokens must be non-negative and elements_per_token positive")
    legacy = 2 * elements_per_token * tokens * (tokens + 1) // 2
    static = 2 * elements_per_token * tokens
    return {
        "legacy_cat_elements": legacy,
        "preallocated_elements": static,
        "saved_elements": max(0, legacy - static),
    }
