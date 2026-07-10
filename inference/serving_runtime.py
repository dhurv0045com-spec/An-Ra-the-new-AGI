"""Deterministic serving primitives for continuous batching and paged KV state.

This module is model-agnostic on purpose.  The model runner owns tensor
creation; the scheduler owns fair request grouping and the cache owns bounded
page lifecycle.  Both are directly testable without a GPU.
"""

from __future__ import annotations

import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from threading import RLock
from typing import Any


@dataclass(frozen=True)
class GenerationWork:
    prompt_token_count: int
    max_new_tokens: int
    model_id: str
    adapter_id: str | None = None
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    submitted_at: float = field(default_factory=time.monotonic)

    def __post_init__(self) -> None:
        if self.prompt_token_count < 0 or self.max_new_tokens <= 0:
            raise ValueError("token counts must be non-negative and max_new_tokens positive")
        if not self.model_id:
            raise ValueError("model_id is required")

    @property
    def token_cost(self) -> int:
        return self.prompt_token_count + self.max_new_tokens


@dataclass(frozen=True)
class ScheduledBatch:
    model_id: str
    adapter_id: str | None
    requests: tuple[GenerationWork, ...]

    @property
    def token_cost(self) -> int:
        return sum(request.token_cost for request in self.requests)


class ContinuousBatchScheduler:
    """FIFO scheduler that batches only compatible model/adapter requests."""

    def __init__(self, *, max_batch_size: int = 8, max_batch_tokens: int = 8192) -> None:
        if max_batch_size <= 0 or max_batch_tokens <= 0:
            raise ValueError("batch limits must be positive")
        self.max_batch_size = int(max_batch_size)
        self.max_batch_tokens = int(max_batch_tokens)
        self._pending: deque[GenerationWork] = deque()
        self._lock = RLock()

    def submit(self, work: GenerationWork) -> str:
        with self._lock:
            self._pending.append(work)
        return work.request_id

    def next_batch(self) -> ScheduledBatch | None:
        with self._lock:
            if not self._pending:
                return None
            first = self._pending.popleft()
            accepted = [first]
            deferred: deque[GenerationWork] = deque()
            while self._pending:
                candidate = self._pending.popleft()
                compatible = (
                    candidate.model_id == first.model_id
                    and candidate.adapter_id == first.adapter_id
                )
                current_cost = sum(item.token_cost for item in accepted)
                fits = current_cost + candidate.token_cost <= self.max_batch_tokens
                if compatible and fits and len(accepted) < self.max_batch_size:
                    accepted.append(candidate)
                else:
                    deferred.append(candidate)
            self._pending.extend(deferred)
        return ScheduledBatch(first.model_id, first.adapter_id, tuple(accepted))

    def report(self) -> dict[str, int]:
        with self._lock:
            return {
                "pending_requests": len(self._pending),
                "pending_tokens": sum(work.token_cost for work in self._pending),
                "max_batch_size": self.max_batch_size,
                "max_batch_tokens": self.max_batch_tokens,
            }


class PagedKVCache:
    """Opaque fixed-size page allocator for per-request KV payloads."""

    def __init__(self, *, page_size: int = 16, max_pages: int = 256) -> None:
        if page_size <= 0 or max_pages <= 0:
            raise ValueError("page_size and max_pages must be positive")
        self.page_size = int(page_size)
        self.max_pages = int(max_pages)
        self._pages: dict[int, list[Any]] = {}
        self._requests: dict[str, list[int]] = {}
        self._next_page_id = 0
        self._lock = RLock()

    def append(self, request_id: str, values: list[Any]) -> tuple[int, ...]:
        """Append opaque KV values, allocating pages atomically or failing closed."""
        if not request_id:
            raise ValueError("request_id is required")
        if not values:
            return tuple(self._requests.get(request_id, ()))
        with self._lock:
            request_pages = self._requests.get(request_id, [])
            remaining = len(values)
            available = (
                self.page_size - len(self._pages[request_pages[-1]])
                if request_pages
                else 0
            )
            remaining = max(0, remaining - available)
            needed = (remaining + self.page_size - 1) // self.page_size
            if len(self._pages) + needed > self.max_pages:
                raise MemoryError("paged KV cache exhausted")
            request_pages = self._requests.setdefault(request_id, [])
            pending = list(values)
            while pending:
                if request_pages and len(self._pages[request_pages[-1]]) < self.page_size:
                    page = self._pages[request_pages[-1]]
                else:
                    page_id = self._next_page_id
                    self._next_page_id += 1
                    page = []
                    self._pages[page_id] = page
                    request_pages.append(page_id)
                room = self.page_size - len(page)
                page.extend(pending[:room])
                del pending[:room]
            return tuple(request_pages)

    def read(self, request_id: str) -> list[Any]:
        with self._lock:
            return [
                value
                for page_id in self._requests.get(request_id, [])
                for value in self._pages[page_id]
            ]

    def release(self, request_id: str) -> bool:
        with self._lock:
            pages = self._requests.pop(request_id, None)
            if pages is None:
                return False
            for page_id in pages:
                self._pages.pop(page_id, None)
            return True

    def report(self) -> dict[str, int]:
        with self._lock:
            return {
                "page_size": self.page_size,
                "allocated_pages": len(self._pages),
                "max_pages": self.max_pages,
                "active_requests": len(self._requests),
                "stored_values": sum(len(page) for page in self._pages.values()),
            }
