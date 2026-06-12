"""Inference-only prefix cache keyed by model and token sequence."""

from __future__ import annotations

from collections import OrderedDict
import hashlib
from typing import Any


class PrefixCache:
    def __init__(self, max_entries: int = 32) -> None:
        self.max_entries = int(max_entries)
        self._entries: OrderedDict[str, Any] = OrderedDict()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key(model_id: str, token_ids: list[int] | tuple[int, ...]) -> str:
        payload = f"{model_id}:" + ",".join(str(token) for token in token_ids)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, model_id: str, token_ids: list[int] | tuple[int, ...]) -> Any | None:
        key = self.key(model_id, token_ids)
        value = self._entries.get(key)
        if value is None:
            self.misses += 1
            return None
        self.hits += 1
        self._entries.move_to_end(key)
        return value

    def put(self, model_id: str, token_ids: list[int] | tuple[int, ...], value: Any) -> None:
        key = self.key(model_id, token_ids)
        self._entries[key] = value
        self._entries.move_to_end(key)
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)

    def report(self) -> dict[str, float | int]:
        total = self.hits + self.misses
        return {
            "entries": len(self._entries),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / total if total else 0.0,
        }
