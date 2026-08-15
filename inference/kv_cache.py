"""Tiered KV retention with bounded arousal-weighted eviction priority."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class KVSegment:
    key: torch.Tensor
    value: torch.Tensor
    token_start: int
    salience: float
    identity_critical: bool = False

    @property
    def tokens(self) -> int:
        return int(self.key.shape[-2])


class TieredKVCache:
    def __init__(self, max_tokens: int, arousal_weight_cap: float = 0.25) -> None:
        self.max_tokens = int(max_tokens)
        self.arousal_weight_cap = float(arousal_weight_cap)
        self.layers: dict[int, list[KVSegment]] = {}

    def update(
        self,
        layer: int,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        token_start: int,
        salience: float = 0.0,
        arousal: float = 0.0,
        identity_critical: bool = False,
    ) -> None:
        bounded_arousal = max(
            0.0, min(self.arousal_weight_cap, float(arousal) * self.arousal_weight_cap)
        )
        priority = max(0.0, min(1.0, float(salience))) + bounded_arousal
        self.layers.setdefault(int(layer), []).append(
            KVSegment(
                key=key.detach(),
                value=value.detach(),
                token_start=int(token_start),
                salience=priority,
                identity_critical=bool(identity_critical),
            )
        )
        self._evict(int(layer))

    def _evict(self, layer: int) -> None:
        segments = self.layers[layer]
        while sum(segment.tokens for segment in segments) > self.max_tokens:
            candidates = [
                (index, segment)
                for index, segment in enumerate(segments)
                if not segment.identity_critical
            ]
            if not candidates:
                break
            index, _ = min(candidates, key=lambda item: (item[1].salience, item[1].token_start))
            segments.pop(index)

    def get(self, layer: int) -> tuple[torch.Tensor, torch.Tensor] | None:
        segments = sorted(self.layers.get(int(layer), []), key=lambda item: item.token_start)
        if not segments:
            return None
        return (
            torch.cat([segment.key for segment in segments], dim=-2),
            torch.cat([segment.value for segment in segments], dim=-2),
        )

    def clear(self) -> None:
        self.layers.clear()
