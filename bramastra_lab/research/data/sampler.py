"""Deterministic group sampler (B03).

Minibatches are assembled from whole counterfactual groups: a pair is never
split across minibatches, so the pair objective always sees both members.
The unpaired control uses the same examples with grouping ignored, shuffled
independently. The cursor is exact state: restoring it reproduces the next
group deterministically.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

from bramastra_lab.research.data.manifest import DatasetError, Example


@dataclass(frozen=True)
class SamplerState:
    epoch: int
    group_index: int
    groups_consumed: int
    examples_consumed: int

    def to_dict(self) -> dict[str, int]:
        return {
            "epoch": self.epoch, "group_index": self.group_index,
            "groups_consumed": self.groups_consumed,
            "examples_consumed": self.examples_consumed,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, int]) -> "SamplerState":
        for field in ("epoch", "group_index", "groups_consumed", "examples_consumed"):
            value = raw.get(field)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise DatasetError(f"sampler state field {field} must be a nonnegative integer")
        return cls(**raw)


class GroupSampler:
    """Yields minibatches of whole groups, deterministically, with exact counters."""

    def __init__(self, groups: Sequence[Sequence[Example]], *, batch_size: int, seed: int,
                 mode: str = "paired") -> None:
        if mode not in ("paired", "unpaired_control"):
            raise DatasetError("sampler mode must be 'paired' or 'unpaired_control'")
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise DatasetError("batch_size must be a positive integer")
        if not groups:
            raise DatasetError("cannot sample from an empty group list", status="DATA_NOT_READY")
        self.mode = mode
        self.batch_size = batch_size
        self.seed = seed
        if mode == "unpaired_control":
            # Same examples; grouping deliberately discarded for the control.
            flattened = [example for group in groups for example in group]
            self._units: list[tuple[Example, ...]] = [(example,) for example in flattened]
        else:
            for group in groups:
                if not group:
                    raise DatasetError("groups must be nonempty")
                if len({member.split for member in group}) != 1:
                    raise DatasetError("groups must stay within one split")
            self._units = [tuple(group) for group in groups]
        self._order: list[int] = []
        self._epoch = -1
        self._position = 0
        self.groups_consumed = 0
        self.examples_consumed = 0
        self._advance_epoch()

    def _advance_epoch(self) -> None:
        self._epoch += 1
        rng = random.Random(f"{self.seed}:{self._epoch}")
        order = list(range(len(self._units)))
        rng.shuffle(order)
        self._order = order
        self._position = 0

    def state(self) -> SamplerState:
        return SamplerState(epoch=self._epoch, group_index=self._position,
                            groups_consumed=self.groups_consumed,
                            examples_consumed=self.examples_consumed)

    def restore(self, state: SamplerState) -> None:
        """Restore exact sampling state and reproduce the next group."""
        if state.epoch < 0 or state.group_index < 0:
            raise DatasetError("sampler state positions must be nonnegative")
        self._epoch = state.epoch
        rng = random.Random(f"{self.seed}:{self._epoch}")
        order = list(range(len(self._units)))
        rng.shuffle(order)
        self._order = order
        self._position = state.group_index
        self.groups_consumed = state.groups_consumed
        self.examples_consumed = state.examples_consumed
        if self._position >= len(self._order):
            self._advance_epoch()

    def _next_batch(self) -> tuple[tuple[Example, ...], bool]:
        """Return (examples, epoch_rolled). Whole groups only."""
        collected: list[Example] = []
        rolled = False
        while len(collected) < self.batch_size:
            if self._position >= len(self._order):
                self._advance_epoch()
                rolled = True
                if not self._order:  # defensive: empty unit list cannot happen
                    raise DatasetError("sampler has no units")
                continue
            unit = self._units[self._order[self._position]]
            self._position += 1
            self.groups_consumed += 1
            self.examples_consumed += len(unit)
            collected.extend(unit)
            if self.mode == "paired" and len(unit) >= self.batch_size:
                break
        return tuple(collected), rolled

    def take_batch(self) -> tuple[Example, ...]:
        """Return the next minibatch, keeping every group whole."""
        batch, _ = self._next_batch()
        return batch

    def batches_per_epoch(self) -> int:
        """Lower bound on whole-group batches per epoch under this batch size."""
        units_per_batch = max(1, self.batch_size)
        return max(1, (len(self._units) + units_per_batch - 1) // units_per_batch)

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def total_examples(self) -> int:
        return sum(len(unit) for unit in self._units)
