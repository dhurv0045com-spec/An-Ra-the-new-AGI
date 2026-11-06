"""Exposure-matched batching and a two-answer counterfactual objective."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import random
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    import torch

from .benchmark import Example
from .codec import digest, encode


@dataclass(frozen=True, slots=True)
class Method:
    batching: str
    contrast_weight: float = 0.0
    associative: bool = False


METHODS = {
    "reference": Method("reference"),
    "shuffled-ce": Method("shuffled"),
    "paired-ce": Method("paired"),
    "paired-contrast": Method("paired", 0.5),
    "associative": Method("paired", 0.5, True),
}


class CounterfactualBatches:
    """Each block contains exactly two exposures to every row in every arm."""

    def __init__(self, rows: Sequence[Example], batch_size: int):
        groups = defaultdict(list)
        for index, row in enumerate(rows):
            facts, _, query = row.visible.prompt.rpartition("?")
            if row.visible.family != "binding" or query not in ("X>", "Y>"):
                raise ValueError("counterfactual batching requires canonical binding pairs")
            token_ids = encode(row.truth.gold)
            if len(token_ids) != 1:
                raise ValueError("counterfactual batching requires one-token answers")
            groups[facts].append((index, query, token_ids[0]))

        pairs = []
        partners = {}
        targets = {}
        for members in groups.values():
            if (len(members) != 2 or len({row[1] for row in members}) != 2
                    or len({row[2] for row in members}) != 2):
                raise ValueError("each fact view requires two distinct queries and answers")
            (first, _, gold_first), (second, _, gold_second) = members
            pairs.append((first, second))
            partners[first], partners[second] = second, first
            targets[first], targets[second] = (gold_first, gold_second), (gold_second, gold_first)

        if not rows or len(rows) * 2 % batch_size or batch_size % 2:
            raise ValueError("two exposures per row must form complete even batches")
        self.pairs = tuple(pairs)
        self.partners = partners
        self.targets = targets
        self.batch_size = batch_size
        self.block_steps = 2 * len(rows) // batch_size

    def indices(self, seed: int, update: int, method: Method) -> tuple[int, ...]:
        block, offset = divmod(update, self.block_steps)
        rng = random.Random(int(digest(["counterfactual-block-v1", seed, block]), 16))
        pairs = list(self.pairs) * 2
        rng.shuffle(pairs)
        stream = []
        for first, second in pairs:
            stream.extend((second, first) if rng.getrandbits(1) else (first, second))
        if method.batching == "shuffled":
            rng.shuffle(stream)
        start = offset * self.batch_size
        return tuple(stream[start:start + self.batch_size])

    def colocated(self, indices: tuple[int, ...]) -> int:
        present = set(indices)
        return sum(self.partners[index] in present for index in indices)


def method_identity(method: Method) -> dict:
    return {"schema": "elendril-counterfactual-v1", **asdict(method),
            "exposures_per_block": 2, "contrast_margin": 1.0}


def answer_contrast(logits: torch.Tensor, eligible: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Apply a smooth margin at the pre-answer position, using the swapped answer as distractor."""
    import torch
    from torch.nn import functional as F

    answer_position = eligible.long().argmax(dim=1) - 1
    answer_logits = logits[torch.arange(logits.size(0), device=logits.device), answer_position]
    gold, distractor = answer_logits.gather(1, targets).unbind(dim=1)
    return F.softplus(1.0 + distractor - gold).mean()
