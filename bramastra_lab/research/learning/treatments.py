"""Training-only logit treatments (B04).

Treatments reshape softmax competition during training only. The model always
emits full-vocabulary logits; evaluation and primary free generation always
use the unmodified full vocabulary. Treatment arguments come only from the
declared training schema and tokenizer identities. An inconsistent batch is
rejected, never silently re-targeted.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import torch

from bramastra_lab.research.config import BYTE_TOKENIZER_NAME, tokenizer_identity

TREATMENTS = frozenset({"full", "participating_mask", "inactive_offset"})


class TreatmentError(ValueError):
    """A treatment argument or batch is inconsistent with the declared schema."""


@dataclass(frozen=True)
class TreatmentSchema:
    """The declared set of token classes allowed to compete during training.

    ``participating`` must cover every structural token the task requires
    (including EOS) plus the task schema's answer classes. It is derived from
    reviewed task-level schemas, never from the current minibatch's difficult
    competitors.
    """

    schema_id: str
    tokenizer_identity: str = BYTE_TOKENIZER_NAME
    participating: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.schema_id, str) or not self.schema_id.strip():
            raise TreatmentError("schema_id must be a nonempty string")
        if self.tokenizer_identity != BYTE_TOKENIZER_NAME:
            raise TreatmentError("schema tokenizer identity does not match the declared tokenizer")
        if not self.participating:
            raise TreatmentError("participating set must be nonempty")
        if any(not isinstance(token, int) or isinstance(token, bool) or token < 0
               for token in self.participating):
            raise TreatmentError("participating tokens must be nonnegative integers")
        if len(set(self.participating)) != len(self.participating):
            raise TreatmentError("participating tokens must be unique")

    def validated_for_vocab(self, vocab: int) -> "TreatmentSchema":
        if any(token >= vocab for token in self.participating):
            raise TreatmentError("participating tokens must be inside the physical vocabulary")
        return self

    def covers(self, labels: Iterable[int]) -> bool:
        return all(label in self.participating for label in labels)


def validate_targets_in_schema(labels: torch.Tensor, loss_mask: torch.Tensor,
                               schema: TreatmentSchema) -> None:
    """Every supervised target must participate. Inconsistent batches reject.

    Masking can never remove a valid gold class: if a supervised position
    carries a label outside the schema, that is a schema/batch contradiction
    and the batch is rejected instead of losing the target.
    """
    supervised_labels = labels[loss_mask]
    if bool((supervised_labels < 0).any()):
        raise TreatmentError("supervised positions must carry nonnegative labels")
    outside = [int(label) for label in supervised_labels.unique().tolist()
               if label not in schema.participating]
    if outside:
        raise TreatmentError(
            f"supervised targets outside the declared participating schema: {sorted(outside)[:8]}")


def _inactive_offset(vocab: int, participating: int, effective_vocab: int) -> float:
    active = participating
    if not active < effective_vocab <= vocab:
        raise TreatmentError(
            f"inactive_offset requires |A| < K <= V; got |A|={active}, K={effective_vocab}, "
            f"V={vocab}")
    offset = math.log((vocab - active) / (effective_vocab - active))
    if not math.isfinite(offset) or offset < 0:
        raise TreatmentError("computed inactive offset must be finite and nonnegative")
    return offset


def apply_treatment(logits: torch.Tensor, schema: TreatmentSchema, treatment: str,
                    *, effective_vocab: int | None = None) -> torch.Tensor:
    """Return logits reshaped for training competition; input tensor is untouched.

    - ``full``: unchanged logits (the compatibility control).
    - ``participating_mask``: non-participating classes are excluded from
      competition (-inf).
    - ``inactive_offset``: non-participating classes are shifted down by
      ``log((V-|A|)/(K-|A|))``, reducing their competition as if the physical
      vocabulary were the declared effective size K.
    """
    if treatment not in TREATMENTS:
        raise TreatmentError(f"unknown treatment {treatment!r}")
    if treatment == "full":
        return logits
    vocab = logits.shape[-1]
    participating = torch.zeros(vocab, dtype=torch.bool)
    participating[list(schema.participating)] = True
    if treatment == "participating_mask":
        treated = logits.clone()
        treated.masked_fill_(~participating, -math.inf)
        return treated
    # inactive_offset
    if effective_vocab is None:
        raise TreatmentError("inactive_offset requires an effective_vocab K")
    offset = _inactive_offset(vocab, len(schema.participating), effective_vocab)
    treated = logits.clone()
    inactive = ~participating
    treated[..., inactive] = treated[..., inactive] - offset
    return treated
