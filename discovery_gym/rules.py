from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Iterable

from .contracts import canonical_json


@dataclass(frozen=True, slots=True)
class RuleSpec:
    structure_id: str
    variant: int
    parents: tuple[int, ...]
    threshold: int = 0
    flip: bool = False
    swap_branches: bool = False

    def __post_init__(self) -> None:
        if self.structure_id not in STRUCTURE_IDS:
            raise ValueError(f"unknown causal structure: {self.structure_id}")
        if self.variant < 0:
            raise ValueError("variant must be nonnegative")
        if len(self.parents) < 2 or len(set(self.parents)) != len(self.parents):
            raise ValueError("causal rules require distinct parents")
        if any(parent < 0 or parent > 2 for parent in self.parents):
            raise ValueError("parents must refer to the three public variables")
        if self.structure_id == "threshold2" and self.threshold not in (1, 2):
            raise ValueError("threshold2 requires threshold 1 or 2")
        if self.structure_id == "neg_threshold3" and self.threshold not in (1, 2):
            raise ValueError("neg_threshold3 requires threshold 1 or 2")

    @property
    def graph_signature(self) -> str:
        return f"{self.structure_id}:{','.join(str(item) for item in self.parents)}"

    @property
    def semantic_signature(self) -> str:
        return canonical_json(
            {
                "structure_id": self.structure_id,
                "variant": self.variant,
                "parents": list(self.parents),
                "threshold": self.threshold,
                "flip": self.flip,
                "swap_branches": self.swap_branches,
            }
        )

    @property
    def semantic_sha256(self) -> str:
        return hashlib.sha256(self.semantic_signature.encode("utf-8")).hexdigest()

    def evaluate(self, assignment: Iterable[int]) -> int:
        values = tuple(assignment)
        if len(values) != 3 or any(
            isinstance(value, bool) or not isinstance(value, int) or value not in (0, 1)
            for value in values
        ):
            raise ValueError("assignment must contain three binary integers")
        first, second, third = values
        if self.structure_id == "xor2":
            result = first ^ second
        elif self.structure_id == "and2":
            result = first & second
        elif self.structure_id == "or2":
            result = first | second
        elif self.structure_id == "threshold2":
            result = int(first + second >= self.threshold)
        elif self.structure_id == "xor3":
            result = first ^ second ^ third
        elif self.structure_id == "and3":
            result = first & second & third
        elif self.structure_id == "or3":
            result = first | second | third
        elif self.structure_id == "majority3":
            result = int(first + second + third >= 2)
        elif self.structure_id == "nested":
            result = first ^ (second & third)
        elif self.structure_id == "mux":
            if self.swap_branches:
                result = third if first else second
            else:
                result = second if first else third
        elif self.structure_id == "xor_and":
            result = (first ^ second) & third
        elif self.structure_id == "neg_threshold3":
            result = int(first + second + third <= self.threshold)
        else:
            raise ValueError(f"unsupported causal structure: {self.structure_id}")
        return int(result) ^ int(self.flip)

    def truth_table(self) -> tuple[int, ...]:
        return tuple(
            self.evaluate(((index >> 2) & 1, (index >> 1) & 1, index & 1))
            for index in range(8)
        )


STRUCTURE_IDS = (
    "xor2",
    "and2",
    "or2",
    "threshold2",
    "xor3",
    "and3",
    "or3",
    "majority3",
    "nested",
    "mux",
    "xor_and",
    "neg_threshold3",
)

STRUCTURE_SPLITS = {
    "train": ("xor2", "and2", "threshold2"),
    "development": ("or2", "xor3", "and3"),
    "confirmation": ("or3", "majority3", "nested"),
    "transfer": ("mux", "xor_and", "neg_threshold3"),
}


def _catalog() -> tuple[RuleSpec, ...]:
    values: list[RuleSpec] = []
    for structure_id in STRUCTURE_IDS:
        parents = (0, 1) if structure_id in {"xor2", "and2", "or2", "threshold2"} else (0, 1, 2)
        for variant in range(2):
            values.append(
                RuleSpec(
                    structure_id=structure_id,
                    variant=variant,
                    parents=parents,
                    threshold=(variant + 1) if structure_id in {"threshold2", "neg_threshold3"} else 0,
                    flip=bool(variant),
                    swap_branches=bool(variant) if structure_id == "mux" else False,
                )
            )
    return tuple(values)


CATALOG = _catalog()
CATALOG_BY_STRUCTURE = {
    structure_id: tuple(item for item in CATALOG if item.structure_id == structure_id)
    for structure_id in STRUCTURE_IDS
}


def catalog_digest() -> str:
    return hashlib.sha256(canonical_json([item.semantic_signature for item in CATALOG]).encode("utf-8")).hexdigest()


def evaluate_rule(spec: RuleSpec, assignment: Iterable[int]) -> int:
    return spec.evaluate(assignment)


def all_assignments() -> tuple[tuple[int, int, int], ...]:
    return tuple(((index >> 2) & 1, (index >> 1) & 1, index & 1) for index in range(8))


def overridden_assignment(context: Iterable[int], variable: int, value: int) -> tuple[int, int, int]:
    values = list(context)
    if len(values) != 3 or any(
        isinstance(item, bool) or not isinstance(item, int) or item not in (0, 1)
        for item in values
    ):
        raise ValueError("context must contain three binary integers")
    if isinstance(variable, bool) or not isinstance(variable, int) or variable not in (0, 1, 2):
        raise ValueError("intervention variable must be an integer index")
    if isinstance(value, bool) or not isinstance(value, int) or value not in (0, 1):
        raise ValueError("intervention value must be a binary integer")
    values[variable] = value
    return tuple(values)
