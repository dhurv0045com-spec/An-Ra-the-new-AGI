"""Deterministic Boolean-rule worlds and exact teaching utilities."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import math
from collections.abc import Sequence


@dataclass(frozen=True)
class RuleWorld:
    world_id: str
    family: str
    bits: int
    table: tuple[int, ...]


@lru_cache(maxsize=None)
def inputs(bits: int) -> tuple[tuple[int, ...], ...]:
    """Return bit vectors in integer order, least-significant bit first."""
    if bits < 1:
        raise ValueError("bits must be positive")
    return tuple(tuple((index >> bit) & 1 for bit in range(bits)) for index in range(1 << bits))


def _add_world(
    result: list[RuleWorld], seen: set[tuple[int, ...]], family: str, bits: int,
    description: str, table: tuple[int, ...],
) -> None:
    if table in seen or len(set(table)) == 1:
        return
    seen.add(table)
    digest = hashlib.sha256(bytes(table)).hexdigest()[:16]
    result.append(RuleWorld(f"{family}:{description}:{digest}", family, bits, table))


def make_worlds(bits: int = 6) -> list[RuleWorld]:
    """Generate all supported rules, deduplicated by Boolean semantics."""
    xs = inputs(bits)
    worlds: list[RuleWorld] = []
    seen: set[tuple[int, ...]] = set()

    # Every nonempty XOR subset and its complement.
    for mask in range(1, 1 << bits):
        for flip in (0, 1):
            table = tuple((sum(x[i] for i in range(bits) if mask >> i & 1) & 1) ^ flip for x in xs)
            _add_world(worlds, seen, "parity", bits, f"m{mask}-f{flip}", table)

    # Ternary encoding: absent, positive, or negated literal.
    for code in range(1, 3**bits):
        value = code
        literals: list[tuple[int, int]] = []
        for bit in range(bits):
            state, value = value % 3, value // 3
            if state:
                literals.append((bit, 1 if state == 1 else 0))
        table = tuple(int(all(x[bit] == wanted for bit, wanted in literals)) for x in xs)
        desc = ".".join(f"{bit}{'p' if wanted else 'n'}" for bit, wanted in literals)
        _add_world(worlds, seen, "conjunction", bits, desc, table)

    # At-least-k over every nonempty variable subset.
    for mask in range(1, 1 << bits):
        selected = tuple(i for i in range(bits) if mask >> i & 1)
        for threshold in range(1, len(selected) + 1):
            table = tuple(int(sum(x[i] for i in selected) >= threshold) for x in xs)
            _add_world(worlds, seen, "threshold", bits, f"m{mask}-k{threshold}", table)

    return sorted(worlds, key=lambda world: (world.family, world.world_id))


def split_worlds(worlds: Sequence[RuleWorld]) -> dict[str, list[RuleWorld]]:
    """Assign semantic truth tables to stable approximately 70/15/15 splits."""
    result: dict[str, list[RuleWorld]] = {"train": [], "dev": [], "test": []}
    for world in worlds:
        bucket = int.from_bytes(hashlib.sha256(bytes(world.table)).digest()[:8], "big") % 100
        split = "train" if bucket < 70 else "dev" if bucket < 85 else "test"
        result[split].append(world)
    for members in result.values():
        members.sort(key=lambda world: (world.family, world.world_id))
    return result


class Episode:
    """A query-limited episode that keeps the target label hidden."""

    def __init__(self, world: RuleWorld, target: int, budget: int):
        if not 0 <= target < len(world.table):
            raise ValueError("target is out of bounds")
        if budget < 0:
            raise ValueError("budget must be nonnegative")
        self.world = world
        self.target = target
        self.budget = budget
        self.history: list[tuple[int, int]] = []

    def legal_actions(self) -> tuple[int, ...]:
        queried = {action for action, _ in self.history}
        if len(self.history) >= self.budget:
            return ()
        return tuple(i for i in range(len(self.world.table)) if i != self.target and i not in queried)

    def observe(self, action: int) -> int:
        if not isinstance(action, int) or isinstance(action, bool):
            raise TypeError("action must be an integer")
        if not 0 <= action < len(self.world.table):
            raise IndexError("action is out of bounds")
        if action == self.target:
            raise ValueError("the target cannot be queried")
        if any(prior == action for prior, _ in self.history):
            raise ValueError("an action cannot be repeated")
        if len(self.history) >= self.budget:
            raise RuntimeError("query budget exhausted")
        observation = self.world.table[action]
        self.history.append((action, observation))
        return observation


def posterior_indices(hypotheses: Sequence[RuleWorld], history: Sequence[tuple[int, int]]) -> list[int]:
    """Return hypotheses consistent with every observed action/label pair."""
    return [
        index for index, world in enumerate(hypotheses)
        if all(0 <= action < len(world.table) and world.table[action] == label for action, label in history)
    ]


def target_probability(
    hypotheses: Sequence[RuleWorld], history: Sequence[tuple[int, int]], target: int,
) -> float:
    posterior = posterior_indices(hypotheses, history)
    if not posterior:
        raise ValueError("history has an empty posterior")
    if not 0 <= target < len(hypotheses[posterior[0]].table):
        raise IndexError("target is out of bounds")
    try:
        positives = sum(hypotheses[index].table[target] for index in posterior)
    except IndexError as exc:
        raise IndexError("target is out of bounds") from exc
    return positives / len(posterior)


def _binary_entropy(probability: float) -> float:
    if probability in (0.0, 1.0):
        return 0.0
    return -probability * math.log2(probability) - (1.0 - probability) * math.log2(1.0 - probability)


def information_gains(
    hypotheses: Sequence[RuleWorld], history: Sequence[tuple[int, int]], target: int,
) -> dict[int, float]:
    """Expected reduction in target-label entropy for each legal query."""
    posterior = posterior_indices(hypotheses, history)
    if not posterior:
        raise ValueError("history has an empty posterior")
    worlds = [hypotheses[index] for index in posterior]
    if not 0 <= target < len(worlds[0].table):
        raise IndexError("target is out of bounds")
    used = {action for action, _ in history}
    prior_entropy = _binary_entropy(sum(world.table[target] for world in worlds) / len(worlds))
    gains: dict[int, float] = {}
    for action in range(len(worlds[0].table)):
        if action == target or action in used:
            continue
        expected_entropy = 0.0
        for observation in (0, 1):
            branch = [world for world in worlds if world.table[action] == observation]
            if branch:
                p_target = sum(world.table[target] for world in branch) / len(branch)
                expected_entropy += len(branch) / len(worlds) * _binary_entropy(p_target)
        gain = prior_entropy - expected_entropy
        gains[action] = 0.0 if abs(gain) < 1e-15 else gain
    return gains
