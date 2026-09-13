"""Deterministic stratified replay (B07).

Replay samples whole accepted episodes through per-family cyclic cursors over
seeded shuffles — an exact sampling state that survives checkpoints. Replays
increase presentation counters, never unique example counts. Declared replay
proportions are reconciled against exact consumed counters and shortfalls are
reported, never silently absorbed.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from bramastra_lab.research.experience.ledger import EpisodeReceipt


class ReplayStateError(ValueError):
    """Replay state or configuration is inconsistent."""


@dataclass(frozen=True)
class ReplayBatch:
    entries: tuple[EpisodeReceipt, ...]
    shortfall: int          # requested minus delivered; must be surfaced
    epoch: int
    is_replay: bool = True  # presentations from replay never count as new


class ReplayEngine:
    """Cyclic stratified sampler over ledger receipts with exact counters.

    The engine binds an explicit ``dataset_identity`` (typically the ledger
    content identity). A saved replay state can only be restored against the
    same identity: changed dataset contents, family weights or replay policy
    force a new identity and can never silently reuse old cursors (W11).
    """

    def __init__(self, entries: Sequence[EpisodeReceipt], *, family_weights: Mapping[str, float],
                 batch_size: int, seed: int, dataset_identity: str,
                 allowed_quality: Sequence[str] = ("accepted",)) -> None:
        if batch_size <= 0:
            raise ReplayStateError("batch_size must be positive")
        if not isinstance(dataset_identity, str) or not dataset_identity:
            raise ReplayStateError("dataset_identity must be a nonempty string")
        self.dataset_identity = dataset_identity
        self.batch_size = batch_size
        self.seed = seed
        self.allowed_quality = tuple(allowed_quality)
        self.family_weights = dict(family_weights)
        for family, weight in self.family_weights.items():
            if not isinstance(weight, (int, float)) or weight <= 0:
                raise ReplayStateError(f"family weight for {family!r} must be positive")
        seen_content: set[str] = set()
        self._pools: dict[str, list[EpisodeReceipt]] = {}
        for receipt in entries:
            if receipt.quality not in self.allowed_quality:
                continue
            if receipt.episode_content_identity in seen_content:
                continue  # overlapping manifests must not inflate unique supply
            seen_content.add(receipt.episode_content_identity)
            self._pools.setdefault(receipt.family, []).append(receipt)
        from bramastra_lab.research.contracts.core import content_identity

        self.identity = content_identity({
            "dataset_identity": dataset_identity,
            "family_weights": dict(sorted(self.family_weights.items())),
            "batch_size": batch_size, "seed": seed,
            "allowed_quality": list(self.allowed_quality),
        })
        self._orders: dict[str, list[int]] = {}
        self._cursors: dict[str, int] = {}
        self._epochs: dict[str, int] = {}
        for family, pool in self._pools.items():
            rng = random.Random(f"{seed}:{family}:0")
            order = list(range(len(pool)))
            rng.shuffle(order)
            self._orders[family] = order
            self._cursors[family] = 0
            self._epochs[family] = 0
        self.consumed_total = 0
        self.consumed_by_family: dict[str, int] = {family: 0 for family in self._pools}
        self.planned_total = 0

    @property
    def families(self) -> tuple[str, ...]:
        return tuple(sorted(self._pools))

    def _roll_epoch(self, family: str) -> None:
        self._epochs[family] += 1
        rng = random.Random(f"{self.seed}:{family}:{self._epochs[family]}")
        order = list(range(len(self._pools[family])))
        rng.shuffle(order)
        self._orders[family] = order
        self._cursors[family] = 0

    def _draw(self, family: str, count: int) -> list[EpisodeReceipt]:
        """Draw up to ``count`` fresh entries; never re-wraps mid-request.

        A request consumes each stored episode at most once; the epoch rolls
        on the next request. Entries a request cannot supply fresh become an
        explicit shortfall instead of silent repetition.
        """
        pool = self._pools[family]
        if not pool or count <= 0:
            return []
        if self._cursors[family] >= len(self._orders[family]):
            self._roll_epoch(family)
        order = self._orders[family]
        cursor = self._cursors[family]
        take = min(count, len(order) - cursor)
        delivered = [pool[order[cursor + index]] for index in range(take)]
        self._cursors[family] = cursor + take
        return delivered

    def sample(self, count: int | None = None) -> ReplayBatch:
        """Draw the next replay batch, weighted by declared family weights.

        With fewer available accepted entries than requested, the delivered
        batch carries an explicit ``shortfall`` instead of silently repeating
        within an epoch or inflating the count.
        """
        requested = self.batch_size if count is None else count
        if requested <= 0:
            raise ReplayStateError("requested replay count must be positive")
        weighted: list[str] = []
        families = self.families
        if not families:
            return ReplayBatch(entries=(), shortfall=requested, epoch=0)
        for family in families:
            weighted.extend([family] * int(self.family_weights.get(family, 1.0)))
        plan: list[tuple[str, int]] = []
        remaining = requested
        for index, family in enumerate(weighted):
            if remaining <= 0:
                break
            share = requested // len(weighted) if index < len(weighted) - 1 else remaining
            share = min(max(share, 0), remaining)
            if share:
                plan.append((family, share))
                remaining -= share
        entries: list[EpisodeReceipt] = []
        for family, share in plan:
            drawn = self._draw(family, share)
            entries.extend(drawn)
            self.consumed_by_family[family] = self.consumed_by_family.get(family, 0) + len(drawn)
            self.consumed_total += len(drawn)
        # A family short on data reduces the delivered batch explicitly.
        shortfall = requested - len(entries)
        epoch = min(self._epochs.values()) if self._epochs else 0
        return ReplayBatch(entries=tuple(entries), shortfall=shortfall, epoch=epoch)

    def declare_planned(self, count: int) -> None:
        """Record the schedule's planned replay volume for reconciliation."""
        if count < 0:
            raise ReplayStateError("planned count must be nonnegative")
        self.planned_total += count

    def reconciliation(self) -> dict[str, Any]:
        """Exact consumed counters versus the declared plan."""
        return {
            "planned": self.planned_total,
            "consumed": self.consumed_total,
            "shortfall": max(0, self.planned_total - self.consumed_total),
            "by_family": dict(self.consumed_by_family),
        }

    def state(self) -> dict[str, Any]:
        return {
            "replay_identity": self.identity,
            "cursors": dict(self._cursors), "epochs": dict(self._epochs),
            "consumed_total": self.consumed_total,
            "consumed_by_family": dict(self.consumed_by_family),
            "planned_total": self.planned_total,
            "orders": {family: list(order) for family, order in self._orders.items()},
        }

    def restore(self, state: Mapping[str, Any]) -> None:
        if state.get("replay_identity") != self.identity:
            raise ReplayStateError(
                "replay state identity does not match this engine; changed dataset "
                "contents, weights or replay policy require a new replay identity "
                "and cannot silently reuse old cursors")
        for family, cursor in state["cursors"].items():
            if family not in self._pools:
                raise ReplayStateError(f"restored cursor names unknown family {family!r}")
            if cursor < 0:
                raise ReplayStateError("cursor positions must be nonnegative")
            # Cursor rollback protection: a restored cursor beyond the pool's
            # shuffled order is invalid rather than silently wrapped.
            if cursor > len(self._orders[family]):
                raise ReplayStateError(f"cursor for {family!r} exceeds its epoch order")
        self._cursors = dict(state["cursors"])
        self._epochs = dict(state["epochs"])
        self.consumed_total = state["consumed_total"]
        self.consumed_by_family = dict(state["consumed_by_family"])
        self.planned_total = state["planned_total"]
        self._orders = {family: list(order) for family, order in state["orders"].items()}


class ReplaySchedule:
    """Persisted fractional replay-slot scheduler (B2.2 R3).

    A slot fires when accumulated credit crosses one whole slot, computed in
    exact rational arithmetic, so a proportion like 0.3 produces the same
    deterministic fire pattern in an uninterrupted run and in any split run
    that persists the credit. Scheduling is a function of committed optimizer
    updates, never of loop-local indexes.
    """

    schema = "bramastra-replay-schedule/v1"

    def __init__(self, proportion: float, *, credit_num: int = 0,
                 slots_fired: int = 0) -> None:
        from fractions import Fraction

        if not 0.0 < proportion <= 1.0:
            raise ReplayStateError("replay proportion must lie in (0, 1]")
        fraction = Fraction(proportion).limit_denominator(1000)
        self.numerator = int(fraction.numerator)
        self.denominator = int(fraction.denominator)
        if credit_num < 0 or slots_fired < 0:
            raise ReplayStateError("replay schedule state must be nonnegative")
        if credit_num >= self.denominator:
            raise ReplayStateError("replay credit must be smaller than one slot")
        self.credit_num = credit_num
        self.slots_fired = slots_fired

    def slot_due(self) -> bool:
        """Advance one decision; True when this update replays."""
        self.credit_num += self.numerator
        if self.credit_num >= self.denominator:
            self.credit_num -= self.denominator
            self.slots_fired += 1
            return True
        return False

    def peek(self) -> bool:
        """Non-mutating preview of the next decision (for trace comparison)."""
        return self.credit_num + self.numerator >= self.denominator

    def state(self) -> dict[str, Any]:
        return {"schema": self.schema,
                "proportion": self.numerator / self.denominator,
                "credit_num": self.credit_num, "denominator": self.denominator,
                "slots_fired": self.slots_fired}

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "ReplaySchedule":
        if state.get("schema") != cls.schema:
            raise ReplayStateError(
                f"replay schedule schema mismatch: {state.get('schema')!r}")
        if int(state.get("denominator", 0)) != cls._denominator_for(
                float(state.get("proportion", 0.0))):
            raise ReplayStateError(
                "replay proportion does not match its recorded denominator; "
                "changed policy requires a new schedule identity")
        return cls(float(state["proportion"]), credit_num=int(state["credit_num"]),
                   slots_fired=int(state["slots_fired"]))

    @staticmethod
    def _denominator_for(proportion: float) -> int:
        from fractions import Fraction

        return int(Fraction(proportion).limit_denominator(1000).denominator)
