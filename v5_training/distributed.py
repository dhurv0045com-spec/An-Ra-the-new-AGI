"""Fail-closed metadata contract for distributed checkpoint shards.

The schema is deliberately independent of XLA, PyTorch, or a storage vendor.
It makes every rank's RNG, optimizer shard, cursor, and token contribution
explicit so a coordinator cannot silently restore an incomplete world.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Mapping


RANK_SCHEMA = "anra-v5-rank-checkpoint/v1"
DISTRIBUTED_SCHEMA = "anra-v5-distributed-checkpoint/v1"


def _assert_sha256(name: str, value: str | None) -> None:
    if value is None:
        return
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256")


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


@dataclass(frozen=True, slots=True)
class RankCheckpoint:
    schema: str
    rank: int
    world_size: int
    global_update: int
    token_contribution: int
    cursor_sha256: str
    rng_state_sha256: str
    optimizer_shard_sha256: str
    data_shard_identity: str
    collective_barrier_sha256: str

    def assert_valid(self) -> None:
        if self.schema != RANK_SCHEMA:
            raise ValueError("unsupported rank-checkpoint schema")
        if self.world_size <= 0 or not 0 <= self.rank < self.world_size:
            raise ValueError("rank is outside the declared world")
        if self.global_update < 0 or self.token_contribution < 0:
            raise ValueError("rank counters cannot be negative")
        if not self.data_shard_identity:
            raise ValueError("data shard identity is required")
        for name, value in (
            ("cursor", self.cursor_sha256),
            ("RNG state", self.rng_state_sha256),
            ("optimizer shard", self.optimizer_shard_sha256),
            ("collective barrier", self.collective_barrier_sha256),
        ):
            _assert_sha256(name, value)

    def canonical(self) -> dict[str, object]:
        self.assert_valid()
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DistributedCheckpoint:
    schema: str
    parent_checkpoint_sha256: str | None
    global_update: int
    global_tokens: int
    world_size: int
    topology: str
    ranks: tuple[RankCheckpoint, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "ranks", tuple(sorted(self.ranks, key=lambda item: item.rank)))

    def assert_valid(self) -> None:
        if self.schema != DISTRIBUTED_SCHEMA:
            raise ValueError("unsupported distributed-checkpoint schema")
        _assert_sha256("parent checkpoint", self.parent_checkpoint_sha256)
        if self.world_size <= 0 or self.global_update < 0 or self.global_tokens < 0:
            raise ValueError("distributed checkpoint counters are invalid")
        if not self.topology or len(self.ranks) != self.world_size:
            raise ValueError("distributed checkpoint has an incomplete rank set")
        ranks = sorted(self.ranks, key=lambda item: item.rank)
        if [item.rank for item in ranks] != list(range(self.world_size)):
            raise ValueError("distributed checkpoint must contain each rank exactly once")
        barrier_ids: set[str] = set()
        data_shards: set[str] = set()
        optimizer_shards: set[str] = set()
        total_tokens = 0
        for item in ranks:
            item.assert_valid()
            if item.world_size != self.world_size or item.global_update != self.global_update:
                raise ValueError("rank metadata disagrees with distributed checkpoint")
            barrier_ids.add(item.collective_barrier_sha256)
            if item.data_shard_identity in data_shards:
                raise ValueError("data shard identity is duplicated")
            if item.optimizer_shard_sha256 in optimizer_shards:
                raise ValueError("optimizer shard identity is duplicated")
            data_shards.add(item.data_shard_identity)
            optimizer_shards.add(item.optimizer_shard_sha256)
            total_tokens += item.token_contribution
        if len(barrier_ids) != 1:
            raise ValueError("ranks did not cross the same collective barrier")
        if total_tokens != self.global_tokens:
            raise ValueError("rank token contributions do not equal global tokens")

    def canonical(self) -> dict[str, object]:
        self.assert_valid()
        value = asdict(self)
        value["ranks"] = [item.canonical() for item in sorted(self.ranks, key=lambda item: item.rank)]
        return value

    def sha256(self) -> str:
        return hashlib.sha256(_canonical_json(self.canonical())).hexdigest()

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "DistributedCheckpoint":
        expected = {"schema", "parent_checkpoint_sha256", "global_update", "global_tokens", "world_size", "topology", "ranks"}
        if set(value) != expected:
            raise ValueError("distributed checkpoint fields do not match schema")
        ranks = tuple(RankCheckpoint(**item) for item in value["ranks"])
        checkpoint = cls(
            schema=str(value["schema"]),
            parent_checkpoint_sha256=value["parent_checkpoint_sha256"],
            global_update=int(value["global_update"]),
            global_tokens=int(value["global_tokens"]),
            world_size=int(value["world_size"]),
            topology=str(value["topology"]),
            ranks=ranks,
        )
        checkpoint.assert_valid()
        return checkpoint
