"""Versioned full-resume metadata for replicated distributed training.

The v1 distributed record describes independently sharded optimizer state and
is kept unchanged. This v2 contract models replicated model/Adam checkpoints
with rank-local RNG and sampler-cursor payloads. A record is useful for resume
only when those payload bytes are stored in the matching transaction and a
restored worker verifies its next-batch fingerprint.
"""

from __future__ import annotations

import base64
from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Mapping, Sequence


DISTRIBUTED_CHECKPOINT_SCHEMA = "anra-v5-distributed-checkpoint/v2"
DISTRIBUTED_RANK_STATE_SCHEMA = "anra-v5-distributed-rank-state/v2"
DISTRIBUTED_MANIFEST_SCHEMA = "anra-v5-checkpoint-transaction/v2"
RANK_STATE_BUNDLE_SCHEMA = "anra-v5-rank-state-payload-bundle/v1"
DISTRIBUTED_COMPONENTS = frozenset({"distributed.json", "rank_states.bin"})
RANK_STATE_FIELDS = frozenset({"rng_state", "cursor_state"})
COLLECTIVE_BARRIER_SCHEMA = "anra-v5-collective-barrier/v1"


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def _sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _payload_hash(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def rank_state_payload_sha256(*, rng_state: bytes, cursor_state: bytes) -> str:
    """Bind the two opaque per-rank continuation payloads as one identity."""

    return hashlib.sha256(_canonical_json({
        "rng_state_sha256": _payload_hash(rng_state),
        "cursor_state_sha256": _payload_hash(cursor_state),
    })).hexdigest()


def aggregate_rng_state_sha256(rng_state_sha256_by_rank: Sequence[str]) -> str:
    """Bind an ordered sequence of rank RNG identities into one run identity."""

    if any(not _sha256(value) for value in rng_state_sha256_by_rank):
        raise ValueError("rank RNG identities must be lowercase SHA-256 values")
    return hashlib.sha256(_canonical_json({
        "schema": "anra-v5-rank-rng-aggregate/v1",
        "ranks": [
            {"rank": rank, "rng_state_sha256": value}
            for rank, value in enumerate(rng_state_sha256_by_rank)
        ],
    })).hexdigest()


def aggregate_rank_rng_sha256(
    ranks: tuple["DistributedRankState", ...] | list["DistributedRankState"],
) -> str:
    """Bind the shared training-state RNG identity to ordered rank RNGs."""

    ordered = sorted(ranks, key=lambda row: row.rank)
    return aggregate_rng_state_sha256([row.rng_state_sha256 for row in ordered])


def collective_barrier_sha256(
    *, global_update: int, world_size: int, topology: str,
    ranks: tuple["DistributedRankState", ...] | list["DistributedRankState"],
) -> str:
    """Derive a barrier identity from its update, topology, and rank receipts."""

    return hashlib.sha256(_canonical_json({
        "schema": COLLECTIVE_BARRIER_SCHEMA,
        "global_update": global_update,
        "world_size": world_size,
        "topology": topology,
        "ranks": [
            {"rank": item.rank, "collective_receipt_sha256": item.collective_receipt_sha256}
            for item in sorted(ranks, key=lambda row: row.rank)
        ],
    })).hexdigest()


def encode_rank_state_bundle(
    rank_states: Mapping[int, Mapping[str, bytes]], *, world_size: int,
) -> bytes:
    """Serialize every rank's opaque RNG/cursor bytes in a canonical envelope."""

    if type(world_size) is not int or world_size <= 0:
        raise ValueError("rank-state bundle world size must be positive")
    if any(type(rank) is not int for rank in rank_states):
        raise ValueError("rank-state bundle ranks must be integers")
    if set(rank_states) != set(range(world_size)):
        raise ValueError("rank-state bundle must include every rank exactly once")
    rows: list[dict[str, object]] = []
    for rank in range(world_size):
        payloads = rank_states[rank]
        if set(payloads) != RANK_STATE_FIELDS:
            raise ValueError("rank-state payload needs exactly RNG and cursor bytes")
        if any(not isinstance(payloads[name], bytes) or not payloads[name]
               for name in RANK_STATE_FIELDS):
            raise ValueError("rank-state payload bytes must be non-empty")
        rows.append({
            "rank": rank,
            "rng_state_base64": base64.b64encode(payloads["rng_state"]).decode("ascii"),
            "cursor_state_base64": base64.b64encode(payloads["cursor_state"]).decode("ascii"),
        })
    return _canonical_json({
        "schema": RANK_STATE_BUNDLE_SCHEMA,
        "world_size": world_size,
        "ranks": rows,
    })


def decode_rank_state_bundle(payload: bytes, *, world_size: int) -> dict[int, dict[str, bytes]]:
    """Decode a bundle and reject malformed, duplicate, or missing ranks."""

    if type(world_size) is not int or world_size <= 0:
        raise ValueError("rank-state bundle world size must be positive")
    try:
        document = json.loads(payload)
    except (TypeError, ValueError) as exc:
        raise ValueError("distributed rank-state bundle is not valid JSON") from exc
    if (
        not isinstance(document, dict)
        or set(document) != {"schema", "world_size", "ranks"}
        or document.get("schema") != RANK_STATE_BUNDLE_SCHEMA
        or type(document.get("world_size")) is not int
        or document["world_size"] != world_size
        or not isinstance(document.get("ranks"), list)
    ):
        raise ValueError("distributed rank-state bundle identity is invalid")
    rows = document["ranks"]
    if len(rows) != world_size:
        raise ValueError("distributed rank-state bundle has an incomplete rank set")
    result: dict[int, dict[str, bytes]] = {}
    for row in rows:
        if (
            not isinstance(row, dict)
            or set(row) != {"rank", "rng_state_base64", "cursor_state_base64"}
            or type(row.get("rank")) is not int
            or row["rank"] in result
            or not isinstance(row.get("rng_state_base64"), str)
            or not isinstance(row.get("cursor_state_base64"), str)
        ):
            raise ValueError("distributed rank-state bundle row is invalid")
        try:
            rng_state = base64.b64decode(row["rng_state_base64"], validate=True)
            cursor_state = base64.b64decode(row["cursor_state_base64"], validate=True)
        except (ValueError, TypeError) as exc:
            raise ValueError("distributed rank-state bundle has invalid base64") from exc
        if not rng_state or not cursor_state:
            raise ValueError("distributed rank-state payloads cannot be empty")
        result[row["rank"]] = {"rng_state": rng_state, "cursor_state": cursor_state}
    if set(result) != set(range(world_size)):
        raise ValueError("distributed rank-state bundle must include every rank exactly once")
    return result


@dataclass(frozen=True, slots=True)
class DistributedRankState:
    schema: str
    rank: int
    world_size: int
    global_update: int
    cumulative_tokens: int
    data_shard_identity: str
    rng_state_sha256: str
    cursor_state_sha256: str
    rank_state_sha256: str
    model_state_sha256: str
    optimizer_state_sha256: str
    next_batch_sha256: str
    collective_receipt_sha256: str

    def assert_valid(self) -> None:
        if self.schema != DISTRIBUTED_RANK_STATE_SCHEMA:
            raise ValueError("unsupported distributed rank-state schema")
        if type(self.rank) is not int or type(self.world_size) is not int:
            raise ValueError("distributed rank and world size must be integers")
        if self.world_size <= 0 or not 0 <= self.rank < self.world_size:
            raise ValueError("distributed rank is outside the declared world")
        if (
            type(self.global_update) is not int
            or type(self.cumulative_tokens) is not int
            or min(self.global_update, self.cumulative_tokens) < 0
        ):
            raise ValueError("distributed rank counters cannot be negative")
        if not isinstance(self.data_shard_identity, str) or not self.data_shard_identity:
            raise ValueError("rank-local data-shard identity is required")
        for name, value in (
            ("RNG state", self.rng_state_sha256),
            ("cursor state", self.cursor_state_sha256),
            ("rank state", self.rank_state_sha256),
            ("model state", self.model_state_sha256),
            ("optimizer state", self.optimizer_state_sha256),
            ("next batch", self.next_batch_sha256),
            ("collective receipt", self.collective_receipt_sha256),
        ):
            if not _sha256(value):
                raise ValueError(f"distributed {name} identity must be a lowercase SHA-256")
        expected = hashlib.sha256(_canonical_json({
            "rng_state_sha256": self.rng_state_sha256,
            "cursor_state_sha256": self.cursor_state_sha256,
        })).hexdigest()
        if self.rank_state_sha256 != expected:
            raise ValueError("rank-state hash does not bind the RNG and cursor hashes")

    def canonical(self) -> dict[str, object]:
        self.assert_valid()
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ReplicatedTrainingCheckpoint:
    """One shared model/optimizer state plus every rank's continuation state."""

    schema: str
    parent_checkpoint_sha256: str | None
    global_update: int
    global_tokens: int
    world_size: int
    topology: str
    optimizer_layout: str
    training_state_sha256: str
    model_state_sha256: str
    optimizer_state_sha256: str
    collective_barrier_sha256: str
    ranks: tuple[DistributedRankState, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "ranks", tuple(sorted(self.ranks, key=lambda item: item.rank)))

    def assert_valid(
        self,
        *,
        training_state: Any | None = None,
        model_payload: bytes | None = None,
        optimizer_payload: bytes | None = None,
        rank_state_payload: bytes | None = None,
    ) -> None:
        if self.schema != DISTRIBUTED_CHECKPOINT_SCHEMA:
            raise ValueError("unsupported replicated distributed-checkpoint schema")
        if self.parent_checkpoint_sha256 is not None and not _sha256(self.parent_checkpoint_sha256):
            raise ValueError("distributed checkpoint parent is not a lowercase SHA-256")
        if (
            type(self.global_update) is not int
            or type(self.global_tokens) is not int
            or type(self.world_size) is not int
            or min(self.global_update, self.global_tokens) < 0
            or self.world_size <= 0
        ):
            raise ValueError("distributed checkpoint counters are invalid")
        if not isinstance(self.topology, str) or not self.topology:
            raise ValueError("distributed checkpoint topology is required")
        if self.optimizer_layout != "replicated":
            raise ValueError("v2 distributed checkpoints require replicated optimizer state")
        for name, value in (
            ("training state", self.training_state_sha256),
            ("model state", self.model_state_sha256),
            ("optimizer state", self.optimizer_state_sha256),
            ("collective barrier", self.collective_barrier_sha256),
        ):
            if not _sha256(value):
                raise ValueError(f"distributed {name} must be a lowercase SHA-256")
        if len(self.ranks) != self.world_size:
            raise ValueError("distributed checkpoint has an incomplete rank set")
        if [rank.rank for rank in self.ranks] != list(range(self.world_size)):
            raise ValueError("distributed checkpoint must contain every rank exactly once")
        rank_payloads = (
            decode_rank_state_bundle(rank_state_payload, world_size=self.world_size)
            if rank_state_payload is not None
            else None
        )
        shards: set[str] = set()
        total_tokens = 0
        for rank in self.ranks:
            rank.assert_valid()
            if rank.world_size != self.world_size or rank.global_update != self.global_update:
                raise ValueError("rank metadata disagrees with distributed checkpoint")
            if (
                rank.model_state_sha256 != self.model_state_sha256
                or rank.optimizer_state_sha256 != self.optimizer_state_sha256
            ):
                raise ValueError("rank model or optimizer hash disagrees with replicated state")
            if rank.data_shard_identity in shards:
                raise ValueError("rank-local data-shard identity is duplicated")
            if rank.cumulative_tokens < 0:
                raise ValueError("rank-local token counters cannot be negative")
            if rank_payloads is not None:
                rank_bytes = rank_payloads[rank.rank]
                if _payload_hash(rank_bytes["rng_state"]) != rank.rng_state_sha256:
                    raise ValueError(f"rank {rank.rank} RNG payload hash mismatch")
                if _payload_hash(rank_bytes["cursor_state"]) != rank.cursor_state_sha256:
                    raise ValueError(f"rank {rank.rank} cursor payload hash mismatch")
            shards.add(rank.data_shard_identity)
            total_tokens += rank.cumulative_tokens
        expected_barrier = collective_barrier_sha256(
            global_update=self.global_update,
            world_size=self.world_size,
            topology=self.topology,
            ranks=self.ranks,
        )
        if expected_barrier != self.collective_barrier_sha256:
            raise ValueError("collective barrier hash does not bind update, topology, and rank receipts")
        if total_tokens != self.global_tokens:
            raise ValueError("rank cumulative-token totals do not equal the global total")
        if training_state is not None:
            training_state.assert_valid()
            if (
                training_state.sha256() != self.training_state_sha256
                or training_state.parent_checkpoint_sha256 != self.parent_checkpoint_sha256
                or training_state.global_update != self.global_update
                or training_state.cumulative_tokens != self.global_tokens
                or training_state.rng_state_sha256 != aggregate_rank_rng_sha256(self.ranks)
            ):
                raise ValueError("distributed metadata disagrees with shared training state or rank RNGs")
        if model_payload is not None and _payload_hash(model_payload) != self.model_state_sha256:
            raise ValueError("distributed model hash disagrees with model.bin")
        if optimizer_payload is not None and _payload_hash(optimizer_payload) != self.optimizer_state_sha256:
            raise ValueError("distributed optimizer hash disagrees with optimizer.bin")

    def canonical(self) -> dict[str, object]:
        self.assert_valid()
        value = asdict(self)
        value["ranks"] = [rank.canonical() for rank in self.ranks]
        return value

    def sha256(self) -> str:
        return hashlib.sha256(_canonical_json(self.canonical())).hexdigest()

    def assert_next_batch(self, *, rank: int, observed_sha256: str) -> None:
        if type(rank) is not int or not 0 <= rank < self.world_size:
            raise ValueError("next-batch rank is outside the distributed checkpoint world")
        if not _sha256(observed_sha256):
            raise ValueError("observed next-batch fingerprint is invalid")
        expected = self.ranks[rank].next_batch_sha256
        if observed_sha256 != expected:
            raise ValueError(
                f"rank {rank} next batch differs from the uninterrupted checkpoint boundary"
            )

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "ReplicatedTrainingCheckpoint":
        if not isinstance(value, Mapping):
            raise ValueError("replicated distributed-checkpoint metadata must be an object")
        expected = {
            "schema", "parent_checkpoint_sha256", "global_update", "global_tokens",
            "world_size", "topology", "optimizer_layout", "training_state_sha256", "model_state_sha256",
            "optimizer_state_sha256", "collective_barrier_sha256", "ranks",
        }
        if set(value) != expected or not isinstance(value.get("ranks"), list):
            raise ValueError("replicated distributed-checkpoint fields do not match schema")
        rank_fields = set(DistributedRankState.__dataclass_fields__)
        ranks: list[DistributedRankState] = []
        for row in value["ranks"]:
            if not isinstance(row, dict) or set(row) != rank_fields:
                raise ValueError("replicated rank-state fields do not match schema")
            ranks.append(DistributedRankState(**row))
        checkpoint = cls(
            schema=value["schema"],
            parent_checkpoint_sha256=value["parent_checkpoint_sha256"],
            global_update=value["global_update"],
            global_tokens=value["global_tokens"],
            world_size=value["world_size"],
            topology=value["topology"],
            optimizer_layout=value["optimizer_layout"],
            training_state_sha256=value["training_state_sha256"],
            model_state_sha256=value["model_state_sha256"],
            optimizer_state_sha256=value["optimizer_state_sha256"],
            collective_barrier_sha256=value["collective_barrier_sha256"],
            ranks=tuple(ranks),
        )
        checkpoint.assert_valid()
        return checkpoint


__all__ = [
    "DISTRIBUTED_CHECKPOINT_SCHEMA",
    "DISTRIBUTED_COMPONENTS",
    "DISTRIBUTED_MANIFEST_SCHEMA",
    "DISTRIBUTED_RANK_STATE_SCHEMA",
    "COLLECTIVE_BARRIER_SCHEMA",
    "RANK_STATE_BUNDLE_SCHEMA",
    "DistributedRankState",
    "ReplicatedTrainingCheckpoint",
    "aggregate_rank_rng_sha256",
    "aggregate_rng_state_sha256",
    "collective_barrier_sha256",
    "decode_rank_state_bundle",
    "encode_rank_state_bundle",
    "rank_state_payload_sha256",
]
