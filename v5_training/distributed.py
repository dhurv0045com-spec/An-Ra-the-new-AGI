"""Fail-closed distributed checkpoint contracts and rank-zero coordination.

The rank metadata schema is independent of XLA, PyTorch, or a storage vendor.
The checkpoint coordinator adds a shared commit boundary: all ranks must
agree on the same logical training state, rank zero alone writes the store,
and the committed SHA or writer error is returned to every rank. Its v1 path
keeps the legacy shared inventory; its v2 path also gathers per-rank RNG and
sampler-cursor captures. Host tests cover v2 assembly and restore, not exact
TPU resume.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from typing import Any, Callable, Mapping, Sequence


RANK_SCHEMA = "anra-v5-rank-checkpoint/v1"
DISTRIBUTED_SCHEMA = "anra-v5-distributed-checkpoint/v1"
CHECKPOINT_CLAIM_SCHEMA = "anra-v5-checkpoint-publish-claim/v1"
UPDATE_DECISION_SCHEMA = "anra-v5-distributed-update-decision/v1"
UPDATE_RNG_SCHEMA = "anra-v5-distributed-update-rng/v1"
UPDATE_LOSS_SCHEMA = "anra-v5-distributed-update-loss/v1"
OBSERVATION_SCHEMA = "anra-v5-checkpoint-observation/v1"
RANK_CAPTURE_SCHEMA = "anra-v5-checkpoint-rank-capture/v1"


class CheckpointCoordinationError(RuntimeError):
    """A rank-zero checkpoint operation failed on the distributed world."""


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_publish_claims(values: list[object], *, world_size: int) -> dict[str, object]:
    def rejected(reason: str) -> dict[str, object]:
        return {
            "schema": CHECKPOINT_CLAIM_SCHEMA,
            "status": "REJECTED",
            "world_size": world_size,
            "reason": reason[:1000],
        }

    if len(values) != world_size or any(not isinstance(value, dict) for value in values):
        return rejected("checkpoint publish claim set does not cover the full world")
    expected_fields = {
        "schema", "rank", "world_size", "lineage_id", "generation",
        "global_update", "cumulative_tokens", "state_sha256",
        "parent_checkpoint_sha256", "local_error",
    }
    claims = list(values)
    if any(set(claim) != expected_fields for claim in claims):
        return rejected("checkpoint publish claim fields do not match schema")
    if any(claim.get("schema") != CHECKPOINT_CLAIM_SCHEMA for claim in claims):
        return rejected("unsupported checkpoint publish claim schema")
    invalid_errors = [claim.get("rank") for claim in claims
                      if claim.get("local_error") is not None
                      and not isinstance(claim.get("local_error"), str)]
    if invalid_errors:
        return rejected("checkpoint publish claim has an invalid local preflight error")
    local_errors = [f"rank {claim.get('rank')}: {claim['local_error']}"
                    for claim in claims if claim.get("local_error")]
    if local_errors:
        return rejected("; ".join(local_errors))
    if any(type(claim.get("rank")) is not int for claim in claims):
        return rejected("checkpoint publish claim rank must be an integer")
    if any(type(claim.get("world_size")) is not int for claim in claims):
        return rejected("checkpoint publish claim world size must be an integer")
    ranks = sorted(claim.get("rank") for claim in claims)
    if ranks != list(range(world_size)):
        return rejected("checkpoint publish claims must include every rank exactly once")
    if any(claim.get("world_size") != world_size for claim in claims):
        return rejected("checkpoint publish claims disagree on world size")
    for claim in claims:
        if (
            not isinstance(claim.get("lineage_id"), str)
            or not claim["lineage_id"]
            or type(claim.get("generation")) is not int
            or type(claim.get("global_update")) is not int
            or type(claim.get("cumulative_tokens")) is not int
            or min(claim["generation"], claim["global_update"], claim["cumulative_tokens"]) < 0
            or not _is_sha256(claim.get("state_sha256"))
        ):
            return rejected("checkpoint publish claim contains invalid state identity")
        parent = claim.get("parent_checkpoint_sha256")
        if parent is not None and not _is_sha256(parent):
            return rejected("checkpoint publish claim has an invalid parent SHA")
    shared_fields = expected_fields - {"rank", "local_error"}
    canonical = {key: claims[0][key] for key in sorted(shared_fields)}
    if any({key: claim[key] for key in sorted(shared_fields)} != canonical for claim in claims[1:]):
        return rejected("ranks reached the checkpoint boundary with different state or parent")
    return {
        "schema": CHECKPOINT_CLAIM_SCHEMA,
        "status": "READY",
        "world_size": world_size,
        "claim_sha256": hashlib.sha256(_canonical_json(canonical)).hexdigest(),
    }


def _select_rank_zero_result(values: list[object], *, world_size: int) -> dict[str, object]:
    if len(values) != world_size:
        raise ValueError("checkpoint result collective did not cover the full world")
    results = [value for value in values if value is not None]
    if len(results) != 1 or not isinstance(results[0], dict):
        raise ValueError("checkpoint result collective requires one rank-zero result")
    result = results[0]
    if result.get("writer_rank") != 0:
        raise ValueError("checkpoint result was not produced by rank zero")
    return result


def _validate_rank_capture_claims(
    values: list[object], *, world_size: int,
) -> dict[str, object]:
    """Gather small production continuation captures in rank order."""

    def rejected(reason: str) -> dict[str, object]:
        return {
            "schema": RANK_CAPTURE_SCHEMA,
            "status": "REJECTED",
            "world_size": world_size,
            "reason": reason[:1000],
        }

    if len(values) != world_size or any(not isinstance(value, dict) for value in values):
        return rejected("rank-capture set does not cover the full world")
    expected_fields = {
        "schema", "rank", "world_size", "cumulative_tokens", "global_tokens",
        "capture", "local_error",
    }
    claims = list(values)
    if any(set(claim) != expected_fields for claim in claims):
        return rejected("rank-capture claim fields do not match schema")
    if any(claim.get("schema") != RANK_CAPTURE_SCHEMA for claim in claims):
        return rejected("unsupported rank-capture claim schema")
    failures = [f"rank {claim.get('rank')}: {claim['local_error']}"
                for claim in claims if isinstance(claim.get("local_error"), str)
                and claim["local_error"]]
    if failures:
        return rejected("; ".join(failures))
    if any(claim.get("local_error") is not None for claim in claims):
        return rejected("rank-capture claim has an invalid local error")
    if any(type(claim.get("rank")) is not int for claim in claims):
        return rejected("rank-capture rank must be an integer")
    ordered = sorted(claims, key=lambda claim: claim["rank"])
    if [claim["rank"] for claim in ordered] != list(range(world_size)):
        return rejected("rank captures must include every rank exactly once")
    if any(type(claim.get("world_size")) is not int
           or claim["world_size"] != world_size for claim in ordered):
        return rejected("rank-capture claims disagree on world size")
    if any(type(claim.get("global_tokens")) is not int
           or claim["global_tokens"] < 0 for claim in ordered):
        return rejected("rank-capture global token total is invalid")
    expected_global_tokens = ordered[0]["global_tokens"]
    if any(claim["global_tokens"] != expected_global_tokens for claim in ordered[1:]):
        return rejected("ranks disagree on the checkpoint global token total")
    if any(type(claim.get("cumulative_tokens")) is not int
           or claim["cumulative_tokens"] < 0 for claim in ordered):
        return rejected("rank-capture cumulative token count is invalid")
    for claim in ordered:
        capture = claim.get("capture")
        if capture is None:
            return rejected(f"rank {claim['rank']} has no continuation capture")
        if (getattr(capture, "rank", None) != claim["rank"]
                or getattr(capture, "cumulative_tokens", None) != claim["cumulative_tokens"]):
            return rejected(f"rank {claim['rank']} capture identity disagrees with its claim")
    if sum(claim["cumulative_tokens"] for claim in ordered) != expected_global_tokens:
        return rejected("rank cumulative-token totals do not equal the global total")
    return {
        "schema": RANK_CAPTURE_SCHEMA,
        "status": "READY",
        "world_size": world_size,
        "captures": [claim["capture"] for claim in ordered],
    }


def _aggregate_update_rng_states(values: list[object], *, world_size: int) -> dict[str, object]:
    def rejected(reason: str) -> dict[str, object]:
        return {
            "schema": UPDATE_RNG_SCHEMA,
            "status": "REJECTED",
            "world_size": world_size,
            "reason": reason[:1000],
        }

    if len(values) != world_size or any(not isinstance(value, dict) for value in values):
        return rejected("distributed update RNG set does not cover the full world")
    expected_fields = {
        "schema", "rank", "world_size", "rng_state_sha256", "local_real_tokens",
        "cumulative_tokens_before", "expected_global_tokens", "expected_prior_global_tokens",
        "local_error",
    }
    claims = list(values)
    if any(set(claim) != expected_fields for claim in claims):
        return rejected("distributed update RNG fields do not match schema")
    if any(claim.get("schema") != UPDATE_RNG_SCHEMA for claim in claims):
        return rejected("unsupported distributed update RNG schema")
    if any(claim.get("local_error") is not None
           and not isinstance(claim.get("local_error"), str) for claim in claims):
        return rejected("distributed update RNG claim has an invalid local error")
    failures = [f"rank {claim.get('rank')}: {claim['local_error']}"
                for claim in claims if claim.get("local_error")]
    if failures:
        return rejected("; ".join(failures))
    if any(type(claim.get("rank")) is not int for claim in claims):
        return rejected("distributed update RNG rank must be an integer")
    if any(type(claim.get("world_size")) is not int for claim in claims):
        return rejected("distributed update RNG world size must be an integer")
    if sorted(claim["rank"] for claim in claims) != list(range(world_size)):
        return rejected("distributed update RNG claims must include every rank exactly once")
    if any(claim["world_size"] != world_size for claim in claims):
        return rejected("distributed update RNG claims disagree on world size")
    ordered = sorted(claims, key=lambda claim: claim["rank"])
    hashes = [claim.get("rng_state_sha256") for claim in ordered]
    if any(not _is_sha256(value) for value in hashes):
        return rejected("distributed update RNG claim contains an invalid SHA-256")
    counter_fields = (
        "local_real_tokens", "cumulative_tokens_before", "expected_global_tokens",
        "expected_prior_global_tokens",
    )
    if any(type(claim.get(name)) is not int or claim[name] < 0
           for claim in ordered for name in counter_fields):
        return rejected("distributed update token counters must be nonnegative integers")
    expected_global = ordered[0]["expected_global_tokens"]
    expected_prior_global = ordered[0]["expected_prior_global_tokens"]
    if any(claim["expected_global_tokens"] != expected_global
           or claim["expected_prior_global_tokens"] != expected_prior_global
           for claim in ordered[1:]):
        return rejected("ranks disagree on global update or prior token totals")
    if sum(claim["cumulative_tokens_before"] for claim in ordered) != expected_prior_global:
        return rejected("rank cumulative tokens do not sum to the prior global token total")
    if sum(claim["local_real_tokens"] for claim in ordered) != expected_global:
        return rejected("rank token contributions do not sum to the global update ledger")
    from .distributed_checkpoint import aggregate_rng_state_sha256

    aggregate = aggregate_rng_state_sha256(hashes)
    return {
        "schema": UPDATE_RNG_SCHEMA,
        "status": "READY",
        "world_size": world_size,
        "rng_state_sha256": aggregate,
        "cumulative_tokens_by_rank": [
            claim["cumulative_tokens_before"] + claim["local_real_tokens"]
            for claim in ordered
        ],
    }


def _validate_update_decisions(values: list[object], *, world_size: int) -> dict[str, object]:
    def rejected(reason: str) -> dict[str, object]:
        return {
            "schema": UPDATE_DECISION_SCHEMA,
            "status": "REJECTED",
            "world_size": world_size,
            "reason": reason[:1000],
        }

    if len(values) != world_size or any(not isinstance(value, dict) for value in values):
        return rejected("distributed update decision set does not cover the full world")
    expected_fields = {
        "schema", "rank", "world_size", "lineage_id", "generation",
        "global_update", "cumulative_tokens", "state_sha256",
        "checkpoint_requested", "stop_requested", "local_error",
    }
    decisions = list(values)
    if any(set(item) != expected_fields for item in decisions):
        return rejected("distributed update decision fields do not match schema")
    if any(item.get("schema") != UPDATE_DECISION_SCHEMA for item in decisions):
        return rejected("unsupported distributed update decision schema")
    if any(item.get("local_error") is not None
           and not isinstance(item.get("local_error"), str) for item in decisions):
        return rejected("distributed update decision has an invalid local error")
    failures = [f"rank {item.get('rank')}: {item['local_error']}"
                for item in decisions if item.get("local_error")]
    if failures:
        return rejected("; ".join(failures))
    if any(type(item.get("rank")) is not int for item in decisions):
        return rejected("distributed update rank must be an integer")
    if any(type(item.get("world_size")) is not int for item in decisions):
        return rejected("distributed update world size must be an integer")
    if sorted(item["rank"] for item in decisions) != list(range(world_size)):
        return rejected("distributed update decisions must include every rank exactly once")
    if any(item["world_size"] != world_size for item in decisions):
        return rejected("distributed update decisions disagree on world size")
    for item in decisions:
        if (
            not isinstance(item.get("lineage_id"), str)
            or not item["lineage_id"]
            or type(item.get("generation")) is not int
            or type(item.get("global_update")) is not int
            or type(item.get("cumulative_tokens")) is not int
            or min(item["generation"], item["global_update"], item["cumulative_tokens"]) < 0
            or not _is_sha256(item.get("state_sha256"))
            or type(item.get("checkpoint_requested")) is not bool
            or type(item.get("stop_requested")) is not bool
        ):
            return rejected("distributed update decision contains invalid state or action")
    shared_fields = expected_fields - {"rank", "local_error"}
    canonical = {key: decisions[0][key] for key in sorted(shared_fields)}
    if any({key: item[key] for key in sorted(shared_fields)} != canonical
           for item in decisions[1:]):
        return rejected("ranks disagree on updated state, checkpoint, or stop action")
    return {
        "schema": UPDATE_DECISION_SCHEMA,
        "status": "READY",
        "world_size": world_size,
        "decision_sha256": hashlib.sha256(_canonical_json(canonical)).hexdigest(),
    }


def _validate_update_loss_claims(
    values: list[object], *, world_size: int,
) -> dict[str, object]:
    """Combine rank-local cross-entropy numerators into one global mean."""

    def rejected(reason: str) -> dict[str, object]:
        return {
            "schema": UPDATE_LOSS_SCHEMA,
            "status": "REJECTED",
            "world_size": world_size,
            "reason": reason[:1000],
        }

    if len(values) != world_size or any(not isinstance(value, dict) for value in values):
        return rejected("distributed loss contribution set does not cover the full world")
    expected_fields = {
        "schema", "rank", "world_size", "global_update", "global_tokens",
        "local_eligible_tokens", "local_loss_numerator",
    }
    claims = list(values)
    if any(set(claim) != expected_fields for claim in claims):
        return rejected("distributed loss contribution fields do not match schema")
    if any(claim.get("schema") != UPDATE_LOSS_SCHEMA for claim in claims):
        return rejected("unsupported distributed loss contribution schema")
    if any(type(claim.get("rank")) is not int for claim in claims):
        return rejected("distributed loss contribution rank must be an integer")
    ordered = sorted(claims, key=lambda claim: claim["rank"])
    if [claim["rank"] for claim in ordered] != list(range(world_size)):
        return rejected("distributed loss contributions must include each rank exactly once")
    if any(type(claim.get("world_size")) is not int
           or claim["world_size"] != world_size for claim in ordered):
        return rejected("distributed loss contributions disagree on world size")
    for claim in ordered:
        numerator = claim.get("local_loss_numerator")
        if (
            type(claim.get("global_update")) is not int
            or claim["global_update"] < 1
            or type(claim.get("global_tokens")) is not int
            or claim["global_tokens"] <= 0
            or type(claim.get("local_eligible_tokens")) is not int
            or claim["local_eligible_tokens"] < 0
            or isinstance(numerator, bool)
            or not isinstance(numerator, (int, float))
            or not math.isfinite(float(numerator))
            or float(numerator) < 0.0
        ):
            return rejected("distributed loss contribution contains invalid counts or value")
        if claim["local_eligible_tokens"] == 0 and float(numerator) != 0.0:
            return rejected("zero-token rank contribution must have a zero loss numerator")
    shared_update = ordered[0]["global_update"]
    shared_total = ordered[0]["global_tokens"]
    if any(claim["global_update"] != shared_update for claim in ordered[1:]):
        return rejected("ranks disagree on the loss update index")
    if any(claim["global_tokens"] != shared_total for claim in ordered[1:]):
        return rejected("ranks disagree on the global eligible-token denominator")
    local_total = sum(claim["local_eligible_tokens"] for claim in ordered)
    if local_total != shared_total:
        return rejected(
            "rank-local eligible-token counts do not equal the global loss denominator"
        )
    numerator_total = sum(float(claim["local_loss_numerator"]) for claim in ordered)
    global_loss = numerator_total / shared_total
    if not math.isfinite(numerator_total) or not math.isfinite(global_loss):
        return rejected("distributed global loss is not finite")
    rank_contributions = [{
        "rank": claim["rank"],
        "eligible_tokens": claim["local_eligible_tokens"],
        "loss_numerator": float(claim["local_loss_numerator"]),
    } for claim in ordered]
    receipt: dict[str, object] = {
        "schema": UPDATE_LOSS_SCHEMA,
        "status": "READY",
        "world_size": world_size,
        "global_update": shared_update,
        "global_tokens": shared_total,
        "loss_numerator": numerator_total,
        "global_loss": global_loss,
        "rank_contributions": rank_contributions,
    }
    receipt["receipt_sha256"] = hashlib.sha256(_canonical_json(receipt)).hexdigest()
    return receipt


def _validate_observations(values: list[object], *, world_size: int) -> dict[str, object]:
    def rejected(reason: str) -> dict[str, object]:
        return {
            "schema": OBSERVATION_SCHEMA,
            "status": "REJECTED",
            "world_size": world_size,
            "reason": reason[:1000],
        }

    if len(values) != world_size or any(not isinstance(value, dict) for value in values):
        return rejected("checkpoint observation set does not cover the full world")
    expected_fields = {
        "schema", "rank", "world_size", "state_sha256", "checkpoint_sha256",
        "callback_requested", "local_error",
    }
    observations = list(values)
    if any(set(item) != expected_fields for item in observations):
        return rejected("checkpoint observation fields do not match schema")
    if any(item.get("schema") != OBSERVATION_SCHEMA for item in observations):
        return rejected("unsupported checkpoint observation schema")
    if any(item.get("local_error") is not None
           and not isinstance(item.get("local_error"), str) for item in observations):
        return rejected("checkpoint observation has an invalid local error")
    failures = [f"rank {item.get('rank')}: {item['local_error']}"
                for item in observations if item.get("local_error")]
    if failures:
        return rejected("; ".join(failures))
    if any(type(item.get("rank")) is not int for item in observations):
        return rejected("checkpoint observation rank must be an integer")
    if any(type(item.get("world_size")) is not int for item in observations):
        return rejected("checkpoint observation world size must be an integer")
    if sorted(item["rank"] for item in observations) != list(range(world_size)):
        return rejected("checkpoint observations must include every rank exactly once")
    for item in observations:
        if (
            item["world_size"] != world_size
            or type(item.get("callback_requested")) is not bool
            or not _is_sha256(item.get("state_sha256"))
            or not _is_sha256(item.get("checkpoint_sha256"))
        ):
            return rejected("checkpoint observation contains invalid identity")
    shared_fields = expected_fields - {"rank", "local_error"}
    canonical = {key: observations[0][key] for key in sorted(shared_fields)}
    if any({key: item[key] for key in sorted(shared_fields)} != canonical
           for item in observations[1:]):
        return rejected("ranks disagree on checkpoint observation identity")
    return {
        "schema": OBSERVATION_SCHEMA,
        "status": "READY",
        "world_size": world_size,
        "observation_sha256": hashlib.sha256(_canonical_json(canonical)).hexdigest(),
    }


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


@dataclass(slots=True)
class RankZeroCheckpointCoordinator:
    """Coordinate one local CheckpointStore commit across a replicated world.

    ``mesh_reduce`` must be an out-of-graph all-rank gather/reduce that returns
    the reducer's same result on every rank. Rank zero alone builds payloads,
    publishes, and runs commit side effects. All ranks verify one identical
    TrainingState/parent claim and receive the same committed SHA. The current
    checkpoint inventory still stores rank zero's RNG and cursor only.
    """

    rank: int
    world_size: int
    mesh_reduce: Callable[[str, Any, Callable[[list[Any]], Any]], Any]
    initial_rank_cumulative_tokens: int = 0
    _call_index: int = field(default=0, init=False, repr=False)
    _update_index: int = field(default=0, init=False, repr=False)
    _rng_index: int = field(default=0, init=False, repr=False)
    _loss_index: int = field(default=0, init=False, repr=False)
    _rank_cumulative_tokens: int = field(default=0, init=False, repr=False)
    _observation_index: int = field(default=0, init=False, repr=False)
    _last_collective_prefix: str | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.world_size <= 0 or not 0 <= self.rank < self.world_size:
            raise ValueError("checkpoint coordinator rank is outside its world")
        if not callable(self.mesh_reduce):
            raise ValueError("checkpoint coordinator needs a mesh-reduce collective")
        if (type(self.initial_rank_cumulative_tokens) is not int
                or self.initial_rank_cumulative_tokens < 0):
            raise ValueError("initial rank token count must be a nonnegative integer")
        self._rank_cumulative_tokens = self.initial_rank_cumulative_tokens

    @property
    def is_writer(self) -> bool:
        return self.rank == 0

    @property
    def rank_cumulative_tokens(self) -> int:
        """This worker's committed-through-current-update real-token count."""

        return self._rank_cumulative_tokens

    def restore_rank_progress(self, *, cumulative_tokens: int) -> None:
        """Seed a fresh coordinator from its verified v2 rank receipt."""

        if type(cumulative_tokens) is not int or cumulative_tokens < 0:
            raise ValueError("restored rank token count must be a nonnegative integer")
        if any((self._call_index, self._update_index, self._rng_index, self._loss_index,
                self._observation_index)):
            raise ValueError("rank progress must be restored before collectives begin")
        if (self.initial_rank_cumulative_tokens not in (0, cumulative_tokens)):
            raise ValueError("configured initial rank token count differs from checkpoint")
        self._rank_cumulative_tokens = cumulative_tokens

    def agree_update_result(
        self,
        *,
        state: Any,
        checkpoint_requested: bool,
        stop_requested: bool,
        local_error: str | None = None,
    ) -> None:
        """Make every rank agree on each update before branching or stopping.

        Call this on every training update, including ranks that did not
        request a checkpoint. It converts boundary-policy or post-step
        validation differences into a shared failure instead of letting one
        process enter a publication collective while its peers continue.
        """

        claim: dict[str, object] = {
            "schema": UPDATE_DECISION_SCHEMA,
            "rank": self.rank,
            "world_size": self.world_size,
            "lineage_id": "",
            "generation": -1,
            "global_update": -1,
            "cumulative_tokens": -1,
            "state_sha256": "0" * 64,
            "checkpoint_requested": bool(checkpoint_requested),
            "stop_requested": bool(stop_requested),
            "local_error": local_error,
        }
        if local_error is None:
            try:
                state.assert_valid()
                claim.update({
                    "lineage_id": state.lineage_id,
                    "generation": state.generation,
                    "global_update": state.global_update,
                    "cumulative_tokens": state.cumulative_tokens,
                    "state_sha256": state.sha256(),
                })
            except Exception as exc:
                claim.update({
                    "lineage_id": "",
                    "generation": -1,
                    "global_update": -1,
                    "cumulative_tokens": -1,
                    "state_sha256": "0" * 64,
                    "local_error": f"{type(exc).__name__}: {str(exc)[:500]}",
                })
        prefix = f"anra-v5-training-update-{self._update_index:08d}"
        decision = self.mesh_reduce(
            f"{prefix}-decisions", claim,
            lambda values: _validate_update_decisions(values, world_size=self.world_size),
        )
        self._update_index += 1
        if not isinstance(decision, dict) or decision.get("schema") != UPDATE_DECISION_SCHEMA:
            raise CheckpointCoordinationError("distributed update decision collective is invalid")
        if decision.get("status") != "READY":
            raise CheckpointCoordinationError(
                "distributed update rejected: "
                f"{decision.get('reason', 'unknown decision error')}"
            )
        if (
            decision.get("world_size") != self.world_size
            or not _is_sha256(decision.get("decision_sha256"))
        ):
            raise CheckpointCoordinationError("distributed update decision identity is invalid")

    def aggregate_update_loss(
        self,
        *,
        global_update: int,
        global_tokens: int,
        local_eligible_tokens: int,
        local_loss_numerator: float,
    ) -> dict[str, object]:
        """Return one validated global mean from rank-local loss numerators.

        All ranks must call this after local-update success agreement and
        before gradient reduction. The shared result binds every rank's
        numerator and denominator contribution into one receipt hash.
        """

        claim = {
            "schema": UPDATE_LOSS_SCHEMA,
            "rank": self.rank,
            "world_size": self.world_size,
            "global_update": global_update,
            "global_tokens": global_tokens,
            "local_eligible_tokens": local_eligible_tokens,
            "local_loss_numerator": local_loss_numerator,
        }
        tag = f"anra-v5-training-loss-{self._loss_index:08d}"
        result = self.mesh_reduce(
            tag,
            claim,
            lambda values: _validate_update_loss_claims(
                values, world_size=self.world_size,
            ),
        )
        self._loss_index += 1
        if not isinstance(result, dict) or result.get("schema") != UPDATE_LOSS_SCHEMA:
            raise CheckpointCoordinationError("distributed update-loss collective is invalid")
        if result.get("status") != "READY":
            raise CheckpointCoordinationError(
                "distributed update loss rejected: "
                f"{result.get('reason', 'unknown aggregation error')}"
            )
        if (
            result.get("world_size") != self.world_size
            or result.get("global_update") != global_update
            or result.get("global_tokens") != global_tokens
            or not _is_sha256(result.get("receipt_sha256"))
        ):
            raise CheckpointCoordinationError("distributed update-loss identity is invalid")
        return result

    def aggregate_update_rng_state(
        self,
        *,
        local_rng_state_sha256: str | None,
        local_real_tokens: int | None,
        expected_global_tokens: int | None,
        expected_prior_global_tokens: int,
        local_error: str | None = None,
    ) -> str:
        """Bind rank-local post-update RNG identities before advancing shared state.

        Every rank must call this once per attempted update, including ranks
        whose local backend step failed. The shared TrainingState stores the
        ordered aggregate; v2 checkpoint metadata separately keeps each rank's
        actual RNG payload for exact restoration.
        """

        claim: dict[str, object] = {
            "schema": UPDATE_RNG_SCHEMA,
            "rank": self.rank,
            "world_size": self.world_size,
            "rng_state_sha256": local_rng_state_sha256,
            "local_real_tokens": local_real_tokens,
            "cumulative_tokens_before": self._rank_cumulative_tokens,
            "expected_global_tokens": expected_global_tokens,
            "expected_prior_global_tokens": expected_prior_global_tokens,
            "local_error": local_error,
        }
        if local_error is None and not _is_sha256(local_rng_state_sha256):
            claim["local_error"] = "rank-local RNG identity is not a lowercase SHA-256"
        tag = f"anra-v5-training-rng-{self._rng_index:08d}"
        result = self.mesh_reduce(
            tag,
            claim,
            lambda values: _aggregate_update_rng_states(values, world_size=self.world_size),
        )
        self._rng_index += 1
        if not isinstance(result, dict) or result.get("schema") != UPDATE_RNG_SCHEMA:
            raise CheckpointCoordinationError("distributed update RNG collective is invalid")
        if result.get("status") != "READY":
            raise CheckpointCoordinationError(
                "distributed update RNG aggregation rejected: "
                f"{result.get('reason', 'unknown RNG aggregation error')}"
            )
        aggregate = result.get("rng_state_sha256")
        if result.get("world_size") != self.world_size or not _is_sha256(aggregate):
            raise CheckpointCoordinationError("distributed update RNG result is invalid")
        by_rank = result.get("cumulative_tokens_by_rank")
        if (not isinstance(by_rank, list) or len(by_rank) != self.world_size
                or any(type(value) is not int or value < 0 for value in by_rank)):
            raise CheckpointCoordinationError("distributed update token-counter result is invalid")
        self._rank_cumulative_tokens = by_rank[self.rank]
        return aggregate

    def _collective_prefix(self) -> str:
        # The tag intentionally does not depend on rank-local state. Divergent
        # state must meet in one collective so it can be rejected, not wait on
        # different tags forever.
        return f"anra-v5-checkpoint-call-{self._call_index:08d}"

    def publish_checkpoint(
        self,
        *,
        state: Any,
        parent_checkpoint_sha256: str | None,
        store: Any,
        payload_builder: Callable[[Any], Mapping[str, bytes]],
    ) -> str:
        """Publish a v1 checkpoint once on rank zero after boundary agreement."""

        return self._publish_checkpoint(
            state=state,
            parent_checkpoint_sha256=parent_checkpoint_sha256,
            store=store,
            payload_builder=payload_builder,
            rank_capture=None,
            rank_capture_error=None,
            distributed_payload_builder=None,
        )

    def publish_distributed_checkpoint(
        self,
        *,
        state: Any,
        parent_checkpoint_sha256: str | None,
        store: Any,
        rank_capture: Any,
        rank_capture_error: str | None,
        payload_builder: Callable[[Any, Sequence[Any]], Mapping[str, bytes]],
    ) -> str:
        """Gather every rank's continuation and publish one v2 generation."""

        return self._publish_checkpoint(
            state=state,
            parent_checkpoint_sha256=parent_checkpoint_sha256,
            store=store,
            payload_builder=None,
            rank_capture=rank_capture,
            rank_capture_error=rank_capture_error,
            distributed_payload_builder=payload_builder,
        )

    def _publish_checkpoint(
        self,
        *,
        state: Any,
        parent_checkpoint_sha256: str | None,
        store: Any,
        payload_builder: Callable[[Any], Mapping[str, bytes]] | None,
        rank_capture: Any,
        rank_capture_error: str | None,
        distributed_payload_builder: (
            Callable[[Any, Sequence[Any]], Mapping[str, bytes]] | None
        ),
    ) -> str:
        """Run the common agreement, optional capture gather, and commit."""

        distributed_mode = distributed_payload_builder is not None
        if distributed_mode == (payload_builder is not None):
            raise ValueError("checkpoint publication must select exactly one inventory builder")

        claim: dict[str, object] = {
            "schema": CHECKPOINT_CLAIM_SCHEMA,
            "rank": self.rank,
            "world_size": self.world_size,
            "lineage_id": "",
            "generation": -1,
            "global_update": -1,
            "cumulative_tokens": -1,
            "state_sha256": "0" * 64,
            "parent_checkpoint_sha256": None,
            "local_error": None,
        }
        try:
            state.assert_valid()
            if parent_checkpoint_sha256 is not None and not _is_sha256(parent_checkpoint_sha256):
                raise ValueError("checkpoint coordinator parent must be a SHA-256")
            claim.update({
                "lineage_id": state.lineage_id,
                "generation": state.generation,
                "global_update": state.global_update,
                "cumulative_tokens": state.cumulative_tokens,
                "state_sha256": state.sha256(),
                "parent_checkpoint_sha256": parent_checkpoint_sha256,
            })
        except Exception as exc:
            # Local preflight errors still enter the shared claim collective so
            # healthy peers reject together instead of waiting at a later tag.
            claim["local_error"] = f"{type(exc).__name__}: {str(exc)[:500]}"
        prefix = self._collective_prefix()
        ready = self.mesh_reduce(
            f"{prefix}-claims", claim,
            lambda values: _validate_publish_claims(values, world_size=self.world_size),
        )
        if not isinstance(ready, dict) or ready.get("schema") != CHECKPOINT_CLAIM_SCHEMA:
            raise CheckpointCoordinationError("checkpoint ranks did not agree on a shared boundary")
        if ready.get("status") != "READY":
            raise CheckpointCoordinationError(
                "checkpoint ranks rejected the shared boundary: "
                f"{ready.get('reason', 'unknown claim error')}"
            )
        if (
            ready.get("world_size") != self.world_size
            or not _is_sha256(ready.get("claim_sha256"))
        ):
            raise CheckpointCoordinationError("checkpoint ranks returned an invalid shared boundary")

        rank_captures: list[Any] | None = None
        if distributed_mode:
            capture_error = rank_capture_error
            captured_rank = getattr(rank_capture, "rank", None)
            captured_tokens = getattr(rank_capture, "cumulative_tokens", None)
            if capture_error is None and (
                type(captured_rank) is not int or captured_rank != self.rank
                or type(captured_tokens) is not int
                or captured_tokens != self._rank_cumulative_tokens
            ):
                capture_error = "rank capture identity or cumulative tokens disagree with coordinator"
            capture_claim = {
                "schema": RANK_CAPTURE_SCHEMA,
                "rank": self.rank,
                "world_size": self.world_size,
                "cumulative_tokens": (
                    captured_tokens if type(captured_tokens) is int else -1
                ),
                "global_tokens": state.cumulative_tokens,
                "capture": rank_capture,
                "local_error": capture_error,
            }
            gathered = self.mesh_reduce(
                f"{prefix}-rank-captures",
                capture_claim,
                lambda values: _validate_rank_capture_claims(
                    values, world_size=self.world_size,
                ),
            )
            if (not isinstance(gathered, dict)
                    or gathered.get("schema") != RANK_CAPTURE_SCHEMA
                    or gathered.get("status") != "READY"
                    or gathered.get("world_size") != self.world_size
                    or not isinstance(gathered.get("captures"), list)
                    or len(gathered["captures"]) != self.world_size):
                reason = gathered.get("reason", "invalid rank-capture result") if isinstance(
                    gathered, dict
                ) else "invalid rank-capture result"
                raise CheckpointCoordinationError(
                    f"distributed checkpoint captures rejected: {reason}"
                )
            rank_captures = gathered["captures"]

        writer_result: dict[str, object] | None = None
        if self.is_writer:
            try:
                if distributed_mode:
                    assert distributed_payload_builder is not None
                    assert rank_captures is not None
                    checkpoint_sha256 = store.publish_distributed(
                        state=state,
                        payloads=distributed_payload_builder(state, rank_captures),
                        expected_parent_sha256=parent_checkpoint_sha256,
                    )
                else:
                    assert payload_builder is not None
                    checkpoint_sha256 = store.publish(
                        state=state,
                        payloads=payload_builder(state),
                        expected_parent_sha256=parent_checkpoint_sha256,
                    )
                writer_result = {
                    "schema": CHECKPOINT_CLAIM_SCHEMA,
                    "status": "COMMITTED",
                    "writer_rank": 0,
                    "checkpoint_sha256": checkpoint_sha256,
                    "claim_sha256": ready["claim_sha256"],
                }
            except Exception as exc:
                writer_result = {
                    "schema": CHECKPOINT_CLAIM_SCHEMA,
                    "status": "FAILED",
                    "writer_rank": 0,
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:1000],
                    "claim_sha256": ready["claim_sha256"],
                }
        result = self.mesh_reduce(
            f"{prefix}-publication", writer_result,
            lambda values: _select_rank_zero_result(values, world_size=self.world_size),
        )
        if (
            not isinstance(result, dict)
            or result.get("schema") != CHECKPOINT_CLAIM_SCHEMA
            or result.get("claim_sha256") != ready["claim_sha256"]
        ):
            raise CheckpointCoordinationError("rank-zero checkpoint result failed identity validation")
        if result.get("status") != "COMMITTED":
            raise CheckpointCoordinationError(
                "rank-zero checkpoint publication failed: "
                f"{result.get('error_type', 'unknown')}: {result.get('error', 'no detail')}"
            )
        checkpoint_sha256 = result.get("checkpoint_sha256")
        if not _is_sha256(checkpoint_sha256):
            raise CheckpointCoordinationError("rank zero broadcast an invalid checkpoint SHA")
        self._last_collective_prefix = prefix
        self._call_index += 1
        return checkpoint_sha256

    def run_writer_callback(
        self,
        *,
        callback: Callable[[Any, str], None] | None,
        state: Any,
        checkpoint_sha256: str,
    ) -> None:
        """Run shared-storage side effects on rank zero and fan out failures."""

        if not _is_sha256(checkpoint_sha256):
            raise ValueError("checkpoint callback needs a valid checkpoint SHA")
        prefix = self._last_collective_prefix
        if prefix is None:
            raise ValueError("writer callback has no preceding coordinated publication")
        writer_result: dict[str, object] | None = None
        if self.is_writer:
            try:
                if callback is not None:
                    callback(state, checkpoint_sha256)
                writer_result = {
                    "schema": CHECKPOINT_CLAIM_SCHEMA,
                    "status": "CALLBACK_OK",
                    "writer_rank": 0,
                    "checkpoint_sha256": checkpoint_sha256,
                }
            except Exception as exc:
                writer_result = {
                    "schema": CHECKPOINT_CLAIM_SCHEMA,
                    "status": "CALLBACK_FAILED",
                    "writer_rank": 0,
                    "checkpoint_sha256": checkpoint_sha256,
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:1000],
                }
        try:
            result = self.mesh_reduce(
                f"{prefix}-callbacks", writer_result,
                lambda values: _select_rank_zero_result(values, world_size=self.world_size),
            )
        finally:
            # A failed callback consumes this phase. Reusing its tag would
            # pair a later operation with a stale publication result.
            self._last_collective_prefix = None
        if (
            not isinstance(result, dict)
            or result.get("schema") != CHECKPOINT_CLAIM_SCHEMA
            or result.get("checkpoint_sha256") != checkpoint_sha256
        ):
            raise CheckpointCoordinationError("rank-zero checkpoint callback result is invalid")
        if result.get("status") != "CALLBACK_OK":
            raise CheckpointCoordinationError(
                "rank-zero checkpoint side effect failed: "
                f"{result.get('error_type', 'unknown')}: {result.get('error', 'no detail')}"
            )
    def run_observed_callback(
        self,
        *,
        callback: Callable[[Any, str], None] | None,
        state: Any,
        checkpoint_sha256: str,
    ) -> None:
        """Run volatile per-rank receipt updates and fan out local failures."""

        state_sha256 = "0" * 64
        local_error: str | None = None
        try:
            state.assert_valid()
            state_sha256 = state.sha256()
            if not _is_sha256(checkpoint_sha256):
                raise ValueError("checkpoint observation needs a valid checkpoint SHA")
            if callback is not None:
                callback(state, checkpoint_sha256)
        except Exception as exc:
            local_error = f"{type(exc).__name__}: {str(exc)[:500]}"
        observation = {
            "schema": OBSERVATION_SCHEMA,
            "rank": self.rank,
            "world_size": self.world_size,
            "state_sha256": state_sha256,
            "checkpoint_sha256": checkpoint_sha256 if _is_sha256(checkpoint_sha256) else "0" * 64,
            "callback_requested": callback is not None,
            "local_error": local_error,
        }
        tag = f"anra-v5-checkpoint-observation-{self._observation_index:08d}"
        result = self.mesh_reduce(
            tag, observation,
            lambda values: _validate_observations(values, world_size=self.world_size),
        )
        self._observation_index += 1
        if not isinstance(result, dict) or result.get("schema") != OBSERVATION_SCHEMA:
            raise CheckpointCoordinationError("checkpoint observation collective is invalid")
        if result.get("status") != "READY":
            raise CheckpointCoordinationError(
                "checkpoint observation rejected: "
                f"{result.get('reason', 'unknown observation error')}"
            )
        if (
            result.get("world_size") != self.world_size
            or not _is_sha256(result.get("observation_sha256"))
        ):
            raise CheckpointCoordinationError("checkpoint observation identity is invalid")
