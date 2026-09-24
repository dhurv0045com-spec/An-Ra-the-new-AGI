"""CPU-verifiable bridge between the production bucket sampler and v2 resume.

The XLA production gate remains closed. This module gives the real packed-row
sampler a strict checkpoint encoding and a fresh-process restore check while
keeping runtime RNG handling explicitly qualified per device family.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from v5_data.bucket_cursor import BUCKET_CURSOR_SCHEMA, BucketCursorState
from v5_training.checkpoint import CheckpointStore
from v5_training.distributed_checkpoint import (
    DISTRIBUTED_CHECKPOINT_SCHEMA,
    DistributedRankState,
    ReplicatedTrainingCheckpoint,
    aggregate_rng_state_sha256,
    collective_barrier_sha256,
    encode_rank_state_bundle,
    rank_state_payload_sha256,
)
from v5_training.production_microsteps import MaterializedMicrostep
from v5_training.production_backend import (
    ProductionTrainingBackend,
    restore_production_shared,
)
from v5_training.production_sampler import ProductionSampler
from v5_training.state import TrainingState


PRODUCTION_RANK_RESUME_SCHEMA = "anra-v5-production-rank-resume/v3"
NEXT_MICROSTEP_SCHEMA = "anra-v5-next-microstep-fingerprint/v1"
MicrostepFingerprintInput = Mapping[str, object] | MaterializedMicrostep


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def rank_data_shard_identity(
    *, cursor: BucketCursorState, rank: int, world_size: int,
) -> str:
    """Bind a replica's assigned rows to the frozen pack and lane ordering."""

    if type(rank) is not int or type(world_size) is not int:
        raise ValueError("rank and world size must be integers")
    if world_size <= 0 or not 0 <= rank < world_size:
        raise ValueError("rank is outside the sampler world")
    cursor.assert_valid()
    return _sha256(_canonical_json({
        "schema": "anra-v5-rank-data-shard/v1",
        "pack_manifest_sha256": cursor.pack_manifest_sha256,
        "lanes_sha256": cursor.lanes_sha256,
        "rank": rank,
        "world_size": world_size,
    }))


def _normalize_microstep(
    value: Mapping[str, object] | MaterializedMicrostep,
) -> dict[str, object]:
    if isinstance(value, MaterializedMicrostep):
        value = value.fingerprint_mapping()
    required = {
        "bucket", "family", "subfamily", "tokens", "segment_ids", "eligible",
        "tokens_by_source", "planned_total",
    }
    if set(value) != required:
        raise ValueError("next microstep fields do not match the production fingerprint schema")
    bucket = value["bucket"]
    if type(bucket) is not int or bucket <= 0:
        raise ValueError("next microstep bucket must be a positive integer")
    if not isinstance(value["family"], str) or not isinstance(value["subfamily"], str):
        raise ValueError("next microstep family identities must be strings")
    if type(value["planned_total"]) is not int or value["planned_total"] <= 0:
        raise ValueError("next microstep global eligible-token denominator must be positive")
    tokens = value["tokens"]
    segments = value["segment_ids"]
    eligible = value["eligible"]
    if not all(isinstance(rows, (list, tuple)) and rows for rows in (tokens, segments, eligible)):
        raise ValueError("next microstep tensors must contain at least one rank-local row")
    if not (len(tokens) == len(segments) == len(eligible)):
        raise ValueError("next microstep tensors must have the same row count")
    normalized_tokens: list[list[int]] = []
    normalized_segments: list[list[int]] = []
    normalized_eligible: list[list[bool]] = []
    for token_row, segment_row, eligible_row in zip(tokens, segments, eligible):
        if not all(isinstance(row, (list, tuple)) for row in (token_row, segment_row, eligible_row)):
            raise ValueError("next microstep tensor rows must be sequences")
        if not (len(token_row) == len(segment_row) == len(eligible_row) == bucket):
            raise ValueError("next microstep rows must be padded to their declared bucket")
        if any(type(token) is not int or token < 0 for token in token_row):
            raise ValueError("next microstep token ids must be nonnegative integers")
        if any(type(segment) is not int or segment < -1 for segment in segment_row):
            raise ValueError("next microstep segment ids must be integers >= -1")
        if any(type(flag) is not bool for flag in eligible_row):
            raise ValueError("next microstep eligibility values must be booleans")
        if any(flag and segment < 0 for flag, segment in zip(eligible_row, segment_row)):
            raise ValueError("padding positions cannot be eligible training targets")
        normalized_tokens.append(list(token_row))
        normalized_segments.append(list(segment_row))
        normalized_eligible.append(list(eligible_row))
    tokens_by_source = value["tokens_by_source"]
    if not isinstance(tokens_by_source, Mapping) or not tokens_by_source or any(
        not isinstance(name, str) or not name or type(count) is not int or count < 0
        for name, count in tokens_by_source.items()
    ):
        raise ValueError("next microstep source ledger is invalid")
    return {
        "bucket": bucket,
        "family": value["family"],
        "subfamily": value["subfamily"],
        "row_count": len(normalized_tokens),
        "tokens_sha256": _sha256(_canonical_json(normalized_tokens)),
        "segment_ids_sha256": _sha256(_canonical_json(normalized_segments)),
        "eligible_sha256": _sha256(_canonical_json(normalized_eligible)),
        "tokens_by_source": dict(sorted(tokens_by_source.items())),
        "planned_total": value["planned_total"],
    }


def next_microstep_fingerprint_sha256(
    *, rank: int, world_size: int, topology: str, checkpoint_global_update: int,
    cursor: BucketCursorState,
    microsteps: Sequence[MicrostepFingerprintInput],
) -> str:
    """Hash the actual next update's rank-local packed rows and global denominator."""

    if type(checkpoint_global_update) is not int or checkpoint_global_update < 0:
        raise ValueError("checkpoint update must be a nonnegative integer")
    if not isinstance(topology, str) or not topology:
        raise ValueError("resume topology is required")
    identity = rank_data_shard_identity(
        cursor=cursor, rank=rank, world_size=world_size,
    )
    if not microsteps:
        raise ValueError("next update needs at least one packed microstep")
    receipt = {
        "schema": NEXT_MICROSTEP_SCHEMA,
        "rank": rank,
        "world_size": world_size,
        "topology": topology,
        "checkpoint_global_update": checkpoint_global_update,
        "next_global_update": checkpoint_global_update + 1,
        "data_shard_identity": identity,
        "cursor_sha256": _sha256(_canonical_json(cursor.canonical())),
        "microsteps": [_normalize_microstep(step) for step in microsteps],
    }
    return _sha256(_canonical_json(receipt))


def terminal_microstep_fingerprint_sha256(
    *, rank: int, world_size: int, topology: str, state: TrainingState,
) -> str:
    """Bind the fact that this committed state has no next training update."""

    state.assert_valid()
    if not state.complete:
        raise ValueError("terminal sampler fingerprint requires a complete training state")
    if type(rank) is not int or type(world_size) is not int:
        raise ValueError("terminal sampler rank and world size must be integers")
    if world_size <= 0 or not 0 <= rank < world_size:
        raise ValueError("terminal sampler rank is outside its world")
    if not isinstance(topology, str) or not topology:
        raise ValueError("terminal sampler topology is required")
    return _sha256(_canonical_json({
        "schema": "anra-v5-terminal-next-microstep/v1",
        "rank": rank,
        "world_size": world_size,
        "topology": topology,
        "global_update": state.global_update,
        "cumulative_tokens": state.cumulative_tokens,
        "cursor_sha256": _sha256(_canonical_json(state.cursor.canonical())),
    }))


@dataclass(frozen=True, slots=True)
class ProductionRankCapture:
    rank: int
    cumulative_tokens: int
    rng_state: bytes
    cursor: BucketCursorState
    next_microsteps: tuple[MicrostepFingerprintInput, ...]
    collective_receipt_sha256: str
    model_state_sha256: str
    optimizer_state_sha256: str
    next_microstep_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class ProductionRankResume:
    state: TrainingState
    metadata: ReplicatedTrainingCheckpoint
    rank: int
    cursor: BucketCursorState
    cumulative_tokens: int
    sampler_spec_sha256: str
    next_microstep_sha256: str
    shared_payloads: Mapping[str, bytes]
    rng_state: bytes


def _rank_cursor_payload(
    *, rank: int, world_size: int, topology: str, state: TrainingState,
    cursor: BucketCursorState, sampler_spec_sha256: str,
    cumulative_tokens: int, next_microstep_sha256: str,
) -> tuple[bytes, str, str]:
    if not isinstance(state.cursor, BucketCursorState):
        raise ValueError("production distributed resume requires a bucket-lane TrainingState cursor")
    if cursor.pack_manifest_sha256 != state.cursor.pack_manifest_sha256:
        raise ValueError("rank cursor pack identity differs from shared training state")
    if cursor.lanes_sha256 != state.cursor.lanes_sha256:
        raise ValueError("rank cursor lane identity differs from shared training state")
    if _canonical_json(cursor.canonical()) != _canonical_json(state.cursor.canonical()):
        raise ValueError("rank cursor progress differs from shared training cursor")
    if type(cumulative_tokens) is not int or cumulative_tokens < 0:
        raise ValueError("rank cumulative tokens must be a nonnegative integer")
    if (not isinstance(sampler_spec_sha256, str) or len(sampler_spec_sha256) != 64
            or any(character not in "0123456789abcdef" for character in sampler_spec_sha256)):
        raise ValueError("production sampler identity must be a lowercase SHA-256")
    shard_identity = rank_data_shard_identity(
        cursor=cursor, rank=rank, world_size=world_size,
    )
    if (not isinstance(next_microstep_sha256, str) or len(next_microstep_sha256) != 64
            or any(character not in "0123456789abcdef" for character in next_microstep_sha256)):
        raise ValueError("next-microstep identity must be a lowercase SHA-256")
    payload = {
        "schema": PRODUCTION_RANK_RESUME_SCHEMA,
        "rank": rank,
        "world_size": world_size,
        "topology": topology,
        "checkpoint_global_update": state.global_update,
        "data_shard_identity": shard_identity,
        "sampler_spec_sha256": sampler_spec_sha256,
        "cursor": cursor.canonical(),
        "cumulative_tokens": cumulative_tokens,
        "next_microstep_sha256": next_microstep_sha256,
    }
    return _canonical_json(payload), shard_identity, next_microstep_sha256


def build_production_distributed_payloads(
    *, state: TrainingState, topology: str, model_payload: bytes,
    optimizer_payload: bytes, scheduler_payload: bytes,
    rank_captures: Sequence[ProductionRankCapture],
    sampler: ProductionSampler,
) -> dict[str, bytes]:
    """Build the exact store v2 inventory from real bucket-cursor captures."""

    state.assert_valid()
    if not isinstance(state.cursor, BucketCursorState):
        raise ValueError("production distributed checkpoints require a bucket-lane cursor")
    if not isinstance(topology, str) or not topology:
        raise ValueError("distributed topology is required")
    if not isinstance(sampler, ProductionSampler):
        raise ValueError("distributed production checkpoint requires its sampler specification")
    if sampler.pack_manifest_sha256 != state.identities.pack_manifest_sha256:
        raise ValueError("production sampler pack identity differs from training state")
    if state.identities.sampler_spec_sha256 != sampler.sha256:
        raise ValueError("production sampler identity differs from training state")
    if not all(isinstance(payload, bytes) for payload in
               (model_payload, optimizer_payload, scheduler_payload)):
        raise ValueError("model, optimizer, and scheduler payloads must be bytes")
    if not rank_captures:
        raise ValueError("distributed checkpoint needs rank-local captures")
    world_size = len(rank_captures)
    if sampler.topology["replicas"] != world_size:
        raise ValueError("production sampler world size differs from rank captures")
    captures = sorted(rank_captures, key=lambda capture: capture.rank)
    if [capture.rank for capture in captures] != list(range(world_size)):
        raise ValueError("production rank captures must include every rank exactly once")
    model_sha256 = _sha256(model_payload)
    optimizer_sha256 = _sha256(optimizer_payload)
    rank_records: list[DistributedRankState] = []
    rank_payloads: dict[int, dict[str, bytes]] = {}
    for capture in captures:
        if capture.cumulative_tokens < 0 or not capture.rng_state:
            raise ValueError("rank token count and RNG state must be present")
        if _canonical_json(capture.cursor.canonical()) != _canonical_json(
            state.cursor.canonical()
        ):
            raise ValueError("rank cursor progress differs from shared training cursor")
        if capture.model_state_sha256 != model_sha256:
            raise ValueError(f"rank {capture.rank} model hash differs from the saved replica")
        if capture.optimizer_state_sha256 != optimizer_sha256:
            raise ValueError(f"rank {capture.rank} optimizer hash differs from the saved replica")
        if state.complete:
            expected_next_sha256 = terminal_microstep_fingerprint_sha256(
                rank=capture.rank,
                world_size=world_size,
                topology=topology,
                state=state,
            )
        else:
            expected_next_plan = sampler.materialize_update(state, rank=capture.rank)
            expected_next_sha256 = next_microstep_fingerprint_sha256(
                rank=capture.rank,
                world_size=world_size,
                topology=topology,
                checkpoint_global_update=state.global_update,
                cursor=state.cursor,
                microsteps=expected_next_plan.rank_microsteps,
            )
        if capture.next_microsteps:
            if state.complete:
                raise ValueError("complete checkpoint cannot capture a next microstep")
            captured_next_sha256 = next_microstep_fingerprint_sha256(
                rank=capture.rank,
                world_size=world_size,
                topology=topology,
                checkpoint_global_update=state.global_update,
                cursor=capture.cursor,
                microsteps=capture.next_microsteps,
            )
            if captured_next_sha256 != expected_next_sha256:
                raise ValueError(f"rank {capture.rank} captured next microstep differs from sampler")
        if (capture.next_microstep_sha256 is not None
                and capture.next_microstep_sha256 != expected_next_sha256):
            raise ValueError(f"rank {capture.rank} next-microstep fingerprint differs from sampler")
        cursor_bytes, shard_identity, next_batch_sha256 = _rank_cursor_payload(
            rank=capture.rank,
            world_size=world_size,
            topology=topology,
            state=state,
            cursor=capture.cursor,
            sampler_spec_sha256=sampler.sha256,
            cumulative_tokens=capture.cumulative_tokens,
            next_microstep_sha256=expected_next_sha256,
        )
        rank_payloads[capture.rank] = {
            "rng_state": capture.rng_state,
            "cursor_state": cursor_bytes,
        }
        rng_sha256 = _sha256(capture.rng_state)
        cursor_sha256 = _sha256(cursor_bytes)
        rank_records.append(DistributedRankState(
            schema="anra-v5-distributed-rank-state/v2",
            rank=capture.rank,
            world_size=world_size,
            global_update=state.global_update,
            cumulative_tokens=capture.cumulative_tokens,
            data_shard_identity=shard_identity,
            rng_state_sha256=rng_sha256,
            cursor_state_sha256=cursor_sha256,
            rank_state_sha256=rank_state_payload_sha256(
                rng_state=capture.rng_state, cursor_state=cursor_bytes,
            ),
            model_state_sha256=model_sha256,
            optimizer_state_sha256=optimizer_sha256,
            next_batch_sha256=next_batch_sha256,
            collective_receipt_sha256=capture.collective_receipt_sha256,
        ))
    metadata = ReplicatedTrainingCheckpoint(
        schema=DISTRIBUTED_CHECKPOINT_SCHEMA,
        parent_checkpoint_sha256=state.parent_checkpoint_sha256,
        global_update=state.global_update,
        global_tokens=state.cumulative_tokens,
        world_size=world_size,
        topology=topology,
        optimizer_layout="replicated",
        training_state_sha256=state.sha256(),
        model_state_sha256=model_sha256,
        optimizer_state_sha256=optimizer_sha256,
        collective_barrier_sha256=collective_barrier_sha256(
            global_update=state.global_update,
            world_size=world_size,
            topology=topology,
            ranks=rank_records,
        ),
        ranks=tuple(rank_records),
    )
    metadata.assert_valid(training_state=state, model_payload=model_payload,
                          optimizer_payload=optimizer_payload)
    rng_cursor_bundle = encode_rank_state_bundle(rank_payloads, world_size=world_size)
    return {
        "model.bin": model_payload,
        "optimizer.bin": optimizer_payload,
        "scheduler.json": scheduler_payload,
        "cursor.json": _canonical_json(state.cursor.canonical()),
        "ledger.json": _canonical_json(dict(state.tokens_by_source)),
        "training_state.json": _canonical_json(state.canonical()),
        "distributed.json": _canonical_json(metadata.canonical()),
        "rank_states.bin": rng_cursor_bundle,
    }


def decode_production_rank_cursor_receipt(
    payload: bytes, *, rank: int, world_size: int, topology: str,
    checkpoint_global_update: int,
) -> tuple[BucketCursorState, str, str]:
    """Validate and decode one rank's serialized production bucket cursor."""

    cursor, sampler_sha, next_sha, _cumulative_tokens = (
        decode_production_rank_checkpoint_receipt(
            payload,
            rank=rank,
            world_size=world_size,
            topology=topology,
            checkpoint_global_update=checkpoint_global_update,
        )
    )
    return cursor, sampler_sha, next_sha


def decode_production_rank_checkpoint_receipt(
    payload: bytes, *, rank: int, world_size: int, topology: str,
    checkpoint_global_update: int,
) -> tuple[BucketCursorState, str, str, int]:
    """Decode the rank cursor, sampler, next-batch, and cumulative-token receipt."""

    try:
        value = json.loads(payload)
    except (TypeError, ValueError) as exc:
        raise ValueError("production rank cursor is not valid JSON") from exc
    expected = {
        "schema", "rank", "world_size", "topology", "checkpoint_global_update",
        "data_shard_identity", "sampler_spec_sha256", "cursor", "cumulative_tokens",
        "next_microstep_sha256",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("production rank cursor fields do not match schema")
    if (
        value["schema"] != PRODUCTION_RANK_RESUME_SCHEMA
        or type(value["rank"]) is not int or value["rank"] != rank
        or type(value["world_size"]) is not int or value["world_size"] != world_size
        or value["topology"] != topology
        or type(value["checkpoint_global_update"]) is not int
        or value["checkpoint_global_update"] != checkpoint_global_update
    ):
        raise ValueError("production rank cursor disagrees with expected runtime identity")
    if not isinstance(value["cursor"], dict):
        raise ValueError("production rank cursor state must be an object")
    cumulative_tokens = value["cumulative_tokens"]
    if type(cumulative_tokens) is not int or cumulative_tokens < 0:
        raise ValueError("production rank cumulative-token receipt is invalid")
    cursor = BucketCursorState.from_dict(value["cursor"])
    shard_identity = rank_data_shard_identity(
        cursor=cursor, rank=rank, world_size=world_size,
    )
    if value["data_shard_identity"] != shard_identity:
        raise ValueError("production rank cursor data-shard identity mismatch")
    sampler_sha = value["sampler_spec_sha256"]
    if (not isinstance(sampler_sha, str) or len(sampler_sha) != 64
            or any(character not in "0123456789abcdef" for character in sampler_sha)):
        raise ValueError("production rank cursor sampler identity is invalid")
    next_sha = value["next_microstep_sha256"]
    if (
        not isinstance(next_sha, str)
        or len(next_sha) != 64
        or any(character not in "0123456789abcdef" for character in next_sha)
    ):
        raise ValueError("production rank cursor next-microstep hash is invalid")
    if payload != _canonical_json(value):
        raise ValueError("production rank cursor must use canonical JSON")
    return cursor, sampler_sha, next_sha, cumulative_tokens


def decode_production_rank_cursor(
    payload: bytes, *, rank: int, world_size: int, topology: str,
    checkpoint_global_update: int,
) -> tuple[BucketCursorState, str]:
    """Decode a rank cursor while preserving the original two-value API."""

    cursor, _sampler_sha, next_sha = decode_production_rank_cursor_receipt(
        payload,
        rank=rank,
        world_size=world_size,
        topology=topology,
        checkpoint_global_update=checkpoint_global_update,
    )
    return cursor, next_sha


def capture_cpu_rng_state(torch_module: Any) -> bytes:
    """Capture CPU RNG bytes only; device RNG APIs need separate qualification."""

    state = torch_module.get_rng_state()
    if str(getattr(state, "device", "cpu")) != "cpu":
        raise RuntimeError("CPU RNG capture returned a non-CPU state")
    return bytes(state.detach().cpu().numpy().tobytes())


def restore_cpu_rng_state(torch_module: Any, payload: bytes) -> None:
    """Restore CPU RNG bytes; refuses to guess CUDA or XLA RNG semantics."""

    if not payload:
        raise ValueError("CPU RNG payload cannot be empty")
    tensor = torch_module.tensor(list(payload), dtype=torch_module.uint8, device="cpu")
    torch_module.set_rng_state(tensor)


def restore_production_rank(
    *, store: CheckpointStore, rank: int, expected_world_size: int,
    expected_topology: str, runtime: str, torch_module: Any,
    sampler: ProductionSampler,
    checkpoint_sha256: str | None = None,
    restore_rng: bool = True,
    rng_state_adapter: Any | None = None,
) -> ProductionRankResume:
    """Verify one rank's continuation and optionally apply its RNG state.

    CPU uses the built-in byte-exact generator API. Other device runtimes need
    an explicit adapter with a ``restore_rng_state(payload)`` implementation;
    runtime support still requires target-side next-batch parity evidence.
    """

    if runtime != "cpu" and not callable(
        getattr(rng_state_adapter, "restore_rng_state", None)
    ):
        raise RuntimeError(
            f"production rank resume runtime {runtime!r} needs an explicit RNG adapter")
    if sampler.topology["replicas"] != expected_world_size:
        raise ValueError("production sampler world size differs from the checkpoint request")
    state, metadata, rank_payloads, shared_payloads = store.restore_distributed_artifacts(
        rank=rank,
        expected_world_size=expected_world_size,
        expected_topology=expected_topology,
        checkpoint_sha256=checkpoint_sha256,
    )
    if state.identities.sampler_spec_sha256 != sampler.sha256:
        raise ValueError("production sampler identity differs from training state")
    cursor, stored_sampler_sha256, stored_next_sha256, cumulative_tokens = (
        decode_production_rank_checkpoint_receipt(
        rank_payloads["cursor_state"],
        rank=rank,
        world_size=expected_world_size,
        topology=expected_topology,
        checkpoint_global_update=state.global_update,
        )
    )
    rank_metadata = metadata.ranks[rank]
    if cumulative_tokens != rank_metadata.cumulative_tokens:
        raise ValueError("rank cursor cumulative tokens disagree with distributed metadata")
    shard_identity = rank_data_shard_identity(
        cursor=cursor, rank=rank, world_size=expected_world_size,
    )
    if shard_identity != rank_metadata.data_shard_identity:
        raise ValueError("rank cursor data-shard identity disagrees with distributed metadata")
    if stored_next_sha256 != rank_metadata.next_batch_sha256:
        raise ValueError("rank cursor next-microstep hash disagrees with distributed metadata")
    if stored_sampler_sha256 != sampler.sha256:
        raise ValueError("production sampler specification differs from the checkpoint")
    if _canonical_json(cursor.canonical()) != _canonical_json(state.cursor.canonical()):
        raise ValueError("rank cursor progress differs from shared training cursor")
    if state.complete:
        observed = terminal_microstep_fingerprint_sha256(
            rank=rank,
            world_size=expected_world_size,
            topology=expected_topology,
            state=state,
        )
    else:
        planned = sampler.materialize_update(state, rank=rank)
        if _canonical_json(planned.end_cursor.canonical()) == _canonical_json(cursor.canonical()):
            raise ValueError("production sampler did not advance the checkpoint cursor")
        observed = next_microstep_fingerprint_sha256(
            rank=rank,
            world_size=expected_world_size,
            topology=expected_topology,
            checkpoint_global_update=state.global_update,
            cursor=cursor,
            microsteps=planned.rank_microsteps,
        )
    metadata.assert_next_batch(rank=rank, observed_sha256=observed)
    # Validate sampler, cursor, and actual next data before optional RNG mutation.
    if restore_rng:
        if runtime == "cpu":
            restore_cpu_rng_state(torch_module, rank_payloads["rng_state"])
        else:
            rng_state_adapter.restore_rng_state(rank_payloads["rng_state"])
    return ProductionRankResume(
        state=state,
        metadata=metadata,
        rank=rank,
        cursor=cursor,
        cumulative_tokens=cumulative_tokens,
        sampler_spec_sha256=stored_sampler_sha256,
        next_microstep_sha256=observed,
        shared_payloads=MappingProxyType(dict(shared_payloads)),
        rng_state=bytes(rank_payloads["rng_state"]),
    )


def restore_production_rank_into_backend(
    *,
    backend: ProductionTrainingBackend,
    store: CheckpointStore,
    rank: int,
    expected_world_size: int,
    expected_topology: str,
    sampler: ProductionSampler,
    checkpoint_sha256: str | None = None,
    expected_learning_rate: float | None = None,
    runtime: str = "cpu",
    rng_state_adapter: Any | None = None,
) -> ProductionRankResume:
    """Verify v2 continuation, load shared state, and restore rank RNG last."""

    resume = restore_production_rank(
        store=store,
        rank=rank,
        expected_world_size=expected_world_size,
        expected_topology=expected_topology,
        runtime=runtime,
        torch_module=backend.torch,
        sampler=sampler,
        checkpoint_sha256=checkpoint_sha256,
        restore_rng=False,
        rng_state_adapter=rng_state_adapter,
    )
    restore_production_shared(
        backend,
        payloads=resume.shared_payloads,
        expected_learning_rate=expected_learning_rate,
    )
    if runtime == "cpu":
        restore_cpu_rng_state(backend.torch, resume.rng_state)
    else:
        if not callable(getattr(rng_state_adapter, "restore_rng_state", None)):
            raise RuntimeError(f"runtime {runtime!r} has no rank RNG restore adapter")
        rng_state_adapter.restore_rng_state(resume.rng_state)
    return resume


__all__ = [
    "NEXT_MICROSTEP_SCHEMA",
    "PRODUCTION_RANK_RESUME_SCHEMA",
    "ProductionRankCapture",
    "ProductionRankResume",
    "build_production_distributed_payloads",
    "aggregate_rng_state_sha256",
    "capture_cpu_rng_state",
    "decode_production_rank_cursor",
    "decode_production_rank_checkpoint_receipt",
    "decode_production_rank_cursor_receipt",
    "next_microstep_fingerprint_sha256",
    "rank_data_shard_identity",
    "restore_cpu_rng_state",
    "restore_production_rank",
    "restore_production_rank_into_backend",
    "terminal_microstep_fingerprint_sha256",
]
