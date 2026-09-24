from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

import pytest

from v5_training.checkpoint import CheckpointStore, REQUIRED_COMPONENTS
from v5_training.distributed_checkpoint import aggregate_rng_state_sha256
from v5_training.distributed import (
    CHECKPOINT_CLAIM_SCHEMA,
    UPDATE_DECISION_SCHEMA,
    UPDATE_LOSS_SCHEMA,
    CheckpointCoordinationError,
    RankZeroCheckpointCoordinator,
    _validate_publish_claims,
    _validate_update_loss_claims,
    _validate_update_decisions,
)
from v5_training.runner import RunController
from v5_training.state import (
    CURSOR_SCHEMA,
    IDENTITY_SCHEMA,
    CursorState,
    IdentityBindings,
    TrainingState,
)
from v5_training.trainer import BackendReport, train


PACK_SHA = "1" * 64


def _initial_state(
    *, lineage: str = "distributed-checkpoint-test", token_budget: int = 5,
) -> TrainingState:
    identities = IdentityBindings(
        schema=IDENTITY_SCHEMA,
        source_commit="0123456789abcdef0123456789abcdef01234567",
        model_spec_sha256="2" * 64,
        tokenizer_sha256="3" * 64,
        data_manifest_sha256="4" * 64,
        pack_manifest_sha256=PACK_SHA,
        run_spec_sha256="5" * 64,
        optimizer_spec_sha256="6" * 64,
        schedule_spec_sha256="7" * 64,
        curriculum_spec_sha256="8" * 64,
    )
    return TrainingState.initial(
        lineage_id=lineage,
        token_budget=token_budget,
        tokens_per_update=5,
        cursor=CursorState(CURSOR_SCHEMA, PACK_SHA, 0, 0, 0),
        rng_state_sha256="9" * 64,
        curriculum_phase="test",
        identities=identities,
    )


def _advanced_state(
    state: TrainingState | None = None,
    *,
    parent_checkpoint_sha256: str | None = None,
) -> TrainingState:
    before = state or _initial_state()
    return before.advance(
        tokens_by_source={"synthetic": 5},
        cursor=CursorState(
            CURSOR_SCHEMA, PACK_SHA, before.cursor.shard_ordinal,
            before.cursor.sequence_ordinal, before.cursor.token_offset + 1,
        ),
        rng_state_sha256=f"{before.global_update + 10:064x}",
        parent_checkpoint_sha256=parent_checkpoint_sha256,
    )


def _claim(state: TrainingState, rank: int, world_size: int = 2) -> dict[str, object]:
    return {
        "schema": CHECKPOINT_CLAIM_SCHEMA,
        "rank": rank,
        "world_size": world_size,
        "lineage_id": state.lineage_id,
        "generation": state.generation,
        "global_update": state.global_update,
        "cumulative_tokens": state.cumulative_tokens,
        "state_sha256": state.sha256(),
        "parent_checkpoint_sha256": state.parent_checkpoint_sha256,
        "local_error": None,
    }


def _decision(
    state: TrainingState,
    rank: int,
    *,
    checkpoint: bool = True,
    stop: bool = False,
    world_size: int = 2,
) -> dict[str, object]:
    return {
        "schema": UPDATE_DECISION_SCHEMA,
        "rank": rank,
        "world_size": world_size,
        "lineage_id": state.lineage_id,
        "generation": state.generation,
        "global_update": state.global_update,
        "cumulative_tokens": state.cumulative_tokens,
        "state_sha256": state.sha256(),
        "checkpoint_requested": checkpoint,
        "stop_requested": stop,
        "local_error": None,
    }


def _payloads(state: TrainingState) -> dict[str, bytes]:
    canonical = lambda value: json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    payloads = {
        "model.bin": b"replicated model",
        "optimizer.bin": b"replicated optimizer",
        "scheduler.json": b"{}",
        "rng.bin": b"rank-zero rng",
        "cursor.json": canonical(asdict(state.cursor)),
        "ledger.json": canonical(dict(state.tokens_by_source)),
        "training_state.json": canonical(state.canonical()),
    }
    assert set(payloads) == REQUIRED_COMPONENTS
    return payloads


class _SequentialMeshReduce:
    """Deterministic two-rank collective seam for coordinator unit tests."""

    def __init__(self, *, claims: list[dict[str, object]] | None = None, world_size: int = 2) -> None:
        self.claims = claims
        self.world_size = world_size
        self.results: dict[str, object] = {}
        self.rng_hashes: list[str] | None = None
        self.rng_errors: list[str | None] | None = None
        self.rank_tokens: list[int] | None = None

    def for_rank(self, rank: int):
        def mesh_reduce(tag: str, value: object, reduce_fn):
            if tag.endswith("-claims"):
                values = self.claims or [dict(value, rank=i) for i in range(self.world_size)]
                assert value == values[rank]
                return reduce_fn(values)
            if tag.endswith("-decisions"):
                values = [dict(value, rank=i) for i in range(self.world_size)]
                return reduce_fn(values)
            if tag.startswith("anra-v5-training-rng-"):
                values = [
                    dict(value, rank=i,
                         rng_state_sha256=(self.rng_hashes[i]
                                           if self.rng_hashes is not None
                                           else value["rng_state_sha256"]),
                         local_error=(self.rng_errors[i]
                                      if self.rng_errors is not None
                                      else value["local_error"]),
                         local_real_tokens=(self.rank_tokens[i]
                                            if self.rank_tokens is not None
                                            else value["local_real_tokens"]))
                    for i in range(self.world_size)
                ]
                return reduce_fn(values)
            if tag.endswith("-publication") or tag.endswith("-callbacks"):
                if rank == 0:
                    values = [value, *([None] * (self.world_size - 1))]
                    self.results[tag] = reduce_fn(values)
                    return self.results[tag]
                return reduce_fn([self.results[tag], *([None] * (self.world_size - 1))])
            if tag.startswith("anra-v5-checkpoint-observation-"):
                values = [dict(value, rank=i) for i in range(self.world_size)]
                return reduce_fn(values)
            raise AssertionError(f"unexpected collective tag: {tag}")

        return mesh_reduce


def _coordinators(
    state: TrainingState,
    *,
    mesh: _SequentialMeshReduce | None = None,
) -> tuple[RankZeroCheckpointCoordinator, ...]:
    mesh = mesh or _SequentialMeshReduce(claims=[_claim(state, rank) for rank in range(2)])
    return tuple(
        RankZeroCheckpointCoordinator(
            rank=rank, world_size=2, mesh_reduce=mesh.for_rank(rank),
        )
        for rank in range(2)
    )


def test_publish_claims_reject_missing_duplicate_and_divergent_ranks() -> None:
    state = _advanced_state()
    claims = [_claim(state, 0), _claim(state, 1)]
    assert _validate_publish_claims(claims, world_size=2)["status"] == "READY"

    assert "full world" in _validate_publish_claims(claims[:1], world_size=2)["reason"]
    assert "every rank" in _validate_publish_claims([claims[0], claims[0]], world_size=2)["reason"]

    divergent = [dict(claim) for claim in claims]
    divergent[1]["cumulative_tokens"] = 10
    assert "different state or parent" in _validate_publish_claims(
        divergent, world_size=2,
    )["reason"]

    decisions = [_decision(state, rank) for rank in range(2)]
    decisions[1]["checkpoint_requested"] = False
    assert "checkpoint, or stop action" in _validate_update_decisions(
        decisions, world_size=2,
    )["reason"]


def test_distributed_loss_aggregation_handles_uneven_rank_counts() -> None:
    contributions = [(5, 10.0), (3, 12.0)]
    claims = [{
        "schema": UPDATE_LOSS_SCHEMA,
        "rank": rank,
        "world_size": 2,
        "global_update": 1,
        "global_tokens": 8,
        "local_eligible_tokens": eligible_tokens,
        "local_loss_numerator": numerator,
    } for rank, (eligible_tokens, numerator) in enumerate(contributions)]
    expected = _validate_update_loss_claims(claims, world_size=2)
    assert expected["status"] == "READY"
    assert expected["global_loss"] == 2.75
    assert expected["loss_numerator"] == 22.0
    assert expected["rank_contributions"] == [
        {"rank": 0, "eligible_tokens": 5, "loss_numerator": 10.0},
        {"rank": 1, "eligible_tokens": 3, "loss_numerator": 12.0},
    ]

    coordinators = []
    for rank in range(2):
        def mesh_reduce(tag, value, reduce_fn, *, rank=rank):
            assert tag == "anra-v5-training-loss-00000000"
            assert value == claims[rank]
            return reduce_fn(claims)

        coordinators.append(RankZeroCheckpointCoordinator(
            rank=rank, world_size=2, mesh_reduce=mesh_reduce,
        ))
    observed = [coordinator.aggregate_update_loss(
        global_update=1,
        global_tokens=8,
        local_eligible_tokens=contributions[rank][0],
        local_loss_numerator=contributions[rank][1],
    ) for rank, coordinator in enumerate(coordinators)]
    assert observed == [expected, expected]

    bad_claims = [dict(claim) for claim in claims]
    bad_claims[1]["local_eligible_tokens"] = 2
    rejected = _validate_update_loss_claims(bad_claims, world_size=2)
    assert rejected["status"] == "REJECTED"
    assert "denominator" in rejected["reason"]


def test_distributed_loss_aggregation_accepts_empty_rank_but_not_empty_update() -> None:
    claims = [{
        "schema": UPDATE_LOSS_SCHEMA,
        "rank": rank,
        "world_size": 2,
        "global_update": 1,
        "global_tokens": 5,
        "local_eligible_tokens": tokens,
        "local_loss_numerator": numerator,
    } for rank, (tokens, numerator) in enumerate(((0, 0.0), (5, 12.5)))]

    ready = _validate_update_loss_claims(claims, world_size=2)
    assert ready["status"] == "READY"
    assert ready["global_loss"] == 2.5
    assert ready["rank_contributions"] == [
        {"rank": 0, "eligible_tokens": 0, "loss_numerator": 0.0},
        {"rank": 1, "eligible_tokens": 5, "loss_numerator": 12.5},
    ]

    empty_update = [dict(claim, local_eligible_tokens=0,
                         local_loss_numerator=0.0) for claim in claims]
    rejected = _validate_update_loss_claims(empty_update, world_size=2)
    assert rejected["status"] == "REJECTED"
    assert "denominator" in rejected["reason"]

    malformed_empty_rank = [dict(claim) for claim in claims]
    malformed_empty_rank[0]["local_loss_numerator"] = 1.0
    rejected = _validate_update_loss_claims(malformed_empty_rank, world_size=2)
    assert rejected["status"] == "REJECTED"
    assert "zero loss numerator" in rejected["reason"]

def test_update_rng_aggregation_binds_ordered_rank_local_states() -> None:
    local_hashes = ["a" * 64, "b" * 64]
    mesh = _SequentialMeshReduce()
    mesh.rng_hashes = local_hashes
    mesh.rank_tokens = [3, 2]
    coordinators = _coordinators(_initial_state(), mesh=mesh)
    expected = aggregate_rng_state_sha256(local_hashes)

    assert coordinators[0].aggregate_update_rng_state(
        local_rng_state_sha256=local_hashes[0],
        local_real_tokens=3,
        expected_global_tokens=5,
        expected_prior_global_tokens=0,
    ) == expected
    assert coordinators[1].aggregate_update_rng_state(
        local_rng_state_sha256=local_hashes[1],
        local_real_tokens=2,
        expected_global_tokens=5,
        expected_prior_global_tokens=0,
    ) == expected
    assert [coordinator.rank_cumulative_tokens for coordinator in coordinators] == [3, 2]

    mesh.rng_errors = [None, "RuntimeError: rank-local step failed"]
    with pytest.raises(CheckpointCoordinationError, match="rank 1: RuntimeError"):
        _coordinators(_initial_state(), mesh=mesh)[0].aggregate_update_rng_state(
            local_rng_state_sha256=local_hashes[0],
            local_real_tokens=3,
            expected_global_tokens=5,
            expected_prior_global_tokens=0,
        )


def test_rank_zero_publishes_once_and_all_ranks_receive_the_same_commit(tmp_path: Path) -> None:
    state = _advanced_state()
    coordinators = _coordinators(state)
    store = CheckpointStore(tmp_path, state.lineage_id)
    builds: list[int] = []
    callbacks: list[tuple[int, str]] = []

    def build(live: TrainingState) -> dict[str, bytes]:
        builds.append(0)
        return _payloads(live)

    sha_rank_0 = coordinators[0].publish_checkpoint(
        state=state, parent_checkpoint_sha256=None, store=store, payload_builder=build,
    )
    sha_rank_1 = coordinators[1].publish_checkpoint(
        state=state,
        parent_checkpoint_sha256=None,
        store=object(),  # The non-writer must not even inspect the store.
        payload_builder=lambda _state: (_ for _ in ()).throw(AssertionError("non-writer built payloads")),
    )
    assert sha_rank_0 == sha_rank_1 == store.latest_sha256()
    assert builds == [0]

    def committed(live: TrainingState, checkpoint_sha: str) -> None:
        callbacks.append((live.global_update, checkpoint_sha))

    coordinators[0].run_writer_callback(
        callback=committed, state=state, checkpoint_sha256=sha_rank_0,
    )
    coordinators[1].run_writer_callback(
        callback=None,
        state=state,
        checkpoint_sha256=sha_rank_1,
    )
    restored, restored_payloads = store.restore()
    assert restored == state
    assert restored_payloads == _payloads(state)
    assert callbacks == [(1, sha_rank_0)]


def test_rank_zero_publish_failure_is_broadcast_without_nonwriter_writes() -> None:
    state = _advanced_state()
    coordinators = _coordinators(state)

    class FailingStore:
        def publish(self, **_kwargs: Any) -> str:
            raise OSError("disk unavailable")

    with pytest.raises(CheckpointCoordinationError, match="disk unavailable"):
        coordinators[0].publish_checkpoint(
            state=state,
            parent_checkpoint_sha256=None,
            store=FailingStore(),
            payload_builder=lambda _state: _payloads(state),
        )
    with pytest.raises(CheckpointCoordinationError, match="disk unavailable"):
        coordinators[1].publish_checkpoint(
            state=state,
            parent_checkpoint_sha256=None,
            store=object(),
            payload_builder=lambda _state: (_ for _ in ()).throw(AssertionError("non-writer built payloads")),
        )


def test_rank_zero_callback_failure_fans_out_and_consumes_callback_phase(tmp_path: Path) -> None:
    state = _advanced_state()
    coordinators = _coordinators(state)
    store = CheckpointStore(tmp_path, state.lineage_id)
    sha = coordinators[0].publish_checkpoint(
        state=state, parent_checkpoint_sha256=None, store=store,
        payload_builder=_payloads,
    )
    assert coordinators[1].publish_checkpoint(
        state=state, parent_checkpoint_sha256=None, store=object(),
        payload_builder=lambda _state: (_ for _ in ()).throw(AssertionError("non-writer built payloads")),
    ) == sha
    called: list[int] = []

    def failing_callback(_live: TrainingState, _checkpoint_sha: str) -> None:
        called.append(0)
        raise OSError("mirror unavailable")

    with pytest.raises(CheckpointCoordinationError, match="mirror unavailable"):
        coordinators[0].run_writer_callback(
            callback=failing_callback, state=state, checkpoint_sha256=sha,
        )
    with pytest.raises(CheckpointCoordinationError, match="mirror unavailable"):
        coordinators[1].run_writer_callback(
            callback=None, state=state, checkpoint_sha256=sha,
        )
    assert called == [0]
    for coordinator in coordinators:
        with pytest.raises(ValueError, match="no preceding coordinated publication"):
            coordinator.run_writer_callback(
                callback=None, state=state, checkpoint_sha256=sha,
            )


def test_back_to_back_commits_keep_parent_fencing_and_collective_tags(tmp_path: Path) -> None:
    initial = _initial_state(lineage="distributed-two-generation", token_budget=10)
    first = _advanced_state(initial)
    mesh = _SequentialMeshReduce()
    coordinators = _coordinators(first, mesh=mesh)
    store = CheckpointStore(tmp_path, initial.lineage_id)

    first_sha = coordinators[0].publish_checkpoint(
        state=first, parent_checkpoint_sha256=None, store=store,
        payload_builder=_payloads,
    )
    assert coordinators[1].publish_checkpoint(
        state=first, parent_checkpoint_sha256=None, store=object(),
        payload_builder=lambda _state: (_ for _ in ()).throw(AssertionError("non-writer built payloads")),
    ) == first_sha

    second = _advanced_state(first, parent_checkpoint_sha256=first_sha)
    second_sha = coordinators[0].publish_checkpoint(
        state=second, parent_checkpoint_sha256=first_sha, store=store,
        payload_builder=_payloads,
    )
    assert coordinators[1].publish_checkpoint(
        state=second, parent_checkpoint_sha256=first_sha, store=object(),
        payload_builder=lambda _state: (_ for _ in ()).throw(AssertionError("non-writer built payloads")),
    ) == second_sha
    assert first_sha != second_sha
    assert store.restore()[0] == second


def test_rank_local_preflight_failure_is_shared_before_any_checkpoint_write() -> None:
    state = _advanced_state()
    claims = [_claim(state, rank) for rank in range(2)]
    claims[0].update({
        "lineage_id": "",
        "generation": -1,
        "global_update": -1,
        "cumulative_tokens": -1,
        "state_sha256": "0" * 64,
        "local_error": "ValueError: deliberately invalid state",
    })
    mesh = _SequentialMeshReduce(claims=claims)
    coordinators = tuple(
        RankZeroCheckpointCoordinator(
            rank=rank, world_size=2, mesh_reduce=mesh.for_rank(rank),
        )
        for rank in range(2)
    )

    class InvalidState:
        def assert_valid(self) -> None:
            raise ValueError("deliberately invalid state")

    with pytest.raises(CheckpointCoordinationError, match="rank 0: ValueError"):
        coordinators[0].publish_checkpoint(
            state=InvalidState(),
            parent_checkpoint_sha256=None,
            store=object(),
            payload_builder=lambda _state: (_ for _ in ()).throw(AssertionError("must not write")),
        )
    with pytest.raises(CheckpointCoordinationError, match="rank 0: ValueError"):
        coordinators[1].publish_checkpoint(
            state=state,
            parent_checkpoint_sha256=None,
            store=object(),
            payload_builder=lambda _state: (_ for _ in ()).throw(AssertionError("must not write")),
        )
def test_trainer_uses_one_rank_zero_commit_and_keeps_all_rank_hooks_in_sync(tmp_path: Path) -> None:
    initial = _initial_state(lineage="trainer-rank-zero-test")
    local_advanced = _advanced_state(initial)
    local_rngs = ["d" * 64, "e" * 64]
    aggregate_rng = aggregate_rng_state_sha256(local_rngs)
    advanced = initial.advance(
        tokens_by_source={"synthetic": 5},
        cursor=local_advanced.cursor,
        rng_state_sha256=aggregate_rng,
        parent_checkpoint_sha256=None,
    )
    mesh = _SequentialMeshReduce()
    mesh.rng_hashes = local_rngs
    mesh.rank_tokens = [3, 2]
    coordinators = _coordinators(advanced, mesh=mesh)
    store = CheckpointStore(tmp_path, initial.lineage_id)
    observed: list[int] = []
    writer_callbacks: list[int] = []

    def backend_step(_state: TrainingState, *, rank: int) -> BackendReport:
        return BackendReport(
            tokens_by_source={"synthetic": 5},
            cursor=advanced.cursor,
            rng_state_sha256=local_rngs[rank],
            loss_finite=True,
            grad_finite=True,
            grad_norm_post_clip=0.5,
            tied_preserved=True,
            local_real_tokens=mesh.rank_tokens[rank],
        )

    def run_rank(rank: int) -> TrainingState:
        controller = RunController(target_update=1)
        controller.start()
        return train(
            state=initial,
            controller=controller,
            store=store if rank == 0 else object(),
            payload_builder=lambda live: _payloads(live),
            backend_step=lambda live: backend_step(live, rank=rank),
            updates=1,
            checkpoint_every=1,
            checkpoint_coordinator=coordinators[rank],
            on_committed=lambda live, _sha: writer_callbacks.append(live.global_update),
            on_checkpoint_observed=lambda live, _sha: observed.append(live.global_update),
        )

    final_0 = run_rank(0)
    final_1 = run_rank(1)
    assert final_0 == final_1 == advanced
    assert store.restore()[0] == advanced
    assert writer_callbacks == [1]
    assert observed == [1, 1]
