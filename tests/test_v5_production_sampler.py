from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest

from v5_data.bucket_cursor import BUCKET_CURSOR_SCHEMA, BucketCursorState, build_bucket_lanes
from v5_data.pack import pack_documents
from v5_training.production_sampler import ProductionSampler
from v5_training.state import IDENTITY_SCHEMA, IdentityBindings, TrainingState


def _fixture(*, allow_replay: bool = True):
    documents = []
    cell_of_source = {}
    for family in ("alpha", "beta"):
        for index in range(4):
            source = f"{family}-source-{index}"
            documents.append((f"{family}-doc-{index}", [10 + index] * 298, source))
            cell_of_source[source] = (family, "")
    packed, _audit = pack_documents(
        documents,
        bos=2,
        eos=3,
        pad=0,
        sequences_per_shard=2,
        cell_of_source=cell_of_source,
    )
    pack_sha = hashlib.sha256(json.dumps(
        [json.loads(shard.payload_bytes()) for shard in packed],
        sort_keys=True,
        separators=(",", ":"),
    ).encode()).hexdigest()
    lanes, lane_receipt = build_bucket_lanes(
        packed,
        run_seed=31,
        pattern=[512],
        cell_of_source=cell_of_source,
        required_buckets={512},
    )
    cursor = BucketCursorState(
        BUCKET_CURSOR_SCHEMA,
        pack_sha,
        lane_receipt["lanes_sha256"],
        {key: [0, 0] for key in lanes},
        {},
        {},
        0,
        0,
    )
    identities = IdentityBindings(
        schema=IDENTITY_SCHEMA,
        source_commit="0123456789abcdef0123456789abcdef01234567",
        model_spec_sha256="1" * 64,
        tokenizer_sha256="2" * 64,
        data_manifest_sha256="3" * 64,
        pack_manifest_sha256=pack_sha,
        run_spec_sha256="4" * 64,
        optimizer_spec_sha256="5" * 64,
        schedule_spec_sha256="6" * 64,
        curriculum_spec_sha256="7" * 64,
    )
    state = TrainingState.initial(
        lineage_id="pure-production-sampler",
        token_budget=3000,
        tokens_per_update=1200,
        cursor=cursor,
        rng_state_sha256="8" * 64,
        curriculum_phase="sampler-test",
        identities=identities,
    )
    sampler = ProductionSampler(
        packed=packed,
        run_seed=31,
        topology={
            "replicas": 2,
            "global_tokens_per_microstep": 600,
            "global_tokens_per_update": 1200,
            "supercycle": [512],
        },
        pack_manifest_sha256=pack_sha,
        cell_of_source=cell_of_source,
        mixture_fractions={"alpha": 0.5, "beta": 0.5},
        cognition_fractions=None,
        cognition_mapped=False,
        allow_replay=allow_replay,
    )
    return state, sampler


def _advance(state: TrainingState, plan) -> TrainingState:
    ledger: dict[str, int] = {}
    for _bucket, _family, _subfamily, window in plan.windows:
        for source, amount in window.tokens_by_source.items():
            ledger[source] = ledger.get(source, 0) + amount
    return state.advance(
        tokens_by_source=ledger,
        cursor=plan.end_cursor,
        rng_state_sha256="9" * 64,
        parent_checkpoint_sha256=None,
    )


def test_production_planner_reconstructs_multimicrostep_mixture_and_replay() -> None:
    state, sampler = _fixture()

    first = sampler.materialize_update(state)
    first_again = sampler.materialize_update(state)
    assert first.end_cursor == first_again.end_cursor
    assert first.rank_microsteps == first_again.rank_microsteps
    assert [family for _bucket, family, _sub, _window in first.windows] == ["alpha", "beta"]
    assert len(first.rank_microsteps) == 2
    assert first.expected_real_tokens == 1200
    assert first.eligible_tokens > 0
    assert state.cumulative_tokens == 0
    assert all(position == [0, 0] for position in state.cursor.positions.values())

    state = _advance(state, first)
    second = sampler.materialize_update(state)
    assert [family for _bucket, family, _sub, _window in second.windows] == ["alpha", "beta"]
    state = _advance(state, second)

    checkpoint_cursor = state.cursor.canonical()
    third = sampler.materialize_update(state)
    assert third.expected_real_tokens == 600
    assert [family for _bucket, family, _sub, _window in third.windows] == ["alpha"]
    assert third.end_cursor.epoch == 1
    assert third.end_cursor.replay_count == 1
    assert [dict(event) for event in third.replay_events] == [
        {"epoch": 1, "at_cumulative_tokens": 2400},
    ]
    assert state.cursor.canonical() == checkpoint_cursor


def test_production_planner_rejects_replay_without_mutating_cursor() -> None:
    state, replaying_sampler = _fixture()
    state = _advance(state, replaying_sampler.materialize_update(state))
    state = _advance(state, replaying_sampler.materialize_update(state))
    before = state.cursor.canonical()
    _state, no_replay_sampler = _fixture(allow_replay=False)

    with pytest.raises(ValueError, match="bounded replay not permitted"):
        no_replay_sampler.materialize_update(state)

    assert state.cursor.canonical() == before


def test_production_planner_binds_seed_topology_and_replay_policy() -> None:
    state, sampler = _fixture()
    _other_state, no_replay_sampler = _fixture(allow_replay=False)
    assert sampler.sha256 != no_replay_sampler.sha256

    bound_state = replace(
        state,
        identities=replace(state.identities, sampler_spec_sha256=sampler.sha256),
    )
    with pytest.raises(ValueError, match="sampler identity differs from training state"):
        no_replay_sampler.materialize_update(bound_state)

    with pytest.raises(ValueError, match="sampler pack identity"):
        changed = replace(sampler, pack_manifest_sha256="a" * 64)
        changed.materialize_update(state)
