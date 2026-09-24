from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import pytest

from v5_training.checkpoint import CheckpointStore, DISTRIBUTED_REQUIRED_COMPONENTS
from v5_training.checkpoint import REQUIRED_COMPONENTS
from v5_training.distributed_checkpoint import (
    DISTRIBUTED_CHECKPOINT_SCHEMA,
    DistributedRankState,
    ReplicatedTrainingCheckpoint,
    aggregate_rank_rng_sha256,
    collective_barrier_sha256,
    encode_rank_state_bundle,
    rank_state_payload_sha256,
)
from v5_training.state import (
    CURSOR_SCHEMA,
    IDENTITY_SCHEMA,
    CursorState,
    IdentityBindings,
    TrainingState,
)


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _distributed_fixture():
    model = b"replicated model bytes"
    optimizer = b"replicated Adam bytes"
    rank_payloads = {
        0: {"rng_state": b"rank-0 rng", "cursor_state": b"rank-0 cursor"},
        1: {"rng_state": b"rank-1 rng", "cursor_state": b"rank-1 cursor"},
    }
    ranks = tuple(
        DistributedRankState(
            schema="anra-v5-distributed-rank-state/v2",
            rank=rank,
            world_size=2,
            global_update=1,
            cumulative_tokens=(3 if rank == 0 else 2),
            data_shard_identity=f"pack-shard-{rank}",
            rng_state_sha256=_sha(rank_payloads[rank]["rng_state"]),
            cursor_state_sha256=_sha(rank_payloads[rank]["cursor_state"]),
            rank_state_sha256=rank_state_payload_sha256(**rank_payloads[rank]),
            model_state_sha256=_sha(model),
            optimizer_state_sha256=_sha(optimizer),
            next_batch_sha256=_sha(f"rank-{rank}-next-batch".encode()),
            collective_receipt_sha256=_sha(f"rank-{rank}-collective".encode()),
        )
        for rank in range(2)
    )
    identities = IdentityBindings(
        schema=IDENTITY_SCHEMA,
        source_commit="0123456789abcdef0123456789abcdef01234567",
        model_spec_sha256="2" * 64,
        tokenizer_sha256="3" * 64,
        data_manifest_sha256="4" * 64,
        pack_manifest_sha256="5" * 64,
        run_spec_sha256="6" * 64,
        optimizer_spec_sha256="7" * 64,
        schedule_spec_sha256="8" * 64,
        curriculum_spec_sha256="9" * 64,
    )
    initial = TrainingState.initial(
        lineage_id="distributed-resume",
        token_budget=10,
        tokens_per_update=5,
        cursor=CursorState(CURSOR_SCHEMA, identities.pack_manifest_sha256, 0, 0, 0),
        rng_state_sha256="a" * 64,
        curriculum_phase="qualification",
        identities=identities,
    )
    state = initial.advance(
        tokens_by_source={"corpus": 5},
        cursor=CursorState(CURSOR_SCHEMA, identities.pack_manifest_sha256, 0, 0, 5),
        rng_state_sha256=aggregate_rank_rng_sha256(ranks),
        parent_checkpoint_sha256=None,
    )
    rank_bundle = encode_rank_state_bundle(rank_payloads, world_size=2)
    metadata = ReplicatedTrainingCheckpoint(
        schema=DISTRIBUTED_CHECKPOINT_SCHEMA,
        parent_checkpoint_sha256=state.parent_checkpoint_sha256,
        global_update=state.global_update,
        global_tokens=state.cumulative_tokens,
        world_size=2,
        topology="xla-v2-8",
        optimizer_layout="replicated",
        training_state_sha256=state.sha256(),
        model_state_sha256=_sha(model),
        optimizer_state_sha256=_sha(optimizer),
        collective_barrier_sha256=collective_barrier_sha256(
            global_update=state.global_update,
            world_size=2,
            topology="xla-v2-8",
            ranks=ranks,
        ),
        ranks=ranks,
    )
    payloads = {
        "model.bin": model,
        "optimizer.bin": optimizer,
        "scheduler.json": b"{}",
        "cursor.json": _canonical(asdict(state.cursor)),
        "ledger.json": _canonical(dict(state.tokens_by_source)),
        "training_state.json": _canonical(state.canonical()),
        "distributed.json": _canonical(metadata.canonical()),
        "rank_states.bin": rank_bundle,
    }
    assert set(payloads) == DISTRIBUTED_REQUIRED_COMPONENTS
    return state, metadata, rank_payloads, payloads


def test_v2_store_round_trips_each_ranks_resume_bytes_and_checks_topology(tmp_path: Path) -> None:
    state, expected_metadata, rank_payloads, payloads = _distributed_fixture()
    store = CheckpointStore(tmp_path, state.lineage_id)
    checkpoint_sha = store.publish_distributed(
        state=state, payloads=payloads, expected_parent_sha256=None,
    )

    restored_state, metadata, rank_state = store.restore_distributed(
        rank=1,
        expected_world_size=2,
        expected_topology="xla-v2-8",
        checkpoint_sha256=checkpoint_sha,
    )
    assert restored_state == state
    assert metadata == expected_metadata
    assert rank_state == rank_payloads[1]
    metadata.assert_next_batch(rank=1, observed_sha256=metadata.ranks[1].next_batch_sha256)
    with pytest.raises(ValueError, match="next batch differs"):
        metadata.assert_next_batch(rank=1, observed_sha256="0" * 64)

    with pytest.raises(ValueError, match="topology"):
        store.restore_distributed(
            rank=1, expected_world_size=2, expected_topology="xla-v3-8",
        )
    with pytest.raises(ValueError, match="world size"):
        store.restore_distributed(
            rank=1, expected_world_size=4, expected_topology="xla-v2-8",
        )


def test_v2_publication_rejects_model_optimizer_and_rank_payload_mismatches(tmp_path: Path) -> None:
    state, _metadata, _rank_payloads, payloads = _distributed_fixture()
    store = CheckpointStore(tmp_path, state.lineage_id)

    changed_model = dict(payloads, **{"model.bin": b"different model"})
    with pytest.raises(ValueError, match="model hash"):
        store.publish_distributed(
            state=state, payloads=changed_model, expected_parent_sha256=None,
        )

    changed_optimizer = dict(payloads, **{"optimizer.bin": b"different optimizer"})
    with pytest.raises(ValueError, match="optimizer hash"):
        store.publish_distributed(
            state=state, payloads=changed_optimizer, expected_parent_sha256=None,
        )

    wrong_rank_bytes = encode_rank_state_bundle({
        0: {"rng_state": b"wrong rng", "cursor_state": b"rank-0 cursor"},
        1: {"rng_state": b"rank-1 rng", "cursor_state": b"rank-1 cursor"},
    }, world_size=2)
    changed_rank = dict(payloads, **{"rank_states.bin": wrong_rank_bytes})
    with pytest.raises(ValueError, match="RNG payload hash mismatch"):
        store.publish_distributed(
            state=state, payloads=changed_rank, expected_parent_sha256=None,
        )
    assert store.latest_sha256() is None


def test_v2_contract_rejects_bad_barrier_token_ledger_and_missing_ranks(tmp_path: Path) -> None:
    state, metadata, _rank_payloads, payloads = _distributed_fixture()
    with pytest.raises(ValueError, match="collective barrier"):
        replace(metadata, collective_barrier_sha256="f" * 64).assert_valid()
    with pytest.raises(ValueError, match="token totals"):
        replace(metadata, global_tokens=metadata.global_tokens + 1).assert_valid()
    with pytest.raises(ValueError, match="incomplete rank set"):
        replace(metadata, ranks=(metadata.ranks[0],)).assert_valid()
    divergent_optimizer = replace(
        metadata,
        ranks=(metadata.ranks[0], replace(metadata.ranks[1], optimizer_state_sha256="e" * 64)),
    )
    with pytest.raises(ValueError, match="rank model or optimizer hash"):
        divergent_optimizer.assert_valid()
    with pytest.raises(ValueError, match="every rank"):
        encode_rank_state_bundle({0: {"rng_state": b"rng", "cursor_state": b"cursor"}}, world_size=2)

    store = CheckpointStore(tmp_path, state.lineage_id)
    assert store.latest_sha256() is None


def test_v1_inventory_remains_local_and_v2_requires_explicit_distributed_api(tmp_path: Path) -> None:
    state, metadata, _rank_payloads, distributed_payloads = _distributed_fixture()
    store = CheckpointStore(tmp_path, state.lineage_id)
    local_payloads = {
        name: distributed_payloads[name]
        for name in REQUIRED_COMPONENTS
        if name != "rng.bin"
    }
    local_payloads["rng.bin"] = b"legacy single-process RNG snapshot"
    local_sha = store.publish(
        state=state, payloads=local_payloads, expected_parent_sha256=None,
    )
    assert store.restore(local_sha) == (state, local_payloads)
    with pytest.raises(ValueError, match="local v1"):
        store.restore_distributed(
            rank=0, expected_world_size=2, expected_topology="xla-v2-8",
        )

    migrated_state = replace(state, parent_checkpoint_sha256=local_sha)
    migrated_metadata = replace(
        metadata,
        parent_checkpoint_sha256=local_sha,
        training_state_sha256=migrated_state.sha256(),
    )
    migrated_payloads = dict(distributed_payloads)
    migrated_payloads["training_state.json"] = _canonical(migrated_state.canonical())
    migrated_payloads["distributed.json"] = _canonical(migrated_metadata.canonical())
    distributed_sha = store.publish_distributed(
        state=migrated_state,
        payloads=migrated_payloads,
        expected_parent_sha256=local_sha,
    )
    assert store.restore(local_sha) == (state, local_payloads)
    assert store.restore_distributed(
        rank=0, expected_world_size=2, expected_topology="xla-v2-8",
    )[0] == migrated_state
    with pytest.raises(ValueError, match="distributed v2"):
        store.restore()

    downgraded_state = replace(state, parent_checkpoint_sha256=distributed_sha)
    downgraded_payloads = dict(local_payloads)
    downgraded_payloads["training_state.json"] = _canonical(downgraded_state.canonical())
    with pytest.raises(ValueError, match="inventory mismatch"):
        store.publish(state=state, payloads=distributed_payloads,
                      expected_parent_sha256=local_sha)
    with pytest.raises(ValueError, match="cannot downgrade"):
        store.publish(
            state=downgraded_state,
            payloads=downgraded_payloads,
            expected_parent_sha256=distributed_sha,
        )
