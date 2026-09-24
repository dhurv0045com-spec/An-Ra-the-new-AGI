from __future__ import annotations

import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from anra_v5.miniature_run import MINI_SPEC
from v5_model.core import initialize
from v5_data.bucket_cursor import (
    BUCKET_CURSOR_SCHEMA,
    BucketCursorState,
    build_bucket_lanes,
    cell_key,
    take_cell_window,
)
from v5_data.pack import pack_documents
from v5_training.checkpoint import CheckpointStore
from v5_training.distributed_checkpoint import (
    aggregate_rng_state_sha256,
    decode_rank_state_bundle,
    encode_rank_state_bundle,
    rank_state_payload_sha256,
)
from v5_training.distributed import RankZeroCheckpointCoordinator
from v5_training.optimizer import build_adamw_optimizer
from v5_training.production_backend import (
    ProductionTrainingBackend,
    capture_evidence,
    production_shared_payloads,
)
from v5_training.production_resume import (
    ProductionRankCapture,
    build_production_distributed_payloads,
    capture_cpu_rng_state,
    decode_production_rank_cursor,
    decode_production_rank_cursor_receipt,
    next_microstep_fingerprint_sha256,
    restore_production_rank,
    restore_production_rank_into_backend,
)
from v5_training.production_microsteps import count_rank_real_tokens
from v5_training.production_sampler import ProductionSampler
from v5_training.runner import RunController
from v5_training.schedule import lr_at
from v5_training.state import IDENTITY_SCHEMA, IdentityBindings, TrainingState
from v5_training.trainer import BackendReport, train


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode()


def _production_backend(seed: int) -> ProductionTrainingBackend:
    torch.manual_seed(seed)
    model = initialize(MINI_SPEC, seed=seed)
    optimizer = build_adamw_optimizer(model, torch_module=torch)
    return ProductionTrainingBackend(
        model=model,
        optimizer=optimizer,
        bos_id=2,
        pad_id=0,
        device=torch.device("cpu"),
        schedule=lr_at,
        torch_module=torch,
    )


def _fixture(*, document_count: int = 4):
    documents = [(f"doc-{index}", [index + 10] * 298, "reasoning")
                 for index in range(document_count)]
    packed, _audit = pack_documents(
        documents, bos=2, eos=3, pad=0, sequences_per_shard=2,
    )
    lanes, lane_receipt = build_bucket_lanes(
        packed, run_seed=31, pattern=[512], required_buckets={512},
    )
    key = cell_key(512, "", "")
    pack_sha = _sha(_canonical([json.loads(shard.payload_bytes()) for shard in packed]))
    cursor_start = BucketCursorState(
        BUCKET_CURSOR_SCHEMA, pack_sha, lane_receipt["lanes_sha256"],
        {key: [0, 0]}, {}, {}, 0, 0,
    )
    first_window = take_cell_window(
        packed, lanes[key], 0, 0, real_tokens=600, pad=0, bucket=512,
    )
    cursor_at_checkpoint = BucketCursorState(
        BUCKET_CURSOR_SCHEMA, pack_sha, lane_receipt["lanes_sha256"],
        {key: [first_window.end_lane_index, first_window.end_token_offset]},
        {}, {}, 0, 0,
    )
    sampler = ProductionSampler(
        packed=packed,
        run_seed=31,
        topology={
            "replicas": 2,
            "global_tokens_per_microstep": 600,
            "global_tokens_per_update": 600,
            "supercycle": [512],
        },
        pack_manifest_sha256=pack_sha,
        cell_of_source=None,
        mixture_fractions=None,
        cognition_fractions=None,
        cognition_mapped=False,
        allow_replay=False,
    )
    model_payload = b"production sampler test model"
    optimizer_payload = b"production sampler test Adam"
    torch.manual_seed(101)
    rng0 = capture_cpu_rng_state(torch)
    torch.manual_seed(202)
    rng1 = capture_cpu_rng_state(torch)
    rng_aggregate = aggregate_rng_state_sha256([_sha(rng0), _sha(rng1)])
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
        sampler_spec_sha256=sampler.sha256,
    )
    initial = TrainingState.initial(
        lineage_id="production-sampler-resume",
        token_budget=1200,
        tokens_per_update=600,
        cursor=cursor_start,
        rng_state_sha256="8" * 64,
        curriculum_phase="resume-test",
        identities=identities,
    )
    state = initial.advance(
        tokens_by_source={"reasoning": 600},
        cursor=cursor_at_checkpoint,
        rng_state_sha256=rng_aggregate,
        parent_checkpoint_sha256=None,
    )
    captures = []
    for rank, rng in enumerate((rng0, rng1)):
        planned = sampler.materialize_update(state, rank=rank)
        captures.append(ProductionRankCapture(
            rank=rank,
            cumulative_tokens=300,
            rng_state=rng,
            cursor=cursor_at_checkpoint,
            next_microsteps=planned.rank_microsteps,
            collective_receipt_sha256=_sha(f"collective-{rank}".encode()),
            model_state_sha256=_sha(model_payload),
            optimizer_state_sha256=_sha(optimizer_payload),
        ))
    payloads = build_production_distributed_payloads(
        state=state,
        topology="cpu-production-sampler-world2",
        model_payload=model_payload,
        optimizer_payload=optimizer_payload,
        scheduler_payload=b"{}",
        rank_captures=captures,
        sampler=sampler,
    )
    return (
        state, cursor_at_checkpoint, rng0, rng1, sampler, payloads,
        captures, model_payload, optimizer_payload,
    )


def test_legacy_state_identity_round_trips_without_optional_sampler_field() -> None:
    state, *_ = _fixture()
    legacy = replace(
        state,
        identities=replace(state.identities, sampler_spec_sha256=None),
    )
    payload = legacy.canonical()

    assert "sampler_spec_sha256" not in payload["identities"]
    assert TrainingState.from_dict(payload).sha256() == legacy.sha256()


def test_production_sampler_cursor_and_next_microstep_survive_distributed_restore(
    tmp_path: Path,
) -> None:
    state, cursor, rng0, rng1, sampler, payloads, captures, model, optimizer = _fixture()
    store = CheckpointStore(tmp_path, state.lineage_id)
    checkpoint_sha = store.publish_distributed(
        state=state, payloads=payloads, expected_parent_sha256=None,
    )
    torch.manual_seed(999)

    restored = restore_production_rank(
        store=store,
        rank=1,
        expected_world_size=2,
        expected_topology="cpu-production-sampler-world2",
        runtime="cpu",
        torch_module=torch,
        sampler=sampler,
        checkpoint_sha256=checkpoint_sha,
    )
    assert _canonical(restored.state.canonical()) == _canonical(state.canonical())
    assert _canonical(restored.cursor.canonical()) == _canonical(cursor.canonical())
    assert capture_cpu_rng_state(torch) == rng1
    assert restored.cumulative_tokens == 300
    assert restored.sampler_spec_sha256 == sampler.sha256
    assert restored.next_microstep_sha256 == restored.metadata.ranks[1].next_batch_sha256
    assert restored.shared_payloads == {
        "model.bin": model,
        "optimizer.bin": optimizer,
        "scheduler.json": b"{}",
    }
    _state, _metadata, rank_payloads = store.restore_distributed(
        rank=1,
        expected_world_size=2,
        expected_topology="cpu-production-sampler-world2",
        checkpoint_sha256=checkpoint_sha,
    )
    legacy_cursor, legacy_next_sha = decode_production_rank_cursor(
        rank_payloads["cursor_state"],
        rank=1,
        world_size=2,
        topology="cpu-production-sampler-world2",
        checkpoint_global_update=state.global_update,
    )
    receipt_cursor, receipt_sampler_sha, receipt_next_sha = decode_production_rank_cursor_receipt(
        rank_payloads["cursor_state"],
        rank=1,
        world_size=2,
        topology="cpu-production-sampler-world2",
        checkpoint_global_update=state.global_update,
    )
    assert legacy_cursor.canonical() == receipt_cursor.canonical() == cursor.canonical()
    assert receipt_sampler_sha == sampler.sha256
    assert legacy_next_sha == receipt_next_sha == restored.next_microstep_sha256

    wrong_sha = _sha(b"wrong next update")
    wrong_metadata = json.loads(payloads["distributed.json"])
    wrong_payloads = dict(payloads)
    rank_bundle = decode_rank_state_bundle(payloads["rank_states.bin"], world_size=2)
    rank_cursor = json.loads(rank_bundle[1]["cursor_state"])
    rank_cursor["next_microstep_sha256"] = wrong_sha
    rank_bundle[1]["cursor_state"] = _canonical(rank_cursor)
    wrong_metadata["ranks"][1]["cursor_state_sha256"] = _sha(
        rank_bundle[1]["cursor_state"]
    )
    wrong_metadata["ranks"][1]["rank_state_sha256"] = rank_state_payload_sha256(
        rng_state=rank_bundle[1]["rng_state"],
        cursor_state=rank_bundle[1]["cursor_state"],
    )
    wrong_metadata["ranks"][1]["next_batch_sha256"] = wrong_sha
    wrong_payloads["distributed.json"] = _canonical(wrong_metadata)
    wrong_payloads["rank_states.bin"] = encode_rank_state_bundle(
        rank_bundle, world_size=2,
    )
    wrong_store = CheckpointStore(tmp_path / "wrong-next", state.lineage_id)
    wrong_checkpoint_sha = wrong_store.publish_distributed(
        state=state, payloads=wrong_payloads, expected_parent_sha256=None,
    )

    rng_before_rejected_restore = capture_cpu_rng_state(torch)
    with pytest.raises(ValueError, match="next batch differs"):
        restore_production_rank(
            store=wrong_store,
            rank=1,
            expected_world_size=2,
            expected_topology="cpu-production-sampler-world2",
            runtime="cpu",
            torch_module=torch,
            sampler=sampler,
            checkpoint_sha256=wrong_checkpoint_sha,
        )
    assert capture_cpu_rng_state(torch) == rng_before_rejected_restore
    with pytest.raises(RuntimeError, match="explicit RNG adapter"):
        restore_production_rank(
            store=store,
            rank=0,
            expected_world_size=2,
            expected_topology="cpu-production-sampler-world2",
            runtime="xla",
            torch_module=torch,
            sampler=sampler,
            checkpoint_sha256=checkpoint_sha,
        )

    class RecordingRngAdapter:
        payload: bytes | None = None

        def restore_rng_state(self, payload: bytes) -> None:
            self.payload = payload

    adapter = RecordingRngAdapter()
    adapted = restore_production_rank(
        store=store,
        rank=1,
        expected_world_size=2,
        expected_topology="cpu-production-sampler-world2",
        runtime="xla",
        torch_module=torch,
        sampler=sampler,
        checkpoint_sha256=checkpoint_sha,
        rng_state_adapter=adapter,
    )
    assert adapter.payload == rng1
    assert adapted.rng_state == rng1


def test_v2_backend_loader_applies_shared_state_then_rank_rng_last(tmp_path: Path) -> None:
    (
        state, _cursor, rng0, rng1, sampler, _old_payloads, captures,
        _old_model, _old_optimizer,
    ) = _fixture()
    source_backend = _production_backend(301)
    source_shared = production_shared_payloads(source_backend)
    model_payload = source_shared["model.bin"]
    optimizer_payload = source_shared["optimizer.bin"]
    captures = [
        replace(
            capture,
            model_state_sha256=_sha(model_payload),
            optimizer_state_sha256=_sha(optimizer_payload),
        )
        for capture in captures
    ]
    payloads = build_production_distributed_payloads(
        state=state,
        topology="cpu-production-sampler-world2",
        model_payload=model_payload,
        optimizer_payload=optimizer_payload,
        scheduler_payload=source_shared["scheduler.json"],
        rank_captures=captures,
        sampler=sampler,
    )
    store = CheckpointStore(tmp_path / "rank-loader", state.lineage_id)
    checkpoint_sha = store.publish_distributed(
        state=state, payloads=payloads, expected_parent_sha256=None,
    )

    target_backend = _production_backend(302)
    torch.manual_seed(303)
    restored = restore_production_rank_into_backend(
        backend=target_backend,
        store=store,
        rank=1,
        expected_world_size=2,
        expected_topology="cpu-production-sampler-world2",
        sampler=sampler,
        checkpoint_sha256=checkpoint_sha,
    )

    source_evidence = capture_evidence(
        source_backend.model, source_backend.optimizer, torch=torch,
    )
    target_evidence = capture_evidence(
        target_backend.model, target_backend.optimizer, torch=torch,
    )
    assert target_evidence.parameter_sha256 == source_evidence.parameter_sha256
    assert target_evidence.moment_sha256 == source_evidence.moment_sha256
    assert target_evidence.optimizer_steps == source_evidence.optimizer_steps
    assert capture_cpu_rng_state(torch) == rng1
    assert restored.rng_state == rng1
    assert restored.rng_state != rng0


class _ThreadedMeshReduce:
    """Barrier-backed all-rank collective for trainer-to-store integration."""

    def __init__(self, world_size: int, *, timeout: float = 15.0) -> None:
        self.world_size = world_size
        self.timeout = timeout
        self.condition = threading.Condition()
        self.slots: dict[str, dict[str, object]] = {}
        self.calls: list[tuple[str, int]] = []

    def for_rank(self, rank: int):
        def mesh_reduce(tag: str, value: object, reduce_fn):
            with self.condition:
                self.calls.append((tag, rank))
                slot = self.slots.setdefault(tag, {"values": {}, "readers": 0})
                values = slot["values"]
                assert isinstance(values, dict)
                if rank in values:
                    raise AssertionError(f"rank {rank} entered {tag} more than once")
                values[rank] = value
                if len(values) == self.world_size:
                    try:
                        slot["result"] = reduce_fn(
                            [values[index] for index in range(self.world_size)]
                        )
                    except BaseException as exc:
                        slot["error"] = exc
                    self.condition.notify_all()
                ready = self.condition.wait_for(
                    lambda: "result" in slot or "error" in slot,
                    timeout=self.timeout,
                )
                if not ready:
                    raise TimeoutError(f"mesh collective {tag} did not receive every rank")
                if "error" in slot:
                    error = slot["error"]
                    assert isinstance(error, BaseException)
                    raise RuntimeError(f"mesh reducer failed for {tag}: {error}") from error
                result = slot["result"]
                slot["readers"] = int(slot["readers"]) + 1
                if slot["readers"] == self.world_size:
                    del self.slots[tag]
                return result

        return mesh_reduce


def test_trainer_gathers_rank_captures_and_publishes_v2_then_restores_each_rank(
    tmp_path: Path,
) -> None:
    state, _cursor, _rng0, _rng1, sampler, _old_payloads, _captures, _model, _optimizer = (
        _fixture(document_count=8)
    )
    state = replace(state, token_budget=1800)
    update_plans = [sampler.materialize_update(state, rank=rank) for rank in range(2)]
    local_tokens = [count_rank_real_tokens(plan.rank_microsteps) for plan in update_plans]
    assert sum(local_tokens) == 600
    assert update_plans[0].end_cursor == update_plans[1].end_cursor

    torch.manual_seed(707)
    rng_states = [capture_cpu_rng_state(torch)]
    torch.manual_seed(808)
    rng_states.append(capture_cpu_rng_state(torch))
    rng_hashes = [_sha(payload) for payload in rng_states]
    shared_backend = _production_backend(909)
    shared_payloads = production_shared_payloads(shared_backend)
    topology = "cpu-production-sampler-world2"

    class _CountingStore(CheckpointStore):
        def __init__(self, root: Path, lineage_id: str) -> None:
            super().__init__(root, lineage_id)
            self.distributed_writes = 0

        def publish(self, **_kwargs):
            raise AssertionError("v2 integration must not publish a v1 generation")

        def publish_distributed(self, **kwargs):
            self.distributed_writes += 1
            return super().publish_distributed(**kwargs)

    store = _CountingStore(tmp_path / "trainer-v2", state.lineage_id)
    mesh = _ThreadedMeshReduce(world_size=2)
    coordinators = [
        RankZeroCheckpointCoordinator(
            rank=rank,
            world_size=2,
            mesh_reduce=mesh.for_rank(rank),
            initial_rank_cumulative_tokens=300,
        )
        for rank in range(2)
    ]
    observed: list[tuple[int, int, str]] = []
    payload_builds: list[int] = []

    def run_rank(rank: int) -> TrainingState:
        plan = update_plans[rank]
        controller = RunController(target_update=state.global_update + 1)
        controller.start()

        def capture(live: TrainingState) -> ProductionRankCapture:
            next_plan = sampler.materialize_update(live, rank=rank)
            fingerprint = next_microstep_fingerprint_sha256(
                rank=rank,
                world_size=2,
                topology=topology,
                checkpoint_global_update=live.global_update,
                cursor=live.cursor,
                microsteps=next_plan.rank_microsteps,
            )
            return ProductionRankCapture(
                rank=rank,
                cumulative_tokens=coordinators[rank].rank_cumulative_tokens,
                rng_state=rng_states[rank],
                cursor=live.cursor,
                next_microsteps=(),
                collective_receipt_sha256=_sha(f"update-collective-{rank}".encode()),
                model_state_sha256=_sha(shared_payloads["model.bin"]),
                optimizer_state_sha256=_sha(shared_payloads["optimizer.bin"]),
                next_microstep_sha256=fingerprint,
            )

        def build_v2(live: TrainingState, captures) -> dict[str, bytes]:
            payload_builds.append(rank)
            return build_production_distributed_payloads(
                state=live,
                topology=topology,
                model_payload=shared_payloads["model.bin"],
                optimizer_payload=shared_payloads["optimizer.bin"],
                scheduler_payload=shared_payloads["scheduler.json"],
                rank_captures=captures,
                sampler=sampler,
            )

        return train(
            state=state,
            controller=controller,
            store=store if rank == 0 else object(),
            payload_builder=lambda _live: (_ for _ in ()).throw(
                AssertionError("v1 payload builder must not be called")
            ),
            backend_step=lambda _live: BackendReport(
                tokens_by_source={"reasoning": 600},
                cursor=plan.end_cursor,
                rng_state_sha256=rng_hashes[rank],
                loss_finite=True,
                grad_finite=True,
                grad_norm_post_clip=0.0,
                tied_preserved=True,
                local_real_tokens=local_tokens[rank],
            ),
            updates=1,
            checkpoint_every=1,
            checkpoint_coordinator=coordinators[rank],
            rank_capture_builder=capture,
            distributed_payload_builder=build_v2,
            on_checkpoint_observed=lambda live, sha: observed.append(
                (rank, live.global_update, sha)
            ),
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        final_states = list(executor.map(run_rank, range(2)))
    assert final_states[0] == final_states[1]
    final_state, metadata, _rank_payloads, shared_restored = (
        store.restore_distributed_artifacts(
            rank=0,
            expected_world_size=2,
            expected_topology=topology,
        )
    )
    assert _canonical(final_state.canonical()) == _canonical(final_states[0].canonical())
    assert metadata.global_tokens == final_state.cumulative_tokens
    assert store.distributed_writes == 1
    assert payload_builds == [0]
    assert {sha for _, _, sha in observed} == {store.latest_sha256()}
    assert sorted(rank for rank, update, _ in observed if update == final_state.global_update) == [0, 1]
    rank_capture_calls = [(tag, rank) for tag, rank in mesh.calls
                          if tag.endswith("-rank-captures")]
    assert len(rank_capture_calls) == 2
    assert len({tag for tag, _rank in rank_capture_calls}) == 1
    assert {rank for _tag, rank in rank_capture_calls} == {0, 1}

    for rank in range(2):
        restored_backend = _production_backend(1000 + rank)
        resumed = restore_production_rank_into_backend(
            backend=restored_backend,
            store=store,
            rank=rank,
            expected_world_size=2,
            expected_topology=topology,
            sampler=sampler,
        )
        assert resumed.state == final_state
        assert resumed.cumulative_tokens == metadata.ranks[rank].cumulative_tokens
        assert resumed.rng_state == rng_states[rank]
        assert capture_cpu_rng_state(torch) == rng_states[rank]
    assert sum(rank.cumulative_tokens for rank in metadata.ranks) == final_state.cumulative_tokens
    assert set(shared_restored) == {"model.bin", "optimizer.bin", "scheduler.json"}


def test_distributed_checkpoint_rejects_rank_cursor_drift(tmp_path: Path) -> None:
    state, cursor, _rng0, _rng1, sampler, _payloads, captures, model, optimizer = _fixture()
    cell = next(iter(cursor.positions))
    lane_index, token_offset = cursor.positions[cell]
    stale_cursors = (
        replace(cursor, positions={**cursor.positions, cell: (lane_index, token_offset + 1)}),
        replace(cursor, epoch=cursor.epoch + 1),
        replace(cursor, mixture_consumed={"reasoning": 1}),
    )
    for stale_cursor in stale_cursors:
        stale_capture = replace(captures[0], cursor=stale_cursor)
        with pytest.raises(ValueError, match="progress differs from shared training cursor"):
            build_production_distributed_payloads(
                state=state,
                topology="cpu-production-sampler-world2",
                model_payload=model,
                optimizer_payload=optimizer,
                scheduler_payload=b"{}",
                rank_captures=(stale_capture, captures[1]),
                sampler=sampler,
            )


def test_restore_rejects_rank_shard_identity_tampering(tmp_path: Path) -> None:
    state, _cursor, _rng0, _rng1, sampler, payloads, *_ = _fixture()
    document = json.loads(payloads["distributed.json"])
    document["ranks"][1]["data_shard_identity"] = _sha(b"wrong-rank-shard")
    tampered_payloads = dict(payloads)
    tampered_payloads["distributed.json"] = _canonical(document)
    store = CheckpointStore(tmp_path, state.lineage_id)
    checkpoint_sha = store.publish_distributed(
        state=state, payloads=tampered_payloads, expected_parent_sha256=None,
    )

    with pytest.raises(ValueError, match="data-shard identity disagrees"):
        restore_production_rank(
            store=store,
            rank=1,
            expected_world_size=2,
            expected_topology="cpu-production-sampler-world2",
            runtime="cpu",
            torch_module=torch,
            sampler=sampler,
            checkpoint_sha256=checkpoint_sha,
        )


def test_restore_rejects_future_sampler_policy_drift(tmp_path: Path) -> None:
    state, _cursor, _rng0, _rng1, sampler, payloads, captures, model, optimizer = _fixture()
    changed_sampler = replace(sampler, allow_replay=True)
    with pytest.raises(ValueError, match="identity differs from training state"):
        build_production_distributed_payloads(
            state=state,
            topology="cpu-production-sampler-world2",
            model_payload=model,
            optimizer_payload=optimizer,
            scheduler_payload=b"{}",
            rank_captures=captures,
            sampler=changed_sampler,
        )
    store = CheckpointStore(tmp_path, state.lineage_id)
    checkpoint_sha = store.publish_distributed(
        state=state, payloads=payloads, expected_parent_sha256=None,
    )

    with pytest.raises(ValueError, match="sampler identity differs from training state"):
        restore_production_rank(
            store=store,
            rank=1,
            expected_world_size=2,
            expected_topology="cpu-production-sampler-world2",
            runtime="cpu",
            torch_module=torch,
            sampler=changed_sampler,
            checkpoint_sha256=checkpoint_sha,
        )
