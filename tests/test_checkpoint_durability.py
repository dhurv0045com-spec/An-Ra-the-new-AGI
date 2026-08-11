from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest
import torch

from scripts.build_brain import (
    _assert_token_window_start,
    _cap_batch_to_token_budget,
    _prepare_resume_target,
    _resolve_token_window_contract,
)
from training.checkpoint_durability import (
    ArtifactClass,
    CheckpointDurabilitySession,
    CheckpointOutbox,
    DurabilityCorruptionError,
    DurabilityState,
    FilesystemReplica,
    MonolithicFilesystemReplica,
    PublicationError,
    ResumeArtifactError,
    SnapshotPublisher,
    assert_resume_artifact_class,
    create_fp16_inference_artifact,
    plan_hot_retention,
)
from training.v2_runtime import CheckpointCompatibilityError, load_checkpoint


def _lineage(step: int) -> dict[str, object]:
    return {
        "lineage_id": "test-foundation/anra-v4-180m",
        "checkpoint_schema_version": 9,
        "source_commit": "abc123",
        "architecture": {"sha256": "architecture", "config": {"n_layer": 2}},
        "tokenizer": {"sha256": "tokenizer", "schema_version": 4},
        "data": {"contract_sha256": "data", "manifest_hashes": {"a": "b"}},
        "training": {"recipe_sha256": "recipe", "seed_contract": {"seed": 1301}},
        "progress": {"global_step": step, "tokens_seen": step * 1024},
        "continuity": {
            "completed_optimizer_boundary": True,
            "accum_micro_steps": 0,
            "components": {
                "model": True,
                "optimizer": True,
                "scheduler": True,
                "scaler": True,
                "rng_states": True,
            },
        },
    }


def _checkpoint(path: Path, size: int = 4096) -> bytes:
    payload = bytes(index % 251 for index in range(size))
    path.write_bytes(payload)
    return payload


def test_full_resume_snapshot_round_trips_and_publishes_verified_pointer(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    expected = _checkpoint(checkpoint, 12_000)
    outbox = CheckpointOutbox(tmp_path / "outbox", chunk_size_bytes=1024)
    ref = outbox.register_checkpoint(checkpoint, lineage=_lineage(100))
    primary = FilesystemReplica(
        "drive",
        tmp_path / "drive",
        kind="mounted_drive",
        canonical=True,
    )
    laptop = FilesystemReplica("laptop", tmp_path / "laptop")
    publisher = SnapshotPublisher(
        outbox,
        [primary, laptop],
        min_protected_replicas=2,
        max_copy_streams=2,
    )
    try:
        publisher.submit(ref)
        result = publisher.wait_for(ref, DurabilityState.PROTECTED, timeout_seconds=10)
    finally:
        publisher.close(wait=True)

    restored = outbox.materialize(ref.snapshot_id, tmp_path / "restored.pt", for_resume=True)
    pointer = json.loads((primary.root / "canonical.json").read_text(encoding="utf-8"))
    manifest = json.loads(ref.manifest_path.read_text(encoding="utf-8"))

    assert restored.read_bytes() == expected
    assert result.state is DurabilityState.PROTECTED
    assert set(result.verified_replicas) == {"drive", "laptop"}
    assert pointer["snapshot_id"] == ref.snapshot_id
    assert pointer["checkpoint_sha256"] == manifest["source"]["sha256"]
    assert manifest["artifact_class"] == "full_resume"
    assert manifest["lineage"]["progress"]["global_step"] == 100


def test_mounted_drive_single_file_replaces_only_after_verification(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    first_payload = _checkpoint(checkpoint, 12_000)
    outbox = CheckpointOutbox(tmp_path / "outbox", chunk_size_bytes=1024)
    replica = MonolithicFilesystemReplica(
        "drive-vault",
        tmp_path / "drive" / "checkpoint-vault",
        canonical=True,
    )
    (replica.root / "chunks" / "aa").mkdir(parents=True)
    (replica.root / "chunks" / "aa" / "legacy.chunk").write_bytes(b"legacy")
    (replica.root / "manifests").mkdir()
    (replica.root / "manifests" / "legacy.json").write_text("{}")
    (replica.root / "receipts").mkdir()
    (replica.root / "canonical.json").write_text("{}")
    (replica.root / "canonical-full-resume.json").write_text("{}")
    publisher = SnapshotPublisher(outbox, [replica])
    first = outbox.register_checkpoint(checkpoint, lineage=_lineage(100))
    try:
        publisher.submit(first)
        publisher.wait_for(first, DurabilityState.PROTECTED, timeout_seconds=10)

        first_files = list(replica.root.glob("*.pt"))
        assert [path.name for path in first_files] == [
            "anra-v4-current-full-resume.pt"
        ]
        assert first_files[0].read_bytes() == first_payload
        assert not list(replica.root.rglob("*.chunk"))
        assert not (replica.root / "manifests").exists()
        assert not (replica.root / "receipts").exists()
        assert not (replica.root / "canonical.json").exists()
        assert not (replica.root / "canonical-full-resume.json").exists()

        second_payload = bytes((index * 7) % 251 for index in range(14_000))
        checkpoint.write_bytes(second_payload)
        second = outbox.register_checkpoint(checkpoint, lineage=_lineage(200))
        publisher.submit(second)
        publisher.wait_for(second, DurabilityState.PROTECTED, timeout_seconds=10)
    finally:
        publisher.close(wait=True)

    final_files = list(replica.root.glob("*.pt"))
    assert [path.name for path in final_files] == [
        "anra-v4-current-full-resume.pt"
    ]
    assert final_files[0].read_bytes() == second_payload
    metadata = json.loads(
        (replica.root / "anra-v4-current-full-resume.json").read_text(encoding="utf-8")
    )
    assert metadata["global_step"] == 200
    assert not list(replica.root.glob("*.uploading"))


def test_mounted_drive_rejects_second_live_canonical_writer(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    _checkpoint(checkpoint, 4_096)
    outbox = CheckpointOutbox(tmp_path / "outbox", chunk_size_bytes=1024)
    ref = outbox.register_checkpoint(checkpoint, lineage=_lineage(100))
    replica = MonolithicFilesystemReplica("drive-vault", tmp_path / "drive")
    replica.root.mkdir(parents=True)
    lease = replica._writer_lease_path(ArtifactClass.FULL_RESUME)
    lease.write_text('{"token": "another-writer"}\n', encoding="utf-8")

    with pytest.raises(PublicationError, match="Another canonical writer"):
        replica.publish_manifest(ref, ref.manifest_path.read_bytes())

    assert not list(replica.root.glob("*.pt"))


def test_mounted_drive_reclaims_expired_writer_lease(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    expected = _checkpoint(checkpoint, 4_096)
    outbox = CheckpointOutbox(tmp_path / "outbox", chunk_size_bytes=1024)
    ref = outbox.register_checkpoint(checkpoint, lineage=_lineage(100))
    replica = MonolithicFilesystemReplica("drive-vault", tmp_path / "drive")
    replica.root.mkdir(parents=True)
    lease = replica._writer_lease_path(ArtifactClass.FULL_RESUME)
    lease.write_text('{"token": "abandoned"}\n', encoding="utf-8")
    expired = time.time() - 4 * 60 * 60
    os.utime(lease, (expired, expired))

    replica.publish_manifest(ref, ref.manifest_path.read_bytes())

    target = replica.root / "anra-v4-current-full-resume.pt"
    assert target.read_bytes() == expected
    assert not lease.exists()


def test_mounted_drive_fences_another_training_session(tmp_path: Path) -> None:
    root = tmp_path / "drive"
    first = MonolithicFilesystemReplica("drive-vault", root, canonical=True)
    second = MonolithicFilesystemReplica("drive-vault", root, canonical=True)

    first.acquire_writer_session()
    try:
        with pytest.raises(PublicationError, match="Another canonical training session"):
            second.acquire_writer_session()
    finally:
        first.release_writer_session()

    second.acquire_writer_session()
    second.release_writer_session()
    assert not list(root.glob("*.session-lease.json"))


def test_active_training_session_heartbeat_prevents_expiry(tmp_path: Path) -> None:
    root = tmp_path / "drive"
    first = MonolithicFilesystemReplica("drive-vault", root, canonical=True)
    second = MonolithicFilesystemReplica("drive-vault", root, canonical=True)

    first.acquire_writer_session()
    try:
        lease = first._session_lease_path(ArtifactClass.FULL_RESUME)
        expired = time.time() - 2 * 60 * 60
        os.utime(lease, (expired, expired))
        first.refresh_writer_session()
        with pytest.raises(PublicationError, match="Another canonical training session"):
            second.acquire_writer_session()
    finally:
        first.release_writer_session()


def test_nearly_expired_training_session_waits_then_reclaims(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "drive"
    first = MonolithicFilesystemReplica("drive-vault", root, canonical=True)
    second = MonolithicFilesystemReplica("drive-vault", root, canonical=True)
    first.acquire_writer_session()
    lease = first._session_lease_path(ArtifactClass.FULL_RESUME)
    lease_seconds = 15 * 60
    almost_expired = time.time() - lease_seconds + 1.0
    os.utime(lease, (almost_expired, almost_expired))
    waits: list[float] = []

    def expire_during_wait(seconds: float) -> None:
        waits.append(seconds)
        expired = time.time() - lease_seconds - 1.0
        os.utime(lease, (expired, expired))

    monkeypatch.setattr("training.checkpoint_durability.time.sleep", expire_during_wait)
    second.acquire_writer_session()
    try:
        assert waits and waits[0] <= 2.0
    finally:
        second.release_writer_session()
        first.release_writer_session()


def test_mounted_drive_reconciles_stale_future_pointer_from_actual_payload(
    tmp_path: Path,
) -> None:
    """A JSON-only future step must not trap every resume at an older payload."""
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"global_step": 7600}, checkpoint)
    outbox = CheckpointOutbox(tmp_path / "outbox", chunk_size_bytes=1024)
    replica = MonolithicFilesystemReplica(
        "drive-vault",
        tmp_path / "drive",
        canonical=True,
    )
    first = outbox.register_checkpoint(checkpoint, lineage=_lineage(7600))
    publisher = SnapshotPublisher(outbox, [replica])
    try:
        publisher.submit(first)
        publisher.wait_for(first, DurabilityState.PROTECTED, timeout_seconds=10)
    finally:
        publisher.close(wait=True)

    pointer_path = replica.root / "anra-v4-current-full-resume.json"
    stale_pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    stale_pointer["global_step"] = 7845
    pointer_path.write_text(json.dumps(stale_pointer), encoding="utf-8")

    torch.save({"global_step": 7799}, checkpoint)
    replacement = outbox.register_checkpoint(checkpoint, lineage=_lineage(7799))
    replica.publish_manifest(replacement, replacement.manifest_path.read_bytes())

    recovered_pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    assert recovered_pointer["global_step"] == 7799


def test_failed_single_file_replacement_preserves_previous_checkpoint(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    first_payload = _checkpoint(checkpoint, 12_000)
    outbox = CheckpointOutbox(tmp_path / "outbox", chunk_size_bytes=1024)
    replica = MonolithicFilesystemReplica(
        "drive-vault",
        tmp_path / "drive" / "checkpoint-vault",
        canonical=True,
    )
    publisher = SnapshotPublisher(outbox, [replica])
    first = outbox.register_checkpoint(checkpoint, lineage=_lineage(100))
    try:
        publisher.submit(first)
        publisher.wait_for(first, DurabilityState.PROTECTED, timeout_seconds=10)

        checkpoint.write_bytes(b"replacement" * 2000)
        second = outbox.register_checkpoint(checkpoint, lineage=_lineage(200))
        second_manifest = outbox.load_manifest(second.snapshot_id)
        corrupt_record = second_manifest["chunks"][0]
        outbox.chunk_path(corrupt_record["sha256"]).write_bytes(
            b"x" * corrupt_record["size_bytes"]
        )
        publisher.submit(second)
        with pytest.raises(PublicationError):
            publisher.wait_for(
                second,
                DurabilityState.PROTECTED,
                timeout_seconds=10,
            )
    finally:
        publisher.close(wait=False)

    remaining = list(replica.root.glob("*.pt"))
    assert [path.name for path in remaining] == [
        "anra-v4-current-full-resume.pt"
    ]
    assert remaining[0].read_bytes() == first_payload
    assert not list(replica.root.glob("*.uploading"))


def test_incomplete_future_pointer_cannot_block_recovery_checkpoint(
    tmp_path: Path,
) -> None:
    replica = FilesystemReplica("drive-vault", tmp_path / "drive", canonical=True)
    replica.root.mkdir(parents=True)
    invalid = {
        "schema_version": 1,
        "snapshot_id": "step-000000000253-incomplete",
        "manifest_sha256": "missing",
        "artifact_class": "full_resume",
        "checkpoint_sha256": "missing",
        "global_step": 253,
    }
    (replica.root / "canonical-full-resume.json").write_text(
        json.dumps(invalid),
        encoding="utf-8",
    )
    (replica.root / "canonical.json").write_text(
        json.dumps(invalid),
        encoding="utf-8",
    )
    recovered = {
        **invalid,
        "snapshot_id": "step-000000000200-recovered",
        "manifest_sha256": "recovered-manifest",
        "checkpoint_sha256": "recovered-checkpoint",
        "global_step": 200,
    }

    replica.publish_pointer(recovered)

    pointer = json.loads(
        (replica.root / "canonical.json").read_text(encoding="utf-8")
    )
    assert pointer["global_step"] == 200
    audit = replica.root / "recovery" / (
        "invalid-canonical-step-000000000253-incomplete.json"
    )
    assert json.loads(audit.read_text(encoding="utf-8"))["reason"].startswith(
        "missing manifest"
    )


def test_complete_future_pointer_still_blocks_rewind(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    _checkpoint(checkpoint, 4096)
    outbox = CheckpointOutbox(tmp_path / "outbox", chunk_size_bytes=1024)
    ref = outbox.register_checkpoint(checkpoint, lineage=_lineage(253))
    replica = FilesystemReplica("drive-vault", tmp_path / "drive", canonical=True)
    publisher = SnapshotPublisher(outbox, [replica])
    try:
        publisher.submit(ref)
        publisher.wait_for(ref, DurabilityState.PROTECTED, timeout_seconds=10)
    finally:
        publisher.close(wait=True)
    current = json.loads(
        (replica.root / "canonical.json").read_text(encoding="utf-8")
    )
    rewind = {
        **current,
        "snapshot_id": "step-000000000200-rewind",
        "manifest_sha256": "rewind-manifest",
        "checkpoint_sha256": "rewind-checkpoint",
        "global_step": 200,
    }

    with pytest.raises(PublicationError, match="Refusing to rewind"):
        replica.publish_pointer(rewind)


def test_partial_replica_chunk_resumes_without_recopying_prefix(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    _checkpoint(checkpoint, 8192)
    outbox = CheckpointOutbox(tmp_path / "outbox", chunk_size_bytes=8192)
    ref = outbox.register_checkpoint(checkpoint, lineage=_lineage(25))
    manifest = outbox.load_manifest(ref.snapshot_id)
    chunk = manifest["chunks"][0]
    local_chunk = outbox.chunk_path(chunk["sha256"])
    replica = FilesystemReplica("drive", tmp_path / "drive", canonical=True)
    partial = replica.partial_chunk_path(chunk["sha256"])
    partial.parent.mkdir(parents=True)
    partial.write_bytes(local_chunk.read_bytes()[:4096])

    publisher = SnapshotPublisher(outbox, [replica])
    try:
        publisher.submit(ref)
        result = publisher.wait_for(ref, timeout_seconds=10)
    finally:
        publisher.close(wait=True)

    assert result.state is DurabilityState.PROTECTED
    assert replica.chunk_path(chunk["sha256"]).read_bytes() == local_chunk.read_bytes()
    assert not partial.exists()
    assert (replica.root / "canonical.json").is_file()


def test_corrupt_replica_chunk_blocks_manifest_and_canonical_pointer(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    _checkpoint(checkpoint, 4096)
    outbox = CheckpointOutbox(tmp_path / "outbox", chunk_size_bytes=4096)
    ref = outbox.register_checkpoint(checkpoint, lineage=_lineage(10))
    manifest = outbox.load_manifest(ref.snapshot_id)
    chunk = manifest["chunks"][0]
    replica = FilesystemReplica("drive", tmp_path / "drive", canonical=True)
    corrupt = replica.chunk_path(chunk["sha256"])
    corrupt.parent.mkdir(parents=True)
    corrupt.write_bytes(b"x" * chunk["size_bytes"])

    publisher = SnapshotPublisher(outbox, [replica])
    try:
        publisher.submit(ref)
        with pytest.raises(PublicationError, match="wrong digest"):
            publisher.wait_for(ref, timeout_seconds=10)
    finally:
        publisher.close(wait=True)

    assert not (replica.root / "canonical.json").exists()
    assert not (replica.root / "manifests" / f"{ref.snapshot_id}.json").exists()


def test_failed_publication_removes_unmanifested_remote_chunks(tmp_path: Path) -> None:
    class ManifestFailureReplica(FilesystemReplica):
        def publish_manifest(self, ref, manifest_bytes):  # type: ignore[no-untyped-def]
            raise OSError("simulated Drive quota failure")

    checkpoint = tmp_path / "checkpoint.pt"
    _checkpoint(checkpoint, 8192)
    outbox = CheckpointOutbox(tmp_path / "outbox", chunk_size_bytes=1024)
    ref = outbox.register_checkpoint(checkpoint, lineage=_lineage(10))
    replica = ManifestFailureReplica("drive", tmp_path / "drive", canonical=True)
    publisher = SnapshotPublisher(outbox, [replica])
    try:
        publisher.submit(ref)
        with pytest.raises(PublicationError, match="quota failure"):
            publisher.wait_for(ref, timeout_seconds=10)
    finally:
        publisher.close(wait=True)

    assert not list((replica.root / "chunks").rglob("*.chunk"))
    assert not list((replica.root / "manifests").glob("*.json"))

    retry_checkpoint = tmp_path / "retry.pt"
    _checkpoint(retry_checkpoint, 8193)
    retry_ref = outbox.register_checkpoint(retry_checkpoint, lineage=_lineage(20))
    retry = SnapshotPublisher(
        outbox,
        [FilesystemReplica("drive", replica.root, canonical=True)],
    )
    try:
        retry.submit(retry_ref)
        retry.wait_for(retry_ref, DurabilityState.PROTECTED, timeout_seconds=10)
    finally:
        retry.close(wait=True)

    assert [item.snapshot_id for item in outbox.snapshots()] == [retry_ref.snapshot_id]
    assert len(list((replica.root / "manifests").glob("*.json"))) == 1


def test_compact_artifact_is_rejected_for_materialize_and_training_resume(
    tmp_path: Path,
) -> None:
    compact = tmp_path / "compact.pt"
    torch.save(
        {
            "checkpoint_schema_version": 9,
            "checkpoint_artifact_class": "fp16_inference",
            "model": {"weight": torch.ones(2, dtype=torch.float16)},
        },
        compact,
    )
    outbox = CheckpointOutbox(tmp_path / "outbox", chunk_size_bytes=1024)
    ref = outbox.register_checkpoint(
        compact,
        artifact_class=ArtifactClass.FP16_INFERENCE,
        lineage=_lineage(50),
    )

    with pytest.raises(ResumeArtifactError, match="cannot resume"):
        outbox.materialize(ref.snapshot_id, tmp_path / "resume.pt", for_resume=True)
    with pytest.raises(ResumeArtifactError, match="exact training resume"):
        assert_resume_artifact_class(torch.load(compact, weights_only=False), compact)
    with pytest.raises(CheckpointCompatibilityError, match="full_resume"):
        load_checkpoint(
            object(),  # rejection occurs before the model is inspected
            None,
            None,
            None,
            compact,
            device=torch.device("cpu"),
            resume_training=True,
        )


def test_compact_export_contains_only_fp16_model_and_bound_lineage(
    tmp_path: Path,
) -> None:
    source = tmp_path / "full-resume.pt"
    torch.save(
        {
            "checkpoint_schema_version": 9,
            "checkpoint_artifact_class": "full_resume",
            "checkpoint_lineage": _lineage(75),
            "source_commit": "abc123",
            "global_step": 75,
            "tokens_seen": 76_800,
            "model": {
                "weight": torch.ones(4, dtype=torch.float32),
                "position": torch.arange(4, dtype=torch.int64),
            },
            "optimizer": {"state": {"sensitive": torch.ones(1)}},
            "scheduler": {"last_epoch": 75},
            "scaler": {},
            "rng_states": {"torch": torch.get_rng_state()},
        },
        source,
    )
    compact_path = tmp_path / "inference.pt"

    report = create_fp16_inference_artifact(source, compact_path)
    compact = torch.load(compact_path, map_location="cpu", weights_only=True)

    assert report["path"] == str(compact_path)
    assert compact["checkpoint_artifact_class"] == "fp16_inference"
    assert compact["training_resume_allowed"] is False
    assert compact["model"]["weight"].dtype is torch.float16
    assert compact["model"]["position"].dtype is torch.int64
    assert "optimizer" not in compact
    assert "scheduler" not in compact
    assert "rng_states" not in compact
    assert compact["checkpoint_lineage"]["lineage_id"] == _lineage(75)["lineage_id"]
    with pytest.raises(ResumeArtifactError, match="exact training resume"):
        assert_resume_artifact_class(compact, compact_path)


def test_retention_plan_keeps_two_full_two_compact_and_inflight_budget(
    tmp_path: Path,
) -> None:
    outbox = CheckpointOutbox(tmp_path / "outbox", chunk_size_bytes=128)
    for step in range(1, 4):
        full = tmp_path / f"full-{step}.pt"
        _checkpoint(full, 400 + step)
        outbox.register_checkpoint(full, lineage=_lineage(step))
        compact = tmp_path / f"compact-{step}.pt"
        _checkpoint(compact, 100 + step)
        outbox.register_checkpoint(
            compact,
            artifact_class=ArtifactClass.FP16_INFERENCE,
            lineage=_lineage(step),
        )

    plan = plan_hot_retention(outbox, in_flight_bytes=500, hot_limit_bytes=2000)

    assert len(plan.keep_snapshot_ids) == 4
    assert len(plan.delete_snapshot_ids) == 2
    assert plan.retained_logical_bytes + plan.in_flight_bytes <= 2000
    assert plan.fits is True


def test_local_outbox_detects_content_addressed_corruption(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    _checkpoint(checkpoint, 1024)
    outbox = CheckpointOutbox(tmp_path / "outbox", chunk_size_bytes=1024)
    ref = outbox.register_checkpoint(checkpoint, lineage=_lineage(1))
    manifest = outbox.load_manifest(ref.snapshot_id)
    chunk = manifest["chunks"][0]
    outbox.chunk_path(chunk["sha256"]).write_bytes(b"z" * chunk["size_bytes"])

    with pytest.raises(DurabilityCorruptionError, match="content-addressed chunk is corrupt"):
        outbox.register_checkpoint(checkpoint, lineage=_lineage(1))


def test_signed_token_window_is_resolved_and_bound_to_checkpoint_boundary(
    monkeypatch,
) -> None:
    window_id = "a" * 64
    monkeypatch.setenv("ANRA_TOKEN_WINDOW_ID", window_id)
    monkeypatch.setenv("ANRA_TOKEN_WINDOW_START", "1000")
    monkeypatch.setenv("ANRA_TOKEN_WINDOW_END", "2000")

    contract = _resolve_token_window_contract(None, None, None)

    assert contract == {
        "window_id": window_id,
        "start_token": 1000,
        "end_token": 2000,
    }
    _assert_token_window_start(contract, phase_tokens_seen=1000, scratch_run=False)
    with pytest.raises(RuntimeError, match="boundary mismatch"):
        _assert_token_window_start(contract, phase_tokens_seen=999, scratch_run=False)
    with pytest.raises(RuntimeError, match="scratch launch"):
        _assert_token_window_start(contract, phase_tokens_seen=0, scratch_run=True)


def test_final_microbatch_is_capped_exactly_at_signed_token_end() -> None:
    xb = torch.arange(16).reshape(2, 8)
    yb = xb.clone()
    wb = torch.ones_like(yb, dtype=torch.float32)
    answer_mask = torch.ones_like(yb, dtype=torch.bool)
    sample_idx = torch.tensor([3, 4])

    capped = _cap_batch_to_token_budget(
        xb,
        yb,
        wb,
        sample_idx,
        answer_mask,
        remaining_tokens=10,
        pad_id=-1,
    )
    capped_x, capped_y, capped_w, capped_idx, capped_answer, accepted = capped

    assert accepted == 10
    assert capped_x.shape[0] == 2
    assert capped_idx.tolist() == [3, 4]
    assert int((capped_y != -1).sum()) == 10
    assert int(capped_w.sum()) == 10
    assert int(capped_answer.sum()) == 10


def test_compact_pack_commits_final_partial_accumulation_without_replay() -> None:
    source = (
        Path(__file__).resolve().parents[1] / "scripts" / "build_brain.py"
    ).read_text(encoding="utf-8")

    assert "min(padded_sample_budget, len(ds))" in source
    assert "or sampler_budget_boundary" in source
    assert "signed_window_boundary or sampler_budget_boundary" in source
    assert "compact permutation pack has fewer unique windows" not in source


def test_required_session_acks_primary_then_protects_final_snapshot(
    monkeypatch,
    tmp_path: Path,
) -> None:
    replicas = [
        {
            "name": "drive",
            "path": str(tmp_path / "drive"),
            "kind": "mounted_drive",
            "canonical": True,
        },
        {
            "name": "laptop",
            "path": str(tmp_path / "laptop"),
            "kind": "filesystem",
            "canonical": False,
        },
    ]
    monkeypatch.setenv("ANRA_REQUIRE_DURABLE_ACK", "1")
    monkeypatch.setenv("ANRA_DURABILITY_REPLICAS", json.dumps(replicas))
    checkpoint = tmp_path / "checkpoint.pt"
    _checkpoint(checkpoint, 2048)
    payload = {
        "checkpoint_schema_version": 9,
        "checkpoint_artifact_class": "full_resume",
        "source_commit": "abc123",
        "global_step": 1,
        "tokens_seen": 2048,
        "model_config": {"n_layer": 2},
        "tokenizer_contract": {"sha256": "tokenizer", "schema_version": 4},
        "dataset_manifest_hashes": {"train": "data"},
        "training_recipe": {"seed": 1301},
        "seed_contract": {"seed": 1301},
        "completed_optimizer_boundary": True,
        "accum_micro_steps": 0,
        "model": {},
        "optimizer": {},
        "scheduler": {},
        "scaler": {},
        "rng_states": {},
    }
    session = CheckpointDurabilitySession.from_environment(
        tmp_path / "outbox",
        scratch_run=True,
    )

    ref = session.publish_checkpoint(checkpoint, payload)
    assert ref is not None
    assert session.initial_acknowledged is True
    session.publish_checkpoint(checkpoint, payload, final=True)
    session.close()

    status = json.loads(
        session.outbox.status_path(ref.snapshot_id).read_text(encoding="utf-8")
    )
    assert status["state"] == "protected"
    assert (tmp_path / "drive" / "canonical.json").is_file()
    assert (tmp_path / "laptop" / "manifests" / f"{ref.snapshot_id}.json").is_file()


def test_environment_selects_single_file_drive_replica(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("ANRA_REQUIRE_DURABLE_ACK", "1")
    monkeypatch.setenv(
        "ANRA_DURABILITY_REPLICAS",
        json.dumps(
            [
                {
                    "name": "drive-vault",
                    "path": str(tmp_path / "drive"),
                    "kind": "mounted_drive_single_file",
                    "canonical": True,
                }
            ]
        ),
    )
    checkpoint = tmp_path / "checkpoint.pt"
    _checkpoint(checkpoint, 4096)
    payload = {
        "checkpoint_schema_version": 9,
        "checkpoint_artifact_class": "full_resume",
        "source_commit": "abc123",
        "global_step": 200,
        "tokens_seen": 2048,
        "model_config": {"n_layer": 2},
        "tokenizer_contract": {"sha256": "tokenizer", "schema_version": 4},
        "dataset_manifest_hashes": {"train": "data"},
        "training_recipe": {"seed": 1301},
        "seed_contract": {"seed": 1301},
        "completed_optimizer_boundary": True,
        "accum_micro_steps": 0,
        "model": {},
        "optimizer": {},
        "scheduler": {},
        "scaler": {},
        "rng_states": {},
    }
    session = CheckpointDurabilitySession.from_environment(
        tmp_path / "outbox",
        scratch_run=False,
    )
    try:
        ref = session.publish_checkpoint(checkpoint, payload)
        assert ref is not None
    finally:
        session.close()

    files = list((tmp_path / "drive").glob("*.pt"))
    assert [path.name for path in files] == [
        "anra-v4-current-full-resume.pt"
    ]
    assert not (tmp_path / "drive" / "chunks").exists()


def test_signed_resume_source_replaces_a_stale_destination(tmp_path: Path) -> None:
    source = tmp_path / "signed-parent.pt"
    destination = tmp_path / "mutable-output.pt"
    source.write_bytes(b"signed-parent")
    destination.write_bytes(b"stale-output")

    _prepare_resume_target(destination, str(source))

    assert destination.read_bytes() == b"signed-parent"
    assert source.read_bytes() == b"signed-parent"


def test_required_session_keeps_two_full_snapshots_and_one_inflight_slot(
    monkeypatch,
    tmp_path: Path,
) -> None:
    replicas = [
        {
            "name": "drive",
            "path": str(tmp_path / "drive"),
            "kind": "mounted_drive",
            "canonical": True,
        },
        {
            "name": "laptop",
            "path": str(tmp_path / "laptop"),
            "kind": "filesystem",
            "canonical": False,
        },
    ]
    monkeypatch.setenv("ANRA_REQUIRE_DURABLE_ACK", "1")
    monkeypatch.setenv("ANRA_DURABILITY_REPLICAS", json.dumps(replicas))
    monkeypatch.setenv("ANRA_CLUSTER_CAMPAIGN_ID", "retention-campaign")
    session = CheckpointDurabilitySession.from_environment(
        tmp_path / "outbox",
        scratch_run=True,
    )
    payload = {
        "checkpoint_schema_version": 9,
        "checkpoint_artifact_class": "full_resume",
        "source_commit": "abc123",
        "tokens_seen": 2048,
        "model_config": {"n_layer": 2},
        "tokenizer_contract": {"sha256": "tokenizer", "schema_version": 4},
        "dataset_manifest_hashes": {"train": "data"},
        "training_recipe": {"seed": 1301, "model_profile": "anra-v4-180m"},
        "seed_contract": {"seed": 1301},
        "completed_optimizer_boundary": True,
        "accum_micro_steps": 0,
        "model": {},
        "optimizer": {},
        "scheduler": {},
        "scaler": {},
        "rng_states": {},
    }
    try:
        for step in (1, 2, 3):
            checkpoint = tmp_path / f"checkpoint-{step}.pt"
            _checkpoint(checkpoint, 2048 + step)
            payload["global_step"] = step
            # Deliberately leave the third save non-final. Retention must run
            # as soon as it is protected, not only on a later save or shutdown.
            session.publish_checkpoint(checkpoint, payload, final=False)
    finally:
        session.close()

    assert len(session.outbox.snapshots()) == 2
    assert len(list((tmp_path / "drive" / "manifests").glob("*.json"))) == 2
    assert len(list((tmp_path / "laptop" / "manifests").glob("*.json"))) == 2
