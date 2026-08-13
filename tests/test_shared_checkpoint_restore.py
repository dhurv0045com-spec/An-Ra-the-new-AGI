from __future__ import annotations

from pathlib import Path
import os
import time

import training.shared_checkpoint as shared
import pytest


def _patch_drive(monkeypatch, tmp_path: Path) -> Path:
    mounted = tmp_path / "drive"
    my_drive = mounted / "MyDrive"
    anra = my_drive / "AnRa"
    monkeypatch.setattr(shared, "DRIVE_ROOT", my_drive)
    monkeypatch.setattr(shared, "DRIVE_DIR", anra)
    monkeypatch.setattr(shared, "DRIVE_V2_CHECKPOINTS", anra / "v2" / "checkpoints")
    monkeypatch.setattr(shared, "CHECKPOINT_ORIGIN_DIR", tmp_path / "origins")
    monkeypatch.setenv("ANRA_SHARED_DRIVE_API", "0")
    return mounted


def test_restore_shared_checkpoint_from_shared_drive_filesystem(monkeypatch, tmp_path: Path) -> None:
    mounted = _patch_drive(monkeypatch, tmp_path)
    source = mounted / "Shareddrives" / "research" / "anra_frontier_500m.pt"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"checkpoint")
    destination = tmp_path / "repo" / "anra_frontier_500m.pt"

    restored = shared.restore_shared_checkpoint(destination)

    assert restored == source
    assert destination.read_bytes() == b"checkpoint"


def test_restore_shared_checkpoint_from_override_dir(monkeypatch, tmp_path: Path) -> None:
    _patch_drive(monkeypatch, tmp_path)
    override = tmp_path / "shared-with-me"
    source = override / "anra_frontier_500m.pt"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"override")
    destination = tmp_path / "repo" / "anra_frontier_500m.pt"
    monkeypatch.setenv("ANRA_SHARED_CHECKPOINT_DIR", str(override))

    restored = shared.restore_shared_checkpoint(destination)

    assert restored == source
    assert destination.read_bytes() == b"override"


def test_shared_filesystem_checkpoint_is_updated_in_place(monkeypatch, tmp_path: Path) -> None:
    mounted = _patch_drive(monkeypatch, tmp_path)
    source = mounted / "Shareddrives" / "research" / "anra_frontier_500m.pt"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"old")
    destination = tmp_path / "repo" / "anra_frontier_500m.pt"

    restored = shared.restore_shared_checkpoint(destination)
    destination.write_bytes(b"new")
    published = shared.sync_checkpoint_to_origin(destination)

    assert restored == source
    assert published == source
    assert source.read_bytes() == b"new"
    assert not (tmp_path / "drive" / "MyDrive" / "AnRa" / "v2" / "checkpoints" / source.name).exists()


def test_drive_api_origin_uses_api_publisher(monkeypatch, tmp_path: Path) -> None:
    _patch_drive(monkeypatch, tmp_path)
    checkpoint = tmp_path / "repo" / "anra_frontier_500m.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    shared._record_origin(
        checkpoint.name,
        {"kind": "drive_api", "file_id": "master-file", "version": "12"},
    )
    expected = Path("drive-api:master-file")
    monkeypatch.setattr(shared, "_upload_checkpoint_to_drive_api", lambda path, origin: expected)

    published = shared.sync_checkpoint_to_origin(checkpoint)

    assert published == expected


def test_recorded_drive_update_is_pinned_to_restored_id_and_version(
    monkeypatch, tmp_path: Path
) -> None:
    _patch_drive(monkeypatch, tmp_path)
    checkpoint = tmp_path / "anra-v4-current-full-resume.pt"
    checkpoint.write_bytes(b"checkpoint")
    shared._record_origin(
        checkpoint.name,
        {
            "kind": "drive_api",
            "file_id": "canonical-id",
            "version": "41",
            "app_properties": {"snapshot_id": "snapshot-40"},
        },
    )
    observed: dict[str, object] = {}

    def stable_update(source: Path, filename: str, **kwargs: object) -> dict[str, object]:
        observed.update({"source": source, "filename": filename, **kwargs})
        return {"id": "canonical-id", "version": "42"}

    monkeypatch.setattr(shared, "update_drive_file_by_name", stable_update)

    result = shared.update_recorded_drive_file(
        checkpoint,
        checkpoint.name,
        app_properties={"global_step": "200"},
    )

    assert result["id"] == "canonical-id"
    assert observed["preferred_file_id"] == "canonical-id"
    assert observed["expected_version"] == "41"
    assert observed["expected_app_properties"] == {"snapshot_id": "snapshot-40"}
    assert observed["cleanup_duplicates"] is True


def test_drive_generation_match_requires_the_same_nonempty_snapshot_identity() -> None:
    target = {
        "appProperties": {
            "snapshot_id": "snapshot-200",
            "sha256": "abc123",
            "unrelated": "allowed",
        }
    }

    assert shared._matches_recorded_generation(
        target,
        {"snapshot_id": "snapshot-200", "sha256": "abc123"},
    )
    assert not shared._matches_recorded_generation(
        target,
        {"snapshot_id": "another-writer", "sha256": "abc123"},
    )
    assert not shared._matches_recorded_generation(target, {})


def test_required_shared_master_never_creates_private_drive_copy(monkeypatch, tmp_path: Path) -> None:
    _patch_drive(monkeypatch, tmp_path)
    checkpoint = tmp_path / "repo" / "anra_frontier_500m.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setenv(shared.REQUIRE_SHARED_MASTER_ENV, "1")

    with pytest.raises(RuntimeError, match="No shared master checkpoint origin"):
        shared.sync_checkpoint_to_origin(checkpoint)

    assert not (tmp_path / "drive" / "MyDrive" / "AnRa" / "v2" / "checkpoints").exists()


def test_required_master_accepts_owner_mydrive_and_updates_it_in_place(monkeypatch, tmp_path: Path) -> None:
    _patch_drive(monkeypatch, tmp_path)
    filename = "anra_frontier_500m.pt"
    master = shared.DRIVE_V2_CHECKPOINTS / filename
    master.parent.mkdir(parents=True)
    master.write_bytes(b"owner-master")
    destination = tmp_path / "repo" / filename
    monkeypatch.setenv(shared.REQUIRE_SHARED_MASTER_ENV, "1")

    restored = shared.restore_shared_checkpoint(destination)
    destination.write_bytes(b"updated-master")
    published = shared.sync_checkpoint_to_origin(destination)

    assert restored == master
    assert published == master
    assert master.read_bytes() == b"updated-master"


def test_filesystem_resume_chooses_newest_duplicate_checkpoint(monkeypatch, tmp_path: Path) -> None:
    mounted = _patch_drive(monkeypatch, tmp_path)
    filename = "anra_frontier_500m.pt"
    stale = shared.DRIVE_V2_CHECKPOINTS / filename
    newest = mounted / "Shareddrives" / "research" / filename
    stale.parent.mkdir(parents=True)
    newest.parent.mkdir(parents=True)
    stale.write_bytes(b"step-450")
    newest.write_bytes(b"step-800")
    now = time.time()
    os.utime(stale, (now - 60, now - 60))
    os.utime(newest, (now, now))

    assert shared.find_filesystem_checkpoint(filename) == newest
