from __future__ import annotations

from pathlib import Path

import pytest
import training.shared_checkpoint as shared
from scripts.build_brain import _prepare_resume_target, _sync_training_checkpoint_to_drive


def test_explicit_resume_source_is_never_a_publish_target(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "source" / "part1.pt"
    destination = tmp_path / "artifacts" / "part2.pt"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"immutable-part-1")

    monkeypatch.setattr(shared, "CHECKPOINT_ORIGIN_DIR", tmp_path / "origins")
    monkeypatch.setattr(shared, "DRIVE_V2_CHECKPOINTS", tmp_path / "drive")

    _prepare_resume_target(destination, str(source))
    destination.write_bytes(b"continued-part-2")
    _sync_training_checkpoint_to_drive(destination)

    assert source.read_bytes() == b"immutable-part-1"
    assert (tmp_path / "drive" / destination.name).read_bytes() == b"continued-part-2"


def test_unavailable_optional_drive_does_not_abort_local_training(
    monkeypatch, tmp_path: Path
) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"durable-local-state")
    monkeypatch.delenv("ANRA_REQUIRE_SHARED_MASTER", raising=False)
    monkeypatch.setattr(
        "scripts.build_brain.sync_checkpoint_to_origin",
        lambda _path: (_ for _ in ()).throw(PermissionError("no mounted drive")),
    )

    _sync_training_checkpoint_to_drive(checkpoint)
    assert checkpoint.read_bytes() == b"durable-local-state"


def test_required_shared_master_still_fails_closed(monkeypatch, tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"durable-local-state")
    monkeypatch.setenv("ANRA_REQUIRE_SHARED_MASTER", "1")
    monkeypatch.setattr(
        "scripts.build_brain.sync_checkpoint_to_origin",
        lambda _path: (_ for _ in ()).throw(PermissionError("no mounted drive")),
    )

    with pytest.raises(PermissionError, match="no mounted drive"):
        _sync_training_checkpoint_to_drive(checkpoint)
