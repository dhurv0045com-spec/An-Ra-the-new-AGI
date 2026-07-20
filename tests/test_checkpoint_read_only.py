from __future__ import annotations

from pathlib import Path

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
