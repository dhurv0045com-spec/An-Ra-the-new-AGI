from __future__ import annotations

from pathlib import Path

import training.shared_checkpoint as shared


def _patch_drive(monkeypatch, tmp_path: Path) -> Path:
    mounted = tmp_path / "drive"
    my_drive = mounted / "MyDrive"
    anra = my_drive / "AnRa"
    monkeypatch.setattr(shared, "DRIVE_ROOT", my_drive)
    monkeypatch.setattr(shared, "DRIVE_DIR", anra)
    monkeypatch.setattr(shared, "DRIVE_V2_CHECKPOINTS", anra / "v2" / "checkpoints")
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
