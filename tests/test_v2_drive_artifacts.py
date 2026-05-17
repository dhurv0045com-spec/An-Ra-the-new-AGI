from __future__ import annotations

import os
from pathlib import Path

import torch

import training.v2_runtime as rt


def _patch_artifact_paths(monkeypatch, tmp_path: Path) -> tuple[Path, Path]:
    local = tmp_path / "repo"
    drive = tmp_path / "drive" / "AnRa"
    monkeypatch.setattr(rt, "DRIVE_DIR", drive)
    monkeypatch.setattr(rt, "DRIVE_V2_CHECKPOINTS", drive / "v2" / "checkpoints")
    monkeypatch.setattr(rt, "ROOT", local)
    monkeypatch.setattr(rt, "V2_BRAIN_CHECKPOINT", local / "anra_v2_brain.pt")
    monkeypatch.setattr(rt, "V2_IDENTITY_CHECKPOINT", local / "anra_v2_identity.pt")
    monkeypatch.setattr(rt, "V2_OUROBOROS_CHECKPOINT", local / "anra_v2_ouroboros.pt")
    monkeypatch.setattr(rt, "V3_TOKENIZER_FILE", local / "tokenizer" / "tokenizer_v3.json")
    monkeypatch.setattr(rt, "OUTPUT_V2_DIR", local / "output" / "v2")
    return local, drive


def test_sync_to_drive_uses_only_fixed_artifact_names(monkeypatch, tmp_path: Path) -> None:
    _local, drive = _patch_artifact_paths(monkeypatch, tmp_path)
    rt.V2_BRAIN_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    rt.V3_TOKENIZER_FILE.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"step": 7, "model": {}}, rt.V2_BRAIN_CHECKPOINT)
    rt.V3_TOKENIZER_FILE.write_text('{"token_to_id": {}}', encoding="utf-8")

    assert rt.sync_to_drive("brain")
    assert rt.sync_to_drive("tokenizer")

    fixed_files = {
        drive / "anra_v2_brain.pt",
        drive / "tokenizer_v3.json",
        drive / "v2" / "checkpoints" / "anra_v2_brain.pt",
        drive / "v2" / "checkpoints" / "tokenizer_v3.json",
    }
    for path in fixed_files:
        assert path.exists(), path

    assert not (drive / "sessions").exists()
    assert list((drive / "v2" / "checkpoints").glob("*step*.pt")) == []
    assert list((drive / "v2" / "checkpoints").glob("*_v1_*.pt")) == []


def test_restore_v2_artifact_keeps_newer_local_checkpoint(monkeypatch, tmp_path: Path) -> None:
    _local, drive = _patch_artifact_paths(monkeypatch, tmp_path)
    drive_ckpt = drive / "v2" / "checkpoints" / "anra_v2_brain.pt"
    drive_ckpt.parent.mkdir(parents=True, exist_ok=True)
    rt.V2_BRAIN_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)

    torch.save({"step": 1}, drive_ckpt)
    torch.save({"step": 99}, rt.V2_BRAIN_CHECKPOINT)
    future = drive_ckpt.stat().st_mtime + 100
    os.utime(rt.V2_BRAIN_CHECKPOINT, (future, future))

    assert rt.restore_v2_artifact("brain")
    assert rt._read_step(rt.V2_BRAIN_CHECKPOINT) == 99


def test_restore_tokenizer_accepts_legacy_v2_drive_name(monkeypatch, tmp_path: Path) -> None:
    _local, drive = _patch_artifact_paths(monkeypatch, tmp_path)
    legacy_tokenizer = drive / "v2" / "checkpoints" / "tokenizer_v2.json"
    legacy_tokenizer.parent.mkdir(parents=True, exist_ok=True)
    legacy_tokenizer.write_text('{"legacy": true}', encoding="utf-8")

    assert rt.restore_v2_artifact("tokenizer")
    assert rt.V3_TOKENIZER_FILE.read_text(encoding="utf-8") == '{"legacy": true}'
