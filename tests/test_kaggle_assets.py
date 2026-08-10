from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from training.kaggle_assets import resolve_kaggle_training_assets


def _write_home(root: Path, name: str = "private-anra") -> Path:
    home = root / name / "ANRA_T4_TRAINING_HOME"
    home.mkdir(parents=True)
    payload = b"portable-foundation-checkpoint"
    checkpoint = home / "anra-v4-current-full-resume.pt"
    checkpoint.write_bytes(payload)
    (home / "anra-v4-current-full-resume.json").write_text(
        json.dumps(
            {
                "global_step": 10_200,
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    return home


def test_resolves_one_verified_private_dataset_snapshot(tmp_path: Path) -> None:
    home = _write_home(tmp_path)

    assets = resolve_kaggle_training_assets(tmp_path)

    assert assets.training_home == home.resolve()
    assert assets.global_step == 10_200
    assert len(assets.checkpoint_sha256) == 64


def test_rejects_ambiguous_checkpoint_lineages(tmp_path: Path) -> None:
    _write_home(tmp_path, "first")
    _write_home(tmp_path, "second")

    with pytest.raises(RuntimeError, match="exactly one canonical"):
        resolve_kaggle_training_assets(tmp_path)


def test_rejects_checkpoint_pointer_hash_mismatch(tmp_path: Path) -> None:
    home = _write_home(tmp_path)
    pointer = home / "anra-v4-current-full-resume.json"
    metadata = json.loads(pointer.read_text(encoding="utf-8"))
    metadata["sha256"] = "0" * 64
    pointer.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="SHA-256"):
        resolve_kaggle_training_assets(tmp_path)
