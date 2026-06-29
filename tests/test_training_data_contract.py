from __future__ import annotations

from pathlib import Path
import hashlib

import pytest

from scripts.build_brain import (
    _active_training_data_layout,
    _assert_resume_data_layout_compatible,
    _assert_resume_data_profile_compatible,
    _collect_data_manifest_payloads,
    _freeze_training_lineage,
)


def test_resume_accepts_matching_data_profile(monkeypatch) -> None:
    monkeypatch.delenv("ANRA_ALLOW_DATA_PROFILE_CHANGE", raising=False)
    _assert_resume_data_profile_compatible("t4-15gb", "t4-15gb")


def test_resume_rejects_changed_data_profile(monkeypatch) -> None:
    monkeypatch.delenv("ANRA_ALLOW_DATA_PROFILE_CHANGE", raising=False)

    with pytest.raises(RuntimeError, match="different data profile"):
        _assert_resume_data_profile_compatible("t4-15gb", "t4-cached")


def test_resume_allows_explicit_profile_experiment(monkeypatch) -> None:
    monkeypatch.setenv("ANRA_ALLOW_DATA_PROFILE_CHANGE", "1")
    _assert_resume_data_profile_compatible("t4-15gb", "t4-cached")


def test_resume_rejects_changed_data_layout() -> None:
    with pytest.raises(RuntimeError, match="different training data layout"):
        _assert_resume_data_layout_compatible("legacy_padded_v0", "bucket_packed_v1")


@pytest.mark.parametrize(
    ("saved", "active", "phase"),
    [
        ("bucket_packed_v1", "raw_causal_shards_v1", "A"),
        ("raw_causal_shards_v1", "bucket_packed_v1", "D"),
    ],
)
def test_resume_allows_planned_curriculum_layout_transition(
    saved: str,
    active: str,
    phase: str,
) -> None:
    _assert_resume_data_layout_compatible(saved, active, phase)


def test_current_trainer_enforces_packed_layout(monkeypatch) -> None:
    monkeypatch.delenv("ANRA_TRAINING_DATA_LAYOUT", raising=False)
    assert _active_training_data_layout() == "bucket_packed_v1"

    monkeypatch.setenv("ANRA_TRAINING_DATA_LAYOUT", "legacy_padded_v0")
    with pytest.raises(RuntimeError, match="only supports"):
        _active_training_data_layout()


def test_training_lineage_freezes_checkpoint_tokenizer_and_manifests(
    tmp_path,
    monkeypatch,
) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    tokenizer = tmp_path / "tokenizer.json"
    manifest = tmp_path / "manifest.json"
    checkpoint.write_bytes(b"checkpoint-v1")
    tokenizer.write_text('{"token_to_id": {"<pad>": 0}}', encoding="utf-8")
    manifest.write_text('{"shards": []}', encoding="utf-8")
    monkeypatch.setattr("scripts.build_brain.OUTPUT_V2_DIR", tmp_path / "output")

    frozen = _freeze_training_lineage(
        checkpoint_path=checkpoint,
        tokenizer_path=tokenizer,
        model_config={"vocab_size": 8209},
        data_manifests=[manifest],
    )
    checkpoint.write_bytes(b"checkpoint-v2")

    archived = frozen["checkpoint_archive"]
    assert archived is not None
    assert Path(archived).read_bytes() == b"checkpoint-v1"
    assert frozen["data_manifest_sha256"]


def test_checkpoint_collects_complete_manifest_bytes(tmp_path: Path) -> None:
    root = tmp_path / "manifests"
    nested = root / "native" / "manifest.json"
    nested.parent.mkdir(parents=True)
    nested.write_bytes(b'{"shards":[{"sha256":"abc"}]}')

    hashes, payloads = _collect_data_manifest_payloads(root)

    assert payloads == {"native/manifest.json": nested.read_bytes()}
    assert hashes["native/manifest.json"] == hashlib.sha256(nested.read_bytes()).hexdigest()
