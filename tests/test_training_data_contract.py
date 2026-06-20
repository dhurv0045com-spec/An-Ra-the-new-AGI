from __future__ import annotations

import pytest

from scripts.build_brain import (
    _active_training_data_layout,
    _assert_resume_data_layout_compatible,
    _assert_resume_data_profile_compatible,
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


def test_current_trainer_enforces_packed_layout(monkeypatch) -> None:
    monkeypatch.delenv("ANRA_TRAINING_DATA_LAYOUT", raising=False)
    assert _active_training_data_layout() == "bucket_packed_v1"

    monkeypatch.setenv("ANRA_TRAINING_DATA_LAYOUT", "legacy_padded_v0")
    with pytest.raises(RuntimeError, match="only supports"):
        _active_training_data_layout()
