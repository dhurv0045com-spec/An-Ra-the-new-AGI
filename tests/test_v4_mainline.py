from __future__ import annotations

import json

from anra.anra_paths import V4_TOKENIZER_FILE
from training.v2_config import (
    ANRA_V4_MODEL,
    ANRA_V4_MODEL_PARAMETER_COUNT,
    CANONICAL_MODEL_PROFILE,
    EXPECTED_TOKENIZER_VOCAB_SIZE,
    MODEL_SIZES,
    model_parameter_count,
    resolve_model_profile,
)
from training.v2_runtime import active_tokenizer_path, load_or_build_v2_tokenizer


def test_v4_is_the_only_active_model_profile() -> None:
    assert set(MODEL_SIZES) == {CANONICAL_MODEL_PROFILE}
    model, _training = resolve_model_profile(CANONICAL_MODEL_PROFILE)
    assert model is ANRA_V4_MODEL
    assert model_parameter_count(model) == ANRA_V4_MODEL_PARAMETER_COUNT


def test_v4_is_the_only_active_tokenizer(monkeypatch) -> None:
    monkeypatch.delenv("ANRA_TOKENIZER_PATH", raising=False)
    assert active_tokenizer_path() == V4_TOKENIZER_FILE
    assert EXPECTED_TOKENIZER_VOCAB_SIZE == 32_768


def test_active_runtime_loads_canonical_v4() -> None:
    tokenizer = load_or_build_v2_tokenizer()
    assert tokenizer.vocab_size == 32_768
    assert tokenizer.pad_token_id == 0


def test_active_runtime_ignores_retired_tokenizer_override(monkeypatch) -> None:
    monkeypatch.setenv("ANRA_TOKENIZER_PATH", "tokenizer/tokenizer_v3.json")
    assert active_tokenizer_path() == V4_TOKENIZER_FILE


def test_t4_notebook_has_no_legacy_training_path() -> None:
    notebook_path = V4_TOKENIZER_FILE.parents[1] / "notebooks" / "AN_RA_T4_TRAINING.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    text = json.dumps(notebook)
    assert "--model-size anra-v4-180m" in text
    assert "native_foundation_v4" in text
    assert "--model-size frontier" not in text
    assert "native_foundation_v3" not in text
