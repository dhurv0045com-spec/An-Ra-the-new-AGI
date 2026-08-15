"""CIV similarity in the live generation path must be measured, not constant.

Historically the /chat path never set ``civ_score``, so the MoD router's
``router_civ_similarity`` telemetry was a hardcoded default of 1.0 presented
as "residual guard" evidence. These tests lock in the measured behavior: the
session's constitutional identity vector supplies the score, verified evidence
updates it, and diagnostic mode stays neutral.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import generate
from anra_brain import CausalTransformerV2


class _Tokenizer:
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 0
    vocab_size = 64
    special_ids = {"<pad>": 0, "<bos>": 1, "<eos>": 2}

    @staticmethod
    def encode(text: str, *, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [3 + (ord(character) % 50) for character in text]

    @staticmethod
    def decode(_ids: list[int]) -> str:
        return "valid complete response with enough words"


def _tiny_model() -> CausalTransformerV2:
    torch.manual_seed(1)
    return CausalTransformerV2(
        vocab_size=64,
        n_embd=32,
        n_head=4,
        n_kv_head=2,
        n_layer=2,
        block_size=64,
        mod_layers={1},
    ).eval()


@pytest.fixture
def runtime(monkeypatch, tmp_path: Path) -> CausalTransformerV2:
    model = _tiny_model()
    monkeypatch.setattr(generate, "_get_runtime", lambda: (model, _Tokenizer(), tmp_path / "x.pt"))
    monkeypatch.setattr(
        generate, "_RUNTIME_LOAD_STATE", {"load_report": {"exact_native_load": True}}
    )
    monkeypatch.setattr(generate, "_get_hal", lambda _session_id: None)
    monkeypatch.setattr(generate, "_CIV_DIR", tmp_path / "civ")
    monkeypatch.setattr(generate, "_CIV_STORE", {})
    monkeypatch.setattr(generate, "_generation_quality", lambda *_a, **_k: 1.0)
    monkeypatch.setattr(generate, "_language_fragment_detected", lambda _text: False)
    return model


def _router_similarity(trace) -> float:
    return float(trace.subsystem_trace["model"]["router_civ_similarity"])


def test_native_mode_reports_measured_civ_not_constant(runtime) -> None:
    trace = generate.generate_traced(
        "H: probe\nANRA:",
        generate.GenerationConfig(max_tokens=2, mode="native", persist_adaptive_state=False),
        session_id="civ_measured_probe",
    )
    # Fresh profile mean is (0.8 + 0.9 + 0.7 + 0.8) / 4 = 0.8; the old wiring
    # silently defaulted to 1.0 regardless of any identity state.
    assert _router_similarity(trace) == pytest.approx(0.8)


def test_explicit_operator_civ_score_wins(runtime) -> None:
    trace = generate.generate_traced(
        "H: probe\nANRA:",
        generate.GenerationConfig(
            max_tokens=2, mode="native", persist_adaptive_state=False, civ_score=0.42
        ),
        session_id="civ_override_probe",
    )
    assert _router_similarity(trace) == pytest.approx(0.42)


def test_diagnostic_mode_stays_neutral(runtime) -> None:
    trace = generate.generate_traced(
        "H: probe\nANRA:",
        generate.GenerationConfig(max_tokens=2, mode="diagnostic"),
        session_id="civ_diagnostic_probe",
    )
    assert _router_similarity(trace) == pytest.approx(1.0)


def test_accepted_output_updates_and_persists_civ(runtime, tmp_path: Path) -> None:
    session_id = "civ_update_probe"
    generate.generate_traced(
        "H: probe\nANRA:",
        generate.GenerationConfig(
            max_tokens=2, mode="full_system", persist_adaptive_state=True
        ),
        session_id=session_id,
    )
    civ = generate._CIV_STORE[session_id]
    # Quality was 1.0, so measured coherence evidence must pull the profile's
    # coherence upward from its 0.8 default.
    assert float(civ.profile.coherence) > 0.8
    assert generate._civ_path(session_id).exists()

    generate.clear_session_runtime_state(session_id)
    assert session_id not in generate._CIV_STORE
    assert not generate._civ_path(session_id).exists()


def test_rejected_output_does_not_update_civ(runtime, monkeypatch) -> None:
    monkeypatch.setattr(generate, "_generation_quality", lambda *_a, **_k: 0.0)
    session_id = "civ_rejected_probe"
    trace = generate.generate_traced(
        "H: probe\nANRA:",
        generate.GenerationConfig(
            max_tokens=2, mode="full_system", persist_adaptive_state=True
        ),
        session_id=session_id,
    )
    assert trace.quality_state == "rejected"
    civ = generate._CIV_STORE[session_id]
    assert float(civ.profile.coherence) == pytest.approx(0.8)
    assert not generate._civ_path(session_id).exists()
