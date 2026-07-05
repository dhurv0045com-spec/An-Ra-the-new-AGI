"""Phase-3 ghost memory and identity injector must be live, not silently dead.

Historically ``generate.py`` imported ``ghost_memory`` and ``identity_injector``
by bare names that always raised ModuleNotFoundError, leaving both permanently
``None`` while traces still claimed ghost execution. The dead imports also hid
three API mismatches (``GhostMemory.store``/``retrieve(session_id)`` and
``IdentityInjector.clean`` do not exist). These tests lock in the revival.
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


def test_phase3_modules_actually_import() -> None:
    # The classes must load from their canonical packages; a bare-name import
    # regression would silently set them back to None.
    assert generate._GhostMemory is not None
    assert generate._ghost_default_config is not None


def test_ghost_memory_is_session_isolated_and_retrieves(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(generate, "_GHOST_MEMORY_DIR", tmp_path / "ghost")
    monkeypatch.setattr(generate, "_GHOST_MEMORY_SESSIONS", {})
    ghost_a = generate._ghost_memory_for("session-a")
    ghost_b = generate._ghost_memory_for("session-b")
    assert ghost_a is not None and ghost_b is not None and ghost_a is not ghost_b

    ghost_a.add_turn("anra", "the cobalt key is stored in vault seven")
    ghost_b.add_turn("anra", "unrelated text about weather patterns")

    hits_a = ghost_a.retrieve("cobalt key vault")
    assert any("cobalt" in str(hit) for hit in hits_a)
    # Zero cross-session leakage: B must never see A's memory.
    assert not any("cobalt" in str(hit) for hit in ghost_b.retrieve("cobalt key vault"))
    # The threshold must still reject unrelated content within a session.
    assert not any("weather" in str(hit) for hit in hits_a)


def test_accepted_full_system_output_persists_durably(tmp_path: Path, monkeypatch) -> None:
    model = _tiny_model()
    monkeypatch.setattr(generate, "_get_runtime", lambda: (model, _Tokenizer(), tmp_path / "x.pt"))
    monkeypatch.setattr(
        generate, "_RUNTIME_LOAD_STATE", {"load_report": {"exact_native_load": True}}
    )
    monkeypatch.setattr(generate, "_get_hal", lambda _session_id: None)
    monkeypatch.setattr(generate, "_GHOST_MEMORY_DIR", tmp_path / "ghost")
    monkeypatch.setattr(generate, "_GHOST_MEMORY_SESSIONS", {})
    monkeypatch.setattr(generate, "_generation_quality", lambda *_a, **_k: 1.0)
    monkeypatch.setattr(generate, "_language_fragment_detected", lambda _text: False)

    session_id = "ghost_durability_probe"
    trace = generate.generate_traced(
        "H: probe\nANRA:",
        generate.GenerationConfig(max_tokens=2, mode="full_system", persist_adaptive_state=True),
        session_id=session_id,
    )
    assert trace.quality_state == "accepted"
    assert trace.subsystem_trace["ghost_executed"] is True

    # The accepted output must be retrievable from the durable per-session store.
    ghost = generate._ghost_memory_for(session_id)
    assert ghost is not None
    hits = ghost.retrieve("valid complete response")
    assert any("valid complete response" in str(hit) for hit in hits)

    # load_ghost_state must expose ranked snippets through the repaired API.
    state = generate.load_ghost_state(session_id)
    assert state["session_id"] == session_id
    assert isinstance(state["snippets"], list)

    generate.clear_session_runtime_state(session_id)
    assert session_id not in generate._GHOST_STORE
    assert generate._ghost_session_key(session_id) not in generate._GHOST_MEMORY_SESSIONS


def test_identity_cleanup_uses_real_method(tmp_path: Path, monkeypatch) -> None:
    class Injector:
        def clean_response(self, response: str) -> str:
            return response.replace("valid", "verified")

    model = _tiny_model()
    monkeypatch.setattr(generate, "_get_runtime", lambda: (model, _Tokenizer(), tmp_path / "x.pt"))
    monkeypatch.setattr(
        generate, "_RUNTIME_LOAD_STATE", {"load_report": {"exact_native_load": True}}
    )
    monkeypatch.setattr(generate, "_get_hal", lambda _session_id: None)
    monkeypatch.setattr(generate, "_IDENTITY_INJECTOR", Injector())

    trace = generate.generate_traced(
        "H: probe\nANRA:",
        generate.GenerationConfig(max_tokens=2, mode="diagnostic"),
        session_id="identity_cleanup_probe",
    )
    assert trace.output.startswith("verified complete response")


def test_real_identity_injector_rewrites_robotic_phrases() -> None:
    if generate._IDENTITY_INJECTOR is None:
        pytest.skip("no identity file available in this environment")
    cleaned = generate._IDENTITY_INJECTOR.clean_response("I am an AI language model and I help.")
    assert "An-Ra" in cleaned
