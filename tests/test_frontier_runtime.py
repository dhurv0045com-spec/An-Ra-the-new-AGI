from __future__ import annotations

from pathlib import Path

import pytest


class _Tokenizer:
    vocab_size = 8209
    backend = "test"


class _FakeModel:
    d_model = 1280
    n_layer = 28
    n_head = 16
    n_kv_head = 4
    block_size = 1024

    def to(self, _device):
        return self

    def eval(self):
        return self

    def disable_kv_cache(self):
        self.kv_disabled = True


class _CharacterTokenizer:
    @staticmethod
    def encode(text: str, *, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(character) for character in text]

    @staticmethod
    def decode(ids: list[int]) -> str:
        return "".join(chr(value) for value in ids)


def test_frontier_runtime_refuses_missing_checkpoint(monkeypatch, tmp_path: Path) -> None:
    import generate

    generate._reset_runtime_cache()
    missing = tmp_path / "anra_frontier_500m.pt"
    monkeypatch.setenv("ANRA_MODEL_PROFILE", "frontier")
    monkeypatch.setattr(generate, "FRONTIER_CHECKPOINT", missing)
    monkeypatch.setattr(generate, "_requested_checkpoint_path", lambda: missing)
    monkeypatch.setattr(
        "training.shared_checkpoint.restore_shared_checkpoint",
        lambda *_args, **_kwargs: None,
    )

    with pytest.raises(FileNotFoundError, match="Frontier runtime requested"):
        generate._load_runtime()

    generate._reset_runtime_cache()


def test_frontier_runtime_uses_frontier_builder(monkeypatch, tmp_path: Path) -> None:
    import generate

    generate._reset_runtime_cache()
    checkpoint = tmp_path / "anra_frontier_500m.pt"
    checkpoint.write_bytes(b"fake")
    calls: list[str] = []

    monkeypatch.setenv("ANRA_MODEL_PROFILE", "frontier")
    monkeypatch.setattr(generate, "FRONTIER_CHECKPOINT", checkpoint)
    monkeypatch.setattr(generate, "_requested_checkpoint_path", lambda: checkpoint)
    monkeypatch.setattr(generate, "load_or_build_v2_tokenizer", lambda: _Tokenizer())

    def build_frontier():
        calls.append("frontier")
        return _FakeModel()

    def build_legacy(**_kwargs):
        calls.append("legacy")
        raise AssertionError("legacy builder must not be used for frontier runtime")

    monkeypatch.setattr(generate, "build_frontier_model", build_frontier)
    monkeypatch.setattr(generate, "build_v2_model", build_legacy)
    monkeypatch.setattr(
        generate,
        "load_checkpoint",
        lambda *_args, **_kwargs: {
            "loaded": True,
            "global_step": 6927,
            "best_loss": 0.3279,
            "sessions_completed": 3,
            "data_profile": "t4-cached",
            "training_data_layout": "conversation_pack_v2",
        },
    )
    monkeypatch.setattr(
        generate,
        "model_summary",
        lambda _model: {
            "parameters": generate.V2_FRONTIER_PARAMETER_COUNT,
            "trainable_parameters": generate.V2_FRONTIER_PARAMETER_COUNT,
        },
    )

    model, tokenizer, loaded, profile, state = generate._load_runtime()

    assert isinstance(model, _FakeModel)
    assert tokenizer.vocab_size == 8209
    assert loaded == checkpoint
    assert profile == "frontier"
    assert state["global_step"] == 6927
    assert calls == ["frontier"]

    generate._reset_runtime_cache()


def test_model_info_exposes_frontier_proof_fields(monkeypatch, tmp_path: Path) -> None:
    import generate

    generate._reset_runtime_cache()

    checkpoint = tmp_path / "anra_frontier_500m.pt"
    checkpoint.write_bytes(b"fake")
    monkeypatch.setenv("ANRA_MODEL_PROFILE", "frontier")
    monkeypatch.setattr(generate, "FRONTIER_CHECKPOINT", checkpoint)
    monkeypatch.setattr(generate, "_requested_checkpoint_path", lambda: checkpoint)
    monkeypatch.setattr(generate, "load_or_build_v2_tokenizer", lambda: _Tokenizer())
    monkeypatch.setattr(generate, "build_frontier_model", lambda: _FakeModel())
    monkeypatch.setattr(
        generate,
        "load_checkpoint",
        lambda *_args, **_kwargs: {
            "loaded": True,
            "global_step": 6927,
            "best_loss": 0.3279,
            "sessions_completed": 3,
        },
    )
    monkeypatch.setattr(
        generate,
        "model_summary",
        lambda _model: {
            "parameters": generate.V2_FRONTIER_PARAMETER_COUNT,
            "trainable_parameters": generate.V2_FRONTIER_PARAMETER_COUNT,
        },
    )

    info = generate.get_model_info()

    assert info["profile"] == "frontier"
    assert info["checkpoint"] == str(checkpoint)
    assert info["param_count"] == generate.V2_FRONTIER_PARAMETER_COUNT
    assert info["block_size"] == 1024
    assert info["checkpoint_state"]["global_step"] == 6927

    generate._reset_runtime_cache()


def test_repetition_penalty_moves_repeated_logits_down() -> None:
    import torch
    import generate

    logits = torch.tensor([0.0, 2.0, -2.0])
    adjusted = generate._apply_repetition_penalty(
        logits,
        [1, 2],
        generate.GenerationConfig(repetition_penalty=2.0),
    )
    assert adjusted[1].item() == 1.0
    assert adjusted[2].item() == -4.0


def test_request_scoped_runtime_state_isolation_probe() -> None:
    import generate

    esv_keys = set(generate._ESV_STORE)
    ghost_keys = set(generate._GHOST_STORE)
    report = generate.verify_session_state_isolation()

    assert report["verified"] is True
    assert report["generation_serialized"] is True
    assert set(generate._ESV_STORE) == esv_keys
    assert set(generate._GHOST_STORE) == ghost_keys


def test_prompt_assembly_preserves_current_message_and_inserts_memory_once() -> None:
    from inference.optimize_context_window import ContextWindowOptimizer

    optimizer = ContextWindowOptimizer(_CharacterTokenizer(), max_context=180)
    result = optimizer.build_optimized_context(
        [("old question", "old answer")],
        [{"content": "remember cobalt"}],
        "current message must remain complete",
        max_new_tokens=32,
        mode="full_system",
    )
    assert result["formatted_prompt"].endswith("H: current message must remain complete\nANRA:")
    assert result["formatted_prompt"].count("remember cobalt") == 1
    assert result["prompt_tokens"] + result["reserved_output_tokens"] + 1 <= 180


def test_prompt_assembly_truncates_old_history_before_current_message() -> None:
    from inference.optimize_context_window import ContextWindowOptimizer

    optimizer = ContextWindowOptimizer(_CharacterTokenizer(), max_context=96)
    result = optimizer.build_optimized_context(
        [("old " * 20, "answer " * 20)],
        [],
        "newest request",
        max_new_tokens=24,
        mode="diagnostic",
    )
    assert "newest request" in result["formatted_prompt"]
    assert result["context_truncated"] is True
