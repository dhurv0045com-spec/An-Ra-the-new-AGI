from __future__ import annotations

from pathlib import Path

import pytest


def test_local_sft_checkpoint_requires_the_single_current_artifact(tmp_path: Path) -> None:
    from runtime.local_checkpoint import (
        CURRENT_SFT_CHECKPOINT_NAME,
        resolve_local_sft_checkpoint,
    )

    checkpoint = tmp_path / CURRENT_SFT_CHECKPOINT_NAME
    checkpoint.write_bytes(b"checkpoint")

    resolved = resolve_local_sft_checkpoint(checkpoint)

    assert resolved.path == checkpoint.resolve()
    assert resolved.source == "explicit"


def test_local_sft_checkpoint_rejects_ambiguous_or_old_names(tmp_path: Path) -> None:
    from runtime.local_checkpoint import resolve_local_sft_checkpoint

    wrong = tmp_path / "anra_v4_180m_rehearsal.pt"
    wrong.write_bytes(b"checkpoint")

    with pytest.raises(ValueError, match="protected current SFT checkpoint"):
        resolve_local_sft_checkpoint(wrong)


def test_prototype_conversation_uses_the_sft_prompt_contract() -> None:
    from runtime.sft_prototype import PrototypeRuntime

    runtime = PrototypeRuntime()
    first = runtime.conversation_prompt("session", "Say hello")
    runtime.add_turn("session", "Say hello", "Hello!")
    second = runtime.conversation_prompt("session", "What next?")

    assert first == "H: USER: Say hello\nANRA:"
    assert second == "H: USER: Say hello\nANRA: Hello!\nUSER: What next?\nANRA:"


def test_prototype_routes_are_available_without_loading_a_model() -> None:
    from runtime.sft_prototype import create_app

    app = create_app()
    routes = {route.path for route in app.routes}

    assert {"/", "/health", "/api/status", "/api/chat", "/api/evaluations/run"} <= routes


def test_prototype_unload_releases_runtime_without_touching_checkpoint(monkeypatch) -> None:
    import runtime.sft_prototype as prototype

    released: list[bool] = []
    monkeypatch.setattr(prototype, "unload_runtime", lambda: released.append(True))
    runtime = prototype.PrototypeRuntime()
    runtime.unload(reason="operator_stop")

    assert released == [True]
    assert runtime.status()["stage"] == "unloaded"
    assert runtime.status()["shutdown_requested"] is True
