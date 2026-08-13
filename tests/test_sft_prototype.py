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


def test_prototype_defaults_to_visible_proof_first_mode() -> None:
    from runtime.sft_prototype import ChatRequest, PROTOTYPE_HTML

    request = ChatRequest(message="What is 23 plus 34?")

    assert request.assistance.mode == "proof_first"
    assert request.assistance.allow_calculator is False
    assert request.assistance.candidate_count == 2
    assert 'id="assistance"' in PROTOTYPE_HTML
    assert 'id="allow-calculator"' in PROTOTYPE_HTML
    assert "Proof-first (recommended)" in PROTOTYPE_HTML
    assert "assistance:assistance()" in PROTOTYPE_HTML
    assert "$('assistance').onchange=syncModes" in PROTOTYPE_HTML


def test_proof_first_calculator_uses_and_revokes_one_call_grant(monkeypatch) -> None:
    import runtime.tool_broker as tool_broker
    from runtime.sft_prototype import PrototypeRuntime

    monkeypatch.setattr(
        tool_broker,
        "record_experience",
        lambda **_kwargs: ("trace", True),
    )
    runtime = PrototypeRuntime()

    value, receipt = runtime.calculate_for_chat(
        session_id="customer-session",
        expression="23 + 34",
    )

    assert value == 57
    assert receipt["status"] == "completed"
    assert receipt["calls_remaining"] == 0
    assert runtime.status()["tools"]["active_grants"] == 0


def test_prototype_unload_releases_runtime_without_touching_checkpoint(monkeypatch) -> None:
    import runtime.sft_prototype as prototype

    released: list[bool] = []
    monkeypatch.setattr(prototype, "unload_runtime", lambda: released.append(True))
    runtime = prototype.PrototypeRuntime()
    runtime.unload(reason="operator_stop")

    assert released == [True]
    assert runtime.status()["stage"] == "unloaded"
    assert runtime.status()["shutdown_requested"] is True
