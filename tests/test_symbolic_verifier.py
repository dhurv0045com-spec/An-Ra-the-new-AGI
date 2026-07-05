"""The symbolic falsification pass must check answers, not assert confidence.

DFC doctrine: on checkable tasks (math/logic) the runtime derives the answer
independently through the 45Q symbolic bridge and scores the model's output
against it. The score feeds HAL/CIV truthfulness only when a verifier really
ran; natural-language tasks yield no fabricated verdict.
"""

from __future__ import annotations

from pathlib import Path

import torch

import generate
from anra_brain import CausalTransformerV2


class _Tokenizer:
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 0
    vocab_size = 64
    special_ids = {"<pad>": 0, "<bos>": 1, "<eos>": 2}

    def __init__(self, reply: str) -> None:
        self.reply = reply

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [3 + (ord(character) % 50) for character in text]

    def decode(self, _ids: list[int]) -> str:
        return self.reply


def _trace_for(reply: str, monkeypatch, tmp_path: Path, **config_kwargs):
    torch.manual_seed(1)
    model = CausalTransformerV2(
        vocab_size=64,
        n_embd=32,
        n_head=4,
        n_kv_head=2,
        n_layer=2,
        block_size=64,
        mod_layers={1},
    ).eval()
    monkeypatch.setattr(
        generate, "_get_runtime", lambda: (model, _Tokenizer(reply), tmp_path / "x.pt")
    )
    monkeypatch.setattr(
        generate, "_RUNTIME_LOAD_STATE", {"load_report": {"exact_native_load": True}}
    )
    monkeypatch.setattr(generate, "_get_hal", lambda _session_id: None)
    config = generate.GenerationConfig(
        max_tokens=2, mode="full_system", persist_adaptive_state=False, **config_kwargs
    )
    return generate.generate_traced(
        "H: Differentiate x^2 + 3*x\nANRA:", config, session_id="symbolic_probe"
    )


def test_correct_symbolic_answer_scores_one(monkeypatch, tmp_path: Path) -> None:
    trace = _trace_for("The derivative is 2*x + 3 by the power rule.", monkeypatch, tmp_path)
    report = trace.subsystem_trace["symbolic_verifier"]
    assert report["mode"] == "MATH"
    assert report["verdict"] == "VERIFIED"
    assert report["score"] == 1.0


def test_wrong_symbolic_answer_scores_zero(monkeypatch, tmp_path: Path) -> None:
    trace = _trace_for("The derivative is 7*x, obviously.", monkeypatch, tmp_path)
    report = trace.subsystem_trace["symbolic_verifier"]
    assert report["score"] == 0.0


def test_natural_language_prompt_yields_no_verdict() -> None:
    report = generate._symbolic_verify("H: how was your day?\nANRA:", "it was fine")
    assert report is None


def test_extract_user_message_takes_last_human_turn() -> None:
    prompt = "identity block\nH: first question\nANRA: first answer\nH: second question\nANRA:"
    assert generate._extract_user_message(prompt) == "second question"


def test_symbolic_score_feeds_hal_truthfulness(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    class HalRecorder:
        def generation_temperature(self, temperature: float) -> float:
            return temperature

        def update(self, *, verifier_result, session_context) -> None:
            captured["verifier_result"] = verifier_result
            captured["truthfulness"] = session_context["civ_evidence"]["truthfulness"]

        def save(self, *_args: object, **_kwargs: object) -> None:
            return None

    torch.manual_seed(1)
    model = CausalTransformerV2(
        vocab_size=64,
        n_embd=32,
        n_head=4,
        n_kv_head=2,
        n_layer=2,
        block_size=64,
        mod_layers={1},
    ).eval()
    monkeypatch.setattr(
        generate,
        "_get_runtime",
        lambda: (model, _Tokenizer("the answer is 2*x + 3"), tmp_path / "x.pt"),
    )
    monkeypatch.setattr(
        generate, "_RUNTIME_LOAD_STATE", {"load_report": {"exact_native_load": True}}
    )
    monkeypatch.setattr(generate, "_get_hal", lambda _session_id: HalRecorder())
    monkeypatch.setattr(generate, "_attach_hal", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(generate, "_save_hal", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(generate, "_CIV_DIR", tmp_path / "civ")
    monkeypatch.setattr(generate, "_CIV_STORE", {})
    monkeypatch.setattr(generate, "_generation_quality", lambda *_a, **_k: 1.0)
    monkeypatch.setattr(generate, "_language_fragment_detected", lambda _text: False)

    generate.generate_traced(
        "H: Differentiate x^2 + 3*x\nANRA:",
        generate.GenerationConfig(
            max_tokens=2, mode="full_system", persist_adaptive_state=True
        ),
        session_id="symbolic_hal_probe",
    )
    # The symbolic pass verified the answer, so truthfulness is a real 1.0,
    # not None and not the coherence fallback.
    assert captured["verifier_result"] == 1.0
    assert captured["truthfulness"] == 1.0
