"""KV-cache parity gate: cached generation must earn its enablement.

The repo contract is "KV cache remains disabled until cached and uncached
token parity is demonstrated". These tests demonstrate it on a real model,
prove the gate unlocks cached generation afterward, and — equally important —
prove the gate FAILS when the cache is genuinely broken. A gate that cannot
detect a fault is not evidence.
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

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [3 + (ord(character) % 50) for character in text]

    def decode(self, ids: list[int]) -> str:
        return " ".join(str(token) for token in ids)


@pytest.fixture
def runtime(monkeypatch, tmp_path: Path) -> CausalTransformerV2:
    torch.manual_seed(7)
    model = CausalTransformerV2(
        vocab_size=64,
        n_embd=32,
        n_head=4,
        n_kv_head=2,
        n_layer=2,
        block_size=128,
        mod_layers={1},
    ).eval()
    monkeypatch.setattr(
        generate, "_get_runtime", lambda: (model, _Tokenizer(), tmp_path / "x.pt")
    )
    monkeypatch.setattr(
        generate, "_RUNTIME_LOAD_STATE", {"load_report": {"exact_native_load": True}}
    )
    monkeypatch.setattr(generate, "_get_hal", lambda _session_id: None)
    monkeypatch.setattr(generate, "_KV_CACHE_PARITY_VERIFIED", False)
    monkeypatch.setattr(generate, "_KV_CACHE_PARITY_IN_PROGRESS", False)
    return model


def test_kv_cache_blocked_until_parity_proven(runtime) -> None:
    with pytest.raises(RuntimeError, match="parity"):
        generate.generate_traced(
            "H: probe\nANRA:",
            generate.GenerationConfig(
                strategy="greedy", max_tokens=4, seed=0, use_kv_cache=True, mode="diagnostic"
            ),
            session_id="kv_blocked_probe",
        )


def test_parity_gate_verifies_and_unlocks_cached_generation(runtime) -> None:
    report = generate.verify_kv_cache_parity(max_tokens=24)
    assert report["verified"] is True
    assert report["tokens_compared"] > 0
    assert report["uncached_tokens"] == report["cached_tokens"]

    # The proven gate must now admit cached generation, and cached output must
    # still replay the uncached tokens exactly.
    trace = generate.generate_traced(
        "H: a different prompt entirely\nANRA:",
        generate.GenerationConfig(
            strategy="greedy", max_tokens=12, seed=0, use_kv_cache=True, mode="diagnostic"
        ),
        session_id="kv_unlocked_probe",
    )
    assert trace.kv_cache_compressed is True
    baseline = generate.generate_traced(
        "H: a different prompt entirely\nANRA:",
        generate.GenerationConfig(
            strategy="greedy", max_tokens=12, seed=0, use_kv_cache=False, mode="diagnostic"
        ),
        session_id="kv_unlocked_probe",
    )
    assert trace.output_token_ids == baseline.output_token_ids


def test_parity_gate_detects_a_genuinely_broken_cache(runtime) -> None:
    # The realistic cache fault is stale content leaking across requests, not
    # a uniform position shift (RoPE attention is relative, so shifting every
    # position equally leaves outputs unchanged — verified the hard way). On
    # a random tiny model a near-uniform distribution dilutes small faults,
    # so the injection uses a 16-token stale prefix against a short probe
    # prompt; measured step-0 entropy divergence is then well above the
    # gate's 1e-3 tolerance while greedy tokens still coincide.
    model = runtime
    real_clear = model.clear_kv_cache
    stale_prefix = [list(range(5, 61, 3))[:16]]

    def stale_clear() -> None:
        real_clear()
        with torch.no_grad():
            model(torch.tensor(stale_prefix, dtype=torch.long))

    model.clear_kv_cache = stale_clear
    try:
        report = generate.verify_kv_cache_parity("H: probe\nANRA:", max_tokens=8)
    finally:
        model.clear_kv_cache = real_clear
    # Greedy tokens survive the poison; only the distribution check can catch
    # it. If this verifies, the gate is decorative.
    assert report["token_parity"] is True
    assert report["distribution_parity"] is False
    assert report["verified"] is False
