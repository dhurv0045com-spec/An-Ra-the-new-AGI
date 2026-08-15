from __future__ import annotations

from pathlib import Path

import pytest
import torch

import generate
from anra_brain import CausalTransformerV2
from inference.turboquant import (
    TorchTurboQuantCache,
    TurboQuantConfig,
    health_check,
)


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


def _tiny_model() -> CausalTransformerV2:
    torch.manual_seed(1301)
    return CausalTransformerV2(
        vocab_size=64,
        n_embd=32,
        n_head=4,
        n_kv_head=2,
        n_layer=2,
        block_size=128,
        mod_layers={1},
    ).eval()


def test_turboquant_physically_packs_four_bit_codes_and_reports_distortion() -> None:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(7)
    key = torch.randn(1, 2, 24, 64, generator=generator, dtype=torch.float16)
    value = torch.randn(1, 2, 24, 64, generator=generator, dtype=torch.float16)
    cache = TorchTurboQuantCache(
        num_kv_heads=2,
        max_seq_len=32,
        d_head=64,
        config=TurboQuantConfig(bits=4),
    )

    restored_key, restored_value = cache.update(key, value)
    report = cache.memory_report()

    assert restored_key.shape == key.shape
    assert restored_value.shape == value.shape
    assert cache.packed_width == 32
    assert report["paper_complete"] is False
    assert report["qjl_fused"] is False
    assert report["compression_ratio"] == pytest.approx(256 / 68)
    assert report["last_relative_mse"] < 0.08
    assert report["compressed_bytes"] < report["uncompressed_bytes"]
    assert report["occupied_compressed_bytes"] < report["compressed_bytes"]


def test_turboquant_cache_retains_only_bounded_recent_history() -> None:
    cache = TorchTurboQuantCache(
        num_kv_heads=1,
        max_seq_len=4,
        d_head=64,
    )
    first = torch.randn(1, 1, 3, 64)
    second = torch.randn(1, 1, 3, 64)

    cache.update(first, first)
    key, value = cache.update(second, second)

    assert key.shape == value.shape == (1, 1, 4, 64)
    assert cache.current_len == 4
    assert cache.position == 6
    cache.reset()
    assert cache.current_len == 0
    assert cache.position == 0

    oversized = torch.randn(1, 1, 7, 64)
    key, _value = cache.update(oversized, oversized)
    assert key.shape == (1, 1, 4, 64)
    assert cache.current_len == 4
    assert cache.position == 7


def test_real_transformer_uses_compressed_backend_and_exposes_physical_bytes() -> None:
    model = _tiny_model()
    model.enable_kv_cache(backend="turboquant", turboquant_bits=4)
    try:
        model(torch.tensor([[1, 2, 3, 4]], dtype=torch.long))
        model(torch.tensor([[5]], dtype=torch.long))
        report = model.kv_cache_telemetry()
    finally:
        model.clear_kv_cache()
        model.disable_kv_cache()

    assert report["backend"] == "turboquant"
    assert report["layers"] == 2
    assert report["compressed_bytes"] < report["uncompressed_bytes"]
    assert report["compression_ratio"] >= 3.0
    assert report["max_relative_mse"] < 0.08


def test_turboquant_generation_is_fail_closed_until_its_own_gate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model = _tiny_model()
    monkeypatch.setattr(
        generate,
        "_get_runtime",
        lambda: (model, _Tokenizer(), tmp_path / "checkpoint.pt"),
    )
    monkeypatch.setattr(
        generate,
        "_RUNTIME_LOAD_STATE",
        {"load_report": {"exact_native_load": True}},
    )
    monkeypatch.setattr(generate, "_get_hal", lambda _session_id: None)
    monkeypatch.setattr(generate, "_TURBOQUANT_VERIFIED_BITS", set())
    monkeypatch.setattr(generate, "_TURBOQUANT_BITS_IN_PROGRESS", set())

    with pytest.raises(RuntimeError, match="TurboQuant"):
        generate.generate_traced(
            "H: pilot\nANRA:",
            generate.GenerationConfig(
                max_tokens=4,
                use_kv_cache=True,
                kv_cache_backend="turboquant",
            ),
        )

    report = generate.verify_turboquant_cache(
        "H: pilot\nANRA:",
        max_tokens=6,
        max_distribution_delta=0.1,
    )

    assert report["backend"] == "turboquant"
    assert report["compression_ratio"] >= 3.0
    assert report["max_relative_mse"] < 0.08
    assert isinstance(report["verified"], bool)
    assert health_check()["status"] == "ok"


def test_turboquant_gate_is_scoped_to_the_verified_precision(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model = _tiny_model()
    monkeypatch.setattr(
        generate,
        "_get_runtime",
        lambda: (model, _Tokenizer(), tmp_path / "checkpoint.pt"),
    )
    monkeypatch.setattr(
        generate,
        "_RUNTIME_LOAD_STATE",
        {"load_report": {"exact_native_load": True}},
    )
    monkeypatch.setattr(generate, "_get_hal", lambda _session_id: None)
    monkeypatch.setattr(generate, "_TURBOQUANT_VERIFIED_BITS", {8})
    monkeypatch.setattr(generate, "_TURBOQUANT_BITS_IN_PROGRESS", set())

    with pytest.raises(RuntimeError, match="4-bit"):
        generate.generate_traced(
            "H: pilot\nANRA:",
            generate.GenerationConfig(
                max_tokens=2,
                use_kv_cache=True,
                kv_cache_backend="turboquant",
                turboquant_bits=4,
            ),
        )
