from __future__ import annotations

import torch

from anra_brain import CausalTransformerV2
from inference.exact_kv_cache import ExactStaticKVCache, analytical_copy_elements


def test_exact_static_cache_reuses_storage_and_preserves_recent_tokens() -> None:
    cache = ExactStaticKVCache(num_kv_heads=2, max_seq_len=4, d_head=8)
    first = torch.arange(48, dtype=torch.float32).reshape(1, 2, 3, 8)
    key, value = cache.update(first, first + 1)
    pointers = cache.storage_pointers

    second = torch.full((1, 2, 2, 8), 99.0)
    key, value = cache.update(second, second + 1)
    assert cache.storage_pointers == pointers
    assert key.shape == value.shape == (1, 2, 4, 8)
    assert torch.equal(key[:, :, -2:], second)
    assert torch.equal(value[:, :, -2:], second + 1)
    assert cache.position == 5
    assert cache.memory_report()["allocation_count"] == 1

    cache.reset()
    assert cache.storage_pointers == pointers
    assert cache.position == 0
    assert cache.current_len == 0


def test_default_exact_cache_matches_legacy_cache_logits_and_tokens() -> None:
    torch.manual_seed(1301)
    model = CausalTransformerV2(
        vocab_size=64,
        n_embd=32,
        n_head=4,
        n_kv_head=2,
        n_layer=2,
        block_size=32,
        mod_layers=(),
        use_rim=False,
        use_dstp=False,
        use_qk_norm=True,
        sliding_window=4,
        full_attention_every=0,
    ).eval()
    checkpoint_keys = tuple(model.state_dict())
    prompt = torch.tensor([[1, 7, 11, 13]], dtype=torch.long)

    def run(backend: str) -> tuple[list[torch.Tensor], list[int], dict[str, object]]:
        model.enable_kv_cache(backend=backend)
        model.clear_kv_cache()
        current = prompt
        distributions: list[torch.Tensor] = []
        tokens: list[int] = []
        try:
            with torch.no_grad():
                logits, _ = model(current)
                for _ in range(8):
                    step = logits[:, -1].clone()
                    distributions.append(step)
                    token = int(step.argmax(dim=-1).item())
                    tokens.append(token)
                    current = torch.tensor([[token]], dtype=torch.long)
                    logits, _ = model(current)
            report = model.kv_cache_telemetry()
        finally:
            model.disable_kv_cache()
        return distributions, tokens, report

    exact_logits, exact_tokens, exact_report = run("float")
    legacy_logits, legacy_tokens, _ = run("legacy-float")
    assert tuple(model.state_dict()) == checkpoint_keys
    assert exact_tokens == legacy_tokens
    assert exact_report["bytes_shifted"] == 0
    for exact, legacy in zip(exact_logits, legacy_logits, strict=True):
        torch.testing.assert_close(exact, legacy, rtol=1e-6, atol=1e-6)


def test_static_cache_changes_append_copy_growth_from_quadratic_to_linear() -> None:
    report = analytical_copy_elements(tokens=2048, elements_per_token=2 * 64)
    assert report["legacy_cat_elements"] == 537_133_056
    assert report["preallocated_elements"] == 524_288
    assert report["saved_elements"] == 536_608_768
