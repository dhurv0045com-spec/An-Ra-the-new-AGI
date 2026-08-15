from __future__ import annotations

import torch

from anra_brain import CausalTransformerV2, RotaryEmbedding
from identity.hal import HALModule
from runtime.hal_telemetry import publish_hal_state
from training.v2_runtime import ensure_tied_lm_head, hal_state_dict, restore_hal_state
from training.tpu_runtime import freeze_parametrized_spectral_norms_for_xla


def test_rope_cache_rebuilds_when_cached_device_differs() -> None:
    rope = RotaryEmbedding(dim=8)
    rope._cached_seq_len = 16
    rope._cached_cos = torch.empty(1, 1, 16, 8, device="meta")
    rope._cached_sin = torch.empty(1, 1, 16, 8, device="meta")

    rope._build_cache(8, torch.device("cpu"), torch.float32)

    assert rope._cached_cos is not None
    assert rope._cached_sin is not None
    assert rope._cached_cos.device.type == "cpu"
    assert rope._cached_sin.device.type == "cpu"


def test_ensure_tied_lm_head_repairs_weight_alias() -> None:
    model = CausalTransformerV2(
        vocab_size=64,
        n_embd=32,
        n_head=4,
        n_kv_head=2,
        n_layer=2,
        block_size=16,
        use_hal=False,
    )
    model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.detach().clone())
    assert model.lm_head.weight is not model.token_embedding_table.weight

    assert ensure_tied_lm_head(model) is True
    assert model.lm_head.weight is model.token_embedding_table.weight


def test_hal_state_publishes_and_round_trips_checkpoint_payload(tmp_path) -> None:
    hal = HALModule()
    before = hal.state.dopamine
    hal.update(verifier_result=0.8, civ_score=0.72)
    assert hal.state.dopamine > before

    report = publish_hal_state(hal, source="test", path=tmp_path / "hal_state.json")
    assert report["hormones"]["dopamine"] == hal.state.dopamine

    model = CausalTransformerV2(
        vocab_size=64,
        n_embd=32,
        n_head=4,
        n_kv_head=2,
        n_layer=2,
        block_size=16,
        use_hal=True,
        hal_module=hal,
    )
    payload = hal_state_dict(model)
    model.hal_module.state.dopamine = 0.0
    assert restore_hal_state(model, payload) is True
    assert model.hal_module.state.dopamine == payload["dopamine"]


def test_freeze_parametrized_spectral_norms_materializes_rim_weights() -> None:
    model = CausalTransformerV2(
        vocab_size=64,
        n_embd=32,
        n_head=4,
        n_kv_head=2,
        n_layer=2,
        block_size=16,
        use_hal=False,
    )
    before = [name for name, module in model.named_modules() if hasattr(module, "parametrizations")]
    assert any("rim_modules" in name for name in before)

    frozen = freeze_parametrized_spectral_norms_for_xla(model)

    assert any("rim_modules" in name for name in frozen)
    after = [name for name, module in model.named_modules() if hasattr(module, "parametrizations")]
    assert not any("rim_modules" in name for name in after)
