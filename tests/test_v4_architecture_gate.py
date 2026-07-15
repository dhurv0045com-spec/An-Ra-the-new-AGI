from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from anra_brain import (
    ANRA_V4_ARCHITECTURE_VERSION,
    CausalTransformerV2,
    MultiHeadAttentionV2,
    RotaryEmbedding,
)
from training.v2_config import ANRA_V4_MODEL, ANRA_V4_TRAINING
from training.v2_runtime import CheckpointCompatibilityError, load_checkpoint
from training.preflight import HardwareProfile, run_preflight


def test_rope_uses_one_phase_per_adjacent_coordinate_pair() -> None:
    rope = RotaryEmbedding(dim=4, base_seq_len=16, target_seq_len=16)
    q = torch.tensor([[[[1.0, 0.0, 1.0, 0.0]]]])
    k = q.clone()

    rotated_q, rotated_k = rope(q, k, position_offset=1)
    phases = rope.inv_freq
    expected = torch.tensor(
        [[[[math.cos(float(phases[0])), math.sin(float(phases[0])),
            math.cos(float(phases[1])), math.sin(float(phases[1]))]]]]
    )

    torch.testing.assert_close(rotated_q, expected)
    torch.testing.assert_close(rotated_k, expected)
    torch.testing.assert_close(rotated_q.norm(dim=-1), q.norm(dim=-1))


def test_attention_temperature_is_bounded_after_controls_compose() -> None:
    torch.manual_seed(1301)
    attention = MultiHeadAttentionV2(32, 4, n_kv_head=2, use_qk_norm=True)
    attention.eval()
    x = torch.randn(2, 6, 32)

    upper = attention(x, attention_temperature=2.0)
    excessive = attention(x, attention_temperature=100.0)
    lower = attention(x, attention_temperature=0.5)
    tiny = attention(x, attention_temperature=0.001)

    torch.testing.assert_close(excessive, upper)
    torch.testing.assert_close(tiny, lower)


def test_native_controls_start_functionally_neutral() -> None:
    model = CausalTransformerV2(
        vocab_size=64,
        n_embd=32,
        n_head=4,
        n_kv_head=2,
        n_layer=4,
        block_size=16,
        mod_layers=(1, 3),
        use_qk_norm=True,
    )

    assert model.architecture_version == ANRA_V4_ARCHITECTURE_VERSION
    torch.testing.assert_close(model.dstp_temperature_log, torch.zeros(4))
    torch.testing.assert_close(model.layer_temperature_bias_log, torch.zeros(4))
    torch.testing.assert_close(model.residual_depth_logits, torch.zeros(4))
    assert all(float(module.raw_alpha.detach()) == 0.0 for module in model.rim_modules)


def test_t4_microbatch_preserves_large_effective_token_batch() -> None:
    assert ANRA_V4_TRAINING.batch_size == 1
    assert ANRA_V4_TRAINING.grad_accum_steps == 32
    assert (
        ANRA_V4_TRAINING.batch_size
        * ANRA_V4_TRAINING.grad_accum_steps
        * ANRA_V4_MODEL.block_size
        == 65_536
    )


def test_preflight_rejects_gpu_below_v4_memory_floor(monkeypatch) -> None:
    readiness = type("Readiness", (), {"blockers": [], "warnings": []})()
    monkeypatch.setattr("training.preflight.assess_training_readiness", lambda: readiness)
    undersized = HardwareProfile(
        "RTX 4050",
        True,
        6 * 1024**3,
        32 * 1024**3,
        100 * 1024**3,
        False,
    )

    decision = run_preflight(
        "anra-v4-180m",
        runtime_class="t4_v4_session",
        hardware=undersized,
    )

    assert not decision.allowed
    assert any("at least 14 GiB VRAM" in blocker for blocker in decision.blockers)


def test_canonical_v4_rejects_checkpoint_without_rotary_contract(
    tmp_path: Path,
) -> None:
    model = CausalTransformerV2(
        vocab_size=32_768,
        n_embd=32,
        n_head=4,
        n_kv_head=2,
        n_layer=2,
        block_size=16,
    )
    # Exercise the canonical gate without allocating the full 181M model.
    model.n_embd = 896
    model.n_layer = 18
    checkpoint = tmp_path / "pre-v4.pt"
    torch.save(
        {
            "checkpoint_schema_version": 8,
            "completed_optimizer_boundary": True,
            "accum_micro_steps": 0,
            "model": model.state_dict(),
            "model_config": {"vocab_size": 32_768},
        },
        checkpoint,
    )

    with pytest.raises(CheckpointCompatibilityError, match="architecture mismatch"):
        load_checkpoint(
            model,
            None,
            None,
            None,
            checkpoint,
            device=torch.device("cpu"),
        )


def test_canonical_v4_rejects_silent_immutable_config_change(tmp_path: Path) -> None:
    model = CausalTransformerV2(
        vocab_size=32_768,
        n_embd=32,
        n_head=4,
        n_kv_head=2,
        n_layer=2,
        block_size=16,
    )
    model.n_embd = 896
    model.n_layer = 18
    saved_config = model.model_config()
    saved_config["rope_base"] = 500_000
    checkpoint = tmp_path / "mutated-v4.pt"
    torch.save(
        {
            "checkpoint_schema_version": 8,
            "completed_optimizer_boundary": True,
            "accum_micro_steps": 0,
            "model": model.state_dict(),
            "model_config": saved_config,
        },
        checkpoint,
    )

    with pytest.raises(CheckpointCompatibilityError, match="immutable architecture"):
        load_checkpoint(
            model,
            None,
            None,
            None,
            checkpoint,
            device=torch.device("cpu"),
        )
