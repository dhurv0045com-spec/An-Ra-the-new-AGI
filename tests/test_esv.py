from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from identity.esv import ESVModule


def test_esv_reads_residual_stream_and_exposes_vad_controls() -> None:
    esv = ESVModule(d_model=8, d_esv=4)
    h = torch.ones(2, 3, 8)
    state = esv(h)
    assert state.shape == (3,)
    assert set(esv.as_dict()) == {"valence", "arousal", "dominance"}
    assert esv.attention_temperature() > 0
    assert 0.0 < esv.memory_write_threshold() < 1.0
    ssm, att = esv.dgsa_gate()
    assert 0.0 < ssm < 1.0
    assert 0.0 < att < 1.0


def test_esv_predictor_zero_init():
    """ESV predictor must start with zero weights so initial state is neutral."""
    esv = ESVModule(d_model=512, d_esv=64)
    for m in esv.predictor.modules():
        if hasattr(m, "weight"):
            assert torch.all(m.weight == 0), (
                f"ESV predictor weights must be zero at init, got max={m.weight.abs().max():.6f}"
            )


def test_esv_initial_state_neutral():
    """ESV state must be (0,0,0) at initialization."""
    esv = ESVModule(d_model=512, d_esv=64)
    assert esv.state.sum().item() == 0.0
    assert esv.valence == 0.0
    assert esv.arousal == 0.0
    assert esv.dominance == 0.0


def test_esv_attention_temperature_neutral():
    """At neutral arousal (0), attention temperature equals tau0."""
    esv = ESVModule()
    assert abs(esv.attention_temperature(tau0=1.0) - 1.0) < 1e-6
