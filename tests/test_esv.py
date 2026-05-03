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
