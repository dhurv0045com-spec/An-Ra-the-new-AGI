from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from identity.esv import ESVModule
from training.v2_data_mix import TrainingExample, V2ConversationDataset


def test_esv_reads_residual_stream_and_exposes_vad_controls() -> None:
    esv = ESVModule(d_model=8, d_esv=4)
    h = torch.ones(2, 3, 8)
    state = esv(h)
    assert state.shape == (2, 3)
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


def test_esv_attention_temperature_is_per_sample() -> None:
    esv = ESVModule(d_model=8, d_esv=4)
    state = torch.tensor([[0.0, -1.0, 0.0], [0.0, 1.0, 0.0]])
    temperature = esv.attention_temperature_tensor(state)

    assert temperature.shape == (2, 1, 1, 1)
    assert temperature[0].item() != temperature[1].item()


def test_esv_temporal_consistency_is_differentiable_per_sample() -> None:
    esv = ESVModule(d_model=8, d_esv=4)
    with torch.no_grad():
        esv.predictor[0].weight.normal_(mean=0.0, std=0.1)
    residual = torch.randn(2, 5, 8, requires_grad=True)
    prediction = esv(residual)
    temporal_loss = esv.temporal_consistency_loss()

    assert prediction.shape == (2, 3)
    assert temporal_loss.shape == ()
    assert temporal_loss.item() > 0.0
    temporal_loss.backward()
    assert residual.grad is not None


def test_verified_esv_targets_reject_unverified_or_invalid_labels() -> None:
    dataset = object.__new__(V2ConversationDataset)
    dataset.examples = [
        TrainingExample(
            bucket="teacher",
            prompt="verified",
            answer="answer",
            source="test",
            metadata={
                "verifier_status": "verified",
                "vad": {"valence": 0.5, "arousal": -0.25, "dominance": 0.75},
            },
        ),
        TrainingExample(
            bucket="teacher",
            prompt="unverified",
            answer="answer",
            source="test",
            metadata={"vad": [0.1, 0.2, 0.3]},
        ),
        TrainingExample(
            bucket="teacher",
            prompt="out of range",
            answer="answer",
            source="test",
            metadata={"verified": True, "vad": [2.0, 0.0, 0.0]},
        ),
    ]

    targets, mask = dataset.verified_esv_targets(
        [0, 1, 2],
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    torch.testing.assert_close(targets[0], torch.tensor([0.5, -0.25, 0.75]))
    assert mask.tolist() == [True, False, False]
    assert torch.count_nonzero(targets[1:]).item() == 0
