from __future__ import annotations

import torch

from anra_brain import CausalTransformerV2
from evaluation.intelligence_telemetry import ANRAIntelligenceSession, subsystem_specs


def test_anra_subsystem_map_covers_core_model_parts() -> None:
    ids = {item.subsystem_id for item in subsystem_specs()}

    assert ids >= {
        "anra.embeddings",
        "anra.attention",
        "anra.mlp",
        "anra.esv",
        "anra.rim",
        "anra.mod",
        "anra.output",
    }


def test_anra_session_collects_deep_signals_without_changing_forward() -> None:
    torch.manual_seed(11)
    model = CausalTransformerV2(
        vocab_size=128,
        n_embd=32,
        n_head=4,
        n_kv_head=2,
        n_layer=2,
        block_size=8,
        mod_layers=(1,),
        use_hal=False,
    )
    tokens = torch.randint(0, 128, (1, 8))
    with torch.no_grad():
        expected, _ = model(tokens)

    session = ANRAIntelligenceSession(model, sample_every=1)
    session.begin_step(0)
    actual, loss = model(tokens, tokens)
    assert loss is not None
    loss.backward()
    session.record_optimizer_step(
        step=0,
        loss=float(loss.item()),
        learning_rate=1e-3,
        gradient_norm=1.0,
        tokens=tokens.numel(),
    )
    session.hooks.close()

    assert torch.equal(expected, actual)
    observed = {signal.subsystem_id for signal in session.monitor.collector.signals}
    assert observed >= {"anra.embeddings", "anra.attention", "anra.mlp", "anra.output"}
