from __future__ import annotations

import torch

from anra.architecture import verify_canonical_counts
from anra_brain import CausalTransformerV3
from identity.esv import ESVModule


def test_canonical_parameter_contract() -> None:
    counts = verify_canonical_counts()
    assert counts["anra_3b_full"] == 2_925_174_103
    assert counts["draft_full"] == 8_004_291


def test_esv_forward_is_pure_until_commit() -> None:
    esv = ESVModule(d_model=8, d_esv=4)
    with torch.no_grad():
        esv.predictor[0].weight.fill_(0.1)
    predicted = esv(torch.ones(2, 3, 8))
    assert esv.state.tolist() == [0.0, 0.0, 0.0]
    esv.commit_state(predicted)
    torch.testing.assert_close(esv.state, predicted)


def test_v3_attention_bound_and_sparse_router() -> None:
    model = CausalTransformerV3(
        vocab_size=64,
        n_embd=32,
        n_head=4,
        n_kv_head=2,
        n_layer=2,
        block_size=16,
        mod_layers=(1,),
    )
    assert model.blocks[0].attn.lba_bound > 0.0
    x = torch.randint(0, 64, (1, 12))
    logits, loss = model(x, x)
    assert logits.shape == (1, 12, 64)
    assert loss is not None
    loss.backward()
    assert model.mod_routers["1"].gate.weight.grad is not None
