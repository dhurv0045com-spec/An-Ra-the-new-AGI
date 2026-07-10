from __future__ import annotations

from pathlib import Path

import torch
import yaml

from anra.architecture import verify_canonical_counts
from anra_brain import CausalTransformerV3
from identity.esv import ESVModule


def test_canonical_parameter_contract() -> None:
    counts = verify_canonical_counts()
    assert counts["frontier_full"] == 499_167_075
    assert counts["draft_full"] == 8_004_291


def test_frontier_yaml_matches_iterate500_contract() -> None:
    config = yaml.safe_load(Path("config/anra_frontier.yaml").read_text(encoding="utf-8"))
    model = config["model"]
    training = config["training"]

    assert model["n_embd"] == 1280
    assert model["n_layer"] == 28
    assert model["n_head"] == 16
    assert model["n_kv_head"] == 4
    assert model["d_ff"] == 3456
    assert model["block_size"] == 1024
    assert model["mod_layers"] == [4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]
    assert training["seq_len"] == 1024
    assert training["gradient_accumulation"] == 16


def test_esv_forward_is_pure_until_commit() -> None:
    esv = ESVModule(d_model=8, d_esv=4)
    with torch.no_grad():
        esv.predictor[0].weight.fill_(0.1)
    predicted = esv(torch.ones(2, 3, 8))
    assert esv.state.tolist() == [0.0, 0.0, 0.0]
    esv.commit_state(predicted)
    torch.testing.assert_close(esv.state, predicted.mean(dim=0))


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
