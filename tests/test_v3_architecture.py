from __future__ import annotations

from pathlib import Path

import torch
import pytest
import yaml

from anra.architecture import verify_canonical_counts
from anra_brain import CausalTransformerV3
from identity.esv import ESVModule
from training.v2_config import (
    ANRA_V4_MODEL,
    ANRA_V4_MODEL_PARAMETER_COUNT,
    CANONICAL_MODEL_PROFILE,
    model_parameter_count,
    resolve_model_profile,
)


def test_canonical_parameter_contract() -> None:
    counts = verify_canonical_counts()
    assert counts["frontier_full"] == 181_132_071
    assert set(counts) == {"frontier_transformer", "frontier_full"}


def test_canonical_v4_profile_has_exact_parameter_contract() -> None:
    assert model_parameter_count(ANRA_V4_MODEL) == ANRA_V4_MODEL_PARAMETER_COUNT
    assert resolve_model_profile(CANONICAL_MODEL_PROFILE)[0] is ANRA_V4_MODEL
    with pytest.raises(ValueError, match="Unknown model profile"):
        resolve_model_profile("pilot-150m")


def test_frontier_yaml_matches_iterate500_contract() -> None:
    config = yaml.safe_load(Path("config/anra_frontier.yaml").read_text(encoding="utf-8"))
    model = config["model"]
    training = config["training"]

    assert model["vocab_size"] == 32768
    assert model["n_embd"] == 896
    assert model["n_layer"] == 18
    assert model["n_head"] == 14
    assert model["n_kv_head"] == 2
    assert model["d_ff"] == 2432
    assert model["block_size"] == 2048
    assert model["mod_layers"] == [4, 6, 8, 10, 12, 14, 16]
    assert model["subsystem_policy"] == "explicit_trained_recipe_only"
    assert model["use_mod"] is False
    assert model["use_rim"] is False
    assert model["use_dstp"] is False
    assert training["seq_len"] == 2048
    assert training["batch_size"] == 1
    assert training["gradient_accumulation"] == 32


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
