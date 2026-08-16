from pathlib import Path

import pytest

from anra_core.config import CANONICAL_CONFIG
from anra_core.brain import ThoughtPolicy
from anra_core.tokenizer import V4Tokenizer


TOKENIZER = Path(__file__).parents[1] / "anra_core" / "assets" / "tokenizer_v4_32k.json"


def test_canonical_configuration() -> None:
    config = CANONICAL_CONFIG
    assert config.dense_parameter_count == 180_093_312
    assert (config.vocab_size, config.d_model, config.n_layers) == (32_768, 896, 18)
    assert (config.n_heads, config.n_kv_heads, config.head_dim) == (14, 2, 64)
    assert config.block_size == 2_048


def test_thought_policy_is_bounded() -> None:
    assert ThoughtPolicy(mode="deliberate", candidates=4).candidates == 4
    with pytest.raises(ValueError, match="between 1 and 4"):
        ThoughtPolicy(mode="deliberate", candidates=5)
    with pytest.raises(ValueError, match="exactly one"):
        ThoughtPolicy(mode="direct", candidates=2)


def test_tokenizer_contract_and_roundtrip() -> None:
    tokenizer = V4Tokenizer.load(TOKENIZER)
    assert len(tokenizer.id_to_token) == 32_768
    assert [tokenizer.token_to_id[token] for token in tokenizer.special_tokens] == [
        *range(13), *range(8_192, 8_209)
    ]
    text = "An-Ra learns. नमस्ते"
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_model_contract_when_torch_is_available() -> None:
    torch = pytest.importorskip("torch")
    from anra_core.model import AnRaCore

    model = AnRaCore()
    assert model.lm_head.weight.data_ptr() == model.token_embedding_table.weight.data_ptr()
    assert sum(parameter.numel() for parameter in model.parameters()) == 180_093_312
    with pytest.raises(ValueError, match="sequence exceeds"):
        model(torch.zeros((1, 2_049), dtype=torch.long))
