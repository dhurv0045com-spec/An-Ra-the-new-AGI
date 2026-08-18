from pathlib import Path

import pytest

from anra_core.config import CANONICAL_CONFIG
from anra_core.brain import ThoughtPolicy, _has_invalid_terminal
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


def test_brain_only_penalizes_the_actual_unknown_terminal() -> None:
    assert not _has_invalid_terminal("A normal completed response.")
    assert _has_invalid_terminal("A malformed response <unk>")


def test_tokenizer_contract_and_roundtrip() -> None:
    tokenizer = V4Tokenizer.load(TOKENIZER)
    assert len(tokenizer.id_to_token) == 32_768
    assert [tokenizer.token_to_id[token] for token in tokenizer.special_tokens] == [
        *range(13), *range(8_192, 8_209)
    ]
    text = "An-Ra learns. नमस्ते"
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_tokenizer_golden_vectors_guard_v4_representation() -> None:
    tokenizer = V4Tokenizer.load(TOKENIZER)
    cases = [
        ("Hello, An-Ra!", [4140, 16, 13, 88, 33, 112, 1225]),
        (" spaces\tand\n punctuation... ", [13, 4261, 7088, 26, 29, 13, 420, 15387, 227, 270, 1147, 62, 11540, 14, 14, 14, 13]),
        ("<bos><eos>", [2, 3]),
        ("café 東京", [27783, 8404, 8378, 13, 8439, 8366, 8386, 8437, 8395, 8381]),
        ("नमस्ते दुनिया", [8433, 8373, 8377, 8433, 8373, 8383, 8433, 8373, 8393, 8433, 8374, 8350, 8433, 8373, 8373, 8433, 8374, 8344, 13, 8433, 8373, 8375, 8433, 8374, 8338, 8433, 8373, 8377, 8433, 8373, 8400, 8433, 8373, 8384, 8433, 8373, 8399]),
        ("🧠🚀", [8449, 8368, 8376, 8369, 8449, 8368, 8363, 8337]),
    ]
    for text, expected_ids in cases:
        assert tokenizer.encode(text) == expected_ids
        assert tokenizer.decode(expected_ids) == text


def test_model_contract_when_torch_is_available() -> None:
    torch = pytest.importorskip("torch")
    from anra_core.model import AnRaCore

    model = AnRaCore()
    assert model.lm_head.weight.data_ptr() == model.token_embedding_table.weight.data_ptr()
    assert sum(parameter.numel() for parameter in model.parameters()) == 180_093_312
    with pytest.raises(ValueError, match="sequence exceeds"):
        model(torch.zeros((1, 2_049), dtype=torch.long))
