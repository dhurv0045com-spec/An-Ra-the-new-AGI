from __future__ import annotations

from pathlib import Path

import pytest
import torch

from anra_brain import CausalTransformerV2
from anra.anra_paths import DATASET_CANONICAL
from scripts.train_tokenizer_v3 import SPECIAL_TOKENS
from tokenizer.subword_tokenizer import SubwordTokenizer
from tokenizer.validate_tokenizer_v3 import build_append_only_v4, validate_tokenizer
from training.v2_runtime import generate_text, tokenizer_special_ids


def test_tokenizer_v3_train_and_validate(tmp_path: Path) -> None:
    json_path = tmp_path / 'tokenizer_v3.json'
    texts = [
        'H: Hello\\nANRA: I can write Python code: def f(x): return x+1',
        '<system> Keep format <user> and <assistant> markers.',
        DATASET_CANONICAL.read_text(encoding='utf-8', errors='replace')[:20000],
    ]
    tok = SubwordTokenizer.train_from_texts(texts, vocab_size=8209, special_tokens=SPECIAL_TOKENS)
    tok.save(json_path)

    assert tok.vocab_size == 8209
    assert tok.backend == "hf"
    assert tok.model_type == "bpe"
    assert len(tok.token_to_id) == 8209
    assert max(tok.encode('def f(x): return x+1')) < 8209
    assert tok.special_ids["<state>"] == 8192

    stats = validate_tokenizer(json_path, DATASET_CANONICAL)
    assert stats['roundtrip_ok']
    assert stats['unk_rate'] < 0.25
    assert stats['code_token_density'] >= 0.0
    assert stats['special_roundtrip_ok']
    assert stats['vocab_size_ok']
    assert stats['special_tokens_ok']


def test_tokenizer_training_refuses_silent_fallback(monkeypatch) -> None:
    monkeypatch.setattr(SubwordTokenizer, "_try_import_tokenizers", staticmethod(lambda: None))

    with pytest.raises(RuntimeError, match="refusing to train a fallback vocab"):
        SubwordTokenizer.train_from_texts(["native language data"], vocab_size=64)


def test_append_only_v4_preserves_ids_and_is_byte_safe(tmp_path: Path) -> None:
    base = DATASET_CANONICAL.parents[1] / "tokenizer" / "tokenizer_v3.json"
    output = tmp_path / "tokenizer_v4.json"
    audit = {
        "eligible_for_schema_v4": True,
        "candidate_tokens": ["phosphorylation", "oxidative", "nutrients"],
        "sampled_units": 1_000_000,
        "projected_reduction": 0.20,
    }

    build_append_only_v4(base, output, audit)
    old = SubwordTokenizer.load(base)
    new = SubwordTokenizer.load(output)

    assert new.vocab_size == 16_384
    assert new.backend == "native_append_v4"
    for token, token_id in old.special_ids.items():
        assert new.token_to_id.get(token) == token_id
    assert len(new.encode("phosphorylation")) == 1
    probe = "native bytes: cafe \u03bb \u2211 \U0001f680"
    assert new.decode(new.encode(probe)) == probe


def test_runtime_accepts_dict_special_ids_surface() -> None:
    class DictTokenizer:
        special_ids = {"<bos>": 2, "<eos>": 3}

    assert tokenizer_special_ids(DictTokenizer()) == {"<bos>": 2, "<eos>": 3}


def test_runtime_accepts_callable_special_ids_surface() -> None:
    class MethodTokenizer:
        def special_ids(self) -> dict[str, int]:
            return {"<bos>": 2, "<eos>": 3}

    assert tokenizer_special_ids(MethodTokenizer()) == {"<bos>": 2, "<eos>": 3}


def test_generate_text_accepts_dict_special_ids_surface() -> None:
    class DictTokenizer:
        special_ids = {"<bos>": 2, "<eos>": 3}

        def encode(self, text: str) -> list[int]:
            return [4, 5][: max(1, len(text.split()))]

        def decode(self, ids: list[int]) -> str:
            return " ".join(str(token) for token in ids)

    torch.manual_seed(1)
    model = CausalTransformerV2(
        vocab_size=16,
        n_embd=16,
        n_head=2,
        n_kv_head=1,
        n_layer=1,
        block_size=8,
        use_hal=False,
    )
    text = generate_text(
        model,
        DictTokenizer(),
        "hello",
        device=torch.device("cpu"),
        max_new_tokens=2,
        top_k=4,
    )

    assert isinstance(text, str)
