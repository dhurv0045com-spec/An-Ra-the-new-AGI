from __future__ import annotations

from pathlib import Path

import torch

from anra_brain import CausalTransformerV2
from anra.anra_paths import DATASET_CANONICAL
from scripts.train_tokenizer_v3 import SPECIAL_TOKENS
from tokenizer.subword_tokenizer import SubwordTokenizer
from tokenizer.validate_tokenizer_v3 import validate_tokenizer
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
