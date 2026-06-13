from __future__ import annotations

from pathlib import Path

from anra.anra_paths import DATASET_CANONICAL
from scripts.train_tokenizer_v3 import SPECIAL_TOKENS
from tokenizer.subword_tokenizer import SubwordTokenizer
from tokenizer.validate_tokenizer_v3 import validate_tokenizer


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
