from __future__ import annotations

from pathlib import Path

from tokenizer.subword_tokenizer import SubwordTokenizer
from tokenizer.validate_tokenizer_v3 import validate_tokenizer
from scripts.train_tokenizer_v3 import SPECIAL_TOKENS


def test_tokenizer_v3_train_and_validate(tmp_path: Path) -> None:
    json_path = tmp_path / 'tokenizer_v3.json'
    texts = [
        'H: Hello\\nANRA: I can write Python code: def f(x): return x+1',
        '<system> Keep format <user> and <assistant> markers.',
        Path('training_data/anra_dataset_v6_1.txt').read_text(encoding='utf-8', errors='replace')[:20000],
    ]
    tok = SubwordTokenizer.train_from_texts(texts, vocab_size=8192, special_tokens=SPECIAL_TOKENS)
    tok.save(json_path)

    assert tok.vocab_size == 8192
    assert len(tok.token_to_id) == 8192
    assert max(tok.encode('def f(x): return x+1')) < 8192
    assert all(tok.special_ids[token] == i for i, token in enumerate(SPECIAL_TOKENS))

    stats = validate_tokenizer(json_path, Path('training_data/anra_dataset_v6_1.txt'))
    assert stats['roundtrip_ok']
    assert stats['unk_rate'] < 0.25
    assert stats['code_token_density'] >= 0.0
    assert stats['special_roundtrip_ok']
    assert stats['vocab_size_ok']
    assert stats['special_tokens_ok']
