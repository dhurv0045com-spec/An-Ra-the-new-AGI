from __future__ import annotations

from pathlib import Path

import pytest

from tokenizer.tokenizer_adapter import SPECIAL_TOKENS, TokenizerAdapter
from tokenizer.validate_tokenizer_v3 import validate_tokenizer


def test_tokenizer_v3_train_and_validate(tmp_path: Path) -> None:
    pytest.importorskip('sentencepiece')
    json_path = tmp_path / 'tokenizer_v3.json'
    model_path = tmp_path / 'tokenizer_v3.model'

    texts = [
        'H: Hello\\nANRA: I can write Python code: def f(x): return x+1',
        '<system> Keep format <user> and <assistant> markers.',
    ]
    tok = TokenizerAdapter.train_from_texts(texts, vocab_size=8192, output_json=json_path, output_model=model_path)

    assert tok.vocab_size() > 0
    assert all(tok.special_ids()[token] == i for i, token in enumerate(SPECIAL_TOKENS))

    stats = validate_tokenizer(json_path, Path('training_data/anra_dataset_v6_1.txt'))
    assert stats['roundtrip_ok']
    assert stats['unk_rate'] < 0.25
    assert stats['code_token_density'] >= 0.0
    assert stats['special_roundtrip_ok']
