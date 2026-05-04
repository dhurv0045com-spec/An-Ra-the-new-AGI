from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tokenizer.subword_tokenizer import SubwordTokenizer


SPECIAL_TOKENS = [
    "<pad>",
    "<unk>",
    "<bos>",
    "<eos>",
    "<sep>",
    "<code>",
    "</code>",
    "<think>",
    "</think>",
    "<goal>",
    "<ESV:v>",
    "<ESV:a>",
    "<ESV:d>",
]
TARGET_VOCAB = 8192


def validate_tokenizer(tokenizer_json: Path, dataset_path: Path) -> dict[str, float | bool]:
    tok = SubwordTokenizer.load(tokenizer_json)
    if len(tok.token_to_id) != TARGET_VOCAB:
        raise AssertionError(f"[tokenizer_v3] expected {TARGET_VOCAB} tokens, found {len(tok.token_to_id)}")
    for expected_id, token in enumerate(SPECIAL_TOKENS):
        actual = tok.token_to_id.get(token)
        if actual != expected_id:
            raise AssertionError(f"[tokenizer_v3] special token {token!r} has id {actual}, expected {expected_id}")

    text = dataset_path.read_text(encoding='utf-8', errors='replace')[:20000]

    ids = tok.encode(text)
    roundtrip_ok = tok.decode(ids)[:200] == text[:200]
    unk_rate = (sum(1 for i in ids if i == tok.special_ids["<unk>"]) / max(1, len(ids)))

    code_chars = sum(1 for c in text if c in "{}[]()=;:_<>/\\")
    token_density = len(ids) / max(1, len(text))

    special_ok = all(tok.decode([idx]) == piece for piece, idx in tok.special_ids.items() if piece in SPECIAL_TOKENS)
    return {
        "roundtrip_ok": roundtrip_ok,
        "unk_rate": float(unk_rate),
        "code_token_density": float(token_density * code_chars / max(1, len(text))),
        "special_roundtrip_ok": special_ok,
        "vocab_size_ok": True,
        "special_tokens_ok": True,
    }


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    results = validate_tokenizer(root / 'tokenizer' / 'tokenizer_v3.json', root / 'training_data' / 'anra_training.txt')
    print(results)
