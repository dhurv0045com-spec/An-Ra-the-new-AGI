from __future__ import annotations

from pathlib import Path

from tokenizer.tokenizer_adapter import SPECIAL_TOKENS, TokenizerAdapter


def validate_tokenizer(tokenizer_json: Path, dataset_path: Path) -> dict[str, float | bool]:
    tok = TokenizerAdapter.load(tokenizer_json, model_path=tokenizer_json.with_suffix('.model'))
    text = dataset_path.read_text(encoding='utf-8', errors='replace')[:20000]

    ids = tok.encode(text)
    roundtrip_ok = tok.decode(ids)[:200] == text[:200]
    unk_rate = (sum(1 for i in ids if i == tok.special_ids()["<unk>"]) / max(1, len(ids)))

    code_chars = sum(1 for c in text if c in "{}[]()=;:_<>/\\")
    token_density = len(ids) / max(1, len(text))

    special_ok = all(tok.decode([idx]) == piece for piece, idx in tok.special_ids().items() if piece in SPECIAL_TOKENS)
    return {
        "roundtrip_ok": roundtrip_ok,
        "unk_rate": float(unk_rate),
        "code_token_density": float(token_density * code_chars / max(1, len(text))),
        "special_roundtrip_ok": special_ok,
    }


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    results = validate_tokenizer(root / 'tokenizer' / 'tokenizer_v3.json', root / 'training_data' / 'anra_dataset_v6_1.txt')
    print(results)
