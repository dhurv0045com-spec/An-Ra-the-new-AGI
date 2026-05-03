from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Iterable

from anra_paths import TOKENIZER_DIR, V3_TOKENIZER_FILE

SPECIAL_TOKENS = [
    "<unk>",
    "<pad>",
    "<bos>",
    "<eos>",
    "<mask>",
    "<sep>",
    "<cls>",
    "<system>",
    "<user>",
    "<assistant>",
    "<tool>",
    "<think>",
    "<code>",
]


class TokenizerAdapter:
    """SentencePiece-BPE adapter with a minimal surface API."""

    def __init__(self, processor, special_ids: dict[str, int]):
        self._sp = processor
        self._special_ids = dict(special_ids)

    @classmethod
    def train_from_texts(
        cls,
        texts: Iterable[str],
        *,
        vocab_size: int = 8192,
        output_json: str | Path = V3_TOKENIZER_FILE,
        output_model: str | Path = TOKENIZER_DIR / "tokenizer_v3.model",
    ) -> "TokenizerAdapter":
        import sentencepiece as spm

        output_json = Path(output_json)
        output_model = Path(output_model)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_model.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory() as td:
            corpus = Path(td) / "spm_corpus.txt"
            corpus.write_text("\n".join(texts), encoding="utf-8")
            model_prefix = str(Path(td) / "tokenizer_v3")
            spm.SentencePieceTrainer.train(
                input=str(corpus),
                model_prefix=model_prefix,
                model_type="bpe",
                vocab_size=vocab_size,
                unk_id=0,
                pad_id=1,
                bos_id=2,
                eos_id=3,
                unk_piece=SPECIAL_TOKENS[0],
                pad_piece=SPECIAL_TOKENS[1],
                bos_piece=SPECIAL_TOKENS[2],
                eos_piece=SPECIAL_TOKENS[3],
                user_defined_symbols=SPECIAL_TOKENS[4:],
                hard_vocab_limit=False,
                character_coverage=0.9995,
            )
            trained_model = Path(model_prefix + ".model")
            output_model.write_bytes(trained_model.read_bytes())

        adapter = cls.load(output_json, model_path=output_model)
        adapter.save(output_json, model_path=output_model)
        return adapter

    @classmethod
    def load(cls, json_path: str | Path, *, model_path: str | Path | None = None) -> "TokenizerAdapter":
        import sentencepiece as spm

        json_path = Path(json_path)
        meta = json.loads(json_path.read_text(encoding="utf-8")) if json_path.exists() else {}
        model = Path(model_path or meta.get("model_path") or json_path.with_suffix(".model"))
        sp = spm.SentencePieceProcessor(model_file=str(model))

        special_ids = {token: sp.piece_to_id(token) for token in SPECIAL_TOKENS}
        if any(idx < 0 for idx in special_ids.values()):
            raise ValueError("Tokenizer model is missing required special tokens")
        return cls(sp, special_ids)

    def save(self, json_path: str | Path, *, model_path: str | Path = TOKENIZER_DIR / "tokenizer_v3.model") -> None:
        json_path = Path(json_path)
        payload = {
            "version": 3,
            "model_type": "sentencepiece_bpe",
            "vocab_size": self.vocab_size(),
            "model_path": str(Path(model_path)),
            "special_ids": self.special_ids(),
        }
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def encode(self, text: str) -> list[int]:
        return list(self._sp.encode(text, out_type=int))

    def decode(self, ids: list[int]) -> str:
        return self._sp.decode(ids)

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return [self.encode(text) for text in texts]

    def vocab_size(self) -> int:
        return int(self._sp.vocab_size())

    def special_ids(self) -> dict[str, int]:
        return dict(self._special_ids)
