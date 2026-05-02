from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

_TOKEN_PATTERN = re.compile(r"\s+|[A-Za-z0-9_]+|[^\w\s]", re.UNICODE)


class SubwordTokenizer:
    def __init__(self, payload: dict):
        self.token_to_id = payload["token_to_id"]
        self.id_to_token = payload["id_to_token"]
        self.vocab_size = len(self.id_to_token)
        self.special_tokens = payload.get("special_tokens", [])
        self.special_ids = {t: self.token_to_id[t] for t in self.special_tokens if t in self.token_to_id}

    @classmethod
    def train_from_texts(
        cls,
        texts: list[str],
        vocab_size: int = 8192,
        special_tokens: list[str] | None = None,
    ) -> "SubwordTokenizer":
        special_tokens = special_tokens or [
            "<unk>", "<pad>", "<bos>", "<eos>", "<sep>", "<code>", "</code>",
            "<think>", "</think>", "<goal>", "<ESV:v>", "<ESV:a>", "<ESV:d>",
        ]
        counter: Counter[str] = Counter()
        chars: set[str] = set()
        for text in texts:
            counter.update(_TOKEN_PATTERN.findall(text))
            chars.update(text)

        ordered = list(special_tokens)
        room = max(0, vocab_size - len(ordered))
        for tok, _ in counter.most_common(room * 2):
            if tok not in ordered:
                ordered.append(tok)
            if len(ordered) >= vocab_size:
                break
        for ch in sorted(chars):
            if len(ordered) >= vocab_size:
                break
            if ch not in ordered:
                ordered.append(ch)
        while len(ordered) < vocab_size:
            ordered.append(f"<extra_{len(ordered)}>")

        payload = {
            "token_to_id": {t: i for i, t in enumerate(ordered)},
            "id_to_token": ordered,
            "special_tokens": special_tokens,
            "model_type": "bpe",
            "backend": "subword",
        }
        return cls(payload)

    def encode(self, text: str) -> list[int]:
        unk = self.token_to_id.get("<unk>", 0)
        out: list[int] = []
        for p in _TOKEN_PATTERN.findall(text):
            out.append(self.token_to_id.get(p, unk))
        return out

    def decode(self, ids: list[int]) -> str:
        return "".join(self.id_to_token[i] for i in ids if 0 <= i < len(self.id_to_token))

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "token_to_id": self.token_to_id,
            "id_to_token": self.id_to_token,
            "special_tokens": self.special_tokens,
        }
        p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        meta = {
            "vocab_size": self.vocab_size,
            "special_tokens": self.special_tokens,
            "model_type": "bpe",
            "backend": "subword",
        }
        p.with_suffix(p.suffix + ".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "SubwordTokenizer":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        if "id_to_token" not in d and "token_to_id" in d:
            items = sorted(d["token_to_id"].items(), key=lambda kv: kv[1])
            d["id_to_token"] = [k for k, _ in items]
        d.setdefault("special_tokens", [])
        return cls(d)
