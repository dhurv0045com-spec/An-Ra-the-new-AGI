from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable


_TOKEN_PATTERN = re.compile(r"\s+|[A-Za-z0-9_]+|[^\w\s]", re.UNICODE)


class _TokenLookup:
    def __init__(self, owner: "SubwordTokenizer") -> None:
        self._owner = owner

    def __call__(self, token: str) -> int | None:
        return self.get(token)

    def __len__(self) -> int:
        if self._owner.backend == "hf":
            return int(self._owner._tokenizer.get_vocab_size())
        return len(self._owner._tokenizer["id_to_token"])

    def get(self, token: str, default=None):
        if self._owner.backend == "hf":
            idx = self._owner._tokenizer.token_to_id(token)
            return default if idx is None else idx
        return self._owner._tokenizer["token_to_id"].get(token, default)

    def __getitem__(self, token: str) -> int:
        idx = self.get(token)
        if idx is None:
            raise KeyError(token)
        return int(idx)

    def __contains__(self, token: object) -> bool:
        return isinstance(token, str) and self.get(token) is not None


class SubwordTokenizer:
    """V2 tokenizer with a `tokenizers` backend and a dependency-free fallback."""

    def __init__(
        self,
        tokenizer,
        *,
        vocab_size: int,
        special_tokens: list[str],
        model_type: str = "bpe",
        backend: str = "hf",
    ):
        self._tokenizer = tokenizer
        self.vocab_size = int(vocab_size)
        self.special_tokens = list(special_tokens)
        self.model_type = model_type
        self.backend = backend
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
        self.token_to_id = _TokenLookup(self)
        self.special_ids = {token: int(self.token_to_id.get(token, idx)) for idx, token in enumerate(self.special_tokens)}
        self.pad_token_id = self.special_ids.get(self.pad_token, 0)
        self.unk_token_id = self.special_ids.get(self.unk_token, 1)
        self.bos_token_id = self.special_ids.get(self.bos_token, 2)
        self.eos_token_id = self.special_ids.get(self.eos_token, 3)

    @staticmethod
    def _try_import_tokenizers():
        try:
            from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers
        except ImportError:
            return None
        return Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

    @classmethod
    def train_from_texts(
        cls,
        texts: Iterable[str],
        *,
        vocab_size: int = 4096,
        min_frequency: int = 2,
        special_tokens: list[str] | None = None,
    ) -> "SubwordTokenizer":
        imports = cls._try_import_tokenizers()
        special_tokens = list(special_tokens or ["<pad>", "<unk>", "<bos>", "<eos>"])
        material = list(texts)

        if imports is not None:
            Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers = imports
            tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
            tokenizer.normalizer = normalizers.Sequence([normalizers.NFKC()])
            tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
            tokenizer.decoder = decoders.ByteLevel()
            trainer = trainers.BpeTrainer(
                vocab_size=vocab_size,
                min_frequency=min_frequency,
                special_tokens=special_tokens,
            )
            tokenizer.train_from_iterator(material, trainer=trainer)
            current = tokenizer.get_vocab_size()
            if current < vocab_size:
                tokenizer.add_tokens([f"<reserved_{idx:05d}>" for idx in range(current, vocab_size)])
            return cls(
                tokenizer,
                vocab_size=tokenizer.get_vocab_size(),
                special_tokens=special_tokens,
                model_type="bpe",
                backend="hf",
            )

        vocab = cls._train_fallback_vocab(material, vocab_size=vocab_size, special_tokens=special_tokens)
        return cls(
            vocab,
            vocab_size=len(vocab["id_to_token"]),
            special_tokens=special_tokens,
            model_type="fallback_unigram",
            backend="fallback",
        )

    @staticmethod
    def _train_fallback_vocab(texts: list[str], *, vocab_size: int, special_tokens: list[str] | None = None) -> dict[str, object]:
        counter: Counter[str] = Counter()
        chars: set[str] = set()
        for text in texts:
            pieces = _TOKEN_PATTERN.findall(text)
            counter.update(pieces)
            chars.update(text)

        special_tokens = list(special_tokens or ["<pad>", "<unk>", "<bos>", "<eos>"])
        ordered_tokens = list(special_tokens)

        room = max(0, vocab_size - len(ordered_tokens))
        most_common = [token for token, _ in counter.most_common(room)]
        ordered_tokens.extend(most_common)

        for char in sorted(chars):
            if char not in ordered_tokens and len(ordered_tokens) < vocab_size:
                ordered_tokens.append(char)

        while len(ordered_tokens) < vocab_size:
            ordered_tokens.append(f"<reserved_{len(ordered_tokens):05d}>")

        token_to_id = {token: idx for idx, token in enumerate(ordered_tokens)}
        return {
            "token_to_id": token_to_id,
            "id_to_token": ordered_tokens,
        }

    @classmethod
    def load(cls, path: str | Path) -> "SubwordTokenizer":
        path = Path(path)
        meta_path = path.with_suffix(path.suffix + ".meta.json")
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        special_tokens = meta.get("special_tokens", ["<pad>", "<unk>", "<bos>", "<eos>"])
        model_type = str(meta.get("model_type", "bpe"))
        backend = str(meta.get("backend", "hf"))

        if backend == "fallback":
            payload = json.loads(path.read_text(encoding="utf-8"))
            return cls(
                payload,
                vocab_size=int(meta.get("vocab_size", len(payload.get("id_to_token", [])))),
                special_tokens=list(special_tokens),
                model_type=model_type,
                backend="fallback",
            )

        imports = cls._try_import_tokenizers()
        if imports is None:
            raise RuntimeError(
                "tokenizers is not installed and this tokenizer file requires the tokenizers backend."
            )
        Tokenizer = imports[0]
        tokenizer = Tokenizer.from_file(str(path))
        return cls(
            tokenizer,
            vocab_size=int(meta.get("vocab_size", tokenizer.get_vocab_size())),
            special_tokens=list(special_tokens),
            model_type=model_type,
            backend="hf",
        )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.backend == "hf":
            self._tokenizer.save(str(path))
        else:
            path.write_text(json.dumps(self._tokenizer, indent=2), encoding="utf-8")
        meta_path = path.with_suffix(path.suffix + ".meta.json")
        meta_path.write_text(
            json.dumps(
                {
                    "vocab_size": self.vocab_size,
                    "special_tokens": self.special_tokens,
                    "model_type": self.model_type,
                    "backend": self.backend,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return path

    def _fallback_encode_piece(self, piece: str) -> list[int]:
        vocab = self._tokenizer["token_to_id"]
        if piece in vocab:
            return [int(vocab[piece])]

        ids: list[int] = []
        pos = 0
        while pos < len(piece):
            matched = None
            for end in range(len(piece), pos, -1):
                candidate = piece[pos:end]
                if candidate in vocab:
                    matched = candidate
                    break
            if matched is None:
                char = piece[pos]
                ids.append(int(vocab.get(char, vocab[self.unk_token])))
                pos += 1
            else:
                ids.append(int(vocab[matched]))
                pos += len(matched)
        return ids

    def _split_preserving_specials(self, text: str) -> list[str]:
        if not text:
            return []
        specials = sorted((tok for tok in self.special_tokens if tok), key=len, reverse=True)
        if not specials:
            return _TOKEN_PATTERN.findall(text)
        pieces: list[str] = []
        pos = 0
        while pos < len(text):
            matched = None
            for token in specials:
                if text.startswith(token, pos):
                    matched = token
                    break
            if matched is not None:
                pieces.append(matched)
                pos += len(matched)
                continue
            next_special = len(text)
            for token in specials:
                idx = text.find(token, pos + 1)
                if idx != -1:
                    next_special = min(next_special, idx)
            pieces.extend(_TOKEN_PATTERN.findall(text[pos:next_special]))
            pos = next_special
        return pieces

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        if self.backend == "hf":
            ids = self._tokenizer.encode(text).ids
        else:
            ids = []
            for piece in self._split_preserving_specials(text):
                ids.extend(self._fallback_encode_piece(piece))
        if add_special_tokens:
            return [self.bos_token_id, *ids, self.eos_token_id]
        return ids

    def decode(self, ids: list[int]) -> str:
        reverse_special = {idx: token for token, idx in self.special_ids.items()}
        if ids and all(int(token_id) in reverse_special for token_id in ids):
            return "".join(reverse_special[int(token_id)] for token_id in ids)
        filtered = [
            int(token_id)
            for token_id in ids
            if token_id not in {self.pad_token_id, self.bos_token_id, self.eos_token_id}
        ]
        if self.backend == "hf":
            return self._tokenizer.decode(filtered)
        id_to_token = self._tokenizer["id_to_token"]
        return "".join(id_to_token[token_id] for token_id in filtered if 0 <= token_id < len(id_to_token))
