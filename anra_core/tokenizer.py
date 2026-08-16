from __future__ import annotations

import json
import hashlib
import re
from pathlib import Path

_TOKEN_PATTERN = re.compile(r"\s+|\w+|[^\w\s]", re.UNICODE)
_BYTE_PATTERN = re.compile(r"<0x([0-9A-F]{2})>")


class V4Tokenizer:
    def __init__(self, payload: dict[str, object], meta: dict[str, object]) -> None:
        if meta.get("backend") != "native_append_v4" or int(meta.get("vocab_size", -1)) != 32_768:
            raise ValueError("tokenizer is not the canonical native V4 32K artifact")
        if int(meta.get("schema_version", 4)) != 4:
            raise ValueError("unsupported tokenizer schema")
        self.id_to_token = list(payload["id_to_token"])
        self.token_to_id = {str(k): int(v) for k, v in dict(payload["token_to_id"]).items()}
        if len(self.id_to_token) != 32_768 or len(self.token_to_id) != 32_768:
            raise ValueError("tokenizer vocabulary is incomplete")
        self.special_tokens = list(meta["special_tokens"])
        expected_special_ids = [*range(13), *range(8_192, 8_209)]
        if len(self.special_tokens) != len(expected_special_ids):
            raise ValueError("canonical V4 tokenizer must contain exactly 30 special tokens")
        for expected_id, token in zip(expected_special_ids, self.special_tokens, strict=True):
            if self.token_to_id.get(token) != expected_id:
                raise ValueError(f"special token ID drift: {token}")
        self.pad_token_id, self.unk_token_id = 0, 1
        self.bos_token_id, self.eos_token_id = 2, 3
        self._specials = frozenset(self.special_tokens)
        ordered = sorted(self.special_tokens, key=len, reverse=True)
        self._special_pattern = re.compile("(" + "|".join(map(re.escape, ordered)) + ")")
        self._trie: dict[object, object] = {}
        for token, token_id in self.token_to_id.items():
            node = self._trie
            for character in token:
                node = node.setdefault(character, {})  # type: ignore[assignment]
            node[None] = token_id

    @classmethod
    def load(cls, path: str | Path) -> "V4Tokenizer":
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        meta = json.loads(path.with_suffix(path.suffix + ".meta.json").read_text(encoding="utf-8"))
        return cls(payload, meta)

    def identity(self, *, probe_count: int = 500) -> dict[str, object]:
        vocabulary_sha256 = hashlib.sha256(
            json.dumps(self.token_to_id, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        probes: list[list[int]] = []
        for index in range(probe_count):
            text = (
                f"H: An-Ra tokenizer probe {index:03d}: "
                f"code_{index % 17} = ({index % 97} + {index % 31}); "
                f"logic, math, science, memory.\nANRA: verified {index % 11}."
            )
            encoded = self.encode(text)
            if self.encode(self.decode(encoded)) != encoded:
                raise ValueError(f"tokenizer probe {index} is not ID-stable")
            probes.append(encoded)
        probe_sha256 = hashlib.sha256(
            json.dumps(probes, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return {
            "schema_version": 4,
            "vocab_size": len(self.id_to_token),
            "special_token_ids": {
                token: self.token_to_id[token] for token in self.special_tokens
            },
            "vocabulary_sha256": vocabulary_sha256,
            "probe_count": probe_count,
            "probe_sha256": probe_sha256,
        }

    def assert_checkpoint_contract(self, contract: object) -> None:
        if not isinstance(contract, dict) or not contract.get("available"):
            raise ValueError("checkpoint does not contain a usable tokenizer contract")
        identity = self.identity(probe_count=int(contract.get("probe_count", 500)))
        for key in (
            "schema_version", "vocab_size", "special_token_ids",
            "vocabulary_sha256", "probe_count", "probe_sha256",
        ):
            if contract.get(key) != identity[key]:
                raise ValueError(f"checkpoint/tokenizer identity mismatch: {key}")

    def _pieces(self, text: str) -> list[str]:
        result: list[str] = []
        for part in self._special_pattern.split(text):
            if part in self._specials:
                result.append(part)
            elif part:
                result.extend(_TOKEN_PATTERN.findall(part))
        return result

    def encode(self, text: str) -> list[int]:
        output: list[int] = []
        for piece in self._pieces(text):
            direct = self.token_to_id.get(piece)
            if direct is not None:
                output.append(direct)
                continue
            position = 0
            while position < len(piece):
                node = self._trie
                cursor, match, end = position, None, position
                while cursor < len(piece) and isinstance(node.get(piece[cursor]), dict):
                    node = node[piece[cursor]]  # type: ignore[index,assignment]
                    cursor += 1
                    if None in node:
                        match, end = int(node[None]), cursor
                if match is not None:
                    output.append(match)
                    position = end
                else:
                    for value in piece[position].encode("utf-8"):
                        output.append(self.token_to_id.get(f"<0x{value:02X}>", self.unk_token_id))
                    position += 1
        return output

    def decode(self, ids: list[int]) -> str:
        reverse_special = {
            self.token_to_id[token]: token for token in self.special_tokens
        }
        if ids and all(int(token_id) in reverse_special for token_id in ids):
            return "".join(reverse_special[int(token_id)] for token_id in ids)
        pieces: list[str] = []
        pending = bytearray()
        for token_id in ids:
            if token_id in {self.pad_token_id, self.bos_token_id, self.eos_token_id}:
                continue
            if not 0 <= token_id < len(self.id_to_token):
                continue
            token = self.id_to_token[token_id]
            match = _BYTE_PATTERN.fullmatch(token)
            if match:
                pending.append(int(match.group(1), 16))
            else:
                if pending:
                    pieces.append(bytes(pending).decode("utf-8", errors="replace"))
                    pending.clear()
                pieces.append(token)
        if pending:
            pieces.append(bytes(pending).decode("utf-8", errors="replace"))
        return "".join(pieces)
