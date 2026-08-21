from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from .errors import RepresentationIncompatibleError

_TOKEN_PATTERN = re.compile(r"\s+|\w+|[^\w\s]", re.UNICODE)
_BYTE_PATTERN = re.compile(r"<0x([0-9A-F]{2})>")


class V4Tokenizer:
    def __init__(self, payload: dict[str, object], meta: dict[str, object]) -> None:
        if meta.get("backend") != "native_append_v4" or int(meta.get("vocab_size", -1)) != 32_768:
            raise RepresentationIncompatibleError(
                "tokenizer is not the canonical native V4 32K artifact"
            )
        if int(meta.get("schema_version", 4)) != 4:
            raise RepresentationIncompatibleError("unsupported tokenizer schema")
        try:
            self.id_to_token = [str(token) for token in list(payload["id_to_token"])]
            self.token_to_id = {
                str(key): int(value) for key, value in dict(payload["token_to_id"]).items()
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise RepresentationIncompatibleError(
                "tokenizer payload does not contain a valid token mapping",
                details={"cause": type(exc).__name__},
            ) from exc
        if len(self.id_to_token) != 32_768 or len(self.token_to_id) != 32_768:
            raise RepresentationIncompatibleError("tokenizer vocabulary is incomplete")
        for token_id, token in enumerate(self.id_to_token):
            if self.token_to_id.get(token) != token_id:
                raise RepresentationIncompatibleError(
                    "canonical token/ID mapping is inconsistent",
                    details={"token": token, "expected_id": token_id},
                )
        try:
            self.special_tokens = [str(token) for token in list(meta["special_tokens"])]
        except (KeyError, TypeError) as exc:
            raise RepresentationIncompatibleError(
                "tokenizer metadata does not contain special tokens",
                details={"cause": type(exc).__name__},
            ) from exc
        expected_special_ids = [*range(13), *range(8_192, 8_209)]
        if len(self.special_tokens) != len(expected_special_ids):
            raise RepresentationIncompatibleError(
                "canonical V4 tokenizer must contain exactly 30 special tokens"
            )
        for expected_id, token in zip(expected_special_ids, self.special_tokens, strict=True):
            if self.token_to_id.get(token) != expected_id:
                raise RepresentationIncompatibleError(
                    f"special token ID drift: {token}",
                    details={"token": token, "expected_id": expected_id},
                )
        self.pad_token_id, self.unk_token_id = 0, 1
        self.bos_token_id, self.eos_token_id = 2, 3
        self._specials = frozenset(self.special_tokens)
        # Identity is a pure function of the immutable vocabulary; cache it so
        # per-call state validation does not re-run 500 probe encodes.
        self._identity_cache: dict[int, dict[str, object]] = {}
        ordered = sorted(self.special_tokens, key=len, reverse=True)
        self._special_pattern = re.compile("(" + "|".join(map(re.escape, ordered)) + ")")
        self._trie: dict[object, object] = {}
        for token, token_id in self.token_to_id.items():
            node = self._trie
            for character in token:
                node = node.setdefault(character, {})  # type: ignore[assignment]
            node[None] = token_id

    @classmethod
    def load(cls, path: str | Path) -> V4Tokenizer:
        path = Path(path)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            meta = json.loads(
                path.with_suffix(path.suffix + ".meta.json").read_text(encoding="utf-8")
            )
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise RepresentationIncompatibleError(
                "tokenizer artifact is not valid UTF-8 JSON",
                details={"path": str(path), "cause": type(exc).__name__},
            ) from exc
        if not isinstance(payload, dict) or not isinstance(meta, dict):
            raise RepresentationIncompatibleError(
                "tokenizer payload and metadata must be mappings",
                details={"path": str(path)},
            )
        return cls(payload, meta)

    @classmethod
    def load_canonical(cls) -> V4Tokenizer:
        return cls.load(Path(__file__).parent / "assets" / "tokenizer_v4_32k.json")

    def identity(self, *, probe_count: int = 500) -> dict[str, object]:
        cached = self._identity_cache.get(probe_count)
        if cached is not None:
            return dict(cached)
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
                raise RepresentationIncompatibleError(
                    f"tokenizer probe {index} is not ID-stable",
                    details={"probe_index": index},
                )
            probes.append(encoded)
        probe_sha256 = hashlib.sha256(
            json.dumps(probes, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        identity = {
            "schema_version": 4,
            "vocab_size": len(self.id_to_token),
            "special_token_ids": {
                token: self.token_to_id[token] for token in self.special_tokens
            },
            "vocabulary_sha256": vocabulary_sha256,
            "probe_count": probe_count,
            "probe_sha256": probe_sha256,
        }
        self._identity_cache[probe_count] = identity
        return dict(identity)

    def assert_checkpoint_contract(self, contract: object) -> None:
        if not isinstance(contract, dict) or not contract.get("available"):
            raise RepresentationIncompatibleError(
                "checkpoint does not contain a usable tokenizer contract"
            )
        try:
            probe_count = int(contract.get("probe_count", 500))
        except (TypeError, ValueError) as exc:
            raise RepresentationIncompatibleError(
                "checkpoint tokenizer probe count is invalid"
            ) from exc
        if probe_count < 1 or probe_count > 10_000:
            raise RepresentationIncompatibleError(
                "checkpoint tokenizer probe count is outside the supported range",
                details={"probe_count": probe_count},
            )
        identity = self.identity(probe_count=probe_count)
        for key in (
            "schema_version", "vocab_size", "special_token_ids",
            "vocabulary_sha256", "probe_count", "probe_sha256",
        ):
            if contract.get(key) != identity[key]:
                raise RepresentationIncompatibleError(
                    f"checkpoint/tokenizer identity mismatch: {key}",
                    details={"field": key, "expected": identity[key], "got": contract.get(key)},
                )

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
