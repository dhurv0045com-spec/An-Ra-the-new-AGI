from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import TypedDict

from training.v2_config import (
    CANONICAL_SPECIAL_TOKEN_IDS,
    CANONICAL_SPECIAL_TOKENS,
    CANONICAL_VOCAB_SIZE,
)

from tokenizer.subword_tokenizer import SubwordTokenizer

SPECIAL_TOKENS = CANONICAL_SPECIAL_TOKENS
TARGET_VOCAB = CANONICAL_VOCAB_SIZE


class TokenizerCompetenceReport(TypedDict):
    backend: str
    model_type: str
    sampled_characters: int
    sampled_tokens: int
    chars_per_token: float
    unk_rate: float
    roundtrip_ok: bool
    byte_safe: bool
    competent: bool


def tokenizer_competence_report(
    tokenizer_json: Path,
    corpus_paths: list[Path],
    *,
    max_characters: int = 1_000_000,
    minimum_chars_per_token: float = 3.5,
    maximum_unk_rate: float = 0.02,
) -> TokenizerCompetenceReport:
    tokenizer = SubwordTokenizer.load(tokenizer_json)
    chunks: list[str] = []
    remaining = max(1, int(max_characters))
    for path in corpus_paths:
        if remaining <= 0 or not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")[:remaining]
        chunks.append(text)
        remaining -= len(text)
    text = "\n".join(chunks)
    if not text:
        raise ValueError("Tokenizer competence requires non-empty held-out text")
    ids = tokenizer.encode(text)
    unknown = sum(token_id == tokenizer.unk_token_id for token_id in ids)
    unk_rate = unknown / max(1, len(ids))
    chars_per_token = len(text) / max(1, len(ids))
    roundtrip_ok = tokenizer.decode(ids) == text
    byte_probe = "An-Ra byte probe: cafe \u03bb \u2211 \U0001f680"
    byte_safe = tokenizer.decode(tokenizer.encode(byte_probe)) == byte_probe
    competent = (
        tokenizer.backend != "fallback"
        and roundtrip_ok
        and byte_safe
        and unk_rate < maximum_unk_rate
        and chars_per_token > minimum_chars_per_token
    )
    return {
        "backend": tokenizer.backend,
        "model_type": tokenizer.model_type,
        "sampled_characters": len(text),
        "sampled_tokens": len(ids),
        "chars_per_token": round(chars_per_token, 6),
        "unk_rate": round(unk_rate, 8),
        "roundtrip_ok": roundtrip_ok,
        "byte_safe": byte_safe,
        "competent": competent,
    }


def validate_tokenizer(tokenizer_json: Path, dataset_path: Path) -> dict[str, float | bool]:
    tok = SubwordTokenizer.load(tokenizer_json)
    meta_path = tokenizer_json.with_suffix(tokenizer_json.suffix + ".meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    target_vocab = int(meta.get("vocab_size", TARGET_VOCAB))
    expected_specials = list(meta.get("special_tokens", SPECIAL_TOKENS))
    if len(tok.token_to_id) != target_vocab:
        raise AssertionError(
            f"[tokenizer_v3] expected {target_vocab} tokens, "
            f"found {len(tok.token_to_id)}"
        )
    for token, expected_id in CANONICAL_SPECIAL_TOKEN_IDS.items():
        actual = tok.token_to_id.get(token)
        if actual != expected_id:
            raise AssertionError(
                f"[tokenizer_v3] special token {token!r} has id {actual}, "
                f"expected {expected_id}"
            )
    missing = [token for token in expected_specials if tok.token_to_id.get(token) is None]
    if missing:
        raise AssertionError(f"[tokenizer_v3] missing canonical special tokens: {missing}")

    text = dataset_path.read_text(encoding='utf-8', errors='replace')[:20000]

    ids = tok.encode(text)
    roundtrip_ok = tok.decode(ids)[:200] == text[:200]
    unk_rate = (sum(1 for i in ids if i == tok.special_ids["<unk>"]) / max(1, len(ids)))

    code_chars = sum(1 for c in text if c in "{}[]()=;:_<>/\\")
    token_density = len(ids) / max(1, len(text))

    special_ok = all(
        tok.decode([idx]) == piece
        for piece, idx in tok.special_ids.items()
        if piece in expected_specials
    )
    competence = tokenizer_competence_report(
        tokenizer_json,
        [dataset_path],
        max_characters=20_000,
        minimum_chars_per_token=0.0,
        maximum_unk_rate=1.0,
    )
    return {
        "roundtrip_ok": roundtrip_ok,
        "unk_rate": float(unk_rate),
        "code_token_density": float(token_density * code_chars / max(1, len(text))),
        "special_roundtrip_ok": special_ok,
        "vocab_size_ok": True,
        "special_tokens_ok": True,
        "chars_per_token": competence["chars_per_token"],
        "byte_safe": competence["byte_safe"],
        "competent_backend": tok.backend != "fallback",
    }


def audit_token_fertility(
    tokenizer_json: Path,
    corpus_paths: list[Path],
    *,
    max_units: int = 1_000_000,
) -> dict[str, object]:
    tokenizer = SubwordTokenizer.load(tokenizer_json)
    units: list[str] = []
    pattern = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+(?:\.\d+)?|[^\w\s]+")
    for path in corpus_paths:
        with path.open("r", encoding="utf-8", errors="replace") as stream:
            for line in stream:
                units.extend(pattern.findall(line))
                if len(units) >= max_units:
                    break
        if len(units) >= max_units:
            break
    units = units[:max_units]
    frequencies = Counter(units)
    current_tokens = sum(len(tokenizer.encode(unit)) for unit in units)
    existing = getattr(tokenizer, "token_to_id", {})
    candidates = [
        unit
        for unit, _ in frequencies.most_common()
        if len(unit) > 1 and existing.get(unit) is None
    ][: 16_384 - tokenizer.vocab_size]
    candidate_set = set(candidates)
    projected_tokens = sum(
        1 if unit in candidate_set else len(tokenizer.encode(unit)) for unit in units
    )
    reduction = (current_tokens - projected_tokens) / max(1, current_tokens)
    return {
        "schema_version": 1,
        "sampled_units": len(units),
        "current_tokens": current_tokens,
        "projected_tokens": projected_tokens,
        "projected_reduction": round(reduction, 6),
        "candidate_count": len(candidates),
        "eligible_for_schema_v4": len(units) >= max_units and reduction >= 0.15,
        "candidate_tokens": candidates,
    }


def build_append_only_v4(
    tokenizer_json: Path,
    output_json: Path,
    audit: dict[str, object],
) -> Path:
    if not bool(audit.get("eligible_for_schema_v4", False)):
        raise RuntimeError(
            "Tokenizer v4 growth requires a one-million-unit audit and >=15% reduction"
        )
    payload = json.loads(tokenizer_json.read_text(encoding="utf-8"))
    token_to_id = dict(payload["token_to_id"])
    id_to_token = list(payload["id_to_token"])
    if len(id_to_token) != 8209:
        raise ValueError("Append-only v4 growth requires the canonical 8,209-token v3 base")
    if any(
        token_to_id.get(token) != token_id
        for token, token_id in CANONICAL_SPECIAL_TOKEN_IDS.items()
    ):
        raise ValueError("Canonical token IDs changed before v4 growth")
    decompositions: dict[str, list[int]] = {}
    base_tokenizer = SubwordTokenizer.load(tokenizer_json)
    for value in range(256):
        token = f"<0x{value:02X}>"
        if token not in token_to_id and len(id_to_token) < 16_384:
            token_to_id[token] = len(id_to_token)
            id_to_token.append(token)
            decompositions[token] = [base_tokenizer.unk_token_id]
    for candidate in audit.get("candidate_tokens", []):
        token = str(candidate)
        if token and token not in token_to_id and len(id_to_token) < 16_384:
            token_to_id[token] = len(id_to_token)
            id_to_token.append(token)
            decompositions[token] = base_tokenizer.encode(token)
    while len(id_to_token) < 16_384:
        token = f"<v4_reserved_{len(id_to_token):05d}>"
        token_to_id[token] = len(id_to_token)
        id_to_token.append(token)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_json.with_suffix(output_json.suffix + ".tmp")
    temporary.write_text(
        json.dumps({"token_to_id": token_to_id, "id_to_token": id_to_token}, indent=2),
        encoding="utf-8",
    )
    temporary.replace(output_json)
    meta = json.loads(
        tokenizer_json.with_suffix(tokenizer_json.suffix + ".meta.json").read_text(
            encoding="utf-8"
        )
    )
    meta.update(
        {
            "schema_version": 4,
            "vocab_size": 16_384,
            "base_vocab_size": 8_209,
            "append_only": True,
            "backend": "native_append_v4",
            "model_type": "native_greedy_byte_subword",
            "byte_safe": True,
            "token_decompositions": decompositions,
            "fertility_audit": audit,
        }
    )
    output_json.with_suffix(output_json.suffix + ".meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return output_json


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    results = validate_tokenizer(
        root / "tokenizer" / "tokenizer_v3.json",
        root / "training_data" / "anra_training.txt",
    )
    print(results)
