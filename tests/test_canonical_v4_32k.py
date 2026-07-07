"""Stream B proofs: canonical 32k append-only V4 (MASTER_UPGRADE Layer 3).

Proves the plan's five V4 gates at the 32,768 ceiling: IDs 0-8208 unchanged,
exact round-trip / byte-safety, the parameter contract, checkpoint migration
that preserves legacy rows bit-for-bit while appending mean-initialized new
rows, and the fertility-audit eligibility gate. The proven 16,384 fallback is
kept green in test_tokenizer_v3.py / test_definitive_architecture.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from tokenizer.subword_tokenizer import SubwordTokenizer
from tokenizer.validate_tokenizer_v3 import (
    V3_BASE_VOCAB_SIZE,
    audit_token_fertility,
    build_append_only_v4,
)
from training.v2_config import (
    CANONICAL_SPECIAL_TOKEN_IDS,
    CANONICAL_V4_VOCAB_SIZE,
    TOKENIZER_V4_32K_VOCAB_SIZE,
    TOKENIZER_V4_VOCAB_SIZE,
    V2_FRONTIER,
    V2_FRONTIER_PARAMETER_COUNT,
    V4_VOCAB_SIZES,
    frontier_parameter_count,
    is_v4_vocab_size,
)

V3_TOKENIZER = Path(__file__).resolve().parents[1] / "tokenizer" / "tokenizer_v3.json"

_ELIGIBLE_AUDIT = {
    "eligible_for_schema_v4": True,
    "sampled_units": 1_000_000,
    "projected_reduction": 0.22,
    "candidate_tokens": [
        "phosphorylation",
        "oxidative",
        "nutrients",
        "eigenvalue",
        "manifold",
    ],
}


def _copy_v3_base(tmp_path: Path) -> Path:
    target = tmp_path / "tokenizer_v3.json"
    target.write_bytes(V3_TOKENIZER.read_bytes())
    meta = V3_TOKENIZER.with_suffix(V3_TOKENIZER.suffix + ".meta.json")
    target.with_suffix(target.suffix + ".meta.json").write_bytes(meta.read_bytes())
    return target


def _grow_32k(tmp_path: Path) -> Path:
    base = _copy_v3_base(tmp_path)
    output = tmp_path / "tokenizer_v4_32k.json"
    build_append_only_v4(
        base, output, dict(_ELIGIBLE_AUDIT), target_vocab_size=TOKENIZER_V4_32K_VOCAB_SIZE
    )
    return output


def test_canonical_v4_is_32k_and_config_is_consistent() -> None:
    assert CANONICAL_V4_VOCAB_SIZE == 32_768
    assert TOKENIZER_V4_32K_VOCAB_SIZE in V4_VOCAB_SIZES
    assert TOKENIZER_V4_VOCAB_SIZE in V4_VOCAB_SIZES  # proven fallback retained
    assert is_v4_vocab_size(32_768)
    assert is_v4_vocab_size(16_384)
    assert not is_v4_vocab_size(8_209)


def test_32k_growth_preserves_every_v3_token_id(tmp_path: Path) -> None:
    base_ids = json.loads(V3_TOKENIZER.read_text(encoding="utf-8"))["id_to_token"]
    grown_path = _grow_32k(tmp_path)
    grown = json.loads(grown_path.read_text(encoding="utf-8"))

    assert len(grown["id_to_token"]) == TOKENIZER_V4_32K_VOCAB_SIZE
    # IDs 0-8208 are byte-for-byte the frozen V3 prefix.
    assert grown["id_to_token"][:V3_BASE_VOCAB_SIZE] == base_ids
    tokenizer = SubwordTokenizer.load(grown_path)
    for token, token_id in CANONICAL_SPECIAL_TOKEN_IDS.items():
        assert tokenizer.token_to_id.get(token) == token_id


def test_32k_tokenizer_is_byte_safe_and_roundtrips(tmp_path: Path) -> None:
    tokenizer = SubwordTokenizer.load(_grow_32k(tmp_path))
    assert tokenizer.vocab_size == TOKENIZER_V4_32K_VOCAB_SIZE
    assert tokenizer.backend == "native_append_v4"
    for probe in (
        "native bytes: cafe λ ∑ \U0001f680",
        "def f(x):\n    return x + 1  # 中文 comment",
        "H: hello\nANRA: phosphorylation and oxidative eigenvalue",
    ):
        assert tokenizer.decode(tokenizer.encode(probe)) == probe
    # Appended candidate is a single id (the fertility win).
    assert len(tokenizer.encode("phosphorylation")) == 1


def test_32k_meta_declares_schema_v4_contract(tmp_path: Path) -> None:
    grown = _grow_32k(tmp_path)
    meta = json.loads(
        grown.with_suffix(grown.suffix + ".meta.json").read_text(encoding="utf-8")
    )
    assert meta["schema_version"] == 4
    assert meta["vocab_size"] == TOKENIZER_V4_32K_VOCAB_SIZE
    assert meta["base_vocab_size"] == V3_BASE_VOCAB_SIZE
    assert meta["append_only"] is True
    assert meta["backend"] == "native_append_v4"
    assert meta["byte_safe"] is True


def test_32k_parameter_contract() -> None:
    delta = frontier_parameter_count(32_768) - V2_FRONTIER_PARAMETER_COUNT
    assert delta == (32_768 - 8_209) * V2_FRONTIER.n_embd
    # 16k and 32k both satisfy their pinned contracts (no AssertionError).
    assert frontier_parameter_count(16_384) < frontier_parameter_count(32_768)


def test_checkpoint_migration_to_32k_preserves_legacy_and_appends(
    tmp_path: Path, monkeypatch
) -> None:
    from training.v2_runtime import migrate_checkpoint_state

    grown = _grow_32k(tmp_path)
    monkeypatch.setenv("ANRA_TOKENIZER_PATH", str(grown))

    embed_dim = 8
    legacy = torch.randn(V3_BASE_VOCAB_SIZE, embed_dim)
    source = {
        "token_embedding_table.weight": legacy,
        "lm_head.weight": legacy.clone(),
    }
    target = {
        "token_embedding_table.weight": torch.zeros(TOKENIZER_V4_32K_VOCAB_SIZE, embed_dim),
        "lm_head.weight": torch.zeros(TOKENIZER_V4_32K_VOCAB_SIZE, embed_dim),
    }
    migrated, report = migrate_checkpoint_state(source, target)

    # Legacy rows preserved bit-for-bit.
    torch.testing.assert_close(
        migrated["token_embedding_table.weight"][:V3_BASE_VOCAB_SIZE], legacy
    )
    assert migrated["token_embedding_table.weight"].shape[0] == TOKENIZER_V4_32K_VOCAB_SIZE
    assert report["source_vocab_size"] == V3_BASE_VOCAB_SIZE
    assert report["target_vocab_size"] == TOKENIZER_V4_32K_VOCAB_SIZE
    assert report["appended_token_rows"] == TOKENIZER_V4_32K_VOCAB_SIZE - V3_BASE_VOCAB_SIZE
    assert report["tokenizer_schema_version"] == 4
    assert report["legacy_rows_preserved"] is True
    # Appended rows are finite and not all-zero (mean-init + sinusoid offset).
    appended = migrated["token_embedding_table.weight"][V3_BASE_VOCAB_SIZE:]
    assert torch.isfinite(appended).all()
    assert appended.abs().sum() > 0


def test_checkpoint_migration_to_32k_is_deterministic(tmp_path: Path, monkeypatch) -> None:
    from training.v2_runtime import migrate_checkpoint_state

    grown = _grow_32k(tmp_path)
    monkeypatch.setenv("ANRA_TOKENIZER_PATH", str(grown))
    legacy = torch.randn(V3_BASE_VOCAB_SIZE, 6)
    source = {"token_embedding_table.weight": legacy, "lm_head.weight": legacy.clone()}

    def run() -> torch.Tensor:
        migrated, _ = migrate_checkpoint_state(
            source,
            {
                "token_embedding_table.weight": torch.zeros(TOKENIZER_V4_32K_VOCAB_SIZE, 6),
                "lm_head.weight": torch.zeros(TOKENIZER_V4_32K_VOCAB_SIZE, 6),
            },
        )
        return migrated["token_embedding_table.weight"]

    torch.testing.assert_close(run(), run())


def test_fertility_audit_respects_the_32k_ceiling(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.txt"
    # Repeated multi-char units so the audit finds high-frequency merge candidates.
    vocabulary = [f"tokenunit{i:04d}" for i in range(400)]
    lines = [" ".join(vocabulary) for _ in range(400)]
    corpus.write_text("\n".join(lines), encoding="utf-8")

    audit_16k = audit_token_fertility(
        V3_TOKENIZER, [corpus], max_units=1_000_000, target_vocab_size=16_384
    )
    audit_32k = audit_token_fertility(
        V3_TOKENIZER, [corpus], max_units=1_000_000, target_vocab_size=32_768
    )

    assert audit_16k["target_vocab_size"] == 16_384
    assert audit_32k["target_vocab_size"] == 32_768
    # The wider ceiling never has room for fewer candidates.
    assert audit_32k["candidate_count"] >= audit_16k["candidate_count"]
    for audit in (audit_16k, audit_32k):
        assert audit["sampled_units"] <= 1_000_000
        assert 0.0 <= audit["projected_reduction"] <= 1.0


def test_audit_and_build_reject_unknown_ceiling(tmp_path: Path) -> None:
    corpus = tmp_path / "c.txt"
    corpus.write_text("alpha beta gamma\n", encoding="utf-8")
    with pytest.raises(ValueError, match="ceiling"):
        audit_token_fertility(V3_TOKENIZER, [corpus], target_vocab_size=24_000)
    with pytest.raises(ValueError, match="ceiling"):
        build_append_only_v4(
            _copy_v3_base(tmp_path),
            tmp_path / "out.json",
            dict(_ELIGIBLE_AUDIT),
            target_vocab_size=24_000,
        )


def test_build_refuses_ineligible_audit(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="one-million-unit audit"):
        build_append_only_v4(
            _copy_v3_base(tmp_path),
            tmp_path / "out.json",
            {"eligible_for_schema_v4": False},
            target_vocab_size=TOKENIZER_V4_32K_VOCAB_SIZE,
        )
