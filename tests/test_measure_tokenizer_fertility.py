from __future__ import annotations

import json
from pathlib import Path

import pytest

from anra.anra_paths import V3_TOKENIZER_FILE
from scripts.measure_tokenizer_fertility import (
    CODE_TOK_PER_CHAR_GATE,
    ENGLISH_TOK_PER_WORD_GATE,
    build_report,
    heldout_english,
    measure_fertility,
)
from tokenizer.subword_tokenizer import SubwordTokenizer


@pytest.fixture(scope="module")
def v3_tokenizer() -> SubwordTokenizer:
    return SubwordTokenizer.load(V3_TOKENIZER_FILE)


def test_measure_fertility_invariants(v3_tokenizer: SubwordTokenizer) -> None:
    text = "H: What is the deployment process?\nANRA: I observe, then I act.\n"
    stats = measure_fertility(v3_tokenizer, text)
    assert stats["sampled_chars"] == len(text)
    assert stats["sampled_words"] == len(text.split())
    assert stats["tokens"] > 0
    assert stats["tok_per_word"] == pytest.approx(
        stats["tokens"] / stats["sampled_words"], abs=1e-5
    )
    assert stats["tok_per_char"] == pytest.approx(
        stats["tokens"] / stats["sampled_chars"], abs=1e-5
    )
    assert 0.0 <= stats["unk_rate"] <= 1.0


def test_measure_fertility_rejects_empty(v3_tokenizer: SubwordTokenizer) -> None:
    with pytest.raises(ValueError, match="non-empty text"):
        measure_fertility(v3_tokenizer, "")


def test_heldout_english_is_deterministic(tmp_path: Path) -> None:
    source = tmp_path / "corpus.txt"
    source.write_text(
        "\n".join(f"line number {index} about training language models" for index in range(200)),
        encoding="utf-8",
    )
    first = heldout_english(source, max_chars=10_000)
    second = heldout_english(source, max_chars=10_000)
    assert first == second
    assert first  # the hash rule must select a non-empty held-out slice
    full = source.read_text(encoding="utf-8")
    assert len(first) < len(full)  # and it must be a strict subset


def test_build_report_gates_and_audit(tmp_path: Path) -> None:
    english = tmp_path / "english.txt"
    english.write_text(
        "\n".join(
            f"sample sentence {index} with ordinary vocabulary and structure"
            for index in range(300)
        ),
        encoding="utf-8",
    )
    dfc = tmp_path / "dfc.jsonl"
    dfc.write_text(
        "\n".join(
            json.dumps({"text": f"<hyp>hypothesis {index}</hyp><verify>ok</verify>"})
            for index in range(50)
        ),
        encoding="utf-8",
    )

    report = build_report(
        V3_TOKENIZER_FILE,
        english_source=english,
        dfc_source=dfc,
        max_chars=5_000,
        max_units=2_000,
    )

    assert report["vocab_size"] == 8209
    sources = report["sources"]
    assert set(sources) == {"english_prose", "code", "math_dfc"}
    prose = sources["english_prose"]
    assert prose["gate"] == ENGLISH_TOK_PER_WORD_GATE
    assert prose["gate_pass"] == (prose["tok_per_word"] <= ENGLISH_TOK_PER_WORD_GATE)
    code = sources["code"]
    assert code["gate"] == CODE_TOK_PER_CHAR_GATE
    assert code["gate_pass"] == (code["tok_per_char"] <= CODE_TOK_PER_CHAR_GATE)
    audit = report["append_audit"]
    assert audit["sampled_units"] <= 2_000
    assert "candidate_tokens" not in audit  # trimmed to keep the report light
    assert isinstance(report["v4_migration_justified"], bool)
    assert report["v3_tax_confirmed"] == (prose["tok_per_word"] >= 1.5)
