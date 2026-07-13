from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.build_v4_tokenizer import build_v4
from tokenizer.subword_tokenizer import SubwordTokenizer
from training.v2_config import TOKENIZER_V4_32K_VOCAB_SIZE
from training.v2_runtime import active_tokenizer_identity

V3_TOKENIZER = Path(__file__).resolve().parents[1] / "tokenizer" / "tokenizer_v3.json"


def _eligible_corpus(tmp_path: Path) -> Path:
    # >=1M sampled units and a large projected reduction: 500 repeated
    # multi-character terms across enough lines to clear the audit's 1M gate.
    corpus = tmp_path / "campaign.txt"
    vocabulary = [f"campaignterm{i:04d}" for i in range(500)]
    line = " ".join(vocabulary)
    corpus.write_text("\n".join(line for _ in range(2_100)), encoding="utf-8")
    return corpus


def test_build_v4_blocks_without_corpus(tmp_path: Path) -> None:
    report = build_v4(
        [tmp_path / "absent.txt"],
        tmp_path / "out.json",
        ceiling=TOKENIZER_V4_32K_VOCAB_SIZE,
    )
    assert report["status"] == "blocked_on_corpus"


def test_build_v4_32k_end_to_end_and_proves(tmp_path: Path, monkeypatch) -> None:
    corpus = _eligible_corpus(tmp_path)
    output = tmp_path / "tokenizer_v4_32k.json"

    report = build_v4(
        [corpus], output, ceiling=TOKENIZER_V4_32K_VOCAB_SIZE, max_units=1_000_000
    )

    assert report["status"] == "built"
    proof = report["proof"]
    assert proof["frozen_prefix_unchanged"] is True
    assert proof["canonical_ids_stable"] is True
    assert proof["byte_roundtrip_ok"] is True
    assert proof["all_proofs_pass"] is True
    assert proof["grown_vocab_size"] == TOKENIZER_V4_32K_VOCAB_SIZE

    tokenizer = SubwordTokenizer.load(output)
    assert tokenizer.vocab_size == TOKENIZER_V4_32K_VOCAB_SIZE
    assert len(tokenizer.encode("campaignterm0001")) == 1
    monkeypatch.setenv("ANRA_TOKENIZER_PATH", str(output))
    identity = active_tokenizer_identity()
    assert identity["schema_version"] == 4
    assert identity["vocab_size"] == TOKENIZER_V4_32K_VOCAB_SIZE
    assert identity["sha256"] == report["output_sha256"]
    assert identity["probe_count"] == 500


def test_build_v4_reports_ineligible_small_corpus(tmp_path: Path) -> None:
    corpus = tmp_path / "tiny.txt"
    corpus.write_text("alpha beta gamma delta\n", encoding="utf-8")
    report = build_v4(
        [corpus], tmp_path / "out.json", ceiling=TOKENIZER_V4_32K_VOCAB_SIZE
    )
    assert report["status"] == "audit_not_eligible"


def test_build_v4_rejects_non_ready_campaign_manifest(tmp_path: Path) -> None:
    corpus = tmp_path / "campaign.txt"
    corpus.write_text("alpha beta gamma\n" * 100, encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"ready_for_v4": False}), encoding="utf-8")

    with pytest.raises(ValueError, match="ready_for_v4"):
        build_v4(
            [corpus],
            tmp_path / "v4.json",
            base_json=V3_TOKENIZER,
            campaign_manifest=manifest,
        )


def test_build_v4_rejects_corpus_outside_manifest_bound_slice(tmp_path: Path) -> None:
    corpus = tmp_path / "campaign.txt"
    corpus.write_text("alpha beta gamma\n" * 100, encoding="utf-8")
    extra = tmp_path / "legacy.txt"
    extra.write_text("unbound legacy material\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "ready_for_v4": True,
                "meets_min_slice": True,
                "all_heldout_disjoint": True,
                "campaign_mix_verified": True,
                "all_required_sources_present": True,
                "train_path": str(corpus),
                "train_sha256": hashlib.sha256(corpus.read_bytes()).hexdigest(),
                "train_bytes": corpus.stat().st_size,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="exactly the manifest-bound"):
        build_v4(
            [corpus, extra],
            tmp_path / "v4.json",
            base_json=V3_TOKENIZER,
            campaign_manifest=manifest,
        )
