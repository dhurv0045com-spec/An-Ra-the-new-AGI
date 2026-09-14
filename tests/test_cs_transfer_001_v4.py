"""Qualification for CS-TRANSFER-001 Amendment-2 direct-token surface."""
from __future__ import annotations

import json
from pathlib import Path

from anra_v5 import cs_transfer_001_data_v2 as data
from anra_v5 import cs_transfer_001_model as model

ROOT = Path(__file__).resolve().parents[1]


class _Identity:
    vocabulary_size = 24576
    special_token_ids = {"pad": 0, "unk": 1, "bos": 2, "eos": 3}


class _Tokenizer:
    identity = _Identity()


def test_amendment_2_is_prospective_and_triggered_by_real_data_gate():
    a2 = json.loads((ROOT / "experiments/CS_TRANSFER_001/AMENDMENT_2.json").read_text())
    assert a2["status"] == "PROSPECTIVE_PREEXECUTION"
    assert a2["outcomes_observed_before_amendment"] is False
    assert a2["evidence"]["gpu_training_started"] is False
    assert "sealed/identity=0.000" in a2["evidence"]["observed_failure"]


def test_physical_treatment_parameter_counts_unchanged():
    assert model.spec_for(4096).parameter_receipt().total == 8_920_320
    assert model.spec_for(24576).parameter_receipt().total == 14_163_200
    assert 14_163_200 - 8_920_320 == (24576 - 4096) * 256


def test_direct_surface_full_preregistered_counts_are_deterministic_and_clean():
    counts = {"training": 600, "development": 80, "sealed": 120}
    a = data.build_shared_surface(
        tokenizer=_Tokenizer(), seed=2026091303,
        candidate_worlds_per_family=5000, select_counts=counts,
    )
    b = data.build_shared_surface(
        tokenizer=_Tokenizer(), seed=2026091303,
        candidate_worlds_per_family=5000, select_counts=counts,
    )
    assert a["manifest_sha256"] == b["manifest_sha256"]
    assert a["manifest"]["surface_revision"] == "A2_DIRECT_TOKEN_COMMON_SPACE"
    assert a["manifest"]["contamination"]["clean"] is True
    assert a["manifest"]["predictive_shortcut_max"] < 0.35
    assert {k: len(v) for k, v in a["rows"].items()} == {
        "training": 3600, "development": 480, "sealed": 720,
    }
    for split, rows in a["rows"].items():
        assert all(r.prompt_ids and r.answer_ids for r in rows)
        assert all(min(r.content_ids) >= 4 for r in rows)
        assert all(max(r.content_ids) < 4096 for r in rows)
        assert max(len(r.answer_ids) for r in rows) <= 24
        assert max(len(r.content_ids) + 2 for r in rows) <= 256
        for fam in data.FAMILIES:
            assert sum(r.family == fam for r in rows) == counts[split]


def test_surface_rejects_wrong_tokenizer_identity():
    class BadIdentity:
        vocabulary_size = 4096
        special_token_ids = {"pad": 0, "unk": 1, "bos": 2, "eos": 3}
    class BadTokenizer:
        identity = BadIdentity()
    try:
        data.build_shared_surface(
            tokenizer=BadTokenizer(), seed=2026091303,
            candidate_worlds_per_family=5000,
            select_counts={"training": 1, "development": 1, "sealed": 1},
        )
    except ValueError as exc:
        assert "24576-entry" in str(exc)
    else:
        raise AssertionError("wrong tokenizer identity was accepted")


def test_serialization_roundtrip_is_exact():
    surface = data.build_shared_surface(
        tokenizer=_Tokenizer(), seed=2026091303,
        candidate_worlds_per_family=5000,
        select_counts={"training": 3, "development": 2, "sealed": 2},
    )
    for rows in surface["rows"].values():
        assert data.deserialize_rows(data.serialize_rows(rows)) == rows
