"""Qualification tests for CS-TRANSFER-001 causal isolation."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

from anra_v5 import cs_transfer_001_data as data_mod
from anra_v5 import cs_transfer_001_model as model_mod

ROOT = Path(__file__).resolve().parents[1]
PREREG = ROOT / "experiments" / "CS_TRANSFER_001" / "PREREGISTRATION.json"


class _LowIdTokenizer:
    """Deterministic compact lexical tokenizer for qualification only.

    Every lexical/punctuation token maps into IDs 4..4095. This fixture tests
    the CS-TRANSFER low-ID selection contract without introducing an artificial
    one-byte-per-token answer-length failure that is unrelated to the real
    production tokenizer.
    """

    _pattern = re.compile(r"\w+|[^\w\s]", re.UNICODE)

    def encode(self, text: str):
        pieces = self._pattern.findall(text.lower())
        return [
            4 + (int.from_bytes(hashlib.sha256(piece.encode("utf-8")).digest()[:4], "big") % 4092)
            for piece in pieces
        ]

    def decode(self, ids):
        # Qualification never depends on text reconstruction; keep a stable
        # diagnostic representation for callers that exercise decode.
        return " ".join(f"tok{int(i)}" for i in ids)


def test_prereg_primary_contrast_is_physical_not_masked():
    p = json.loads(PREREG.read_text(encoding="utf-8"))
    assert set(p["model"]["arms"]) == {"PHYS_4096", "PHYS_24576"}
    assert p["causal_contrast"]["not_a_masking_proxy"] is True
    assert "byte-identical" in p["causal_contrast"]["token_sequences"]
    assert p["data"]["rendering"]["max_common_token_id"] == 4095


def test_exact_parameter_counts():
    assert model_mod.spec_for(4096).parameter_receipt().total == 8_920_320
    assert model_mod.spec_for(24576).parameter_receipt().total == 14_163_200
    assert 14_163_200 - 8_920_320 == (24576 - 4096) * 256


def test_matched_initialization_shared_bytes_are_exact():
    torch = pytest.importorskip("torch")
    small, full, receipt = model_mod.build_matched_pair(seed=4811, torch_module=torch)
    assert receipt["shared_initialization_exact"] is True
    assert receipt["small_parameters"] == 8_920_320
    assert receipt["full_parameters"] == 14_163_200
    s = dict(small.named_parameters())
    f = dict(full.named_parameters())
    for name in s:
        if name.endswith("embedding.weight"):
            assert torch.equal(s[name], f[name][:4096])
        else:
            assert torch.equal(s[name], f[name])
    assert f["embedding.weight"].shape[0] == 24576
    assert s["embedding.weight"].shape[0] == 4096


def test_naive_same_seed_would_be_confounded_but_matched_constructor_is_not():
    """Guard the reason this experiment has a dedicated pair constructor."""
    torch = pytest.importorskip("torch")
    from v5_model.core import initialize

    naive_small = initialize(model_mod.spec_for(4096), seed=4811, torch_module=torch)
    naive_full = initialize(model_mod.spec_for(24576), seed=4811, torch_module=torch)
    ns = dict(naive_small.named_parameters())
    nf = dict(naive_full.named_parameters())
    shared_non_embedding = [n for n in ns if not n.endswith("embedding.weight")]
    assert any(not torch.equal(ns[n], nf[n]) for n in shared_non_embedding)

    matched_small, matched_full, _ = model_mod.build_matched_pair(seed=4811, torch_module=torch)
    ms = dict(matched_small.named_parameters())
    mf = dict(matched_full.named_parameters())
    assert all(torch.equal(ms[n], mf[n]) for n in shared_non_embedding)
    assert torch.equal(ms["embedding.weight"], mf["embedding.weight"][:4096])


def test_shared_surface_fixture_is_feasible_at_protocol_acceptance_floor():
    candidates = 1_000
    floor = data_mod.MIN_ACCEPTANCE
    assert int(candidates * 0.60 * floor) >= 60
    assert int(candidates * 0.20 * floor) >= 30


def test_shared_surface_is_deterministic_and_low_id():
    tok = _LowIdTokenizer()
    counts = {"training": 60, "development": 30, "sealed": 30}
    a = data_mod.build_shared_surface(
        tokenizer=tok, seed=2026091303, candidate_worlds_per_family=1000,
        select_counts=counts,
    )
    b = data_mod.build_shared_surface(
        tokenizer=tok, seed=2026091303, candidate_worlds_per_family=1000,
        select_counts=counts,
    )
    assert a["manifest_sha256"] == b["manifest_sha256"]
    assert a["manifest"]["contamination"]["clean"] is True
    assert a["manifest"]["predictive_shortcut_max"] < 0.35
    for split, rows in a["rows"].items():
        assert len(rows) == counts[split] * 6
        for row in rows:
            assert row.prompt_ids and row.answer_ids
            assert min(row.content_ids) >= 4
            assert max(row.content_ids) < 4096
            assert len(row.answer_ids) <= 24
            assert len(row.content_ids) + 2 <= 256


def test_low_id_filter_rejects_out_of_range_tokens():
    class BadTokenizer(_LowIdTokenizer):
        def encode(self, text: str):
            ids = super().encode(text)
            return ids + [4096]

    with pytest.raises(ValueError, match="acceptance"):
        data_mod.build_shared_surface(
            tokenizer=BadTokenizer(), seed=2026091303,
            candidate_worlds_per_family=20,
            select_counts={"training": 2, "development": 1, "sealed": 1},
        )


def test_serialized_token_rows_roundtrip_exactly():
    row = data_mod.TokenRow(
        example_id="fixture-example",
        group_id="fixture-group",
        family="identity",
        split="development",
        template_id="copy-list",
        prompt="Copy these words: flint quark.",
        answer="flint quark",
        prompt_ids=(71, 72, 73),
        answer_ids=(81, 82),
    )
    restored = data_mod.deserialize_rows(data_mod.serialize_rows([row]))
    assert restored == [row]


def test_sealed_is_generated_selected_and_hash_bound_before_training():
    source = (ROOT / "anra_v5" / "cs_transfer_001_data.py").read_text(encoding="utf-8")
    assert 'SPLIT_ORDER = ("sealed", "development", "training")' in source
    assert 'split_hashes[split]' in source
    assert 'contamination_screen(selected_examples)' in source


def test_no_production_scale_claim_is_authorized():
    p = json.loads(PREREG.read_text(encoding="utf-8"))
    ceiling = p["claim_ceiling"]
    assert "No production tokenizer" in ceiling
    assert "500M" in ceiling
    assert p["post_result_actions"]["SUPPORTED_PHYSICAL_CLASS_SPACE"].startswith(
        "replicate the physical contrast"
    )
