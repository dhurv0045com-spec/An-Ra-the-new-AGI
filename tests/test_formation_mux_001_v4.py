"""Science-S4 regression tests for FORMATION-MUX-001 sealed custody."""

from __future__ import annotations

import copy
import hashlib
import json

import pytest

pytest.importorskip("torch")

from anra_v5 import formation_mux_train_v4 as train
from v5_experiments import formation_mux_protocol_v4 as proto
from v5_experiments.formation_mux_data import FAMILIES
from v5_experiments.formation_mux_surface_v4 import (
    build_public_surface,
    regenerate_sealed_rows,
    validate_public_surface,
)


class _Identity:
    vocabulary_size = 24576
    special_token_ids = {"pad": 0, "unk": 1, "bos": 2, "eos": 3}
    artifact_sha256 = "a" * 64


class _Tokenizer:
    identity = _Identity()

    @staticmethod
    def encode(text: str) -> list[int]:
        return [1000 + (sum(map(ord, word)) % 2000) for word in text.split()]


def _canonical(value) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def test_public_manifest_contains_no_raw_sealed_rows():
    manifest = build_public_surface(seed=73011, tokenizer=_Tokenizer())
    assert set(manifest["splits"]) == {"training", "development"}
    assert "sealed" not in manifest["splits"]
    assert manifest["sealed_rows_persisted"] is False
    assert len(manifest["splits"]["training"]) == 3600
    assert len(manifest["splits"]["development"]) == 480
    assert set(manifest["sealed_commitments"]) == set(proto.EXPERIMENTS)
    assert all(len(v) == 64 for v in manifest["sealed_commitments"].values())


def test_public_manifest_rejects_even_hash_valid_raw_sealed_injection():
    manifest = build_public_surface(seed=73011, tokenizer=_Tokenizer())
    bad = copy.deepcopy(manifest)
    bad["splits"]["sealed"] = [{"family": "identity", "prompt_ids": [260], "answer_ids": [300]}]
    body = {k: v for k, v in bad.items() if k != "sha256"}
    bad["sha256"] = hashlib.sha256(_canonical(body)).hexdigest()
    with pytest.raises(RuntimeError, match="SEALED_FIREWALL_BREACH"):
        validate_public_surface(bad)


def test_worker_binding_exposes_zero_sealed_examples():
    manifest = build_public_surface(seed=73011, tokenizer=_Tokenizer())
    worker = train._worker_surface(manifest)
    assert worker["splits"]["sealed"] == []
    assert len(worker["splits"]["training"]) == 3600
    assert len(worker["splits"]["development"]) == 480


def test_sealed_regeneration_matches_commitment_without_mutating_public_manifest():
    manifest = build_public_surface(seed=73011, tokenizer=_Tokenizer())
    before = hashlib.sha256(_canonical(manifest)).hexdigest()
    rows = regenerate_sealed_rows(
        public_manifest=manifest,
        tokenizer=_Tokenizer(),
        experiment=proto.EXPERIMENT_A,
    )
    after = hashlib.sha256(_canonical(manifest)).hexdigest()
    assert before == after
    assert len(rows) == 720
    for family in FAMILIES:
        assert sum(1 for row in rows if row["family"] == family) == 120
    assert "sealed" not in manifest["splits"]


def test_protocol_hash_binds_firewall_and_all_execution_cadences():
    a = proto.protocol_payload(proto.EXPERIMENT_A)
    b = proto.protocol_payload(proto.EXPERIMENT_B)
    assert "raw sealed rows" in a["sealed_firewall"]
    assert a["surface_counts_per_family"] == {
        "training": 600, "development": 80, "sealed": 120
    }
    assert a["A"] == {
        "updates": 2000,
        "eligible_from_update": 600,
        "eval_every_updates": 100,
        "checkpoint_every_updates": 250,
    }
    assert b["B"] == {
        "processed_token_budget": 500000,
        "eligible_from_tokens": 150000,
        "eval_every_tokens": 25000,
        "checkpoint_every_tokens": 50000,
        "exposure_mismatch_tolerance": 0.001,
    }
    assert len(proto.protocol_sha(proto.EXPERIMENT_A)) == 64
    assert len(proto.protocol_sha(proto.EXPERIMENT_B)) == 64
    assert proto.protocol_sha(proto.EXPERIMENT_A) != proto.protocol_sha(proto.EXPERIMENT_B)
