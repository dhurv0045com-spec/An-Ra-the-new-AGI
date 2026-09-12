"""Tests for the next-Core compute/parameter calculator (pure CPU, no torch)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from tools.next_core_compute_model import (
    V5A,
    Geometry,
    memory_bytes,
    parameter_receipt,
    scale_ladder,
    training_flops_per_token,
    validate_against_live_v5,
)


def test_v5a_parameter_count_is_exact():
    assert parameter_receipt(V5A)["total"] == 250_216_960


def test_v5a_receipt_components():
    r = parameter_receipt(V5A)
    assert r["embedding"] == 24_576 * 896 == 22_020_096
    assert r["attention_per_layer"] == 2 * 896 * 896 + 2 * 896 * 448 == 2_408_448
    assert r["ffn_per_layer"] == 3 * 896 * 2_368 == 6_365_184
    assert r["block_norms_per_layer"] == 2 * 896 == 1_792
    assert r["qk_norms_per_layer"] == (14 + 7) * 64 == 1_344
    assert r["block_total"] == 8_776_768
    assert r["all_blocks"] == 26 * 8_776_768 == 228_195_968
    assert r["final_norm"] == 896


def test_untied_output_head_doubles_embedding():
    untied = Geometry(**{**V5A.__dict__, "tied_embeddings": False})
    r_tied = parameter_receipt(V5A)
    r_untied = parameter_receipt(untied)
    assert r_untied["output_head"] == 24_576 * 896
    assert r_untied["total"] == r_tied["total"] + 24_576 * 896


def test_invalid_geometry_rejected():
    with pytest.raises(ValueError):
        Geometry(vocabulary_size=100, width=100, layers=2, query_heads=4,
                 kv_heads=2, head_dimension=16, ffn_width=64, context_length=32)


def test_memory_layout_uses_adopted_precision_contract():
    m = memory_bytes(V5A)
    p = parameter_receipt(V5A)["total"]
    assert m["fp32_master_params"] == 4 * p
    assert m["fp32_adam_moments"] == 8 * p
    assert m["optimizer_state_bytes"] == 8 * p
    assert m["checkpoint_bytes_params_plus_moments"] == 12 * p


def test_flops_are_positive_and_training_dominates_forward():
    f = training_flops_per_token(V5A)
    assert f["forward_flops_per_token"] > 0
    assert f["training_flops_per_token"] == 3 * f["forward_flops_per_token"]


def test_ladder_rungs_are_honest():
    names = {rung["name"]: rung["parameters"] for rung in scale_ladder()}
    assert names["byte-6M"] < 7_000_000
    assert 35_000_000 < names["byte-38M"] < 40_000_000
    assert 90_000_000 < names["~97M"] < 100_000_000
    assert names["~250M (V5-A)"] == 250_216_960
    assert 490_000_000 < names["~500M"] < 505_000_000
    assert all(rung["tokens_at_chinchilla_20x"] > 0 for rung in scale_ladder())


def test_live_v5_validation_matches_documented():
    result = validate_against_live_v5()
    assert result["match_documented"] is True
    assert result["calculator_total"] == 250_216_960
