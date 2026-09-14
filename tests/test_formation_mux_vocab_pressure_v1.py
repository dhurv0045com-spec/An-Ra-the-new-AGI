"""Regression tests for the prospective VOCAB-PRESSURE-001 adjunct."""
import math

import torch

from tools import formation_mux_001_vocab_pressure_v1 as vp
from v5_experiments import formation_mux_protocol_v5 as proto


def test_denominator_tax_matches_exact_softmax_decomposition():
    logits = torch.zeros(vp.PHYSICAL_VOCAB, dtype=torch.float32)
    metrics = vp.token_pressure_metrics(logits, 300, torch=torch)
    expected_extra_mass = (vp.PHYSICAL_VOCAB - vp.SHARED_VOCAB) / vp.PHYSICAL_VOCAB
    expected_tax = math.log(vp.PHYSICAL_VOCAB / vp.SHARED_VOCAB)
    assert math.isclose(metrics["extra_probability_mass"], expected_extra_mass, rel_tol=1e-6)
    assert math.isclose(metrics["denominator_tax_nats"], expected_tax, rel_tol=1e-6)
    assert metrics["target_margin_vs_best_extra"] == 0.0
    assert metrics["target_rank_full"] == 1.0
    assert metrics["target_rank_shared"] == 1.0


def _focus_rows(rescue: float):
    return {
        str(seed): {"counterfactual_rescue_exact_valid_eos": rescue}
        for seed in proto.SEED_BUNDLES
    }


def test_focus_classification_is_prospective_and_stable():
    material = vp.classify_focus(_focus_rows(0.10))
    assert material["classification"] == "MATERIAL_ENDPOINT_COMPETITION"
    assert material["positive_rescue_seed_bundles"] == 4

    small = vp.classify_focus(_focus_rows(0.0))
    assert small["classification"] == "SMALL_ENDPOINT_COMPETITION"
    assert small["positive_rescue_seed_bundles"] == 0


def test_diagnostic_constants_match_frozen_cs_mech_contract():
    assert vp.SHARED_VOCAB == 4096
    assert vp.PHYSICAL_VOCAB == 24576
    assert vp.FOCUS_ARM == "M2_EXTRA_FROZEN"
    assert vp.MATERIAL_RESCUE_FLOOR == 0.05
