from __future__ import annotations

import pytest

from anra_v5.cyr_gpu011_entry import (
    _compact_ark_render_batch,
    exposure_aware_final_decision,
    resolve_from_calibrations,
)
from v5_experiments import cyr_gpu011 as core


def _receipt(*, g90_update, batch=64, exposure=1.0, measurement=0.95):
    return {
        "g90_confirm_update": g90_update,
        "batch_rows": batch,
        "ark_exposure_fraction": exposure,
        "reasoning_battery_final": {
            "STANDARD": {"complete_exact_with_valid_stop": measurement}
        },
    }


def _cal(regime: str, batch: int, ups: float, eps: float = 100.0):
    return {
        "status": "PASS", "regime": regime, "batch_rows": batch,
        "training_updates_per_sec": ups, "training_real_tokens_per_sec": ups * batch * 8,
        "semantic_rows_per_sec": ups * batch, "generation_examples_per_sec": eps,
    }


def test_production_null_is_not_called_divergence_when_underexposed() -> None:
    compact = _receipt(g90_update=12_000, batch=64, exposure=12_000 / 18_000)
    production = _receipt(g90_update=None, batch=32, exposure=0.40, measurement=0.05)
    decision = exposure_aware_final_decision(
        compact=compact, production_primary=production, production_replication=None
    )
    assert decision["verdict"] == "PRODUCTION_UNDEREXPOSED_RELATIVE_TO_COMPACT_G90"
    assert decision["exposure_matched_for_divergence"] is False


def test_exposure_matched_compact_vs_production_null_can_be_called_divergence() -> None:
    compact = _receipt(g90_update=12_000, batch=64, exposure=12_000 / 18_000)
    production = _receipt(g90_update=None, batch=32, exposure=0.80, measurement=0.10)
    decision = exposure_aware_final_decision(
        compact=compact, production_primary=production, production_replication=None
    )
    assert decision["verdict"] == "EXPOSURE_MATCHED_BRIDGE_DIVERGENCE_COMPACT_G90_PRODUCTION_NO_G90"
    assert decision["exposure_matched_for_divergence"] is True


def test_controller_g90_without_measurement_support_is_not_qualified() -> None:
    compact = _receipt(g90_update=8_000, batch=64, exposure=0.60, measurement=0.55)
    production = _receipt(g90_update=None, batch=64, exposure=0.60, measurement=0.10)
    decision = exposure_aware_final_decision(
        compact=compact, production_primary=production, production_replication=None
    )
    assert decision["compact_controller_and_measurement_g90"] is False
    assert decision["verdict"] == "NO_G90_WITH_INCOMPLETE_EXPOSURE"


def test_replicated_label_requires_two_measurement_supported_production_g90s() -> None:
    compact = _receipt(g90_update=10_000, batch=64, exposure=0.70, measurement=0.96)
    p1 = _receipt(g90_update=11_000, batch=64, exposure=0.75, measurement=0.94)
    p2_bad = _receipt(g90_update=12_000, batch=64, exposure=0.80, measurement=0.70)
    p2_good = _receipt(g90_update=12_000, batch=64, exposure=0.80, measurement=0.93)
    assert exposure_aware_final_decision(
        compact=compact, production_primary=p1, production_replication=p2_bad
    )["verdict"] == "PRODUCTION_REPRESENTATION_G90_SINGLE_SEED_DEVELOPMENT"
    assert exposure_aware_final_decision(
        compact=compact, production_primary=p1, production_replication=p2_good
    )["verdict"] == "PRODUCTION_REPRESENTATION_G90_REPLICATED_DEVELOPMENT"


def test_resolver_targets_same_semantic_box_for_every_batch_size() -> None:
    calibrations = {
        "COMPACT_B64": _cal("COMPACT", 64, 20.0),
        "PRODUCTION_B64": _cal("PRODUCTION", 64, 0.40),
        "PRODUCTION_B32": _cal("PRODUCTION", 32, 1.30),
        "PRODUCTION_B16": _cal("PRODUCTION", 16, 2.20),
    }
    resolved = resolve_from_calibrations(calibrations)
    assert resolved["compact_target_updates"] == 18_000
    assert resolved["compact_target_updates"] * resolved["compact_batch_rows"] == core.CYR11_ARK_MAX_ROW_PRESENTATIONS
    target = resolved["production_target_updates"] * resolved["production_batch_rows"]
    assert target == core.CYR11_ARK_MAX_ROW_PRESENTATIONS
    if resolved["production_batch_rows"] == 32:
        assert resolved["production_target_updates"] == 36_000
    if resolved["production_batch_rows"] == 16:
        assert resolved["production_target_updates"] == 72_000


def test_resolver_progresses_even_when_full_production_exposure_will_not_fit() -> None:
    calibrations = {
        "COMPACT_B64": _cal("COMPACT", 64, 10.0),
        "PRODUCTION_B16": _cal("PRODUCTION", 16, 0.10, eps=20.0),
    }
    resolved = resolve_from_calibrations(calibrations)
    assert resolved["production_available"] is True
    assert resolved["production_target_updates"] == 72_000
    assert 0 <= resolved["production_projected_ark_exposure_fraction"] < 1.0


def test_compact_renderer_matches_arkenstone_answer_prefix_bos_semantics() -> None:
    torch = pytest.importorskip("torch")
    tok = core.CompactCharTokenizer()
    row = {"prompt": "12 + 13 = ", "answer": "25"}
    tokens, segments, eligible, counted = _compact_ark_render_batch(
        tok, [row], torch=torch, device=torch.device("cpu"), special=tok.special
    )
    expected = [tok.bos_id, *tok.encode(row["prompt"]), tok.bos_id, *tok.encode(row["answer"]), tok.eos_id]
    assert tokens[0, : len(expected)].tolist() == expected
    prompt_len = 1 + len(tok.encode(row["prompt"]))
    assert eligible[0, :prompt_len].tolist() == [False] * prompt_len
    assert eligible[0, prompt_len:len(expected)].tolist() == [True] * (len(expected) - prompt_len)
    assert counted["ark_answer_prefix_bos_supervised"] is True
    assert int((segments >= 0).sum()) == len(expected)
