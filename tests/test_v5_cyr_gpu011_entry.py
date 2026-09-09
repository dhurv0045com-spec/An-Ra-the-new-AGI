from __future__ import annotations

from anra_v5.cyr_gpu011_entry import exposure_aware_final_decision


def _receipt(*, g90_update, batch=64, exposure=1.0, measurement=0.95):
    return {
        "g90_confirm_update": g90_update,
        "batch_rows": batch,
        "ark_exposure_fraction": exposure,
        "reasoning_battery_final": {
            "STANDARD": {"complete_exact_with_valid_stop": measurement}
        },
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
