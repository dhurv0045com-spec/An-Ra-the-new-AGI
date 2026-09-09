"""CYR-GPU-008 final hardware-adaptive contract.

008 supersedes preregistered-but-unexecuted 007 after a final preexecution code
review found that 007's compatibility monkeypatch could recursively call its
own final-decision wrapper. No 007 Colab execution occurred. 008 captures the
immutable 006 decision function before any compatibility patch and also makes
proxy priority explicit: preserve a non-TINY scale test before falling back to
TINY, then maximize causal scope within that scale class.
"""
from __future__ import annotations

from typing import Any, Mapping

from v5_experiments import cyr_gpu006_final as base6
from v5_experiments import cyr_gpu007 as tiering

# Capture before the runner temporarily replaces base6.final_decision.
_BASE6_FINAL_DECISION = base6.final_decision

CYR8_ID = "CYR-GPU-008"
CYR8_PARENT_SEEDS = tiering.CYR7_PARENT_SEEDS
CYR8_ARMS = tiering.CYR7_ARMS
CYR8_WALL_TARGET_MINUTES = tiering.CYR7_WALL_TARGET_MINUTES
CYR8_WALL_HARD_MINUTES = tiering.CYR7_WALL_HARD_MINUTES
CYR8_PACKAGING_RESERVE_MINUTES = tiering.CYR7_PACKAGING_RESERVE_MINUTES
TIER_SPECS = tiering.TIER_SPECS
ARKENSTONE_AUDITED_SHA = tiering.ARKENSTONE_AUDITED_SHA
ARKENSTONE_DISCOVERY_V6_BUNDLE_SHA256 = tiering.ARKENSTONE_DISCOVERY_V6_BUNDLE_SHA256
CYR8_TRANSFER_CANDIDATE = tiering.CYR7_TRANSFER_CANDIDATE
CYR8_TRANSFER_COMPARATOR = tiering.CYR7_TRANSFER_COMPARATOR
CYR8_TRANSFER_MODE = tiering.CYR7_TRANSFER_MODE

proxy_registry = tiering.proxy_registry
assert_proxy_in_registry = tiering.assert_proxy_in_registry
render_t2_worlds = tiering.render_t2_worlds
build_data_manifest = tiering.build_data_manifest
assert_manifest_sha = tiering.assert_manifest_sha
commutation_audit = tiering.commutation_audit
build_future_stream = tiering.build_future_stream
assert_future_tail_equality = tiering.assert_future_tail_equality
fixed_time_switch_point = tiering.fixed_time_switch_point
lr_for_token = tiering.lr_for_token
arm_order = tiering.arm_order
decide_retention = tiering.decide_retention


def _normalize(resolved: Mapping[str, Any]) -> dict[str, Any]:
    body = dict(resolved)
    body["schema"] = "anra-cyr-gpu008-resolved/v1"
    body["experiment"] = CYR8_ID
    return body


def resolve_from_calibrations(
    calibrations: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Hardware-only resolution with a non-TINY scientific-scale floor.

    First search every preregistered tier on MIDI/MICRO/RESEARCH_SMALL. Only if
    none fits do we search TINY. This avoids choosing a 1.6M-parameter full
    campaign when the same GPU could provide a more relevant replicated
    retention test on a materially larger real-V5 proxy.
    """
    for tier in TIER_SPECS:
        for name in ("MIDI", "MICRO", "RESEARCH_SMALL"):
            receipt = calibrations.get(name)
            if receipt:
                resolved = tiering._fit(name=name, receipt=receipt, tier=tier)
                if resolved is not None:
                    return _normalize(resolved)
    tiny = calibrations.get("TINY")
    if tiny:
        for tier in TIER_SPECS:
            resolved = tiering._fit(name="TINY", receipt=tiny, tier=tier)
            if resolved is not None:
                return _normalize(resolved)
    diagnostics = {
        name: {
            "status": receipt.get("status"),
            "training_real_tokens_per_sec": receipt.get("training_real_tokens_per_sec"),
            "generation_examples_per_sec": receipt.get("generation_examples_per_sec"),
            "peak_vram_gb": receipt.get("peak_vram_gb"),
        }
        for name, receipt in calibrations.items()
    }
    raise ValueError(
        "assigned GPU cannot afford even the frozen two-parent matched-fork minimum "
        f"inside 170 minutes; calibration={diagnostics}"
    )


def validate_resolved(resolved: Mapping[str, Any]) -> dict[str, Any]:
    body = tiering.validate_resolved(resolved)
    if resolved.get("experiment") != CYR8_ID:
        raise ValueError("CYR-GPU-008 resolved identity drift")
    if resolved.get("schema") != "anra-cyr-gpu008-resolved/v1":
        raise ValueError("CYR-GPU-008 resolved schema drift")
    return body


def final_decision(
    parent_runs: list[Mapping[str, Any]],
    *, transfer: Mapping[str, Any] | None,
    resolved: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply 006 replicated science, then enforce the 008 claim ceiling."""
    result = _BASE6_FINAL_DECISION(parent_runs, transfer=transfer)
    possible = bool(resolved.get("research_candidate_possible", False))
    if not possible:
        result["research_candidate"] = False
    result["schema"] = "anra-cyr-gpu008-decision/v1"
    result["experiment"] = CYR8_ID
    result["tier"] = resolved["tier"]
    result["proxy"] = resolved["proxy"]
    result["claim_ceiling"] = resolved["claim_ceiling"]
    result["research_candidate_possible"] = possible
    result["production_promotion_authorized"] = False
    return result
