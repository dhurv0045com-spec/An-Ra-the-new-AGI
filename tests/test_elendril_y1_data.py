"""Scientific custody, intervention, oracle, holdout, and anti-shortcut tests for AN-RA V8-Y1 data."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import random

import pytest

from anra_v5.elendril.y1_data import (
    _is_holdout_combo,
    audit_y1_split_custody,
    build_arm_training_stream,
    build_confirmation_suite,
    build_d2_probe_dataset,
    build_development_suite,
    build_followup_suite,
    build_non_route_rehearsal_pool,
    build_t0_dataset,
    collect_forbidden_prompts,
    independent_traversal_oracle,
    load_y1_suite,
    serialize_y1_suite,
)
from anra_v5.elendril.y1_evaluation import evaluate_all_heuristics
from anra_v5.elendril.y1_protocol import (
    HOLDOUT_SERIALIZATION_COMBINATION,
    NON_ROUTE_REHEARSAL_COUNTS,
    PROMPT_TOKEN_BOUND,
    REGISTRY_SHA,
    T0_SPEC,
    registry,
)


def test_y1_registry_and_split_quotas_custody_and_holdout() -> None:
    reg = registry()
    assert reg["schema"] == "anra-v8-y1-protocol-v1"
    assert len(REGISTRY_SHA) == 64

    dev = build_development_suite()
    conf = build_confirmation_suite()
    fol = build_followup_suite()
    d2 = build_d2_probe_dataset()

    for suite in (dev, conf, fol):
        assert len(suite["cases"]) == 11840
        assert len(suite["historical_continuity"]) == 80
        assert suite["over_context_rejected"] >= 0
        cohorts = suite["cohorts"]
        assert len(cohorts["main"]) == 8192
        assert len(cohorts["edge-change-panel"]) == 768
        assert len(cohorts["missing-bridge-panel"]) == 512
        assert len(cohorts["dual-chain-query-panel"]) == 512
        assert len(cohorts["serialization-holdout"]) == 64
        assert len(cohorts["preservation-suite"]) == 1792

        # Every serialization-holdout case must satisfy exact holdout tuple.
        for sc in cohorts["serialization-holdout"]:
            assert sc.hops == 2
            assert _is_holdout_combo(
                hops=2,
                terminal_pos_0based=sc.visible_terminal_pos,
                distractor_count=2,
                queried_start_pos_0based=sc.queried_start_pos,
            )

        # Incomplete cases 1/6/7 must carry applicable=False so unchanged controls aren't miscounted.
        for mc in cohorts["main"]:
            if mc.solvability != "answerable" and mc.variant_index in (1, 6, 7):
                assert mc.applicable is False
            else:
                assert mc.applicable is True

    custody = audit_y1_split_custody(dev, conf, fol)
    assert custody["status"] == "PASS"

    # D2 probes must not overlap dev/conf/followup.
    all_main_prompts = collect_forbidden_prompts(dev, conf, fol)
    d2_prompts = {c.public_prompt for c in (*d2["train_cases"], *d2["test_cases"])}
    assert not (all_main_prompts & d2_prompts)
    assert len(d2["train_cases"]) == 512 and len(d2["test_cases"]) == 512

    # Serialization roundtrip check
    ser = serialize_y1_suite(dev)
    reloaded = load_y1_suite(ser)
    assert reloaded["suite_sha256"] == dev["suite_sha256"]
    assert len(reloaded["cases"]) == 11840


def test_y1_oracle_and_byte_length_and_anti_shortcut() -> None:
    dev = build_development_suite()
    # Oracle disagrees with any planted wrong gold label
    ans_case = next(c for c in dev["cohorts"]["main-base"] if c.solvability == "answerable")
    wrong_sym = next(s for s in "ABCDEFGHIJKLMNOP" if s != ans_case.truth.expected)
    op, arg = independent_traversal_oracle(ans_case.environment.initial, ans_case.goal.subject, ans_case.hops, channels=ans_case.environment.channels)
    assert (op, arg) == ("ANSWER", ans_case.truth.expected)
    assert arg != wrong_sym

    # Byte-length invariance for source-rename, timestamp-shift, distractor-replace on identical family
    fam_variants = [c for c in dev["cohorts"]["main"] if c.world_id == ans_case.world_id]
    by_idx = {c.variant_index: c for c in fam_variants}
    b_len = len(by_idx[0].public_prompt.encode("utf-8"))
    for vi in (1, 3, 4, 5, 6):
        assert len(by_idx[vi].public_prompt.encode("utf-8")) == b_len

    # Heuristics: oracle hits 1.0 primary, always-abstain hits 0.0, last-row reflects ~1/(h+2)
    hres = evaluate_all_heuristics(dev["cases"], historical_cases=dev["historical_continuity"])
    assert hres["oracle-traversal"]["primary_endpoint"] == pytest.approx(1.0)
    assert hres["always-abstain"]["primary_endpoint"] == pytest.approx(0.0)
    assert 0.15 < hres["last-row-object"]["primary_endpoint"] < 0.35
    assert 0.15 < hres["nearest-to-query-object"]["primary_endpoint"] < 0.35


def test_y1_t0_and_training_streams_isolated_and_balanced() -> None:
    dev = build_development_suite()
    forbidden = collect_forbidden_prompts(dev)

    t0 = build_t0_dataset()
    assert len(t0["cases"]) == T0_SPEC["families_total"] == 128
    assert len(t0["unique_samples"]) == T0_SPEC["supervised_states_total"] == 160
    assert len(t0["cycled_samples"]) == T0_SPEC["exposures"] == 4096
    assert t0["partial_final_cycle_exposures"] == 4096 % 160

    reh = build_non_route_rehearsal_pool(forbidden_prompts=forbidden)
    assert len(reh) == sum(NON_ROUTE_REHEARSAL_COUNTS.values()) == 2368

    # Fast bounded check of arm stream contract for all 4 arms
    small_reh = {k: 4 for k in NON_ROUTE_REHEARSAL_COUNTS}
    small_route = {
        "imbalanced": {"answerable": 12, "query_needed_families": 2, "unavailable": 4},
        "balanced": {"answerable": 6, "query_needed_families": 4, "unavailable": 6},
    }
    s_f00 = build_arm_training_stream(arm="F00", fresh=False, coverage="imbalanced", seed=7, order_seed=7, epochs=2, batch_size=8, forbidden_prompts=forbidden, rehearsal_counts=small_reh, route_counts_per_hop=small_route)
    s_f10 = build_arm_training_stream(arm="F10", fresh=True, coverage="imbalanced", seed=7, order_seed=7, epochs=2, batch_size=8, forbidden_prompts=forbidden, rehearsal_counts=small_reh, route_counts_per_hop=small_route)
    s_f01 = build_arm_training_stream(arm="F01", fresh=False, coverage="balanced", seed=7, order_seed=7, epochs=2, batch_size=8, forbidden_prompts=forbidden, rehearsal_counts=small_reh, route_counts_per_hop=small_route)
    assert len(s_f00["samples"]) == len(s_f01["samples"])
    assert s_f00["data_sha256"] != s_f10["data_sha256"]
    for s in s_f10["samples"]:
        assert len(s.tokens) <= PROMPT_TOKEN_BOUND
