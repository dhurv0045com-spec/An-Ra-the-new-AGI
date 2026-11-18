"""Acceptance tests for Avendi V8-Y2 computable odds, routing decisions, P3/P4/P5/P6, and campaign CLI."""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pytest

from anra_v5.elendril.avendi.correction import (
    build_p5_conditional_families,
    build_p5_dose_experiment_spec,
    build_p6_correction_blocks,
    run_p6_bounded_correction_study,
)
from anra_v5.elendril.avendi.metrics import (
    ABSTAIN_TOKEN_ID,
    AP_TOKEN_IDS,
    EOS_TOKEN_ID,
    bootstrap_paired_family_contrasts,
    evaluate_frozen_routing_decision,
    score_crossover_family,
    score_single_case_prediction,
)
from anra_v5.elendril.avendi.objectives import (
    audit_pair_margin_gradient_sign_reversal,
    run_p3_job,
)
from anra_v5.elendril.avendi.transition import (
    AvendiSymbolicTransitionModel,
    run_p4_interchange_protocol,
    select_matched_continuous_width,
)
from anra_v5.elendril.avendi.worlds import build_avendi_suite
from tools.analyze_avendi import analyze_avendi_evidence, render_avendi_results_markdown
from tools.avendi_campaign import run_avendi_campaign
from tools.build_avendi_notebook import build_avendi_notebook


def test_computable_odds_latent_flip_and_degenerate_bootstrap_variance() -> None:
    suite = build_avendi_suite(scale_factor=64)
    fam = suite["splits"]["dev"]["three_hop_families"][0]
    cmap = fam.cell_map()
    id_g = AP_TOKEN_IDS[fam.endpoint_g]
    id_h = AP_TOKEN_IDS[fam.endpoint_h]

    # Case 1: Positive interaction (+10 - 1 - 1 - 1)/4 = +1.75 coexisting with 3 negative signed margins!
    # Cells: 00=(qA,orig)->g, 01=(qA,swap)->h, 10=(qB,orig)->h, 11=(qB,swap)->g
    case_scores = {}
    for ck, s_target in (("00", 10.0), ("01", -1.0), ("10", -1.0), ("11", -1.0)):
        cell = cmap[ck]
        logits = np.zeros(260, dtype=np.float32)
        id_gold = AP_TOKEN_IDS[cell.expected_answer]
        id_riv = AP_TOKEN_IDS[cell.rival_answer]
        logits[id_gold] = s_target
        logits[id_riv] = 0.0
        # Emit repeated abstention ('!')
        case_scores[cell.case_id] = score_single_case_prediction(cell, raw_tokens=[ABSTAIN_TOKEN_ID, EOS_TOKEN_ID], first_logits=logits)

    f_res = score_crossover_family(fam, case_scores)
    assert f_res["interaction"] > 0.0
    assert f_res["min_signed_margin"] < 0.0
    assert f_res["latent_flip"] is False
    assert f_res["positive_interaction_but_not_latent_flip"] is True

    # Case 2: All 4 signed margins > 0 -> latent_flip is True even when greedy output repeats '!'
    for ck in ("00", "01", "10", "11"):
        cell = cmap[ck]
        logits = np.zeros(260, dtype=np.float32)
        logits[AP_TOKEN_IDS[cell.expected_answer]] = 2.0
        logits[AP_TOKEN_IDS[cell.rival_answer]] = 0.5
        logits[ABSTAIN_TOKEN_ID] = 5.0  # '!' wins greedy decoding
        case_scores[cell.case_id] = score_single_case_prediction(cell, raw_tokens=[ABSTAIN_TOKEN_ID, EOS_TOKEN_ID], first_logits=logits)

    f_res2 = score_crossover_family(fam, case_scores)
    assert f_res2["latent_flip"] is True
    assert f_res2["all_four_same_tokens"] is True
    assert f_res2["repeated_greedy_with_latent_flip"] is True

    # Case 3: Degenerate variance across 3 seeds (+0.25 identical difference in every seed)
    t_seed = {
        s: {f"f{i}": {"hops": 1, "all_four_raw_correct": 1.0 if i == 0 else 0.0} for i in range(4)}
        for s in (7, 19, 41)
    }
    c_seed = {
        s: {f"f{i}": {"hops": 1, "all_four_raw_correct": 0.0} for i in range(4)}
        for s in (7, 19, 41)
    }
    boot = bootstrap_paired_family_contrasts(t_seed, c_seed, n_resamples=100)
    assert boot["mean_difference"] == pytest.approx(0.25)
    assert boot["paired_seed_summary"]["variance_status"] == "DEGENERATE_ZERO_VARIANCE"


def test_p2_stage_routing_decisions() -> None:
    def _mock_arm(lit: float, one_raw: float, three_raw: float, one_flip: float = 0.0, three_flip: float = 0.0):
        return {
            s: {
                "literal_copy": {"raw_exact_accuracy": lit},
                "one_hop": {"all_four_raw_rate": one_raw, "latent_flip_rate": one_flip},
                "three_hop": {"all_four_raw_rate": three_raw, "latent_flip_rate": three_flip},
            }
            for s in (7, 19, 41)
        }

    # 1. Literal pass (0.98), 1-hop fail (0.20) -> route to P3_CONTROLLED_ACQUISITION
    r1 = evaluate_frozen_routing_decision({"F00": _mock_arm(0.98, 0.20, 0.05), "F11": _mock_arm(0.96, 0.05, 0.01)})
    assert r1["selected_anchor_arm"] == "F00"
    assert r1["first_failing_dependency"] == "ONE_HOP_TWO_KEY_SELECTION"
    assert r1["next_admitted_stage"] == "P3_CONTROLLED_ACQUISITION"

    # 2. 1-hop pass (0.85), 3-hop fail (0.15) -> route to P4_TRANSITION_INTERCHANGE
    r2 = evaluate_frozen_routing_decision({"F00": _mock_arm(0.98, 0.85, 0.15), "F11": _mock_arm(0.98, 0.40, 0.05)})
    assert r2["first_failing_dependency"] == "THREE_HOP_COMPOSITION"
    assert r2["next_admitted_stage"] == "P4_TRANSITION_INTERCHANGE"

    # 3. Within-A-P sign patterns pass (0.85) while raw fails (0.10) -> DIAGNOSTIC status
    r3 = evaluate_frozen_routing_decision({"F00": _mock_arm(0.98, 0.10, 0.05, one_flip=0.85), "F11": _mock_arm(0.98, 0.05, 0.0)})
    assert r3["status"] == "DIAGNOSTIC"


def test_p3_gradient_sign_and_fresh_process_resume_and_p4_p5_p6(tmp_path: Path) -> None:
    # 1. P3 gradient sign negative control
    neg = audit_pair_margin_gradient_sign_reversal()
    assert neg["status"] == "PASS"

    # 2. P3 tiny lineage training + fresh-instance resume
    suite = build_avendi_suite(scale_factor=64)
    src = "641633a060c2ac84215e149de7dc3d0f4987f748"
    p3_dir = tmp_path / "p3_resume_test"
    rec1 = run_p3_job(
        arm="CE_PAIR_MARGIN",
        seed=7,
        suite=suite,
        output_dir=p3_dir,
        source_commit=src,
        cpu_canary=True,
        canary_updates=2,
        max_steps_this_call=1,
    )
    assert rec1["global_update"] == 1
    assert rec1["update1_gradient_audit"]["pair_margin_logits_grad_norm"] > 0.0

    # Resume from update 1 -> update 2 on a fresh trainer instance
    rec2 = run_p3_job(
        arm="CE_PAIR_MARGIN",
        seed=7,
        suite=suite,
        output_dir=p3_dir,
        source_commit=src,
        cpu_canary=True,
        canary_updates=2,
        resume=True,
    )
    assert rec2["global_update"] == 2
    assert rec2["checkpoint_sha256"] != rec1["checkpoint_sha256"]

    # 3. P4 width search and hop-1 state interchange protocol (including edited terminal suffix)
    w_search = select_matched_continuous_width()
    assert w_search["within_10pct_tolerance"] is True
    assert w_search["selected_continuous_width"] in (48, 56, 64, 72, 80)

    m_p4 = AvendiSymbolicTransitionModel(seed=7)
    inter = run_p4_interchange_protocol(m_p4, suite["splits"]["dev"]["three_hop_families"][:4], seed=7)
    assert inter["identity_swap_drop"] == pytest.approx(0.0)
    assert "donor_unconditional_accuracy_edited_terminal_suffix" in inter

    # 4. P5 6-condition generator & P6 bounded correction block isolation + no-observation chance control
    p5 = build_p5_conditional_families(families_count=4)
    assert p5["cases_total"] == 4 * 2 * 6
    dose = build_p5_dose_experiment_spec()
    assert dose["replaced_slots_in_treatment"] == 4

    p6_blocks = build_p6_correction_blocks(blocks_count=8)
    p6_res = run_p6_bounded_correction_study(p6_blocks, seed=7)
    assert p6_res["block_isolation_verified"] is True
    assert p6_res["identical_pre_feedback_observation_pairs"] is True
    assert p6_res["opposite_target_pairs"] is True
    assert p6_res["no_observation_control_exact_chance"] is True
    assert p6_res["memory_reset_dependence_verified"] is True


def test_campaign_cli_admission_guard_and_notebook_and_analyzer(tmp_path: Path) -> None:
    src = "641633a060c2ac84215e149de7dc3d0f4987f748"
    # Conditional stage must refuse missing --admission receipt
    with pytest.raises(ValueError, match="--admission"):
        run_avendi_campaign(
            argparse.Namespace(
                phase="acquisition",
                source=src,
                output=tmp_path / "no_adm",
                checkpoint_root=None,
                evidence=None,
                admission=None,
                minutes=120,
                owner_enabled=True,
                cpu_canary=True,
                canary_updates=1,
                scale_factor=64,
                skip_source_check=True,
                expected_spec_override=None,
            )
        )

    # Fixture phase runs cleanly and exports verified bundles
    fix_out = tmp_path / "fixture_out"
    res = run_avendi_campaign(
        argparse.Namespace(
            phase="fixture",
            source=src,
            output=fix_out,
            checkpoint_root=None,
            evidence=None,
            admission=None,
            minutes=30,
            owner_enabled=False,
            cpu_canary=True,
            canary_updates=1,
            scale_factor=64,
            skip_source_check=True,
            expected_spec_override=None,
        )
    )
    assert res["status"] == "COMPLETE"
    assert res["stage_statuses"]["P0_instrument"] == "QUALIFIED"
    assert (fix_out / "exports" / "AVENDI_V8_Y2_SUMMARY.zip").is_file()
    assert (fix_out / "exports" / "AVENDI_V8_Y2_EVIDENCE.zip").is_file()

    # Analyzer verifies exported evidence zip and renders RESULTS.md
    metrics = analyze_avendi_evidence(fix_out / "exports" / "AVENDI_V8_Y2_EVIDENCE.zip")
    assert metrics["verification_rows"]["byte_integrity"] == "PASS"
    assert metrics["verification_rows"]["oracle_fixture_correctness"] == "PASS"
    md = render_avendi_results_markdown(metrics)
    assert "Six-Row Independent Verification Matrix" in md

    # Notebook builder compiles all code cells cleanly
    nb = build_avendi_notebook(src)
    for idx, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"<avendi-cell-{idx}>", "exec")
