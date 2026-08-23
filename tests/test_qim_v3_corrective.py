"""Evidence-repair contract tests: QIM-v3 greedy bug fix + honest metrics.

Each test targets one recorded defect of the first clean-replication pass:
the stale-query greedy bug, literal-zero Monte Carlo p-values, the
trajectory metric mislabeled as query lift, collapsed hash semantics,
the inaccurate extraction vocab-disjointness claim, and history rewriting
in the evidence chain.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]


def _src(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


# ------------------- P0/P1/P6: greedy targets use their OWN query ----------

def _synthetic_group():
    """Three obviously different entities A/B/C so prompt mixups are loud."""
    return {
        "format": "prose",
        "displayed_facts": [
            {"entity": "alpha", "code": "AAA-111",
             "line": "Alpha bears tag AAA-111."},
            {"entity": "bravo", "code": "BBB-222",
             "line": "Bravo bears tag BBB-222."},
            {"entity": "charlie", "code": "CCC-333",
             "line": "Charlie bears tag CCC-333."},
        ],
    }


def test_every_greedy_target_builds_its_own_query_prompt() -> None:
    from connector.experiments.query_influence_v3 import build_query_prompt
    g = _synthetic_group()
    prompts = [build_query_prompt(g, i) for i in range(3)]
    # each prompt carries ITS OWN query...
    assert "Return the tag of Alpha." in prompts[0]
    assert "Return the tag of Bravo." in prompts[1]
    assert "Return the tag of Charlie." in prompts[2]
    # ...and NO other member's query
    for i in range(3):
        for j in range(3):
            if i != j:
                other = f"Return the tag of {g['displayed_facts'][j]['entity'].capitalize()}."
                assert other not in prompts[i]
    # order must be A, B, C — never C, C, C
    queries = [p.splitlines()[-2] for p in prompts]
    assert queries[0].endswith("Alpha.") and queries[1].endswith("Bravo.") \
        and queries[2].endswith("Charlie.")


def test_old_stale_prompt_behavior_fails_this_contract() -> None:
    """Reproduce the pre-fix behavior (last-query prompt reused for every
    target) and show it VIOLATES the per-target prompt contract."""
    g = _synthetic_group()
    recs = g["displayed_facts"]
    # emulate the old loop-carried state: prompt variable survives only
    # for the LAST qi
    stale_prompts = [None] * len(recs)
    for qi in range(len(recs)):
        from connector.experiments.query_influence_v3 import (
            _prompt, _query)
        prompt = _prompt("\n".join(r["line"] for r in recs), _query(recs[qi]))
        stale_prompts[qi] = prompt  # old code consumed this AFTER the loop
    # old consumption pattern: every target evaluated under prompts[-1]
    consumed = [stale_prompts[-1]] * len(recs)
    ok = [
        f"Return the tag of {recs[i]['entity'].capitalize()}." in consumed[i]
        for i in range(len(recs))]
    assert ok == [False, False, True], \
        "synthetic stale-prompt reproduction should fail 2 of 3 targets"


def test_lift_and_rank_inputs_unchanged_by_greedy_patch() -> None:
    """The corrective patch must not alter what the likelihood/rank loop
    consumes: build_query_prompt(g, i) equals the historical inline
    construction for both formats."""
    from connector.experiments.query_influence_v3 import (
        _prompt, _query, build_groups, build_query_prompt)
    for g in build_groups():
        recs = g["displayed_facts"]
        if g["format"] == "table":
            block = ("item | tag\n"
                     + "\n".join(f"{r['entity'].capitalize()} | {r['code']}"
                                 for r in recs))
        else:
            block = "\n".join(r["line"] for r in recs)
        for i in range(len(recs)):
            assert build_query_prompt(g, i) == _prompt(block, _query(recs[i]))


# ----------------------------- P7: Monte Carlo plus-one estimator ---------

def test_mc_p_value_never_reports_literal_zero() -> None:
    from connector.experiments.query_influence_v3 import PERM_DRAWS, sign_flip_p
    # n=40 forces the Monte Carlo branch; all-positive deltas make the
    # all-plus flip the unique exceeding permutation (prob 2^-40): the old
    # code sampled zero hits and reported p=0.0.
    vals = [0.5] * 40
    p = sign_flip_p(vals)
    assert p > 0.0, "plus-one estimator must never return literal zero"
    assert p == pytest.approx(1 / (PERM_DRAWS + 1))
    # symmetric sanity: all-negative data -> p near 1, capped below 1
    p_neg = sign_flip_p([-0.5] * 40)
    assert p_neg > 0.99


def test_exact_enumeration_branch_keeps_plain_tail_count() -> None:
    import itertools
    from connector.experiments.query_influence_v3 import sign_flip_p
    vals = [0.5, -0.2, 0.9, 0.3, -0.1, 0.7]
    obs = sum(vals)
    exact = sum(1 for signs in itertools.product((1, -1), repeat=len(vals))
                if sum(s * v for s, v in zip(signs, vals)) >= obs) / 64
    assert sign_flip_p(vals) == pytest.approx(exact)


# ----------------------- P5: trajectory metric identity --------------------

def test_trainer_trajectory_field_is_not_mislabeled_query_lift() -> None:
    src = _src("training/sft_grouped_queryswap.py")
    assert "qim_v2_mean_group_lift" not in src, \
        "historical field name must be retired"
    assert "qim_v2_same_query_candidate_margin" in src
    assert "trajectory_query_lift" in src and "NOT_MEASURED" in src


def test_final_receipt_distinguishes_margin_from_true_lift() -> None:
    receipt = json.loads((ROOT / "output/replication_receipt_final.json")
                         .read_text(encoding="utf-8"))
    traj = receipt["dev_trajectory_optimizer_updates"]
    assert traj, "receipt trajectory missing"
    # the receipt's historical trajectory rows carry the OLD key (they were
    # produced by the old code); the receipt must say so explicitly rather
    # than pretend those numbers were query lift.
    row_keys = set(traj[0].keys())
    has_old_key = any("qim_v2_mean_group_lift" in k for k in row_keys)
    if has_old_key:
        assert receipt["trajectory_metric_status"]["historical_field"] == \
            "qim_v2_mean_group_lift"
        assert receipt["trajectory_metric_status"]["actual_formula"] == \
            "same_query_candidate_margin"
        assert receipt["trajectory_metric_status"][
            "true_query_lift_intermediate_updates"] == "NOT_MEASURED"


# -------------------- P8: hash semantics are explicit ----------------------

def test_hash_semantics_explicitly_separated() -> None:
    audit = json.loads((ROOT / "data/grouped_queryswap/split_audit.json")
                       .read_text(encoding="utf-8"))
    for side in ("train", "heldout"):
        assert f"{side}_canonical_rows_sha256" in audit
        assert f"{side}_file_bytes_sha256" in audit
    assert audit["train_canonical_rows_sha256"] != \
        audit["train_file_bytes_sha256"]
    trainer_src = _src("training/sft_grouped_queryswap.py")
    assert "train_file_bytes_sha256" in trainer_src
    assert "train_canonical_rows_sha256" in trainer_src


# ------------- P9: extraction fixture vocab documentation honesty ----------

def test_extraction_module_does_not_claim_full_vocab_disjointness() -> None:
    src = _src("connector/experiments/context_value_extraction.py")
    assert "NOT guaranteed fully disjoint" in src
    # code prefixes really are disjoint from grouped-queryswap data
    prefixes = ("MRC", "QDX")
    blob = ""
    for name in ("train", "heldout"):
        blob += (ROOT / f"data/grouped_queryswap/{name}.jsonl").read_text(
            encoding="utf-8")
    import re
    for p in prefixes:
        assert not re.search(rf"\b{p}-\d{{3}}\b", blob), \
            f"extraction prefix {p} collides with grouped data"


# --------------- P2/P3/P4: corrective rescore artifacts + lineage ----------

FROZEN_FIXTURE_SHA = "27713accb3aa6825de23cf09540497943a0e56ca3ac177ba94f0f164f740a614"


def test_original_qim_v3_artifacts_preserved_untouched() -> None:
    parent = json.loads((ROOT / "output/qim3_parent_baseline.json")
                        .read_text(encoding="utf-8"))
    child = json.loads((ROOT / "output/qim3_sft6_replication.json")
                       .read_text(encoding="utf-8"))
    # the ORIGINAL (pre-correction) reports keep their original greedy
    # numbers and carry no corrective marker
    assert parent["candidate_diagnostic_only"][
        "greedy_corresponding_accuracy"] == "36/119"
    assert child["candidate_diagnostic_only"][
        "greedy_corresponding_accuracy"] == "24/119"
    assert "evaluator_version" not in parent and "evaluator_version" not in child
    assert parent["fixture_sha256"] == FROZEN_FIXTURE_SHA


def test_corrective_rescore_artifacts_exist_and_are_honest() -> None:
    parent = json.loads((ROOT / "output/qim3_parent_corrective_rescore.json")
                        .read_text(encoding="utf-8"))
    child = json.loads((ROOT / "output/qim3_sft6_corrective_rescore.json")
                       .read_text(encoding="utf-8"))
    for rep in (parent, child):
        assert rep.get("evaluation_class") == \
            "CORRECTIVE_RESCORING_AFTER_EVALUATOR_BUG"
        assert rep["fixture_sha256"] == FROZEN_FIXTURE_SHA
        assert rep["evaluator_version"] == "v3.1-corrective-greedy"


def test_valid_metrics_bit_identical_between_old_and_new_runs() -> None:
    old_p = json.loads((ROOT / "output/qim3_parent_baseline.json").read_text(encoding="utf-8"))
    new_p = json.loads((ROOT / "output/qim3_parent_corrective_rescore.json").read_text(encoding="utf-8"))
    old_c = json.loads((ROOT / "output/qim3_sft6_replication.json").read_text(encoding="utf-8"))
    new_c = json.loads((ROOT / "output/qim3_sft6_corrective_rescore.json").read_text(encoding="utf-8"))
    for old, new in ((old_p, new_p), (old_c, new_c)):
        assert old["parameter_sha256"] == new["parameter_sha256"]
        assert old["per_group_query_lift"] == new["per_group_query_lift"]
        assert old["group_level"]["mean_group_lift"] == \
            new["group_level"]["mean_group_lift"]
        assert old["group_level"]["groups_positive"] == \
            new["group_level"]["groups_positive"]
        assert old["candidate_diagnostic_only"]["correct_rank1_fraction"] == \
            new["candidate_diagnostic_only"]["correct_rank1_fraction"]
        assert old["candidate_diagnostic_only"]["mean_correct_rank"] == \
            new["candidate_diagnostic_only"]["mean_correct_rank"]
        # GREEDY is the ONLY field allowed to change
        assert old["fixture_sha256"] == new["fixture_sha256"]


def test_pr5_lineage_records_invalidation_then_correction() -> None:
    receipt = json.loads((ROOT / "output/replication_receipt_final.json")
                         .read_text(encoding="utf-8"))
    pr5 = receipt["preregistered_predictions_PR5_repaired"]
    assert pr5["original_recorded_verdict"] == "FAIL"
    assert pr5["evidence_status"] == "INVALIDATED_BY_EVALUATOR_BUG"
    assert pr5["reason"].startswith("stale query prompt")
    assert pr5["corrected_verdict"] in ("PASS", "FAIL")
    assert "output/qim3_sft6_corrective_rescore.json" in pr5["evidence"]
    assert pr5["no_retraining_performed"] is True


def test_manifest_points_to_corrective_rescore_without_erasing_history() -> None:
    manifest = json.loads((ROOT / "output/EVIDENCE_MANIFEST.json")
                          .read_text(encoding="utf-8"))
    cr = manifest["clean_replication"]
    orig = cr["original_qim_v3"]
    corr = cr["corrective_rescore"]
    assert orig["greedy_metric_status"] == "INVALIDATED"
    assert "stale" in orig["reason"]
    assert "likelihood/rank metrics remain valid" in orig["note"]
    assert corr["no_retraining_performed"] is True
    assert corr["parent_parameter_sha256"] == corr["child_parameter_sha256"] != "" \
        or (corr["parent_parameter_sha256"] and corr["child_parameter_sha256"])
    assert corr["pr5"] in ("PASS", "FAIL")


def test_verified_intervention_experience_remains_zero_after_repair() -> None:
    receipt = json.loads((ROOT / "output/replication_receipt_final.json")
                         .read_text(encoding="utf-8"))
    assert receipt["verified_intervention_experiences_earned"] == 0
