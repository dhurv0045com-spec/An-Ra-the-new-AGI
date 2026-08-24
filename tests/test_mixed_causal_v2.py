"""Tests: true counterfactual normalization + mixed-causal v2 contracts."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]

from connector.experiments.counterfactual_normalization import (
    normalize_scores, build_counterfactual_queries,
    verify_byte_identical_context, argmax,
    pseudo_normalization_rejected_example)
import connector.experiments.mixed_causal_v1 as mc


def test_true_normalization_changes_raw_ranking() -> None:
    """THE core property: true normalization can flip the argmax vs raw.
    Candidate 0 wins raw (2.0 vs 1.0) but under ITS OWN counterfactual
    query candidate 0 also scores high (strong intrinsic prior: cf[1][0]
    = 1.9), while candidate 1 collapses under the same query (cf[1][1] =
    -3.0). Normalized: cand0 = 2.0-1.9 = 0.1; cand1 = 1.0-(-3.0) = 4.0."""
    raw = [2.0, 1.0]
    cf = {1: [1.9, -3.0]}
    norm = normalize_scores(0, raw, cf)
    assert argmax(raw) == 0
    assert argmax(norm) == 1       # ranking flipped — impossible for pseudo


def test_pseudo_normalization_rejected() -> None:
    """The old same-query formula is rank-preserving and must never be
    mistaken for normalization."""
    raw, pseudo = pseudo_normalization_rejected_example()
    assert argmax(raw) == argmax(pseudo)


def test_normalize_scores_rejects_actual_query_in_cf_set() -> None:
    with pytest.raises(ValueError):
        normalize_scores(0, [1.0, 2.0], {0: [1.0, 2.0]})
    with pytest.raises(ValueError):
        normalize_scores(0, [1.0, 2.0], {})


def test_counterfactual_prompts_change_query_only() -> None:
    task = next(t for t in mc.build_tasks()
                if t["family"] == "selection")
    cfs = mc.counterfactual_queries(task)
    assert len(cfs) == len(task["alt_query_targets"])
    verify_byte_identical_context(cfs)
    # query lines differ
    qlines = {p.splitlines()[-2] for p in cfs.values()}
    assert len(qlines) == len(cfs)


def test_context_byte_identical_across_counterfactuals() -> None:
    for task in mc.build_tasks():
        if task["family"] != "selection":
            continue
        cfs = mc.counterfactual_queries(task)
        ctx = {tuple(p.splitlines()[:-2]) for p in cfs.values()}
        assert len(ctx) == 1
        return


def test_applicability_masks_from_structure_only() -> None:
    tasks = mc.build_tasks()
    by_family = {}
    for t in tasks:
        by_family.setdefault(t["family"], t)
    # composition: arity>1 -> no CONSTRAINED/NORM_EXACT/NORMALIZED single-slot
    comp = mc.applicable_actions(by_family["composition"])
    assert "CONSTRAINED" not in comp and "NORM_EXACT" not in comp
    # copy_single: one candidate, no alt targets -> no normalization
    copy = mc.applicable_actions(by_family["copy_single"])
    assert "NORMALIZED" not in copy and "NORM_EXACT" not in copy
    # selection: everything applicable
    sel = mc.applicable_actions(by_family["selection"])
    assert set(["NO_CHANGE", "CONSTRAINED", "NORMALIZED", "NORM_EXACT",
                "ABSTAIN"]) <= set(sel)


def test_matrix_contains_full_action_outcome_data() -> None:
    r = json.loads((ROOT / "output/mixed_causal_matrix_v2.json")
                   .read_text(encoding="utf-8"))
    assert r["n_tasks"] == 60
    row = r["per_task_rows"][0]
    for action in row["observed"]["applicable_actions"]:
        assert action in row["actions"]
        entry = row["actions"][action]
        assert "verifier_pass" in entry and "cost" in entry


def test_policy_v3_frozen_and_never_sees_gold() -> None:
    p = json.loads((ROOT / "output/self_model_v3.json").read_text(encoding="utf-8"))
    forbidden = {"gold", "family", "RAW_ok", "NORMALIZED_ok",
                 "raw_rank_of_gold", "adj_rank_of_gold"}
    assert not (set(p["feature_names"]) & forbidden)
    assert p["utility_rule"]["lambda"] == 0.25


def test_oracle_is_evaluator_only() -> None:
    """Oracle needs verifier results; the policy interface cannot accept it."""
    from connector.experiments.observed_self_model import AdaptivePolicy
    from connector.experiments.observed_self_model import ObservedArmState
    pol = AdaptivePolicy(weights=(0.0,) * 10, bias=0.0)
    state = ObservedArmState(
        n_candidates=2, format_name="prose", raw_pick_code="A",
        norm_pick_code="B", free_out_code=None, constrained_pick_code=None,
        raw_scores=[1, 0], norm_scores=[0, 1])
    # policy decides from state alone; oracle action is not an input
    assert pol.decide(state) in ("NORMALIZE", "KEEP_RAW")


def test_old_mixed_calibration_marked_invalid() -> None:
    d = json.loads((ROOT / "output/mixed_causal_dev_results.json")
                   .read_text(encoding="utf-8"))
    assert d["invalidation"]["status"] == "INVALID_FOR_CAUSAL_POLICY_TRAINING"
    assert d["invalidation"]["reason"] == "NORMALIZATION ARM MISIMPLEMENTED"
