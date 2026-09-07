import json

import pytest

from bramastra_lab.discovery.compare import compare_stages


def make_row(world, target, policy, correct, probability, *, family="f", budget=1, label=1):
    return {"world_id": world, "family": family, "target": target, "policy": policy,
            "budget": budget, "correct": correct, "probability": probability,
            "label": label, "queries": budget}


def test_compare_reports_scores_deltas_and_family_worst_case():
    parent, child = [], []
    for world, family, old, new in (("a", "easy", False, True), ("b", "hard", True, False)):
        parent.append(make_row(world, 0, "learned", old, 0.75 if old else 0.25, family=family))
        child.append(make_row(world, 0, "learned", new, 0.75 if new else 0.25, family=family))
    result = compare_stages(parent, child, bootstrap_samples=30, seed=4)
    overall = next(x for x in result["comparisons"] if x["family"] is None)
    assert overall["nworlds"] == 2
    assert overall["accuracy_delta"] == 0
    assert overall["brier_delta"] == 0
    assert overall["accuracy_ci95"] is not None
    assert result["retention"]["worst_family_delta"] == -1


def test_compare_requires_exact_stage_keys_and_validates_rows():
    parent = [make_row("a", 0, "learned", True, 1)]
    child = [make_row("a", 1, "learned", True, 1)]
    with pytest.raises(ValueError, match="exact matched"):
        compare_stages(parent, child)
    bad = [make_row("a", 0, "learned", True, 1)]
    bad[0]["correct"] = False
    with pytest.raises(ValueError, match="correct"):
        compare_stages(parent, bad)


def test_compare_rejects_cross_stage_truth_or_family_changes():
    parent = [make_row("a", 0, "learned", True, 1)]
    changed_label = [make_row("a", 0, "learned", True, 0, label=0)]
    with pytest.raises(ValueError, match="labels/families"):
        compare_stages(parent, changed_label)
    changed_family = [make_row("a", 0, "learned", True, 1, family="other")]
    with pytest.raises(ValueError, match="labels/families"):
        compare_stages(parent, changed_family)


def test_one_world_ci_is_null_and_result_is_json_serializable():
    parent = [make_row("a", 0, "random", True, 1)]
    child = [make_row("a", 0, "random", True, 1)]
    result = compare_stages(parent, child)
    assert result["comparisons"][0]["accuracy_ci95"] is None
    json.dumps(result, allow_nan=False)


def test_scores_and_ci_use_equal_world_weight_with_unequal_row_counts():
    parent = [make_row("small", 0, "learned", False, .25)]
    child = [make_row("small", 0, "learned", True, .75)]
    for target in (0, 1, 2):
        parent.append(make_row("large", target, "learned", False, .25))
        child.append(make_row("large", target, "learned", False, .25))
    result = compare_stages(parent, child, bootstrap_samples=40, seed=9)
    overall = next(x for x in result["comparisons"] if x["family"] is None)
    assert overall["accuracy_delta"] == .5
    assert overall["brier_delta"] == -.25
    assert overall["weighting"] == "equal semantic worlds"
    reversed_result = compare_stages(list(reversed(parent)), list(reversed(child)), bootstrap_samples=40, seed=9)
    reversed_overall = next(x for x in reversed_result["comparisons"] if x["family"] is None)
    assert reversed_overall["accuracy_ci95"] == overall["accuracy_ci95"]
