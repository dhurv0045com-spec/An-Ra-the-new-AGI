import pytest

from bramastra_lab.discovery.statistics import summarize


def row(world, target, policy, correct, probability, *, family="f", budget=1, queries=1):
    return {"world_id": world, "family": family, "target": target, "policy": policy,
            "budget": budget, "correct": correct, "probability": probability,
            "label": 1, "queries": queries}


def test_numeric_aggregates_and_paired_bootstrap_are_json_shaped():
    rows = []
    for world, target, learned, baseline in [("a", 0, True, False), ("b", 1, False, False)]:
        rows += [row(world, target, "random", baseline, 0.25), row(world, target, "coverage", baseline, 0.25),
                 row(world, target, "learned", learned, 0.75 if learned else 0.25)]
    result = summarize(rows, bootstrap_samples=20, seed=7)
    assert result["aggregates"]["learned"]["1"]["nworlds"] == 2
    assert result["aggregates"]["learned"]["1"]["accuracy"] == 0.5
    assert result["aggregates"]["learned"]["1"]["brier"] == 0.3125
    assert result["aggregates"]["learned"]["1"]["family"]["f"]["nrows"] == 2
    comparison = next(c for c in result["comparisons"] if c["baseline"] == "random")
    assert comparison["delta_accuracy"] == 0.5
    assert comparison["ci95"] is not None


def test_duplicate_and_missing_pairs_fail():
    duplicate = [row("a", 0, "random", True, 1), row("a", 0, "random", True, 1)]
    with pytest.raises(ValueError, match="duplicate"):
        summarize(duplicate)
    missing = [row("a", 0, "random", True, 1), row("a", 0, "learned", True, 1), row("b", 0, "learned", True, 1)]
    with pytest.raises(ValueError, match="missing paired"):
        summarize(missing)


def test_cluster_bootstrap_gives_each_world_equal_weight():
    rows = [row("small", 0, "random", False, 0), row("small", 0, "learned", True, 1)]
    for target in range(10):
        rows += [row("large", target, "random", False, 0), row("large", target, "learned", False, 0)]
    comparison = next(c for c in summarize(rows, bootstrap_samples=20)["comparisons"] if c["baseline"] == "random")
    assert comparison["delta_accuracy"] == 0.5


def test_one_world_has_explicit_insufficient_uncertainty():
    rows = [row("a", 0, policy, policy == "learned", 1 if policy == "learned" else 0)
            for policy in ("random", "coverage", "learned")]
    result = summarize(rows)
    assert all(c["ci95"] is None and c["uncertainty"] == "insufficient worlds" for c in result["comparisons"])


def test_invalid_probability_and_counts_fail():
    bad_probability = row("a", 0, "random", True, float("nan"))
    with pytest.raises(ValueError, match="probability"):
        summarize([bad_probability])
    bad_probability["probability"] = 1.1
    with pytest.raises(ValueError, match="probability"):
        summarize([bad_probability])
    bad_count = row("a", 0, "random", True, 0)
    bad_count["queries"] = -1
    with pytest.raises(ValueError, match="queries"):
        summarize([bad_count])
    bad_label = row("a", 0, "random", True, 0)
    bad_label["label"] = 2
    with pytest.raises(ValueError, match="label"):
        summarize([bad_label])


@pytest.mark.parametrize("field", ["world_id", "family", "policy"])
def test_identity_fields_are_nonempty_strings(field):
    bad = row("a", 0, "random", True, 1)
    bad[field] = ""
    with pytest.raises(ValueError, match=field):
        summarize([bad])


def test_bounds_consistency_and_correctness_invariants_fail():
    bad_queries = row("a", 0, "random", True, 1, budget=1, queries=2)
    with pytest.raises(ValueError, match="queries"):
        summarize([bad_queries])
    bad_target = row("a", -1, "random", True, 1, queries=0)
    with pytest.raises(ValueError, match="target"):
        summarize([bad_target])
    wrong_correct = row("a", 0, "random", True, 0, queries=0)
    with pytest.raises(ValueError, match="correct"):
        summarize([wrong_correct])


def test_world_target_label_and_family_must_be_consistent():
    first = row("a", 0, "random", True, 1, queries=0)
    second = row("a", 0, "coverage", True, 1, family="other", queries=0)
    with pytest.raises(ValueError, match="consistent"):
        summarize([first, second])
