"""Evidence tooling checks without model updates."""

import pytest

from tools.compare_elendril_pairs import paired_cluster_interval
from tools.verify_elendril_evidence import relative_path


def counts(a, b):
    return {"query_swap_both_correct": [a + b, 4],
            "cluster_query_pairs": {"A": [a, 2], "B": [b, 2]}}


def test_paired_bootstrap_reuses_clusters_across_seeds():
    reference = {7: counts(0, 2), 19: counts(2, 0)}
    candidate = {7: counts(2, 0), 19: counts(0, 2)}
    result = paired_cluster_interval(reference, candidate, draws=100)
    assert result["independent_resampling_units"] == 2
    assert result["matched_seeds"] == [7, 19]
    assert result["delta_candidate_minus_reference"] == 0
    assert result["interval_95"] == [0, 0]


def test_paired_bootstrap_refuses_incomplete_cluster_coverage():
    incomplete = {"query_swap_both_correct": [1, 4],
                  "cluster_query_pairs": {"A": [1, 2], "B": [0, 1]}}
    with pytest.raises(ValueError, match="do not cover"):
        paired_cluster_interval({7: counts(0, 0)}, {7: incomplete}, draws=100)


@pytest.mark.parametrize("path", ["../other", "C:/outside", "/outside", "x\\other"])
def test_evidence_paths_reject_escape(path):
    with pytest.raises(ValueError, match="unsafe archived path"):
        relative_path(path)
    assert relative_path("legitimate/artifact.json").as_posix() == "legitimate/artifact.json"
