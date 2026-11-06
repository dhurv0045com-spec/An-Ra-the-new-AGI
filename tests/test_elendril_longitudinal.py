from dataclasses import replace

import pytest

from anra_v5.elendril.longitudinal import Probe, Work, forward_transfer_delta, retention_delta


def test_retention_uses_identical_probe_and_equal_cluster_weight():
    baseline = Probe("a" * 64, "b" * 64, 7, Work(100, 50), (("a", 1, 1), ("b", 0, 4)))
    after = replace(baseline, work=Work(200, 100), clusters=(("b", 1, 4), ("a", 0, 1)))
    assert retention_delta(baseline, after) == {
        "mean_cluster_delta": [-3, 8], "cluster_deltas": {"a": [-1, 1], "b": [1, 4]},
        "n_clusters": 2,
    }
    with pytest.raises(ValueError, match="intervening"):
        retention_delta(baseline, baseline)
    with pytest.raises(ValueError, match="intervening"):
        retention_delta(baseline, replace(after, work=Work(200, 49)))
    with pytest.raises(ValueError, match="same fixture"):
        retention_delta(baseline, replace(after, fixture_sha256="c" * 64))


def test_forward_transfer_matches_full_work_and_coverage():
    scratch = Probe.from_score(fixture_sha256="a" * 64, model_sha256="b" * 64, seed=7,
                               work=Work(100, 50, 1, 2), metrics={"clusters": {"c": [1, 4]}})
    prior = replace(scratch, clusters=(("c", 3, 4),))
    assert forward_transfer_delta(scratch, prior)["mean_cluster_delta"] == [1, 2]
    with pytest.raises(ValueError, match="matched"):
        forward_transfer_delta(scratch, replace(prior, work=Work(100, 50, 2, 2)))
    with pytest.raises(ValueError, match="coverage"):
        forward_transfer_delta(scratch, replace(prior, clusters=(("other", 3, 4),)))
    with pytest.raises(ValueError, match="same fixture"):
        forward_transfer_delta(scratch, replace(prior, seed=19))
