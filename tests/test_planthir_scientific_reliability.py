import pytest
import torch

from anra_v5.planthir.causal import _paired_bootstrap_ci, run_representation_linear_probe
from anra_v5.planthir.contracts import FalsifierSpec, HypothesisSpec
from anra_v5.planthir.interpretation import synthesize_scientific_interpretation
from anra_v5.planthir.observatory import _calibration_metrics, compute_representation_isotropy


def test_isotropy_detects_antipodal_axis_and_missing_samples():
    axis = torch.tensor([[1., 0.], [-1., 0.], [1., 0.], [-1., 0.]])
    assert compute_representation_isotropy(axis)["isotropy"] == pytest.approx(0.)
    assert compute_representation_isotropy(torch.eye(2))["isotropy"] == pytest.approx(1.)
    assert compute_representation_isotropy(torch.zeros(0, 2))["isotropy"] is None
    assert compute_representation_isotropy(torch.zeros(2, 2))["isotropy"] is None


def test_uncertainty_and_calibration_reject_invalid_inputs():
    assert _paired_bootstrap_ci([]) is None
    assert _paired_bootstrap_ci([1.]) is None
    with pytest.raises(ValueError):
        _calibration_metrics([0.9, 0.2], [True])
    with pytest.raises(ValueError):
        _calibration_metrics([float('nan')], [True])


def test_probe_rejects_reused_worlds_before_evaluation():
    worlds = [{"world_id": str(i)} for i in range(6)]
    with pytest.raises(ValueError, match="disjoint"):
        run_representation_linear_probe(None, None, worlds[:4], worlds[3:])


def test_oracle_and_surviving_falsifier_do_not_certify_mechanism():
    hyp = HypothesisSpec("H", "some mechanism", "mechanism", (FalsifierSpec("F", "check", "x", ">=", 0.),))
    report = synthesize_scientific_interpretation(
        experiment_id="x", instrument_validation={"passed": True},
        oracle_intervention={"delta_oracle": 1., "delta_oracle_95ci": [1., 1.], "oracle_injected_accuracy": 1.},
        hypotheses=[hyp], observed_metrics_for_falsifiers={"x": 1.})
    assert not report.confirmed_causes
    assert any(c.claim_id == "S-ORACLE-INFORMATION-EFFECT" for c in report.suspected_mechanisms)
    assert any(c.claim_id == "U-H" for c in report.unresolved_questions)


def test_probe_rejects_renamed_isomorphic_v8_worlds():
    from copy import deepcopy
    from anra_v5.planthir.causal import _probe_split_keys
    world = {"world_id": "a", "goal_subject": "a", "hops": 1,
             "initial_facts": [{"subject": "a", "source": "s", "tick": 1, "value": "b"}],
             "allowed_public_tools": {"query_available": False, "memory_available": False},
             "hidden": {"query_payload": [], "memory_payload": [], "expected": "b"}}
    renamed = deepcopy(world)
    renamed.update(world_id="b", goal_subject="x")
    renamed["initial_facts"][0].update(subject="x", value="y")
    assert _probe_split_keys([world])[1] == _probe_split_keys([renamed])[1]
    with pytest.raises(ValueError, match="graph signature"):
        run_representation_linear_probe(None, None, [world] * 4, [renamed] * 2)


def test_cluster_bootstrap_requires_assumption_and_groups_variants():
    from anra_v5.planthir.causal import _clustered_bootstrap_ci
    worlds = [{"world_id": "a", "pair_id": "p"}, {"world_id": "b", "pair_id": "p"},
              {"world_id": "c", "parent_world_id": "root"}, {"world_id": "d", "parent_world_id": "root"}]
    assert _clustered_bootstrap_ci([1., 1., -1., -1.], worlds, assume_independent_clusters=False) == (None, None)
    ci, clusters = _clustered_bootstrap_ci([1., 1., -1., -1.], worlds, assume_independent_clusters=True)
    assert clusters == 2 and ci == [-1., 1.]
    assert _clustered_bootstrap_ci([1., 1.], worlds[:2], assume_independent_clusters=True) == (None, 1)


@pytest.mark.parametrize("architecture", ["U", "F-cap"])
def test_ablation_restores_model_after_exception(monkeypatch, architecture):
    from types import SimpleNamespace
    from anra_v5.planthir import causal
    spec = SimpleNamespace(architecture=architecture, recurrent_steps=2)
    model = SimpleNamespace(v8_spec=spec, cap_out=torch.nn.Linear(2, 2))
    policy = SimpleNamespace(v8_spec=spec, model=model)
    weight = model.cap_out.weight.detach().clone()
    bias = model.cap_out.bias.detach().clone()
    calls = iter([(True, None, [("ANSWER", "a")])])
    def episode(*args):
        try:
            return next(calls)
        except StopIteration:
            raise RuntimeError("injected")
    monkeypatch.setattr(causal, "_run_single_episode", episode)
    with pytest.raises(RuntimeError, match="injected"):
        causal.run_surgical_mechanism_ablation(policy, None, [{"world_id": "a"}])
    assert model.v8_spec is spec and policy.v8_spec is spec
    assert torch.equal(model.cap_out.weight, weight) and torch.equal(model.cap_out.bias, bias)
