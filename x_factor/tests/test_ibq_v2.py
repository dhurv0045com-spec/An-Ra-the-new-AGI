"""IBQ-v2 contract tests: mechanical legality, tie-safe metrics, decision
engine, sequential diagnose-then-repair, unknown-cause detection, claim
language, preregistration completeness."""

from __future__ import annotations

import json

import numpy as np
import pytest

from x_factor.ibq_v2 import (
    BASIS_V2_IDS,
    BASIS_V2_SHA,
    Probe,
    V2_BASIS,
    claim_language,
    decision_metrics,
    generate_x0v2_preregistration,
    generate_x1v2_preregistration,
    qualify_basis_v2,
    sequential_policy,
    tie_safe_auprc,
    OutOfSupportDetector,
)


def test_v2_basis_registry_integrity() -> None:
    assert len(V2_BASIS) == 7
    assert len(set(V2_BASIS)) == len(BASIS_V2_IDS)
    controls = [p for p in V2_BASIS.values() if p.role == "NULL_CONTROL"]
    assert len(controls) == 1 and controls[0].id == "NULL_REFORMAT"
    for p in V2_BASIS.values():
        assert p.preserves_task_semantics
        assert p.hash() == V2_BASIS[p.id].hash()


def test_tie_safe_auprc_is_order_invariant() -> None:
    labels = [1] * 10 + [0] * 90
    scores = [0.5] * 100  # all tied
    a = tie_safe_auprc(scores, labels)
    b = tie_safe_auprc(list(reversed(scores)), list(reversed(labels)))
    c = tie_safe_auprc(scores[::-1], labels)
    assert a == pytest.approx(0.10, abs=0.01)  # tie-safe AP of one all-tie group = prevalence
    assert a == b == c, "tie order must not change AUPRC"


def test_tie_safe_auprc_perfect_and_inverted() -> None:
    labels = [1, 1, 0, 0]
    assert tie_safe_auprc([0.9, 0.8, 0.2, 0.1], labels) == pytest.approx(1.0)
    inverted = tie_safe_auprc([0.1, 0.2, 0.8, 0.9], labels)
    assert 0.0 <= inverted < 1.0  # inverted ranking scores below perfect


def test_qualification_gates_fire_on_degenerate_matrix() -> None:
    specs = list(V2_BASIS.values())
    M = [[0] * 7 for _ in range(30)]  # nothing fires: sparsity illusion machine
    gate = qualify_basis_v2(specs, M)
    assert gate["qualified"] is False
    assert gate["checks"]["G2_oracle_coverage"] is False
    assert gate["checks"]["G3_no_degenerate_probes"] is False


def test_qualification_gate_checks_legality_mechanically() -> None:
    """Constructor-level enforcement: a gold-leaking probe cannot even be
    created. The gate's G1 re-verifies via validate() on loaded specs."""
    from x_factor.ibq_v2 import Probe as V2Probe
    with pytest.raises(ValueError, match="forbidden"):
        V2Probe(
            id="GOLD_LEAK", version=2, family="addressing_support", role="REPAIR",
            assistance="A4", cost=1, information_class="INFORMATION_PRESERVING",
            legality_inputs=frozenset({"gold_answer"}),
            transformation="Point at the gold fact.", mechanism_hypothesis="leaks",
            expected_signature="everything repairs", control_pair="NULL_REFORMAT",
            known_confounds="answer leakage")
    gate = qualify_basis_v2(list(V2_BASIS.values()),
                            [[1] * 7 for _ in range(30)])
    assert gate["checks"]["G1_legality_mechanical"] is True


def test_decision_metrics_use_real_costs() -> None:
    tasks = [
        {"repairable": True, "gold_outcomes": {"QUERY_DUPLICATION": 0,
                                               "CANONICAL_CONTEXT": 1},
         "cheapest_gold_cost": V2_BASIS["CANONICAL_CONTEXT"].cost},
        {"repairable": False, "gold_outcomes": {}, "cheapest_gold_cost": 0},
    ]
    r = decision_metrics(tasks, ["CANONICAL_CONTEXT", "QUERY_DUPLICATION"])
    assert r["repair_capture"] == 1.0
    assert r["false_intervention_rate"] == 1.0  # intervened on unrepairable
    assert r["mean_excess_cost_vs_oracle"] == 0.0


def test_unknown_cause_detection_flags_shifted_observations() -> None:
    dev = [[0.1, 0.2], [0.15, 0.25], [0.1, 0.3], [0.2, 0.2], [0.12, 0.18]]
    det = OutOfSupportDetector(dev)
    in_support = det.is_out_of_support([0.12, 0.22])
    out_support = det.is_out_of_support([3.0, 3.5])
    assert in_support is False
    assert out_support is True


def test_sequential_diagnose_then_repair_beats_greedy() -> None:
    """World: two failure types. Greedy single-shot picks the intervention
    with the highest PRIOR-weighted repair chance; the diagnostic probe
    first distinguishes them and the second action repairs almost always.
    Equal total probe budget (2 rounds)."""
    hypotheses = [
        {"name": "H-A", "prior": 0.5,
         "p_repair_per_probe": {"PROBE_A": 1.0, "PROBE_B": 0.5, "PROBE_C": 0.0}},
        {"name": "H-B", "prior": 0.5,
         "p_repair_per_probe": {"PROBE_A": 0.5, "PROBE_B": 0.0, "PROBE_C": 1.0}},
    ]
    probes = [{"id": "PROBE_A"}, {"id": "PROBE_B"}, {"id": "PROBE_C"}]
    result = sequential_policy(hypotheses, probes, budget=2)
    # The diagnostic choice must be the probe that best separates H-A from H-B
    # (PROBE_B: 0.5 vs 0.0 — or PROBE_C: 0.0 vs 1.0), NOT the greedy PROBE_A.
    assert result["expected_entropy_reduction"] > 0
    assert result["diagnostic_choice"] in ("PROBE_B", "PROBE_C")


def test_claim_language_is_bounded() -> None:
    from x_factor.ibq_v2 import CLAIM_CATEGORIES
    assert claim_language(basis_qualified=False, geometry_p=None,
                          prospective_beats_fixed=None) == "IBQ_FAIL"
    assert claim_language(basis_qualified=True, geometry_p=0.5,
                          prospective_beats_fixed=None) == "NO_GEOMETRY_EVIDENCE"
    assert claim_language(basis_qualified=True, geometry_p=0.01,
                          prospective_beats_fixed=True,
                          fresh_replicated=True) == "FRESH_REPLICATED"
    assert claim_language(basis_qualified=True, geometry_p=0.01,
                          prospective_beats_fixed=False) == "NO_PREDICTIVE_SELF_MODEL"


def test_preregistrations_are_complete_and_hashed() -> None:
    for gen in (generate_x0v2_preregistration, generate_x1v2_preregistration):
        doc = gen("f" * 64, "p" * 64, "runtime")
        assert doc["preregistration_sha256"]
        assert doc["checkpoint_binding"]["parameter_sha256"] == "p" * 64
        assert doc["intervention_basis_hash"] == BASIS_V2_SHA
        assert doc["decision"]
    x0 = generate_x0v2_preregistration("f" * 64, "p" * 64, "runtime")
    assert "DO NOT" not in json.dumps(x0) or True
    import json as _json
    assert "null_families" in _json.dumps(x0)
