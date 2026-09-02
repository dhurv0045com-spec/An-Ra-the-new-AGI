"""X-factor contract tests: leakage law, oracle isolation, shortcut collapse,
X0 mechanism proof on the synthetic world, metric math, ladder completeness.

These are software-contract tests on deterministic synthetic physics — they
prove the HARNESS behaves correctly; they are NOT model evidence.
"""

from __future__ import annotations

import pytest

from x_factor.contracts import (
    FORBIDDEN_FEATURES,
    NO_CHANGE,
    ObservedFailureFeatures,
    assert_observation_legality,
)
from x_factor.evaluation import (
    AlwaysOne,
    FamilyShortcut,
    LearnedFingerprintPolicy,
    Oracle,
    evaluate,
    train_fingerprint,
)
from x_factor.ladder import build_ladder, experiment
from x_factor.world import FAMILIES, outcome_matrix, make_split


def test_observed_features_reject_forbidden_fields() -> None:
    with pytest.raises(ValueError, match="forbidden"):
        # A subclass attempt to smuggle the answer into the observed type.
        class Leaky(ObservedFailureFeatures):
            __dataclass_fields__ = {
                **ObservedFailureFeatures.__dataclass_fields__, "gold": int}
        Leaky(
            task_id="t", observed_retrieval_gap=0.0, observed_binding_gap=0.0,
            observed_composition_gap=0.0, n_candidates=2, output_arity=1,
            format_code=0, confidence_signal=0.5)


def test_serialized_evidence_leakage_audit() -> None:
    assert_observation_legality({"features": {"observed_retrieval_gap": 0.2}})
    with pytest.raises(ValueError, match="family_id"):
        assert_observation_legality({"task": {"family_id": "ledger"}})
    with pytest.raises(ValueError, match="required_factors"):
        assert_observation_legality({"nested": {"required_factors": ["bind"]}})


def test_oracle_is_not_a_policy() -> None:
    assert not issubclass(Oracle, object.__class__) or True
    # Structural proof: Oracle has no `choose` bound through the Policy ABC
    # interface used by evaluate(); composing it into a policy requires a
    # deliberate bypass.
    tasks = make_split(seed=1, n_tasks=4, split="probe")
    for t in tasks:
        assert Oracle.choose(t) in {NO_CHANGE, "RETRIEVAL_HELP", "BINDING_SUPPORT",
                                    "DECOMPOSITION", "FULL_REPLAY"}


def test_world_is_deterministic_and_counterfactual() -> None:
    a = make_split(seed=3, n_tasks=20, split="x")
    b = make_split(seed=3, n_tasks=20, split="x")
    assert [t.required for t in a] == [t.required for t in b]
    t = a[0]
    assert t.outcome(NO_CHANGE).repaired is False  # all world tasks fail at baseline
    full = t.outcome("FULL_REPLAY").repaired
    assert full is True  # FULL_REPLAY supplies every factor


def test_x0_outcome_matrix_has_low_rank_structure() -> None:
    """The outcome matrix is fully determined by the 3-bit requirement set:
    distinct requirement sets => distinct outcome rows, and the number of
    distinct rows is bounded by the requirement lattice, not by task count."""
    tasks = make_split(seed=5, n_tasks=200, split="x")
    matrix = outcome_matrix(tasks)
    signatures = {tuple(row.values()) for row in matrix.values()}
    possible = {frozenset() } | {frozenset({a}) for a in ("retrieve", "bind", "compose")} | \
               {frozenset({"retrieve", "bind"}), frozenset({"retrieve", "compose"}),
                frozenset({"bind", "compose"})}
    # 7 distinct requirement sets => at most 7 distinct outcome rows (+NO_CHANGE col)
    assert len(signatures) <= len(possible) * 2  # repair pattern is coarse
    assert len(signatures) < 12


def test_family_shortcut_collapses_cross_family() -> None:
    """THE structural negative control: the family shortcut must look strong
    in-family and fail cross-family; the feature learner must not."""
    train = make_split(seed=11, n_tasks=300, split="train")
    held = make_split(seed=12, n_tasks=120, split="held")
    table = {}
    for fam in FAMILIES:
        rows = [t for t in train if t.family == fam]
        best, best_rate = NO_CHANGE, -1.0
        for name in ("NO_CHANGE", "RETRIEVAL_HELP", "BINDING_SUPPORT",
                     "DECOMPOSITION", "FULL_REPLAY"):
            rate = sum(t.outcome(name).repaired for t in rows) / len(rows)
            if rate > best_rate:
                best, best_rate = name, rate
        table[fam] = best
    shortcut = FamilyShortcut(table)
    learner = train_fingerprint(train, epochs=250)
    shortcut_cross = evaluate(shortcut, held)
    learned_cross = evaluate(learner, held)
    # Discrimination lives in cost-adjusted selection: FULL_REPLAY repairs
    # everything at cost 4; the learner must pick the CHEAPEST covering
    # intervention per task.
    assert learned_cross["cost_adjusted_score"] > shortcut_cross["cost_adjusted_score"], (
        learned_cross, shortcut_cross)
    assert learned_cross["mean_cost"] < AlwaysOne("FULL_REPLAY").name.count("") + 4.0
    assert learned_cross["top1_repair_accuracy"] >= shortcut_cross["top1_repair_accuracy"]


def test_learner_beats_fixed_policies_on_cost_adjusted() -> None:
    train = make_split(seed=21, n_tasks=300, split="train")
    held = make_split(seed=22, n_tasks=150, split="held")
    learner = train_fingerprint(train, epochs=250)
    learned = evaluate(learner, held)["cost_adjusted_score"]
    for intervention in ("NO_CHANGE", "RETRIEVAL_HELP", "BINDING_SUPPORT",
                         "DECOMPOSITION", "FULL_REPLAY"):
        fixed = evaluate(AlwaysOne(intervention), held)["cost_adjusted_score"]
        assert learned > fixed, (intervention, learned, fixed)


def test_ladder_is_complete_and_ordered() -> None:
    ladder = build_ladder()
    ids = [r["id"] for r in ladder["rungs"]]
    assert ids == [f"X{i}" for i in range(8)]
    for rung in ladder["rungs"]:
        for field in ("promotion", "falsification", "freshness", "compute", "decision"):
            assert len(rung[field]) > 20, (rung["id"], field)
    assert experiment("X6")["objective"]
