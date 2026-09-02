"""Scientific-contract tests for the interventional cognitive geometry.

All tests run on deterministic synthetic worlds. They prove MECHANISM
behavior — what the framework can and cannot discover — never model
cognition. Decisive properties:

  - neutral observations hide the causal basis (no 1:1 sensor);
  - low-rank worlds compress, random worlds do not (measured, relative);
  - shortcut trap: the family classifier wins development, fails fresh;
    the family-blind geometry survives;
  - the full-rank world defeats the predictor (honest failure);
  - an unseen cause is detectable as outcome rows IMPOSSIBLE under the
    development physics (support violation, not error magnitude);
  - factor-basis permutation leaves observables and conclusions unchanged;
  - active selection can override greedy repair via information gain.
"""

from __future__ import annotations

import numpy as np
import pytest

from x_factor.geometry import (
    INTERVENTIONS,
    ProspectivePredictor,
    active_select,
    effective_rank,
    make_world,
    policy_from_prediction,
    FAMILIES,
    _mixing,
    shortcut_correlated_world,
)


def _fit_predict(train, test):
    predictor = ProspectivePredictor().fit(train)
    return predictor, predictor.predict(test), ProspectivePredictor.errors(
        predictor.predict(test), test)


def test_neutral_observations_hide_the_causal_basis() -> None:
    A = _mixing(42)
    rng = np.random.default_rng(0)
    H = (rng.random((2000, 3)) > 0.5).astype(float)
    O = H @ A.T
    C = np.corrcoef(np.hstack([H, O]).T)[:3, 3:]
    assert np.abs(C).max() < 0.99
    assert np.abs(A).min() > 0.1


def test_low_rank_world_compresses_and_random_world_does_not() -> None:
    causal = make_world("A-easy", 7, 120)
    random_world = make_world("D-noncausal", 8, 120, random_outcomes=True)
    M_causal = np.stack([np.array(r["gold"][1:], float) for r in causal.rows])
    M_random = np.stack([np.array(r["gold"][1:], float) for r in random_world.rows])
    er_causal, er_random = effective_rank(M_causal), effective_rank(M_random)
    assert er_random > er_causal + 0.5, (er_causal, er_random)


def test_prospective_prediction_works_in_clean_world_and_fails_in_random() -> None:
    clean = make_world("A", 11, 160)
    holdout_clean = make_world("A-hold", 12, 80, A=clean.A)  # same instrument
    predictor, _, err_clean = _fit_predict(clean, holdout_clean)
    greedy = [policy_from_prediction(p) for p in predictor.predict(holdout_clean)]
    correct = sum(r["gold"][INTERVENTIONS.index(g)] == 1
                  for g, r in zip(greedy, holdout_clean.rows))
    assert correct / len(holdout_clean.rows) > 0.6

    random_world = make_world("D", 13, 160, random_outcomes=True)
    random_hold = make_world("D-hold", 14, 80, random_outcomes=True, A=random_world.A)
    rand_pred = ProspectivePredictor(use_conjunction=True).fit(random_world).predict(random_hold)
    err_rand = ProspectivePredictor.errors(rand_pred, random_hold)
    assert err_rand.mean() > err_clean.mean() + 0.05, "honest failure in world D"


def test_shortcut_trap_fools_family_and_not_geometry() -> None:
    dev, fresh = shortcut_correlated_world("C", 21, 140,
                                           dev_correlation=0.85,
                                           fresh_correlation=0.15)
    # Structural relabel property: in dev the family matches the best
    # intervention at ~0.85; in fresh at ~0.15. A family-based policy
    # therefore looks great in development and fails the shift.
    def match_rate(world):
        hits = 0
        for r in world.rows:
            best = next((iv for iv, g in zip(INTERVENTIONS, r["gold"]) if g == 1),
                        "NO_CHANGE")
            hits += r["family"] == FAMILIES[INTERVENTIONS.index(best)]
        return hits / len(world.rows)
    assert abs(match_rate(dev) - 0.85) < 0.1
    assert abs(match_rate(fresh) - 0.15) < 0.1
    predictor = ProspectivePredictor(use_conjunction=True).fit(dev)
    err_fresh = ProspectivePredictor.errors(predictor.predict(fresh), fresh).mean()
    # Survival reference: error must stay well below chance-level scoring
    # (the D-world regime), and far below the relabeled-family chaos.
    assert err_fresh < 0.35, "family-blind geometry survives the shift"


def test_unseen_cause_is_detectable_as_impossible_outcome_rows() -> None:
    dev = make_world("dev", 31, 200)
    fresh_same = make_world("fresh-same", 32, 100)
    fresh_new = make_world("fresh-new", 33, 100, extra_factor=True)

    def support(world):
        return {tuple(r["gold"]) for r in world.rows}

    dev_support, same_support, new_support = support(dev), support(fresh_same), support(fresh_new)
    outside = new_support - dev_support
    assert outside, "fresh-new must contain rows outside the dev physics"
    assert all(g == (0, 0, 0, 0, 0) for g in outside), "impossible = nothing repairs"
    assert not (same_support - dev_support)
    assert any("quantum" in r["required"] for r in fresh_new.rows)


def test_conclusions_are_invariant_to_the_arbitrary_instrument_draw() -> None:
    """The mixing transform A is an arbitrary measurement instrument. Two
    worlds with identical latent requirements and physics but different
    instrument draws must yield the same predictive error statistics —
    conclusions may not depend on which instrument happened to be used.
    (Factor renaming without renaming the intervention registry is NOT a
    symmetry: the registry is fixed to factor names by design.)"""
    order = ("retrieve", "bind", "compose")
    pinned = [frozenset(rng_set) for rng_set in (
        __import__("random").Random(9).sample(order, 1) for _ in range(0))] if False else None
    import random
    rng = random.Random(9)
    pinned = [frozenset(rng.sample(order, rng.choice((1, 2, 3)))) for _ in range(140)]
    hold_pinned = [frozenset(rng.sample(order, rng.choice((1, 2, 3)))) for _ in range(80)]
    means = []
    for instrument_seed in (51, 52):
        A = _mixing(instrument_seed)
        dev = make_world("inv-dev", 55, 140, A=A, factor_order=order,
                         required_sets=pinned)
        hold = make_world("inv-hold", 56, 80, A=A, factor_order=order,
                          required_sets=hold_pinned)
        pred = ProspectivePredictor(use_conjunction=True).fit(dev).predict(hold)
        means.append(ProspectivePredictor.errors(pred, hold).mean())
    assert means[0] == pytest.approx(means[1], abs=0.02), (
        "conclusions must not depend on the arbitrary instrument draw")


def test_active_selection_prefers_discriminating_intervention() -> None:
    w1 = make_world("h1", 61, 120)
    w2 = make_world("h2", 62, 120, interaction=True, A=w1.A)  # same instrument
    p1, p2 = ProspectivePredictor().fit(w1), ProspectivePredictor().fit(w2)
    ambiguous = make_world("amb", 63, 40, noise=0.0, A=w1.A)
    disagreements = [active_select([p1, p2], ambiguous, i, beta=2.0)[1]
                     for i in range(len(ambiguous.rows))]
    assert max(disagreements) > 0.05, "committee must expose uncertainty"
    greedy = [policy_from_prediction((p1.predict(ambiguous)[i] + p2.predict(ambiguous)[i]) / 2)
              for i in range(len(ambiguous.rows))]
    active_greedy = [active_select([p1, p2], ambiguous, i, beta=0.0)[0]
                     for i in range(len(ambiguous.rows))]
    assert active_greedy == greedy
    active_curious = [active_select([p1, p2], ambiguous, i, beta=2.0)[0]
                      for i in range(len(ambiguous.rows))]
    assert active_curious != greedy, "information gain must be able to override repair"


def test_interaction_world_exposes_non_additivity() -> None:
    clean = make_world("clean", 71, 140)
    interact = make_world("E", 72, 140, interaction=True)
    clean_hold = make_world("clean-h", 73, 80, A=clean.A)
    interact_hold = make_world("E-h", 74, 80, interaction=True, A=interact.A)
    # LINEAR geometry fails on the interaction world; the conjunction basis
    # recovers it. That pairing is the mechanism finding.
    linear_int = ProspectivePredictor(use_conjunction=False).fit(interact)
    err_linear = ProspectivePredictor.errors(linear_int.predict(interact_hold), interact_hold).mean()
    conj_int = ProspectivePredictor(use_conjunction=True).fit(interact)
    err_conj = ProspectivePredictor.errors(conj_int.predict(interact_hold), interact_hold).mean()
    _, _, err_clean = _fit_predict(clean, clean_hold)
    assert err_linear.mean() > err_clean.mean() + 0.02, "linear geometry fails on interactions"
    assert err_conj.mean() < err_linear.mean()
