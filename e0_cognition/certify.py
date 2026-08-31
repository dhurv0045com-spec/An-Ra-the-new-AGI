"""Development certification entry point for the E0 benchmark contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

from .baselines import BASELINES, evaluate_all_baselines
from .contracts import Split, assert_split_disjoint
from .evaluation_generators import build_evaluation_suite
from .reference_solvers import assert_reference_solver_agreement
from .preregistration import PROTOCOL, protocol_sha256
from .statistics import (
    approximate_two_proportion_n_per_arm,
    uniform_candidate_chance,
    wilson_interval,
)
from .training_generators import assert_training_eval_disjoint, build_training_examples
from .metrics import selection_eligible


DEFAULT_DEVELOPMENT_SEED = 271828


def _subset_baseline(suite, baseline_name: str, families: set[str]) -> dict[str, float | int]:
    baseline = BASELINES[baseline_name]
    cases = [case for case in suite.cases if case.family in families]
    if not cases:
        raise AssertionError(f"no cases for baseline subset {sorted(families)}")
    correct = sum(baseline(case) == case.answer for case in cases)
    chance = sum(1.0 / len(case.candidates) for case in cases) / len(cases)
    return {"accuracy": correct / len(cases), "chance": chance, "cases": len(cases)}


def _aggregate_shortcut_probe(
    *, split: Split, seed: int, groups_per_family: int, baseline_name: str, families: set[str]
) -> dict[str, float | int]:
    """Pool independent generator seeds before judging a shortcut.

    A single shuffled suite can be noisy: the red-team gate is therefore based
    on a preregistered multi-seed pool, while the canonical receipt still keeps
    the primary development suite hash and metrics.
    """

    suites = [
        build_evaluation_suite(split, seed=seed + offset, groups_per_family=groups_per_family)
        for offset in range(8)
    ]
    baseline = BASELINES[baseline_name]
    cases = [case for suite in suites for case in suite.cases if case.family in families]
    correct = sum(baseline(case) == case.answer for case in cases)
    chance = sum(1.0 / len(case.candidates) for case in cases) / len(cases)
    # Position heuristics do not have uniform-candidate chance. Under a random
    # serialization the final candidate-bearing fact is uniform over all such
    # facts, so enumerate those possible winners analytically. This is exactly
    # equivalent to enumerating every fact permutation and remains tractable as
    # state histories grow. Other heuristics retain uniform-candidate chance.
    if baseline_name in {"latest_fact", "nearest_position"}:
        null_correct = 0.0
        for case in cases:
            candidate_facts = tuple(
                fact for fact in case.facts if any(candidate in fact for candidate in case.candidates)
            )
            if not candidate_facts:
                null_correct += float(case.candidates[0] == case.answer)
                continue
            hits = sum(
                baseline(replace(case, facts=(fact,))) == case.answer
                for fact in candidate_facts
            )
            null_correct += hits / len(candidate_facts)
        calibrated_chance = null_correct / len(cases)
        null_method = "analytic-random-serialization"
    else:
        calibrated_chance = chance
        null_method = "casewise-uniform-candidate"
    return {
        "accuracy": correct / len(cases),
        "chance": chance,
        "calibrated_chance": calibrated_chance,
        "cases": len(cases),
        "seeds": 8,
        "null_method": null_method,
    }


def build_development_certificate(*, seed: int, groups_per_family: int) -> dict[str, object]:
    dev = build_evaluation_suite(Split.DEVELOPMENT, seed=seed, groups_per_family=groups_per_family)
    # Test-only sentinels prove the code enforces namespace isolation. They are not sealed fixtures.
    sealed_sentinel = build_evaluation_suite(Split.SEALED, seed=314159, groups_per_family=2)
    fresh_sentinel = build_evaluation_suite(Split.FRESH, seed=161803, groups_per_family=2)
    assert_split_disjoint((dev, sealed_sentinel, fresh_sentinel))
    assert_reference_solver_agreement(dev)
    training = build_training_examples(seed=seed + 1, count=max(64, groups_per_family * 8))
    assert_training_eval_disjoint(training, {case.template_id for case in dev.cases})
    baselines = evaluate_all_baselines(dev)
    pair_histogram = dict(sorted(Counter(pair.kind.value for pair in dev.pairs).items()))
    chance = uniform_candidate_chance(dev)
    random_result = baselines["deterministic_random"]
    random_interval = wilson_interval(random_result["correct"], random_result["total"])
    answer_positions = Counter(
        case.candidates.index(case.answer) for case in dev.cases if len(case.candidates) > 1
    )
    maximum_answer_position_share = max(answer_positions.values()) / sum(answer_positions.values())
    surface_axes = dev.surface_axis_histograms()
    difficulty_axes = dev.difficulty_axis_histograms()
    state_families = {"state_overwrite", "natural_state_analogue"}
    state_shortcuts = {
        name: _aggregate_shortcut_probe(
            split=Split.DEVELOPMENT,
            seed=seed + 1_000,
            groups_per_family=groups_per_family,
            baseline_name=name,
            families=state_families,
        )
        for name in (
            "first_candidate",
            "last_candidate",
            "latest_fact",
            "nearest_position",
            "lexical_overlap",
            "bag_of_words",
        )
    }
    rule_shortcuts = {
        name: _aggregate_shortcut_probe(
            split=Split.DEVELOPMENT,
            seed=seed + 2_000,
            groups_per_family=groups_per_family,
            baseline_name=name,
            families={"rule_induction"},
        )
        for name in (
            "bag_of_words",
            "fixed_reverse_rule",
            "fixed_identity_rule",
            "fixed_repeat_left_rule",
            "fixed_repeat_right_rule",
        )
    }
    state_shortcut_ceiling = max(
        result["calibrated_chance"] + 0.10 for result in state_shortcuts.values()
    )
    rule_shortcut_ceiling = max(
        result["calibrated_chance"] + 0.10 for result in rule_shortcuts.values()
    )
    pair_effects = dev.pair_effect_histogram()
    rule_structures = {
        value
        for case in dev.cases
        if case.family == "rule_induction"
        for axis, value in case.surface_axes
        if axis == "rule_structure"
    }
    checks = {
        "suite_contracts": True,
        "counterfactual_pairs_mechanical": len(dev.pairs) > 0,
        "split_namespaces_disjoint": True,
        "independent_surface_solver_agreement": True,
        "training_eval_templates_disjoint": True,
        "hidden_truth_excluded_from_model_view": all(
            set(case.model_view()) == {"context", "query", "prompt"} for case in dev.cases
        ),
        "random_baseline_below_50pct": baselines["deterministic_random"]["accuracy"] < 0.50,
        "full_oracle_exact": baselines["full_truth_oracle"]["accuracy"] == 1.0,
        "broken_state_tracker_fails_state": (
            baselines["broken_state_tracker"]["by_family"]["state_overwrite"] < 0.50
        ),
        "direct_retrieval_does_not_solve_composition": all(
            baselines["direct_retrieval_control"]["by_family"][family] < 0.75
            for family in ("relation_2_hop", "relation_3_hop", "natural_composition_analogue")
        ),
        "random_control_matches_calculated_chance": abs(random_result["accuracy"] - chance) < 0.08,
        "no_dominant_answer_position": maximum_answer_position_share < 0.40,
        "context_position_axes_covered": {
            "front", "middle", "back", "distributed", "answer-absent"
        }.issubset(surface_axes["relevant_position"]),
        "output_format_axes_covered": len(surface_axes["answer_format"]) >= 5,
        "state_semantic_query_axes_covered": {
            "latest", "intermediate", "rollback", "precedence"
        }.issubset(surface_axes["state_query"]),
        "state_serialization_is_shuffled": surface_axes["serialization"].get(
            "semantic-shuffled", 0
        ) > 0,
        "state_shortcut_heuristics_fail": all(
            result["accuracy"] <= result["calibrated_chance"] + 0.10
            for result in state_shortcuts.values()
        ),
        "multiple_rule_structures": len(rule_structures) >= 6,
        "rule_shortcuts_fail": all(
            result["accuracy"] <= result["calibrated_chance"] + 0.10
            for result in rule_shortcuts.values()
        ),
        "sensitivity_and_invariance_pairs_present": (
            pair_effects.get("sensitivity", 0) > 0
            and pair_effects.get("invariance", 0) > 0
        ),
        "difficulty_axes_covered": {"cardinality", "hops", "distractors"}.issubset(
            difficulty_axes
        ),
        "natural_analogues_present": all(
            family in dev.family_histogram()
            for family in (
                "natural_binding_analogue",
                "natural_state_analogue",
                "natural_composition_analogue",
            )
        ),
        "copy_is_not_selection_control": all(
            not selection_eligible(case)
            for case in dev.cases
            if case.family == "exact_contextual_copy"
        ),
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    return {
        "schema": "esoes-e0-development-certificate/v2",
        "status": status,
        "scope": "development infrastructure only; not a V5 model result",
        "suite": {
            "split": dev.split.value,
            "generator_version": dev.generator_version,
            "seed": seed,
            "groups_per_family": groups_per_family,
            "cases": len(dev.cases),
            "pairs": len(dev.pairs),
            "sha256": dev.sha256(),
            "family_histogram": dev.family_histogram(),
            "pair_histogram": pair_histogram,
            "surface_axis_histograms": surface_axes,
            "difficulty_axis_histograms": difficulty_axes,
            "pair_effect_histogram": pair_effects,
            "rule_structures": sorted(rule_structures),
        },
        "checks": checks,
        "baselines": baselines,
        "shortcut_audit": {
            "state_families": sorted(state_families),
            "state_heuristics": state_shortcuts,
            "state_shortcut_ceiling": state_shortcut_ceiling,
            "rule_induction_heuristics": rule_shortcuts,
            "rule_shortcut_ceiling": rule_shortcut_ceiling,
            "policy": "every named heuristic must remain within its calibrated null + 10 percentage points",
        },
        "statistical_calibration": {
            "uniform_candidate_chance": chance,
            "deterministic_random_wilson_95": random_interval,
            "answer_position_histogram_multi_candidate": dict(sorted(answer_positions.items())),
            "maximum_answer_position_share": maximum_answer_position_share,
            "approx_n_per_arm_to_detect_chance_plus_10pp_at_80pct_power": {
                "two_candidate": approximate_two_proportion_n_per_arm(0.5, 0.6),
                "four_candidate": approximate_two_proportion_n_per_arm(0.25, 0.35),
            },
            "method_note": "Two-sided normal approximation for planning; final gates use paired/bootstrap or exact methods preregistered per metric.",
        },
        "statistical_protocol": {
            "sha256": protocol_sha256(),
            "protocol": PROTOCOL,
        },
        "training_generator": {
            "examples_checked": len(training),
            "template_ids": sorted({example.template_id for example in training}),
        },
        "sealed_policy": {
            "seed_in_repository": False,
            "fixture_in_repository": False,
            "required_before_promotion": "external seed custody plus committed SHA-256 commitment",
            "sentinel_hash": hashlib.sha256(b"namespace-test-only-not-a-sealed-fixture").hexdigest(),
        },
        "limitations": [
            "No model has been evaluated by this certificate.",
            "Heuristic baselines are leak detectors, not claims of benchmark difficulty.",
            "The real sealed seed and fixture must be created under independent custody.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/e0/development_certificate.json"))
    parser.add_argument("--development-seed", type=int, default=DEFAULT_DEVELOPMENT_SEED)
    parser.add_argument("--groups-per-family", type=int, default=16)
    args = parser.parse_args()
    certificate = build_development_certificate(
        seed=args.development_seed, groups_per_family=args.groups_per_family
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(certificate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": certificate["status"], "output": str(args.output)}, sort_keys=True))
    return 0 if certificate["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
