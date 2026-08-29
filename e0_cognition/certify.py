"""Development certification entry point for the E0 benchmark contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from .baselines import evaluate_all_baselines
from .contracts import PairKind, Split, assert_split_disjoint
from .evaluation_generators import build_evaluation_suite
from .reference_solvers import assert_reference_solver_agreement
from .statistics import (
    approximate_two_proportion_n_per_arm,
    uniform_candidate_chance,
    wilson_interval,
)
from .training_generators import assert_training_eval_disjoint, build_training_examples


DEFAULT_DEVELOPMENT_SEED = 271828


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
            baselines["broken_state_tracker"]["by_family"]["state_overwrite"] < 0.25
        ),
        "direct_retrieval_does_not_solve_composition": all(
            baselines["direct_retrieval_control"]["by_family"][family] < 0.75
            for family in ("relation_2_hop", "relation_3_hop", "natural_composition_analogue")
        ),
        "random_control_matches_calculated_chance": abs(random_result["accuracy"] - chance) < 0.08,
        "no_dominant_answer_position": maximum_answer_position_share < 0.40,
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    return {
        "schema": "esoes-e0-development-certificate/v1",
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
        },
        "checks": checks,
        "baselines": baselines,
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
