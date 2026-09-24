from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace

import pytest

from signac_100m.phase1_eval import (
    BASELINE_GATE_SCHEMA,
    INTERFERENCE_RETRIEVAL_GRID,
    PHASE1_AXES,
    PHASE1_PAIR_FAMILY_KINDS,
    PHASE1_PAIR_KIND_CATEGORIES,
    SUMMARY_SCHEMA,
    PhaseOneBaselineGateSpec,
    assess_phase1_baseline,
)
from signac_100m.rsi import (
    BoundedRSIPolicy,
    INTERVENTION_SPEC_SCHEMA,
    POLICY_SCHEMA,
    RegisteredCurriculumTrial,
    analyze_registered_curriculum_trial,
    build_training_family_coverage_report,
    propose_bounded_curriculum_change,
    validate_registered_curriculum_result,
    validate_rsi_proposal,
)


def _sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def _wilson(successes: int, cases: int = 10) -> list[float]:
    z = 1.959963984540054
    proportion = successes / cases
    denominator = 1 + z * z / cases
    center = (proportion + z * z / (2 * cases)) / denominator
    margin = z * ((proportion * (1 - proportion) / cases + z * z / (4 * cases * cases)) ** 0.5) / denominator
    return [max(0.0, center - margin), min(1.0, center + margin)]


def _summary(seed: int, weakest_axis: str = "composition", weakest_successes: int = 2):
    def rate_record(successes: int, cases: int) -> dict[str, object]:
        return {
            "successes": successes,
            "cases": cases,
            "rate": successes / cases,
            "wilson95": _wilson(successes, cases),
        }

    interference_case_count = 2 * sum(
        len(quartiles) for quartiles in INTERFERENCE_RETRIEVAL_GRID.values()
    )
    family_summary = {}
    axes = {}
    for axis, families in PHASE1_AXES.items():
        if axis == "interference_retrieval":
            rows = {
                family: {
                    "exact": rate_record(interference_case_count, interference_case_count),
                    "valid_eos": rate_record(interference_case_count, interference_case_count),
                    "exact_and_eos": rate_record(interference_case_count, interference_case_count),
                }
                for family in families
            }
        else:
            axis_exact_and_eos = weakest_successes if axis == weakest_axis else 8
            rows = {
                family: {
                    "exact": rate_record(max(8, axis_exact_and_eos), 10),
                    "valid_eos": rate_record(10, 10),
                    "exact_and_eos": rate_record(axis_exact_and_eos, 10),
                }
                for family in families
            }
        family_summary.update(rows)
        axes[axis] = {"families": list(families)}
        for metric in ("exact", "valid_eos", "exact_and_eos"):
            successes = sum(rows[family][metric]["successes"] for family in families)
            cases = sum(rows[family][metric]["cases"] for family in families)
            axes[axis][metric] = rate_record(successes, cases)

    pair_details = {}
    for category, kinds in PHASE1_PAIR_KIND_CATEGORIES.items():
        groups = [
            (family, kind)
            for family, family_kinds in PHASE1_PAIR_FAMILY_KINDS.items()
            for kind in family_kinds
            if kind in kinds
        ]
        base_cases, extra_cases = divmod(10, len(groups))
        for index, (family, kind) in enumerate(groups):
            cases = base_cases + int(index < extra_cases)
            rate = {
                "successes": cases,
                "cases": cases,
                "rate": 1.0,
                "wilson95": _wilson(cases, cases),
            }
            pair_details.setdefault(family, {})[kind] = {
                **rate,
                "valid_eos_pair_rate": dict(rate),
            }
    interference_grid = {
        str(dose): {
            str(quartile): {
                "exact_and_eos": {
                    "successes": 2, "cases": 2, "rate": 1.0, "wilson95": _wilson(2, 2)
                },
                "valid_eos": {
                    "successes": 2, "cases": 2, "rate": 1.0, "wilson95": _wilson(2, 2)
                },
            }
            for quartile in quartiles
        }
        for dose, quartiles in INTERFERENCE_RETRIEVAL_GRID.items()
    }
    report = {
        "schema": SUMMARY_SCHEMA,
        "status": "MEASUREMENT_ONLY",
        "training_authorized": False,
        "split": "development",
        "seed": seed + 1000,
        "training_seed": seed,
        "evaluation_seed": seed + 1000,
        "identities": {
            "model_spec_sha256": "a" * 64,
            "suite_sha256": hashlib.sha256(f"suite-{seed}".encode()).hexdigest(),
            "evaluation_receipt_sha256": hashlib.sha256(f"receipt-{seed}".encode()).hexdigest(),
            "evaluator_sha256": "b" * 64,
            "generator_sha256": "d" * 64,
            "checkpoint_sha256": hashlib.sha256(f"checkpoint-{seed}".encode()).hexdigest(),
            "training_recipe_sha256": "f" * 64,
            "tokenizer_sha256": "1" * 64,
            "training_spec_sha256": "2" * 64,
            "data_manifest_sha256": "3" * 64,
            "pack_manifest_sha256": "4" * 64,
            "optimizer_spec_sha256": "5" * 64,
            "schedule_spec_sha256": "6" * 64,
            "curriculum_spec_sha256": "7" * 64,
            "source_tree_sha256": "8" * 64,
            "receipt_subject_manifest_sha256": hashlib.sha256(f"subject-{seed}".encode()).hexdigest(),
            "subject_manifest_sha256": hashlib.sha256(f"subject-{seed}".encode()).hexdigest(),
            "adapter_sha256": hashlib.sha256(f"adapter-{seed}".encode()).hexdigest(),
        },
        "metrics": {
            "by_skill_axis": axes,
            "by_family": family_summary,
            "causal_pairs": {
                kind: {
                    "wilson95": _wilson(10), "cases": 10, "successes": 10,
                    "valid_eos_pair_rate": {"wilson95": _wilson(10), "cases": 10, "successes": 10},
                }
                for kind in ("sensitivity", "invariance")
            },
            "causal_pairs_by_family_and_kind": pair_details,
            "interference_retrieval_grid": interference_grid,
        },
    }
    report["sha256"] = _sha(report)
    return report


def _policy(**overrides):
    fields = dict(
        schema=POLICY_SCHEMA,
        preregistration_sha256="c" * 64,
        minimum_independent_seeds=2,
        action_threshold_lcb=0.4,
    )
    fields.update(overrides)
    return BoundedRSIPolicy(**fields)


def _baseline_gate(reports):
    spec = PhaseOneBaselineGateSpec(
        schema=BASELINE_GATE_SCHEMA,
        preregistration_sha256="c" * 64,
        minimum_independent_seeds=2,
        minimum_cases_per_axis=1,
        primary_axis="identity",
        primary_floor_upper_max=0.0,
        primary_ceiling_lower_min=1.0,
        run_level_familywise_alpha=0.05,
        maximum_run_level_interval_width=0.5,
        minimum_exact_and_eos_lcb=0.0,
        minimum_eos_lcb=0.0,
        minimum_sensitivity_lcb=0.0,
        minimum_invariance_lcb=0.0,
    )
    return assess_phase1_baseline(reports, spec)


def _baseline_spec():
    return PhaseOneBaselineGateSpec(
        schema=BASELINE_GATE_SCHEMA,
        preregistration_sha256="c" * 64,
        minimum_independent_seeds=2,
        minimum_cases_per_axis=1,
        primary_axis="identity",
        primary_floor_upper_max=0.0,
        primary_ceiling_lower_min=1.0,
        run_level_familywise_alpha=0.05,
        maximum_run_level_interval_width=0.5,
        minimum_exact_and_eos_lcb=0.0,
        minimum_eos_lcb=0.0,
        minimum_sensitivity_lcb=0.0,
        minimum_invariance_lcb=0.0,
    )


def _weights():
    axes = list(PHASE1_AXES)
    quotient, remainder = divmod(1_000_000, len(axes))
    return {
        axis: quotient + int(index < remainder)
        for index, axis in enumerate(axes)
    }


def _intervention_spec(**overrides):
    fields = dict(
        schema=INTERVENTION_SPEC_SCHEMA,
        preregistration_sha256="8" * 64,
        randomization_receipt_sha256="9" * 64,
        training_family="heldout_rule_induction",
        target_axis="composition",
        target_metric="exact_and_eos",
        control_curriculum_sha256="a" * 64,
        treatment_curriculum_sha256="b" * 64,
        minimum_matched_pairs=50,
        familywise_alpha=0.05,
        maximum_interval_width=0.8,
        minimum_target_effect=0.2,
        maximum_non_target_harm=0.4,
    )
    fields.update(overrides)
    return RegisteredCurriculumTrial(**fields)


def _intervention_arm(seed: int, arm: str, curriculum_sha256: str, composition_successes: int):
    summary = _summary(seed, weakest_successes=composition_successes)
    identity = summary["identities"]
    identity["curriculum_spec_sha256"] = curriculum_sha256
    identity["training_recipe_sha256"] = _sha(f"recipe-{arm}")
    identity["checkpoint_sha256"] = _sha(f"checkpoint-{seed}-{arm}")
    identity["evaluation_receipt_sha256"] = _sha(f"receipt-{seed}-{arm}")
    subject_sha256 = _sha(f"subject-{seed}-{arm}")
    identity["subject_manifest_sha256"] = subject_sha256
    identity["receipt_subject_manifest_sha256"] = subject_sha256
    identity["adapter_sha256"] = _sha(f"adapter-{seed}-{arm}")
    summary["sha256"] = _sha({key: value for key, value in summary.items() if key != "sha256"})
    return summary


@pytest.mark.parametrize("threshold", [True, False, "0.4", float("nan"), float("inf"), -0.1, 1.1])
def test_rsi_policy_rejects_invalid_action_thresholds(threshold):
    with pytest.raises(ValueError, match=r"threshold must lie in \[0, 1\]"):
        _policy(action_threshold_lcb=threshold).assert_valid()


@pytest.mark.parametrize("threshold", [0, 0.4, 1, 1.0])
def test_rsi_policy_accepts_finite_numeric_action_thresholds(threshold):
    _policy(action_threshold_lcb=threshold).assert_valid()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("run_level_familywise_alpha", True),
        ("run_level_familywise_alpha", 0.0),
        ("run_level_familywise_alpha", 1.0),
        ("run_level_familywise_alpha", float("nan")),
        ("maximum_run_level_interval_width", 0.0),
        ("maximum_run_level_interval_width", float("inf")),
    ],
)
def test_baseline_spec_rejects_unfrozen_or_invalid_run_level_thresholds(field, value):
    with pytest.raises(ValueError):
        replace(_baseline_spec(), **{field: value}).assert_valid()


def test_baseline_rejects_summaries_from_different_source_trees():
    first, second = _summary(1), _summary(2)
    second["identities"]["source_tree_sha256"] = "9" * 64
    second["sha256"] = _sha({key: value for key, value in second.items() if key != "sha256"})
    with pytest.raises(ValueError, match="source tree"):
        assess_phase1_baseline([first, second], _baseline_spec())


def test_rsi_reports_weak_axis_but_abstains_until_mapping_and_inference_exist():
    reports = [_summary(seed) for seed in range(100, 200)]
    gate = _baseline_gate(reports)
    uncertainty = gate["run_level_uncertainty"]
    proposal = propose_bounded_curriculum_change(
        summaries=reports,
        baseline_gate=gate,
        baseline_gate_spec=_baseline_spec(),
        current_skill_axis_weights_ppm=_weights(),
        policy=_policy(),
    )
    validate_rsi_proposal(proposal)
    assert gate["decision"] == "DIAGNOSTIC_NONFLOOR_NONCEILING"
    assert gate["mechanism_experiment_eligible"] is False
    assert gate["cluster_aware_uncertainty_available"] is True
    assert uncertainty["method"] == "SIMULTANEOUS_TWO_SIDED_HOEFFDING_BONFERRONI"
    assert uncertainty["independent_unit"].startswith("trained-subject/evaluation-surface pair")
    assert "not independently audited" in uncertainty["independence_basis"]
    assert uncertainty["subject_count"] == 100
    assert uncertainty["simultaneous_endpoint_count"] == 2 * len(PHASE1_AXES) + 4
    assert uncertainty["by_skill_axis"]["composition"]["exact_and_eos"]["mean_subject_rate"] == 0.2
    composition_interval = uncertainty["by_skill_axis"]["composition"]["exact_and_eos"]["simultaneous_interval"]
    assert 0.0 < composition_interval[0] < 0.2 < composition_interval[1] < 1.0
    assert uncertainty["causal_pairs"]["sensitivity"]["valid_eos_pair_rate"]["mean_subject_rate"] == 1.0
    assert proposal["decision"] == "ABSTAIN"
    assert proposal["diagnostic_weakest_axis"] == "composition"
    assert proposal["selected_axis"] is None
    assert proposal["split_scope"] == "development_measurement_only"
    assert proposal["skill_axis_weights_proposed_ppm"] == proposal["skill_axis_weights_before_ppm"]
    assert proposal["abstain_reason"] == "prospective_intervention_effect_map_and_randomized_comparison_required"
    assert proposal["comparison_design"]["proposal_applicable"] is False
    assert proposal["training_family_coverage"]["training_effect_claim"] is False
    assert proposal["training_family_coverage"]["proposal_eligible"] is False


def test_run_level_bounds_tighten_with_more_distinct_trained_subjects():
    reports = [_summary(seed) for seed in range(200, 300)]
    gate = assess_phase1_baseline(reports, _baseline_spec())
    bounds = gate["run_level_uncertainty"]["by_skill_axis"]["composition"]["exact_and_eos"]

    assert gate["cluster_aware_uncertainty_available"] is True
    assert gate["run_level_uncertainty"]["subject_count"] == 100
    assert bounds["mean_subject_rate"] == pytest.approx(0.2)
    assert 0.0 < bounds["simultaneous_interval"][0] < 0.2
    assert 0.2 < bounds["simultaneous_interval"][1] < 1.0


def test_wide_run_level_intervals_force_inconclusive_gate():
    reports = [_summary(1), _summary(2)]
    gate = assess_phase1_baseline(reports, _baseline_spec())

    assert gate["decision"] == "BORDERLINE_DIAGNOSTIC"
    assert "wider than the preregistered precision limit" in gate["reason"]
    assert gate["cluster_aware_uncertainty_available"] is True
    assert gate["mechanism_experiment_eligible"] is False


def test_rsi_matched_trial_rejects_different_source_trees():
    spec = _intervention_spec()
    control = _intervention_arm(399, "control", spec.control_curriculum_sha256, 2)
    treatment = _intervention_arm(399, "treatment", spec.treatment_curriculum_sha256, 10)
    treatment["identities"]["source_tree_sha256"] = "9" * 64
    treatment["sha256"] = _sha({
        key: value for key, value in treatment.items() if key != "sha256"
    })
    with pytest.raises(ValueError, match="exposed non-curriculum training component"):
        analyze_registered_curriculum_trial(
            spec=spec,
            matched_pairs=[{"pair_id": "source-tree-mismatch", "control": control, "treatment": treatment}],
        )


def test_registered_curriculum_trial_estimates_matched_effect_but_never_changes_training():
    spec = _intervention_spec()
    pairs = [
        {
            "pair_id": f"composition-{seed}",
            "control": _intervention_arm(seed, "control", spec.control_curriculum_sha256, 2),
            "treatment": _intervention_arm(seed, "treatment", spec.treatment_curriculum_sha256, 10),
        }
        for seed in range(400, 500)
    ]

    report = analyze_registered_curriculum_trial(spec=spec, matched_pairs=pairs)
    validate_registered_curriculum_result(report, spec=spec, matched_pairs=pairs)
    target = report["target_effect"]

    assert report["decision"] == "DEVELOPMENT_EFFECT_SIGNAL"
    assert report["matched_pair_count"] == 100
    assert target["mean_paired_difference"] == pytest.approx(0.8)
    assert target["simultaneous_interval"][0] >= spec.minimum_target_effect
    assert report["randomization_and_custody_audited"] is False
    assert report["trainer_mutation_authorized"] is False
    assert report["training_authorized"] is False
    assert report["promotion_authorized"] is False
    assert "fresh sealed matched seeds" in report["next_action"]

    tampered = copy.deepcopy(report)
    tampered["target_effect"]["mean_paired_difference"] = 0.9
    with pytest.raises(ValueError, match="does not match its frozen inputs"):
        validate_registered_curriculum_result(tampered, spec=spec, matched_pairs=pairs)


def test_registered_curriculum_trial_rejects_wide_small_sample_and_unmapped_target():
    spec = _intervention_spec(minimum_matched_pairs=2)
    pairs = [
        {
            "pair_id": f"small-{seed}",
            "control": _intervention_arm(seed, "control", spec.control_curriculum_sha256, 2),
            "treatment": _intervention_arm(seed, "treatment", spec.treatment_curriculum_sha256, 10),
        }
        for seed in (510, 511)
    ]
    report = analyze_registered_curriculum_trial(spec=spec, matched_pairs=pairs)
    assert report["decision"] == "INCONCLUSIVE_WIDE_INTERVALS"
    assert all(value is False for value in (
        report["trainer_mutation_authorized"],
        report["training_authorized"],
        report["promotion_authorized"],
    ))

    unmapped = _intervention_spec(training_family="counterfactual_sensitivity")
    with pytest.raises(ValueError, match="lacks direct frozen training-family coverage"):
        unmapped.assert_valid()


def test_rsi_abstains_without_independent_seed_evidence():
    proposal = propose_bounded_curriculum_change(
        summaries=[_summary(1)],
        baseline_gate=_baseline_gate([_summary(1)]),
        baseline_gate_spec=_baseline_spec(),
        current_skill_axis_weights_ppm=_weights(),
        policy=_policy(),
    )
    assert proposal["decision"] == "ABSTAIN"
    assert proposal["selected_axis"] is None
    assert proposal["skill_axis_weights_proposed_ppm"] == proposal["skill_axis_weights_before_ppm"]
    assert proposal["abstain_reason"] == "baseline_gate_not_informative"


def test_rsi_refuses_non_development_or_tampered_summaries():
    report = _summary(1)
    sealed = copy.deepcopy(report)
    sealed["split"] = "sealed"
    sealed["sha256"] = _sha({key: value for key, value in sealed.items() if key != "sha256"})
    with pytest.raises(ValueError, match="development summaries"):
        propose_bounded_curriculum_change(
            summaries=[sealed, _summary(2)],
            baseline_gate=_baseline_gate([sealed, _summary(2)]),
            baseline_gate_spec=_baseline_spec(),
            current_skill_axis_weights_ppm=_weights(),
            policy=_policy(),
        )
    report["metrics"]["by_skill_axis"]["identity"]["exact_and_eos"]["wilson95"][0] = 0.0
    with pytest.raises(ValueError, match="hash mismatch"):
        propose_bounded_curriculum_change(
            summaries=[report, _summary(2)],
            baseline_gate=_baseline_gate([report, _summary(2)]),
            baseline_gate_spec=_baseline_spec(),
            current_skill_axis_weights_ppm=_weights(),
            policy=_policy(),
        )


def test_rsi_reports_no_weak_axis_but_still_abstains_from_training_action():
    reports = [_summary(seed, weakest_successes=10) for seed in range(300, 400)]
    proposal = propose_bounded_curriculum_change(
        summaries=reports,
        baseline_gate=_baseline_gate(reports),
        baseline_gate_spec=_baseline_spec(),
        current_skill_axis_weights_ppm=_weights(),
        policy=_policy(),
    )
    assert proposal["decision"] == "ABSTAIN"
    assert proposal["diagnostic_weakest_axis"] is None
    assert proposal["abstain_reason"] == "prospective_intervention_effect_map_and_randomized_comparison_required"


def test_rsi_coverage_report_distinguishes_direct_pooled_proxy_and_missing_families():
    report = build_training_family_coverage_report()
    rows = {row["training_family"]: row for row in report["rows"]}

    assert len(rows) == 9
    assert rows["identity_copy"]["coverage_status"] == "DIRECT_POOLED_AXIS"
    assert rows["heldout_rule_induction"]["skill_axes"] == ["composition"]
    assert rows["counterfactual_sensitivity"]["coverage_status"] == "FAMILY_SPECIFIC_PAIR_PROXY"
    assert "metrics.causal_pairs_by_family_and_kind.counterfactual_premise.relevant_fact_swap" in (
        rows["counterfactual_sensitivity"]["metric_paths"]
    )
    assert rows["interference_retrieval"]["coverage_status"] == "DIRECT_STRESS_GRID"
    assert rows["interference_retrieval"]["skill_axes"] == ["interference_retrieval"]
    assert rows["faithful_realization"]["coverage_status"] == "DIRECT_POOLED_AXIS"
    assert rows["faithful_realization"]["evaluation_families"] == ["faithful_realization"]
    assert all(row["training_effect_claim"] is False for row in report["rows"])
    rows["identity_copy"]["evaluation_families"].append("tampered")
    rebuilt = build_training_family_coverage_report()
    assert rebuilt["rows"][0]["evaluation_families"] == [
        "exact_contextual_copy", "nonce_identifier_retrieval"
    ]


def test_rsi_rejects_coverage_changes_even_when_outer_proposal_hash_is_recomputed():
    reports = [_summary(1), _summary(2)]
    proposal = propose_bounded_curriculum_change(
        summaries=reports,
        baseline_gate=_baseline_gate(reports),
        baseline_gate_spec=_baseline_spec(),
        current_skill_axis_weights_ppm=_weights(),
        policy=_policy(),
    )
    proposal["training_family_coverage"]["rows"][-1]["limitations"] = "changed"
    proposal["sha256"] = _sha({key: value for key, value in proposal.items() if key != "sha256"})

    with pytest.raises(ValueError, match="coverage report differs"):
        validate_rsi_proposal(proposal)


def test_rsi_refuses_a_gate_bound_to_different_summaries():
    reports = [_summary(1), _summary(2)]
    other_gate = _baseline_gate([_summary(1), _summary(3)])
    with pytest.raises(ValueError, match="does not match recomputed summary diagnostics"):
        propose_bounded_curriculum_change(
            summaries=reports,
            baseline_gate=other_gate,
            baseline_gate_spec=_baseline_spec(),
            current_skill_axis_weights_ppm=_weights(),
            policy=_policy(),
        )


def test_rsi_abstains_when_preregistered_baseline_is_not_informative():
    report = _summary(1)
    gate = _baseline_gate([report])
    proposal = propose_bounded_curriculum_change(
        summaries=[report],
        baseline_gate=gate,
        baseline_gate_spec=_baseline_spec(),
        current_skill_axis_weights_ppm=_weights(),
        policy=_policy(),
    )
    assert gate["decision"] == "INCONCLUSIVE_INSUFFICIENT_TRAINING_SEEDS"
    assert proposal["decision"] == "ABSTAIN"
    assert proposal["baseline_gate_sha256"] == gate["sha256"]
    assert proposal["abstain_reason"] == "baseline_gate_not_informative"


def test_different_evaluation_seeds_do_not_count_as_independent_training_runs():
    first = _summary(1)
    second = _summary(1)
    second["seed"] = 2002
    second["evaluation_seed"] = 2002
    for key in ("suite_sha256", "checkpoint_sha256", "evaluation_receipt_sha256"):
        second["identities"][key] = hashlib.sha256(f"second-{key}".encode()).hexdigest()
    second["sha256"] = _sha({key: value for key, value in second.items() if key != "sha256"})

    gate = assess_phase1_baseline([first, second], _baseline_spec())

    assert gate["decision"] == "INCONCLUSIVE_INSUFFICIENT_TRAINING_SEEDS"
    assert gate["independent_training_seed_count"] == 1
    assert gate["independent_evaluation_seed_count"] == 2
    assert gate["cluster_aware_uncertainty_available"] is False
    assert gate["run_level_uncertainty"] is None
    assert gate["mechanism_experiment_eligible"] is False


def test_summary_verifier_recomputes_wilson_bounds_from_reported_counts():
    report = _summary(1)
    report["metrics"]["by_skill_axis"]["identity"]["exact_and_eos"]["wilson95"] = [0.0, 1.0]
    report["sha256"] = _sha({key: value for key, value in report.items() if key != "sha256"})
    with pytest.raises(ValueError, match="interval does not match its counts"):
        assess_phase1_baseline([report], _baseline_spec())


def test_summary_verifier_rejects_tampered_family_pair_counts():
    report = _summary(1)
    metric = report["metrics"]["causal_pairs_by_family_and_kind"][
        "counterfactual_premise"
    ]["relevant_fact_swap"]
    metric["cases"] += 1
    report["sha256"] = _sha({key: value for key, value in report.items() if key != "sha256"})

    with pytest.raises(ValueError, match="counts are inconsistent"):
        assess_phase1_baseline([report], _baseline_spec())


def test_summary_verifier_rejects_receipt_bound_to_a_different_subject_manifest():
    report = _summary(1)
    report["identities"]["receipt_subject_manifest_sha256"] = "8" * 64
    report["sha256"] = _sha({key: value for key, value in report.items() if key != "sha256"})

    with pytest.raises(ValueError, match="different subject manifest"):
        assess_phase1_baseline([report], _baseline_spec())


def test_summary_verifier_requires_every_interference_grid_cell():
    report = _summary(1)
    del report["metrics"]["interference_retrieval_grid"]["32"]["4"]
    report["sha256"] = _sha({key: value for key, value in report.items() if key != "sha256"})

    with pytest.raises(ValueError, match="quartile grid is incomplete"):
        assess_phase1_baseline([report], _baseline_spec())
