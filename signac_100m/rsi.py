"""Bounded cognition diagnostics and preregistered development trial analysis.

This module can measure matched curriculum effects on development summaries,
but cannot propose or apply a training change. Any effect signal remains
conditional on external randomization/custody audit and fresh sealed replication.
"""

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from e0_cognition.training_generators import (
    TRAINING_COGNITION_FAMILIES,
    TRAINING_GENERATOR_VERSION,
)
from signac_100m.phase1_eval import (
    BASELINE_GATE_SCHEMA,
    INTERFERENCE_RETRIEVAL_GRID,
    PHASE1_AXES,
    PHASE1_PAIR_FAMILY_KINDS,
    PhaseOneBaselineGateSpec,
    assess_phase1_baseline,
    verify_phase1_summary,
)


POLICY_SCHEMA = "anra-signac-bounded-rsi-policy/v2"
PROPOSAL_SCHEMA = "anra-signac-rsi-curriculum-diagnostic/v5"
TRAINING_COVERAGE_SCHEMA = "anra-signac-training-evaluation-coverage/v3"
INTERVENTION_SPEC_SCHEMA = "anra-signac-matched-curriculum-intervention/v1"
INTERVENTION_RESULT_SCHEMA = "anra-signac-matched-curriculum-effect-report/v1"
MIXTURE_TOTAL_PPM = 1_000_000
IMMUTABLE_RESEARCH_COMPONENTS = (
    "model_spec",
    "tokenizer",
    "objective_and_eos_policy",
    "optimizer_and_learning_rate",
    "evaluator_and_thresholds",
    "development_sealed_fresh_splits",
)

# This is a measurement taxonomy, not an empirically estimated training-effect
# map. Rows distinguish family-specific evaluation from pooled axes, broad pair
# metrics, incidental proxies, and missing measurements.
_TRAINING_FAMILY_COVERAGE_ROWS: tuple[dict[str, Any], ...] = (
    {
        "training_family": "identity_copy",
        "coverage_status": "DIRECT_POOLED_AXIS",
        "evaluation_families": ["exact_contextual_copy", "nonce_identifier_retrieval"],
        "skill_axes": ["identity"],
        "metric_paths": [
            "metrics.by_family.<evaluation_family>.exact_and_eos",
            "metrics.by_skill_axis.identity.exact_and_eos",
        ],
        "limitations": "Family scores are reported, while the RSI gate uses a pooled identity axis.",
    },
    {
        "training_family": "query_binding",
        "coverage_status": "DIRECT_POOLED_AXIS",
        "evaluation_families": [
            "entity_value_binding", "matched_direct_retrieval", "natural_binding_analogue"
        ],
        "skill_axes": ["binding"],
        "metric_paths": [
            "metrics.by_family.<evaluation_family>.exact_and_eos",
            "metrics.by_skill_axis.binding.exact_and_eos",
        ],
        "limitations": "Family scores are reported, while the RSI gate pools three binding families.",
    },
    {
        "training_family": "semantic_state",
        "coverage_status": "DIRECT_POOLED_AXIS",
        "evaluation_families": ["state_overwrite", "natural_state_analogue"],
        "skill_axes": ["state_order"],
        "metric_paths": [
            "metrics.by_family.<evaluation_family>.exact_and_eos",
            "metrics.by_skill_axis.state_order.exact_and_eos",
        ],
        "limitations": "The RSI gate pools state overwrite and its naturalized analogue.",
    },
    {
        "training_family": "interference_retrieval",
        "coverage_status": "DIRECT_STRESS_GRID",
        "evaluation_families": ["interference_retrieval"],
        "skill_axes": ["interference_retrieval"],
        "metric_paths": [
            "metrics.by_skill_axis.interference_retrieval.exact_and_eos",
            "metrics.interference_retrieval_grid.<distractor_count>.<position_quartile>.exact_and_eos",
            "metrics.causal_pairs_by_family_and_kind.interference_retrieval.relevant_fact_swap",
        ],
        "limitations": (
            "Each dose/position cell has one counterfactual pair (two task rows), so this is sparse "
            "diagnostic coverage; it does not establish training effects or transfer beyond the suite."
        ),
    },
    {
        "training_family": "relational_composition",
        "coverage_status": "DIRECT_POOLED_AXIS",
        "evaluation_families": [
            "relation_2_hop", "relation_3_hop", "natural_composition_analogue"
        ],
        "skill_axes": ["composition"],
        "metric_paths": [
            "metrics.by_family.<evaluation_family>.exact_and_eos",
            "metrics.by_skill_axis.composition.exact_and_eos",
        ],
        "limitations": (
            "The measured relation families cover two and three hops; the pooled composition axis "
            "also contains rule induction, and does not measure the one-hop training condition."
        ),
    },
    {
        "training_family": "counterfactual_sensitivity",
        "coverage_status": "FAMILY_SPECIFIC_PAIR_PROXY",
        "evaluation_families": ["counterfactual_premise"],
        "skill_axes": [],
        "metric_paths": [
            "metrics.by_family.counterfactual_premise.exact_and_eos",
            "metrics.causal_pairs_by_family_and_kind.counterfactual_premise.relevant_fact_swap",
        ],
        "limitations": (
            "The isolated task changes a hypothetical premise, not the training family's explicit "
            "intervention operation; this generated pair diagnostic is not a training-effect estimate."
        ),
    },
    {
        "training_family": "heldout_rule_induction",
        "coverage_status": "DIRECT_POOLED_AXIS",
        "evaluation_families": ["rule_induction"],
        "skill_axes": ["composition"],
        "metric_paths": [
            "metrics.by_family.rule_induction.exact_and_eos",
            "metrics.by_skill_axis.composition.exact_and_eos",
        ],
        "limitations": "Rule induction is directly scored but pooled with relation-composition families.",
    },
    {
        "training_family": "missing_information",
        "coverage_status": "DIRECT_POOLED_AXIS",
        "evaluation_families": ["missing_information"],
        "skill_axes": ["missing_information"],
        "metric_paths": [
            "metrics.by_family.missing_information.exact_and_eos",
            "metrics.by_skill_axis.missing_information.exact_and_eos",
        ],
        "limitations": "This is a dedicated axis, but it does not establish broad calibrated abstention.",
    },
    {
        "training_family": "faithful_realization",
        "coverage_status": "DIRECT_POOLED_AXIS",
        "evaluation_families": ["faithful_realization"],
        "skill_axes": ["faithful_realization"],
        "metric_paths": [
            "metrics.by_family.faithful_realization.exact_and_eos",
            "metrics.by_skill_axis.faithful_realization.exact_and_eos",
            "metrics.causal_pairs_by_family_and_kind.faithful_realization.relevant_fact_swap",
        ],
        "limitations": (
            "The evaluator checks one exact payload/revision format with a relevant-fact pair. It does "
            "not cover broader formats, repetition, long-form fidelity, or training effects."
        ),
    },
)


def build_training_family_coverage_report() -> dict[str, Any]:
    """Return hash-bound descriptive coverage, never an RSI training action."""

    rows = deepcopy(_TRAINING_FAMILY_COVERAGE_ROWS)
    if tuple(row["training_family"] for row in rows) != tuple(TRAINING_COGNITION_FAMILIES):
        raise ValueError("RSI coverage rows must match the frozen training-family registry")
    known_axes = set(PHASE1_AXES)
    for row in rows:
        row["training_effect_claim"] = False
        if not set(row["skill_axes"]).issubset(known_axes):
            raise ValueError("RSI coverage row names an unknown Phase-One skill axis")
    report: dict[str, Any] = {
        "schema": TRAINING_COVERAGE_SCHEMA,
        "scope": "DESCRIPTIVE_MEASUREMENT_COVERAGE_ONLY",
        "training_generator_version": TRAINING_GENERATOR_VERSION,
        "training_families": list(TRAINING_COGNITION_FAMILIES),
        "evaluation_axes": {axis: list(families) for axis, families in PHASE1_AXES.items()},
        "evaluation_pair_families_and_kinds": {
            family: list(kinds) for family, kinds in PHASE1_PAIR_FAMILY_KINDS.items()
        },
        "evaluation_interference_grid": {
            str(dose): list(quartiles)
            for dose, quartiles in INTERFERENCE_RETRIEVAL_GRID.items()
        },
        "rows": rows,
        "training_effect_claim": False,
        "proposal_eligible": False,
        "sha256": "",
    }
    report["sha256"] = _sha256({key: value for key, value in report.items() if key != "sha256"})
    return report


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _assert_sha256(name: str, value: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256")


@dataclass(frozen=True, slots=True)
class BoundedRSIPolicy:
    """Prospectively frozen threshold for a weakest-axis diagnostic only."""

    schema: str
    preregistration_sha256: str
    minimum_independent_seeds: int
    action_threshold_lcb: float

    def assert_valid(self) -> None:
        if self.schema != POLICY_SCHEMA:
            raise ValueError("unsupported bounded RSI policy schema")
        _assert_sha256("RSI preregistration", self.preregistration_sha256)
        if type(self.minimum_independent_seeds) is not int or self.minimum_independent_seeds < 2:
            raise ValueError("RSI diagnostics require at least two distinct training seeds")
        threshold = self.action_threshold_lcb
        if (
            type(threshold) not in (int, float)
            or not math.isfinite(threshold)
            or not 0.0 <= threshold <= 1.0
        ):
            raise ValueError("RSI diagnostic threshold must lie in [0, 1]")

    def sha256(self) -> str:
        self.assert_valid()
        return _sha256(asdict(self))


@dataclass(frozen=True, slots=True)
class RegisteredCurriculumTrial:
    """Frozen matched-pair development comparison; never a trainer command."""

    schema: str
    preregistration_sha256: str
    randomization_receipt_sha256: str
    training_family: str
    target_axis: str
    target_metric: str
    control_curriculum_sha256: str
    treatment_curriculum_sha256: str
    minimum_matched_pairs: int
    familywise_alpha: float
    maximum_interval_width: float
    minimum_target_effect: float
    maximum_non_target_harm: float

    def assert_valid(self) -> None:
        if self.schema != INTERVENTION_SPEC_SCHEMA:
            raise ValueError("unsupported registered curriculum-trial schema")
        for name, value in (
            ("intervention preregistration", self.preregistration_sha256),
            ("randomization receipt", self.randomization_receipt_sha256),
            ("control curriculum", self.control_curriculum_sha256),
            ("treatment curriculum", self.treatment_curriculum_sha256),
        ):
            _assert_sha256(name, value)
        if self.training_family not in TRAINING_COGNITION_FAMILIES:
            raise ValueError("registered intervention names an unknown training family")
        if self.target_axis not in PHASE1_AXES:
            raise ValueError("registered intervention names an unknown Phase-One skill axis")
        if self.target_metric not in {"exact_and_eos", "valid_eos"}:
            raise ValueError("registered intervention names an unsupported target metric")
        coverage = build_training_family_coverage_report()
        coverage_row = next(
            (row for row in coverage["rows"] if row["training_family"] == self.training_family),
            None,
        )
        if (
            coverage_row is None
            or coverage_row["coverage_status"] not in {"DIRECT_POOLED_AXIS", "DIRECT_STRESS_GRID"}
            or self.target_axis not in coverage_row["skill_axes"]
        ):
            raise ValueError("registered target axis lacks direct frozen training-family coverage")
        if self.control_curriculum_sha256 == self.treatment_curriculum_sha256:
            raise ValueError("matched intervention arms must use different curriculum identities")
        if type(self.minimum_matched_pairs) is not int or self.minimum_matched_pairs < 2:
            raise ValueError("matched intervention requires at least two independent seed pairs")
        numeric_values = (
            self.familywise_alpha,
            self.maximum_interval_width,
            self.minimum_target_effect,
            self.maximum_non_target_harm,
        )
        if any(
            type(value) not in (int, float) or not math.isfinite(value)
            for value in numeric_values
        ):
            raise ValueError("registered intervention thresholds must be finite numbers")
        if not 0.0 < self.familywise_alpha < 1.0:
            raise ValueError("intervention familywise alpha must lie strictly between zero and one")
        if not 0.0 < self.maximum_interval_width <= 2.0:
            raise ValueError("paired effect interval width must lie in (0, 2]")
        if not 0.0 <= self.minimum_target_effect <= 1.0:
            raise ValueError("minimum target effect must lie in [0, 1]")
        if not 0.0 <= self.maximum_non_target_harm <= 1.0:
            raise ValueError("maximum non-target harm must lie in [0, 1]")

    def sha256(self) -> str:
        self.assert_valid()
        return _sha256(asdict(self))


_MATCHED_RECIPE_IDENTITY_FIELDS = (
    "model_spec_sha256",
    "generator_sha256",
    "evaluator_sha256",
    "tokenizer_sha256",
    "training_spec_sha256",
    "data_manifest_sha256",
    "pack_manifest_sha256",
    "optimizer_spec_sha256",
    "schedule_spec_sha256",
    "source_tree_sha256",
)


def _bounded_rate(record: Mapping[str, Any], *, label: str) -> tuple[float, int]:
    successes, cases = record.get("successes"), record.get("cases")
    if (
        type(cases) is not int
        or cases <= 0
        or type(successes) is not int
        or not 0 <= successes <= cases
    ):
        raise ValueError(f"{label} has invalid success/case counts")
    return successes / cases, cases


def analyze_registered_curriculum_trial(
    *,
    spec: RegisteredCurriculumTrial,
    matched_pairs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Estimate preregistered paired development effects and off-target harm.

    Every control/treatment pair must use the same training seed and exact
    evaluation surface, and match on all core recipe identities exposed by the
    summary. Different pairs must use distinct training and evaluation seeds.
    Hoeffding bounds operate on paired rate differences in [-1, 1], with
    Bonferroni correction across every frozen endpoint. Opaque subject-manifest
    fields, randomization, and custody still require external audit. A positive
    development signal only earns a sealed-confirmation follow-up; this report
    can never change a curriculum or authorize training.
    """

    spec.assert_valid()
    if not matched_pairs:
        raise ValueError("registered intervention analysis requires at least one matched pair")

    seen_pair_ids: set[str] = set()
    seen_training_seeds: set[int] = set()
    seen_evaluation_seeds: set[int] = set()
    seen_suites: set[str] = set()
    seen_summary_hashes: set[str] = set()
    pair_differences: dict[str, list[float]] = {}
    source_summary_hashes: list[str] = []
    target_endpoint = f"by_skill_axis.{spec.target_axis}.{spec.target_metric}"

    def endpoints(summary: Mapping[str, Any]) -> dict[str, tuple[float, int]]:
        metrics = summary["metrics"]
        values: dict[str, tuple[float, int]] = {}
        for axis in PHASE1_AXES:
            for metric_name in ("exact_and_eos", "valid_eos"):
                path = f"by_skill_axis.{axis}.{metric_name}"
                values[path] = _bounded_rate(
                    metrics["by_skill_axis"][axis][metric_name], label=path
                )
        for pair_kind in ("sensitivity", "invariance"):
            path = f"causal_pairs.{pair_kind}.pair_success"
            values[path] = _bounded_rate(metrics["causal_pairs"][pair_kind], label=path)
            eos_path = f"causal_pairs.{pair_kind}.valid_eos_pair_rate"
            values[eos_path] = _bounded_rate(
                metrics["causal_pairs"][pair_kind]["valid_eos_pair_rate"], label=eos_path
            )
        return values

    for pair in matched_pairs:
        if not isinstance(pair, Mapping) or set(pair) != {"pair_id", "control", "treatment"}:
            raise ValueError("matched-pair record has an unsupported shape")
        pair_id = pair["pair_id"]
        if not isinstance(pair_id, str) or not pair_id.strip() or pair_id in seen_pair_ids:
            raise ValueError("matched-pair identifiers must be nonempty and unique")
        seen_pair_ids.add(pair_id)
        control, treatment = pair["control"], pair["treatment"]
        if not isinstance(control, Mapping) or not isinstance(treatment, Mapping):
            raise ValueError("matched pair must carry both control and treatment summaries")
        verify_phase1_summary(control)
        verify_phase1_summary(treatment)

        control_identity = control["identities"]
        treatment_identity = treatment["identities"]
        if control["training_seed"] != treatment["training_seed"]:
            raise ValueError("matched intervention arms must share their registered training seed")
        if control["evaluation_seed"] != treatment["evaluation_seed"]:
            raise ValueError("matched intervention arms must share their evaluation seed")
        if control_identity["suite_sha256"] != treatment_identity["suite_sha256"]:
            raise ValueError("matched intervention arms must use the exact same evaluation suite")
        if any(
            control_identity[field] != treatment_identity[field]
            for field in _MATCHED_RECIPE_IDENTITY_FIELDS
        ):
            raise ValueError("matched intervention arms differ in an exposed non-curriculum training component")
        if control_identity["curriculum_spec_sha256"] != spec.control_curriculum_sha256:
            raise ValueError("control summary is not bound to the registered curriculum")
        if treatment_identity["curriculum_spec_sha256"] != spec.treatment_curriculum_sha256:
            raise ValueError("treatment summary is not bound to the registered curriculum")
        if control_identity["training_recipe_sha256"] == treatment_identity["training_recipe_sha256"]:
            raise ValueError("matched intervention arms do not identify distinct training recipes")
        if control_identity["checkpoint_sha256"] == treatment_identity["checkpoint_sha256"]:
            raise ValueError("matched intervention arms must identify distinct checkpoints")
        if control_identity["subject_manifest_sha256"] == treatment_identity["subject_manifest_sha256"]:
            raise ValueError("matched intervention arms must identify distinct subjects")

        training_seed = int(control["training_seed"])
        evaluation_seed = int(control["evaluation_seed"])
        suite_hash = str(control_identity["suite_sha256"])
        if (
            training_seed in seen_training_seeds
            or evaluation_seed in seen_evaluation_seeds
            or suite_hash in seen_suites
        ):
            raise ValueError("independent matched pairs must use distinct seed and suite identities")
        seen_training_seeds.add(training_seed)
        seen_evaluation_seeds.add(evaluation_seed)
        seen_suites.add(suite_hash)

        for summary in (control, treatment):
            summary_hash = str(summary["sha256"])
            if summary_hash in seen_summary_hashes:
                raise ValueError("a Phase-One summary cannot be reused across matched pairs")
            seen_summary_hashes.add(summary_hash)
            source_summary_hashes.append(summary_hash)

        control_endpoints = endpoints(control)
        treatment_endpoints = endpoints(treatment)
        if set(control_endpoints) != set(treatment_endpoints):
            raise ValueError("matched arms do not report the same registered outcomes")
        for endpoint in control_endpoints:
            if control_endpoints[endpoint][1] != treatment_endpoints[endpoint][1]:
                raise ValueError("matched arms report different case counts on a shared evaluation surface")
            pair_differences.setdefault(endpoint, []).append(
                treatment_endpoints[endpoint][0] - control_endpoints[endpoint][0]
            )

    pair_count = len(matched_pairs)
    endpoint_count = len(pair_differences)
    radius = math.sqrt(
        2.0 * math.log((2.0 * endpoint_count) / spec.familywise_alpha) / pair_count
    )
    effects: dict[str, dict[str, Any]] = {}
    for endpoint, differences in sorted(pair_differences.items()):
        mean_difference = sum(differences) / pair_count
        effects[endpoint] = {
            "mean_paired_difference": mean_difference,
            "simultaneous_interval": [
                max(-1.0, mean_difference - radius),
                min(1.0, mean_difference + radius),
            ],
        }

    intervals = [row["simultaneous_interval"] for row in effects.values()]
    too_wide = any(upper - lower > spec.maximum_interval_width for lower, upper in intervals)
    target_interval = effects[target_endpoint]["simultaneous_interval"]
    harm_endpoints = {
        endpoint: interval
        for endpoint, result in effects.items()
        if endpoint != target_endpoint
        for interval in (result["simultaneous_interval"],)
    }
    harm_signal = any(
        upper < -spec.maximum_non_target_harm
        for lower, upper in harm_endpoints.values()
    )
    unresolved_harm = any(
        lower < -spec.maximum_non_target_harm
        for lower, upper in harm_endpoints.values()
    )

    if pair_count < spec.minimum_matched_pairs:
        decision = "INCONCLUSIVE_INSUFFICIENT_MATCHED_PAIRS"
    elif too_wide:
        decision = "INCONCLUSIVE_WIDE_INTERVALS"
    elif harm_signal:
        decision = "STOP_PRESPECIFIED_HARM_SIGNAL"
    elif target_interval[0] >= spec.minimum_target_effect and not unresolved_harm:
        decision = "DEVELOPMENT_EFFECT_SIGNAL"
    elif target_interval[1] < spec.minimum_target_effect:
        decision = "NO_MINIMUM_TARGET_EFFECT_SIGNAL"
    else:
        decision = "INCONCLUSIVE_TARGET_OR_HARM"

    report: dict[str, Any] = {
        "schema": INTERVENTION_RESULT_SCHEMA,
        "decision": decision,
        "decision_scope": "MATCHED_DEVELOPMENT_COMPARISON_ONLY",
        "spec_sha256": spec.sha256(),
        "preregistration_sha256": spec.preregistration_sha256,
        "randomization_receipt_sha256": spec.randomization_receipt_sha256,
        "randomization_and_custody_audited": False,
        "training_family": spec.training_family,
        "target_axis": spec.target_axis,
        "target_metric": spec.target_metric,
        "target_effect": {
            "mean_paired_difference": effects[target_endpoint]["mean_paired_difference"],
            "simultaneous_interval": target_interval,
            "minimum_registered_effect": spec.minimum_target_effect,
        },
        "matched_pair_count": pair_count,
        "independent_pair_count_assuming_distinct_seed_provenance": pair_count,
        "control_curriculum_sha256": spec.control_curriculum_sha256,
        "treatment_curriculum_sha256": spec.treatment_curriculum_sha256,
        "source_summary_sha256": sorted(source_summary_hashes),
        "analysis": {
            "method": "SIMULTANEOUS_PAIRED_HOEFFDING_BONFERRONI",
            "independent_unit": "matched training seed with a shared evaluation surface",
            "subject_aggregation": "treatment rate minus control rate within each matched seed",
            "familywise_alpha": spec.familywise_alpha,
            "familywise_confidence_level": 1.0 - spec.familywise_alpha,
            "simultaneous_endpoint_count": endpoint_count,
            "unclipped_half_width": radius,
            "maximum_interval_width": spec.maximum_interval_width,
            "maximum_non_target_harm": spec.maximum_non_target_harm,
            "effects": effects,
        },
        "prespecified_harm_signal_endpoints": sorted(
            endpoint
            for endpoint, (lower, upper) in harm_endpoints.items()
            if upper < -spec.maximum_non_target_harm
        ),
        "trainer_mutation_authorized": False,
        "training_authorized": False,
        "promotion_authorized": False,
        "next_action": (
            "run the exact frozen intervention on fresh sealed matched seeds, then independently audit the randomization receipt and source custody"
            if decision == "DEVELOPMENT_EFFECT_SIGNAL"
            else "retain the result as development evidence; do not change curriculum weights or promote a model"
        ),
        "claim_ceiling": "conditional matched development effect estimate only; randomization/custody audit and sealed replication are outstanding",
    }
    report["sha256"] = _sha256(report)
    return report


def validate_registered_curriculum_result(
    report: Mapping[str, Any],
    *,
    spec: RegisteredCurriculumTrial,
    matched_pairs: Sequence[Mapping[str, Any]],
) -> None:
    """Recompute and compare a trial report against its exact source summaries."""

    expected = analyze_registered_curriculum_trial(spec=spec, matched_pairs=matched_pairs)
    if dict(report) != expected:
        raise ValueError("registered curriculum result does not match its frozen inputs")


def _verify_baseline_gate(
    gate: Mapping[str, Any],
    *,
    summaries: Sequence[Mapping[str, Any]],
    policy: BoundedRSIPolicy,
    gate_spec: PhaseOneBaselineGateSpec,
) -> None:
    if gate.get("schema") != BASELINE_GATE_SCHEMA:
        raise ValueError("RSI requires a Signac Phase-One baseline-gate record")
    claimed = gate.get("sha256")
    body = {key: value for key, value in gate.items() if key != "sha256"}
    if not isinstance(claimed, str) or _sha256(body) != claimed:
        raise ValueError("Phase-One baseline-gate content hash mismatch")
    decision = gate.get("decision")
    if gate.get("training_authorized") is not False:
        raise ValueError("Phase-One baseline gate cannot authorize training")
    if gate.get("mechanism_experiment_eligible") is not False:
        raise ValueError("diagnostic baseline classifications cannot authorize interventions")
    if gate.get("decision_scope") != "WITHIN_SUITE_DIAGNOSTIC_ONLY":
        raise ValueError("Phase-One baseline gate has an unsupported decision scope")
    _assert_sha256("baseline preregistration", str(gate.get("preregistration_sha256", "")))
    _assert_sha256("baseline gate specification", str(gate.get("gate_spec_sha256", "")))
    if gate.get("preregistration_sha256") != policy.preregistration_sha256:
        raise ValueError("RSI policy and baseline gate bind different preregistrations")
    gate_spec.assert_valid()
    if gate_spec.preregistration_sha256 != policy.preregistration_sha256:
        raise ValueError("RSI policy and baseline spec bind different preregistrations")
    recomputed_gate = assess_phase1_baseline(summaries, gate_spec)
    if dict(gate) != recomputed_gate:
        raise ValueError("Phase-One baseline gate does not match recomputed summary diagnostics")
    source_hashes = gate.get("source_summary_sha256")
    expected_hashes = sorted(str(summary.get("sha256", "")) for summary in summaries)
    if source_hashes != expected_hashes:
        raise ValueError("Phase-One baseline gate does not bind these exact summaries")
    unique_seed_count = len({summary.get("training_seed") for summary in summaries})
    if gate.get("independent_training_seed_count") != unique_seed_count:
        raise ValueError("Phase-One baseline-gate training-seed count disagrees with its summaries")
    if decision == "DIAGNOSTIC_NONFLOOR_NONCEILING" and unique_seed_count < policy.minimum_independent_seeds:
        raise ValueError("diagnostic baseline does not meet the RSI training-seed minimum")


def validate_rsi_proposal(proposal: Mapping[str, Any]) -> None:
    """Check that a proposal stays inside its inert training-mixture surface."""

    if proposal.get("schema") != PROPOSAL_SCHEMA:
        raise ValueError("unsupported RSI proposal schema")
    required_fields = {
        "schema", "decision", "status", "policy_sha256", "baseline_gate_sha256",
        "baseline_gate_decision", "source_summary_sha256",
        "diagnostic_weakest_axis", "selected_axis", "action", "weight_shift_ppm", "split_scope",
        "mutable_component", "immutable_components", "skill_axis_weights_before_ppm",
        "skill_axis_weights_proposed_ppm", "training_family_coverage", "abstain_reason",
        "comparison_design", "sha256",
    }
    if set(proposal) != required_fields:
        raise ValueError("RSI proposal carries unsupported fields")
    claimed = proposal.get("sha256")
    body = {key: value for key, value in proposal.items() if key != "sha256"}
    if not isinstance(claimed, str) or _sha256(body) != claimed:
        raise ValueError("RSI proposal content hash mismatch")
    if proposal.get("status") != "DIAGNOSTIC_ONLY_NOT_APPLICABLE_TO_CURRENT_TRAINER":
        raise ValueError("Signac RSI diagnostic cannot authorize or execute training")
    if proposal.get("split_scope") != "development_measurement_only":
        raise ValueError("RSI diagnostic must be restricted to development measurements")
    if tuple(proposal.get("immutable_components", ())) != IMMUTABLE_RESEARCH_COMPONENTS:
        raise ValueError("RSI proposal attempts to alter a protected research component")
    if proposal.get("mutable_component") != "none":
        raise ValueError("Signac RSI diagnostic cannot mutate training components")
    if proposal.get("action") != "none":
        raise ValueError("Signac RSI has no registered training-family action")
    _assert_sha256("RSI policy", str(proposal.get("policy_sha256", "")))
    _assert_sha256("Phase-One baseline gate", str(proposal.get("baseline_gate_sha256", "")))
    baseline_decision = proposal.get("baseline_gate_decision")
    allowed_baseline_decisions = {
        "INCONCLUSIVE_INSUFFICIENT_TRAINING_SEEDS",
        "INCONCLUSIVE_INSUFFICIENT_CASES",
        "NO_GO_PRIMARY_FLOOR_DIAGNOSTIC",
        "NO_GO_PRIMARY_CEILING_DIAGNOSTIC",
        "DIAGNOSTIC_NONFLOOR_NONCEILING",
        "BORDERLINE_DIAGNOSTIC",
    }
    if baseline_decision not in allowed_baseline_decisions:
        raise ValueError("RSI proposal carries an unknown baseline-gate decision")
    if proposal.get("decision") != "ABSTAIN":
        raise ValueError("Signac RSI must abstain until inference and training-axis mapping exist")
    if proposal.get("training_family_coverage") != build_training_family_coverage_report():
        raise ValueError("RSI coverage report differs from the frozen measurement taxonomy")
    source_hashes = proposal.get("source_summary_sha256")
    if not isinstance(source_hashes, list):
        raise ValueError("RSI proposal must bind its development summaries")
    for value in source_hashes:
        _assert_sha256("Phase-One summary", str(value))
    before = proposal.get("skill_axis_weights_before_ppm")
    after = proposal.get("skill_axis_weights_proposed_ppm")
    if not isinstance(before, Mapping) or not isinstance(after, Mapping):
        raise ValueError("RSI proposal must bind before/after family weights")
    if set(before) != set(PHASE1_AXES) or set(after) != set(PHASE1_AXES):
        raise ValueError("RSI proposal family weights must cover the frozen Phase-One axes")
    if any(type(value) is not int or value < 0 for value in (*before.values(), *after.values())):
        raise ValueError("RSI mixture weights must be nonnegative integer ppm values")
    if sum(before.values()) != MIXTURE_TOTAL_PPM or sum(after.values()) != MIXTURE_TOTAL_PPM:
        raise ValueError("RSI mixture weights must each sum to one million ppm")
    deltas = {axis: int(after[axis]) - int(before[axis]) for axis in PHASE1_AXES}
    if sum(deltas.values()) != 0:
        raise ValueError("RSI diagnostic must preserve the current training mixture")
    if type(proposal.get("weight_shift_ppm")) is not int:
        raise ValueError("RSI proposal weight shift must be an integer ppm value")
    selected_axis = proposal.get("selected_axis")
    decision = proposal.get("decision")
    if decision != "ABSTAIN":
        raise ValueError("Signac RSI diagnostic must abstain")
    if selected_axis is not None or any(deltas.values()) or proposal["weight_shift_ppm"] != 0:
        raise ValueError("RSI abstention must not change the training mixture")
    diagnostic_axis = proposal.get("diagnostic_weakest_axis")
    if diagnostic_axis is not None and diagnostic_axis not in PHASE1_AXES:
        raise ValueError("RSI diagnostic names an unknown skill axis")
    expected_reason = (
        "prospective_intervention_effect_map_and_randomized_comparison_required"
        if baseline_decision == "DIAGNOSTIC_NONFLOOR_NONCEILING"
        else "baseline_gate_not_informative"
    )
    if proposal.get("abstain_reason") != expected_reason:
        raise ValueError("RSI abstention reason conflicts with baseline readiness")
    design = proposal.get("comparison_design")
    required_blockers = {
        "externally audited paired inference over randomized intervention/control subjects",
        "prospectively registered training-family-to-outcome effect map",
        "matched prospective intervention preregistration",
        "fresh sealed-set confirmation and independent replication",
    }
    if (
        not isinstance(design, Mapping)
        or design.get("proposal_applicable") is not False
        or design.get("automatic_training_or_promotion") is not False
        or not required_blockers.issubset(set(design.get("blocking_requirements", ())))
    ):
        raise ValueError("RSI diagnostic cannot advertise an applicable training proposal")


def propose_bounded_curriculum_change(
    *,
    summaries: Sequence[Mapping[str, Any]],
    baseline_gate: Mapping[str, Any],
    baseline_gate_spec: PhaseOneBaselineGateSpec,
    current_skill_axis_weights_ppm: Mapping[str, int],
    policy: BoundedRSIPolicy,
) -> dict[str, Any]:
    """Report a weakest-axis diagnostic and abstain from curriculum changes.

    The embedded family-coverage map describes which outcomes are measured; it
    does not estimate training effects. Baseline run-level bounds do not
    estimate intervention effects. Without a matched prospective comparison
    and a registered family-to-outcome effect map, this emits no weight proposal.
    """

    policy.assert_valid()
    weights = {str(axis): value for axis, value in current_skill_axis_weights_ppm.items()}
    if set(weights) != set(PHASE1_AXES):
        raise ValueError("current mixture must cover all Phase-One skill axes")
    if any(type(value) is not int or value < 0 for value in weights.values()):
        raise ValueError("current mixture weights must be nonnegative integer ppm values")
    if sum(weights.values()) != MIXTURE_TOTAL_PPM:
        raise ValueError("current family weights must sum to one million ppm")

    verified = list(summaries)
    for summary in verified:
        verify_phase1_summary(summary)
    _verify_baseline_gate(
        baseline_gate, summaries=verified, policy=policy, gate_spec=baseline_gate_spec
    )
    unique_seeds = {summary.get("training_seed") for summary in verified}
    identities = [summary["identities"] for summary in verified]
    if baseline_gate.get("decision") != "DIAGNOSTIC_NONFLOOR_NONCEILING":
        verified = []
        abstain_reason = "baseline_gate_not_informative"
    elif len(verified) < policy.minimum_independent_seeds or len(unique_seeds) != len(verified):
        verified = []
        abstain_reason = "insufficient_independent_development_seeds"
    elif len({item["suite_sha256"] for item in identities}) != len(identities):
        verified = []
        abstain_reason = "development_surfaces_are_not_independent"
    elif len({item["checkpoint_sha256"] for item in identities}) != len(identities):
        verified = []
        abstain_reason = "checkpoint_seeds_are_not_independent"
    elif (
        len({item["model_spec_sha256"] for item in identities}) != 1
        or len({item["evaluator_sha256"] for item in identities}) != 1
        or len({item["generator_sha256"] for item in identities}) != 1
    ):
        raise ValueError("RSI seed summaries disagree on model, generator, or evaluator identity")
    else:
        abstain_reason = "no_skill_below_preregistered_action_threshold"

    diagnostic_axis: str | None = None
    if verified:
        run_level_bounds = baseline_gate["run_level_uncertainty"]["by_skill_axis"]
        simultaneous_lcb = {
            axis: float(run_level_bounds[axis]["exact_and_eos"]["simultaneous_interval"][0])
            for axis in PHASE1_AXES
        }
        below = [axis for axis, value in simultaneous_lcb.items() if value < policy.action_threshold_lcb]
        if below:
            diagnostic_axis = min(below, key=lambda axis: (simultaneous_lcb[axis], axis))
        abstain_reason = "prospective_intervention_effect_map_and_randomized_comparison_required"

    before = dict(sorted(weights.items()))
    after = before
    decision = "ABSTAIN"
    delta = 0

    proposal: dict[str, Any] = {
        "schema": PROPOSAL_SCHEMA,
        "decision": decision,
        "status": "DIAGNOSTIC_ONLY_NOT_APPLICABLE_TO_CURRENT_TRAINER",
        "policy_sha256": policy.sha256(),
        "baseline_gate_sha256": str(baseline_gate["sha256"]),
        "baseline_gate_decision": str(baseline_gate["decision"]),
        "source_summary_sha256": sorted(str(summary.get("sha256", "")) for summary in summaries),
        "diagnostic_weakest_axis": diagnostic_axis,
        "selected_axis": None,
        "action": "none",
        "weight_shift_ppm": delta,
        "split_scope": "development_measurement_only",
        "mutable_component": "none",
        "immutable_components": list(IMMUTABLE_RESEARCH_COMPONENTS),
        "skill_axis_weights_before_ppm": before,
        "skill_axis_weights_proposed_ppm": after,
        "training_family_coverage": build_training_family_coverage_report(),
        "abstain_reason": (
            abstain_reason
            if abstain_reason == "baseline_gate_not_informative"
            else "prospective_intervention_effect_map_and_randomized_comparison_required"
        ),
        "comparison_design": {
            "proposal_applicable": False,
            "blocking_requirements": [
                "externally audited paired inference over randomized intervention/control subjects",
                "prospectively registered training-family-to-outcome effect map",
                "matched prospective intervention preregistration",
                "fresh sealed-set confirmation and independent replication",
            ],
            "automatic_training_or_promotion": False,
        },
    }
    proposal["sha256"] = _sha256(proposal)
    validate_rsi_proposal(proposal)
    return proposal


__all__ = [
    "BoundedRSIPolicy",
    "INTERVENTION_RESULT_SCHEMA",
    "INTERVENTION_SPEC_SCHEMA",
    "IMMUTABLE_RESEARCH_COMPONENTS",
    "MIXTURE_TOTAL_PPM",
    "POLICY_SCHEMA",
    "PROPOSAL_SCHEMA",
    "RegisteredCurriculumTrial",
    "TRAINING_COVERAGE_SCHEMA",
    "analyze_registered_curriculum_trial",
    "build_training_family_coverage_report",
    "propose_bounded_curriculum_change",
    "validate_registered_curriculum_result",
    "validate_rsi_proposal",
]
