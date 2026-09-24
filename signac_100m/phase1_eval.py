"""Signac Phase-One candidate-free cognition evaluation bridge.

This module converts the repository's causal E0 suite into the hash-bound V5
evaluation protocol. It exposes model prompts only, records raw free
generation and EOS stop status, scores causal pairs, and emits a diagnostic
summary. It never authorizes training or consumes sealed/fresh fixtures.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from e0_cognition.contracts import EvaluationSuite, Split
from e0_cognition.evaluation_generators import (
    INTERFERENCE_RETRIEVAL_GRID,
    build_evaluation_suite,
)
from e0_cognition.statistics import wilson_interval
from signac_100m.spec import MODEL_SPEC
from v5_evaluation.checkpoint_adapter import (
    SCORING_CONTRACT_SHA256,
    assert_canonical_checkpoint_adapter,
)
from v5_evaluation.fixture import TaskFixtureBatch
from v5_evaluation.protocol import (
    EvaluationProtocol,
    EvaluationReceipt,
    TaskLevelEvidence,
    evaluator_source_sha256,
    run_evaluation,
)
from v5_registry.subject import CoreSubjectManifest


SUMMARY_SCHEMA = "anra-signac-phase1-cognition-summary/v5"
PROTOCOL_ID = "signac-phase1-e0-candidate-free/v5"
PHASE1_AXES: dict[str, tuple[str, ...]] = {
    "identity": ("exact_contextual_copy", "nonce_identifier_retrieval"),
    "binding": (
        "entity_value_binding",
        "matched_direct_retrieval",
        "natural_binding_analogue",
    ),
    "interference_retrieval": ("interference_retrieval",),
    "state_order": ("state_overwrite", "natural_state_analogue"),
    "composition": (
        "relation_2_hop",
        "relation_3_hop",
        "natural_composition_analogue",
        "rule_induction",
    ),
    "faithful_realization": ("faithful_realization",),
    "missing_information": ("missing_information",),
}
PHASE1_METRICS = (
    "EXACT_ACCURACY",
    "BALANCED_ACCURACY",
    "VALID_EOS_RATE",
    "EXACT_AND_EOS_RATE",
    "PAIRED_COUNTERFACTUAL_SENSITIVITY",
    "PAIRED_INVARIANCE_STABILITY",
)
PHASE1_PAIR_FAMILY_KINDS: dict[str, tuple[str, ...]] = {
    "counterfactual_premise": ("relevant_fact_swap",),
    "entity_value_binding": (
        "irrelevant_fact_swap", "order_permutation", "query_swap", "relevant_fact_swap"
    ),
    "rule_induction": ("query_swap",),
    "state_overwrite": ("state_swap",),
    "interference_retrieval": ("relevant_fact_swap",),
    "faithful_realization": ("relevant_fact_swap",),
}
PHASE1_PAIR_KIND_CATEGORIES: dict[str, tuple[str, ...]] = {
    "sensitivity": ("query_swap", "relevant_fact_swap", "state_swap"),
    "invariance": ("irrelevant_fact_swap", "order_permutation"),
}
BASELINE_GATE_SCHEMA = "anra-signac-phase1-baseline-gate/v5"
RUN_LEVEL_UNCERTAINTY_SCHEMA = "anra-signac-run-level-uncertainty/v1"


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _verify_rate_record(value: Any, label: str) -> tuple[int, int]:
    if not isinstance(value, Mapping) or set(value) != {
        "successes", "cases", "rate", "wilson95"
    }:
        raise ValueError(f"{label} has an invalid metric shape")
    successes, cases, rate, interval = (
        value["successes"], value["cases"], value["rate"], value["wilson95"]
    )
    if (
        type(cases) is not int
        or cases <= 0
        or type(successes) is not int
        or not 0 <= successes <= cases
        or type(rate) not in (int, float)
        or not isinstance(interval, list)
        or len(interval) != 2
        or not 0.0 <= float(interval[0]) <= float(interval[1]) <= 1.0
    ):
        raise ValueError(f"{label} has malformed metric counts or interval")
    if not math.isclose(float(rate), successes / cases, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{label} rate disagrees with its counts")
    expected_interval = wilson_interval(successes, cases)
    if any(
        not math.isclose(float(interval[index]), expected_interval[index], rel_tol=0.0, abs_tol=1e-12)
        for index in range(2)
    ):
        raise ValueError(f"{label} interval does not match its counts")
    return successes, cases


def _training_recipe_sha256(subject: CoreSubjectManifest) -> str:
    return _sha256({
        "model_spec_sha256": subject.model_spec_sha256,
        "tokenizer_artifact_sha256": subject.tokenizer_artifact_sha256,
        "tokenizer_identity_sha256": subject.tokenizer_identity_sha256,
        "training_spec_sha256": subject.training_spec_sha256,
        "data_manifest_sha256": subject.data_manifest_sha256,
        "pack_manifest_sha256": subject.pack_manifest_sha256,
        "optimizer_spec_sha256": subject.optimizer_spec_sha256,
        "schedule_spec_sha256": subject.schedule_spec_sha256,
        "curriculum_spec_sha256": subject.curriculum_spec_sha256,
        "source_commit": subject.source_commit,
        "source_tree_sha256": subject.source_tree_sha256,
        "parent_checkpoint_sha256": subject.parent_checkpoint_sha256,
        "training_stage": subject.training_stage,
    })


def _assert_checkpoint_source_binding(
    *, subject: CoreSubjectManifest, adapter_identity: Any,
) -> None:
    """Require the subject and restored checkpoint to name the same code bundle."""

    expected = subject.source_tree_sha256
    actual = getattr(adapter_identity, "source_tree_sha256", None)
    if expected is None:
        raise ValueError("Signac Phase-One subject must bind the source tree used for training")
    if actual != expected:
        raise ValueError("Phase-One checkpoint source tree identity disagrees with the subject")


def verify_phase1_summary(summary: Mapping[str, Any]) -> None:
    """Verify a diagnostic report before it enters a multi-seed decision."""

    if summary.get("schema") != SUMMARY_SCHEMA or summary.get("split") != "development":
        raise ValueError("baseline analysis accepts only Signac development summaries")
    if summary.get("training_authorized") is not False:
        raise ValueError("Phase-One score summaries cannot authorize training")
    if (
        type(summary.get("training_seed")) is not int
        or summary["training_seed"] < 0
        or type(summary.get("evaluation_seed")) is not int
        or summary["evaluation_seed"] < 0
        or summary.get("seed") != summary.get("evaluation_seed")
    ):
        raise ValueError("Phase-One summary must distinguish training and evaluation seeds")
    claimed = summary.get("sha256")
    body = {key: value for key, value in summary.items() if key != "sha256"}
    if not isinstance(claimed, str) or _sha256(body) != claimed:
        raise ValueError("Phase-One summary content hash mismatch")
    identities = summary.get("identities")
    metrics = summary.get("metrics")
    if not isinstance(identities, Mapping) or not isinstance(metrics, Mapping):
        raise ValueError("Phase-One summary is missing identity or metric mappings")
    for name in (
        "model_spec_sha256", "suite_sha256", "evaluation_receipt_sha256",
        "evaluator_sha256", "checkpoint_sha256", "generator_sha256",
        "training_recipe_sha256", "tokenizer_sha256", "training_spec_sha256",
        "data_manifest_sha256", "pack_manifest_sha256", "optimizer_spec_sha256",
        "schedule_spec_sha256", "curriculum_spec_sha256",
        "source_tree_sha256",
        "receipt_subject_manifest_sha256", "subject_manifest_sha256", "adapter_sha256",
    ):
        value = identities.get(name)
        if not isinstance(value, str) or len(value) != 64 or any(
            character not in "0123456789abcdef" for character in value
        ):
            raise ValueError(f"Phase-One summary has invalid {name}")
    if identities["receipt_subject_manifest_sha256"] != identities["subject_manifest_sha256"]:
        raise ValueError("Phase-One receipt is bound to a different subject manifest")
    axes = metrics.get("by_skill_axis")
    if not isinstance(axes, Mapping) or set(axes) != set(PHASE1_AXES):
        raise ValueError("Phase-One summary does not cover every registered skill axis")
    for axis in PHASE1_AXES:
        for metric in ("exact_and_eos", "valid_eos"):
            summary_metric = axes[axis].get(metric)
            interval = summary_metric.get("wilson95") if isinstance(summary_metric, Mapping) else None
            if (
                not isinstance(interval, list)
                or len(interval) != 2
                or not 0.0 <= float(interval[0]) <= float(interval[1]) <= 1.0
            ):
                raise ValueError(f"Phase-One axis {axis} has invalid {metric} evidence")
            cases = summary_metric.get("cases") if isinstance(summary_metric, Mapping) else None
            successes = summary_metric.get("successes") if isinstance(summary_metric, Mapping) else None
            if (
                type(cases) is not int
                or cases <= 0
                or type(successes) is not int
                or not 0 <= successes <= cases
            ):
                raise ValueError(f"Phase-One axis {axis} has invalid {metric} counts")
            expected_interval = wilson_interval(successes, cases)
            if any(
                not math.isclose(float(interval[index]), expected_interval[index], rel_tol=0.0, abs_tol=1e-12)
                for index in range(2)
            ):
                raise ValueError(f"Phase-One axis {axis} {metric} interval does not match its counts")
    by_family = metrics.get("by_family")
    expected_families = {
        family for families in PHASE1_AXES.values() for family in families
    }
    if not isinstance(by_family, Mapping) or not expected_families.issubset(by_family):
        raise ValueError("Phase-One summary lacks the complete registered family breakdown")
    family_counts: dict[str, dict[str, tuple[int, int]]] = {}
    for family, family_metrics in by_family.items():
        if not isinstance(family, str) or not family:
            raise ValueError("Phase-One family names must be nonempty strings")
        if not isinstance(family_metrics, Mapping) or set(family_metrics) != {
            "exact", "valid_eos", "exact_and_eos"
        }:
            raise ValueError(f"Phase-One family {family} has an invalid metric breakdown")
        counts = {
            metric_name: _verify_rate_record(
                family_metrics[metric_name], f"Phase-One family {family} {metric_name}"
            )
            for metric_name in ("exact", "valid_eos", "exact_and_eos")
        }
        if (
            counts["exact"][1] != counts["valid_eos"][1]
            or counts["exact"][1] != counts["exact_and_eos"][1]
            or counts["exact_and_eos"][0] > counts["exact"][0]
            or counts["exact_and_eos"][0] > counts["valid_eos"][0]
        ):
            raise ValueError(f"Phase-One family {family} metric counts disagree")
        family_counts[family] = counts
    for axis, registered_families in PHASE1_AXES.items():
        axis_metrics = axes[axis]
        expected_axis_families = list(registered_families)
        if not isinstance(axis_metrics, Mapping) or set(axis_metrics) != {
            "families", "exact", "valid_eos", "exact_and_eos"
        } or axis_metrics.get("families") != expected_axis_families:
            raise ValueError(f"Phase-One axis {axis} has an invalid family breakdown")
        for metric_name in ("exact", "valid_eos", "exact_and_eos"):
            expected_counts = (
                sum(family_counts[family][metric_name][0] for family in registered_families),
                sum(family_counts[family][metric_name][1] for family in registered_families),
            )
            actual_counts = _verify_rate_record(
                axis_metrics[metric_name], f"Phase-One axis {axis} {metric_name}"
            )
            if actual_counts != expected_counts:
                raise ValueError(
                    f"Phase-One axis {axis} {metric_name} disagrees with family metrics"
                )

    pairs = metrics.get("causal_pairs")
    if not isinstance(pairs, Mapping) or not {"sensitivity", "invariance"}.issubset(pairs):
        raise ValueError("Phase-One summary lacks both causal-pair classes")
    for pair_kind in ("sensitivity", "invariance"):
        pair_metrics = pairs[pair_kind]
        interval = pair_metrics.get("wilson95") if isinstance(pair_metrics, Mapping) else None
        if (
            not isinstance(interval, list)
            or len(interval) != 2
            or not 0.0 <= float(interval[0]) <= float(interval[1]) <= 1.0
        ):
            raise ValueError(f"Phase-One {pair_kind} pair evidence has an invalid interval")
        cases = pair_metrics.get("cases") if isinstance(pair_metrics, Mapping) else None
        successes = pair_metrics.get("successes") if isinstance(pair_metrics, Mapping) else None
        if (
            type(cases) is not int
            or cases <= 0
            or type(successes) is not int
            or not 0 <= successes <= cases
        ):
            raise ValueError(f"Phase-One {pair_kind} pair evidence has invalid counts")
        expected_interval = wilson_interval(successes, cases)
        if any(
            not math.isclose(float(interval[index]), expected_interval[index], rel_tol=0.0, abs_tol=1e-12)
            for index in range(2)
        ):
            raise ValueError(f"Phase-One {pair_kind} interval does not match its counts")
        eos_rate = pair_metrics.get("valid_eos_pair_rate")
        if not isinstance(eos_rate, Mapping):
            raise ValueError(f"Phase-One {pair_kind} lacks valid-EOS pair counts")
        eos_interval = eos_rate.get("wilson95")
        eos_cases, eos_successes = eos_rate.get("cases"), eos_rate.get("successes")
        if (
            not isinstance(eos_interval, list)
            or len(eos_interval) != 2
            or type(eos_cases) is not int
            or eos_cases != cases
            or type(eos_successes) is not int
            or not 0 <= eos_successes <= eos_cases
        ):
            raise ValueError(f"Phase-One {pair_kind} valid-EOS pair evidence is malformed")
        expected_eos_interval = wilson_interval(eos_successes, eos_cases)
        if any(
            not math.isclose(float(eos_interval[index]), expected_eos_interval[index], rel_tol=0.0, abs_tol=1e-12)
            for index in range(2)
        ):
            raise ValueError(f"Phase-One {pair_kind} valid-EOS interval does not match its counts")

    by_family_kind = metrics.get("causal_pairs_by_family_and_kind")
    if not isinstance(by_family_kind, Mapping) or set(by_family_kind) != set(
        PHASE1_PAIR_FAMILY_KINDS
    ):
        raise ValueError("Phase-One summary lacks the registered family/pair-kind breakdown")
    aggregate_counts = {
        category: {"cases": 0, "successes": 0, "valid_eos_successes": 0}
        for category in PHASE1_PAIR_KIND_CATEGORIES
    }
    for family, expected_kinds in PHASE1_PAIR_FAMILY_KINDS.items():
        family_metrics = by_family_kind[family]
        if not isinstance(family_metrics, Mapping) or set(family_metrics) != set(expected_kinds):
            raise ValueError(f"Phase-One {family} has an unexpected causal-pair breakdown")
        for pair_kind in expected_kinds:
            pair_metric = family_metrics[pair_kind]
            if not isinstance(pair_metric, Mapping) or set(pair_metric) != {
                "successes", "cases", "rate", "wilson95", "valid_eos_pair_rate"
            }:
                raise ValueError("Phase-One family/pair-kind metric has an invalid shape")
            pair_eos = pair_metric["valid_eos_pair_rate"]
            if not isinstance(pair_eos, Mapping) or set(pair_eos) != {
                "successes", "cases", "rate", "wilson95"
            }:
                raise ValueError("Phase-One family/pair-kind EOS metric has an invalid shape")
            cases = pair_metric["cases"]
            successes = pair_metric["successes"]
            eos_cases = pair_eos["cases"]
            eos_successes = pair_eos["successes"]
            if (
                type(cases) is not int
                or cases <= 0
                or type(successes) is not int
                or not 0 <= successes <= cases
                or type(eos_cases) is not int
                or eos_cases != cases
                or type(eos_successes) is not int
                or not successes <= eos_successes <= cases
            ):
                raise ValueError("Phase-One family/pair-kind counts are inconsistent")
            for label, rate_record, rate_successes in (
                ("pair", pair_metric, successes),
                ("valid-EOS pair", pair_eos, eos_successes),
            ):
                rate = rate_record.get("rate")
                interval = rate_record.get("wilson95")
                if (
                    type(rate) not in (int, float)
                    or not isinstance(interval, list)
                    or len(interval) != 2
                    or not 0.0 <= float(interval[0]) <= float(interval[1]) <= 1.0
                ):
                    raise ValueError(f"Phase-One family/pair-kind {label} rate is malformed")
                if not math.isclose(float(rate), rate_successes / cases, rel_tol=0.0, abs_tol=1e-12):
                    raise ValueError(f"Phase-One family/pair-kind {label} rate disagrees with its counts")
                expected_interval = wilson_interval(rate_successes, cases)
                if any(
                    not math.isclose(
                        float(interval[index]), expected_interval[index], rel_tol=0.0, abs_tol=1e-12
                    )
                    for index in range(2)
                ):
                    raise ValueError(
                        f"Phase-One family/pair-kind {label} interval does not match its counts"
                    )
            category = next(
                name
                for name, kinds in PHASE1_PAIR_KIND_CATEGORIES.items()
                if pair_kind in kinds
            )
            aggregate_counts[category]["cases"] += cases
            aggregate_counts[category]["successes"] += successes
            aggregate_counts[category]["valid_eos_successes"] += eos_successes
    for category, totals in aggregate_counts.items():
        aggregate = pairs[category]
        aggregate_eos = aggregate["valid_eos_pair_rate"]
        if (
            totals["cases"] != aggregate["cases"]
            or totals["successes"] != aggregate["successes"]
            or totals["cases"] != aggregate_eos["cases"]
            or totals["valid_eos_successes"] != aggregate_eos["successes"]
        ):
            raise ValueError("Phase-One family/pair-kind totals disagree with aggregate pair metrics")

    interference_grid = metrics.get("interference_retrieval_grid")
    expected_dose_keys = {str(dose) for dose in INTERFERENCE_RETRIEVAL_GRID}
    if not isinstance(interference_grid, Mapping) or set(interference_grid) != expected_dose_keys:
        raise ValueError("Phase-One summary lacks the frozen interference-retrieval dose grid")
    grid_totals = {
        "exact_and_eos": [0, 0],
        "valid_eos": [0, 0],
    }
    for distractor_count, quartiles in INTERFERENCE_RETRIEVAL_GRID.items():
        dose_row = interference_grid[str(distractor_count)]
        expected_quartile_keys = {str(quartile) for quartile in quartiles}
        if not isinstance(dose_row, Mapping) or set(dose_row) != expected_quartile_keys:
            raise ValueError("Phase-One interference-retrieval quartile grid is incomplete")
        for quartile in quartiles:
            condition = dose_row[str(quartile)]
            if not isinstance(condition, Mapping) or set(condition) != {
                "exact_and_eos", "valid_eos"
            }:
                raise ValueError("Phase-One interference-retrieval condition has an invalid shape")
            exact_successes, exact_cases = _verify_rate_record(
                condition["exact_and_eos"], "interference-retrieval exact-and-EOS metric"
            )
            eos_successes, eos_cases = _verify_rate_record(
                condition["valid_eos"], "interference-retrieval EOS metric"
            )
            if eos_cases != exact_cases or exact_successes > eos_successes:
                raise ValueError("Phase-One interference-retrieval exact/EOS counts disagree")
            grid_totals["exact_and_eos"][0] += exact_successes
            grid_totals["exact_and_eos"][1] += exact_cases
            grid_totals["valid_eos"][0] += eos_successes
            grid_totals["valid_eos"][1] += eos_cases
    by_family = metrics.get("by_family")
    if not isinstance(by_family, Mapping) or "interference_retrieval" not in by_family:
        raise ValueError("Phase-One summary lacks interference-retrieval family metrics")
    for metric_name, totals in grid_totals.items():
        family_successes, family_cases = _verify_rate_record(
            by_family["interference_retrieval"][metric_name],
            f"interference-retrieval family {metric_name}",
        )
        if totals != [family_successes, family_cases]:
            raise ValueError("Phase-One interference-retrieval grid totals disagree with family metrics")


@dataclass(frozen=True, slots=True)
class PhaseOneBaselineGateSpec:
    """Explicit sensitivity thresholds; values must come from a frozen preregistration."""

    schema: str
    preregistration_sha256: str
    minimum_independent_seeds: int
    minimum_cases_per_axis: int
    primary_axis: str
    primary_floor_upper_max: float
    primary_ceiling_lower_min: float
    run_level_familywise_alpha: float
    maximum_run_level_interval_width: float
    minimum_exact_and_eos_lcb: float
    minimum_eos_lcb: float
    minimum_sensitivity_lcb: float
    minimum_invariance_lcb: float

    def assert_valid(self) -> None:
        if self.schema != BASELINE_GATE_SCHEMA:
            raise ValueError("unsupported Phase-One baseline gate schema")
        if len(self.preregistration_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.preregistration_sha256
        ):
            raise ValueError("baseline gate must bind its preregistration SHA-256")
        if (
            type(self.minimum_independent_seeds) is not int
            or type(self.minimum_cases_per_axis) is not int
            or self.minimum_independent_seeds < 2
            or self.minimum_cases_per_axis <= 0
        ):
            raise ValueError("baseline gate requires multiple seeds and positive case counts")
        if self.primary_axis not in PHASE1_AXES:
            raise ValueError("baseline primary axis is not part of the frozen suite")
        values = (
            self.primary_floor_upper_max,
            self.primary_ceiling_lower_min,
            self.run_level_familywise_alpha,
            self.maximum_run_level_interval_width,
            self.minimum_exact_and_eos_lcb,
            self.minimum_eos_lcb,
            self.minimum_sensitivity_lcb,
            self.minimum_invariance_lcb,
        )
        if any(
            type(value) not in (int, float) or not math.isfinite(value) or not 0.0 <= value <= 1.0
            for value in values
        ):
            raise ValueError("baseline gate thresholds must lie in [0, 1]")
        if not 0.0 < self.run_level_familywise_alpha < 1.0:
            raise ValueError("run-level familywise alpha must lie strictly between zero and one")
        if self.maximum_run_level_interval_width <= 0.0:
            raise ValueError("maximum run-level interval width must be positive")
        if self.primary_floor_upper_max >= self.primary_ceiling_lower_min:
            raise ValueError("primary floor and ceiling bands overlap")

    def sha256(self) -> str:
        self.assert_valid()
        return _sha256(asdict(self))


def assess_phase1_baseline(
    summaries: Sequence[Mapping[str, Any]], spec: PhaseOneBaselineGateSpec
) -> dict[str, Any]:
    """Classify whether Signac cognition measurements resolve beyond floor/ceiling.

    The returned classification is diagnostic only. Task-level Wilson intervals
    describe each sampled surface; the separate run-level bounds aggregate one
    rate per trained subject. Neither supplies a preregistered training-effect
    map, and neither licenses a mechanism experiment or 100M training run.
    """

    spec.assert_valid()
    for summary in summaries:
        verify_phase1_summary(summary)
    unique_training_seed_count = len({summary.get("training_seed") for summary in summaries})
    unique_evaluation_seed_count = len({summary.get("evaluation_seed") for summary in summaries})
    identities = [summary["identities"] for summary in summaries]
    independent = (
        len(identities) == len(summaries)
        and unique_training_seed_count == len(summaries)
        and unique_evaluation_seed_count == len(summaries)
        and len({identity["suite_sha256"] for identity in identities}) == len(identities)
        and len({identity["checkpoint_sha256"] for identity in identities}) == len(identities)
    )
    common_identity = (
        len({identity["model_spec_sha256"] for identity in identities}) <= 1
        and len({identity["evaluator_sha256"] for identity in identities}) <= 1
        and len({identity["generator_sha256"] for identity in identities}) <= 1
        and len({identity["training_recipe_sha256"] for identity in identities}) <= 1
        and len({identity["tokenizer_sha256"] for identity in identities}) <= 1
        and len({identity["training_spec_sha256"] for identity in identities}) <= 1
        and len({identity["data_manifest_sha256"] for identity in identities}) <= 1
        and len({identity["pack_manifest_sha256"] for identity in identities}) <= 1
        and len({identity["optimizer_spec_sha256"] for identity in identities}) <= 1
        and len({identity["schedule_spec_sha256"] for identity in identities}) <= 1
        and len({identity["curriculum_spec_sha256"] for identity in identities}) <= 1
        and len({identity["source_tree_sha256"] for identity in identities}) <= 1
    )
    if not common_identity:
        raise ValueError(
            "baseline seeds disagree on source tree, model, training recipe, generator, or evaluator identity"
        )

    decision = "INCONCLUSIVE_INSUFFICIENT_TRAINING_SEEDS"
    reason = "need distinct trained subjects from independent training seeds and distinct evaluation surfaces"
    run_level_uncertainty: dict[str, Any] | None = None
    if unique_training_seed_count >= spec.minimum_independent_seeds and independent:
        enough_cases = all(
            int(summary["metrics"]["by_skill_axis"][axis]["exact_and_eos"]["cases"])
            >= spec.minimum_cases_per_axis
            and int(summary["metrics"]["by_skill_axis"][axis]["valid_eos"]["cases"])
            >= spec.minimum_cases_per_axis
            for summary in summaries
            for axis in PHASE1_AXES
        ) and all(
            int(summary["metrics"]["causal_pairs"][pair_kind]["cases"])
            >= spec.minimum_cases_per_axis
            for summary in summaries
            for pair_kind in ("sensitivity", "invariance")
        )
        if not enough_cases:
            decision, reason = "INCONCLUSIVE_INSUFFICIENT_CASES", "at least one skill axis is underpowered"
        else:
            run_level_uncertainty = _run_level_uncertainty(
                summaries, familywise_alpha=spec.run_level_familywise_alpha
            )
            all_run_intervals = [
                metric["simultaneous_interval"]
                for axis_metrics in run_level_uncertainty["by_skill_axis"].values()
                for metric in axis_metrics.values()
            ] + [
                metric["simultaneous_interval"]
                for pair_metrics in run_level_uncertainty["causal_pairs"].values()
                for metric in pair_metrics.values()
            ]
            sufficiently_precise = all(
                float(interval[1]) - float(interval[0])
                <= spec.maximum_run_level_interval_width
                for interval in all_run_intervals
            )
            primary_interval = run_level_uncertainty["by_skill_axis"][spec.primary_axis][
                "exact_and_eos"
            ]["simultaneous_interval"]
            if not sufficiently_precise:
                decision, reason = (
                    "BORDERLINE_DIAGNOSTIC",
                    "simultaneous run-level uncertainty is wider than the preregistered precision limit",
                )
            elif float(primary_interval[1]) <= spec.primary_floor_upper_max:
                decision, reason = "NO_GO_PRIMARY_FLOOR_DIAGNOSTIC", "primary capability remains below the preregistered floor band"
            elif float(primary_interval[0]) >= spec.primary_ceiling_lower_min:
                decision, reason = "NO_GO_PRIMARY_CEILING_DIAGNOSTIC", "primary capability is above the preregistered comparison band"
            else:
                capable = all(
                    run_level_uncertainty["by_skill_axis"][axis][metric_name][
                        "simultaneous_interval"
                    ][0] >= threshold
                    for axis in PHASE1_AXES
                    for metric_name, threshold in (
                        ("exact_and_eos", spec.minimum_exact_and_eos_lcb),
                        ("valid_eos", spec.minimum_eos_lcb),
                    )
                )
                pairs_pass = all(
                    run_level_uncertainty["causal_pairs"][pair_kind]["pair_success"][
                        "simultaneous_interval"
                    ][0] >= threshold
                    for pair_kind, threshold in (
                        ("sensitivity", spec.minimum_sensitivity_lcb),
                        ("invariance", spec.minimum_invariance_lcb),
                    )
                )
                if capable and pairs_pass:
                    decision, reason = "DIAGNOSTIC_NONFLOOR_NONCEILING", "within-suite diagnostics fall inside the preregistered planning band"
                else:
                    decision, reason = "BORDERLINE_DIAGNOSTIC", "one or more within-suite skill, EOS, or causal-pair bounds remain unresolved"
    body: dict[str, Any] = {
        "schema": BASELINE_GATE_SCHEMA,
        "decision": decision,
        "reason": reason,
        "decision_scope": "WITHIN_SUITE_DIAGNOSTIC_ONLY",
        "preregistration_sha256": spec.preregistration_sha256,
        "gate_spec_sha256": spec.sha256(),
        "source_summary_sha256": sorted(str(summary["sha256"]) for summary in summaries),
        "independent_training_seed_count": unique_training_seed_count,
        "independent_evaluation_seed_count": unique_evaluation_seed_count,
        "run_level_uncertainty": run_level_uncertainty,
        "cluster_aware_uncertainty_available": run_level_uncertainty is not None,
        "training_authorized": False,
        "mechanism_experiment_eligible": False,
        "next_action": (
            "retain as a development diagnostic; independently audit the run-level bounds and preregister a matched training-family-to-outcome intervention before any change"
            if decision == "DIAGNOSTIC_NONFLOOR_NONCEILING"
            else "resolve the named measurement/sensitivity failure; no intervention is eligible"
        ),
        "claim_ceiling": "Signac candidate-free development diagnostic with simultaneous run-level bounds when eligible and conditional on independent seed generation; no custody audit, training intervention effect, run authorization, or AGI claim.",
    }
    body["sha256"] = _sha256(body)
    return body


def _run_level_uncertainty(
    summaries: Sequence[Mapping[str, Any]], *, familywise_alpha: float
) -> dict[str, Any]:
    """Bound mean per-subject rates without treating tasks within a subject as iid.

    Each trained subject contributes one bounded [0, 1] rate per endpoint. The
    two-sided Hoeffding intervals use a Bonferroni correction across all frozen
    axis/EOS and paired endpoints. They are intentionally conservative and are
    descriptive over the sampled subjects and evaluation surfaces; they do not
    estimate a training intervention effect.
    """

    if not summaries:
        raise ValueError("run-level uncertainty requires at least one subject")
    if type(familywise_alpha) not in (int, float) or not math.isfinite(familywise_alpha):
        raise ValueError("run-level familywise alpha must be finite")
    if not 0.0 < familywise_alpha < 1.0:
        raise ValueError("run-level familywise alpha must lie strictly between zero and one")

    def subject_rate(record: Any, label: str) -> float:
        if not isinstance(record, Mapping):
            raise ValueError(f"{label} has no subject-level counts")
        successes, cases = record.get("successes"), record.get("cases")
        if (
            type(cases) is not int
            or cases <= 0
            or type(successes) is not int
            or not 0 <= successes <= cases
        ):
            raise ValueError(f"{label} has invalid subject-level counts")
        return successes / cases

    endpoint_rates: dict[str, list[float]] = {}
    for axis in PHASE1_AXES:
        for metric_name in ("exact_and_eos", "valid_eos"):
            endpoint = f"by_skill_axis.{axis}.{metric_name}"
            endpoint_rates[endpoint] = [
                subject_rate(summary["metrics"]["by_skill_axis"][axis][metric_name], endpoint)
                for summary in summaries
            ]
    for pair_kind in ("sensitivity", "invariance"):
        endpoint = f"causal_pairs.{pair_kind}.pair_success"
        endpoint_rates[endpoint] = [
            subject_rate(summary["metrics"]["causal_pairs"][pair_kind], endpoint)
            for summary in summaries
        ]
        eos_endpoint = f"causal_pairs.{pair_kind}.valid_eos_pair_rate"
        endpoint_rates[eos_endpoint] = [
            subject_rate(
                summary["metrics"]["causal_pairs"][pair_kind]["valid_eos_pair_rate"],
                eos_endpoint,
            )
            for summary in summaries
        ]

    subject_count = len(summaries)
    endpoint_count = len(endpoint_rates)
    radius = math.sqrt(
        math.log((2.0 * endpoint_count) / familywise_alpha) / (2.0 * subject_count)
    )

    def interval(values: Sequence[float]) -> dict[str, Any]:
        mean_rate = math.fsum(values) / subject_count
        return {
            "mean_subject_rate": mean_rate,
            "minimum_subject_rate": min(values),
            "maximum_subject_rate": max(values),
            "simultaneous_interval": [
                max(0.0, mean_rate - radius),
                min(1.0, mean_rate + radius),
            ],
        }

    by_axis = {
        axis: {
            metric_name: interval(endpoint_rates[f"by_skill_axis.{axis}.{metric_name}"])
            for metric_name in ("exact_and_eos", "valid_eos")
        }
        for axis in PHASE1_AXES
    }
    pair_bounds = {
        pair_kind: {
            "pair_success": interval(endpoint_rates[f"causal_pairs.{pair_kind}.pair_success"]),
            "valid_eos_pair_rate": interval(
                endpoint_rates[f"causal_pairs.{pair_kind}.valid_eos_pair_rate"]
            ),
        }
        for pair_kind in ("sensitivity", "invariance")
    }
    return {
        "schema": RUN_LEVEL_UNCERTAINTY_SCHEMA,
        "method": "SIMULTANEOUS_TWO_SIDED_HOEFFDING_BONFERRONI",
        "independent_unit": "trained-subject/evaluation-surface pair, conditional on independent seed generation",
        "independence_basis": "unique training/evaluation seeds and unique checkpoint/suite hashes; seed generation and custody are not independently audited here",
        "subject_aggregation": "equal-weight mean of each subject's within-surface rate",
        "subject_count": subject_count,
        "familywise_alpha": float(familywise_alpha),
        "familywise_confidence_level": 1.0 - float(familywise_alpha),
        "simultaneous_endpoint_count": endpoint_count,
        "unclipped_half_width": radius,
        "by_skill_axis": by_axis,
        "causal_pairs": pair_bounds,
        "claim_limit": "development subjects and sampled evaluation surfaces only, conditional on independent seed generation; not a training intervention effect or custody audit",
    }


def _source_bundle_sha256(relative_paths: Sequence[str]) -> str:
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for relative in sorted(relative_paths):
        payload = (root / relative).read_bytes()
        digest.update(relative.encode("utf-8"))
        digest.update(hashlib.sha256(payload).digest())
    return digest.hexdigest()


def build_development_fixture(
    *, seed: int, groups_per_family: int = 8
) -> tuple[EvaluationSuite, TaskFixtureBatch]:
    """Build one reproducible public development surface from the E0 generator."""

    if seed < 0 or groups_per_family <= 0:
        raise ValueError("development seed must be nonnegative and group count positive")
    suite = build_evaluation_suite(
        Split.DEVELOPMENT, seed=seed, groups_per_family=groups_per_family
    )
    suite.assert_valid()
    pair_memberships: dict[str, list[dict[str, str]]] = {}
    for pair in suite.pairs:
        for role, case in (("base", pair.base), ("changed", pair.changed)):
            pair_memberships.setdefault(case.case_id, []).append(
                {"pair_id": pair.pair_id, "pair_kind": pair.kind.value, "pair_role": role}
            )

    records: list[dict[str, Any]] = []
    for case in suite.cases:
        model_view = case.model_view()
        record: dict[str, Any] = {
            "task_id": case.case_id,
            "cluster_id": f"{case.family}:{case.seed}",
            "family": case.family,
            "difficulty": json.dumps(dict(case.difficulty), sort_keys=True, separators=(",", ":")),
            "split": Split.DEVELOPMENT.value,
            "prompt": model_view["prompt"],
            "candidates": list(case.candidates),
            "gold": case.answer,
        }
        if case.case_id in pair_memberships:
            record["causal_pairs"] = pair_memberships[case.case_id]
        records.append(record)

    generator_sha = _source_bundle_sha256((
        "e0_cognition/contracts.py",
        "e0_cognition/evaluation_generators.py",
    ))
    config_sha = _sha256({
        "generator_version": suite.generator_version,
        "seed": seed,
        "groups_per_family": groups_per_family,
        "split": Split.DEVELOPMENT.value,
        "causal_suite_sha256": suite.sha256(),
    })
    fixture = TaskFixtureBatch.freeze(
        generator_id="esoes-e0-causal-cognition",
        generator_sha256=generator_sha,
        generator_config_sha256=config_sha,
        seed=seed,
        split=Split.DEVELOPMENT.value,
        cases=records,
    )
    return suite, fixture


def build_phase1_protocol(fixture: TaskFixtureBatch, *, seed: int) -> EvaluationProtocol:
    """Bind a development fixture to candidate-free raw generation and causal metrics."""

    fixture.assert_valid()
    if fixture.split != Split.DEVELOPMENT.value or fixture.seed != seed:
        raise ValueError("Signac Phase-One protocol accepts only its matching development fixture")
    groups = sum(str(case["family"]) == "exact_contextual_copy" for case in fixture.cases)
    _suite, canonical_fixture = build_development_fixture(seed=seed, groups_per_family=groups)
    if canonical_fixture.sha256() != fixture.sha256():
        raise ValueError("fixture is not the canonical E0 development surface")
    return EvaluationProtocol(
        protocol_id=PROTOCOL_ID,
        generator_id=fixture.generator_id,
        generator_sha256=fixture.generator_sha256,
        generator_config_sha256=fixture.generator_config_sha256,
        fixture_sha256=fixture.sha256(),
        evaluator_sha256=evaluator_source_sha256(),
        split=fixture.split,
        seed=seed,
        n_cases=len(fixture.cases),
        decoding_mode="RAW_FREE_GENERATION",
        candidate_scoring_mode=SCORING_CONTRACT_SHA256,
        metrics=PHASE1_METRICS,
        statistical_rule="WILSON_BINOMIAL",
    )


def run_phase1_evaluation(
    *,
    protocol: EvaluationProtocol,
    subject: CoreSubjectManifest,
    adapter: Any,
    fixture: TaskFixtureBatch,
    evidence_path: Path,
) -> tuple[EvaluationReceipt, list[TaskLevelEvidence]]:
    """Run Signac Phase-One only on the exact M102 spec and a dev-only fixture."""

    subject.assert_valid()
    if subject.model_spec_sha256 != MODEL_SPEC.sha256():
        raise ValueError("Signac Phase-One subject must bind the exact M102 ModelSpec")
    assert_canonical_checkpoint_adapter(adapter)
    adapter_identity = adapter.identity
    _assert_checkpoint_source_binding(subject=subject, adapter_identity=adapter_identity)
    for label, actual, expected in (
        ("checkpoint", adapter_identity.checkpoint_sha256, subject.checkpoint_sha256),
        ("parameters", adapter_identity.parameter_sha256, subject.parameter_sha256),
        ("model spec", adapter_identity.model_spec_sha256, subject.model_spec_sha256),
        ("tokenizer", adapter_identity.tokenizer_artifact_sha256, subject.tokenizer_artifact_sha256),
    ):
        if actual != expected:
            raise ValueError(f"Phase-One adapter {label} identity disagrees with the subject manifest")
    if fixture.split != "development" or protocol.split != "development":
        raise ValueError("Signac Phase-One baseline scoring cannot consume sealed or fresh data")
    if protocol.protocol_id != PROTOCOL_ID or protocol.metrics != PHASE1_METRICS:
        raise ValueError("Signac Phase-One requires the frozen candidate-free protocol")
    if build_phase1_protocol(fixture, seed=protocol.seed).sha256() != protocol.sha256():
        raise ValueError("Signac Phase-One protocol does not match canonical fixture identity")
    return run_evaluation(
        protocol=protocol,
        subject=subject,
        adapter=adapter,
        fixture=fixture,
        evidence_path=evidence_path,
    )


def _rate(successes: int, total: int) -> dict[str, float | int]:
    lower, upper = wilson_interval(successes, total)
    return {
        "successes": successes,
        "cases": total,
        "rate": successes / total,
        "wilson95": [lower, upper],
    }


def _summarize_pair_outcomes(outcomes: Sequence[tuple[bool, bool]]) -> dict[str, Any]:
    if not outcomes:
        raise ValueError("cannot summarize an empty causal-pair group")
    successes = sum(success for success, _valid_eos in outcomes)
    valid_eos_pairs = sum(valid_eos for _success, valid_eos in outcomes)
    result = _rate(successes, len(outcomes))
    result["valid_eos_pair_rate"] = _rate(valid_eos_pairs, len(outcomes))
    return result


def summarize_phase1_evidence(
    *,
    receipt: EvaluationReceipt,
    evidence: Sequence[TaskLevelEvidence],
    protocol: EvaluationProtocol,
    fixture: TaskFixtureBatch,
    suite: EvaluationSuite,
    subject: CoreSubjectManifest,
) -> dict[str, Any]:
    """Derive an audit summary from the receipt-bound task-level evidence."""

    suite.assert_valid()
    subject.assert_valid()
    fixture.assert_valid()
    if protocol.protocol_id != PROTOCOL_ID or protocol.split != "development":
        raise ValueError("summary requires the Signac Phase-One development protocol")
    if fixture.sha256() != protocol.fixture_sha256 or fixture.split != "development":
        raise ValueError("summary fixture does not match the frozen development protocol")
    if receipt.receipt_schema != "anra-v5-evaluation-receipt/v3":
        raise ValueError("unsupported Phase-One evaluation receipt schema")
    if receipt.protocol_sha256 != protocol.sha256() or not receipt.derived_from_task_evidence:
        raise ValueError("receipt is not derived from this frozen protocol")
    if receipt.subject_manifest_sha256 != subject.sha256():
        raise ValueError("receipt does not bind the supplied training subject")
    if receipt.n_tasks != len(evidence) or receipt.n_tasks != len(fixture.cases):
        raise ValueError("receipt/evidence/fixture case counts disagree")
    if tuple(record.sha256() for record in evidence) != receipt.task_evidence_sha256:
        raise ValueError("task evidence hashes disagree with the evaluation receipt")
    if len({record.task_id for record in evidence}) != len(evidence):
        raise ValueError("task evidence repeats a task id")
    groups_per_family = sum(
        str(case["family"]) == "exact_contextual_copy" for case in fixture.cases
    )
    rebuilt_suite, rebuilt_fixture = build_development_fixture(
        seed=protocol.seed, groups_per_family=groups_per_family
    )
    if rebuilt_suite.sha256() != suite.sha256() or rebuilt_fixture.sha256() != fixture.sha256():
        raise ValueError("suite or fixture is not the deterministic surface bound by the protocol")
    fixture_by_id = {str(case["task_id"]): case for case in fixture.cases}
    if set(fixture_by_id) != {record.task_id for record in evidence}:
        raise ValueError("task evidence does not cover the frozen fixture exactly")
    checkpoint_hashes = {record.checkpoint_sha256 for record in evidence}
    subject_hashes = {record.subject_manifest_sha256 for record in evidence}
    adapter_hashes = {record.adapter_sha256 for record in evidence}
    if (
        len(checkpoint_hashes) != 1
        or checkpoint_hashes != {subject.checkpoint_sha256}
        or subject_hashes != {receipt.subject_manifest_sha256}
    ):
        raise ValueError("task evidence mixes checkpoint or subject identities")
    if adapter_hashes != {receipt.adapter_sha256}:
        raise ValueError("task evidence mixes adapter identities")
    for record in evidence:
        if (
            record.evidence_schema != "anra-v5-task-evidence/v2"
            or record.split != "development"
            or record.evaluation_mode != "RAW_FREE_GENERATION"
            or record.protocol_sha256 != protocol.sha256()
            or record.termination_valid is None
        ):
            raise ValueError("task evidence has wrong identity, split, mode, or missing EOS status")
        source = fixture_by_id[record.task_id]
        expected_pairs = tuple(sorted(
            (str(pair["pair_id"]), str(pair["pair_kind"]), str(pair["pair_role"]))
            for pair in source.get("causal_pairs", ())
        ))
        if (
            record.family != source["family"]
            or record.visible_prompt != source["prompt"]
            or record.gold_reference != source["gold"]
            or record.causal_pairs != expected_pairs
        ):
            raise ValueError("task evidence does not match its model-visible/gold fixture record")
    if fixture.generator_config_sha256 != protocol.generator_config_sha256:
        raise ValueError("fixture generator configuration differs from protocol")
    if len({record.seed for record in evidence}) != 1 or evidence[0].seed != protocol.seed:
        raise ValueError("task evidence seed disagrees with protocol")

    by_family: dict[str, list[TaskLevelEvidence]] = {}
    for record in evidence:
        by_family.setdefault(record.family, []).append(record)
    family_summary: dict[str, Any] = {}
    for family, records in sorted(by_family.items()):
        family_summary[family] = {
            "exact": _rate(sum(record.correct for record in records), len(records)),
            "valid_eos": _rate(sum(record.termination_valid is True for record in records), len(records)),
            "exact_and_eos": _rate(
                sum(record.correct and record.termination_valid is True for record in records),
                len(records),
            ),
        }

    interference_by_condition: dict[tuple[int, int], list[TaskLevelEvidence]] = {}
    for record in by_family.get("interference_retrieval", ()):
        difficulty_text = fixture_by_id[record.task_id].get("difficulty")
        try:
            difficulty = json.loads(str(difficulty_text))
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("interference-retrieval fixture has malformed difficulty metadata") from exc
        distractors = difficulty.get("distractors") if isinstance(difficulty, Mapping) else None
        quartile = (
            difficulty.get("context_position_quartile")
            if isinstance(difficulty, Mapping)
            else None
        )
        if (
            type(distractors) is not int
            or type(quartile) is not int
            or distractors not in INTERFERENCE_RETRIEVAL_GRID
            or quartile not in INTERFERENCE_RETRIEVAL_GRID[distractors]
        ):
            raise ValueError("interference-retrieval task is outside the frozen dose/position grid")
        interference_by_condition.setdefault((distractors, quartile), []).append(record)
    expected_conditions = {
        (distractors, quartile)
        for distractors, quartiles in INTERFERENCE_RETRIEVAL_GRID.items()
        for quartile in quartiles
    }
    if set(interference_by_condition) != expected_conditions:
        raise ValueError("development fixture does not cover the full interference-retrieval grid")
    interference_grid_summary = {
        str(distractors): {
            str(quartile): {
                "exact_and_eos": _rate(
                    sum(
                        record.correct and record.termination_valid is True
                        for record in interference_by_condition[(distractors, quartile)]
                    ),
                    len(interference_by_condition[(distractors, quartile)]),
                ),
                "valid_eos": _rate(
                    sum(
                        record.termination_valid is True
                        for record in interference_by_condition[(distractors, quartile)]
                    ),
                    len(interference_by_condition[(distractors, quartile)]),
                ),
            }
            for quartile in quartiles
        }
        for distractors, quartiles in INTERFERENCE_RETRIEVAL_GRID.items()
    }

    axis_summary: dict[str, Any] = {}
    for axis, families in PHASE1_AXES.items():
        records = [record for family in families for record in by_family.get(family, ())]
        if not records:
            raise ValueError(f"development fixture has no evidence for required axis {axis}")
        axis_summary[axis] = {
            "families": [family for family in families if family in by_family],
            "exact": _rate(sum(record.correct for record in records), len(records)),
            "valid_eos": _rate(sum(record.termination_valid is True for record in records), len(records)),
            "exact_and_eos": _rate(
                sum(record.correct and record.termination_valid is True for record in records),
                len(records),
            ),
        }

    pair_groups: dict[str, list[tuple[TaskLevelEvidence, str, str]]] = {}
    for record in evidence:
        for pair_id, kind, role in record.causal_pairs:
            pair_groups.setdefault(pair_id, []).append((record, kind, role))
    pair_outcomes_by_category: dict[str, list[tuple[bool, bool]]] = {
        category: [] for category in PHASE1_PAIR_KIND_CATEGORIES
    }
    pair_outcomes_by_family_kind: dict[str, dict[str, list[tuple[bool, bool]]]] = {}
    for members in pair_groups.values():
        if len(members) != 2 or {role for _record, _kind, role in members} != {"base", "changed"}:
            raise ValueError("malformed pair evidence in Phase-One task records")
        if len({kind for _record, kind, _role in members}) != 1:
            raise ValueError("causal-pair members disagree on intervention kind")
        base = next(record for record, _kind, role in members if role == "base")
        changed = next(record for record, _kind, role in members if role == "changed")
        if base.family != changed.family:
            raise ValueError("causal-pair members disagree on evaluation family")
        pair_kind = members[0][1]
        category = next(
            (
                name
                for name, kinds in PHASE1_PAIR_KIND_CATEGORIES.items()
                if pair_kind in kinds
            ),
            None,
        )
        if category is None:
            raise ValueError(f"unregistered Phase-One causal-pair kind: {pair_kind}")
        both_eos_valid = (
            base.termination_valid is True and changed.termination_valid is True
        )
        if category == "sensitivity":
            success = (
                both_eos_valid
                and base.correct
                and changed.correct
                and base.raw_output != changed.raw_output
            )
        else:
            success = (
                both_eos_valid
                and base.correct
                and changed.correct
                and base.raw_output == changed.raw_output
            )
        outcome = (success, both_eos_valid)
        pair_outcomes_by_category[category].append(outcome)
        pair_outcomes_by_family_kind.setdefault(base.family, {}).setdefault(
            pair_kind, []
        ).append(outcome)
    if any(not outcomes for outcomes in pair_outcomes_by_category.values()):
        missing = [name for name, outcomes in pair_outcomes_by_category.items() if not outcomes]
        raise ValueError(f"development suite has no {missing} causal pairs")
    pair_summary = {
        category: _summarize_pair_outcomes(outcomes)
        for category, outcomes in pair_outcomes_by_category.items()
    }
    pair_family_kind_summary = {
        family: {
            pair_kind: _summarize_pair_outcomes(outcomes)
            for pair_kind, outcomes in sorted(kind_groups.items())
        }
        for family, kind_groups in sorted(pair_outcomes_by_family_kind.items())
    }

    metric_values = dict(receipt.metric_values)
    computed_metrics = {
        "EXACT_ACCURACY": sum(record.correct for record in evidence) / len(evidence),
        "BALANCED_ACCURACY": sum(
            sum(record.correct for record in records) / len(records)
            for records in by_family.values()
        ) / len(by_family),
        "VALID_EOS_RATE": sum(record.termination_valid is True for record in evidence) / len(evidence),
        "EXACT_AND_EOS_RATE": sum(
            record.correct and record.termination_valid is True for record in evidence
        ) / len(evidence),
        "PAIRED_COUNTERFACTUAL_SENSITIVITY": pair_summary["sensitivity"]["rate"],
        "PAIRED_INVARIANCE_STABILITY": pair_summary["invariance"]["rate"],
    }
    if set(metric_values) != set(PHASE1_METRICS) or any(
        not math.isclose(float(metric_values[name]), float(value), rel_tol=0.0, abs_tol=1e-12)
        for name, value in computed_metrics.items()
    ):
        raise ValueError("receipt metrics are not derivable from the frozen task evidence")
    if not math.isclose(
        receipt.aggregate_correct_rate,
        computed_metrics["EXACT_ACCURACY"],
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("receipt aggregate is not derivable from task outcomes")
    body: dict[str, Any] = {
        "schema": SUMMARY_SCHEMA,
        "status": "MEASUREMENT_ONLY",
        "decision": "NO_PROMOTION_RULE_APPLIED",
        "training_authorized": False,
        "split": "development",
        "seed": protocol.seed,
        "evaluation_seed": protocol.seed,
        "training_seed": subject.seed,
        "identities": {
            "model_spec_sha256": MODEL_SPEC.sha256(),
            "suite_sha256": suite.sha256(),
            "fixture_sha256": fixture.sha256(),
            "protocol_sha256": protocol.sha256(),
            "generator_sha256": fixture.generator_sha256,
            "evaluation_receipt_sha256": receipt.sha256(),
            "task_evidence_sha256": receipt.evidence_artifact_sha256,
            "receipt_subject_manifest_sha256": receipt.subject_manifest_sha256,
            "checkpoint_sha256": evidence[0].checkpoint_sha256,
            "adapter_sha256": receipt.adapter_sha256,
            "evaluator_sha256": protocol.evaluator_sha256,
            "subject_manifest_sha256": subject.sha256(),
            "training_recipe_sha256": _training_recipe_sha256(subject),
            "tokenizer_sha256": subject.tokenizer_artifact_sha256,
            "training_spec_sha256": subject.training_spec_sha256,
            "data_manifest_sha256": subject.data_manifest_sha256,
            "pack_manifest_sha256": subject.pack_manifest_sha256,
            "optimizer_spec_sha256": subject.optimizer_spec_sha256,
            "schedule_spec_sha256": subject.schedule_spec_sha256,
            "curriculum_spec_sha256": subject.curriculum_spec_sha256,
            "source_tree_sha256": subject.source_tree_sha256,
        },
        "counts": {"tasks": len(evidence), "families": len(by_family), "causal_pairs": len(pair_groups)},
        "metrics": {
            "overall_exact": metric_values["EXACT_ACCURACY"],
            "balanced_family_exact": metric_values["BALANCED_ACCURACY"],
            "valid_eos": metric_values["VALID_EOS_RATE"],
            "exact_and_eos": metric_values["EXACT_AND_EOS_RATE"],
            "causal_pair_sensitivity": metric_values["PAIRED_COUNTERFACTUAL_SENSITIVITY"],
            "causal_pair_invariance": metric_values["PAIRED_INVARIANCE_STABILITY"],
            "by_family": family_summary,
            "by_skill_axis": axis_summary,
            "interference_retrieval_grid": interference_grid_summary,
            "causal_pairs": pair_summary,
            "causal_pairs_by_family_and_kind": pair_family_kind_summary,
        },
        "uncertainty_note": "Wilson intervals are within-suite diagnostics; correlated causal pairs and multiple tasks per generated episode require seed/cluster-aware analysis before promotion.",
        "claim_ceiling": "Candidate-free development behavior on this generated cognition suite only; no scale-transfer, training authorization, or AGI claim.",
    }
    body["sha256"] = _sha256(body)
    return body


__all__ = [
    "BASELINE_GATE_SCHEMA",
    "RUN_LEVEL_UNCERTAINTY_SCHEMA",
    "PHASE1_AXES",
    "PHASE1_METRICS",
    "INTERFERENCE_RETRIEVAL_GRID",
    "PHASE1_PAIR_FAMILY_KINDS",
    "PHASE1_PAIR_KIND_CATEGORIES",
    "PhaseOneBaselineGateSpec",
    "PROTOCOL_ID",
    "SUMMARY_SCHEMA",
    "assess_phase1_baseline",
    "build_development_fixture",
    "build_phase1_protocol",
    "run_phase1_evaluation",
    "summarize_phase1_evidence",
    "verify_phase1_summary",
]
