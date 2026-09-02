"""Canonical 4-tier evaluation interface, counterfactual pair metrics, and scorer firewall guards."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Mapping, Sequence

from e0_cognition.contracts import (
    CausalCase,
    EvaluationSuite,
    INVARIANCE_PAIR_KINDS,
    SENSITIVITY_PAIR_KINDS,
    PairKind,
)
from e0_cognition.metrics import (
    accuracy_by_difficulty,
    measure_assistance,
    measure_pair_behavior,
    measure_realization,
    measure_selection,
    selection_eligible,
)


class EvaluationTier(str, Enum):
    RAW_CORE = "raw_core"
    CONSTRAINED_REALIZATION = "constrained_realization"
    ASSISTED = "assisted"
    ORACLE_SCORER = "oracle_scorer"


class ScorerFirewallBlockedError(RuntimeError):
    """Raised when candidate suffix scoring is attempted without a verified scorer firewall pass."""
    pass


@dataclass(frozen=True, slots=True)
class CasePrediction:
    case_id: str
    raw_output: str
    constrained_output: str
    assisted_output: str | None = None
    candidate_logprobs: Mapping[str, float] | None = None


@dataclass(frozen=True, slots=True)
class EvaluationSummary:
    schema: str
    suite_split: str
    case_count: int
    raw_core_accuracy: float
    constrained_accuracy: float
    assisted_accuracy: float | None
    intervention_dependence_rate: float | None
    assistance_harm_rate: float | None
    family_accuracies: Mapping[str, float]
    difficulty_curves: Mapping[str, Mapping[int, float]]
    pair_sensitivity_flip_rate: float
    pair_invariance_stable_rate: float
    natural_analogue_macro_accuracy: float
    candidate_scoring_status: str
    candidate_selection_accuracy: float | None


class SenoraEvaluator:
    """Canonical evaluator strictly maintaining the 4 distinct evaluation tiers."""

    def __init__(
        self,
        suite: EvaluationSuite,
        *,
        scorer_firewall_status: str = "FAIL_DEVELOPMENT_POLICY",
    ) -> None:
        self.suite = suite
        self.scorer_firewall_status = scorer_firewall_status

    def evaluate_predictions(
        self,
        predictions: Sequence[CasePrediction],
        *,
        general_substrate_loss: float | None = None,
    ) -> EvaluationSummary:
        pred_map = {p.case_id: p for p in predictions}
        cases = self.suite.cases

        raw_hits: list[bool] = []
        constrained_hits: list[bool] = []
        assisted_hits: list[bool] = []
        intervention_deps: list[bool] = []
        assistance_harms: list[bool] = []

        family_counts: dict[str, int] = {}
        family_raw_hits: dict[str, int] = {}
        natural_analogue_families = {
            "natural_binding_analogue",
            "natural_state_analogue",
            "natural_composition_analogue",
        }
        natural_hits: list[bool] = []

        # Tier 1, 2, 3 processing
        raw_pred_strings: dict[str, str] = {}
        for case in cases:
            pred = pred_map.get(case.case_id)
            if pred is None:
                raise ValueError(f"missing prediction for case {case.case_id}")

            raw_ok = pred.raw_output.strip() == case.answer
            constrained_ok = pred.constrained_output.strip() == case.answer
            raw_hits.append(raw_ok)
            constrained_hits.append(constrained_ok)
            raw_pred_strings[case.case_id] = pred.raw_output.strip()

            family_counts[case.family] = family_counts.get(case.family, 0) + 1
            if raw_ok:
                family_raw_hits[case.family] = family_raw_hits.get(case.family, 0) + 1

            if case.family in natural_analogue_families:
                natural_hits.append(raw_ok)

            if pred.assisted_output is not None:
                asst_ok = pred.assisted_output.strip() == case.answer
                assisted_hits.append(asst_ok)
                intervention_deps.append(asst_ok and not raw_ok)
                assistance_harms.append(raw_ok and not asst_ok)

        # Counterfactual pair behavior
        pair_metrics = measure_pair_behavior(self.suite, raw_pred_strings)
        sens_rate = (
            pair_metrics.sensitivity_correct_flip / pair_metrics.sensitivity_total
            if pair_metrics.sensitivity_total > 0
            else 0.0
        )
        inv_rate = (
            pair_metrics.invariance_stable / pair_metrics.invariance_total
            if pair_metrics.invariance_total > 0
            else 0.0
        )

        # Difficulty curves
        difficulty = accuracy_by_difficulty(self.suite, raw_pred_strings)

        # Family accuracies
        family_accs = {
            fam: family_raw_hits.get(fam, 0) / count
            for fam, count in sorted(family_counts.items())
        }

        # Tier 4: Candidate scoring / Scorer firewall check
        selection_acc = None
        if any(p.candidate_logprobs is not None for p in predictions):
            if self.scorer_firewall_status != "PASSED":
                scorer_status = f"BLOCKED_BY_SCORER_FIREWALL:{self.scorer_firewall_status}"
            else:
                scorer_status = "CERTIFIED_CANDIDATE_SCORING"
                eligible = [c for c in cases if selection_eligible(c)]
                hits = 0
                for c in eligible:
                    p = pred_map[c.case_id]
                    if p.candidate_logprobs:
                        top = max(p.candidate_logprobs.items(), key=lambda x: x[1])[0]
                        hits += int(top == c.answer)
                selection_acc = hits / max(1, len(eligible))
        else:
            scorer_status = "NOT_EVALUATED"

        return EvaluationSummary(
            schema="senora-evaluation-summary/v1",
            suite_split=self.suite.split.value,
            case_count=len(cases),
            raw_core_accuracy=sum(raw_hits) / len(raw_hits),
            constrained_accuracy=sum(constrained_hits) / len(constrained_hits),
            assisted_accuracy=sum(assisted_hits) / len(assisted_hits) if assisted_hits else None,
            intervention_dependence_rate=sum(intervention_deps) / len(intervention_deps) if intervention_deps else None,
            assistance_harm_rate=sum(assistance_harms) / len(assistance_harms) if assistance_harms else None,
            family_accuracies=family_accs,
            difficulty_curves=difficulty,
            pair_sensitivity_flip_rate=sens_rate,
            pair_invariance_stable_rate=inv_rate,
            natural_analogue_macro_accuracy=sum(natural_hits) / max(1, len(natural_hits)),
            candidate_scoring_status=scorer_status,
            candidate_selection_accuracy=selection_acc,
        )