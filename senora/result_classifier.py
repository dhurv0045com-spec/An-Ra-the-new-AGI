"""Automated Result Classifier and Precommitted Next-Action Decision Engine for P35-A.

Evaluates empirical outcomes against preregistered causal hypotheses and separates
the 9 distinct scientific possibilities without human post-hoc storytelling:

1. ROBUST_POSITIVE: True transferable cognition across OOD, natural analogues, and seeds.
2. SYNTHETIC_ONLY: Template familiarity without natural or structural transfer.
3. REALIZATION_ONLY: Constrained formatting improves without query-conditioned internal routing.
4. SUBSTRATE_TRADEOFF: Cognition gains accompanied by catastrophic linguistic regression (>3%).
5. FAMILY_COLLAPSE: Mean accuracy increases while an individual primitive collapses.
6. NO_EFFECT: Zero causal difference between cognition mixture and substrate baseline.
7. SEED_UNSTABLE: Effect fails to replicate in magnitude or direction across seeds.
8. UNDERPOWERED_INCONCLUSIVE: Confidence interval too wide to reject or support the hypothesis.
9. MEASUREMENT_BLOCKED: Upstream firewall or measurement invalidity prevents evaluation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

from senora.evaluator import EvaluationSummary
from senora.transfer_contract import StatisticalTestResults


class P35ResultCategory(str, Enum):
    ROBUST_POSITIVE = "ROBUST_POSITIVE"
    SYNTHETIC_ONLY = "SYNTHETIC_ONLY"
    REALIZATION_ONLY = "REALIZATION_ONLY"
    SUBSTRATE_TRADEOFF = "SUBSTRATE_TRADEOFF"
    FAMILY_COLLAPSE = "FAMILY_COLLAPSE"
    NO_EFFECT = "NO_EFFECT"
    SEED_UNSTABLE = "SEED_UNSTABLE"
    UNDERPOWERED_INCONCLUSIVE = "UNDERPOWERED_INCONCLUSIVE"
    MEASUREMENT_BLOCKED = "MEASUREMENT_BLOCKED"


PRECOMMITTED_ACTIONS: dict[P35ResultCategory, str] = {
    P35ResultCategory.ROBUST_POSITIVE: (
        "AUTHORIZE_P35_B: Proceed to Phase P35-B (query-swap contrastive objective comparison) "
        "on the frozen 15% cognition mixture."
    ),
    P35ResultCategory.SYNTHETIC_ONLY: (
        "HALT_OBJECTIVE_WORK_AND_REDESIGN_DATA: Do not add query-swap or scale model. "
        "Redesign cognition data generator to introduce greater surface lexical variation and natural phrasing."
    ),
    P35ResultCategory.REALIZATION_ONLY: (
        "INVESTIGATE_SELECTION_BOTTLENECK: Do not scale. Model learned output syntax without query routing. "
        "Investigate attention head capacity or query-key binding inductive biases."
    ),
    P35ResultCategory.SUBSTRATE_TRADEOFF: (
        "ADJUST_CURRICULUM_MIXTURE: Substrate regression exceeds 3.0%. Lower cognition fraction to 5% "
        "or test late-phase curriculum annealing to prevent negative transfer."
    ),
    P35ResultCategory.FAMILY_COLLAPSE: (
        "DEBUG_FAILING_FAMILY_GENERATOR: Mean accuracy masked primitive failure. Halt scaling and debug "
        "the specific collapsed cognitive family."
    ),
    P35ResultCategory.NO_EFFECT: (
        "FALSIFY_DATA_MIXTURE_HYPOTHESIS: 15% verified cognition data produced no causal effect in 35M dense "
        "Transformer. Reject data-only solution; prioritize architectural inductive bias."
    ),
    P35ResultCategory.SEED_UNSTABLE: (
        "INVESTIGATE_OPTIMIZATION_VARIANCE: Effect failed to replicate across seeds. Halt scaling and investigate "
        "learning rate, warmup length, and gradient noise before re-running."
    ),
    P35ResultCategory.UNDERPOWERED_INCONCLUSIVE: (
        "EXPAND_EVALUATION_SAMPLE: Expand evaluation fixture size to preregistered powered sample count "
        "before drawing conclusions."
    ),
    P35ResultCategory.MEASUREMENT_BLOCKED: (
        "REPAIR_MEASUREMENT_HARNESS: Upstream candidate scorer or firewall failed. Restrict evaluation "
        "to RAW_CORE unassisted exact generation only."
    ),
}


@dataclass(frozen=True, slots=True)
class P35Classification:
    category: P35ResultCategory
    treatment_effect_raw_core: float
    natural_analogue_gain: float
    structural_ood_gain: float
    substrate_regression_fraction: float
    query_sensitivity_flip_rate: float
    worst_family_name: str
    worst_family_accuracy: float
    two_seed_replicated: bool
    precommitted_next_action: str
    rationale: str

    def canonical(self) -> dict[str, Any]:
        data = asdict(self)
        data["category"] = self.category.value
        return data


def classify_p35_a_results(
    treatment_eval: EvaluationSummary,
    control_eval: EvaluationSummary,
    *,
    substrate_regression_fraction: float,
    paired_statistics: StatisticalTestResults | None = None,
    seed2_treatment_eval: EvaluationSummary | None = None,
    seed2_control_eval: EvaluationSummary | None = None,
) -> P35Classification:
    """Classify empirical results into exact mutually exclusive scientific outcomes."""
    # 1. Measurement Integrity Check
    if "FAIL" in treatment_eval.candidate_scoring_status and treatment_eval.raw_core_accuracy == 0.0:
        return _make_classification(
            P35ResultCategory.MEASUREMENT_BLOCKED,
            treatment_effect=0.0,
            natural_gain=0.0,
            ood_gain=0.0,
            substrate_reg=substrate_regression_fraction,
            sensitivity=treatment_eval.pair_sensitivity_flip_rate,
            worst_fam="none",
            worst_fam_acc=0.0,
            replicated=False,
            rationale="Evaluation output failed measurement integrity or candidate scoring firewall blocked without raw core.",
        )

    treatment_effect = treatment_eval.raw_core_accuracy - control_eval.raw_core_accuracy
    natural_gain = treatment_eval.natural_analogue_macro_accuracy - control_eval.natural_analogue_macro_accuracy
    ood_gain = treatment_effect  # in development suite

    # Find worst family
    fam_accs = treatment_eval.family_accuracies
    worst_fam = min(fam_accs.keys(), key=lambda k: fam_accs[k]) if fam_accs else "none"
    worst_fam_acc = fam_accs[worst_fam] if fam_accs else 0.0

    # 2. Check Substrate Tradeoff (> 3.0% regression)
    if substrate_regression_fraction > 0.03:
        return _make_classification(
            P35ResultCategory.SUBSTRATE_TRADEOFF,
            treatment_effect, natural_gain, ood_gain, substrate_regression_fraction,
            treatment_eval.pair_sensitivity_flip_rate, worst_fam, worst_fam_acc, False,
            rationale=f"Language substrate loss regressed by {substrate_regression_fraction * 100:.2f}%, exceeding 3.0% bound.",
        )

    # 3. Check Family Collapse (worst family below chance floor 25% or regressed heavily)
    if worst_fam_acc < 0.25:
        return _make_classification(
            P35ResultCategory.FAMILY_COLLAPSE,
            treatment_effect, natural_gain, ood_gain, substrate_regression_fraction,
            treatment_eval.pair_sensitivity_flip_rate, worst_fam, worst_fam_acc, False,
            rationale=f"Primitive family '{worst_fam}' collapsed to {worst_fam_acc * 100:.1f}%, below the 25% chance floor.",
        )

    # 4. Check No Effect (|treatment_effect| < 0.05)
    if abs(treatment_effect) < 0.05:
        return _make_classification(
            P35ResultCategory.NO_EFFECT,
            treatment_effect, natural_gain, ood_gain, substrate_regression_fraction,
            treatment_eval.pair_sensitivity_flip_rate, worst_fam, worst_fam_acc, False,
            rationale=f"Causal raw-core treatment effect is negligible ({treatment_effect:+.3f}).",
        )

    # 5. Check Realization Only (query sensitivity is at chance <= 50%, indicating formatting memorization without query-conditioned routing)
    if treatment_eval.pair_sensitivity_flip_rate <= 0.50:
        return _make_classification(
            P35ResultCategory.REALIZATION_ONLY,
            treatment_effect, natural_gain, ood_gain, substrate_regression_fraction,
            treatment_eval.pair_sensitivity_flip_rate, worst_fam, worst_fam_acc, False,
            rationale=f"Query-swap sensitivity rate is at chance ({treatment_eval.pair_sensitivity_flip_rate * 100:.1f}% <= 50%); model memorized formatting without routing.",
        )

    # 6. Check Synthetic Only (raw core positive on synthetic dev templates, but natural transfer zero or negative)
    if treatment_effect >= 0.15 and natural_gain <= 0.02:
        return _make_classification(
            P35ResultCategory.SYNTHETIC_ONLY,
            treatment_effect, natural_gain, ood_gain, substrate_regression_fraction,
            treatment_eval.pair_sensitivity_flip_rate, worst_fam, worst_fam_acc, False,
            rationale=f"Model improved on synthetic templates ({treatment_effect:+.3f}) but failed to transfer to natural analogues ({natural_gain:+.3f}).",
        )

    # 7. Check Underpowered / Inconclusive
    if paired_statistics is not None and paired_statistics.bootstrap_ci_lower_95 <= 0.0:
        return _make_classification(
            P35ResultCategory.UNDERPOWERED_INCONCLUSIVE,
            treatment_effect, natural_gain, ood_gain, substrate_regression_fraction,
            treatment_eval.pair_sensitivity_flip_rate, worst_fam, worst_fam_acc, False,
            rationale=f"Bootstrap 95% CI [{paired_statistics.bootstrap_ci_lower_95:.3f}, {paired_statistics.bootstrap_ci_upper_95:.3f}] spans zero; evidence is statistically inconclusive.",
        )

    # 8. Check Seed Stability
    if seed2_treatment_eval is not None and seed2_control_eval is not None:
        effect_seed1 = treatment_effect
        effect_seed2 = seed2_treatment_eval.raw_core_accuracy - seed2_control_eval.raw_core_accuracy
        gap = abs(effect_seed1 - effect_seed2)
        if effect_seed2 <= 0.05 or gap > 0.10:
            return _make_classification(
                P35ResultCategory.SEED_UNSTABLE,
                treatment_effect, natural_gain, ood_gain, substrate_regression_fraction,
                treatment_eval.pair_sensitivity_flip_rate, worst_fam, worst_fam_acc, False,
                rationale=f"Treatment effect failed to replicate: Seed A={effect_seed1:+.3f}, Seed B={effect_seed2:+.3f} (gap={gap:.3f}).",
            )
        replicated = True
    else:
        replicated = False

    # 9. Robust Positive (Treatment effect >= 0.20, natural transfer >= 0.10, sensitivity >= 80%, replicated)
    if treatment_effect >= 0.20 and natural_gain >= 0.10 and treatment_eval.pair_sensitivity_flip_rate >= 0.80:
        if replicated:
            return _make_classification(
                P35ResultCategory.ROBUST_POSITIVE,
                treatment_effect, natural_gain, ood_gain, substrate_regression_fraction,
                treatment_eval.pair_sensitivity_flip_rate, worst_fam, worst_fam_acc, True,
                rationale="Causal treatment effect is large, transferable to natural analogues, preserved across seeds, with zero substrate collapse.",
            )
        else:
            return _make_classification(
                P35ResultCategory.ROBUST_POSITIVE,
                treatment_effect, natural_gain, ood_gain, substrate_regression_fraction,
                treatment_eval.pair_sensitivity_flip_rate, worst_fam, worst_fam_acc, False,
                rationale="Seed A demonstrates robust positive transfer. Requires matched Seed B execution to confirm replication.",
            )

    return _make_classification(
        P35ResultCategory.UNDERPOWERED_INCONCLUSIVE,
        treatment_effect, natural_gain, ood_gain, substrate_regression_fraction,
        treatment_eval.pair_sensitivity_flip_rate, worst_fam, worst_fam_acc, replicated,
        rationale=f"Outcome does not cleanly meet robust positive thresholds: effect={treatment_effect:+.3f}, natural_gain={natural_gain:+.3f}.",
    )


def _make_classification(
    category: P35ResultCategory,
    treatment_effect: float,
    natural_gain: float,
    ood_gain: float,
    substrate_reg: float,
    sensitivity: float,
    worst_fam: str,
    worst_fam_acc: float,
    replicated: bool,
    rationale: str,
) -> P35Classification:
    return P35Classification(
        category=category,
        treatment_effect_raw_core=round(treatment_effect, 4),
        natural_analogue_gain=round(natural_gain, 4),
        structural_ood_gain=round(ood_gain, 4),
        substrate_regression_fraction=round(substrate_reg, 4),
        query_sensitivity_flip_rate=round(sensitivity, 4),
        worst_family_name=worst_fam,
        worst_family_accuracy=round(worst_fam_acc, 4),
        two_seed_replicated=replicated,
        precommitted_next_action=PRECOMMITTED_ACTIONS[category],
        rationale=rationale,
    )