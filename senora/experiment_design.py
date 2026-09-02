"""Sequential P35 Experiment Plan: P35-CMS-1 (Cognition Mixture & Query-Swap Causal Screen).

Implements a sequential decision architecture:
- Phase P35-A (Data Mixture Causal Screen):
    Control Substrate (0% cognition) vs Cognition Mixture (15% cognition) under pure CE.
    Preserves 65:20 natural:code ratio in the non-cognition remainder.
    Evaluated strictly on DEVELOPMENT + Structural-OOD Development.
    If no credible effect: STOP (do not spend compute on query-swap).
- Phase P35-B (Objective Screen, conditional on P35-A):
    Cognition Mixture 15% CE vs Cognition Mixture 15% CE + Query-Swap Contrastive (lambda=0.10).
    Token-matched (50M tokens) with explicit accounting for auxiliary forward calculations.
- Prospective Confirmation:
    FRESH / SEALED suites evaluated ONLY prospective to confirm the frozen winning recipe.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from senora.data_pipeline import (
    BASE_CODE_PARTS,
    BASE_NATURAL_PARTS,
    MIXTURE_COGNITION_15,
    MIXTURE_CONTROL_SUBSTRATE,
)
from senora.model import (
    EXPECTED_P35_PARAMETER_COUNT,
    P35_MODEL_SPEC,
    get_p35_parameter_receipt,
)


@dataclass(frozen=True, slots=True)
class ExperimentArm:
    name: str
    phase: str  # "P35-A" or "P35-B"
    description: str
    cognition_fraction: float
    natural_fraction: float
    code_fraction: float
    query_swap_lambda: float
    token_budget: int
    idealized_6nd_flops: int
    matching_basis: str  # "FLOP_MATCHED" or "TOKEN_MATCHED"


@dataclass(frozen=True, slots=True)
class P35ExperimentPlan:
    schema: str
    experiment_id: str
    title: str
    hypothesis: str
    uncertain_assumption: str
    sequential_decision_structure: str
    control_arm: str
    single_intended_change: str
    compute_accounting: str
    primary_cognition_metric: str
    substrate_metric: str
    development_evaluation_suite: str
    prospective_confirmation_suite: str
    failure_abort_criteria: list[str]
    result_that_would_change_mind: str
    model_spec: dict[str, Any]
    arms: list[dict[str, Any]]
    prelaunch_status: str

    def canonical(self) -> dict[str, Any]:
        return asdict(self)

    def sha256(self) -> str:
        payload = json.dumps(self.canonical(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def build_p35_cms1_plan() -> P35ExperimentPlan:
    token_budget = 50_000_000
    param_receipt = get_p35_parameter_receipt()
    param_count = param_receipt.total_parameters  # Exactly 35,411,328
    flops_base = 6 * param_count * token_budget  # 10,623,398,400,000,000 (1.0623e16)

    # Arm 0: Control Substrate (0% cognition, normalized 65:20 natural:code ratio)
    arm_control = ExperimentArm(
        name="control-substrate-00",
        phase="P35-A",
        description=(
            "Pure pretraining baseline: 0% cognition data. Remaining 100% tokens allocated in "
            f"exact {int(BASE_NATURAL_PARTS)}:{int(BASE_CODE_PARTS)} ratio "
            f"({MIXTURE_CONTROL_SUBSTRATE.natural_fraction:.4f} natural, {MIXTURE_CONTROL_SUBSTRATE.code_fraction:.4f} code) "
            "with pure Cross-Entropy objective."
        ),
        cognition_fraction=0.0,
        natural_fraction=MIXTURE_CONTROL_SUBSTRATE.natural_fraction,
        code_fraction=MIXTURE_CONTROL_SUBSTRATE.code_fraction,
        query_swap_lambda=0.0,
        token_budget=token_budget,
        idealized_6nd_flops=flops_base,
        matching_basis="FLOP_MATCHED",
    )

    # Arm 1: 15% Cognition Mixture, Pure CE
    arm_cog_ce = ExperimentArm(
        name="cognition-mixture-15-ce",
        phase="P35-A",
        description=(
            "Cognition-infused mixture: 15% verified cognition data. Remaining 85% tokens allocated in "
            f"exact {int(BASE_NATURAL_PARTS)}:{int(BASE_CODE_PARTS)} ratio "
            f"({MIXTURE_COGNITION_15.natural_fraction:.4f} natural, {MIXTURE_COGNITION_15.code_fraction:.4f} code) "
            "with pure Cross-Entropy objective."
        ),
        cognition_fraction=0.15,
        natural_fraction=MIXTURE_COGNITION_15.natural_fraction,
        code_fraction=MIXTURE_COGNITION_15.code_fraction,
        query_swap_lambda=0.0,
        token_budget=token_budget,
        idealized_6nd_flops=flops_base,
        matching_basis="FLOP_MATCHED",
    )

    # Arm 2: 15% Cognition Mixture, CE + Query-Swap (Conditional on Phase P35-A pass)
    arm_cog_qswap = ExperimentArm(
        name="cognition-mixture-15-qswap",
        phase="P35-B",
        description=(
            "Cognition-infused mixture: 15% verified cognition data, holding data stream and base token "
            "count identical to Arm 1, with composite objective: CE + query-swap contrastive (lambda=0.10)."
        ),
        cognition_fraction=0.15,
        natural_fraction=MIXTURE_COGNITION_15.natural_fraction,
        code_fraction=MIXTURE_COGNITION_15.code_fraction,
        query_swap_lambda=0.10,
        token_budget=token_budget,
        idealized_6nd_flops=flops_base,
        matching_basis="TOKEN_MATCHED",
    )

    return P35ExperimentPlan(
        schema="senora-p35-experiment-plan/v2",
        experiment_id="P35-CMS-1",
        title="Cognition Mixture & Query-Swap Sequential Causal Screen",
        hypothesis=(
            "Dense Transformer pretraining fails on query-conditioned value binding and state tracking because "
            "causal counterfactual pressure is too sparse in raw text; adding 15% verified cognition data under "
            "pure CE induces robust internal routing without degrading general linguistic substrate."
        ),
        uncertain_assumption=(
            "Whether pure Cross-Entropy on synthetic cognition instances generalizes to out-of-distribution "
            "domains and natural language analogues, or merely memorizes template surface forms; and whether "
            "an explicit counterfactual query-swap contrastive objective is necessary to prevent candidate-prior bias."
        ),
        sequential_decision_structure=(
            "Phase P35-A executes Arm 0 (Control) vs Arm 1 (15% CE). Evaluated strictly on Split.DEVELOPMENT. "
            "If Arm 1 fails to produce statistically significant raw-Core OOD gain without substrate regression, "
            "the experiment STOPS (falsifying the data-only hypothesis; query-swap compute is saved). "
            "Only if Arm 1 passes does Phase P35-B execute Arm 2 (15% CE+qswap) to test objective efficacy."
        ),
        control_arm="control-substrate-00",
        single_intended_change=(
            "P35-A: Isolate the causal effect of adding 15% verified cognition data, preserving the 65:20 "
            "natural:code ratio in the non-cognition remainder. "
            "P35-B: Isolate the causal effect of the query-swap contrastive objective holding data identical."
        ),
        compute_accounting=(
            f"Model parameters: {param_count:,} (verified exact). "
            f"Base token budget: {token_budget:,} tokens per arm. "
            f"Base idealized 6ND FLOPs: {flops_base:,}. "
            "Arm 0 and Arm 1 are strictly FLOP-matched. "
            "Arm 2 is token-matched with explicit accounting for paired query-swap forward calculations."
        ),
        primary_cognition_metric="Macro raw-Core exact match accuracy on Development Causal Suite (Split.DEVELOPMENT).",
        substrate_metric="Validation cross-entropy loss on held-out natural/code corpus (regression <= 3.0%).",
        development_evaluation_suite="Split.DEVELOPMENT (used for arm comparison and recipe selection).",
        prospective_confirmation_suite="Split.FRESH (strictly prospectively locked; evaluated ONLY after recipe freeze).",
        failure_abort_criteria=[
            "Non-finite loss (NaN/Inf) or gradient explosion norm > 100.0.",
            "Substrate regression > 3.0% relative to control-substrate-00.",
            "Zero parameter movement or silent Adam optimizer failure.",
            "Synthetic-only gain: performance improves on training templates but fails to transfer to natural analogues (p >= 0.05).",
            "Assisted-only gain: improvements appear in Assisted mode but Raw Core shows zero gain.",
            "Worst-family collapse: any single cognition primitive family falls below chance.",
        ],
        result_that_would_change_mind=(
            "If Arm 1 (15% CE) produces a large, transferable raw-Core gain on Development without substrate "
            "regression, and Arm 2 (query-swap) yields no additional benefit, we falsify the necessity of auxiliary "
            "contrastive objectives and freeze pure CE. If Arm 1 shows zero OOD gain, we falsify the hypothesis "
            "that data mixture alone solves small-model routing."
        ),
        model_spec=P35_MODEL_SPEC.canonical(),
        arms=[asdict(a) for a in [arm_control, arm_cog_ce, arm_cog_qswap]],
        prelaunch_status="FROZEN_SEQUENTIAL_SPEC_READY_FOR_REMOTE_LAUNCH",
    )


def main() -> int:
    plan = build_p35_cms1_plan()
    output_path = Path("artifacts/v5/p35_cms1_plan.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(plan.canonical(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote P35-CMS-1 plan to {output_path} (plan_sha256={plan.sha256()[:16]}...)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())