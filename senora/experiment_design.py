"""First recommended P35 experiment design: Cognition Mixture & Query-Swap Causal Screen (P35-CMS-1)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from v5_contracts.model_spec import ModelSpec


P35_MODEL_SPEC = ModelSpec(
    schema="anra-v5-p35-model-spec/v1",
    family="dense-decoder-transformer",
    vocabulary_size=24_576,
    width=384,
    layers=16,
    query_heads=6,
    kv_heads=3,  # 2:1 GQA
    head_dimension=64,
    ffn_width=1024,
    context_length=2048,
    rope_base=10_000.0,
    norm_epsilon=1e-5,
    tied_embeddings=True,
    qk_norm=True,
    qk_norm_affine=True,
    linear_bias=False,
    dropout=0.0,
)


@dataclass(frozen=True, slots=True)
class ExperimentArm:
    name: str
    description: str
    cognition_fraction: float
    natural_fraction: float
    code_fraction: float
    query_swap_lambda: float
    token_budget: int
    idealized_6nd_flops: int


@dataclass(frozen=True, slots=True)
class P35ExperimentPlan:
    schema: str
    experiment_id: str
    title: str
    hypothesis: str
    uncertain_assumption: str
    control_arm: str
    single_intended_change: str
    token_flop_matching: str
    primary_cognition_metric: str
    substrate_metric: str
    ood_test: str
    natural_transfer_requirement: str
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
    param_count = P35_MODEL_SPEC.parameter_receipt().total  # 35,414,400
    flops = 6 * param_count * token_budget  # 6ND FLOPs

    arms = [
        ExperimentArm(
            name="control-substrate-00",
            description="Pure pretraining control (75% natural text, 25% code, 0% cognition) with Cross-Entropy only.",
            cognition_fraction=0.0,
            natural_fraction=0.75,
            code_fraction=0.25,
            query_swap_lambda=0.0,
            token_budget=token_budget,
            idealized_6nd_flops=flops,
        ),
        ExperimentArm(
            name="cognition-mixture-15-ce",
            description="Cognition-infused mixture (65% natural, 20% code, 15% cognition) with Cross-Entropy only.",
            cognition_fraction=0.15,
            natural_fraction=0.65,
            code_fraction=0.20,
            query_swap_lambda=0.0,
            token_budget=token_budget,
            idealized_6nd_flops=flops,
        ),
        ExperimentArm(
            name="cognition-mixture-15-qswap",
            description="Cognition-infused mixture (65% natural, 20% code, 15% cognition) with CE + query-swap auxiliary contrast (lambda=0.10).",
            cognition_fraction=0.15,
            natural_fraction=0.65,
            code_fraction=0.20,
            query_swap_lambda=0.10,
            token_budget=token_budget,
            idealized_6nd_flops=flops,
        ),
    ]

    return P35ExperimentPlan(
        schema="senora-p35-experiment-plan/v1",
        experiment_id="P35-CMS-1",
        title="Cognition Mixture & Query-Swap Causal Screen",
        hypothesis=(
            "Standard dense Transformer pretraining on web text fails to acquire query-conditioned binding "
            "and state tracking because causal counterfactual pressure is too sparse; mixing 15% mechanically "
            "verified cognition data under standard Cross-Entropy induces robust internal binding and state "
            "tracking without regressing natural text representations."
        ),
        uncertain_assumption=(
            "Whether pure Cross-Entropy on synthetic cognition instances generalizes to fresh OOD domains "
            "and natural language analogues, or merely memorizes template syntax; and whether an explicit "
            "query-swap contrastive objective is necessary for query-conditioned routing."
        ),
        control_arm="control-substrate-00",
        single_intended_change=(
            "Arm 0 vs Arm 1: isolate the effect of 15% verified cognition data under identical CE loss. "
            "Arm 1 vs Arm 2: isolate the effect of counterfactual query-swap contrastive auxiliary loss (lambda=0.10) "
            "holding data stream and token budget identical."
        ),
        token_flop_matching=(
            f"All 3 arms consume exactly {token_budget:,} tokens and {flops:,} idealized 6ND FLOPs "
            "using the exact 35.4M parameter 2:1 GQA P35 architecture."
        ),
        primary_cognition_metric="Macro raw-Core exact match accuracy on Fresh-OOD Causal Suite (Split.FRESH).",
        substrate_metric="Cross-entropy loss on held-out general natural/code validation corpus (must regress <= 3.0%).",
        ood_test="Evaluation on Split.FRESH cases featuring unseen domains, relations, entity prefixes, and rule structures.",
        natural_transfer_requirement=(
            "Statistically significant positive gain (paired sign test p < 0.01) on natural language analogues "
            "for binding, state tracking, and relational composition."
        ),
        failure_abort_criteria=[
            "Non-finite loss (NaN/Inf) or gradient explosion norm > 100.0.",
            "Substrate regression > 3.0% compared to control-substrate-00.",
            "Synthetic-only gain: performance improves on training templates but fails to transfer to natural analogues (p >= 0.05).",
            "Assisted-only gain: improvements appear in Assisted mode but Raw Core shows zero gain.",
            "Worst-family collapse: any single cognition primitive family falls below chance.",
        ],
        result_that_would_change_mind=(
            "If Arm 1 (15% CE) matches or outperforms Arm 2 (query-swap) on fresh OOD and query sensitivity pairs "
            "with zero substrate loss regression, we will reject auxiliary contrastive objectives and freeze pure CE. "
            "If both Arm 1 and Arm 2 fail on fresh OOD despite learning synthetic templates, we will falsify the data-only "
            "hypothesis and prioritize architectural inductive bias."
        ),
        model_spec=P35_MODEL_SPEC.canonical(),
        arms=[asdict(a) for a in arms],
        prelaunch_status="FROZEN_SPEC_READY_FOR_REMOTE_LAUNCH",
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