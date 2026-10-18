from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .contracts import content_hash
from .generator import GENERATOR_VERSION, SuiteManifest


PROTOCOL_SCHEMA = "anra-discovery-gym/protocol/v2"


@dataclass(frozen=True, slots=True)
class ProtocolSpec:
    protocol_id: str
    stage: str
    hypothesis: str
    generator_version: str
    suite_manifest_sha256: str
    split_contract: Mapping[str, str]
    policy_ids: tuple[str, ...]
    baseline_ids: tuple[str, ...]
    budget: int
    equal_action_budget: bool
    primary_metrics: tuple[str, ...]
    secondary_metrics: tuple[str, ...]
    prospective_thresholds: Mapping[str, Any]
    confirmation_policy: str
    tuning_policy: str
    model_status: str
    attribution_policy: str

    def __post_init__(self) -> None:
        if self.stage != "DG0":
            raise ValueError("the first protocol is bounded to DG0")
        if self.budget <= 0:
            raise ValueError("protocol budget must be positive")
        if not self.equal_action_budget:
            raise ValueError("DG0 requires equal action budgets")
        if self.model_status != "NOT_RUN":
            raise ValueError("DG0 cannot claim a trained model")

    def to_dict(self) -> dict[str, Any]:
        body = {
            "schema": PROTOCOL_SCHEMA,
            "protocol_id": self.protocol_id,
            "stage": self.stage,
            "hypothesis": self.hypothesis,
            "generator_version": self.generator_version,
            "suite_manifest_sha256": self.suite_manifest_sha256,
            "split_contract": dict(self.split_contract),
            "policy_ids": list(self.policy_ids),
            "baseline_ids": list(self.baseline_ids),
            "budget": self.budget,
            "equal_action_budget": self.equal_action_budget,
            "primary_metrics": list(self.primary_metrics),
            "secondary_metrics": list(self.secondary_metrics),
            "prospective_thresholds": dict(self.prospective_thresholds),
            "confirmation_policy": self.confirmation_policy,
            "tuning_policy": self.tuning_policy,
            "model_status": self.model_status,
            "attribution_policy": self.attribution_policy,
        }
        return {**body, "protocol_sha256": content_hash(body)}


def freeze_dg0_protocol(suite_manifest: SuiteManifest, *, budget: int = 8) -> ProtocolSpec:
    return ProtocolSpec(
        protocol_id="DG0-instrument-validity/v2",
        stage="DG0",
        hypothesis="A deterministic causal-program environment can expose legal, costly evidence and independently score withheld outcomes without hidden-state leakage.",
        generator_version=GENERATOR_VERSION,
        suite_manifest_sha256=content_hash(suite_manifest.to_dict()),
        split_contract={
            "train": "structure families fixed before rendering",
            "development": "disjoint structures for design diagnostics",
            "confirmation": "disjoint structures reserved for frozen protocol",
            "transfer": "held-out graph and operator combinations",
        },
        policy_ids=("information_gain",),
        baseline_ids=(
            "random",
            "fixed",
            "uncertainty",
            "information_gain",
            "no_experiment",
            "surface_shortcut",
            "task_identity_shortcut",
            "oracle",
        ),
        budget=budget,
        equal_action_budget=True,
        primary_metrics=(
            "evaluator_recomputation_agreement",
            "legal_action_and_cost_audit",
            "public_firewall_pass",
            "structural_split_disjointness",
        ),
        secondary_metrics=(
            "final_task_success",
            "prediction_accuracy",
            "prediction_brier",
            "information_gain_per_cost",
            "invalid_action_rate",
            "budget_overrun_rate",
        ),
        prospective_thresholds={
            "evaluator_recomputation_agreement": 1.0,
            "legal_action_and_cost_audit": "all_checks_pass",
            "public_firewall_pass": True,
            "structural_split_disjointness": True,
            "scientific_learning_threshold": "not_defined_in_DG0",
        },
        confirmation_policy="confirmation worlds are generated and audited but no model is trained or tuned",
        tuning_policy="no policy, threshold, or generator choice may use confirmation outcomes",
        model_status="NOT_RUN",
        attribution_policy="every result names policy, environment, evaluator, and any oracle separately",
    )


def protocol_template() -> dict[str, Any]:
    return {
        "schema": PROTOCOL_SCHEMA,
        "protocol_id": "DG1-prospective-replace-me",
        "stage": "DG1",
        "hypothesis": "state the single learnability hypothesis before generating data",
        "generator_version": "pin exact generator version",
        "suite_manifest_sha256": "pin suite manifest hash",
        "split_contract": {
            "train": "training structures only",
            "development": "development structures only",
            "confirmation": "untouched confirmation structures",
            "transfer": "one declared structural axis per result",
        },
        "policy_ids": ["learned-policy-id"],
        "baseline_ids": ["random", "fixed", "uncertainty", "information_gain", "no_experiment", "oracle"],
        "budget": 0,
        "equal_action_budget": True,
        "equal_inference_budget": True,
        "primary_metrics": ["final_task_success", "prediction_accuracy"],
        "secondary_metrics": ["prediction_brier", "information_gain_per_cost", "regret"],
        "prospective_thresholds": {
            "primary_effect": "declare before confirmation",
            "uncertainty": "cluster by independent world",
            "model_training": "from scratch only; record initialization and checkpoint",
        },
        "confirmation_policy": "freeze hashes and thresholds before opening confirmation outcomes",
        "tuning_policy": "development only; confirmation and transfer are never tuning pools",
        "model_status": "PLANNED_NOT_RUN",
        "attribution_policy": "separate learner, environment, evaluator, teacher, and external-tool decisions",
    }
