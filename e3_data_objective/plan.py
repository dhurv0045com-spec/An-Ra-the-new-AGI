"""Build the staged E3 P35 data/objective experiment plan.

The plan fixes mixture arithmetic and causal promotion gates before results are
available.  Phase B cannot be instantiated until Phase A selects one mixture
and one adjacent mixture.  No training code lives in this package.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from v5_contracts.run_spec import DataMixture


SCREEN_TOKENS = 200_000_000
EVALUATION_BOUNDARIES = (50_000_000, 100_000_000, 200_000_000)
COGNITION_FRACTIONS = (0.05, 0.15, 0.30)
QUERY_SWAP_LAMBDAS = (0.0, 0.05, 0.15)


def _valid_sha256(value: str | None) -> bool:
    return value is not None and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _mixture(cognition_fraction: float) -> DataMixture:
    remaining = 1.0 - cognition_fraction
    return DataMixture(
        natural=remaining * 65 / 85,
        code_math_formal=remaining * 20 / 85,
        verified_cognition=cognition_fraction,
    )


@dataclass(frozen=True, slots=True)
class MixtureArm:
    name: str
    cognition_fraction: float
    mixture: DataMixture
    objective: str = "causal-next-token-ce"

    def assert_valid(self) -> None:
        if self.cognition_fraction not in COGNITION_FRACTIONS:
            raise ValueError("E3 cognition fraction is not preregistered")
        self.mixture.assert_valid()
        if abs(self.mixture.verified_cognition - self.cognition_fraction) > 1e-12:
            raise ValueError("mixture cognition fraction drift")
        expected = _mixture(self.cognition_fraction)
        if any(
            abs(getattr(self.mixture, field) - getattr(expected, field)) > 1e-12
            for field in ("natural", "code_math_formal", "verified_cognition")
        ):
            raise ValueError("non-cognition ratio must remain 65:20")
        if self.objective != "causal-next-token-ce":
            raise ValueError("Phase A must remain CE-only")

    def receipt(self) -> dict[str, Any]:
        self.assert_valid()
        return {
            "name": self.name,
            "cognition_fraction": self.cognition_fraction,
            "mixture": asdict(self.mixture),
            "token_allocation": self.mixture.token_allocation(SCREEN_TOKENS),
            "objective": self.objective,
        }


@dataclass(frozen=True, slots=True)
class ObjectiveArm:
    name: str
    query_swap_lambda: float
    scope: str = "mechanically-verified-query-swap-pairs-only"

    def assert_valid(self) -> None:
        if self.query_swap_lambda not in QUERY_SWAP_LAMBDAS:
            raise ValueError("query-swap lambda is not preregistered")
        if self.scope != "mechanically-verified-query-swap-pairs-only":
            raise ValueError("auxiliary scope drift")


@dataclass(frozen=True, slots=True)
class E3Plan:
    schema: str
    experiment_id: str
    tokenizer_sha256: str | None
    corpus_manifest_sha256: str | None
    generator_sha256: str | None
    e2_winner_sha256: str | None
    model_constructor_sha256: str | None
    raw_byte_budget: int | None
    screen_tokens: int
    evaluation_boundaries: tuple[int, ...]
    mixture_arms: tuple[MixtureArm, ...]
    objective_arms: tuple[ObjectiveArm, ...]
    selected_mixture: str | None
    selected_neighbor: str | None
    trace_arm_enabled: bool
    trace_trigger_receipt_sha256: str | None

    def assert_valid(self) -> None:
        if self.schema != "esoes-e3-plan/v1":
            raise ValueError("unexpected E3 plan schema")
        if self.screen_tokens != SCREEN_TOKENS or self.evaluation_boundaries != EVALUATION_BOUNDARIES:
            raise ValueError("E3 screen budget or boundaries drift")
        if tuple(arm.cognition_fraction for arm in self.mixture_arms) != COGNITION_FRACTIONS:
            raise ValueError("E3 must contain ordered 5/15/30 percent mixture arms")
        if tuple(arm.query_swap_lambda for arm in self.objective_arms) != QUERY_SWAP_LAMBDAS:
            raise ValueError("E3 objective arms must be CE/0.05/0.15")
        for arm in self.mixture_arms:
            arm.assert_valid()
            if sum(arm.mixture.token_allocation(self.screen_tokens).values()) != self.screen_tokens:
                raise ValueError("E3 mixture token allocation mismatch")
        for arm in self.objective_arms:
            arm.assert_valid()
        names = [arm.name for arm in self.mixture_arms]
        if (self.selected_mixture is None) != (self.selected_neighbor is None):
            raise ValueError("Phase B requires both winner and adjacent neighbor")
        if self.selected_mixture is not None:
            if self.selected_mixture not in names or self.selected_neighbor not in names:
                raise ValueError("Phase B selected an unknown mixture")
            left = names.index(self.selected_mixture)
            right = names.index(self.selected_neighbor)
            if abs(left - right) != 1:
                raise ValueError("Phase B neighbor must be adjacent to the Phase A winner")
        if self.trace_arm_enabled and not _valid_sha256(self.trace_trigger_receipt_sha256):
            raise ValueError("trace arm requires a hashed composition-transfer trigger")
        if not self.trace_arm_enabled and self.trace_trigger_receipt_sha256 is not None:
            raise ValueError("disabled trace arm cannot carry a trigger receipt")
        if self.raw_byte_budget is not None and self.raw_byte_budget <= 0:
            raise ValueError("raw-byte budget must be positive")

    def status(self) -> str:
        self.assert_valid()
        dependencies = (
            self.tokenizer_sha256,
            self.corpus_manifest_sha256,
            self.generator_sha256,
            self.e2_winner_sha256,
            self.model_constructor_sha256,
        )
        if any(value is None for value in dependencies) or self.raw_byte_budget is None:
            return "BLOCKED_UPSTREAM_INPUTS"
        if not all(_valid_sha256(value) for value in dependencies):
            raise ValueError("E3 dependency hashes must be lowercase SHA-256")
        if self.selected_mixture is None:
            return "READY_FOR_PHASE_A_MIXTURE_SCREEN"
        return "READY_FOR_PHASE_B_OBJECTIVE_SCREEN"

    def as_dict(self) -> dict[str, Any]:
        self.assert_valid()
        payload: dict[str, Any] = {
            "schema": self.schema,
            "experiment_id": self.experiment_id,
            "status": self.status(),
            "dependencies": {
                "tokenizer_sha256": self.tokenizer_sha256,
                "corpus_manifest_sha256": self.corpus_manifest_sha256,
                "generator_sha256": self.generator_sha256,
                "e2_winner_sha256": self.e2_winner_sha256,
                "model_constructor_sha256": self.model_constructor_sha256,
            },
            "raw_byte_budget": self.raw_byte_budget,
            "screen_tokens": self.screen_tokens,
            "evaluation_boundaries": list(self.evaluation_boundaries),
            "phase_a_mixture_arms": [arm.receipt() for arm in self.mixture_arms],
            "phase_b": {
                "selected_mixture": self.selected_mixture,
                "selected_neighbor": self.selected_neighbor,
                "objective_arms": [asdict(arm) for arm in self.objective_arms],
            },
            "trace_arm": {
                "enabled": self.trace_arm_enabled,
                "trigger_receipt_sha256": self.trace_trigger_receipt_sha256,
                "maximum_trace_exposure_fraction": 0.25,
                "required_evaluation": "trace-free",
            },
            "comparison_policy": {
                "fixed": [
                    "tokenizer", "model", "optimizer", "schedule", "source order",
                    "raw bytes", "tokens", "evaluation fixtures",
                ],
                "phase_a_changes_only": "verified cognition fraction",
                "phase_b_changes_only": "query-swap auxiliary lambda",
            },
            "promotion_gates": {
                "primary": "worst-family fresh-OOD cognition per measured training FLOP",
                "natural_analogue_transfer_required": True,
                "candidate_free_improvement_required": True,
                "raw_core_improvement_required": True,
                "maximum_substrate_loss_regression_fraction": 0.03,
                "synthetic_only_gain_rejected": True,
                "assisted_only_gain_rejected": True,
            },
            "limitations": [
                "This artifact freezes experiment logic; it is not a training result.",
                "Phase B cannot be chosen before Phase A evidence exists.",
                "The trace arm remains disabled unless a hashed transfer failure triggers it.",
            ],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        payload["plan_sha256"] = hashlib.sha256(encoded).hexdigest()
        return payload


def build_plan(
    *,
    tokenizer_sha256: str | None = None,
    corpus_manifest_sha256: str | None = None,
    generator_sha256: str | None = None,
    e2_winner_sha256: str | None = None,
    model_constructor_sha256: str | None = None,
    raw_byte_budget: int | None = None,
    selected_mixture: str | None = None,
    selected_neighbor: str | None = None,
    trace_arm_enabled: bool = False,
    trace_trigger_receipt_sha256: str | None = None,
) -> E3Plan:
    mixture_arms = tuple(
        MixtureArm(
            name=f"cognition-{int(fraction * 100):02d}-ce",
            cognition_fraction=fraction,
            mixture=_mixture(fraction),
        )
        for fraction in COGNITION_FRACTIONS
    )
    objective_arms = tuple(
        ObjectiveArm(
            name="ce-control" if value == 0 else f"query-swap-{value:.2f}",
            query_swap_lambda=value,
        )
        for value in QUERY_SWAP_LAMBDAS
    )
    plan = E3Plan(
        schema="esoes-e3-plan/v1",
        experiment_id="E3-P35-data-objective-screen-v1",
        tokenizer_sha256=tokenizer_sha256,
        corpus_manifest_sha256=corpus_manifest_sha256,
        generator_sha256=generator_sha256,
        e2_winner_sha256=e2_winner_sha256,
        model_constructor_sha256=model_constructor_sha256,
        raw_byte_budget=raw_byte_budget,
        screen_tokens=SCREEN_TOKENS,
        evaluation_boundaries=EVALUATION_BOUNDARIES,
        mixture_arms=mixture_arms,
        objective_arms=objective_arms,
        selected_mixture=selected_mixture,
        selected_neighbor=selected_neighbor,
        trace_arm_enabled=trace_arm_enabled,
        trace_trigger_receipt_sha256=trace_trigger_receipt_sha256,
    )
    plan.assert_valid()
    return plan


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer-sha256")
    parser.add_argument("--corpus-manifest-sha256")
    parser.add_argument("--generator-sha256")
    parser.add_argument("--e2-winner-sha256")
    parser.add_argument("--model-constructor-sha256")
    parser.add_argument("--raw-byte-budget", type=int)
    parser.add_argument("--selected-mixture")
    parser.add_argument("--selected-neighbor")
    args = parser.parse_args()
    plan = build_plan(
        tokenizer_sha256=args.tokenizer_sha256,
        corpus_manifest_sha256=args.corpus_manifest_sha256,
        generator_sha256=args.generator_sha256,
        e2_winner_sha256=args.e2_winner_sha256,
        model_constructor_sha256=args.model_constructor_sha256,
        raw_byte_budget=args.raw_byte_budget,
        selected_mixture=args.selected_mixture,
        selected_neighbor=args.selected_neighbor,
    )
    payload = plan.as_dict()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
