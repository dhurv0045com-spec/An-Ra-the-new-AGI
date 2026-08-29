"""Data, optimization, and compute-center contracts for experiment planning."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .model_spec import ModelSpec, V5A_250M


@dataclass(frozen=True, slots=True)
class DataMixture:
    natural: float
    code_math_formal: float
    verified_cognition: float

    def assert_valid(self) -> None:
        values = (self.natural, self.code_math_formal, self.verified_cognition)
        if any(value < 0 for value in values):
            raise ValueError("mixture fractions cannot be negative")
        if abs(sum(values) - 1.0) > 1e-9:
            raise ValueError("mixture fractions must sum to one")

    def token_allocation(self, total_tokens: int) -> dict[str, int]:
        self.assert_valid()
        natural = round(total_tokens * self.natural)
        code = round(total_tokens * self.code_math_formal)
        cognition = total_tokens - natural - code
        return {
            "natural": natural,
            "code_math_formal": code,
            "verified_cognition": cognition,
        }


@dataclass(frozen=True, slots=True)
class RunSpec:
    schema: str
    token_budget: int
    tokens_per_update: int
    optimizer: str
    beta1: float
    beta2: float
    weight_decay: float
    gradient_clip: float
    precision: str
    schedule_family: str
    peak_learning_rate: float
    mixture: DataMixture

    def assert_valid(self, model: ModelSpec) -> None:
        model.assert_valid()
        self.mixture.assert_valid()
        if self.token_budget <= 0 or self.tokens_per_update <= 0:
            raise ValueError("token counts must be positive")
        if self.token_budget < 15 * model.parameter_receipt().total:
            raise ValueError("center run must budget at least 15 tokens per parameter")
        if not 0 < self.peak_learning_rate < 0.01:
            raise ValueError("peak learning rate is implausible")

    def receipt(self, model: ModelSpec) -> dict[str, object]:
        self.assert_valid(model)
        parameters = model.parameter_receipt().total
        idealized_flops = 6 * parameters * self.token_budget
        return {
            "schema": self.schema,
            "model_sha256": model.sha256(),
            "parameters": parameters,
            "token_budget": self.token_budget,
            "tokens_per_parameter": self.token_budget / parameters,
            "tokens_per_update": self.tokens_per_update,
            "optimizer_updates_floor": self.token_budget // self.tokens_per_update,
            "idealized_6nd_flops": idealized_flops,
            "checkpoint_storage_planning_bytes": {
                "bf16_parameters": 2 * parameters,
                "fp32_master_parameters": 4 * parameters,
                "adam_moments": 8 * parameters,
                "full_resume_without_gradients": 14 * parameters,
                "full_resume_with_bf16_gradients": 16 * parameters,
            },
            "data_tokens": self.mixture.token_allocation(self.token_budget),
            "run_spec": {
                **asdict(self),
                "mixture": asdict(self.mixture),
            },
        }


V5A_RUN_CENTER = RunSpec(
    schema="anra-v5-run-spec/v1",
    token_budget=5_000_000_000,
    tokens_per_update=131_072,
    optimizer="AdamW",
    beta1=0.9,
    beta2=0.95,
    weight_decay=0.1,
    gradient_clip=1.0,
    precision="bf16-compute/fp32-reduction-and-state",
    schedule_family="warmup-stable-decay",
    peak_learning_rate=3e-4,
    mixture=DataMixture(natural=0.65, code_math_formal=0.20, verified_cognition=0.15),
)
