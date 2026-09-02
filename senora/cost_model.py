"""Analytical Remote Run Budget and Hardware Cost Model for P35.

Computes theoretical computational, storage, and duration bounds for P35 scientific
experiments across LOW (conservative), MEDIAN, and HIGH (optimistic) accelerator
throughput scenarios without pretending estimates are measurements.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from senora.experiment_design import build_p35_cms1_plan
from senora.model import EXPECTED_P35_PARAMETER_COUNT


THROUGHPUT_SCENARIOS_TOKENS_PER_SEC = {
    "conservative_low": 8_000.0,   # Single A100/H100 with conservative batching/unfused kernels
    "median": 15_000.0,            # Single H100 standard PyTorch SDPA execution
    "optimistic_high": 25_000.0,   # Single H100 with optimal tensor core utilization / FlashAttention
}


@dataclass(frozen=True, slots=True)
class ArmCostBreakdown:
    arm_name: str
    phase: str
    parameters: int
    token_budget: int
    updates_count: int
    sequence_length: int
    theoretical_6nd_flops: int
    effective_training_flops: int
    auxiliary_overhead_factor: float
    activation_memory_gb: float
    checkpoint_storage_mb: float
    evaluation_overhead_flops: int
    seed_count: int
    gpu_hours_conservative: float
    gpu_hours_median: float
    gpu_hours_optimistic: float

    def canonical(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ExperimentBudget:
    schema: str
    experiment_id: str
    total_token_budget_per_seed: int
    total_effective_flops_per_seed: int
    total_gpu_hours_conservative: float
    total_gpu_hours_median: float
    total_gpu_hours_optimistic: float
    total_checkpoint_storage_gb: float
    arms: list[ArmCostBreakdown]

    def canonical(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "experiment_id": self.experiment_id,
            "total_token_budget_per_seed": self.total_token_budget_per_seed,
            "total_effective_flops_per_seed": self.total_effective_flops_per_seed,
            "total_gpu_hours_conservative": round(self.total_gpu_hours_conservative, 2),
            "total_gpu_hours_median": round(self.total_gpu_hours_median, 2),
            "total_gpu_hours_optimistic": round(self.total_gpu_hours_optimistic, 2),
            "total_checkpoint_storage_gb": round(self.total_checkpoint_storage_gb, 2),
            "arms": [a.canonical() for a in self.arms],
        }


def compute_arm_cost(
    arm_name: str,
    phase: str,
    token_budget: int,
    parameters: int = EXPECTED_P35_PARAMETER_COUNT,
    batch_tokens: int = 131_072,
    sequence_length: int = 2048,
    is_auxiliary_qswap: bool = False,
    seeds: int = 1,
    checkpoints_count: int = 5,
) -> ArmCostBreakdown:
    # 6ND FLOPs: 6 * N * D
    base_flops = 6 * parameters * token_budget
    overhead = 1.25 if is_auxiliary_qswap else 1.0  # auxiliary contrastive pass overhead
    effective_flops = int(base_flops * overhead)

    updates = token_budget // batch_tokens

    # Memory: Model params (FP32 master + BF16) + Adam moments (FP32 m, v) + activation buffer
    # 35.4M * 4 bytes master + 2 bytes BF16 + 8 bytes Adam = 14 bytes/param ~= 495 MB
    # Activations at batch 64, seq 2048, 16 layers ~= 3.5 GB
    activation_mem = 4.2  # GB

    # Checkpoints: 495 MB per full training checkpoint * count
    ckpt_storage = checkpoints_count * 495.0

    # Eval: 240 cases * 2048 tokens * 2 * N FLOPs ~= 3.5e13 FLOPs (negligible < 0.1% of training)
    eval_flops = 240 * sequence_length * 2 * parameters

    # Hours per seed
    toks_total = token_budget * seeds
    h_cons = (toks_total * overhead) / (THROUGHPUT_SCENARIOS_TOKENS_PER_SEC["conservative_low"] * 3600.0)
    h_med = (toks_total * overhead) / (THROUGHPUT_SCENARIOS_TOKENS_PER_SEC["median"] * 3600.0)
    h_opt = (toks_total * overhead) / (THROUGHPUT_SCENARIOS_TOKENS_PER_SEC["optimistic_high"] * 3600.0)

    return ArmCostBreakdown(
        arm_name=arm_name,
        phase=phase,
        parameters=parameters,
        token_budget=token_budget,
        updates_count=updates,
        sequence_length=sequence_length,
        theoretical_6nd_flops=base_flops,
        effective_training_flops=effective_flops,
        auxiliary_overhead_factor=overhead,
        activation_memory_gb=activation_mem,
        checkpoint_storage_mb=ckpt_storage,
        evaluation_overhead_flops=eval_flops,
        seed_count=seeds,
        gpu_hours_conservative=round(h_cons, 2),
        gpu_hours_median=round(h_med, 2),
        gpu_hours_optimistic=round(h_opt, 2),
    )


def compute_p35_cms1_budget(seeds_per_arm: int = 1) -> ExperimentBudget:
    plan = build_p35_cms1_plan()
    arm_breakdowns = []
    for arm in plan.arms:
        is_qswap = "qswap" in arm["name"]
        breakdown = compute_arm_cost(
            arm_name=arm["name"],
            phase=arm["phase"],
            token_budget=arm["token_budget"],
            is_auxiliary_qswap=is_qswap,
            seeds=seeds_per_arm,
        )
        arm_breakdowns.append(breakdown)

    total_tokens = sum(a.token_budget for a in arm_breakdowns)
    total_effective_flops = sum(a.effective_training_flops for a in arm_breakdowns)
    total_h_cons = sum(a.gpu_hours_conservative for a in arm_breakdowns)
    total_h_med = sum(a.gpu_hours_median for a in arm_breakdowns)
    total_h_opt = sum(a.gpu_hours_optimistic for a in arm_breakdowns)
    total_ckpt_gb = sum(a.checkpoint_storage_mb for a in arm_breakdowns) / 1024.0

    return ExperimentBudget(
        schema="senora-experiment-budget/v1",
        experiment_id=plan.experiment_id,
        total_token_budget_per_seed=total_tokens,
        total_effective_flops_per_seed=total_effective_flops,
        total_gpu_hours_conservative=total_h_cons,
        total_gpu_hours_median=total_h_med,
        total_gpu_hours_optimistic=total_h_opt,
        total_checkpoint_storage_gb=total_ckpt_gb,
        arms=arm_breakdowns,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="P35 analytical hardware run cost model")
    parser.add_argument("--output", type=Path, default=Path("artifacts/v5/p35_cost_model.json"))
    parser.add_argument("--seeds", type=int, default=1, help="Seeds per arm")
    args = parser.parse_args()

    budget = compute_p35_cms1_budget(seeds_per_arm=args.seeds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(budget.canonical(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote analytical run budget to {args.output}")
    print(f"Total Effective FLOPs: {budget.total_effective_flops_per_seed:,}")
    print(f"Estimated GPU-Hours: Conservative={budget.total_gpu_hours_conservative:.2f}h, Median={budget.total_gpu_hours_median:.2f}h, Optimistic={budget.total_gpu_hours_optimistic:.2f}h")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())