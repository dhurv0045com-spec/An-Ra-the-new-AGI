from __future__ import annotations

import torch
from torch import nn

from evaluation.capability_ladder import evaluate_capability_ladder
from robotics.sim_to_real import decide_next_mode
from training.distillation import (
    DistillationExample,
    DistillationSource,
    accept_distillation_example,
)
from training.distributed import estimate_campaign, recommended_profile
from training.qat import attach_qat
from training.sadl import normalized_mix, owner_weight


def test_sadl_frontier_owner_weight_and_floor() -> None:
    assert round(owner_weight(499_167_019), 4) == 0.65
    mix = normalized_mix(499_167_019)
    assert abs(sum(mix.values()) - 1.0) < 1e-9
    assert mix["owner"] >= 0.50


def test_qat_keeps_master_weights_trainable() -> None:
    model = nn.Sequential(nn.Linear(8, 8))
    attached = attach_qat(model)
    assert attached == ["0"]
    output = model(torch.randn(2, 8)).sum()
    output.backward()
    assert model[0].base.weight.grad is not None


def test_distributed_estimates_require_measured_throughput() -> None:
    profile = recommended_profile(world_size=4, bf16_supported=True)
    assert profile.precision == "bf16"
    estimate = estimate_campaign(
        token_target=60_000_000_000,
        measured_tokens_per_second=1000,
        hourly_cost=8,
    )
    assert estimate["hours"] > 0


def test_distillation_checks_output_rights_separately() -> None:
    source = DistillationSource("teacher", "Apache-2.0", False, "MIT", False, True)
    accepted, failures = accept_distillation_example(
        DistillationExample("p", "o", True, 0.8, source)
    )
    assert not accepted
    assert "output_use_not_allowed" in failures


def test_capability_and_sim_to_real_ladders() -> None:
    ladder = evaluate_capability_ladder(
        {
            "novel_problem_solving": 0.8,
            "cross_domain_transfer": 0.8,
            "continual_learning_gain": 0.1,
            "calibration": 0.9,
        }
    )
    assert ladder["adaptive_passed"]
    assert decide_next_mode(
        current_mode="simulation", randomized_sim_success=0.85
    ).allowed
