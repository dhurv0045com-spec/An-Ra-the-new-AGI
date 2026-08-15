from __future__ import annotations

import torch

from scripts.profile_checkpoint_pathologies import (
    _ActivationAccumulator,
    _normalized_entropy,
    profile_checkpoint,
)
from runtime.experience_ledger import content_hash


def test_checkpoint_pathology_profile_detects_dormant_router_context(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "checkpoint_schema_version": 4,
            "global_step": 7,
            "best_loss": 0.5,
            "model": {
                "token_embedding_table.weight": torch.ones(8, 4),
                "residual_depth_logits": torch.zeros(2),
                "dstp_temperature_log": torch.zeros(2),
                "rim_modules.0.raw_alpha": torch.zeros(()),
                "mod_routers.0.context_weights": torch.zeros(3),
                "mod_routers.0.capacity_control": torch.zeros(()),
                "mod_routers.0.gate.weight": torch.ones(1, 4),
            },
        },
        checkpoint,
    )
    report = profile_checkpoint(checkpoint)

    assert report["passed_numerical_integrity"] is True
    assert report["router_context_weights_all_zero"] is True
    assert report["alerts"][0]["code"] == "router_context_dormant"  # type: ignore[index]
    unsigned = {key: value for key, value in report.items() if key != "report_hash"}
    assert report["report_hash"] == content_hash(unsigned)


def test_checkpoint_pathology_profile_rejects_nonfinite_weights(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(
        {"model": {"token_embedding_table.weight": torch.tensor([[float("nan")]])}},
        checkpoint,
    )
    report = profile_checkpoint(checkpoint)

    assert report["passed_numerical_integrity"] is False
    assert report["nonfinite_elements"] == 1


def test_activation_accumulator_and_router_entropy_are_finite() -> None:
    accumulator = _ActivationAccumulator()
    accumulator.add("layer", torch.tensor([[[1.0, 2.0], [3.0, float("nan")]]]))
    row = accumulator.report()["layer"]
    assert row["nonfinite"] == 1
    assert row["mean_rms"] > 0
    assert _normalized_entropy(torch.ones(16)) == 1.0
