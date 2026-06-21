from __future__ import annotations

import torch
import pytest

from training.anra_optimizer import (
    build_optimizer_with_report,
    candidate_report,
    repair_optimizer_param_group_defaults,
)
from training.v2_config import V2_FRONTIER_TRAINING
from training.v2_runtime import v2_report_path


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.token_embedding = torch.nn.Embedding(16, 8)
        self.proj = torch.nn.Linear(8, 8)
        self.norm = torch.nn.LayerNorm(8)

    def forward(self, x):
        return self.proj(self.token_embedding(x))


def test_candidate_report_lists_optimizer_bakeoff_options() -> None:
    report = candidate_report(TinyModel())

    names = {candidate["name"] for candidate in report["candidates"]}
    assert names == {
        "adamw",
        "adam8bit",
        "adafactor",
        "muon",
        "scale",
        "galore",
        "qgalore",
    }
    assert report["trainable_params"] > 0


def test_build_optimizer_selects_adamw_baseline() -> None:
    optimizer, report = build_optimizer_with_report(TinyModel(), optimizer_name="adamw", lr=1e-4)

    assert isinstance(optimizer, torch.optim.AdamW)
    assert report["selected"]["requested"] == "adamw"
    assert report["selected"]["actual"] == "adamw"
    assert report["selected"]["status"] == "active"


def test_unavailable_scale_falls_back_without_claiming_active_scale() -> None:
    optimizer, report = build_optimizer_with_report(TinyModel(), optimizer_name="scale")

    assert isinstance(optimizer, torch.optim.AdamW)
    assert report["selected"]["requested"] == "scale"
    assert report["selected"]["actual"] == "adamw"
    assert report["selected"]["status"] == "fallback"


def test_auto_prefers_memory_light_adafactor_when_muon_unavailable() -> None:
    pytest.importorskip("transformers")

    _optimizer, report = build_optimizer_with_report(TinyModel(), optimizer_name="auto")

    assert report["selected"]["requested"] == "auto"
    assert report["selected"]["actual"] in {"muon", "adafactor"}
    if report["selected"]["actual"] == "adafactor":
        assert "memory-light Adafactor" in report["selected"]["reason"]


def test_adafactor_checkpoint_without_beta1_is_repaired_before_step() -> None:
    transformers = pytest.importorskip("transformers")
    model = TinyModel()
    optimizer = transformers.Adafactor(
        model.parameters(),
        lr=1e-3,
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
    )
    legacy_state = optimizer.state_dict()
    for group in legacy_state["param_groups"]:
        group.pop("beta1", None)
    optimizer.load_state_dict(legacy_state)

    repaired = repair_optimizer_param_group_defaults(optimizer)
    model(torch.tensor([[1, 2, 3]])).sum().backward()
    optimizer.step()

    assert "beta1" in repaired


def test_iterate500_frontier_training_defaults_are_fast_t4_profile() -> None:
    assert V2_FRONTIER_TRAINING.batch_size == 1
    assert V2_FRONTIER_TRAINING.grad_accum_steps == 8
    assert V2_FRONTIER_TRAINING.learning_rate == 4e-4
    assert V2_FRONTIER_TRAINING.warmup_steps == 32


def test_optimizer_bakeoff_report_path_is_registered() -> None:
    assert v2_report_path("optimizer_bakeoff").name == "v2_optimizer_bakeoff_report.json"
