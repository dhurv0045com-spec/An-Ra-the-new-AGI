from __future__ import annotations

import torch

from training.anra_optimizer import build_optimizer_with_report, candidate_report
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


def test_optimizer_bakeoff_report_path_is_registered() -> None:
    assert v2_report_path("optimizer_bakeoff").name == "v2_optimizer_bakeoff_report.json"
