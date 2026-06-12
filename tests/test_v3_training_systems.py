from __future__ import annotations

import torch

from training.cdr import CorrectedFailure, CorrectedFailureCurriculum
from training.data_pipeline_v3 import ShardedDataPipeline, SourceRecord
from training.data_ledger import DataQuality
from training.pcgrad import project_conflicting_gradient
from training.stages import CampaignState, TrainingStage


def _quality() -> DataQuality:
    return DataQuality(
        difficulty_percentile=0.5,
        novelty=1.0,
        provenance=1.0,
        verification=1.0,
        identity_relevance=0.8,
        license_score=1.0,
    )


def test_pcgrad_removes_negative_conflict() -> None:
    primary = torch.tensor([1.0, 0.0])
    secondary = torch.tensor([-1.0, 1.0])
    projected, telemetry = project_conflicting_gradient(primary, secondary)
    assert telemetry.conflict
    assert torch.dot(projected, secondary) >= -1e-6


def test_data_pipeline_is_hash_reproducible(tmp_path) -> None:
    records = [
        SourceRecord("ordinary example", "unit", "MIT", "owner", _quality()),
        SourceRecord(
            "[GOAL] g [CONSTRAINT] c [HYPOTHESIS] h [ACTION] a "
            "[RESULT] r [VERIFY] v [UPDATE] u",
            "unit",
            "MIT",
            "dfc",
            _quality(),
            "verified",
        ),
    ]
    first = ShardedDataPipeline(tmp_path / "a", tokenizer_version="v3").preprocess(records)
    second = ShardedDataPipeline(tmp_path / "b", tokenizer_version="v3").preprocess(records)
    assert first["shards"][0]["sha256"] == second["shards"][0]["sha256"]


def test_cdr_accepts_only_verified_corrections(tmp_path) -> None:
    curriculum = CorrectedFailureCurriculum(tmp_path / "failures.jsonl")
    curriculum.append(
        CorrectedFailure(
            prompt="p",
            failed_output="bad",
            diagnosis="wrong tool",
            corrected_target="good",
            category="tool_selection",
            verifier="unit",
            verified=True,
        )
    )
    assert len(curriculum.load()) == 1


def test_campaign_stages_resume(tmp_path) -> None:
    path = tmp_path / "campaign.json"
    state = CampaignState(path)
    assert state.next_stage().stage == TrainingStage.FOUNDATION
    state.update(
        TrainingStage.FOUNDATION,
        step=50_000,
        status="complete",
        checkpoint="foundation.pt",
    )
    resumed = CampaignState(path)
    assert resumed.next_stage().stage == TrainingStage.OWNER_ADAPTATION
