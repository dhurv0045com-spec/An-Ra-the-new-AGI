from __future__ import annotations

import json

import pytest

from training.data_pipeline import ShardedDataPipeline, TokenShardPublisher
from training.stages import (
    FOUNDATION_MILESTONES,
    FOUNDATION_STAGES,
    FoundationCampaignState,
    training_progress_report,
)
from training.train_unified import CANONICAL_CAMPAIGNS


@pytest.mark.parametrize("tokenizer_version", ["v3", "v3-16384", "v4-16384", "16k"])
def test_new_data_publications_reject_legacy_tokenizers(
    tmp_path, tokenizer_version: str
) -> None:
    with pytest.raises(ValueError, match="V3 and 16k lineages are retired"):
        ShardedDataPipeline(tmp_path / "records", tokenizer_version=tokenizer_version)
    with pytest.raises(ValueError, match="V3 and 16k lineages are retired"):
        TokenShardPublisher(tmp_path / "tokens", tokenizer_version=tokenizer_version)


def test_new_data_publications_require_schema_four(tmp_path) -> None:
    with pytest.raises(ValueError, match="require tokenizer schema 4"):
        TokenShardPublisher(
            tmp_path / "tokens",
            tokenizer_version="v4-32768",
            tokenizer_schema_version=3,
        )


def test_only_cumulative_dense_v4_campaigns_are_operational() -> None:
    assert CANONICAL_CAMPAIGNS == (
        "v4-foundation",
        "foundation_200m",
        "foundation_500m",
        "foundation_1b",
        "foundation_3_6b",
    )
    assert tuple(stage.token_target for stage in FOUNDATION_STAGES) == FOUNDATION_MILESTONES


def test_legacy_campaign_state_cannot_resume_v4_lineage(tmp_path) -> None:
    path = tmp_path / "campaign.json"
    path.write_text(
        json.dumps({"foundation": {"status": "complete"}}),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="Legacy campaign state"):
        FoundationCampaignState(path)


def test_legacy_rescue_and_stage_labels_are_not_progress_targets() -> None:
    with pytest.raises(ValueError, match="Unknown V4 foundation milestone"):
        training_progress_report(
            milestone="rescue",
            tokens_seen=0,
            tokens_per_second=1_000.0,
        )
