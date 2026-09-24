from __future__ import annotations

import pytest

from v5_training.kaggle_xla_development import (
    SCHEMA,
    DevelopmentConfig,
    _SyntheticTokenizer,
    _rank_receipt_sha256,
    _synthetic_documents,
    aggregate_group_receipts,
    validate_fresh_worker_group_pair,
)
from v5_training.production_entry import frozen_topology, prepare_data
from v5_training.production_sampler import campaign_microstep_plan


def _group(session: int, *, world_size: int = 2) -> list[dict[str, object]]:
    receipts: list[dict[str, object]] = []
    for rank in range(world_size):
        receipt: dict[str, object] = {
            "schema": SCHEMA,
            "status": "PASS",
            "session": session,
            "ordinal": rank,
            "world_size": world_size,
            "source_tree_sha256": "a" * 64,
            "run_id": "test-run",
            "python_version": "3.12.1",
            "torch_version": "2.x",
            "platform": "Linux-test",
            "xla_status": {
                "device_type": "TPU",
                "world_size": world_size,
                "ordinal": rank,
                "versions": {"torch_xla": "test-xla"},
            },
            "model_spec_sha256": "b" * 64,
            "execution_mode": "xla-development",
            "campaign_tokens": 200,
            "updates_executed": session,
            "cumulative_tokens": 100 * session,
            "state_complete": session == 2,
            "termination": "COMPLETE" if session == 2 else "MANUAL_BOUNDARY",
            "resumed": session == 2,
            "resume_equal": None,
            "resume_verification": "DEFERRED_TO_FRESH_WORKER_GROUP",
            "checkpoint_head": ("c" if session == 1 else "d") * 64,
            "data_manifest_sha256": "e" * 64,
            "pack_manifest_sha256": "f" * 64,
        }
        receipt["receipt_sha256"] = _rank_receipt_sha256(receipt)
        receipts.append(receipt)
    return receipts


def test_aggregate_accepts_complete_first_and_resumed_groups() -> None:
    first = aggregate_group_receipts(_group(1), expected_world_size=2, session=1)
    second = aggregate_group_receipts(_group(2), expected_world_size=2, session=2)

    assert first["status"] == second["status"] == "PASS"
    assert first["resumed"] is False and second["resumed"] is True
    assert first["state_complete"] is False and second["state_complete"] is True
    assert first["checkpoint_head"] != second["checkpoint_head"]
    assert first["fresh_worker_group_resume_passed"] is False
    assert second["fresh_worker_group_resume_passed"] is True
    assert second["runtime_identity"]["xla_status"]["device_type"] == "TPU"
    assert first["research_training_authorized"] is False
    validate_fresh_worker_group_pair(first, second)


def test_fresh_worker_group_pair_rejects_runtime_or_data_drift() -> None:
    first = aggregate_group_receipts(_group(1), expected_world_size=2, session=1)
    second = aggregate_group_receipts(_group(2), expected_world_size=2, session=2)

    changed_runtime = {**second, "runtime_identity": {"platform": "different TPU runtime"}}
    with pytest.raises(ValueError, match="runtime_identity"):
        validate_fresh_worker_group_pair(first, changed_runtime)

    changed_source = {**second, "source_tree_sha256": "9" * 64}
    with pytest.raises(ValueError, match="source_tree_sha256"):
        validate_fresh_worker_group_pair(first, changed_source)


def test_aggregate_rejects_missing_rank_or_divergent_checkpoint() -> None:
    with pytest.raises(ValueError, match="cover"):
        aggregate_group_receipts(_group(1)[:1], expected_world_size=2, session=1)

    receipts = _group(1)
    receipts[1]["checkpoint_head"] = "9" * 64
    with pytest.raises(ValueError, match="receipt hash"):
        aggregate_group_receipts(receipts, expected_world_size=2, session=1)

    receipts[1]["receipt_sha256"] = _rank_receipt_sha256(receipts[1])
    with pytest.raises(ValueError, match="checkpoint_head"):
        aggregate_group_receipts(receipts, expected_world_size=2, session=1)

    with pytest.raises(ValueError, match="session"):
        aggregate_group_receipts(_group(1), expected_world_size=2, session=2)


def test_development_profile_requires_bound_identity_and_uses_fixed_tokenizer() -> None:
    config = DevelopmentConfig(
        source_tree_sha256="a" * 64,
        output_dir="out",
        run_id="test-run",
        cymek_sha="b" * 40,
        identity_kind="source-tree-derived-development-id; not a git commit",
        campaign_tokens=200,
    )
    config.validate()

    tokenizer = _SyntheticTokenizer()
    assert tokenizer.encode("alpha beta theta") == [4, 5, 11]
    assert len(tokenizer.identity.artifact_sha256) == 64
    with pytest.raises(ValueError, match="campaign identity"):
        DevelopmentConfig(
            source_tree_sha256="a" * 64,
            output_dir="out",
            run_id="test-run",
            cymek_sha="not-a-commit",
            identity_kind="invalid",
        ).validate()


def test_synthetic_campaign_data_supplies_every_frozen_bucket_lane() -> None:
    topology = frozen_topology()
    config = DevelopmentConfig(
        source_tree_sha256="a" * 64,
        output_dir="out",
        run_id="bucket-supply-test",
        cymek_sha="b" * 40,
        identity_kind="test",
        campaign_tokens=2 * topology["global_tokens_per_update"],
    )
    documents = _synthetic_documents(config, topology)
    data = prepare_data(
        documents=documents,
        tokenizer=_SyntheticTokenizer(),
        run_id=config.run_id,
        seed=config.seed,
        development_mode=True,
    )
    supply: dict[int, int] = {}
    for shard in data["packed"]:
        supply[shard.bucket] = supply.get(shard.bucket, 0) + shard.real_tokens
    demand: dict[int, int] = {}
    for bucket, tokens in campaign_microstep_plan(
        start_tokens=0, campaign_tokens=config.campaign_tokens, topo=topology,
    ):
        demand[bucket] = demand.get(bucket, 0) + tokens

    assert set(demand) == {512, 1024, 2048, 4096}
    assert all(supply.get(bucket, 0) >= tokens for bucket, tokens in demand.items())
    assert all(
        document["authorization_category"] == "first-party-development-only"
        for document in documents
    )
