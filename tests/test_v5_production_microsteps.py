from __future__ import annotations

import hashlib
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from anra_v5.miniature_run import MINI_SPEC
from v5_data.bucket_cursor import BUCKET_CURSOR_SCHEMA, BucketCursorState, LaneWindow
from v5_training import production_entry
from v5_training.production_entry import run_campaign
from v5_training.production_microsteps import (
    count_eligible_targets,
    count_rank_real_tokens,
    materialize_rank_microsteps,
)
from v5_training.production_resume import next_microstep_fingerprint_sha256
from v5_training.distributed import RankZeroCheckpointCoordinator


def _window(bucket: int, rows: list[list[int]], source: str) -> LaneWindow:
    tokens = tuple(tuple(row) for row in rows)
    segments = tuple(
        tuple([row_index] * len(row)) for row_index, row in enumerate(rows)
    )
    eligible = tuple(
        tuple([True] * len(row)) for row in rows
    )
    return LaneWindow(
        tokens=tokens,
        segment_ids=segments,
        eligible=eligible,
        tokens_by_source={source: sum(map(len, rows))},
        tokens_by_family={},
        real_tokens=sum(map(len, rows)),
        row_widths=tuple(map(len, rows)),
        end_lane_index=0,
        end_token_offset=0,
    )


def test_production_microstep_materialization_is_rank_reversible_and_fingerprinted() -> None:
    windows = [
        (4, "reasoning", "proof", _window(4, [[1, 2], [3, 4, 5, 6], [7], [8, 9, 10]], "source-a")),
        (2, "reasoning", "proof", _window(2, [[11, 12], [13], [14, 15], [16, 17]], "source-b")),
    ]
    planned_total = sum(count_eligible_targets(window) for _, _, _, window in windows)
    cursor = BucketCursorState(
        BUCKET_CURSOR_SCHEMA,
        hashlib.sha256(b"pack").hexdigest(),
        hashlib.sha256(b"lanes").hexdigest(),
        {}, {}, {}, 0, 0,
    )

    by_rank = {
        rank: materialize_rank_microsteps(
            windows, replicas=2, rank=rank, planned_total=planned_total,
        )
        for rank in (0, 1)
    }
    rank_token_counts = {
        rank: sum(count_rank_real_tokens(steps) for steps in (by_rank[rank],))
        for rank in (0, 1)
    }
    rank_eligible_counts = {
        rank: sum(count_eligible_targets(step) for step in by_rank[rank])
        for rank in (0, 1)
    }
    assert sum(rank_token_counts.values()) == sum(
        window.real_tokens for _, _, _, window in windows
    )
    assert sum(rank_eligible_counts.values()) == planned_total
    assert all(count > 0 for count in rank_eligible_counts.values())
    assert len(set(rank_token_counts.values())) == 2
    hashes = [
        next_microstep_fingerprint_sha256(
            rank=rank,
            world_size=2,
            topology="cpu-materializer-world2",
            checkpoint_global_update=3,
            cursor=cursor,
            microsteps=by_rank[rank],
        )
        for rank in (0, 1)
    ]
    assert hashes[0] != hashes[1]

    global_rows = materialize_rank_microsteps(
        windows, replicas=2, rank=None, planned_total=planned_total,
    )
    for index, (bucket, family, subfamily, window) in enumerate(windows):
        left, right = by_rank[0][index], by_rank[1][index]
        full = global_rows[index]
        assert left.bucket == right.bucket == bucket
        assert left.family == right.family == family
        assert left.subfamily == right.subfamily == subfamily
        assert left.planned_total == right.planned_total == planned_total
        assert left.tokens_by_source == right.tokens_by_source == window.tokens_by_source
        assert left.tokens + right.tokens == full.tokens
        assert left.segment_ids + right.segment_ids == full.segment_ids
        assert left.eligible + right.eligible == full.eligible
        assert all(len(row) == bucket for row in full.tokens)
        assert all(len(row) == bucket for row in full.segment_ids)
        assert all(len(row) == bucket for row in full.eligible)
        assert all(
            segment == -1 and not eligible and token == 0
            for token_row, segment_row, eligible_row in zip(
                full.tokens, full.segment_ids, full.eligible,
            )
            for token, segment, eligible in zip(token_row, segment_row, eligible_row)
            if segment == -1
        )


def test_production_microstep_materialization_pads_unshardable_rows_without_losing_targets() -> None:
    window = _window(4, [[1, 2], [3], [4, 5]], "source")
    planned_total = count_eligible_targets(window)
    full = materialize_rank_microsteps(
        [(4, "", "", window)], replicas=2, rank=None,
        planned_total=planned_total,
    )[0]
    ranks = [
        materialize_rank_microsteps(
            [(4, "", "", window)], replicas=2, rank=rank,
            planned_total=planned_total,
        )[0]
        for rank in range(2)
    ]

    # One inert row is appended after the original ordered rows, then the
    # equal contiguous shards can be concatenated to recover the source data.
    assert ranks[0].tokens + ranks[1].tokens == full.tokens + ((0, 0, 0, 0),)
    assert ranks[0].segment_ids + ranks[1].segment_ids == (
        full.segment_ids + ((-1, -1, -1, -1),)
    )
    assert ranks[0].eligible + ranks[1].eligible == (
        full.eligible + ((False, False, False, False),)
    )
    assert sum(count_eligible_targets(step) for step in ranks) == planned_total
    assert sum(count_rank_real_tokens((step,)) for step in ranks) == window.real_tokens


def test_shorter_than_replica_world_tail_materializes_zero_target_ranks() -> None:
    window = _window(4, [[5, 6]], "tail-source")
    planned_total = count_eligible_targets(window)
    rank_steps = [
        materialize_rank_microsteps(
            [(4, "", "", window)], replicas=4, rank=rank,
            planned_total=planned_total,
        )[0]
        for rank in range(4)
    ]

    local_counts = [count_eligible_targets(step) for step in rank_steps]
    assert local_counts == [planned_total, 0, 0, 0]
    assert all(len(step.tokens) == 1 for step in rank_steps)
    assert sum(local_counts) == planned_total
    assert sum(count_rank_real_tokens((step,)) for step in rank_steps) == window.real_tokens
    for step in rank_steps[1:]:
        assert step.tokens == ((0, 0, 0, 0),)
        assert step.segment_ids == ((-1, -1, -1, -1),)
        assert step.eligible == ((False, False, False, False),)


def test_production_microstep_validates_layout_before_device_transfer() -> None:
    window = _window(4, [[5, 6, 7]], "source")
    nonmonotonic = replace(window, segment_ids=((0, 2, 1),))
    with pytest.raises(ValueError, match="segment ids must be nondecreasing"):
        materialize_rank_microsteps(
            [(4, "", "", nonmonotonic)], replicas=1, rank=0, planned_total=1,
        )

    padded_with_real_token = replace(
        window,
        segment_ids=((0, -1, 0),),
        eligible=((True, False, True),),
    )
    with pytest.raises(ValueError, match="padding must use the pad token id"):
        materialize_rank_microsteps(
            [(4, "", "", padded_with_real_token)], replicas=1, rank=0, planned_total=1,
        )


def test_rank_real_token_counter_rejects_malformed_padding() -> None:
    malformed = _window(2, [[5, 6]], "source")
    rank_step = materialize_rank_microsteps(
        [(2, "", "", malformed)], replicas=1, rank=0,
        planned_total=count_eligible_targets(malformed),
    )[0]
    broken = type(rank_step)(
        bucket=rank_step.bucket,
        family=rank_step.family,
        subfamily=rank_step.subfamily,
        tokens=((0, 6),),
        segment_ids=rank_step.segment_ids,
        eligible=rank_step.eligible,
        tokens_by_source=rank_step.tokens_by_source,
        planned_total=rank_step.planned_total,
    )
    with pytest.raises(ValueError, match="real-token segment cannot contain padding"):
        count_rank_real_tokens((broken,))


def test_local_campaign_executes_the_shared_microstep_materializer(tmp_path) -> None:
    class ToyTokenizer:
        identity = SimpleNamespace(artifact_sha256="a" * 64)

        def encode(self, text: str) -> list[int]:
            return [5 + index % 13 for index, _ in enumerate(text.split())]

    class TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.spec = MINI_SPEC
            self.embedding = torch.nn.Embedding(MINI_SPEC.vocabulary_size, 8)

        def forward_hidden(
            self, tokens, _positions, _mask, *, use_activation_checkpointing=False,
        ):
            return self.embedding(tokens)

        def forward(self, tokens, positions, mask, *, use_activation_checkpointing=False):
            hidden = self.forward_hidden(
                tokens, positions, mask,
                use_activation_checkpointing=use_activation_checkpointing,
            )
            return hidden @ self.embedding.weight.transpose(0, 1)

    def build_tiny_model(_spec, _seed):
        return TinyModel()

    with patch.object(production_entry, "initialize", side_effect=build_tiny_model):
        receipt = run_campaign(
            documents=[{
                "doc_id": "toy-document",
                "source_id": "toy-source",
                "text": "word " * 120,
                "family": "natural",
            }],
            tokenizer=ToyTokenizer(),
            model_spec=MINI_SPEC,
            run_id="shared-microstep-local-smoke",
            seed=17,
            store_root=str(tmp_path),
            device=torch.device("cpu"),
            torch_module=torch,
            xb=object(),
            campaign_tokens=100,
            development_mode=True,
            cymek_sha="ab" * 20,
            milestones=(),
            recovery_tokens=10**12,
        )

    assert receipt["termination"] == "COMPLETE"
    assert receipt["cumulative_tokens"] == 100
    assert receipt["microstep_shapes"][0]["execution"] == "local-global"
    assert receipt["microstep_shapes"][0]["executing_rank_rows"] == [1]


def test_campaign_v2_checkpoint_resumes_across_sessions(tmp_path) -> None:
    class ToyTokenizer:
        identity = SimpleNamespace(artifact_sha256="b" * 64)

        def encode(self, text: str) -> list[int]:
            return [5 + index % 13 for index, _ in enumerate(text.split())]

    class TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.spec = MINI_SPEC
            self.embedding = torch.nn.Embedding(MINI_SPEC.vocabulary_size, 8)

        def forward_hidden(
            self, tokens, _positions, _mask, *, use_activation_checkpointing=False,
        ):
            return self.embedding(tokens)

        def forward(self, tokens, positions, mask, *, use_activation_checkpointing=False):
            hidden = self.forward_hidden(
                tokens, positions, mask,
                use_activation_checkpointing=use_activation_checkpointing,
            )
            return hidden @ self.embedding.weight.transpose(0, 1)

    def build_tiny_model(_spec, _seed):
        return TinyModel()

    topology = dict(production_entry.frozen_topology())
    topology.update({
        "replicas": 1,
        "gradient_accumulation_microsteps": 1,
        "tokens_per_replica_microstep": 64,
        "global_tokens_per_microstep": 64,
        "global_tokens_per_update": 64,
        "sequences_per_replica_by_bucket": {1024: 1},
        "supercycle": [1024],
        "recovery_threshold_tokens": 10**12,
        "recovery_generations_retained": 2,
    })

    def single_rank_reduce(_tag, value, reduce_fn):
        return reduce_fn([value])

    def coordinator():
        return RankZeroCheckpointCoordinator(
            rank=0, world_size=1, mesh_reduce=single_rank_reduce,
        )

    with patch.object(production_entry, "initialize", side_effect=build_tiny_model), \
            patch.object(production_entry, "frozen_topology", return_value=topology):
        receipts = []
        for _session in range(3):
            receipts.append(production_entry.run_campaign(
                documents=[{
                    "doc_id": "long-toy-document",
                    "source_id": "toy-source",
                    "text": "word " * 700,
                    "family": "natural",
                }],
                tokenizer=ToyTokenizer(),
                model_spec=MINI_SPEC,
                run_id="campaign-v2-resume-smoke",
                seed=29,
                campaign_tokens=192,
                max_updates=1,
                store_root=str(tmp_path / "campaign-v2"),
                device=torch.device("cpu"),
                torch_module=torch,
                xb=object(),
                development_mode=True,
                cymek_sha="cd" * 20,
                milestones=(),
                recovery_tokens=10**12,
                checkpoint_coordinator=coordinator(),
            ))

    assert [receipt["cumulative_tokens"] for receipt in receipts] == [64, 128, 192]
    assert [receipt["resumed"] for receipt in receipts] == [False, True, True]
    assert all(receipt["resume_equal"] for receipt in receipts)
    assert all(
        receipt["last_update_receipt"]["loss_scope"] == "GLOBAL_BATCH_MEAN"
        and receipt["last_update_receipt"]["loss_denominator"]
        == receipt["last_update_receipt"]["supervised_tokens"]
        and len(receipt["last_update_receipt"]["loss_aggregation_sha256"]) == 64
        for receipt in receipts
    )
    assert receipts[-1]["termination"] == "COMPLETE"
    source_tree_sha256 = receipts[-1]["identity_bundle"]["source_tree_sha256"]
    assert len(source_tree_sha256) == 64
    from v5_training.checkpoint import CheckpointStore
    stored_state, _metadata, _rank_state, _shared = CheckpointStore(
        tmp_path / "campaign-v2", "campaign-v2-resume-smoke",
    ).restore_distributed_artifacts(
        rank=0,
        expected_world_size=1,
        expected_topology=(
            f"signac-v2-world1-{receipts[-1]['topology_sha256']}"
        ),
        checkpoint_sha256=receipts[-1]["checkpoint_head"],
    )
    assert stored_state.identities.source_tree_sha256 == source_tree_sha256
    with patch.object(production_entry, "initialize", side_effect=build_tiny_model), \
            patch.object(production_entry, "frozen_topology", return_value=topology):
        completed_receipt = production_entry.run_campaign(
            documents=[{
                "doc_id": "long-toy-document",
                "source_id": "toy-source",
                "text": "word " * 700,
                "family": "natural",
            }],
            tokenizer=ToyTokenizer(),
            model_spec=MINI_SPEC,
            run_id="campaign-v2-resume-smoke",
            seed=29,
            campaign_tokens=192,
            store_root=str(tmp_path / "campaign-v2"),
            device=torch.device("cpu"),
            torch_module=torch,
            xb=object(),
            development_mode=True,
            cymek_sha="cd" * 20,
            milestones=(),
            recovery_tokens=10**12,
            checkpoint_coordinator=coordinator(),
        )
    assert completed_receipt["already_complete"] is True
    store = production_entry.CheckpointStore(
        tmp_path / "campaign-v2", "campaign-v2-resume-smoke",
    )
    state, metadata, _rank_payloads, shared = store.restore_distributed_artifacts(
        rank=0,
        expected_world_size=1,
        expected_topology=f"signac-v2-world1-{production_entry.topology_sha256(topology)}",
    )
    assert state.cumulative_tokens == metadata.global_tokens == 192
    assert metadata.ranks[0].cumulative_tokens == 192
    assert set(shared) == {"model.bin", "optimizer.bin", "scheduler.json"}
