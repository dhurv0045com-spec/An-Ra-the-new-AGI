from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from training.v2_config import CANONICAL_SPECIAL_TOKEN_IDS
from training.v2_data_mix import RawCausalShardDataset
from training.verified_process import (
    VERIFIED_PROCESS_OBJECTIVE,
    apply_verified_process_weights,
)


def test_only_complete_spans_from_verified_rows_receive_extra_weight() -> None:
    ids = CANONICAL_SPECIAL_TOKEN_IDS
    targets = torch.tensor(
        [
            [ids["<hyp>"], 101, 102, ids["</hyp>"], ids["<verify>"], 103, ids["</verify>"]],
            [ids["<hyp>"], 201, ids["</hyp>"], 202, 203, 204, 205],
            [ids["<upd>"], 301, 302, 303, 304, 305, 306],
        ]
    )
    weights, report = apply_verified_process_weights(
        targets,
        torch.ones_like(targets, dtype=torch.float32),
        verified_rows=torch.tensor([True, False, True]),
        special_token_ids=ids,
        multiplier=1.25,
    )

    assert VERIFIED_PROCESS_OBJECTIVE == "verified_dfc_process_spans_v1"
    assert weights[0].tolist() == [1.0, 1.25, 1.25, 1.0, 1.0, 1.25, 1.0]
    assert torch.equal(weights[1], torch.ones(7))
    assert torch.equal(weights[2], torch.ones(7))
    assert report.eligible_rows == 2
    assert report.rows_with_complete_spans == 1
    assert report.complete_spans == 2
    assert report.weighted_tokens == 3
    assert report.malformed_or_truncated_spans == 1


def test_weighting_contract_rejects_unbounded_multiplier() -> None:
    targets = torch.zeros((1, 4), dtype=torch.long)
    with pytest.raises(ValueError, match="multiplier"):
        apply_verified_process_weights(
            targets,
            torch.ones_like(targets, dtype=torch.float32),
            verified_rows=torch.tensor([True]),
            special_token_ids=CANONICAL_SPECIAL_TOKEN_IDS,
            multiplier=3.0,
        )


def test_raw_dataset_routes_weighting_only_from_verified_dfc(tmp_path) -> None:
    ids = CANONICAL_SPECIAL_TOKEN_IDS
    shard = tmp_path / "verified.npy"
    np.save(
        shard,
        np.asarray(
            [999, ids["<hyp>"], 101, ids["</hyp>"], 102, 103, 104, 105],
            dtype=np.uint16,
        ),
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "tokenizer_sha256": "test",
                "shards": [
                    {
                        "path": shard.name,
                        "tokens": 8,
                        "source_class": "verified_dfc",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    tokenizer = SimpleNamespace(pad_token_id=0, special_ids=ids)
    dataset = RawCausalShardDataset(
        manifest,
        tokenizer,
        block_size=7,
        verify_hashes=False,
        verified_process_multiplier=1.25,
    )

    _, _, weights, _, _ = dataset[0]
    assert weights.tolist() == [1.0, 1.25, 1.0, 1.0, 1.0, 1.0, 1.0]
