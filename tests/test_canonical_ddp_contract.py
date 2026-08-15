from __future__ import annotations

import pytest
import torch

from scripts.build_brain import (
    _canonical_distributed_contract,
    _merge_rank_strided_batches,
    _train_anra_v2,
    _validate_canonical_distributed_resume,
)
from training.distributed import DistributedContext


def _context(*, rank: int = 0, world_size: int = 2) -> DistributedContext:
    return DistributedContext(
        enabled=True,
        backend="nccl",
        rank=rank,
        local_rank=rank,
        world_size=world_size,
        device=torch.device("cpu"),
    )


def test_canonical_ddp_contract_binds_global_batch_and_topology() -> None:
    contract = _canonical_distributed_contract(_context(), batch_size=2, accumulation=4)
    assert contract["world_size"] == 2
    assert contract["micro_batch_size_per_rank"] == 2
    assert contract["gradient_accumulation"] == 4
    assert contract["global_sequences_per_step"] == 16
    assert contract["checkpoint_owner"] == "rank_zero_only"
    assert contract["rng_ownership"] == "every_rank"
    assert contract["same_host"] is True
    assert contract["rank_to_local_rank"] == {"0": 0, "1": 1}

    _validate_canonical_distributed_resume(contract, contract)
    changed = dict(contract)
    changed["world_size"] = 4
    with pytest.raises(RuntimeError, match="topology changed"):
        _validate_canonical_distributed_resume(contract, changed)
    with pytest.raises(RuntimeError, match="single-GPU checkpoints"):
        _validate_canonical_distributed_resume({}, contract)


def test_rank_batches_reconstruct_global_sampler_order() -> None:
    assert _merge_rank_strided_batches([[10, 12], [11, 13]]) == [10, 11, 12, 13]
    with pytest.raises(RuntimeError, match="unequal local microbatches"):
        _merge_rank_strided_batches([[10, 12], [11]])
    with pytest.raises(RuntimeError, match="overlapping"):
        _merge_rank_strided_batches([[10], [10]])


def test_canonical_ddp_fails_before_mutation_for_unsupported_layout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("scripts.build_brain.print_session_dashboard", lambda: None)
    with pytest.raises(RuntimeError, match="structured/conversation layout"):
        _train_anra_v2(
            data_path="unused.txt",
            distributed_context=_context(),
        )


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"model_size": "anra-v4-500m-growth"}, "growth"),
        ({"token_window_id": "window"}, "token-window trimming"),
        ({"max_phase_tokens": 1_000}, "phase-token trimming"),
        ({"continuation_phase": "D"}, "PCGrad continuation"),
        ({"post_session_eval": True}, "post-session evaluation"),
    ],
)
def test_canonical_ddp_rejects_unproven_training_paths_before_cuda(
    monkeypatch: pytest.MonkeyPatch,
    options: dict[str, object],
    message: str,
) -> None:
    from training.v2_data_mix import RawCausalShardDataset

    monkeypatch.setattr("scripts.build_brain.print_session_dashboard", lambda: None)
    with pytest.raises(RuntimeError, match=message):
        _train_anra_v2(
            data_path="unused.txt",
            training_layout=RawCausalShardDataset.PACKING_LAYOUT,
            distributed_context=_context(),
            **options,
        )
