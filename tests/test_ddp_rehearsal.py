from __future__ import annotations

import copy

import pytest
import torch

from scripts.compare_ddp_rehearsals import compare_rehearsals
from scripts.run_ddp_rehearsal import (
    CHECKPOINT_SCHEMA,
    _training_state_fingerprint,
    _validate_resume_payload,
)
from training.curriculum_sampler import DeterministicPermutationSampler
from training.distributed import DistributedContext, barrier_or_raise


def _contract() -> dict[str, object]:
    return {
        "distributed": {
            "schema": "anra-ddp-contract/v1",
            "backend": "nccl",
            "world_size": 2,
            "micro_batch_size_per_rank": 1,
            "gradient_accumulation": 2,
            "global_sequences_per_step": 4,
            "sampler_partition": "rank_strided_global_position_v1",
            "gradient_reduction": "ddp_mean_v1",
        },
        "seed": 1301,
        "windows": 16,
        "sequence_length": 8,
        "vocab_size": 32,
        "width": 8,
        "learning_rate": 0.001,
        "weight_decay": 0.1,
        "model": "tiny_tied_embedding_dropout_v1",
        "sampler": "counter_based_sha256_v1",
    }


def _payload() -> tuple[dict[str, object], DeterministicPermutationSampler]:
    sampler = DeterministicPermutationSampler(16, num_samples=16, seed=1301)
    consumed = [sampler.index_at(position) for position in range(8)]
    return (
        {
            "schema": CHECKPOINT_SCHEMA,
            "global_step": 2,
            "global_cursor": 8,
            "consumed_indices": consumed,
            "rehearsal_contract": _contract(),
            "distributed_rng_states": {"0": {}, "1": {}},
        },
        sampler,
    )


def test_resume_payload_requires_canonical_prefix_and_optimizer_boundary() -> None:
    payload, sampler = _payload()
    _validate_resume_payload(
        payload, expected_contract=_contract(), base_sampler=sampler
    )

    corrupted = copy.deepcopy(payload)
    corrupted["consumed_indices"][3] = corrupted["consumed_indices"][4]
    with pytest.raises(RuntimeError, match="canonical sampler prefix"):
        _validate_resume_payload(
            corrupted, expected_contract=_contract(), base_sampler=sampler
        )

    partial = copy.deepcopy(payload)
    partial["global_cursor"] = 6
    partial["consumed_indices"] = partial["consumed_indices"][:6]
    with pytest.raises(RuntimeError, match="optimizer-step boundary"):
        _validate_resume_payload(
            partial, expected_contract=_contract(), base_sampler=sampler
        )


def test_resume_payload_binds_dataset_and_topology_contract() -> None:
    payload, sampler = _payload()
    changed = _contract()
    changed["windows"] = 20
    with pytest.raises(RuntimeError, match="topology or lineage"):
        _validate_resume_payload(
            payload, expected_contract=changed, base_sampler=sampler
        )


def test_training_state_fingerprint_is_deterministic_and_sensitive() -> None:
    model = {"weight": torch.tensor([[1.0, 2.0]])}
    optimizer = {"state": {0: {"step": torch.tensor(2), "exp_avg": torch.ones(2)}}}
    kwargs = {
        "global_step": 2,
        "global_cursor": 8,
        "consumed_indices": [1, 3, 5, 7, 0, 2, 4, 6],
        "distributed_rng_states": {"0": {"torch": torch.tensor([1], dtype=torch.uint8)}},
    }
    first = _training_state_fingerprint(model, optimizer, **kwargs)
    second = _training_state_fingerprint(model, optimizer, **kwargs)
    assert first == second
    changed = dict(kwargs)
    changed["global_cursor"] = 9
    assert _training_state_fingerprint(model, optimizer, **changed) != first


def test_comparator_recomputes_and_matches_complete_training_state(tmp_path) -> None:
    model = {"weight": torch.tensor([[1.0, 2.0]])}
    optimizer = {"state": {0: {"step": torch.tensor(2), "exp_avg": torch.ones(2)}}}
    rng = {"0": {"torch": torch.tensor([1], dtype=torch.uint8)}}
    fingerprint = _training_state_fingerprint(
        model,
        optimizer,
        global_step=2,
        global_cursor=8,
        consumed_indices=list(range(8)),
        distributed_rng_states=rng,
    )
    payload = {
        "schema": CHECKPOINT_SCHEMA,
        "model": model,
        "optimizer": optimizer,
        "global_step": 2,
        "global_cursor": 8,
        "consumed_indices": list(range(8)),
        "rehearsal_contract": _contract(),
        "distributed_rng_states": rng,
        "state_fingerprint": fingerprint,
    }
    reference = tmp_path / "reference.pt"
    resumed = tmp_path / "resumed.pt"
    torch.save(payload, reference)
    torch.save(copy.deepcopy(payload), resumed)
    assert compare_rehearsals(reference, resumed)["status"] == "exact_match"

    changed = copy.deepcopy(payload)
    changed["global_cursor"] = 9
    torch.save(changed, resumed)
    with pytest.raises(RuntimeError, match="fingerprint does not verify"):
        compare_rehearsals(reference, resumed)


def test_rank_zero_barrier_envelope_accepts_success_and_propagates_error() -> None:
    context = DistributedContext(False, "none", 0, 0, 1, torch.device("cpu"))
    barrier_or_raise(context)
    with pytest.raises(RuntimeError, match="rank-zero operation failed: disk full"):
        barrier_or_raise(context, primary_error="disk full")
