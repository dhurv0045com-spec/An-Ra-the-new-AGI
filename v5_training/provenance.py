"""Checkpoint provenance binding every training identity in one manifest.

A manifest names the checkpoint SHA only indirectly (the store computes
content hashes); it binds model spec, tokenizer, data manifest, optimizer
step, token ledger, cursor, RNG, topology, precision, parent, and source
commit. This checkpoint later becomes research subject material, so its
identity must survive hostile audit.
"""

from __future__ import annotations

from v5_contracts.lineage import CheckpointManifest

from .state import TrainingState


def cursor_identity(state: TrainingState) -> str:
    """Render the sampler cursor as a stable identity string."""

    cursor = state.cursor
    return (
        f"{cursor.pack_manifest_sha256}"
        f":{cursor.shard_ordinal}:{cursor.sequence_ordinal}:{cursor.token_offset}"
    )


def build_manifest(
    state: TrainingState,
    *,
    lineage_id: str,
    checkpoint_id: str,
    parent_checkpoint_sha256: str | None,
    source_commit: str,
    model_spec_sha256: str,
    tokenizer_sha256: str,
    data_manifest_sha256: str,
    parameter_sha256: str,
    rng_state_sha256: str,
    distributed_topology: str,
    precision: str,
) -> CheckpointManifest:
    """Bind a validated training state plus measured evidence into a manifest."""

    state.assert_valid()
    manifest = CheckpointManifest(
        schema="anra-v5-checkpoint/v1",
        lineage_id=lineage_id,
        checkpoint_id=checkpoint_id,
        parent_checkpoint_sha256=parent_checkpoint_sha256,
        source_commit=source_commit,
        model_spec_sha256=model_spec_sha256,
        tokenizer_sha256=tokenizer_sha256,
        data_manifest_sha256=data_manifest_sha256,
        global_update=state.global_update,
        cumulative_tokens=state.cumulative_tokens,
        tokens_by_source=dict(state.tokens_by_source),
        curriculum_phase=state.curriculum_phase,
        sampler_cursor=cursor_identity(state),
        distributed_topology=distributed_topology,
        precision=precision,
        parameter_sha256=parameter_sha256,
        optimizer_step_max=state.optimizer_step_max,
        rng_state_sha256=rng_state_sha256,
    )
    manifest.assert_valid()
    return manifest


__all__ = ["build_manifest", "cursor_identity"]
