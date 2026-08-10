"""Immutable parent-stage and lineage contract for V4 model growth."""

from __future__ import annotations

from collections.abc import Mapping

GROWTH_PARENT_LINEAGE_SCHEMA = "anra-growth-parent-lineage/v1"
GROWTH_PARENT_POLICIES = frozenset({"pretrained-parent", "post-trained-parent"})


def build_growth_parent_lineage(
    payload: Mapping[str, object],
    *,
    checkpoint_sha256: str,
    parent_stage_policy: str,
) -> dict[str, object]:
    """Copy the parent identity without flattening SFT into pretraining state."""

    if parent_stage_policy not in GROWTH_PARENT_POLICIES:
        raise ValueError(f"invalid growth parent-stage policy: {parent_stage_policy!r}")
    sft = payload.get("sft")
    is_post_trained = isinstance(sft, Mapping) and sft.get("stage") == "sft"
    expected_policy = "post-trained-parent" if is_post_trained else "pretrained-parent"
    if parent_stage_policy != expected_policy:
        raise ValueError(
            f"growth policy {parent_stage_policy!r} disagrees with checkpoint stage "
            f"{expected_policy!r}"
        )
    model_config = payload.get("model_config")
    if not isinstance(model_config, Mapping):
        raise ValueError("growth parent has no recorded model_config")
    approved = tuple(model_config.get("approved_subsystems", ()))
    active_flags = {
        name: bool(model_config.get(name, False))
        for name in (
            "use_mtp",
            "use_moe",
            "use_mod",
            "use_rim",
            "use_dstp",
            "use_esv_control",
            "use_residual_depth",
            "use_hal",
        )
    }
    if approved or any(active_flags.values()):
        raise ValueError(
            "registered 181M-to-500M growth currently requires an explicitly dense parent"
        )
    checkpoint_lineage = payload.get("checkpoint_lineage")
    if not isinstance(checkpoint_lineage, Mapping):
        raise ValueError("growth parent has no immutable checkpoint_lineage")
    return {
        "schema": GROWTH_PARENT_LINEAGE_SCHEMA,
        "parent_stage_policy": parent_stage_policy,
        "checkpoint_sha256": str(checkpoint_sha256),
        "checkpoint_schema_version": int(payload.get("checkpoint_schema_version", 0)),
        "checkpoint_artifact_class": str(payload.get("checkpoint_artifact_class", "")),
        "lineage_id": str(payload.get("lineage_id", "")),
        "source_commit": str(payload.get("source_commit", "")),
        "checkpoint_lineage": dict(checkpoint_lineage),
        "sft": dict(sft) if isinstance(sft, Mapping) else None,
        "data_profile": str(payload.get("data_profile", "unknown")),
        "training_data_layout": str(payload.get("training_data_layout", "unknown")),
        "data_manifests": dict(
            payload.get("data_manifests", payload.get("dataset_manifest_hashes", {}))
        ),
        "model_config": dict(model_config),
    }
