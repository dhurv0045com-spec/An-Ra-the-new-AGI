"""Create the truthful signed launch manifest after a cloud worker is ready."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from training.launch_manifest import build_launch_manifest, sign_manifest
from training.v2_config import (
    ANRA_V4_GROWTH_MODEL_PROFILE,
    CANONICAL_MODEL_PROFILE,
)


def create_cloud_launch(
    *,
    pack_root: Path,
    output: Path,
    artifact_path: str,
    checkpoint_source: str,
    worker_id: str,
    runtime_estimate_hours: float,
    batch_size: int,
    accumulation: int,
    model_profile: str = CANONICAL_MODEL_PROFILE,
    stage: str | None = None,
    growth_manifest: str | None = None,
    growth_parent_checkpoint: str | None = None,
) -> dict[str, object]:
    pack_root = pack_root.resolve()
    pack = json.loads((pack_root / "pack_manifest.json").read_text(encoding="utf-8"))
    active_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    if active_commit != str(pack.get("builder_commit", "")):
        raise RuntimeError(
            "cloud worker checkout does not match the pack builder commit: "
            f"worker={active_commit} pack={pack.get('builder_commit')}"
        )
    tokenizer = pack_root / str(pack["tokenizer_path"])
    tokenizer_metadata = pack_root / str(pack["tokenizer_metadata_path"])
    train_manifest = pack_root / str(pack["train_manifest"])
    validation_manifest = pack_root / str(pack["validation_manifest"])
    tokenizer_hash = hashlib.sha256(tokenizer.read_bytes()).hexdigest()
    if tokenizer_hash != str(pack["tokenizer_sha256"]):
        raise ValueError("cloud pack tokenizer hash mismatch")
    tokenizer_metadata_hash = hashlib.sha256(tokenizer_metadata.read_bytes()).hexdigest()
    if tokenizer_metadata_hash != str(pack["tokenizer_metadata_sha256"]):
        raise ValueError("cloud pack tokenizer metadata hash mismatch")
    window_tokens = int(pack["training_tokens_requested"])
    cumulative_tokens = int(pack.get("cumulative_phase_tokens", window_tokens))
    window_start = max(0, cumulative_tokens - window_tokens)
    is_growth = model_profile == ANRA_V4_GROWTH_MODEL_PROFILE
    is_continuation_window = checkpoint_source.strip().lower() != "scratch"
    if is_growth and (not growth_manifest or not growth_parent_checkpoint):
        raise ValueError(
            "The 500M child launch requires --growth-manifest and "
            "--growth-parent-checkpoint"
        )
    if not is_growth and (growth_manifest or growth_parent_checkpoint):
        raise ValueError("The 181M foundation cannot bind growth artifacts")
    launch_stage = stage or ("growth_alignment" if is_growth else "foundation")
    manifest = build_launch_manifest(
        model_profile=model_profile,
        extension_profile="none",
        tokenizer_hash=tokenizer_hash,
        tokenizer_path=str(tokenizer),
        data_manifests=[str(train_manifest), str(validation_manifest)],
        data_manifest_roles={
            str(train_manifest): "train",
            str(validation_manifest): "validation",
        },
        stage=launch_stage,
        optimizer="adamw",
        batch_size=batch_size,
        accumulation=accumulation,
        schedule={
            "kind": "cosine_with_warmup",
            "warmup_fraction": 0.02,
            "min_lr": 5e-6 if is_growth else 1e-5,
        },
        seeds=[int(pack["seed"])],
        checkpoint_source=checkpoint_source,
        expected_tokens=cumulative_tokens,
        runtime_estimate_hours=float(runtime_estimate_hours),
        owner_authorized=True,
        worker_id=worker_id,
        worker_role="canonical_trainer",
        artifact_path=artifact_path,
        shard_assignment=[0],
        checkpoint_read_only=True,
        # A later pack is a new signed token window in the same corpus
        # lineage.  It must never silently reset the accepted sampler cursor.
        # Each portable pack is a deterministic slice of the immutable corpus.
        # A continuation therefore preserves the global token boundary while
        # resetting only the pack-local permutation cursor.
        allow_data_profile_change=is_continuation_window,
        reset_data_sampler=is_continuation_window,
        token_window={
            "start_token": window_start,
            "end_token": cumulative_tokens,
            "pack_sha256": hashlib.sha256(
                (pack_root / "pack_manifest.json").read_bytes()
            ).hexdigest(),
        },
        artifact_destinations=[
            {
                "kind": "full_resume",
                "uri": artifact_path,
                "required": True,
            }
        ],
        resource_limits={
            "session_budget_minutes": max(60, int(runtime_estimate_hours * 60)),
            "drain_reserve_minutes": 30,
            "checkpoint_steps": 100,
            "checkpoint_minutes": 15,
        },
        growth_manifest=growth_manifest,
        growth_parent_checkpoint=growth_parent_checkpoint,
    )
    return sign_manifest(manifest, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--artifact-path",
        default="output/v2/checkpoints/anra_v4_dense_170m_seed1301.pt",
    )
    parser.add_argument("--checkpoint-source", default="scratch")
    parser.add_argument("--worker-id", default="io-net-worker")
    parser.add_argument("--runtime-estimate-hours", type=float, default=6.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--accumulation", type=int, default=8)
    parser.add_argument(
        "--model-profile",
        choices=(CANONICAL_MODEL_PROFILE, ANRA_V4_GROWTH_MODEL_PROFILE),
        default=CANONICAL_MODEL_PROFILE,
    )
    parser.add_argument(
        "--stage",
        default=None,
        help="Defaults to foundation for 181M and growth_alignment for 500M.",
    )
    parser.add_argument("--growth-manifest", default=None)
    parser.add_argument("--growth-parent-checkpoint", default=None)
    args = parser.parse_args()
    signed = create_cloud_launch(
        pack_root=args.pack_root,
        output=args.output,
        artifact_path=args.artifact_path,
        checkpoint_source=args.checkpoint_source,
        worker_id=args.worker_id,
        runtime_estimate_hours=args.runtime_estimate_hours,
        batch_size=args.batch_size,
        accumulation=args.accumulation,
        model_profile=args.model_profile,
        stage=args.stage,
        growth_manifest=args.growth_manifest,
        growth_parent_checkpoint=args.growth_parent_checkpoint,
    )
    print(
        json.dumps(
            {
                "path": str(args.output),
                "run_id": signed["run_id"],
                "git_commit": signed["git_commit"],
                "hardware": signed["hardware"],
                "expected_tokens": signed["expected_tokens"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
