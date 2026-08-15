"""Owner CLI for signing a single-GPU V4 -> canonical same-host DDP migration."""

from __future__ import annotations

import argparse
from pathlib import Path

from training.ddp_bootstrap import (
    SAMPLER_POLICIES,
    create_bootstrap_manifest,
    current_source_commit,
    file_bindings,
)
from training.distributed import canonical_training_ddp_contract


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-checkpoint", required=True)
    parser.add_argument("--child-checkpoint", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--child-lineage-id", required=True)
    parser.add_argument("--data-manifest", action="append", default=[], metavar="ROLE=PATH")
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--accumulation", type=int, required=True)
    parser.add_argument("--visible-device-order", required=True)
    parser.add_argument("--seed", type=int, default=1301)
    parser.add_argument(
        "--sampler-policy",
        choices=sorted(SAMPLER_POLICIES),
        default="preserve_global_cursor_repartition_by_rank_v1",
    )
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    args = parser.parse_args()
    pairs: dict[str, Path] = {}
    for item in args.data_manifest:
        if "=" not in item:
            parser.error("--data-manifest must use ROLE=PATH")
        role, raw_path = item.split("=", 1)
        if not role.strip() or role in pairs:
            parser.error("data manifest roles must be unique and non-empty")
        pairs[role.strip()] = Path(raw_path)
    contract = canonical_training_ddp_contract(
        backend="nccl",
        world_size=args.world_size,
        micro_batch_size_per_rank=args.batch_size,
        gradient_accumulation=args.accumulation,
        visible_device_order=args.visible_device_order,
    )
    manifest = create_bootstrap_manifest(
        parent_checkpoint=args.parent_checkpoint,
        child_checkpoint=args.child_checkpoint,
        output_manifest=args.output_manifest,
        child_lineage_id=args.child_lineage_id,
        target_source_commit=current_source_commit(Path(args.repo_root)),
        target_ddp_contract=contract,
        target_data_bindings=file_bindings(pairs),
        seed=args.seed,
        sampler_policy=args.sampler_policy,
    )
    print(f"Signed DDP bootstrap: {args.output_manifest}")
    print(f"Body SHA-256: {manifest['body_sha256']}")


if __name__ == "__main__":
    main()
