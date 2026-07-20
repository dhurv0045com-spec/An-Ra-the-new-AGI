"""Create the truthful signed launch manifest after a cloud worker is ready."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from training.launch_manifest import build_launch_manifest, sign_manifest


def create_cloud_launch(
    *,
    pack_root: Path,
    output: Path,
    artifact_path: str,
    checkpoint_source: str,
    worker_id: str,
    runtime_estimate_hours: float,
) -> dict[str, object]:
    pack_root = pack_root.resolve()
    pack = json.loads((pack_root / "pack_manifest.json").read_text(encoding="utf-8"))
    tokenizer = pack_root / str(pack["tokenizer_path"])
    train_manifest = pack_root / str(pack["train_manifest"])
    validation_manifest = pack_root / str(pack["validation_manifest"])
    tokenizer_hash = hashlib.sha256(tokenizer.read_bytes()).hexdigest()
    if tokenizer_hash != str(pack["tokenizer_sha256"]):
        raise ValueError("cloud pack tokenizer hash mismatch")
    manifest = build_launch_manifest(
        model_profile="anra-v4-180m",
        extension_profile="none",
        tokenizer_hash=tokenizer_hash,
        tokenizer_path=str(tokenizer),
        data_manifests=[str(train_manifest), str(validation_manifest)],
        data_manifest_roles={
            str(train_manifest): "train",
            str(validation_manifest): "validation",
        },
        stage="baseline_170m",
        optimizer="adamw",
        batch_size=1,
        accumulation=32,
        schedule={
            "kind": "cosine_with_warmup",
            "warmup_fraction": 0.02,
            "min_lr": 1e-5,
        },
        seeds=[int(pack["seed"])],
        checkpoint_source=checkpoint_source,
        expected_tokens=int(pack["training_tokens_requested"]),
        runtime_estimate_hours=float(runtime_estimate_hours),
        owner_authorized=True,
        worker_id=worker_id,
        worker_role="v4_dense_170m_baseline",
        artifact_path=artifact_path,
        shard_assignment=[0],
        checkpoint_read_only=True,
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
    args = parser.parse_args()
    signed = create_cloud_launch(
        pack_root=args.pack_root,
        output=args.output,
        artifact_path=args.artifact_path,
        checkpoint_source=args.checkpoint_source,
        worker_id=args.worker_id,
        runtime_estimate_hours=args.runtime_estimate_hours,
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
