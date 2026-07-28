"""Create the truthful signed launch manifest after a cloud worker is ready."""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runtime.safe_load import safe_torch_load
from training.launch_manifest import build_launch_manifest, sign_manifest
from training.v2_config import (
    ANRA_V4_GROWTH_MODEL_PROFILE,
    CANONICAL_MODEL_PROFILE,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_pack_files(pack_root: Path, pack: dict[str, object]) -> None:
    """Verify every immutable pack member without requiring its builder checkout."""

    files = pack.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("cloud pack must declare a non-empty immutable file inventory")
    for raw in files:
        if not isinstance(raw, dict):
            raise ValueError("cloud pack file inventory entries must be objects")
        relative = Path(str(raw.get("path", "")))
        if not str(relative) or relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe cloud pack member path: {relative}")
        target = (pack_root / relative).resolve()
        if pack_root not in target.parents:
            raise ValueError(f"cloud pack member escapes its root: {relative}")
        if not target.is_file():
            raise FileNotFoundError(f"cloud pack member is missing: {relative}")
        expected_size = int(raw.get("bytes", -1))
        if target.stat().st_size != expected_size:
            raise ValueError(f"cloud pack member size mismatch: {relative}")
        expected_hash = str(raw.get("sha256", "")).lower()
        if len(expected_hash) != 64 or not hmac.compare_digest(_sha256(target), expected_hash):
            raise ValueError(f"cloud pack member hash mismatch: {relative}")


def _materialize_tokenizer_metadata(
    pack_root: Path,
    pack: dict[str, object],
    tokenizer: Path,
    tokenizer_hash: str,
) -> tuple[Path, str, str]:
    """Bind V4 metadata, including for immutable packs built before sidecars."""

    declared_path = str(pack.get("tokenizer_metadata_path", "")).strip()
    if declared_path:
        metadata = pack_root / declared_path
        metadata_hash = _sha256(metadata)
        if not hmac.compare_digest(
            metadata_hash,
            str(pack.get("tokenizer_metadata_sha256", "")).lower(),
        ):
            raise ValueError("cloud pack tokenizer metadata hash mismatch")
        return metadata, metadata_hash, "pack"

    canonical_tokenizer = REPO_ROOT / "tokenizer" / "tokenizer_v4_32k.json"
    canonical_metadata = canonical_tokenizer.with_suffix(
        canonical_tokenizer.suffix + ".meta.json"
    )
    if not canonical_tokenizer.is_file() or not canonical_metadata.is_file():
        raise FileNotFoundError(
            "historical cloud pack requires the canonical V4 tokenizer metadata sidecar"
        )
    if not hmac.compare_digest(_sha256(canonical_tokenizer), tokenizer_hash):
        try:
            canonical_payload = json.loads(canonical_tokenizer.read_text(encoding="utf-8"))
            packed_payload = json.loads(tokenizer.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                "historical cloud pack tokenizer cannot be compared to active V4"
            ) from exc
        if canonical_payload != packed_payload:
            raise ValueError(
                "historical cloud pack tokenizer differs from the active V4 tokenizer"
            )
    metadata = tokenizer.with_suffix(tokenizer.suffix + ".meta.json")
    if metadata.exists() and not hmac.compare_digest(
        _sha256(metadata), _sha256(canonical_metadata)
    ):
        raise ValueError("historical cloud pack has conflicting tokenizer metadata")
    if not metadata.exists():
        shutil.copy2(canonical_metadata, metadata)
    return metadata, _sha256(metadata), "canonical_v4_compatibility_sidecar"


def _continuation_start_token(
    checkpoint_source: str,
    *,
    pack_window_start: int,
    cumulative_tokens: int,
    continuation_phase: str = "A",
) -> int:
    """Bind a continuation window to the exact phase boundary in its parent."""

    source = str(checkpoint_source).strip()
    if not source or source.lower() == "scratch":
        return pack_window_start
    checkpoint = Path(source)
    if not checkpoint.is_absolute():
        checkpoint = (REPO_ROOT / checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"continuation checkpoint is missing: {checkpoint}")
    payload = safe_torch_load(checkpoint, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("continuation checkpoint must contain a mapping")
    counts = payload.get("continuation_token_counts", {})
    if not isinstance(counts, dict):
        raise ValueError("continuation checkpoint has no phase-token accounting")
    phase_tokens = int(counts.get(continuation_phase.upper(), 0))
    if phase_tokens < pack_window_start:
        raise ValueError(
            "continuation checkpoint precedes this data pack: "
            f"checkpoint={phase_tokens:,} pack_start={pack_window_start:,}"
        )
    if phase_tokens >= cumulative_tokens:
        raise ValueError(
            "continuation checkpoint has already consumed this data window: "
            f"checkpoint={phase_tokens:,} pack_end={cumulative_tokens:,}"
        )
    return phase_tokens


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
    if not isinstance(pack, dict):
        raise ValueError("cloud pack manifest must contain an object")
    _verify_pack_files(pack_root, pack)
    active_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    pack_builder_commit = str(pack.get("builder_commit", "")).strip()
    if not pack_builder_commit:
        raise ValueError("cloud pack does not identify its builder commit")
    tokenizer = pack_root / str(pack["tokenizer_path"])
    train_manifest = pack_root / str(pack["train_manifest"])
    validation_manifest = pack_root / str(pack["validation_manifest"])
    tokenizer_hash = hashlib.sha256(tokenizer.read_bytes()).hexdigest()
    if tokenizer_hash != str(pack["tokenizer_sha256"]):
        raise ValueError("cloud pack tokenizer hash mismatch")
    tokenizer_metadata, tokenizer_metadata_hash, tokenizer_metadata_source = (
        _materialize_tokenizer_metadata(pack_root, pack, tokenizer, tokenizer_hash)
    )
    window_tokens = int(pack["training_tokens_requested"])
    cumulative_tokens = int(pack.get("cumulative_phase_tokens", window_tokens))
    pack_window_start = max(0, cumulative_tokens - window_tokens)
    window_start = _continuation_start_token(
        checkpoint_source,
        pack_window_start=pack_window_start,
        cumulative_tokens=cumulative_tokens,
    )
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
    manifest["data_pack_provenance"] = {
        "builder_commit": pack_builder_commit,
        "training_commit": active_commit,
        "manifest_sha256": hashlib.sha256(
            (pack_root / "pack_manifest.json").read_bytes()
        ).hexdigest(),
        "declared_window_start": pack_window_start,
        "resume_window_start": window_start,
        "window_end": cumulative_tokens,
        "tokenizer_metadata_path": str(tokenizer_metadata),
        "tokenizer_metadata_sha256": tokenizer_metadata_hash,
        "tokenizer_metadata_source": tokenizer_metadata_source,
    }
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
