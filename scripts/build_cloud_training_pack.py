"""Build a compact, immutable V4 foundation pack for a bounded cloud run."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import os
import shutil
from pathlib import Path

import numpy as np
from anra.anra_paths import ROOT
from training.curriculum_sampler import PERMUTATION_SAMPLER_ALGORITHM

PHASE_A_MIX = {
    "fineweb_edu": 11 / 18,
    "permissive_code": 1 / 6,
    "finemath": 2 / 15,
    "science_technical": 4 / 45,
}
DEFAULT_SOURCE = ROOT / "output" / "v2" / "data_manifests" / "native_foundation_v4" / "30gb"
DEFAULT_OUTPUT = ROOT / "output" / "v2" / "cloud_packs" / "v4_phase_a_170m_seed1301"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sign_pack_manifest(path: Path, *, key: str | None = None) -> dict[str, object]:
    signing_key = key or os.environ.get("ANRA_MANIFEST_SIGNING_KEY", "")
    if not signing_key:
        raise PermissionError("ANRA_MANIFEST_SIGNING_KEY is required")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.pop("signature", None)
    payload["signature_algorithm"] = "hmac-sha256"
    unsigned = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["signature"] = hmac.new(
        signing_key.encode(), unsigned, hashlib.sha256
    ).hexdigest()
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)
    return payload


def verify_cloud_pack(root: Path, *, key: str | None = None) -> dict[str, object]:
    manifest_path = root / "pack_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    for item in payload.get("files", []):
        path = root / str(item["path"])
        if not path.is_file():
            failures.append(f"missing:{item['path']}")
        elif path.stat().st_size != int(item["bytes"]):
            failures.append(f"size:{item['path']}")
        elif _sha256(path) != str(item["sha256"]):
            failures.append(f"hash:{item['path']}")
    signing_key = key or os.environ.get("ANRA_MANIFEST_SIGNING_KEY", "")
    signature_valid: bool | None = None
    if payload.get("signature"):
        unsigned_payload = dict(payload)
        signature = str(unsigned_payload.pop("signature"))
        unsigned = json.dumps(
            unsigned_payload, sort_keys=True, separators=(",", ":")
        ).encode()
        expected = hmac.new(signing_key.encode(), unsigned, hashlib.sha256).hexdigest()
        signature_valid = bool(signing_key and hmac.compare_digest(signature, expected))
        if not signature_valid:
            failures.append("signature")
    return {
        "valid": not failures,
        "signature_valid": signature_valid,
        "files_verified": len(payload.get("files", [])),
        "failures": failures,
    }


def finalize_cloud_pack_metadata(root: Path) -> dict[str, object]:
    manifest_path = root / "pack_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    launch_template = {
        "schema_version": 1,
        "pack": payload["name"],
        "model_profile": payload["model_profile"],
        "seed": payload["seed"],
        "expected_tokens": payload["training_tokens_requested"],
        "optimizer": "adamw",
        "batch_size": 1,
        "accumulation": 32,
        "block_size": payload["block_size"],
        "warmup_fraction": 0.02,
        "min_lr": 1e-5,
        "qk_norm": "on",
        "attention": "hybrid",
        "mtp": "off",
        "moe": "off",
        "curriculum": "none",
        "budget_policy": {
            "benchmark_minutes": 10,
            "recommended_gpu": "RTX4090 x1",
            "maximum_spend_usd": 4.0,
            "reserve_usd": 1.0,
            "shutdown_after_artifact_download": True,
        },
        "must_generate_and_sign_on_worker": True,
    }
    (root / "launch_template.json").write_text(
        json.dumps(launch_template, indent=2, sort_keys=True), encoding="utf-8"
    )
    (root / "RUNBOOK.md").write_text(
        "# An-Ra V4 170M cloud run\n\n"
        "This pack contains 170,000,384 usable Phase-A tokens and a separate "
        "10,485,760-token validation split. It uses V4 tokenizer hash binding "
        "and a deterministic global permutation, so the first pass has zero "
        "window repeats.\n\n"
        "1. Check out the repository at the intended clean commit on one CUDA worker.\n"
        "2. Copy this directory to `output/v2/cloud_packs/` without changing files.\n"
        "3. Set `ANRA_MANIFEST_SIGNING_KEY` only in the worker environment.\n"
        "4. Run `python -m scripts.create_cloud_launch --pack-root <pack> "
        "--output output/v2/launch_manifests/v4_170m_scratch.json`.\n"
        "5. Run a ten-minute session with `python -m training.train_unified "
        "--mode session --launch-manifest output/v2/launch_manifests/v4_170m_scratch.json "
        "--prepare_data never --session_minutes 10 --post-session-eval none "
        "--data_path training_data/anra_training.txt`.\n"
        "6. Inspect loss, throughput, VRAM, sampler cursor, unique/repeated windows, "
        "checkpoint hash, and projected cost before continuing.\n"
        "7. Generate a new signed resume manifest naming the benchmark checkpoint "
        "as `--checkpoint-source` and a different artifact path. Continue only "
        "while the hard spending cap can still cover checkpoint download.\n"
        "8. Download checkpoint, run report, launch manifests, and logs, then "
        "terminate the VM.\n",
        encoding="utf-8",
    )
    inventory = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path != manifest_path:
            inventory.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    payload["files"] = inventory
    payload["total_bytes"] = sum(int(item["bytes"]) for item in inventory)
    payload.pop("signature", None)
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    if os.environ.get("ANRA_MANIFEST_SIGNING_KEY"):
        payload = sign_pack_manifest(manifest_path)
    return payload


def allocate_windows(total_windows: int, mix: dict[str, float]) -> dict[str, int]:
    total_mass = sum(float(value) for value in mix.values())
    exact = {
        name: total_windows * float(weight) / total_mass for name, weight in mix.items()
    }
    allocated = {name: int(math.floor(value)) for name, value in exact.items()}
    remaining = total_windows - sum(allocated.values())
    order = sorted(mix, key=lambda name: (exact[name] - allocated[name], name), reverse=True)
    for name in order[:remaining]:
        allocated[name] += 1
    return allocated


def _load_manifest(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("shards"), list):
        raise ValueError(f"invalid token manifest: {path}")
    return payload


def _select_source_shards(
    manifest: dict[str, object], source_class: str
) -> list[dict[str, object]]:
    selected = [
        dict(item)
        for item in manifest["shards"]
        if isinstance(item, dict) and str(item.get("source_class", "")) == source_class
    ]
    if not selected:
        raise ValueError(f"source manifest has no {source_class} shards")
    return selected


def _write_split(
    *,
    source_manifest_path: Path,
    output_dir: Path,
    requested_tokens: int,
    block_size: int,
    seed: int,
    max_windows_per_shard: int,
) -> dict[str, object]:
    source_manifest = _load_manifest(source_manifest_path)
    source_root = source_manifest_path.parent
    total_windows = math.ceil(int(requested_tokens) / int(block_size))
    windows_by_source = allocate_windows(total_windows, PHASE_A_MIX)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_shards: list[dict[str, object]] = []
    selected_source_hashes: dict[str, str] = {}

    for source_class, source_windows in windows_by_source.items():
        candidates = _select_source_shards(source_manifest, source_class)
        chunk_count = math.ceil(source_windows / max_windows_per_shard)
        remaining = source_windows
        for chunk_index in range(chunk_count):
            windows = min(max_windows_per_shard, remaining)
            candidate_index = min(
                len(candidates) - 1,
                int((chunk_index + 0.5) * len(candidates) / chunk_count),
            )
            candidate = candidates[candidate_index]
            source_path = source_root / str(candidate["path"])
            actual_hash = selected_source_hashes.get(str(source_path))
            if actual_hash is None:
                actual_hash = _sha256(source_path)
                if actual_hash != str(candidate.get("sha256", "")):
                    raise ValueError(f"source shard hash mismatch: {source_path}")
                selected_source_hashes[str(source_path)] = actual_hash
            source_array = np.load(source_path, mmap_mode="r", allow_pickle=False)
            required_values = windows * block_size + 1
            available_windows = max(0, (len(source_array) - 1) // block_size)
            if available_windows < windows:
                raise ValueError(
                    f"selected source shard is too small: {source_path} "
                    f"has {available_windows} windows, needs {windows}"
                )
            start_span = available_windows - windows + 1
            start_digest = hashlib.sha256(
                f"{seed}:{source_class}:{chunk_index}:{candidate['sha256']}".encode()
            ).digest()
            start_window = int.from_bytes(start_digest[:8], "big") % start_span
            start = start_window * block_size
            values = np.asarray(
                source_array[start : start + required_values], dtype=np.uint16
            )
            output_name = f"{source_class}-{chunk_index:03d}.npy"
            output_path = output_dir / output_name
            temporary = output_path.with_suffix(".tmp.npy")
            np.save(temporary, values, allow_pickle=False)
            temporary.replace(output_path)
            output_shards.append(
                {
                    "dtype": "uint16",
                    "partial": windows < max_windows_per_shard,
                    "path": output_name,
                    "sha256": _sha256(output_path),
                    "source_class": source_class,
                    "tokens": int(values.size),
                    "training_windows": windows,
                    "source_manifest_sha256": _sha256(source_manifest_path),
                    "source_shard_path": str(candidate["path"]),
                    "source_shard_sha256": str(candidate["sha256"]),
                    "source_window_start": start_window,
                }
            )
            remaining -= windows

    realized = {
        name: count / total_windows for name, count in windows_by_source.items()
    }
    manifest: dict[str, object] = {
        "schema_version": 3,
        "pack_schema_version": 1,
        "tokenizer_schema_version": source_manifest.get("tokenizer_schema_version", 3),
        "tokenizer_version": source_manifest.get("tokenizer_version", "v4-32768"),
        "tokenizer_sha256": source_manifest["tokenizer_sha256"],
        "sampling_policy": PERMUTATION_SAMPLER_ALGORITHM,
        "block_size": block_size,
        "requested_training_tokens": int(requested_tokens),
        "usable_training_tokens": total_windows * block_size,
        "total_training_windows": total_windows,
        "total_tokens": sum(int(item["tokens"]) for item in output_shards),
        "source_class_token_mix": {
            name: count * block_size for name, count in windows_by_source.items()
        },
        "campaign_mix_target": PHASE_A_MIX,
        "campaign_mix_realized": realized,
        "campaign_mix_verified": all(
            abs(realized[name] - PHASE_A_MIX[name]) <= 1 / total_windows
            for name in PHASE_A_MIX
        ),
        "seed": seed,
        "source_manifest": str(source_manifest_path),
        "source_manifest_sha256": _sha256(source_manifest_path),
        "shards": output_shards,
    }
    manifest_path = output_dir / "manifest.json"
    temporary_manifest = manifest_path.with_suffix(".tmp")
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary_manifest.replace(manifest_path)
    return manifest


def build_cloud_pack(
    *,
    source_root: Path = DEFAULT_SOURCE,
    output_root: Path = DEFAULT_OUTPUT,
    training_tokens: int = 170_000_000,
    validation_tokens: int = 10_485_760,
    block_size: int = 2048,
    seed: int = 1301,
) -> dict[str, object]:
    if output_root.exists():
        raise FileExistsError(f"cloud pack publication is immutable: {output_root}")
    train_dir = output_root / "train"
    validation_dir = output_root / "validation"
    train_manifest = _write_split(
        source_manifest_path=source_root / "manifest.json",
        output_dir=train_dir,
        requested_tokens=training_tokens,
        block_size=block_size,
        seed=seed,
        max_windows_per_shard=2048,
    )
    validation_manifest = _write_split(
        source_manifest_path=source_root / "validation" / "manifest.json",
        output_dir=validation_dir,
        requested_tokens=validation_tokens,
        block_size=block_size,
        seed=seed + 1,
        max_windows_per_shard=1024,
    )
    tokenizer = ROOT / "tokenizer" / "tokenizer_v4_32k.json"
    tokenizer_hash = _sha256(tokenizer)
    if tokenizer_hash != str(train_manifest["tokenizer_sha256"]):
        raise ValueError("active V4 tokenizer does not match the source token shards")
    shutil.copy2(tokenizer, output_root / tokenizer.name)

    inventory = []
    for path in sorted(output_root.rglob("*")):
        if path.is_file():
            inventory.append(
                {
                    "path": path.relative_to(output_root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    pack: dict[str, object] = {
        "schema_version": 1,
        "name": "anra-v4-phase-a-170m-seed1301",
        "model_profile": "anra-v4-180m",
        "seed": seed,
        "training_tokens_requested": training_tokens,
        "training_tokens_effective": train_manifest["usable_training_tokens"],
        "validation_tokens": validation_manifest["usable_training_tokens"],
        "block_size": block_size,
        "sampling_policy": PERMUTATION_SAMPLER_ALGORITHM,
        "tokenizer_path": tokenizer.name,
        "tokenizer_sha256": tokenizer_hash,
        "train_manifest": "train/manifest.json",
        "train_manifest_sha256": _sha256(train_dir / "manifest.json"),
        "validation_manifest": "validation/manifest.json",
        "validation_manifest_sha256": _sha256(validation_dir / "manifest.json"),
        "files": inventory,
        "total_bytes": sum(int(item["bytes"]) for item in inventory),
        "launch_manifest_policy": (
            "Generate and sign on the cloud worker after checkout so commit, runtime, "
            "hardware, and worker paths are truthful."
        ),
    }
    signing_key = os.environ.get("ANRA_MANIFEST_SIGNING_KEY", "")
    if signing_key:
        pack["signature_algorithm"] = "hmac-sha256"
        unsigned = json.dumps(pack, sort_keys=True, separators=(",", ":")).encode("utf-8")
        pack["signature"] = hmac.new(
            signing_key.encode("utf-8"), unsigned, hashlib.sha256
        ).hexdigest()
    pack_path = output_root / "pack_manifest.json"
    pack_path.write_text(json.dumps(pack, indent=2, sort_keys=True), encoding="utf-8")
    return finalize_cloud_pack_metadata(output_root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--training-tokens", type=int, default=170_000_000)
    parser.add_argument("--validation-tokens", type=int, default=10_485_760)
    parser.add_argument("--block-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=1301)
    args = parser.parse_args()
    report = build_cloud_pack(
        source_root=args.source_root,
        output_root=args.output_root,
        training_tokens=args.training_tokens,
        validation_tokens=args.validation_tokens,
        block_size=args.block_size,
        seed=args.seed,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
