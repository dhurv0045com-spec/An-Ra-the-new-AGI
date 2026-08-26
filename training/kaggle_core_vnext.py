"""Fail-closed Kaggle TPU launch path for core-vnext.

This module owns the notebook-facing orchestration. It deliberately disables
train_xla's sparse candidate path because candidate serialization is not yet
XLA-safe; recovery is provided by the canonical xm.save latest checkpoint.
After training, the checkpoint is reloaded, hashed, copied to /kaggle/working,
and uploaded to an authenticated Kaggle Dataset. The remote manifest is then
downloaded back and verified before success is reported.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle.getmembers():
            target = (destination / member.name).resolve()
            if member.issym() or member.islnk():
                raise RuntimeError(f"archive links are refused: {member.name}")
            if root not in target.parents and target != root:
                raise RuntimeError(f"unsafe archive member: {member.name}")
        bundle.extractall(destination)


def _find_checkpoint(input_root: Path) -> Path:
    name = os.environ.get("ANRA_TPU_CHECKPOINT", "anra-v4-current-full-resume.pt")
    if not name:
        raise RuntimeError("ANRA_TPU_CHECKPOINT must name the exact parent checkpoint")
    matches = [p for p in input_root.rglob("*.pt") if p.name == name]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one {name!r}; found {len(matches)}")
    return matches[0]


def _find_pack(input_root: Path) -> Path:
    campaign_manifests = sorted(input_root.rglob("pack_manifest.json"))
    if len(campaign_manifests) == 1:
        train_root = campaign_manifests[0].parent / "train"
        if not (train_root / "manifest.json").is_file():
            raise RuntimeError("Campaign pack lacks train/manifest.json")
        return train_root
    if len(campaign_manifests) > 1:
        raise RuntimeError(f"Expected one campaign pack; found {len(campaign_manifests)}")
    archives = sorted(input_root.rglob("*.tar.gz"))
    if len(archives) != 1:
        raise RuntimeError(f"Expected one pack archive; found {len(archives)}")
    destination = Path("/kaggle/working/pack")
    _safe_extract(archives[0], destination)
    if not (destination / "manifest.json").is_file():
        raise RuntimeError("Pack archive lacks manifest.json")
    return destination


def _next_run_dir() -> Path:
    root = Path("/kaggle/working/runs")
    root.mkdir(parents=True, exist_ok=True)
    numbers: list[int] = []
    for path in root.glob("run-*"):
        if path.is_dir() and path.name.removeprefix("run-").isdigit():
            numbers.append(int(path.name.removeprefix("run-")))
    return root / f"run-{max(numbers, default=0) + 1:03d}"


def _remote_persist(
    *, export_dir: Path, checkpoint: Path, manifest_path: Path,
    receipt_path: Path, checkpoint_sha: str, parameter_sha: str,
    global_step: int, source_commit: str,
) -> str:
    import kagglehub

    who = kagglehub.whoami()
    username = who.get("username") if isinstance(who, dict) else None
    if not username:
        raise RuntimeError(f"Could not resolve authenticated Kaggle username: {who!r}")
    slug = os.environ.get("ANRA_RECOVERY_DATASET_SLUG", "anra-core-vnext-recovery")
    handle = f"{username}/{slug}"

    stage = Path("/kaggle/working/anra_remote_stage")
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    staged_checkpoint = stage / checkpoint.name
    shutil.copy2(checkpoint, staged_checkpoint)
    shutil.copy2(manifest_path, stage / "EXPORT_MANIFEST.json")
    shutil.copy2(receipt_path, stage / "receipt.json")
    if sha256_file(staged_checkpoint) != checkpoint_sha:
        raise RuntimeError("Remote-stage checkpoint hash mismatch")

    notes = (
        f"core-vnext step={global_step} commit={source_commit[:12]} "
        f"sha256={checkpoint_sha[:16]}"
    )
    kagglehub.dataset_upload(handle, str(stage), version_notes=notes)

    verify_root = Path("/kaggle/working/anra_remote_verify")
    if verify_root.exists():
        shutil.rmtree(verify_root)
    last_error: Exception | None = None
    for _ in range(18):
        try:
            downloaded = kagglehub.dataset_download(
                handle,
                path="EXPORT_MANIFEST.json",
                output_dir=str(verify_root),
                force_download=True,
            )
            candidates = list(verify_root.rglob("EXPORT_MANIFEST.json"))
            if not candidates and downloaded:
                candidate = Path(downloaded)
                if candidate.is_file() and candidate.name == "EXPORT_MANIFEST.json":
                    candidates = [candidate]
            if candidates:
                remote = json.loads(candidates[0].read_text(encoding="utf-8"))
                if remote.get("checkpoint_sha256") != checkpoint_sha:
                    raise RuntimeError("Remote manifest checkpoint SHA mismatch")
                if remote.get("parameter_sha256") != parameter_sha:
                    raise RuntimeError("Remote manifest parameter SHA mismatch")
                return handle
        except Exception as exc:  # dataset version may need processing time
            last_error = exc
        time.sleep(10)
    raise RuntimeError(f"Kaggle Dataset upload could not be verified: {last_error}")


def main() -> None:
    from anra_core.checkpoint import load_core_checkpoint
    from anra_core.config import CANONICAL_CONFIG
    from training.pack_verify import verify_pack
    from training.train_xla import preflight, write_run_receipt

    input_root = Path("/kaggle/input")
    checkpoint = _find_checkpoint(input_root)
    pack_root = _find_pack(input_root)
    pack = verify_pack(pack_root)

    config = {
        "max_steps": 0,
        "max_minutes": 430,
        "batch_size": 1,
        "grad_accum_steps": 8,
        "learning_rate": 2e-4,
        "weight_decay": 0.1,
        "save_interval": 200,
        # Hard safety invariant: do not enter plain torch.save candidate path.
        "candidate_interval": 0,
        "warmup_fraction": 0.0,
        "decay_fraction": 0.1,
        "log_interval": 1,
        "seed": 1301,
    }
    assert config["candidate_interval"] == 0
    available_steps = pack.total_windows // (8 * config["batch_size"] * config["grad_accum_steps"])
    if available_steps <= 0:
        raise RuntimeError("Pack cannot form one complete distributed optimizer update")
    config["_pack_total_steps"] = available_steps

    identity = preflight(
        dataset_path=pack_root,
        checkpoint_path=checkpoint,
        block_size=CANONICAL_CONFIG.block_size,
        vocab_size=CANONICAL_CONFIG.vocab_size,
        expected_resume_step=20_000,
        start_new_pack=True,
        allow_legacy_resume=True,
    )
    run_dir = _next_run_dir()
    receipt = write_run_receipt(run_dir, identity_block=identity, config=config, world_size=8)
    source_commit = os.environ.get("ANRA_SOURCE_COMMIT", "unknown")

    output = run_dir / "anra-v4-tpu-latest.pt"
    command = [
        sys.executable, "-m", "training.train_xla",
        "--dataset-path", str(pack_root),
        "--output-checkpoint", str(output),
        "--resume-from", str(checkpoint),
        "--expected-resume-step", "20000",
        "--allow-legacy-resume", "--start-new-pack", "--no-gradient-checkpointing",
        "--max-steps", "0", "--max-minutes", str(config["max_minutes"]),
        "--batch-size", str(config["batch_size"]),
        "--grad-accum-steps", str(config["grad_accum_steps"]),
        "--learning-rate", str(config["learning_rate"]),
        "--weight-decay", str(config["weight_decay"]),
        "--warmup-fraction", str(config["warmup_fraction"]),
        "--decay-fraction", str(config["decay_fraction"]),
        "--save-interval", str(config["save_interval"]),
        "--candidate-interval", "0",
        "--log-interval", str(config["log_interval"]),
        "--seed", str(config["seed"]),
    ]
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Training exited {result.returncode}; inspect logs")
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError("Training returned success without a non-empty checkpoint")

    parent_model, _, parent_identity = load_core_checkpoint(checkpoint, legacy_unverified=True)
    trained_model, payload, trained_identity = load_core_checkpoint(output)
    assert trained_identity.artifact_class == "full_resume"
    assert payload.get("checkpoint_schema_version") == 3
    assert payload.get("optimizer_state_dict", {}).get("state")
    assert payload.get("trainer_state", {}).get("pack_manifest_sha256") == pack.manifest_sha256
    assert payload.get("lr_schedule", {}).get("name") == "wsd_pack_v1"
    assert trained_identity.parameter_sha256 != parent_identity.parameter_sha256
    optimizer_steps = [int(state["step"]) for state in payload["optimizer_state_dict"]["state"].values()]
    assert optimizer_steps and max(optimizer_steps) == payload["global_step"]
    del parent_model, trained_model

    export_root = Path("/kaggle/working/anra_exports")
    export_root.mkdir(parents=True, exist_ok=True)
    export_dir = export_root / run_dir.name
    if export_dir.exists():
        raise RuntimeError(f"Refusing to overwrite existing export: {export_dir}")
    shutil.copytree(run_dir, export_dir)
    exported = export_dir / output.name
    checkpoint_sha = sha256_file(output)
    if sha256_file(exported) != checkpoint_sha:
        raise RuntimeError("Local export checkpoint SHA mismatch")
    reload_model, reload_payload, reload_identity = load_core_checkpoint(exported)
    assert reload_identity.parameter_sha256 == trained_identity.parameter_sha256
    assert reload_payload.get("global_step") == payload.get("global_step")
    del reload_model

    manifest = {
        "schema": "anra-export/v2",
        "global_step": int(payload["global_step"]),
        "checkpoint_file": exported.name,
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_bytes": exported.stat().st_size,
        "parameter_sha256": trained_identity.parameter_sha256,
        "source_commit": source_commit,
    }
    manifest_path = export_dir / "EXPORT_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    archive_path = Path(shutil.make_archive(
        str(export_root / f"{run_dir.name}-verified"), "zip",
        root_dir=export_dir.parent, base_dir=export_dir.name,
    ))
    if not archive_path.is_file() or archive_path.stat().st_size == 0:
        raise RuntimeError("Verified local archive was not created")

    handle = _remote_persist(
        export_dir=export_dir,
        checkpoint=exported,
        manifest_path=manifest_path,
        receipt_path=receipt,
        checkpoint_sha=checkpoint_sha,
        parameter_sha=trained_identity.parameter_sha256,
        global_step=int(payload["global_step"]),
        source_commit=source_commit,
    )
    print(json.dumps({
        "REMOTE_PERSISTENCE_GATE": "PASSED",
        "dataset_handle": handle,
        "run_dir": str(run_dir),
        "local_archive": str(archive_path),
        "global_step": int(payload["global_step"]),
        "checkpoint_sha256": checkpoint_sha,
        "parameter_sha256": trained_identity.parameter_sha256,
    }, indent=2))


if __name__ == "__main__":
    main()
