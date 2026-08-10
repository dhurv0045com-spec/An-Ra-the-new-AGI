"""Discover one immutable An-Ra training input mounted by Kaggle."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

CURRENT_CHECKPOINT = "anra-v4-current-full-resume.pt"
CURRENT_METADATA = "anra-v4-current-full-resume.json"
SIGNING_KEY_NAMES = (
    "training-signing-keys.json",
    "anra-v4-recovery-signing-keys.json",
)


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class KaggleTrainingAssets:
    training_home: Path
    checkpoint: Path
    metadata: Path
    checkpoint_sha256: str
    global_step: int
    signing_key: Path | None


def _candidate_homes(input_root: Path) -> list[Path]:
    root = input_root.resolve()
    homes = {
        checkpoint.parent.resolve()
        for checkpoint in root.rglob(CURRENT_CHECKPOINT)
        if checkpoint.is_file() and checkpoint.stat().st_size > 0
    }
    return sorted(homes, key=lambda path: path.as_posix())


def resolve_kaggle_training_assets(
    input_root: str | Path = "/kaggle/input",
) -> KaggleTrainingAssets:
    """Resolve and verify exactly one private Kaggle training-home snapshot."""

    root = Path(input_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Kaggle input root is not mounted: {root}")
    homes = _candidate_homes(root)
    if not homes:
        raise FileNotFoundError(
            "No Kaggle input contains anra-v4-current-full-resume.pt. Add one private "
            "Dataset snapshot of ANRA_T4_TRAINING_HOME to this notebook."
        )
    if len(homes) != 1:
        raise RuntimeError(
            "Expected exactly one canonical training-home snapshot, found: "
            + ", ".join(str(path) for path in homes)
        )
    home = homes[0]
    checkpoint = home / CURRENT_CHECKPOINT
    metadata = home / CURRENT_METADATA
    if not metadata.is_file():
        raise FileNotFoundError(f"Checkpoint metadata is missing: {metadata}")
    pointer = json.loads(metadata.read_text(encoding="utf-8-sig"))
    actual_size = checkpoint.stat().st_size
    actual_hash = sha256_file(checkpoint)
    if int(pointer.get("size_bytes", -1)) != actual_size:
        raise ValueError("Kaggle checkpoint size differs from its canonical metadata")
    if str(pointer.get("sha256", "")).lower() != actual_hash:
        raise ValueError("Kaggle checkpoint SHA-256 differs from its canonical metadata")
    signing_keys = [home / name for name in SIGNING_KEY_NAMES if (home / name).is_file()]
    if len(signing_keys) > 1:
        raise RuntimeError("Kaggle input contains multiple campaign signing-key files")
    return KaggleTrainingAssets(
        training_home=home,
        checkpoint=checkpoint,
        metadata=metadata,
        checkpoint_sha256=actual_hash,
        global_step=int(pointer.get("global_step", pointer.get("step", -1))),
        signing_key=signing_keys[0] if signing_keys else None,
    )
