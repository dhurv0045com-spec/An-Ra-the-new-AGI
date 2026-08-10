"""Strict local selection for the one checkpoint used by the SFT prototype.

The Colab vault can contain training history, manifests, and archive copies.
The local prototype must never scan or stage that collection.  It accepts one
explicit full-resume SFT artifact and keeps a small path-only preference file
for future launches.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from anra.anra_paths import ROOT, STATE_DIR

CURRENT_SFT_CHECKPOINT_NAME = "anra-v4-current-full-resume.pt"
_PREFERENCE_PATH = STATE_DIR / "sft_prototype_checkpoint.json"


@dataclass(frozen=True)
class LocalSFTCheckpoint:
    """One locally accessible SFT source artifact, never a checkpoint set."""

    path: Path
    source: str


def _configured_path() -> Path | None:
    for variable in ("ANRA_SFT_CHECKPOINT", "ANRA_CHECKPOINT_PATH"):
        value = os.environ.get(variable, "").strip()
        if value:
            return Path(value).expanduser()
    try:
        payload = json.loads(_PREFERENCE_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    value = payload.get("checkpoint") if isinstance(payload, dict) else None
    return Path(str(value)).expanduser() if value else None


def resolve_local_sft_checkpoint(value: str | Path | None = None) -> LocalSFTCheckpoint:
    """Resolve exactly one full SFT checkpoint without broad filesystem scans."""

    explicit = Path(value).expanduser() if value else _configured_path()
    candidates = [
        explicit,
        ROOT / "checkpoints" / CURRENT_SFT_CHECKPOINT_NAME,
        ROOT / "state" / "sft-v4" / CURRENT_SFT_CHECKPOINT_NAME,
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        resolved = candidate.resolve()
        if resolved.is_file():
            if resolved.name != CURRENT_SFT_CHECKPOINT_NAME:
                raise ValueError(
                    "The local prototype accepts only the protected current SFT checkpoint "
                    f"named {CURRENT_SFT_CHECKPOINT_NAME}, not {resolved.name!r}."
                )
            return LocalSFTCheckpoint(
                path=resolved,
                source="explicit" if candidate == explicit else "standard_local_location",
            )
    raise FileNotFoundError(
        "The finished SFT checkpoint is not on this computer. Download or sync only "
        f"{CURRENT_SFT_CHECKPOINT_NAME} from the shared sft-v4 Drive folder, then run "
        "start_sft_prototype.ps1 -Checkpoint <that-file>."
    )


def remember_local_sft_checkpoint(value: str | Path) -> LocalSFTCheckpoint:
    """Store only a local path preference; never copy or upload the checkpoint."""

    resolved = resolve_local_sft_checkpoint(value)
    _PREFERENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = _PREFERENCE_PATH.with_suffix(".tmp")
    try:
        temporary.write_text(
            json.dumps({"checkpoint": str(resolved.path)}, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, _PREFERENCE_PATH)
    finally:
        temporary.unlink(missing_ok=True)
    return resolved
