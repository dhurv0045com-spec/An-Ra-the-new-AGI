"""Install one verified high-quality SFT bundle into the shared training home."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

_FILES = (
    "sft-v4-train.jsonl",
    "sft-v4-train.manifest.json",
    "sft-v4-validation.jsonl",
    "sft-v4-validation.manifest.json",
    "sft-v4-test.jsonl",
    "sft-v4-test.manifest.json",
    "sft-v4-source-receipts.json",
    "sft-v4-quality-audit.json",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def install(bundle: str | Path, training_home: str | Path) -> dict[str, object]:
    source = Path(bundle).resolve()
    destination = Path(training_home).resolve() / "sft-v4"
    missing = [name for name in _FILES if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError(f"SFT bundle is incomplete: {missing}")
    audit = json.loads((source / "sft-v4-quality-audit.json").read_text(encoding="utf-8"))
    if audit.get("schema") != "anra-sft-quality-audit/v2":
        raise ValueError("unsupported SFT quality audit")
    if int(audit.get("selected_examples", 0)) < 1_000:
        raise ValueError("SFT bundle is too small for the prepared canonical pilot")
    for split in ("train", "validation", "test"):
        manifest_path = source / f"sft-v4-{split}.manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        artifact = source / str(manifest["artifacts"][0]["path"])
        if _sha256(artifact) != str(manifest["artifacts"][0]["sha256"]):
            raise ValueError(f"SFT {split} artifact hash mismatch")

    destination.mkdir(parents=True, exist_ok=True)
    installed: dict[str, str] = {}
    for name in _FILES:
        target = destination / name
        temporary = destination / f".{name}.{os.getpid()}.tmp"
        shutil.copyfile(source / name, temporary)
        os.replace(temporary, target)
        installed[name] = _sha256(target)
    return {"sft_home": str(destination), "installed": installed}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--training-home", required=True)
    args = parser.parse_args()
    print(json.dumps(install(args.bundle, args.training_home), indent=2))


if __name__ == "__main__":
    main()
