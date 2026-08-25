"""Fail-closed final-checkpoint verification for Kaggle TPU notebooks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from anra_core.checkpoint import load_core_checkpoint


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(checkpoint: Path, parent: Path, expected_step: int) -> dict[str, object]:
    if not checkpoint.is_file():
        raise FileNotFoundError(f"final checkpoint is missing: {checkpoint}")

    parent_model, _parent_payload, parent_identity = load_core_checkpoint(
        parent, legacy_unverified=True
    )
    trained_model, payload, trained_identity = load_core_checkpoint(checkpoint)
    optimizer_states = payload.get("optimizer_state_dict", {}).get("state", {})
    optimizer_steps = [
        int(state["step"])
        for state in optimizer_states.values()
        if isinstance(state, dict) and "step" in state
    ]
    optimizer_step = max(optimizer_steps, default=0)

    failures: list[str] = []
    if trained_identity.artifact_class != "full_resume":
        failures.append(f"artifact class is {trained_identity.artifact_class!r}")
    if payload.get("checkpoint_schema_version") != 3:
        failures.append(f"schema is {payload.get('checkpoint_schema_version')!r}, not 3")
    if payload.get("global_step") != expected_step:
        failures.append(
            f"global step is {payload.get('global_step')!r}, not {expected_step}"
        )
    if optimizer_step != expected_step:
        failures.append(f"Adam step is {optimizer_step}, not {expected_step}")
    if trained_identity.parameter_sha256 == parent_identity.parameter_sha256:
        failures.append("trained parameters are byte-identical to the parent")
    if not payload.get("tokenizer_contract"):
        failures.append("tokenizer contract is missing")
    if failures:
        raise RuntimeError("INVALID FINAL CHECKPOINT: " + "; ".join(failures))

    result: dict[str, object] = {
        "verified": True,
        "checkpoint": str(checkpoint),
        "size_bytes": checkpoint.stat().st_size,
        "file_sha256": _file_sha256(checkpoint),
        "checkpoint_schema_version": 3,
        "artifact_class": "full_resume",
        "global_step": expected_step,
        "optimizer_step": optimizer_step,
        "parameter_sha256": trained_identity.parameter_sha256,
        "parent_parameter_sha256": parent_identity.parameter_sha256,
        "tokenizer_loader_succeeded": True,
    }
    del parent_model, trained_model, payload
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--expected-step", type=int, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = verify(args.checkpoint, args.parent, args.expected_step)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.manifest.with_suffix(args.manifest.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
    temporary.replace(args.manifest)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
