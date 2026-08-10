"""Compare uninterrupted and interrupted/resumed DDP rehearsal checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from scripts.run_ddp_rehearsal import (
    CHECKPOINT_SCHEMA,
    _training_state_fingerprint,
)


def _load_and_verify(path: Path) -> dict[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema") != CHECKPOINT_SCHEMA:
        raise RuntimeError(f"not an exact-resume DDP rehearsal checkpoint: {path}")
    recorded = str(payload.get("state_fingerprint", ""))
    computed = _training_state_fingerprint(
        payload["model"],
        payload["optimizer"],
        global_step=int(payload["global_step"]),
        global_cursor=int(payload["global_cursor"]),
        consumed_indices=[int(value) for value in payload["consumed_indices"]],
        distributed_rng_states=payload["distributed_rng_states"],
    )
    if recorded != computed:
        raise RuntimeError(f"checkpoint fingerprint does not verify: {path}")
    return payload


def compare_rehearsals(uninterrupted: Path, resumed: Path) -> dict[str, object]:
    reference = _load_and_verify(uninterrupted.resolve())
    candidate = _load_and_verify(resumed.resolve())
    fields = (
        "rehearsal_contract",
        "global_step",
        "global_cursor",
        "consumed_indices",
        "state_fingerprint",
    )
    mismatches = [field for field in fields if reference[field] != candidate[field]]
    if mismatches:
        raise RuntimeError(
            "interrupted/resumed rehearsal differs from uninterrupted reference: "
            + ", ".join(mismatches)
        )
    return {
        "schema": "anra-ddp-rehearsal-comparison/v1",
        "status": "exact_match",
        "global_step": int(reference["global_step"]),
        "global_cursor": int(reference["global_cursor"]),
        "state_fingerprint": str(reference["state_fingerprint"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uninterrupted", type=Path, required=True)
    parser.add_argument("--resumed", type=Path, required=True)
    result = compare_rehearsals(**vars(parser.parse_args()))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
