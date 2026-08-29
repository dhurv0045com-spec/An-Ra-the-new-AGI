"""Fail-closed lineage and Adam-state evidence for An-Ra checkpoints."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from anra_core.checkpoint import load_core_checkpoint


SCHEMA = "anra-checkpoint-lineage-audit/v1"
CHUNK = 4 * 1024 * 1024


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(CHUNK):
            digest.update(block)
    return digest.hexdigest()


def tensor_update(digest: "hashlib._Hash", name: str, tensor: torch.Tensor) -> None:
    value = tensor.detach().cpu().contiguous()
    digest.update(f"{name}\0{tuple(value.shape)}\0{value.dtype}\0".encode())
    raw = value.view(torch.uint8).reshape(-1)
    for start in range(0, raw.numel(), CHUNK):
        digest.update(raw[start:start + CHUNK].numpy().tobytes())
    digest.update(b"\0")


def scalar_step(value) -> int:
    if torch.is_tensor(value):
        return int(value.detach().cpu().item())
    return int(value)


def inspect_checkpoint(path: Path, *, legacy: bool) -> dict:
    model, _metadata, identity = load_core_checkpoint(path, legacy_unverified=legacy)
    del model
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    optimizer = payload.get("optimizer_state_dict") or payload.get("optimizer") or {}
    states = optimizer.get("state") or {}
    avg_hash = hashlib.sha256()
    sq_hash = hashlib.sha256()
    steps = []
    avg_count = 0
    sq_count = 0
    for state_id in sorted(states, key=lambda item: str(item)):
        state = states[state_id]
        if "step" in state:
            steps.append(scalar_step(state["step"]))
        if torch.is_tensor(state.get("exp_avg")):
            tensor_update(avg_hash, str(state_id), state["exp_avg"])
            avg_count += 1
        if torch.is_tensor(state.get("exp_avg_sq")):
            tensor_update(sq_hash, str(state_id), state["exp_avg_sq"])
            sq_count += 1

    trainer = payload.get("trainer_state") or {}
    global_step = int(payload.get("global_step", payload.get("step", 0)))
    trainer_step = int(trainer.get("global_step", global_step))
    optimizer_step = max(steps, default=0)
    pack_step = trainer.get("pack_step")
    tokens_per_step = trainer.get("tokens_per_optimizer_step")
    exact_pack_tokens = (
        int(pack_step) * int(tokens_per_step)
        if pack_step is not None and tokens_per_step is not None else None
    )
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "file_sha256": file_sha256(path),
        "parameter_sha256_canonical": identity.parameter_sha256,
        "metadata_parameter_sha256": payload.get("parameter_sha256"),
        "schema_version": payload.get("checkpoint_schema_version"),
        "artifact_class": payload.get("checkpoint_artifact_class"),
        "tokenizer_contract_verified": identity.tokenizer_contract_verified,
        "legacy_forensic_mode": identity.legacy_unverified,
        "global_step": global_step,
        "trainer_step": trainer_step,
        "optimizer_max_step": optimizer_step,
        "optimizer_min_step": min(steps, default=0),
        "optimizer_state_count": len(states),
        "exp_avg_count": avg_count,
        "exp_avg_sq_count": sq_count,
        "exp_avg_sha256": avg_hash.hexdigest(),
        "exp_avg_sq_sha256": sq_hash.hexdigest(),
        "steps_consistent": global_step == trainer_step == optimizer_step,
        "source_checkpoint": payload.get("source_checkpoint"),
        "source_commit": payload.get("source_commit"),
        "pack_manifest_sha256": payload.get("pack_manifest_sha256"),
        "pack_step": pack_step,
        "pack_total_steps": trainer.get("pack_total_steps"),
        "tokens_per_optimizer_step": tokens_per_step,
        "exact_pack_tokens_consumed": exact_pack_tokens,
        "historical_tokens_seen": payload.get("tokens_seen"),
        "metrics": payload.get("metrics"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", required=True)
    parser.add_argument("--intermediate", required=True)
    parser.add_argument("--final", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    checkpoints = {
        "parent": inspect_checkpoint(Path(args.parent), legacy=True),
        "intermediate": inspect_checkpoint(Path(args.intermediate), legacy=False),
        "final": inspect_checkpoint(Path(args.final), legacy=False),
    }
    parent = checkpoints["parent"]
    intermediate = checkpoints["intermediate"]
    final = checkpoints["final"]
    comparisons = {
        "parent_to_intermediate": {
            "parameter_hash_changed": parent["parameter_sha256_canonical"] != intermediate["parameter_sha256_canonical"],
            "exp_avg_changed": parent["exp_avg_sha256"] != intermediate["exp_avg_sha256"],
            "exp_avg_sq_changed": parent["exp_avg_sq_sha256"] != intermediate["exp_avg_sq_sha256"],
            "step_delta": intermediate["global_step"] - parent["global_step"],
        },
        "intermediate_to_final": {
            "parameter_hash_changed": intermediate["parameter_sha256_canonical"] != final["parameter_sha256_canonical"],
            "exp_avg_changed": intermediate["exp_avg_sha256"] != final["exp_avg_sha256"],
            "exp_avg_sq_changed": intermediate["exp_avg_sq_sha256"] != final["exp_avg_sq_sha256"],
            "step_delta": final["global_step"] - intermediate["global_step"],
        },
        "parent_to_final": {
            "parameter_hash_changed": parent["parameter_sha256_canonical"] != final["parameter_sha256_canonical"],
            "exp_avg_changed": parent["exp_avg_sha256"] != final["exp_avg_sha256"],
            "exp_avg_sq_changed": parent["exp_avg_sq_sha256"] != final["exp_avg_sq_sha256"],
            "step_delta": final["global_step"] - parent["global_step"],
            "certified_continuation_tokens": final["exact_pack_tokens_consumed"],
            "total_token_count_certifiable": False,
            "total_token_reason": (
                "The parent records historical_tokens_seen, while the TPU continuation records a new "
                "tokens/optimizer-step contract. Multiplying all 22,517 steps by the TPU contract is invalid."
            ),
        },
    }
    checks = {
        "all_full_resume": all(row["artifact_class"] == "full_resume" for row in checkpoints.values()),
        "all_steps_consistent": all(row["steps_consistent"] for row in checkpoints.values()),
        "final_tokenizer_strict": final["tokenizer_contract_verified"],
        "parent_final_parameters_differ": comparisons["parent_to_final"]["parameter_hash_changed"],
        "parent_final_adam_moments_differ": (
            comparisons["parent_to_final"]["exp_avg_changed"]
            and comparisons["parent_to_final"]["exp_avg_sq_changed"]
        ),
        "final_pack_complete": final["pack_step"] == final["pack_total_steps"],
        "intermediate_step_matches_pack": (
            intermediate["pack_step"] is not None
            and intermediate["global_step"] - parent["global_step"]
            == intermediate["pack_step"]
        ),
        "final_step_matches_pack": (
            final["pack_step"] is not None
            and final["global_step"] - parent["global_step"] == final["pack_step"]
        ),
        "pack_contract_stable": (
            intermediate["pack_manifest_sha256"] == final["pack_manifest_sha256"]
            and intermediate["tokens_per_optimizer_step"]
            == final["tokens_per_optimizer_step"]
        ),
    }
    reality_checks = dict(checks)
    provenance = {
        "continuation_step_token_math_consistent": (
            checks["intermediate_step_matches_pack"]
            and checks["final_step_matches_pack"]
            and checks["final_pack_complete"]
            and checks["pack_contract_stable"]
            and final["exact_pack_tokens_consumed"]
            == final["pack_step"] * final["tokens_per_optimizer_step"]
        ),
        "campaign_lifetime_token_provenance_consistent": False,
        "reason": (
            "The parent records 327,827,071 historical tokens and a 324,550,271→500,000,000 "
            "token window, while the continuation pack is labeled 170M→500M and adds ~330M tokens."
        ),
    }
    report = {
        "schema": SCHEMA,
        "checkpoints": checkpoints,
        "comparisons": comparisons,
        "checks": checks,
        "reality_checks": reality_checks,
        "provenance": provenance,
        "checkpoint_is_real": all(reality_checks.values()),
        "campaign_provenance_consistent": (
            provenance["continuation_step_token_math_consistent"]
            and provenance["campaign_lifetime_token_provenance_consistent"]
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output.resolve()), **checks, **provenance, "checkpoint_is_real": report["checkpoint_is_real"], "campaign_provenance_consistent": report["campaign_provenance_consistent"]}, indent=2))
    if not report["checkpoint_is_real"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
