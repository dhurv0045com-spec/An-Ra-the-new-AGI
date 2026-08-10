"""Build a hash-bound V4 growth initialization from a trained parent.

This command creates model weights for a larger child.  It deliberately does
not create a resumable training checkpoint: the child's AdamW state must start
fresh after parity validation and low-learning-rate stabilization.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path

import torch

from training.csii import CrossScaleIdentityInheritance
from training.growth_contract import GROWTH_PARENT_POLICIES, build_growth_parent_lineage
from training.v2_config import (
    ANRA_V4_GROWTH_MODEL_PROFILE,
    CANONICAL_MODEL_PROFILE,
    model_profile_registration,
)
from training.v2_runtime import build_model_for_profile, load_or_build_v2_tokenizer

DEFAULT_PARITY_PROMPTS = (
    "An-Ra checks that the inherited model produces the same logits.",
    "Reason carefully: if every verified checkpoint has a hash, what does the hash prove?",
    "def triangular(n):\n    return n * (n + 1) // 2",
    "Calculate 37 * 19 and explain the arithmetic in one sentence.",
    "Return a valid JSON object with status, evidence, and confidence fields.",
    "When evidence is missing, explain why uncertainty is more honest than invention.",
    "Rewrite this sentence clearly: The measurements was recorded incorrectly.",
    "A long context must preserve the opening constraint while reasoning through several "
    "intermediate facts. Constraint: answer with the word cobalt only after checking that "
    "every stated premise is internally consistent. Premise one is consistent. Premise two "
    "does not contradict premise one. Premise three asks for the original constraint.",
)


def _checkpoint_model_state(payload: object) -> dict[str, torch.Tensor]:
    if not isinstance(payload, dict):
        raise TypeError("Growth requires a structured V4 parent checkpoint")
    state = payload.get("model_state_dict", payload.get("model"))
    if not isinstance(state, dict) or not state:
        raise ValueError("Parent checkpoint has no model state")
    if not all(
        isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in state.items()
    ):
        raise TypeError("Parent checkpoint model state contains non-tensor entries")
    return state


def _assert_parent_config(payload: object, model: object) -> None:
    if not isinstance(payload, dict) or not isinstance(payload.get("model_config"), dict):
        raise ValueError("Growth requires the parent's recorded model_config")
    saved = payload["model_config"]
    active = model.model_config()
    fields = (
        "architecture_version",
        "vocab_size",
        "n_embd",
        "n_head",
        "n_kv_head",
        "n_layer",
        "d_ff",
        "block_size",
        "rope_base",
        "mod_layers",
        "use_qk_norm",
        "sliding_window",
        "full_attention_every",
        "use_mtp",
        "use_moe",
        "use_mod",
        "use_rim",
        "use_dstp",
        "use_esv_control",
        "use_residual_depth",
        "use_hal",
        "approved_subsystems",
        "initialization_scheme",
    )
    mismatches: dict[str, dict[str, object]] = {}
    for field in fields:
        tuple_fields = {"mod_layers", "approved_subsystems"}
        saved_value = tuple(saved.get(field, ())) if field in tuple_fields else saved.get(field)
        active_value = tuple(active.get(field, ())) if field in tuple_fields else active.get(field)
        if saved_value != active_value:
            mismatches[field] = {"checkpoint": saved_value, "registered": active_value}
    if mismatches:
        raise ValueError(f"Parent checkpoint does not match its registered profile: {mismatches}")


def _load_prompts(path: str | None) -> tuple[str, ...]:
    if path is None:
        return DEFAULT_PARITY_PROMPTS
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if (
        not isinstance(payload, list)
        or not payload
        or not all(isinstance(item, str) and item.strip() for item in payload)
    ):
        raise ValueError("Parity prompts must be a non-empty JSON string array")
    return tuple(item.strip() for item in payload)


def _parity_tokens(
    prompts: tuple[str, ...], *, max_length: int = 256
) -> tuple[torch.Tensor, torch.Tensor]:
    tokenizer = load_or_build_v2_tokenizer()
    encoded = [tokenizer.encode(prompt, add_special_tokens=True)[:max_length] for prompt in prompts]
    if not encoded or any(not row for row in encoded):
        raise ValueError("Parity prompts produced an empty token sequence")
    width = max(len(row) for row in encoded)
    padded = [row + [tokenizer.pad_token_id] * (width - len(row)) for row in encoded]
    mask = [[True] * len(row) + [False] * (width - len(row)) for row in encoded]
    return torch.tensor(padded, dtype=torch.long), torch.tensor(mask, dtype=torch.bool)


def _atomic_torch_save(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json_save(payload: dict[str, object], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def grow_checkpoint(
    *,
    source_checkpoint: str | Path,
    output_checkpoint: str | Path,
    report_path: str | Path,
    source_profile: str = CANONICAL_MODEL_PROFILE,
    target_profile: str = ANRA_V4_GROWTH_MODEL_PROFILE,
    parity_prompts: tuple[str, ...] = DEFAULT_PARITY_PROMPTS,
    minimum_cosine: float = 0.99,
    parent_stage_policy: str,
    device: str = "auto",
    overwrite: bool = False,
) -> dict[str, object]:
    source_path = Path(source_checkpoint).resolve()
    output_path = Path(output_checkpoint).resolve()
    report_target = Path(report_path).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Parent checkpoint does not exist: {source_path}")
    if source_path == output_path:
        raise ValueError("Growth output must not overwrite its parent checkpoint")
    if (output_path.exists() or report_target.exists()) and not overwrite:
        raise FileExistsError("Growth output/report already exists; pass --overwrite explicitly")

    source_registration = model_profile_registration(source_profile)
    target_registration = model_profile_registration(target_profile)
    if source_registration.name != CANONICAL_MODEL_PROFILE:
        raise ValueError(
            "The registered V4 growth path currently requires the canonical 181M parent"
        )
    if (
        target_registration.parent_profile != source_registration.name
        or not target_registration.requires_growth_manifest
        or target_registration.scratch_training_allowed
    ):
        raise ValueError("Target profile is not a registered child of the requested parent")

    source = build_model_for_profile(source_profile)
    # Growth is an operational V4 path, not a legacy checkpoint-inspection
    # path.  It therefore fails closed instead of permitting arbitrary pickle
    # execution through a weights_only=False compatibility fallback.
    payload = torch.load(source_path, map_location="cpu", weights_only=True)
    _assert_parent_config(payload, source)
    parent_lineage = build_growth_parent_lineage(
        payload,
        checkpoint_sha256=_sha256_file(source_path),
        parent_stage_policy=parent_stage_policy,
    )
    state = _checkpoint_model_state(payload)
    parent_progress = {
        "tokens_seen": int(payload.get("tokens_seen", 0)),
        "continuation_token_counts": dict(payload.get("continuation_token_counts", {})),
        "raw_window_consumption": dict(payload.get("raw_window_consumption", {})),
        "data_sampler_state": dict(payload.get("data_sampler_state", {})),
        "data_profile": str(payload.get("data_profile", "unknown")),
        "training_data_layout": str(payload.get("training_data_layout", "unknown")),
        "seed_contract": dict(payload.get("seed_contract", {})),
        "data_manifests": dict(
            payload.get("data_manifests", payload.get("dataset_manifest_hashes", {}))
        ),
        "best_validation_loss": float(payload.get("best_validation_loss", float("inf"))),
        "best_answer_validation_loss": float(
            payload.get("best_answer_validation_loss", float("inf"))
        ),
        "validation_history": list(payload.get("validation_history", [])),
    }
    incompatible = source.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError(f"Parent checkpoint did not load exactly: {incompatible}")
    del state, payload
    gc.collect()

    target = build_model_for_profile(target_profile, allow_experimental=True)
    report = CrossScaleIdentityInheritance.grow(
        source,
        target,
        source_checkpoint=source_path,
        source_profile=source_profile,
        target_profile=target_profile,
    )
    parity_ids, parity_mask = _parity_tokens(parity_prompts)
    resolved_device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device == "auto"
        else torch.device(device)
    )
    source.to(resolved_device)
    target.to(resolved_device)
    parity = CrossScaleIdentityInheritance.verify_parity(
        source,
        target,
        parity_ids.to(resolved_device),
        valid_token_mask=parity_mask.to(resolved_device),
    )
    report = CrossScaleIdentityInheritance.bind_parity(
        report,
        parity,
        minimum_cosine=minimum_cosine,
    )
    CrossScaleIdentityInheritance.write_report(report, report_target)
    if report.parity_passed is not True:
        raise RuntimeError(
            "Growth child failed real-logits parity; no child artifact was published: "
            f"cosine={report.parity_cosine:.8f}, KL={report.parity_mean_kl:.8f}, "
            f"top1={report.parity_top1_agreement:.4f}"
        )
    manifest = CrossScaleIdentityInheritance.validate_growth_report(report)
    target.to("cpu")
    artifact = {
        "artifact_schema_version": 1,
        "artifact_class": "growth_initialization",
        "training_resume_allowed": False,
        "optimizer_restart_required": True,
        "optimizer_state_inherited": False,
        "model_profile": target_profile,
        "parent_model_profile": source_profile,
        "model": target.state_dict(),
        "model_config": target.model_config(),
        "growth_manifest": manifest,
        "parent_progress": parent_progress,
        "parent_lineage": parent_lineage,
    }
    _atomic_torch_save(artifact, output_path)
    artifact_sha256 = _sha256_file(output_path)
    report_sha256 = _sha256_file(report_target)
    metadata_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    _atomic_json_save(
        {
            "schema": "anra-growth-initialization/v1",
            "artifact_class": "growth_initialization",
            "artifact_sha256": artifact_sha256,
            "growth_manifest_sha256": report_sha256,
            "source_checkpoint_sha256": str(manifest["source_checkpoint_sha256"]),
            "source_profile": source_profile,
            "target_profile": target_profile,
            "optimizer_restart_required": True,
            "optimizer_state_inherited": False,
            "training_resume_allowed": False,
        },
        metadata_path,
    )
    return {
        "output_checkpoint": str(output_path),
        "output_checkpoint_sha256": artifact_sha256,
        "output_metadata": str(metadata_path),
        "report": str(report_target),
        "report_sha256": report_sha256,
        "source_profile": source_profile,
        "target_profile": target_profile,
        "parity_cosine": report.parity_cosine,
        "optimizer_restart_required": True,
        "training_resume_allowed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a parity-gated 181M -> 500M V4 growth initialization"
    )
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--output-checkpoint", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--source-profile", default=CANONICAL_MODEL_PROFILE)
    parser.add_argument("--target-profile", default=ANRA_V4_GROWTH_MODEL_PROFILE)
    parser.add_argument("--parity-prompts", default=None)
    parser.add_argument("--minimum-cosine", type=float, default=0.99)
    parser.add_argument(
        "--parent-stage-policy",
        choices=sorted(GROWTH_PARENT_POLICIES),
        required=True,
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = grow_checkpoint(
        source_checkpoint=args.source_checkpoint,
        output_checkpoint=args.output_checkpoint,
        report_path=args.report,
        source_profile=args.source_profile,
        target_profile=args.target_profile,
        parity_prompts=_load_prompts(args.parity_prompts),
        minimum_cosine=args.minimum_cosine,
        parent_stage_policy=args.parent_stage_policy,
        device=args.device,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
