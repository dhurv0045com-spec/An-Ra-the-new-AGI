"""Run one bounded, full-parameter V4 GPU optimizer step.

This is an execution and reproducibility canary, not a training launcher.  It
never writes a checkpoint and it cannot run more than one optimizer step per
repeat.  The default sequence length is deliberately small enough for the
local 6 GiB RTX 4050; longer contexts require an explicit acknowledgement.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from typing import Any

import torch
from anra.extensions import attach_candidate_adapters
from training.anra_optimizer import build_optimizer
from training.reproducibility import CANONICAL_TRAINING_SEED, seed_everything
from training.v2_config import (
    ANRA_V4_MODEL,
    ANRA_V4_TRAINING,
    CANONICAL_MODEL_PROFILE,
    model_parameter_count,
)
from training.v2_runtime import build_model_for_profile

from scripts.build_brain import _configure_continuation_phase, _masked_logit_z_loss

SAFE_LOCAL_CONTEXT = 256


def _initialization_fingerprint(model: torch.nn.Module) -> str:
    """Hash tensor identity plus deterministic probes without copying the model."""

    digest = hashlib.sha256()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            flat = parameter.detach().reshape(-1)
            digest.update(name.encode("utf-8"))
            digest.update(str(tuple(parameter.shape)).encode("ascii"))
            if flat.numel():
                indices = sorted({0, flat.numel() // 3, (2 * flat.numel()) // 3, flat.numel() - 1})
                probe = flat[indices].float().cpu().numpy()
                digest.update(probe.tobytes())
    return digest.hexdigest()


def run_canary(
    *,
    variant: str,
    seed: int,
    sequence_length: int,
    adapter: str = "off",
) -> dict[str, Any]:
    if variant not in {"dense", "mtp"}:
        raise ValueError("variant must be dense or mtp")
    if adapter not in {"off", "lora", "dora"}:
        raise ValueError("adapter must be off, lora, or dora")
    if sequence_length < 4 or sequence_length > ANRA_V4_MODEL.block_size:
        raise ValueError(
            f"sequence length must be in [4, {ANRA_V4_MODEL.block_size}]"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("V4 GPU canary requires a CUDA-visible GPU")

    seed_report = seed_everything(seed)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    use_mtp = variant == "mtp"
    model = build_model_for_profile(
        CANONICAL_MODEL_PROFILE,
        block_size=sequence_length,
        use_mtp=use_mtp,
    )
    expected_parameters = model_parameter_count(
        ANRA_V4_MODEL,
        mtp_depth=2 if use_mtp else 0,
    )
    actual_parameters = sum(parameter.numel() for parameter in model.parameters())
    if actual_parameters != expected_parameters:
        raise AssertionError(
            f"parameter contract mismatch: {actual_parameters} != {expected_parameters}"
        )

    device = torch.device("cuda")
    model = model.to(device)
    phase = _configure_continuation_phase(model, "A")
    adapter_targets: tuple[str, ...] = ()
    if adapter != "off":
        adapter_targets = tuple(
            name
            for name, module in model.named_modules()
            if isinstance(module, torch.nn.Linear)
            and name.endswith(("attn.q_proj", "attn.v_proj", "mlp.down_proj"))
        )
        attach_candidate_adapters(
            model,
            rank=8,
            alpha=16.0,
            dora=adapter == "dora",
            target_modules=adapter_targets,
        )
    fingerprint = _initialization_fingerprint(model)
    installed_parameters = sum(parameter.numel() for parameter in model.parameters())
    adapter_parameters = installed_parameters - actual_parameters
    model.train()
    model.disable_kv_cache()
    optimizer = build_optimizer(
        model,
        lr=ANRA_V4_TRAINING.learning_rate,
        weight_decay=ANRA_V4_TRAINING.weight_decay,
        optimizer_name=ANRA_V4_TRAINING.optimizer,
    )
    inputs = torch.randint(
        0,
        ANRA_V4_MODEL.vocab_size,
        (1, sequence_length),
        device=device,
    )
    targets = torch.randint(
        0,
        ANRA_V4_MODEL.vocab_size,
        (1, sequence_length),
        device=device,
    )
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits, _ = model(inputs)
        base_loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
        )
        z_loss = _masked_logit_z_loss(
            logits,
            targets,
            pad_id=ANRA_V4_MODEL.pad_token_id,
            weight=ANRA_V4_TRAINING.logit_z_loss_weight,
        )
        mtp_loss = (
            model.multi_token_prediction_loss(targets)
            if use_mtp
            else torch.zeros((), device=device, dtype=base_loss.dtype)
        )
        total_loss = base_loss + z_loss + mtp_loss
    if not bool(torch.isfinite(total_loss)):
        raise RuntimeError("canary produced a non-finite loss")
    total_loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), ANRA_V4_TRAINING.max_grad_norm
    )
    if not bool(torch.isfinite(gradient_norm)):
        raise RuntimeError("canary produced non-finite gradients")
    optimizer.step()
    torch.cuda.synchronize()

    return {
        "status": "passed",
        "variant": variant,
        "seed": seed,
        "sequence_length": sequence_length,
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "base_parameters": actual_parameters,
        "expected_base_parameters": expected_parameters,
        "installed_parameters": installed_parameters,
        "parameters": installed_parameters,
        "expected_parameters": expected_parameters + adapter_parameters,
        "adapter_parameters": adapter_parameters,
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "active_native_parameters": len(phase["active_subsystem_parameters"]),
        "adapter": adapter,
        "adapter_target_count": len(adapter_targets),
        "initialization_fingerprint": fingerprint,
        "logit_probe": logits.detach().float().reshape(-1)[:8].cpu().tolist(),
        "base_loss": float(base_loss.detach()),
        "logit_z_loss": float(z_loss.detach()),
        "mtp_weighted_loss": float(mtp_loss.detach()),
        "total_loss": float(total_loss.detach()),
        "preclip_gradient_norm": float(gradient_norm.detach()),
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1_048_576,
        "elapsed_seconds": time.perf_counter() - started,
        "seed_contract": seed_report.to_dict(),
        "checkpoint_written": False,
        "optimizer_steps": 1,
    }


def _replay_matches(first: dict[str, Any], second: dict[str, Any]) -> bool:
    exact_fields = (
        "initialization_fingerprint",
        "logit_probe",
        "base_loss",
        "logit_z_loss",
        "mtp_weighted_loss",
        "total_loss",
        "preclip_gradient_norm",
    )
    return all(first[field] == second[field] for field in exact_fields)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=["dense", "mtp"], default="dense")
    parser.add_argument("--adapter", choices=["off", "lora", "dora"], default="off")
    parser.add_argument("--seed", type=int, default=CANONICAL_TRAINING_SEED)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--repeat", type=int, choices=[1, 2], default=1)
    parser.add_argument(
        "--allow-large-context",
        action="store_true",
        help=(
            f"Acknowledge the OOM risk above sequence length {SAFE_LOCAL_CONTEXT}; "
            "still performs exactly one optimizer step per repeat."
        ),
    )
    args = parser.parse_args()
    if args.sequence_length > SAFE_LOCAL_CONTEXT and not args.allow_large_context:
        parser.error(
            f"sequence length above {SAFE_LOCAL_CONTEXT} requires --allow-large-context"
        )

    runs: list[dict[str, Any]] = []
    for _ in range(args.repeat):
        runs.append(
            run_canary(
                variant=args.variant,
                seed=args.seed,
                sequence_length=args.sequence_length,
                adapter=args.adapter,
            )
        )
        gc.collect()
        torch.cuda.empty_cache()
    report: dict[str, Any] = {
        "schema_version": 1,
        "purpose": "bounded_execution_and_same_stack_replay_only",
        "runs": runs,
        "replay_checked": args.repeat == 2,
        "replay_exact": _replay_matches(runs[0], runs[1]) if args.repeat == 2 else None,
        "quality_claim": False,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["replay_exact"] is not False else 2


if __name__ == "__main__":
    raise SystemExit(main())
