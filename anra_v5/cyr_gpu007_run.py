"""CYR-GPU-007 Colab runner.

This is a thin compatibility layer over the already-audited CYR-GPU-006 CUDA,
checkpoint, generation and transfer machinery. 007 changes only the
prospectively hardware-resolved scope: 2 or 3 acquisition parents, optional
transfer, and a TINY last-resort development proxy. Scientific outcomes never
select the tier.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Mapping

from anra_v5 import cyr_gpu006_run_final as impl
from v5_experiments import cyr_gpu006_final as core6
from v5_experiments import cyr_gpu007 as core

BUNDLE_NAME = "CYMEK_GPU_RESEARCH_V7_RESULTS.zip"
production_tokenizer = impl.production_tokenizer
write_json = impl.write_json
read_json = impl.read_json


def calibrate_candidates(
    *, registry: Mapping[str, Mapping[str, Any]], tokenizer: Any, torch: Any,
    device: Any, special: Mapping[str, int], train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Calibrate every scientific fallback, including TINY as last resort."""
    receipts: dict[str, Any] = {}
    calibrate_one = impl.impl.legacy.calibrate_candidate
    for name in ("MIDI", "MICRO", "RESEARCH_SMALL", "TINY"):
        receipts[name] = calibrate_one(
            name=name,
            spec=registry[name]["spec"],
            tokenizer=tokenizer,
            torch=torch,
            device=device,
            special=special,
            train_rows=train_rows,
            eval_rows=eval_rows,
        )
    return receipts


@contextmanager
def _compatibility_contract(resolved: Mapping[str, Any]):
    """Temporarily parameterize the frozen 006 engine with the 007 contract.

    The engine is single-process/single-threaded in Colab. Every mutated global
    is restored in finally so imports remain deterministic after the run.
    """
    resolved = core.validate_resolved(resolved)
    parent_seeds = tuple(int(seed) for seed in resolved["parent_seeds"])
    transfer_target = int(resolved["transfer_target_parents"])
    transfer_min = int(resolved["transfer_min_parents"])

    legacy = impl.impl.legacy
    legacy_core = legacy.core
    saved = {
        "core6_id": core6.CYR6_ID,
        "core6_parents": core6.CYR6_PARENT_SEEDS,
        "core6_transfer_target": core6.CYR6_TRANSFER_TARGET_PARENTS,
        "core6_transfer_min": core6.CYR6_TRANSFER_MIN_PARENTS,
        "core6_validate": core6.validate_resolved,
        "core6_final_decision": core6.final_decision,
        "legacy_id": legacy_core.CYR6_ID,
        "legacy_bundle": legacy.BUNDLE_NAME,
    }
    try:
        core6.CYR6_ID = core.CYR7_ID
        core6.CYR6_PARENT_SEEDS = parent_seeds
        core6.CYR6_TRANSFER_TARGET_PARENTS = transfer_target
        core6.CYR6_TRANSFER_MIN_PARENTS = transfer_min
        core6.validate_resolved = core.validate_resolved
        core6.final_decision = lambda parent_runs, *, transfer: core.final_decision(
            parent_runs, transfer=transfer, resolved=resolved
        )
        legacy_core.CYR6_ID = core.CYR7_ID
        legacy.BUNDLE_NAME = BUNDLE_NAME
        yield resolved
    finally:
        core6.CYR6_ID = saved["core6_id"]
        core6.CYR6_PARENT_SEEDS = saved["core6_parents"]
        core6.CYR6_TRANSFER_TARGET_PARENTS = saved["core6_transfer_target"]
        core6.CYR6_TRANSFER_MIN_PARENTS = saved["core6_transfer_min"]
        core6.validate_resolved = saved["core6_validate"]
        core6.final_decision = saved["core6_final_decision"]
        legacy_core.CYR6_ID = saved["legacy_id"]
        legacy.BUNDLE_NAME = saved["legacy_bundle"]


def run_campaign(
    *, repo: Path, out: Path, preregistration: Mapping[str, Any],
    resolved: Mapping[str, Any], calibrations: Mapping[str, Any],
    torch: Any = None, device: Any = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    if torch is None:
        import torch as torch_module
        torch = torch_module
    if not torch.cuda.is_available():
        raise RuntimeError("CYR-GPU-007 requires a CUDA Colab GPU")
    if device is None:
        device = torch.device("cuda")
    if getattr(device, "type", None) != "cuda":
        raise RuntimeError("CYR-GPU-007 refuses non-CUDA scientific execution")
    with _compatibility_contract(resolved) as validated:
        campaign = impl.run_campaign(
            repo=repo,
            out=out,
            preregistration=preregistration,
            resolved=validated,
            calibrations=calibrations,
            torch=torch,
            device=device,
            progress=progress,
        )
    campaign["experiment"] = core.CYR7_ID
    campaign["hardware_tier"] = validated["tier"]
    campaign["claim_ceiling"] = validated["claim_ceiling"]
    return campaign
