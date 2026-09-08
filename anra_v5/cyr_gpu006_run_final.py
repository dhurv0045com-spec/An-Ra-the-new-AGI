"""Final CYR-GPU-006 Colab orchestrator candidate.

This layer uses the already-tested CUDA/shared-parent machinery in
``cyr_gpu006_run_v2`` but owns the final campaign semantics. In particular,
the non-arithmetic plasticity comparison is equal-age/equal-exposure:
HYSTERETIC_HIGH_LOW versus LOW_CONTINUE from the same parent after the same
continuation-token budget. Both states then learn the same order-robust
binding stream at HIGH LR with a fixed 18:2 new-skill:T2 replay row ratio.

The old pre-continuation-parent versus post-retention-state comparison is not
used because it confounded retention policy with additional model age and
same-task exposure.
"""
from __future__ import annotations

import time
import traceback
from pathlib import Path
from typing import Any, Callable, Mapping

from anra_v5 import cyr_gpu006_run_v2 as impl
from v5_experiments import cyr_gpu006_final as core

BUNDLE_NAME = impl.BUNDLE_NAME
production_tokenizer = impl.production_tokenizer
calibrate_candidates = impl.calibrate_candidates
generate_rates_batched = impl.generate_rates_batched
write_json = impl.write_json
read_json = impl.read_json
robust_binding_task = impl.robust_binding_task


def _source_eligible(parent: Mapping[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if parent.get("parent_status") != "G90_CONFIRMED":
        reasons.append("parent not G90_CONFIRMED")
    if not parent.get("parent_equivalence", {}).get("identical", False):
        reasons.append("fork source bytes not equivalent")
    if not parent.get("future_tail", {}).get("identical", False):
        reasons.append("fork future-tail mismatch")
    arms = parent.get("arms", {})
    candidate = arms.get(core.CYR6_TRANSFER_CANDIDATE)
    comparator = arms.get(core.CYR6_TRANSFER_COMPARATOR)
    for label, receipt in (("candidate", candidate), ("comparator", comparator)):
        if not receipt:
            reasons.append(f"missing {label} retention state")
            continue
        if receipt.get("status") != "COMPLETE":
            reasons.append(f"{label} retention state incomplete")
        if not receipt.get("redteam_pass", False):
            reasons.append(f"{label} redteam failed")
        if float(receipt.get("final_g", 0.0)) < core.CYR6_TRANSFER_SOURCE_MIN_T2:
            reasons.append(f"{label} source not T2-qualified at transfer fork")
    if candidate and comparator:
        if int(candidate.get("actual_real_tokens", -1)) != int(
            comparator.get("actual_real_tokens", -2)
        ):
            reasons.append("candidate/comparator continuation exposure differs")
        if candidate.get("future_tail_sha256") != comparator.get("future_tail_sha256"):
            reasons.append("candidate/comparator continuation identity differs")
        if candidate.get("parent_model_sha256") != comparator.get("parent_model_sha256"):
            reasons.append("candidate/comparator parent identity differs")
    return not reasons, reasons


def _transfer_pair_summary(pair: Mapping[str, Any]) -> dict[str, Any]:
    comparator = pair["comparator"]
    candidate = pair["candidate"]
    reasons: list[str] = []
    if comparator.get("status") != "COMPLETE" or candidate.get("status") != "COMPLETE":
        return {
            "compatible": False,
            "advantage": False,
            "reasons": ["incomplete transfer exposure"],
        }

    for label, receipt in (("comparator", comparator), ("candidate", candidate)):
        if receipt.get("robust_g90_confirm_tokens") is None:
            reasons.append(f"{label} did not acquire robust binding")
        sealed = receipt.get("sealed_binding") or {}
        if not sealed.get("qualified", False):
            reasons.append(f"{label} final sealed binding not robust-qualified")

    slowdown = None
    comp_confirm = comparator.get("robust_g90_confirm_tokens")
    cand_confirm = candidate.get("robust_g90_confirm_tokens")
    if comp_confirm is not None and cand_confirm is not None:
        slowdown = float(cand_confirm) / max(float(comp_confirm), 1.0) - 1.0
        if slowdown > core.CYR6_TRANSFER_MAX_PLASTICITY_SLOWDOWN:
            reasons.append("adaptive candidate robust acquisition materially slower")

    comp_sealed = comparator.get("sealed_binding") or {}
    cand_sealed = candidate.get("sealed_binding") or {}
    floor_diff = None
    if comp_sealed and cand_sealed:
        floor_diff = float(cand_sealed.get("robust_floor", 0.0)) - float(
            comp_sealed.get("robust_floor", 0.0)
        )
        if floor_diff < -core.CYR6_TRANSFER_FINAL_TOLERANCE:
            reasons.append("adaptive candidate final robust binding materially worse")

    comp_old = comparator.get("old_t2_measurement_final")
    cand_old = candidate.get("old_t2_measurement_final")
    old_diff = None
    if comp_old and cand_old:
        old_diff = float(cand_old["complete_exact_with_valid_stop"]) - float(
            comp_old["complete_exact_with_valid_stop"]
        )
        if old_diff < -core.CYR6_TRANSFER_FINAL_TOLERANCE:
            reasons.append("adaptive candidate old-skill retention materially worse")

    compatible = not reasons
    speed_advantage = bool(
        compatible
        and comp_confirm is not None
        and cand_confirm is not None
        and float(cand_confirm)
        <= float(comp_confirm) * (1.0 - core.CYR6_TRANSFER_ADVANTAGE_SPEED_FRACTION)
    )
    retention_advantage = bool(
        compatible
        and old_diff is not None
        and old_diff >= core.CYR6_TRANSFER_ADVANTAGE_RETENTION
    )
    return {
        "compatible": compatible,
        "advantage": bool(speed_advantage or retention_advantage),
        "reasons": reasons,
        "robust_acquisition_slowdown": slowdown,
        "sealed_robust_floor_difference": floor_diff,
        "old_t2_final_difference": old_diff,
        "speed_advantage": speed_advantage,
        "retention_advantage": retention_advantage,
    }


def transfer_plasticity_replicated(
    *,
    parent_runs: list[Mapping[str, Any]],
    spec: Any,
    tokenizer: Any,
    torch: Any,
    device: Any,
    special: Mapping[str, int],
    t2_train: list[dict[str, Any]],
    t2_measurement: list[dict[str, Any]],
    resolved: Mapping[str, Any],
    deadline: float,
    out: Path,
) -> dict[str, Any]:
    binding = robust_binding_task()
    eligible: list[Mapping[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for parent in parent_runs:
        ok, reasons = _source_eligible(parent)
        if ok:
            eligible.append(parent)
        else:
            rejected.append({"seed": parent.get("seed"), "reasons": reasons})
    eligible = eligible[: int(resolved["transfer_target_parents"])]
    if len(eligible) < int(resolved["transfer_min_parents"]):
        return {
            "schema": "anra-cyr-gpu006-transfer/final-v1",
            "status": "INCONCLUSIVE",
            "reason": "fewer than two equal-age T2-qualified candidate/comparator source pairs",
            "candidate_arm": core.CYR6_TRANSFER_CANDIDATE,
            "comparator_arm": core.CYR6_TRANSFER_COMPARATOR,
            "binding_manifest": binding["manifest"],
            "eligible_seeds": [parent.get("seed") for parent in eligible],
            "rejected": rejected,
            "pairs": [],
        }

    pairs: list[dict[str, Any]] = []
    for parent_index, parent in enumerate(eligible):
        seed = int(parent["seed"])
        source_arms = parent["arms"]
        checkpoints = {
            "comparator": source_arms[core.CYR6_TRANSFER_COMPARATOR]["final_checkpoint"],
            "candidate": source_arms[core.CYR6_TRANSFER_CANDIDATE]["final_checkpoint"],
        }
        # Alternate execution order across independent parents to avoid making
        # one treatment systematically later in the wall-time box.
        labels = ["comparator", "candidate"]
        if parent_index % 2:
            labels.reverse()
        outputs: dict[str, Any] = {}
        for label in labels:
            if time.monotonic() >= deadline:
                outputs[label] = {"status": "TIMEBOX"}
                continue
            state_out = out / "transfer" / f"parent-{seed}" / label
            outputs[label] = impl._train_transfer_state(
                label=label,
                checkpoint=checkpoints[label],
                seed=seed,
                mode=core.CYR6_TRANSFER_MODE,
                spec=spec,
                tokenizer=tokenizer,
                torch=torch,
                device=device,
                special=special,
                binding=binding,
                t2_train=t2_train,
                t2_measurement=t2_measurement,
                target_actual_tokens=int(resolved["transfer_target_actual_tokens"]),
                eval_interval_tokens=int(resolved["transfer_eval_interval_tokens"]),
                deadline=deadline,
                out=state_out,
            )
        pair = {
            "seed": seed,
            "mode": core.CYR6_TRANSFER_MODE,
            "candidate_arm": core.CYR6_TRANSFER_CANDIDATE,
            "comparator_arm": core.CYR6_TRANSFER_COMPARATOR,
            "source_parent_model_sha256": source_arms[core.CYR6_TRANSFER_CANDIDATE][
                "parent_model_sha256"
            ],
            "source_continuation_tokens": source_arms[core.CYR6_TRANSFER_CANDIDATE][
                "actual_real_tokens"
            ],
            "source_future_tail_sha256": source_arms[core.CYR6_TRANSFER_CANDIDATE][
                "future_tail_sha256"
            ],
            "comparator": outputs.get("comparator", {"status": "TIMEBOX"}),
            "candidate": outputs.get("candidate", {"status": "TIMEBOX"}),
        }
        pair["summary"] = _transfer_pair_summary(pair)
        pairs.append(pair)

    compatible = [pair for pair in pairs if pair["summary"]["compatible"]]
    advantages = [pair for pair in compatible if pair["summary"]["advantage"]]
    required = int(resolved["transfer_min_parents"])
    if len(compatible) < required:
        status = "INCONCLUSIVE"
    elif len(advantages) >= required:
        status = "REPLICATED_PLASTICITY_ADVANTAGE"
    else:
        status = "REPLICATED_PLASTICITY_NONINFERIOR"
    return {
        "schema": "anra-cyr-gpu006-transfer/final-v1",
        "status": status,
        "candidate_arm": core.CYR6_TRANSFER_CANDIDATE,
        "comparator_arm": core.CYR6_TRANSFER_COMPARATOR,
        "candidate_selection": "prospective from ARK-011; not selected from CYR arithmetic outcomes",
        "comparison": (
            "equal acquired parent, equal same-task continuation exposure, then equal HIGH-LR "
            "order-augmented binding+replay stream"
        ),
        "binding_manifest": binding["manifest"],
        "eligible_seeds": [int(parent["seed"]) for parent in eligible],
        "compatible_pairs": len(compatible),
        "advantage_pairs": len(advantages),
        "minimum_required": required,
        "rejected": rejected,
        "pairs": pairs,
        "claim_limit": (
            "tests plasticity of the complete post-retention training state; it does not "
            "separate parameter-state from Adam-moment causality"
        ),
    }


def run_campaign(
    *,
    repo: Path,
    out: Path,
    preregistration: Mapping[str, Any],
    resolved: Mapping[str, Any],
    calibrations: Mapping[str, Any],
    torch: Any = None,
    device: Any = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Execute the final unfrozen CYR-GPU-006 candidate; never recalibrate here."""
    if torch is None:
        import torch as torch_module

        torch = torch_module
    if not torch.cuda.is_available():
        raise RuntimeError("CYR-GPU-006 requires a CUDA Colab GPU")
    if device is None:
        device = torch.device("cuda")
    if getattr(device, "type", None) != "cuda":
        raise RuntimeError("CYR-GPU-006 full run refuses non-CUDA device")
    resolved = core.validate_resolved(resolved)
    if (
        resolved["proxy"] not in calibrations
        or calibrations[resolved["proxy"]].get("status") != "PASS"
    ):
        raise ValueError("resolved proxy lacks a passing CELL-0 calibration")

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    hard_deadline = started + float(resolved["wall_budget_minutes"]) * 60.0
    packaging_deadline = hard_deadline - core.CYR6_PACKAGING_RESERVE_MINUTES * 60.0
    budgets = resolved["stage_budgets_seconds"]
    acquisition_deadline = min(started + float(budgets["acquisition"]), packaging_deadline)
    continuation_deadline = min(
        acquisition_deadline + float(budgets["retention"]), packaging_deadline
    )
    transfer_deadline = packaging_deadline

    campaign: dict[str, Any] = {
        "schema": "anra-cyr-gpu006-campaign/final-v1",
        "experiment": core.CYR6_ID,
        "status": "RUNNING",
        "resolved": dict(resolved),
        "calibrations": dict(calibrations),
        "environment": impl.legacy._environment(torch, device),
        "arkenstone_audited_sha": core.ARKENSTONE_AUDITED_SHA,
        "discovery_v6_bundle_sha256": core.ARKENSTONE_DISCOVERY_V6_BUNDLE_SHA256,
        "resume_semantics": (
            "completed parents/arms/transfer states are skipped; incomplete states restart "
            "from their immutable source checkpoint and deterministic stream"
        ),
    }
    failure: dict[str, Any] | None = None
    try:
        tokenizer, identity = production_tokenizer(repo)
        special = {
            "pad_id": identity["pad_id"],
            "bos_id": identity["bos_id"],
            "eos_id": identity["eos_id"],
        }
        if (
            preregistration.get("tokenizer", {}).get("artifact_sha256")
            != identity["artifact_sha256"]
        ):
            raise ValueError("runtime tokenizer differs from preregistration")

        splits = core.render_t2_worlds()
        manifest = core.build_data_manifest(splits)
        core.assert_manifest_sha(manifest)
        expected_manifest = preregistration.get("data", {}).get(
            "data_manifest_sha256_full"
        )
        if expected_manifest and manifest["sha256"] != expected_manifest:
            raise ValueError("runtime rendered-data manifest differs from preregistration")
        leak_audit = core.commutation_audit(splits, tv_bound=0.20)
        if not leak_audit["commutation_free"]:
            raise ValueError(f"data leak audit failed: {leak_audit['findings']}")
        campaign["data_manifest"] = manifest
        campaign["split_manifest"] = {
            name: [row["world_id"] for row in rows] for name, rows in splits.items()
        }
        campaign["leak_audit"] = leak_audit

        registry = core.proxy_registry(vocab_size=identity["vocabulary_size"])
        campaign["proxy_registry"] = {
            name: {"parameters": entry["parameters"], "role": entry["role"]}
            for name, entry in registry.items()
        }
        spec = registry[resolved["proxy"]]["spec"]

        parent_receipts: list[dict[str, Any]] = []
        streams: dict[int, Any] = {}
        for seed in core.CYR6_PARENT_SEEDS:
            stream = core.build_future_stream(
                seed=seed,
                world_count=len(splits["train"]),
                prefix_rows=max(
                    64, int(resolved["target_actual_tokens_acquisition"] // 128)
                ),
                tail_rows=max(
                    64, int(resolved["target_actual_tokens_continuation"] // 128)
                ),
            )
            streams[seed] = stream
            parent_receipts.append(
                impl.legacy.acquire_parent(
                    seed=seed,
                    spec=spec,
                    proxy_name=resolved["proxy"],
                    tokenizer=tokenizer,
                    torch=torch,
                    device=device,
                    special=special,
                    train_rows=splits["train"],
                    controller_rows=splits["dev_controller"],
                    probe_rows=splits["train"][:16],
                    target_actual_tokens=int(
                        resolved["target_actual_tokens_acquisition"]
                    ),
                    stream=stream,
                    store_root=out,
                    stage_deadline=acquisition_deadline,
                    eval_interval_tokens=int(
                        resolved["acquisition_eval_interval_tokens"]
                    ),
                    progress=progress,
                )
            )
        campaign["parents"] = parent_receipts

        parent_runs: list[dict[str, Any]] = []
        for parent_index, parent in enumerate(parent_receipts):
            seed = int(parent["seed"])
            run_path = out / f"parent-{seed}" / "parent-run-final.json"
            existing_run = read_json(run_path)
            if existing_run and existing_run.get("complete_parent_experiment"):
                parent_runs.append(existing_run)
                continue
            if parent.get("parent_status") != "G90_CONFIRMED":
                record = {
                    "seed": seed,
                    "parent_status": parent["parent_status"],
                    "parent": parent,
                    "arms": {},
                    "parent_equivalence": {"identical": False},
                    "future_tail": {"identical": False},
                    "complete_parent_experiment": True,
                }
                write_json(run_path, record)
                parent_runs.append(record)
                continue

            equivalence = impl.legacy.verify_parent_equivalence(
                parent=parent, spec=spec, torch=torch, device=device
            )
            arms: dict[str, Any] = {}
            for arm in core.arm_order(parent_index):
                arms[arm] = impl.legacy.continuation_arm(
                    arm=arm,
                    parent=parent,
                    spec=spec,
                    tokenizer=tokenizer,
                    torch=torch,
                    device=device,
                    special=special,
                    train_rows=splits["train"],
                    controller_rows=splits["dev_controller"],
                    measurement_rows=splits["dev_measurement"],
                    stream=streams[seed],
                    target_actual_tokens=int(
                        resolved["target_actual_tokens_continuation"]
                    ),
                    store_root=out,
                    stage_deadline=continuation_deadline,
                    eval_interval_tokens=int(
                        resolved["continuation_eval_interval_tokens"]
                    ),
                    progress=progress,
                )
            compared = min(
                (len(receipt["consumed_batch_shas"]) for receipt in arms.values()),
                default=0,
            )
            tail = (
                {"identical": False, "batches_compared": 0}
                if compared == 0
                else core.assert_future_tail_equality(
                    {
                        arm: receipt["consumed_batch_shas"][:compared]
                        for arm, receipt in arms.items()
                    }
                )
            )
            record = {
                "seed": seed,
                "parent_status": parent["parent_status"],
                "parent": parent,
                "arms": arms,
                "parent_equivalence": equivalence,
                "future_tail": tail,
                "complete_parent_experiment": all(
                    receipt.get("status") == "COMPLETE" for receipt in arms.values()
                ),
            }
            write_json(run_path, record)
            parent_runs.append(record)
        campaign["parent_runs"] = parent_runs
        campaign["retention_preliminary"] = core.decide_retention(parent_runs)

        transfer = None
        if resolved.get("transfer_enabled") and time.monotonic() < transfer_deadline:
            transfer = transfer_plasticity_replicated(
                parent_runs=parent_runs,
                spec=spec,
                tokenizer=tokenizer,
                torch=torch,
                device=device,
                special=special,
                t2_train=splits["train"],
                t2_measurement=splits["dev_measurement"],
                resolved=resolved,
                deadline=transfer_deadline,
                out=out,
            )
        campaign["transfer"] = transfer
        decision = core.final_decision(parent_runs, transfer=transfer)
        campaign["decision"] = decision

        # Arithmetic SEALED is never a controller and is consumed only after
        # all optimization and transfer decisions are complete. Measure every
        # completed arm, not just the selected arm, to preserve negative data.
        sealed: dict[str, Any] = {"status": "MEASURED_AFTER_DECISION", "arms": {}}
        for parent_run in parent_runs:
            seed = int(parent_run.get("seed", -1))
            for arm, arm_receipt in parent_run.get("arms", {}).items():
                if arm_receipt.get("status") != "COMPLETE":
                    continue
                from v5_model.core import initialize
                from v5_training.optimizer import build_adamw_optimizer

                model = initialize(spec, seed, torch_module=torch).to(device)
                optimizer = build_adamw_optimizer(model, torch_module=torch)
                impl.legacy._load_checkpoint(
                    Path(arm_receipt["final_checkpoint"]),
                    model=model,
                    optimizer=optimizer,
                    torch=torch,
                )
                sealed["arms"][f"{seed}/{arm}"] = generate_rates_batched(
                    model,
                    tokenizer,
                    splits["sealed_reserved"],
                    torch=torch,
                    device=device,
                    special=special,
                    batch_size=32,
                )
                del model, optimizer
        campaign["sealed"] = sealed
        campaign["status"] = (
            "COMPLETE_RESEARCH_CANDIDATE"
            if decision.get("research_candidate")
            else "COMPLETE_INCONCLUSIVE"
            if decision.get("verdict") == "INCONCLUSIVE"
            else "COMPLETE"
        )
    except Exception as exc:
        failure = {
            "schema": "anra-cyr-gpu006-failure/final-v1",
            "exception": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
            "wall_seconds": time.monotonic() - started,
        }
        campaign["status"] = "FAILED"
        campaign["decision"] = campaign.get(
            "decision",
            {
                "verdict": "INCONCLUSIVE",
                "winner": None,
                "reason": "campaign failure; see FAILURE.json",
                "production_promotion_authorized": False,
            },
        )
    finally:
        campaign["wall_seconds"] = time.monotonic() - started
        bundle = impl.legacy.package_bundle(
            out,
            campaign=campaign,
            preregistration=preregistration,
            failure=failure,
        )
        campaign["bundle"] = bundle
        write_json(out / "campaign_receipt.json", campaign)
    if failure is not None:
        raise RuntimeError(
            f"CYR-GPU-006 failed after packaging partial evidence: {failure['message']}"
        )
    return campaign
