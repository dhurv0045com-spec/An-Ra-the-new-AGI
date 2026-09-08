"""CYR-GPU-006 revised orchestrator after Arkenstone Discovery V6 audit.

The original 006 executable candidate repaired CUDA propagation, shared-parent
forks and replication. This revision keeps those repairs and changes the
non-arithmetic stage so it answers the actual V6 bottlenecks:

* transfer uses a prospectively chosen HYSTERETIC_HIGH_LOW state, never the
  post-hoc arithmetic winner;
* the new skill is deterministic order-augmented registry binding, because
  ARK-014 showed canonical-only binding is a known brittle subject;
* canonical/query-only/order-only/query+order metrics stay orthogonal;
* binding acquisition and old T2 retention are measured together;
* a fixed replay-row condition is primary because ARK-013 already established
  the no-replay cross-task interference boundary at Micro scale;
* an optional pure-new-skill condition runs only when the hardware-only
  resolver can afford it.

The scientific campaign remains CUDA-only and uses the real Cymek V5 core.
"""
from __future__ import annotations

import hashlib
import json
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Mapping

from anra_v5 import cyr_gpu006_run as legacy
from v5_experiments import cyr_gpu006_v2 as core

BUNDLE_NAME = legacy.BUNDLE_NAME
production_tokenizer = legacy.production_tokenizer
calibrate_candidates = legacy.calibrate_candidates
generate_rates_batched = legacy.generate_rates_batched
sha256_file = legacy.sha256_file
write_json = legacy.write_json
read_json = legacy.read_json


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _to_row(record: Mapping[str, Any]) -> dict[str, str]:
    text = str(record["text"])
    answer = str(record["answer"])
    if not text.endswith(answer):
        raise ValueError("binding render does not terminate in its answer")
    return {"prompt": text[: len(text) - len(answer)], "answer": answer}


def robust_binding_task() -> dict[str, Any]:
    """Build a disjoint, order-augmented non-arithmetic transfer subject."""
    from v5_experiments.cyr_tournament import eval_variant_texts, render_worlds

    worlds = render_worlds(
        family=core.CYR6_TRANSFER_FAMILY,
        split_seeds={"train": 9101, "control": 9102, "sealed": 9103},
        worlds_per_split={"train": 96, "control": 32, "sealed": 32},
    )
    train_rows: list[dict[str, str]] = []
    for world in worlds["train"]:
        variants = eval_variant_texts(world)
        # Pure semantic augmentation: query identity and presentation order are
        # crossed while answer semantics remain correct.
        for key in ("base", "query_only", "order_only", "query_and_order"):
            train_rows.append(_to_row(variants[key]))

    def eval_sets(split: str) -> dict[str, list[dict[str, str]]]:
        result = {key: [] for key in ("canonical", "query_only", "order_only", "query_order")}
        for world in worlds[split]:
            variants = eval_variant_texts(world)
            result["canonical"].append(_to_row(variants["base"]))
            result["query_only"].append(_to_row(variants["query_only"]))
            result["order_only"].append(_to_row(variants["order_only"]))
            result["query_order"].append(_to_row(variants["query_and_order"]))
        return result

    manifest_body = {
        "schema": "anra-cyr-gpu006-binding-transfer/v2",
        "family": core.CYR6_TRANSFER_FAMILY,
        "split_seeds": {"train": 9101, "control": 9102, "sealed": 9103},
        "world_counts": {key: len(value) for key, value in worlds.items()},
        "train_rows": train_rows,
        "control_world_ids": [world["world_id"] for world in worlds["control"]],
        "sealed_world_ids": [world["world_id"] for world in worlds["sealed"]],
        "augmentation": ["base", "query_only", "order_only", "query_and_order"],
    }
    manifest_sha = hashlib.sha256(_canonical_json(manifest_body)).hexdigest()
    return {
        "train_rows": train_rows,
        "control": eval_sets("control"),
        "sealed": eval_sets("sealed"),
        "manifest": {
            "schema": manifest_body["schema"],
            "family": manifest_body["family"],
            "split_seeds": manifest_body["split_seeds"],
            "world_counts": manifest_body["world_counts"],
            "augmentation": manifest_body["augmentation"],
            "sha256": manifest_sha,
        },
    }


def binding_metrics(
    model: Any,
    tokenizer: Any,
    eval_sets: Mapping[str, list[dict[str, str]]],
    *,
    torch: Any,
    device: Any,
    special: Mapping[str, int],
) -> dict[str, Any]:
    metrics = {
        key: generate_rates_batched(
            model,
            tokenizer,
            rows,
            torch=torch,
            device=device,
            special=special,
            batch_size=32,
        )
        for key, rows in eval_sets.items()
    }
    complete = {
        key: float(value["complete_exact_with_valid_stop"])
        for key, value in metrics.items()
    }
    thresholds = core.CYR6_TRANSFER_ROBUST_THRESHOLDS
    qualified = (
        complete["canonical"] >= thresholds["canonical"]
        and complete["order_only"] >= thresholds["order_only"]
        and complete["query_order"] >= thresholds["query_order"]
    )
    return {
        "variants": metrics,
        "complete_exact": complete,
        "robust_floor": min(
            complete["canonical"], complete["order_only"], complete["query_order"]
        ),
        "qualified": qualified,
        "thresholds": dict(thresholds),
    }


def _mixed_transfer_batch(
    *,
    binding_rows: list[dict[str, str]],
    t2_rows: list[dict[str, Any]],
    update: int,
    mode: str,
) -> list[dict[str, Any]]:
    """Deterministic identical stream for baseline/candidate comparisons."""
    if mode == core.CYR6_TRANSFER_OPTIONAL_MODE:
        width = 20
        return [binding_rows[(update * width + i) % len(binding_rows)] for i in range(width)]
    if mode != core.CYR6_TRANSFER_PRIMARY_MODE:
        raise ValueError(f"unknown transfer mode {mode}")
    # Exactly 18 new-skill rows + 2 old-skill replay rows per logical batch.
    # This is 10% by row count, not a claimed token mixture; receipts report
    # actual token counts for each task.
    new_rows = [
        binding_rows[(update * 18 + i) % len(binding_rows)] for i in range(18)
    ]
    replay_rows = [
        t2_rows[(update * 2 + i) % len(t2_rows)] for i in range(2)
    ]
    return [*new_rows, *replay_rows]


def _train_transfer_state(
    *,
    label: str,
    checkpoint: str,
    seed: int,
    mode: str,
    spec: Any,
    tokenizer: Any,
    torch: Any,
    device: Any,
    special: Mapping[str, int],
    binding: Mapping[str, Any],
    t2_train: list[dict[str, Any]],
    t2_measurement: list[dict[str, Any]],
    target_actual_tokens: int,
    eval_interval_tokens: int,
    deadline: float,
    out: Path,
) -> dict[str, Any]:
    from v5_model.core import initialize
    from v5_training.optimizer import build_adamw_optimizer

    receipt_path = out / "receipt.json"
    existing = read_json(receipt_path)
    if existing and existing.get("status") == "COMPLETE":
        return existing

    model = initialize(spec, seed, torch_module=torch).to(device)
    optimizer = build_adamw_optimizer(model, torch_module=torch)
    legacy._load_checkpoint(Path(checkpoint), model=model, optimizer=optimizer, torch=torch)
    backend = legacy._make_backend(
        model=model,
        optimizer=optimizer,
        special=special,
        device=device,
        torch=torch,
        lr=core.CYR6_LRS["HIGH"],
    )

    consumed = 0
    binding_real_tokens = 0
    replay_real_tokens = 0
    updates = 0
    next_eval = eval_interval_tokens
    robust_flags: list[bool] = []
    robust_g90_onset = None
    robust_g90_confirm = None
    trace: list[dict[str, Any]] = []
    data_sha = binding["manifest"]["sha256"]

    while consumed < target_actual_tokens:
        if time.monotonic() >= deadline:
            break
        batch = _mixed_transfer_batch(
            binding_rows=binding["train_rows"],
            t2_rows=t2_train,
            update=updates,
            mode=mode,
        )
        # Count task-local real tokens from the actual rendered batch so replay
        # exposure is never inferred from row count.
        binding_count = len(batch) if mode == core.CYR6_TRANSFER_OPTIONAL_MODE else 18
        new_part = batch[:binding_count]
        replay_part = batch[binding_count:]
        _tok_new, _seg_new, _elig_new, new_counted = legacy.render_batch(
            tokenizer, new_part, torch=torch, device=device, special=special
        )
        if replay_part:
            _tok_old, _seg_old, _elig_old, old_counted = legacy.render_batch(
                tokenizer, replay_part, torch=torch, device=device, special=special
            )
        else:
            old_counted = {"real_tokens": 0, "supervised_tokens": 0}

        update = legacy._one_update(
            backend=backend,
            tokenizer=tokenizer,
            rows=batch,
            torch=torch,
            device=device,
            special=special,
            cumulative=consumed,
            update=updates + 1,
            data_sha=data_sha,
        )
        amount = int(update["counted"]["real_tokens"])
        consumed += amount
        binding_real_tokens += int(new_counted["real_tokens"])
        replay_real_tokens += int(old_counted["real_tokens"])
        updates += 1

        if consumed >= next_eval or consumed >= target_actual_tokens:
            bind = binding_metrics(
                model,
                tokenizer,
                binding["control"],
                torch=torch,
                device=device,
                special=special,
            )
            old = generate_rates_batched(
                model,
                tokenizer,
                t2_measurement,
                torch=torch,
                device=device,
                special=special,
                batch_size=32,
            )
            robust_flags.append(bool(bind["qualified"]))
            if bind["qualified"] and robust_g90_onset is None:
                robust_g90_onset = consumed
            if (
                robust_g90_confirm is None
                and len(robust_flags) >= core.CYR6_TRANSFER_CONFIRMATIONS
                and all(robust_flags[-core.CYR6_TRANSFER_CONFIRMATIONS :])
            ):
                robust_g90_confirm = consumed
            trace.append(
                {
                    "actual_real_tokens": consumed,
                    "binding_control": bind,
                    "old_t2_measurement": old,
                }
            )
            while next_eval <= consumed:
                next_eval += eval_interval_tokens

    status = "COMPLETE" if consumed >= target_actual_tokens else "TIMEBOX"
    sealed = None
    if status == "COMPLETE":
        sealed = binding_metrics(
            model,
            tokenizer,
            binding["sealed"],
            torch=torch,
            device=device,
            special=special,
        )
    final_old = (
        generate_rates_batched(
            model,
            tokenizer,
            t2_measurement,
            torch=torch,
            device=device,
            special=special,
            batch_size=32,
        )
        if status == "COMPLETE"
        else None
    )
    result = {
        "schema": "anra-cyr-gpu006-transfer-state/v2",
        "label": label,
        "seed": seed,
        "mode": mode,
        "status": status,
        "actual_real_tokens": consumed,
        "target_actual_real_tokens": target_actual_tokens,
        "binding_real_tokens": binding_real_tokens,
        "old_t2_replay_real_tokens": replay_real_tokens,
        "actual_replay_token_fraction": (
            replay_real_tokens / max(binding_real_tokens + replay_real_tokens, 1)
        ),
        "updates": updates,
        "robust_g90_onset_tokens": robust_g90_onset,
        "robust_g90_confirm_tokens": robust_g90_confirm,
        "control_trace": trace,
        "sealed_binding": sealed,
        "old_t2_measurement_final": final_old,
        "full_state_transfer": True,
        "optimizer_state_reset": False,
        "interpretation": (
            "plasticity of the complete post-policy training state; parameter-state "
            "and optimizer-state contributions are not separated in this stage"
        ),
    }
    write_json(receipt_path, result)
    return result


def _pair_summary(pair: Mapping[str, Any]) -> dict[str, Any]:
    baseline = pair["baseline"]
    candidate = pair["candidate"]
    reasons: list[str] = []
    if baseline.get("status") != "COMPLETE" or candidate.get("status") != "COMPLETE":
        reasons.append("incomplete exposure")
        return {"compatible": False, "reasons": reasons}
    if baseline.get("robust_g90_confirm_tokens") is None:
        reasons.append("baseline did not acquire robust binding")
    if candidate.get("robust_g90_confirm_tokens") is None:
        reasons.append("candidate did not acquire robust binding")
    base_sealed = baseline.get("sealed_binding") or {}
    cand_sealed = candidate.get("sealed_binding") or {}
    if not base_sealed.get("qualified", False):
        reasons.append("baseline final sealed binding not qualified")
    if not cand_sealed.get("qualified", False):
        reasons.append("candidate final sealed binding not qualified")

    slowdown = None
    if (
        baseline.get("robust_g90_confirm_tokens") is not None
        and candidate.get("robust_g90_confirm_tokens") is not None
    ):
        base_tokens = max(float(baseline["robust_g90_confirm_tokens"]), 1.0)
        slowdown = float(candidate["robust_g90_confirm_tokens"]) / base_tokens - 1.0
        if slowdown > core.CYR6_TRANSFER_MAX_PLASTICITY_SLOWDOWN:
            reasons.append("candidate robust acquisition materially slower")

    floor_diff = None
    if base_sealed and cand_sealed:
        floor_diff = float(cand_sealed.get("robust_floor", 0.0)) - float(
            base_sealed.get("robust_floor", 0.0)
        )
        if floor_diff < -core.CYR6_TRANSFER_FINAL_TOLERANCE:
            reasons.append("candidate final robust binding materially worse")

    old_diff = None
    base_old = baseline.get("old_t2_measurement_final")
    cand_old = candidate.get("old_t2_measurement_final")
    if base_old and cand_old:
        old_diff = float(cand_old["complete_exact_with_valid_stop"]) - float(
            base_old["complete_exact_with_valid_stop"]
        )
        if old_diff < -core.CYR6_TRANSFER_FINAL_TOLERANCE:
            reasons.append("candidate old-skill retention materially worse")

    return {
        "compatible": not reasons,
        "reasons": reasons,
        "robust_acquisition_slowdown": slowdown,
        "sealed_robust_floor_difference": floor_diff,
        "old_t2_final_difference": old_diff,
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
    candidate_name = str(resolved["transfer_candidate"])
    eligible = [
        parent
        for parent in parent_runs
        if parent.get("parent_status") == "G90_CONFIRMED"
        and parent.get("arms", {}).get(candidate_name, {}).get("status") == "COMPLETE"
    ]
    eligible = eligible[: core.CYR6_TRANSFER_MIN_PARENTS]
    if len(eligible) < core.CYR6_TRANSFER_MIN_PARENTS:
        return {
            "schema": "anra-cyr-gpu006-transfer/v2",
            "status": "INCONCLUSIVE",
            "reason": "fewer than two complete prospective-candidate parent states",
            "candidate": candidate_name,
            "binding_manifest": binding["manifest"],
            "pairs": [],
        }

    pairs: list[dict[str, Any]] = []
    for mode_index, mode in enumerate(resolved["transfer_modes"]):
        for parent_index, parent in enumerate(eligible):
            seed = int(parent["seed"])
            checkpoints = {
                "baseline": parent["parent"]["parent_checkpoint"],
                "candidate": parent["arms"][candidate_name]["final_checkpoint"],
            }
            # Counterbalance execution order to reduce deadline/order bias.
            labels = ["baseline", "candidate"]
            if (mode_index + parent_index) % 2:
                labels.reverse()
            outputs: dict[str, Any] = {}
            for label in labels:
                if time.monotonic() >= deadline:
                    outputs[label] = {"status": "TIMEBOX"}
                    continue
                state_out = out / "transfer" / f"parent-{seed}" / mode / label
                outputs[label] = _train_transfer_state(
                    label=label,
                    checkpoint=checkpoints[label],
                    seed=seed,
                    mode=mode,
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
                "mode": mode,
                "candidate": candidate_name,
                "baseline": outputs.get("baseline", {"status": "TIMEBOX"}),
                "candidate_state": outputs.get("candidate", {"status": "TIMEBOX"}),
            }
            # Stable alias for the scorer.
            pair["candidate"] = pair.pop("candidate_state")
            pair["summary"] = _pair_summary(pair)
            pairs.append(pair)

    by_mode: dict[str, Any] = {}
    for mode in resolved["transfer_modes"]:
        mode_pairs = [pair for pair in pairs if pair["mode"] == mode]
        complete = [pair for pair in mode_pairs if pair["summary"]["compatible"]]
        by_mode[mode] = {
            "compatible_pairs": len(complete),
            "required": core.CYR6_TRANSFER_MIN_PARENTS,
            "status": (
                "REPLICATED_COMPATIBLE"
                if len(complete) >= core.CYR6_TRANSFER_MIN_PARENTS
                else "INCONCLUSIVE"
            ),
            "pairs": mode_pairs,
        }

    primary_ok = (
        by_mode.get(core.CYR6_TRANSFER_PRIMARY_MODE, {}).get("status")
        == "REPLICATED_COMPATIBLE"
    )
    return {
        "schema": "anra-cyr-gpu006-transfer/v2",
        "status": (
            "REPLICATED_PLASTICITY_COMPATIBLE" if primary_ok else "INCONCLUSIVE"
        ),
        "candidate": candidate_name,
        "candidate_selection": "prospective from ARK-011 prior evidence",
        "binding_manifest": binding["manifest"],
        "modes": by_mode,
        "pairs": pairs,
        "claim_limit": (
            "tests plasticity of full post-policy training state on robust binding; "
            "does not isolate optimizer-state vs parameter-state causality"
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
    """Execute the revised unfrozen CYR-GPU-006 plan."""
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
    packaging_deadline = (
        hard_deadline - core.CYR6_PACKAGING_RESERVE_MINUTES * 60.0
    )
    science_window = packaging_deadline - started
    acquisition_deadline = started + science_window * 0.35
    continuation_deadline = started + science_window * 0.78
    transfer_deadline = packaging_deadline

    campaign: dict[str, Any] = {
        "schema": "anra-cyr-gpu006-campaign/v2",
        "experiment": core.CYR6_ID,
        "status": "RUNNING",
        "resolved": dict(resolved),
        "calibrations": dict(calibrations),
        "environment": legacy._environment(torch, device),
        "arkenstone_audited_sha": core.ARKENSTONE_AUDITED_SHA,
        "discovery_v6_bundle_sha256": core.ARKENSTONE_DISCOVERY_V6_BUNDLE_SHA256,
        "resume_semantics": (
            "completed acquisition/retention/transfer states are skipped; incomplete "
            "states restart from their immutable source checkpoint and deterministic stream"
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
        audit = core.commutation_audit(splits, tv_bound=0.20)
        if not audit["commutation_free"]:
            raise ValueError(f"data leak audit failed: {audit['findings']}")
        campaign["data_manifest"] = manifest
        campaign["split_manifest"] = {
            name: [row["world_id"] for row in rows] for name, rows in splits.items()
        }
        campaign["leak_audit"] = audit

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
                legacy.acquire_parent(
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
            run_path = out / f"parent-{seed}" / "parent-run-v2.json"
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

            equivalence = legacy.verify_parent_equivalence(
                parent=parent, spec=spec, torch=torch, device=device
            )
            arms: dict[str, Any] = {}
            for arm in core.arm_order(parent_index):
                arms[arm] = legacy.continuation_arm(
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

        retention_preliminary = core.decide_retention(parent_runs)
        campaign["retention_preliminary"] = retention_preliminary

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

        # Arithmetic sealed set is consumed only after all training and all
        # controller-dependent decisions are over.
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
                legacy._load_checkpoint(
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
            "COMPLETE"
            if decision.get("verdict") != "INCONCLUSIVE"
            or transfer is not None
            else "COMPLETE_INCONCLUSIVE"
        )
    except Exception as exc:
        failure = {
            "schema": "anra-cyr-gpu006-failure/v2",
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
        bundle = legacy.package_bundle(
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
