from __future__ import annotations

import argparse
import math
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments" / "COLAB"))

from discovery_v7_common import (
    ARK014_BINDING_MANIFEST_SHA,
    MASTER_PLAN_SHA,
    ReceiptWriter,
    RunContext,
    bind_ark11_runtime,
    cpu_tree,
    current_device,
    file_sha256,
    git_head,
    gradient_norm_and_clip,
    import_experiment_module,
    load_ark11,
    load_ark14,
    model_state_equal,
    optimizer_step_with_delta,
    package_all,
    parameter_sha,
)

RUNNER_PATH = Path(__file__)
DEFAULT_BUDGET_MINUTES = 300


def nested_equal(a: Any, b: Any) -> bool:
    if torch.is_tensor(a) and torch.is_tensor(b):
        return torch.equal(a.detach().cpu(), b.detach().cpu())
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(nested_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(nested_equal(x, y) for x, y in zip(a, b))
    return a == b


def smoke_test(device: torch.device, head: str) -> dict:
    print("\n=== DISCOVERY V7 GPU SMOKE TEST ===", flush=True)
    ctx = RunContext(device=device, head=head, started=time.time(), budget_minutes=20)
    writer = ReceiptWriter(
        ctx,
        experiment_id="MASTER_DISCOVERY_V7_SMOKE",
        plan_sha=MASTER_PLAN_SHA,
        runner_path=RUNNER_PATH,
    )

    ark11 = load_ark11()
    bind_ark11_runtime(ark11, device, head)
    t2_manifest = ark11.load_manifest()
    t2_train = [(p, a) for p, a in t2_manifest["train"]]
    t2_test = [(p, a) for p, a in t2_manifest["test"]]
    t2_control, t2_sealed, t2_split = ark11.build_control_sealed_split(t2_test)
    if not t2_control or not t2_sealed or set(t2_control) & set(t2_sealed):
        raise RuntimeError("T2 CONTROL/SEALED smoke invariant failed")

    ark14 = load_ark14()
    train_meta, bind_control, bind_sealed, bind_manifest = ark14.build_binding_manifest()
    if bind_manifest.get("manifest_sha256") != ARK014_BINDING_MANIFEST_SHA:
        raise RuntimeError(
            f"binding manifest drift: {bind_manifest.get('manifest_sha256')} != {ARK014_BINDING_MANIFEST_SHA}"
        )
    if bind_manifest.get("factset_overlap_train_test") != 0 or bind_manifest.get("factset_overlap_control_sealed") != 0:
        raise RuntimeError("binding fact-set overlap smoke invariant failed")
    sample = train_meta[7]
    aug1 = ark14.augmented_facts(sample, 2301, 17, 3)
    aug2 = ark14.augmented_facts(sample, 2301, 17, 3)
    if aug1 != aug2:
        raise RuntimeError("order augmentation is not deterministic")

    # Import the exact V7 experiment modules before any expensive training.
    import_experiment_module(15)
    import_experiment_module(16)

    vocab = ark11.CompactVocab()
    torch.manual_seed(70707)
    torch.cuda.manual_seed_all(70707)
    model = ark11.Micro(vocab.size, 128).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1
    )
    rows = t2_train[:8]
    loss, count = ark11.loss_and_positions(model, vocab, rows, device)
    if not torch.isfinite(loss) or int(count.item()) <= 0:
        raise RuntimeError("GPU forward/loss smoke failure")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradient_norm_and_clip(model, 1.0)
    optimizer_step_with_delta(model, optimizer)

    snap = ark11.snapshot_state(model, optimizer)
    _, restored_model, restored_optimizer = ark11.load_fork(snap, 1e-3)
    if not model_state_equal(model.state_dict(), restored_model.state_dict()):
        raise RuntimeError("snapshot model reload mismatch")
    if not nested_equal(snap["optimizer"], cpu_tree(restored_optimizer.state_dict())):
        raise RuntimeError("snapshot optimizer reload mismatch")

    # Deterministic next-update identity.
    v1, m1, o1 = ark11.load_fork(snap, 1e-3)
    v2, m2, o2 = ark11.load_fork(snap, 1e-3)
    ark11.train_step(m1, o1, v1, rows)
    ark11.train_step(m2, o2, v2, rows)
    if parameter_sha(m1) != parameter_sha(m2):
        raise RuntimeError("same snapshot/minibatch/LR did not reproduce identical next parameters")

    # Verify the post-step cap primitive using two identical forks.
    vc1, mc1, oc1 = ark11.load_fork(snap, 1e-3)
    loss1, _ = ark11.loss_and_positions(mc1, vc1, rows, device)
    oc1.zero_grad(set_to_none=True)
    loss1.backward()
    gradient_norm_and_clip(mc1, 1.0)
    uncapped = optimizer_step_with_delta(mc1, oc1)
    raw = float(uncapped["raw_delta_norm"])
    if not math.isfinite(raw) or raw <= 0:
        raise RuntimeError(f"invalid raw update norm in cap smoke: {raw}")

    vc2, mc2, oc2 = ark11.load_fork(snap, 1e-3)
    loss2, _ = ark11.loss_and_positions(mc2, vc2, rows, device)
    oc2.zero_grad(set_to_none=True)
    loss2.backward()
    gradient_norm_and_clip(mc2, 1.0)
    requested_cap = raw * 0.5
    capped = optimizer_step_with_delta(mc2, oc2, cap_norm=requested_cap)
    applied = float(capped["applied_delta_norm"])
    if not bool(capped["cap_fired"]):
        raise RuntimeError("delta-cap smoke expected cap to fire")
    if applied > requested_cap * (1.0 + 1e-6) + 1e-12:
        raise RuntimeError(f"applied delta exceeded cap: {applied} > {requested_cap}")
    for state in oc2.state.values():
        for value in state.values():
            if torch.is_tensor(value) and not torch.isfinite(value).all():
                raise RuntimeError("nonfinite optimizer moment after capped step")

    payload = {
        "status": "PASS",
        "canonical_t2_sha256": t2_manifest["split_sha256"],
        "t2_control_sha256": t2_split["control_sha256"],
        "t2_sealed_sha256": t2_split["sealed_sha256"],
        "binding_manifest_sha256": bind_manifest["manifest_sha256"],
        "binding_factset_overlap_train_test": bind_manifest["factset_overlap_train_test"],
        "binding_factset_overlap_control_sealed": bind_manifest["factset_overlap_control_sealed"],
        "order_augmentation_reproducible": True,
        "gpu_forward_backward": True,
        "snapshot_model_reload_exact": True,
        "snapshot_optimizer_reload_exact": True,
        "equal_next_step_parameter_hash": True,
        "delta_cap_primitive": {
            "raw_delta_norm": raw,
            "requested_cap": requested_cap,
            "applied_delta_norm": applied,
            "cap_fired": True,
            "optimizer_moments_finite": True,
        },
    }
    writer.save("DISCOVERY_V7_SMOKE_TEST.json", payload)
    print("DISCOVERY V7 GPU SMOKE TEST PASS", flush=True)
    return payload


def training_design_decision(summary015: dict | None, summary016: dict | None) -> dict:
    s15 = summary015 or {}
    s16 = summary016 or {}
    v15 = s15.get("verdict")
    v16 = s16.get("primary_verdict")
    flags16 = list(s16.get("mechanism_flags") or [])

    if v15 == "SUPPORTED_NONARITHMETIC_INVARIANCE_PROTECTION":
        if "TRUST_REGION_CANDIDATE" in flags16:
            decision = "CANDIDATE_UPDATE_TRUST_REGION"
            intervention = {
                "candidate": "capability-state-aware applied-update trust region",
                "components": [
                    "CONTROL capability probe registry",
                    "raw/applied optimizer update-norm measurement",
                    "state-conditional update budget",
                    "cumulative parameter path + displacement logging",
                    "SEALED evaluation firewall",
                ],
            }
        elif "LOW_LR_SPECIFIC_BEYOND_STEP_NORM" in flags16 and "TRUST_REGION_CANDIDATE" not in flags16:
            decision = "CANDIDATE_STATE_LR_CONTROLLER"
            intervention = {
                "candidate": "hysteretic capability-state LR controller",
                "components": [
                    "CONTROL capability probe registry",
                    "ACQUIRE/RECOVER -> CONSOLIDATE state transition",
                    "state-triggered LR change",
                    "raw/applied update telemetry retained as red-team",
                    "SEALED evaluation firewall",
                ],
            }
        else:
            decision = "RETENTION_EFFECT_TRANSFERRED_MECHANISM_UNRESOLVED"
            intervention = {
                "candidate": None,
                "next_test": "larger proxy mechanism study before hard-coding LR or update-norm control",
            }
    elif v15 == "TRANSFER_NOT_SUPPORTED":
        decision = "ARITHMETIC_OR_STRESS_SPECIFIC_ONLY"
        intervention = {
            "candidate": None,
            "next_test": "do not promote the Micro-T2 optimizer mechanism to general training infrastructure",
        }
    else:
        decision = "INSUFFICIENT_EVENT_EVIDENCE"
        intervention = {
            "candidate": None,
            "next_test": "improve event-generating test design or use an independently preregistered larger proxy",
        }

    measurement = {
        "recommended_measurement_only": True,
        "components": [
            "capability probe registry with explicit CONTROL/SEALED roles",
            "raw and applied optimizer update-norm logging",
            "cumulative parameter path length",
            "relative displacement from capability milestones",
            "acquired/unstable/recovered/consolidated state-transition log",
            "data-mixture or presentation-regime identifier",
            "exact resume identity for controller state",
            "old-skill and new-skill metrics across curriculum/mixture shifts",
        ],
    }

    return {
        "decision": decision,
        "ARK-015_verdict": v15,
        "ARK-016_primary_verdict": v16,
        "ARK-016_mechanism_flags": flags16,
        "intervention_recommendation": intervention,
        "measurement_infrastructure_recommendation": measurement,
        "cymek_production_scheduler_change_authorized": False,
        "pre500m_or_500m_authorized": False,
        "claim_boundary": "research handoff recommendation only; requires larger proxy/TPU promotion gates",
    }


def run_full(total_budget_minutes: float, device: torch.device, head: str) -> dict:
    overall = RunContext(device=device, head=head, started=time.time(), budget_minutes=total_budget_minutes)
    writer = ReceiptWriter(
        overall,
        experiment_id="MASTER_DISCOVERY_V7",
        plan_sha=MASTER_PLAN_SHA,
        runner_path=RUNNER_PATH,
    )
    program = {
        "status": "RUNNING",
        "total_budget_minutes": total_budget_minutes,
        "campaigns": {},
    }

    # Preserve a large independent reserve for the mechanism red-team.
    reserve016 = 140.0
    alloc015 = min(135.0, max(0.0, overall.minutes_left - reserve016))
    result015 = None
    if alloc015 >= 60.0:
        print(f"\n=== MASTER V7: ARK-015 allocation {alloc015:.1f} min ===", flush=True)
        started = time.time()
        ark15 = import_experiment_module(15)
        subctx = RunContext(device=device, head=head, started=started, budget_minutes=alloc015)
        result015 = ark15.run_campaign(subctx)
        program["campaigns"]["ARK-015"] = {
            "status": "EXECUTED_OR_PARTIAL",
            "allocation_minutes": alloc015,
            "actual_minutes": (time.time() - started) / 60.0,
            "summary": result015.get("summary"),
        }
    else:
        program["campaigns"]["ARK-015"] = {
            "status": "BUDGET_BLOCKED",
            "allocation_minutes": alloc015,
        }
    writer.save("DISCOVERY_V7_PROGRAM_PARTIAL.json", program)

    alloc016 = min(165.0, max(0.0, overall.minutes_left))
    result016 = None
    if alloc016 >= 75.0:
        print(f"\n=== MASTER V7: ARK-016 allocation {alloc016:.1f} min ===", flush=True)
        started = time.time()
        ark16 = import_experiment_module(16)
        subctx = RunContext(device=device, head=head, started=started, budget_minutes=alloc016)
        result016 = ark16.run_campaign(subctx)
        program["campaigns"]["ARK-016"] = {
            "status": "EXECUTED_OR_PARTIAL",
            "allocation_minutes": alloc016,
            "actual_minutes": (time.time() - started) / 60.0,
            "summary": result016.get("summary"),
        }
    else:
        program["campaigns"]["ARK-016"] = {
            "status": "BUDGET_BLOCKED",
            "allocation_minutes": alloc016,
            "remaining_master_minutes": overall.minutes_left,
        }
    writer.save("DISCOVERY_V7_PROGRAM_PARTIAL.json", program)

    decision = training_design_decision(
        result015.get("summary") if result015 else None,
        result016.get("summary") if result016 else None,
    )
    writer.save("TRAINING_DESIGN_DECISION.json", decision)

    program["status"] = "COMPLETE_OR_BUDGETED_PARTIAL"
    program["total_actual_minutes"] = overall.minutes_used
    program["training_design_decision"] = decision["decision"]
    writer.save("DISCOVERY_V7_PROGRAM_SUMMARY.json", program)
    return program


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--budget-minutes", type=float, default=DEFAULT_BUDGET_MINUTES)
    parser.add_argument("--expected-head", type=str, default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = current_device()
    head = git_head()
    print("DEVICE:", torch.cuda.get_device_name(0), "| torch", torch.__version__, flush=True)
    print("HEAD:", head, flush=True)
    print("MASTER V7 PLAN:", MASTER_PLAN_SHA, flush=True)
    print("RUNNER SHA256:", file_sha256(RUNNER_PATH), flush=True)
    if args.expected_head and head != args.expected_head:
        raise RuntimeError(f"checked-out HEAD {head} != expected {args.expected_head}")

    if args.smoke_test:
        smoke_test(device, head)
        package_all(download=False)
        return 0

    try:
        run_full(float(args.budget_minutes), device, head)
        return 0
    except Exception as exc:
        fail_ctx = RunContext(device=device, head=head, started=time.time(), budget_minutes=1)
        writer = ReceiptWriter(
            fail_ctx,
            experiment_id="MASTER_DISCOVERY_V7",
            plan_sha=MASTER_PLAN_SHA,
            runner_path=RUNNER_PATH,
        )
        writer.save("DISCOVERY_V7_FAILURE_RECEIPT.json", {
            "status": "FAILED",
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "traceback": traceback.format_exc(),
        })
        raise
    finally:
        package_all(download=True)


if __name__ == "__main__":
    raise SystemExit(main())
