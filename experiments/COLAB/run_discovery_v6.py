from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments" / "COLAB"))

from discovery_v6_common import (
    MASTER_PLAN_SHA,
    ReceiptWriter,
    RunContext,
    bind_ark11_runtime,
    current_device,
    file_sha256,
    git_head,
    import_experiment_module,
    load_ark11,
    package_all,
    parameter_sha,
)

RUNNER_PATH = Path(__file__)
DEFAULT_BUDGET_MINUTES = 240


def smoke_test(device: torch.device, head: str) -> dict:
    print("\n=== DISCOVERY V6 GPU SMOKE TEST ===", flush=True)
    ctx = RunContext(device=device, head=head, started=time.time(), budget_minutes=20)
    writer = ReceiptWriter(
        ctx,
        experiment_id="MASTER_DISCOVERY_V6_SMOKE",
        plan_sha=MASTER_PLAN_SHA,
        runner_path=RUNNER_PATH,
    )

    ark11 = load_ark11()
    bind_ark11_runtime(ark11, device, head)
    manifest = ark11.load_manifest()
    train = [(p, a) for p, a in manifest["train"]]
    test = [(p, a) for p, a in manifest["test"]]
    control, sealed, split = ark11.build_control_sealed_split(test)
    assert control and sealed and not (set(control) & set(sealed))
    assert set(control) | set(sealed) == set(test)

    ark13 = import_experiment_module(13)
    t3_train, t3_control, t3_sealed, t3_manifest = ark13.build_t3carry_manifest()
    assert len(t3_train) == 500 and len(t3_control) + len(t3_sealed) == 200
    assert t3_manifest["commutation_overlap"] == 0
    for prompt, _ in t3_train[:50] + t3_control[:25] + t3_sealed[:25]:
        a, b = ark13._pair_from_prompt(prompt)
        assert a % 10 + b % 10 >= 10

    ark14 = import_experiment_module(14)
    train_meta, bind_control, bind_sealed, bind_manifest = ark14.build_binding_manifest()
    assert len(train_meta) == 1200
    assert bind_manifest["factset_overlap_train_test"] == 0
    assert bind_manifest["factset_overlap_control_sealed"] == 0
    sample = train_meta[7]
    p1 = ark14.augmented_facts(sample, 2201, 17, 3)
    p2 = ark14.augmented_facts(sample, 2201, 17, 3)
    assert p1 == p2
    for variants in [bind_control, bind_sealed]:
        assert len(variants["canonical"]) == len(variants["order_only"]) == len(variants["query_order"])
        for i in range(min(20, len(variants["canonical"]))):
            assert variants["canonical"][i][1].isdigit()
            assert variants["order_only"][i][1].isdigit()
            assert variants["query_order"][i][1].isdigit()

    vocab = ark11.CompactVocab()
    torch.manual_seed(60606)
    torch.cuda.manual_seed_all(60606)
    model = ark11.Micro(vocab.size, 128).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1
    )
    rows = train[:8]
    loss, count = ark11.loss_and_positions(model, vocab, rows, device)
    assert torch.isfinite(loss) and int(count.item()) > 0
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    snap = ark11.snapshot_state(model, optimizer)
    _, restored, _ = ark11.load_fork(snap, 1e-3)
    assert ark11.model_state_equal(model.state_dict(), restored.state_dict())

    v1, m1, o1 = ark11.load_fork(snap, 1e-3)
    v2, m2, o2 = ark11.load_fork(snap, 1e-3)
    ark11.train_step(m1, o1, v1, rows)
    ark11.train_step(m2, o2, v2, rows)
    assert parameter_sha(m1) == parameter_sha(m2)

    payload = {
        "status": "PASS",
        "canonical_t2_sha256": manifest["split_sha256"],
        "t2_control_sha256": split["control_sha256"],
        "t2_sealed_sha256": split["sealed_sha256"],
        "t3_manifest_sha256": t3_manifest["manifest_sha256"],
        "binding_manifest_sha256": bind_manifest["manifest_sha256"],
        "gpu_forward_backward": True,
        "snapshot_reload_exact": True,
        "equal_next_step_parameter_hash": True,
        "order_augmentation_reproducible": True,
    }
    writer.save("DISCOVERY_V6_SMOKE_TEST.json", payload)
    print("DISCOVERY V6 GPU SMOKE TEST PASS", flush=True)
    return payload


def run_full(total_budget_minutes: float, device: torch.device, head: str) -> dict:
    overall = RunContext(device=device, head=head, started=time.time(), budget_minutes=total_budget_minutes)
    writer = ReceiptWriter(
        overall,
        experiment_id="MASTER_DISCOVERY_V6",
        plan_sha=MASTER_PLAN_SHA,
        runner_path=RUNNER_PATH,
    )
    program = {
        "status": "RUNNING",
        "total_budget_minutes": total_budget_minutes,
        "campaigns": {},
    }

    def overall_left() -> float:
        return overall.minutes_left

    reserve_after_011 = 35 + 50 + 40
    alloc011 = max(0.0, min(105.0, overall_left() - reserve_after_011))
    if alloc011 >= 45:
        print(f"\n=== MASTER: ARK-011 allocation {alloc011:.1f} min ===", flush=True)
        started = time.time()
        ark11 = load_ark11()
        bind_ark11_runtime(ark11, device, head)
        result011 = ark11.full_campaign(int(alloc011))
        ark11.package_results(download=False)
        program["campaigns"]["ARK-011"] = {
            "status": "EXECUTED_OR_PARTIAL",
            "allocation_minutes": alloc011,
            "actual_minutes": (time.time() - started) / 60.0,
            "summary": result011.get("summary"),
        }
    else:
        program["campaigns"]["ARK-011"] = {"status": "BUDGET_BLOCKED", "allocation_minutes": alloc011}
    writer.save("DISCOVERY_V6_PROGRAM_PARTIAL.json", program)

    if overall_left() >= 35:
        reserve = 50 + 40
        alloc012 = max(35.0, min(55.0, overall_left() - reserve)) if overall_left() > reserve else 0.0
        if alloc012 >= 35:
            print(f"\n=== MASTER: ARK-012 allocation {alloc012:.1f} min ===", flush=True)
            started = time.time()
            ark12 = import_experiment_module(12)
            subctx = RunContext(device=device, head=head, started=started, budget_minutes=alloc012)
            result012 = ark12.run_campaign(subctx)
            program["campaigns"]["ARK-012"] = {
                "status": "EXECUTED_OR_PARTIAL",
                "allocation_minutes": alloc012,
                "actual_minutes": (time.time() - started) / 60.0,
                "summary": result012.get("summary"),
            }
        else:
            program["campaigns"]["ARK-012"] = {"status": "BUDGET_BLOCKED", "allocation_minutes": alloc012}
    else:
        program["campaigns"]["ARK-012"] = {"status": "BUDGET_BLOCKED", "remaining_minutes": overall_left()}
    writer.save("DISCOVERY_V6_PROGRAM_PARTIAL.json", program)

    if overall_left() >= 50:
        alloc013 = max(50.0, min(75.0, overall_left() - 40)) if overall_left() > 40 else 0.0
        if alloc013 >= 50:
            print(f"\n=== MASTER: ARK-013 allocation {alloc013:.1f} min ===", flush=True)
            started = time.time()
            ark13 = import_experiment_module(13)
            subctx = RunContext(device=device, head=head, started=started, budget_minutes=alloc013)
            result013 = ark13.run_campaign(subctx)
            program["campaigns"]["ARK-013"] = {
                "status": "EXECUTED_OR_PARTIAL",
                "allocation_minutes": alloc013,
                "actual_minutes": (time.time() - started) / 60.0,
                "summary": result013.get("summary"),
            }
        else:
            program["campaigns"]["ARK-013"] = {"status": "BUDGET_BLOCKED", "allocation_minutes": alloc013}
    else:
        program["campaigns"]["ARK-013"] = {"status": "BUDGET_BLOCKED", "remaining_minutes": overall_left()}
    writer.save("DISCOVERY_V6_PROGRAM_PARTIAL.json", program)

    if overall_left() >= 40:
        alloc014 = overall_left()
        print(f"\n=== MASTER: ARK-014 allocation {alloc014:.1f} min ===", flush=True)
        started = time.time()
        ark14 = import_experiment_module(14)
        subctx = RunContext(device=device, head=head, started=started, budget_minutes=alloc014)
        result014 = ark14.run_campaign(subctx)
        program["campaigns"]["ARK-014"] = {
            "status": "EXECUTED_OR_PARTIAL",
            "allocation_minutes": alloc014,
            "actual_minutes": (time.time() - started) / 60.0,
            "summary": result014.get("summary"),
        }
    else:
        program["campaigns"]["ARK-014"] = {"status": "BUDGET_BLOCKED", "remaining_minutes": overall_left()}

    program["status"] = "COMPLETE_OR_BUDGETED_PARTIAL"
    program["total_actual_minutes"] = overall.minutes_used
    writer.save("DISCOVERY_V6_PROGRAM_SUMMARY.json", program)
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
    print("MASTER PLAN:", MASTER_PLAN_SHA, flush=True)
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
            experiment_id="MASTER_DISCOVERY_V6",
            plan_sha=MASTER_PLAN_SHA,
            runner_path=RUNNER_PATH,
        )
        writer.save("DISCOVERY_V6_FAILURE_RECEIPT.json", {
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
