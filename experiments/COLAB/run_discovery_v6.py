"""Bounded integrity preflight and explicit single-campaign entry point.

CPU preflight is the default. Training requires --campaign and CUDA; no implicit
multi-hour run or experimental success is inferred from a passing preflight.
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
import statistics
import time
import traceback

import torch

from discovery_v6_common import (
    BudgetExhausted, MASTER_PLAN_SHA, ReceiptWriter, RunContext,
    bind_ark11_runtime, current_device, generate_indices, git_head,
    import_experiment_module, load_ark11, order_sha256, package_all,
    parameter_sha,
)


def tree_equal(left, right) -> bool:
    if torch.is_tensor(left):
        return torch.is_tensor(right) and torch.equal(left.detach().cpu(), right.detach().cpu())
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(tree_equal(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(tree_equal(a, b) for a, b in zip(left, right))
    return left == right


def preflight(ctx: RunContext) -> dict:
    """Exercise the real Micro optimizer fork, manifests, and sampler on CPU/GPU."""
    ark11 = load_ark11()
    bind_ark11_runtime(ark11, ctx.device, ctx.head, ctx=ctx)
    t2 = ark11.load_manifest()
    control, sealed, split = ark11.build_control_sealed_split(t2["test"])
    assert control and sealed and not set(control) & set(sealed)
    assert set(control) | set(sealed) == {tuple(row) for row in t2["test"]}
    ark13 = import_experiment_module(13)
    train3, control3, sealed3, manifest3 = ark13.build_t3carry_manifest()
    assert manifest3 == ark13.build_t3carry_manifest()[3]
    assert not set(control3) & set(sealed3)
    old_pairs = {tuple(sorted(ark13._pair_from_prompt(p))) for p, _ in t2["train"]}
    new_ood_pairs = {tuple(sorted(ark13._pair_from_prompt(p))) for p, _ in control3 + sealed3}
    assert not old_pairs & new_ood_pairs

    # No trained checkpoint: these are a few correctness updates from random weights.
    torch.manual_seed(9191)
    vocab = ark11.CompactVocab()
    model = ark11.Micro(vocab.size, 128).to(ctx.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95),
                                 eps=1e-8, weight_decay=0.1)
    rows = [tuple(row) for row in t2["train"][:8]]
    count = ark11.train_step(model, optimizer, vocab, rows)
    snapshot = ark11.snapshot_state(model, optimizer)
    original = copy.deepcopy(snapshot)
    fork_hashes = []
    for _ in range(2):
        fork_vocab, fork_model, fork_optimizer = ark11.load_fork(snapshot, 1e-3)
        assert parameter_sha(fork_model) == parameter_sha(model)
        assert torch.equal(torch.get_rng_state(), snapshot["torch_rng"])
        assert tree_equal(fork_optimizer.state_dict(), snapshot["optimizer"])
        ark11.train_step(fork_model, fork_optimizer, fork_vocab, rows)
        fork_hashes.append(parameter_sha(fork_model))
        assert tree_equal(snapshot, original), "fork mutated its source snapshot"
    assert fork_hashes[0] == fork_hashes[1], "matched next updates diverged"
    # A different LR must actually change the next update.
    low_vocab, low_model, low_optimizer = ark11.load_fork(snapshot, 1e-5)
    ark11.train_step(low_model, low_optimizer, low_vocab, rows)
    assert parameter_sha(low_model) != fork_hashes[0]
    assert tree_equal(snapshot, original)

    samples, batches, batch_size, pool_size = 5, 18000, 64, 500
    def legacy():
        rng = torch.Generator().manual_seed(6801)
        return [torch.randint(0, pool_size, (batch_size,), generator=rng).tolist()
                for _ in range(batches)]
    expected = legacy()
    actual = generate_indices(6801, batches, batch_size, pool_size)
    assert actual == expected == ark11.generate_continuation_indices(6801, batches, batch_size, pool_size)
    assert actual != generate_indices(6802, batches, batch_size, pool_size)
    old_times, new_times = [], []
    for trial in range(samples):
        # Alternate order to reduce warmup/order bias; tensor construction + tolist included.
        pairs = [(legacy, old_times), (lambda: generate_indices(6801, batches, batch_size, pool_size), new_times)]
        for fn, timings in (pairs if trial % 2 == 0 else pairs[::-1]):
            started = time.perf_counter()
            value = fn()
            timings.append(time.perf_counter() - started)
            del value
    return {
        "status": "PASS", "scope": "bounded engineering integrity; no capability evidence",
        "t2_manifest_sha256": t2["split_sha256"],
        "t2_assignment_sha256": split["assignment_sha256"],
        "t3_manifest_sha256": manifest3["manifest_sha256"],
        "t3_counts": manifest3["counts"],
        "train_ood_commutation_overlap": 0,
        "snapshot_immutable_after_forks": True, "optimizer_restore_exact": True,
        "torch_rng_restore_exact": True, "matched_next_update_sha256": fork_hashes[0],
        "different_lr_changes_update": True, "supervised_positions_initial_update": count,
        "sampler": {"seed": 6801, "batches": batches, "batch_size": batch_size,
                    "pool_size": pool_size, "identical_legacy_stream": True,
                    "order_sha256": order_sha256(actual), "repetitions": samples,
                    "legacy_seconds": old_times, "vectorized_seconds": new_times,
                    "median_speedup": statistics.median(old_times) / statistics.median(new_times)},
        "unverified": ["CUDA runtime" if ctx.device.type == "cpu" else "CPU runtime",
                       "full training horizons", "cognition gain", "ARK-014", "full V6 orchestration",
                       "durable checkpoint restart"],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", choices=["ARK-012", "ARK-013"],
                        help="explicitly run one CUDA campaign instead of the default CPU preflight")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--budget-minutes", type=float, default=None)
    parser.add_argument("--output-dir", type=Path, help="must be a new directory")
    args = parser.parse_args(argv)
    if args.campaign and args.device != "cuda":
        parser.error("training campaigns require --device cuda; CPU mode is a bounded preflight")
    budget = args.budget_minutes if args.budget_minutes is not None else 2.0
    minimum = {"ARK-012": 35, "ARK-013": 50}
    if args.campaign and budget < minimum[args.campaign]:
        parser.error(f"{args.campaign} needs an explicit budget of at least {minimum[args.campaign]} minutes")
    ctx = RunContext(torch.device(args.device), git_head(), time.time(), budget, args.output_dir)
    writer = ReceiptWriter(ctx, experiment_id=args.campaign or "V6_PREFLIGHT",
                           plan_sha=MASTER_PLAN_SHA, runner_path=Path(__file__))
    result = 1
    try:
        if args.device == "cuda":
            current_device()
        else:
            torch.set_num_threads(1)
        writer.save("PREFLIGHT.json", preflight(ctx))
        if args.campaign:
            module = import_experiment_module(int(args.campaign[-3:]))
            module.run_campaign(ctx)
        result = 0
    except (Exception, KeyboardInterrupt) as exc:
        writer.save("FAILURE_RECEIPT.json", {
            "status": "BUDGET_EXHAUSTED" if isinstance(exc, BudgetExhausted) else "FAILED",
            "exception_type": type(exc).__name__, "exception": str(exc),
            "traceback": traceback.format_exc(),
        })
        print(f"FAILED: {type(exc).__name__}: {exc}", flush=True)
    finally:
        package_all(download=False, ctx=ctx)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
