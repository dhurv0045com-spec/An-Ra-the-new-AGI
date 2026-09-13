"""Audited CS-TRANSFER-001 executable.

Supersedes the unexecuted development runner ``cs_transfer_001_run.py``.
Static review found two orchestration defects before any scientific GPU run:
(1) state certification reconstructed a synthetic pre-update state instead of
retaining the real object, and (2) resume at a development checkpoint could
advance before regenerating a missing development evaluation.  This module
keeps the preregistered science unchanged and repairs only those execution
semantics.  All non-run-arm modes delegate to the original implementation.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from typing import Any

from anra_v5 import cs_transfer_001_run as r


def _assert_trace_matches_state(trace: list[dict[str, Any]], update: int) -> None:
    observed = [int(x["update"]) for x in trace]
    expected = list(range(1, update + 1))
    if observed != expected:
        raise SystemExit(
            f"FAIL_CLOSED RESUME: durable trace {observed[-3:] if observed else []} "
            f"does not cover checkpoint 1..{update}"
        )


def _endpoint_parameter_diagnostics(backend, *, pair_index: int, arm: str) -> dict[str, Any]:
    """Measure preregistered secondary displacement diagnostics.

    This runs only after the fixed endpoint and cannot control training.  The
    matched update-zero model is regenerated from the frozen seed/constructor;
    common embedding rows and all non-embedding tensors therefore have an
    exact paired reference.  For the full arm, extra-row current/delta norms
    quantify how much the rows that do not exist in V4096 participated.
    """
    import torch

    p = r.load_prereg()
    seed = int(p["matching"]["model_seeds"][pair_index])
    pair = r.build_matched_pair(seed=seed, torch_module=torch)
    initial, _ = r.select_arm(pair, arm)
    current = dict(backend.model.named_parameters())
    start = dict(initial.named_parameters())
    if set(current) != set(start):
        raise SystemExit("FAIL_CLOSED DIAGNOSTIC: endpoint/init parameter names differ")

    shared_embed_delta_sq = 0.0
    shared_embed_init_sq = 0.0
    nonembed_delta_sq = 0.0
    nonembed_init_sq = 0.0
    extra_current_sq = 0.0
    extra_init_sq = 0.0
    extra_delta_sq = 0.0

    for name, cur in current.items():
        ini = start[name]
        if name.endswith("embedding.weight"):
            cur_common = cur[:4096].detach().float().cpu()
            ini_common = ini[:4096].detach().float().cpu()
            shared_embed_delta_sq += float(torch.sum((cur_common - ini_common) ** 2).item())
            shared_embed_init_sq += float(torch.sum(ini_common ** 2).item())
            if arm == "PHYS_24576":
                cur_extra = cur[4096:].detach().float().cpu()
                ini_extra = ini[4096:].detach().float().cpu()
                extra_current_sq += float(torch.sum(cur_extra ** 2).item())
                extra_init_sq += float(torch.sum(ini_extra ** 2).item())
                extra_delta_sq += float(torch.sum((cur_extra - ini_extra) ** 2).item())
        else:
            cur_cpu = cur.detach().float().cpu()
            ini_cpu = ini.detach().float().cpu()
            nonembed_delta_sq += float(torch.sum((cur_cpu - ini_cpu) ** 2).item())
            nonembed_init_sq += float(torch.sum(ini_cpu ** 2).item())

    def norm(x: float) -> float:
        return math.sqrt(max(0.0, x))

    shared_delta = norm(shared_embed_delta_sq)
    nonembed_delta = norm(nonembed_delta_sq)
    return {
        "schema": "anra-cs-transfer-001-parameter-diagnostics/v1",
        "shared_embedding_rows": 4096,
        "shared_embedding_l2_delta_from_init": shared_delta,
        "shared_embedding_init_l2": norm(shared_embed_init_sq),
        "shared_embedding_relative_l2_delta": (
            shared_delta / max(1e-30, norm(shared_embed_init_sq))
        ),
        "non_embedding_l2_delta_from_init": nonembed_delta,
        "non_embedding_init_l2": norm(nonembed_init_sq),
        "non_embedding_relative_l2_delta": (
            nonembed_delta / max(1e-30, norm(nonembed_init_sq))
        ),
        "extra_rows": (24576 - 4096) if arm == "PHYS_24576" else 0,
        "extra_rows_current_l2": norm(extra_current_sq) if arm == "PHYS_24576" else None,
        "extra_rows_init_l2": norm(extra_init_sq) if arm == "PHYS_24576" else None,
        "extra_rows_l2_delta_from_init": norm(extra_delta_sq) if arm == "PHYS_24576" else None,
        "post_endpoint_only": True,
        "controls_training": False,
    }


def run_arm(*, pair_index: int, arm: str, cuda: bool) -> dict[str, Any]:
    import torch

    if arm not in r.ARMS:
        raise SystemExit(f"FAIL_CLOSED: arm must be one of {r.ARMS}")
    if cuda and not torch.cuda.is_available():
        raise SystemExit("FAIL_CLOSED HARDWARE: CUDA requested but unavailable")

    p = r.load_prereg()
    prepared = r.load_prepared()
    device = torch.device("cuda") if cuda else None
    backend, init_receipt, plan = r._make_backend(
        pair_index=pair_index, arm=arm, device=device,
    )
    state, store, arm_dir, run_spec = r._state_and_store(
        pair_index=pair_index, arm=arm, prepared=prepared, backend=backend, plan=plan,
    )

    latest = store.latest_sha256()
    if latest is not None:
        restored, payloads = store.restore(latest)
        if restored.identities != state.identities:
            raise SystemExit("FAIL_CLOSED RESUME: checkpoint identity drift")
        r.base.restore_production(backend, payloads=payloads)
        state = restored
    elif r._latest_trace(arm_dir):
        raise SystemExit("FAIL_CLOSED RESUME: training trace exists without a checkpoint")

    target = int(p["training"]["target_updates"])
    checkpoint_every = int(p["training"]["checkpoint_every_updates"])
    eval_updates = set(int(x) for x in p["training"]["development_eval_updates"])
    order_seed = int(p["matching"]["order_seeds"][pair_index])
    per_epoch, epochs = r._stream_layout(prepared, order_seed=order_seed, target=target)

    # A trace may be ahead of LATEST only if the process died after the
    # pre-publication receipt and before checkpoint publication. Trim and replay.
    trace = [
        row for row in r._latest_trace(arm_dir)
        if int(row["update"]) <= int(state.global_update)
    ]
    _assert_trace_matches_state(trace, int(state.global_update))
    dev_trace = [
        item for item in r._latest_development(arm_dir)
        if int(item["update"]) <= int(state.global_update)
    ]

    # Development checkpoints are fixed prospectively. If a crash occurred
    # after checkpoint publication but before evaluation publication, evaluate
    # the RESTORED durable checkpoint before allowing another update.
    current_update = int(state.global_update)
    if current_update in eval_updates and not any(
        int(item["update"]) == current_update for item in dev_trace
    ):
        dev_trace.append({
            "update": current_update,
            "checkpoint_sha256": latest,
            "metrics": r._evaluate(backend, prepared["rows"]["development"]),
        })
        dev_trace.sort(key=lambda x: int(x["update"]))
        r._persist_dev(arm_dir, dev_trace)

    parent_sha = latest
    t0 = time.time()
    for update_index in range(current_update, target):
        epoch, position = divmod(update_index, per_epoch)
        if epoch >= len(epochs):
            raise SystemExit("FAIL_CLOSED DATA: frozen stream cache exhausted")
        window = epochs[epoch][position]
        batch = r.base.batch_from_window(
            window,
            pack_manifest_sha256=prepared["pack_manifest_sha256"],
            update_ordinal=update_index,
            device=device,
        )

        before = state
        pre_tokens = int(before.schedule_tokens)
        report = backend.step(before, batch)
        after = before.advance(
            tokens_by_source=report.tokens_by_source,
            cursor=report.cursor,
            rng_state_sha256=report.rng_state_sha256,
            parent_checkpoint_sha256=parent_sha,
        )
        r.base.certify_update(
            before=before,
            after=after,
            tokens_by_source=report.tokens_by_source,
            loss_finite=report.loss_finite,
            grad_finite=report.grad_finite,
            grad_norm_post_clip=report.grad_norm_post_clip,
            tied_preserved=report.tied_preserved,
        )
        state = after

        expected_lr = float(r.base.canary_lr_at(plan)(cumulative_tokens=pre_tokens))
        actual_lr = float(backend.optimizer.param_groups[0]["lr"])
        if actual_lr != expected_lr:
            raise SystemExit(
                f"FAIL_CLOSED SCHEDULE: update {state.global_update} "
                f"expected {expected_lr}, got {actual_lr}"
            )

        trace.append({
            "update": int(state.global_update),
            "tokens_seen": int(state.cumulative_tokens),
            "epoch": int(epoch),
            "epoch_update": int(position + 1),
            "loss": float(backend.last_receipt["loss"]),
            "grad_norm_post_clip": float(report.grad_norm_post_clip),
            "lr_expected": expected_lr,
            "lr_actual": actual_lr,
        })
        _assert_trace_matches_state(trace, int(state.global_update))

        due_checkpoint = int(state.global_update) % checkpoint_every == 0
        due_eval = int(state.global_update) in eval_updates
        if due_eval and not due_checkpoint:
            raise SystemExit(
                "FAIL_CLOSED CONTRACT: every development evaluation must be a checkpoint boundary"
            )
        if due_checkpoint:
            # Receipt precedes publication; if publish fails, resume trims it.
            r._persist_training(arm_dir, trace, "IN_PROGRESS")
            published = store.publish(
                state=state,
                payloads=r.base.production_payloads(backend, state=state),
                expected_parent_sha256=parent_sha,
            )
            parent_sha = published
            trace[-1]["checkpoint_sha256"] = published
        if due_eval and not any(
            int(item["update"]) == int(state.global_update) for item in dev_trace
        ):
            dev_trace.append({
                "update": int(state.global_update),
                "checkpoint_sha256": parent_sha,
                "metrics": r._evaluate(backend, prepared["rows"]["development"]),
            })
            dev_trace.sort(key=lambda x: int(x["update"]))
            r._persist_dev(arm_dir, dev_trace)

    expected_dev = sorted(eval_updates)
    observed_dev = sorted(int(x["update"]) for x in dev_trace)
    if observed_dev != expected_dev:
        raise SystemExit(
            f"FAIL_CLOSED EVALUATION: expected dev checkpoints {expected_dev}, got {observed_dev}"
        )
    if int(state.global_update) != target or int(state.cumulative_tokens) != int(p["training"]["token_budget"]):
        raise SystemExit("FAIL_CLOSED ENDPOINT: update/token endpoint mismatch")

    r._persist_training(arm_dir, trace, "COMPLETE_ENDPOINT")
    r._persist_dev(arm_dir, dev_trace)
    parameter_diagnostics = _endpoint_parameter_diagnostics(
        backend, pair_index=pair_index, arm=arm,
    )
    r._write_receipt("ARM_RESULT", {
        "schema": "anra-cs-transfer-001-arm-result/v2",
        "pair_index": pair_index,
        "arm": arm,
        "run_spec": run_spec,
        "matched_init": init_receipt,
        "global_update": int(state.global_update),
        "cumulative_tokens": int(state.cumulative_tokens),
        "checkpoint_sha256": store.latest_sha256(),
        "development_trace_updates": observed_dev,
        "parameter_diagnostics": parameter_diagnostics,
        "wall_seconds_this_invocation": round(time.time() - t0, 3),
        "status": "COMPLETE",
        "runner_revision": "v2",
    }, arm_dir=arm_dir)
    return {
        "pair_index": pair_index,
        "arm": arm,
        "status": "COMPLETE",
        "global_update": int(state.global_update),
        "checkpoint": store.latest_sha256(),
        "development_trace_updates": observed_dev,
        "parameter_diagnostics": parameter_diagnostics,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True,
                        choices=("prepare", "preflight", "scan", "run-arm", "development", "finalize"))
    parser.add_argument("--pair-index", type=int, default=0)
    parser.add_argument("--arm", choices=r.ARMS, default="PHYS_4096")
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()
    if args.mode == "prepare":
        result = r.prepare_data()["receipt"]
    elif args.mode == "preflight":
        result = r.preflight(cuda=args.cuda)
    elif args.mode == "scan":
        result = r.scan()
    elif args.mode == "run-arm":
        result = run_arm(pair_index=args.pair_index, arm=args.arm, cuda=args.cuda)
    elif args.mode == "development":
        result = r.development_aggregate()
    else:
        result = r.finalize(cuda=args.cuda)
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
