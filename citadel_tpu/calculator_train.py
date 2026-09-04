"""Calculator training driver (experiment T1, AMENDMENT_001). Platform-neutral TPU
backend (Colab first, Kaggle secondary); Cymek production code resolves from
the pinned read-only Cymek runtime (see runtime_bootstrap).

Pipeline (single call, preregistered order): bootstrap+probe → data receipt →
untrained DEV+TEST generation baseline → dev-gated update ladder [5,20,100,200]
→ trained TEST evaluated EXACTLY ONCE at endpoint → save/destroy/reload with
prediction-hash gate → heuristic nulls → numeric gate → receipt.

Objective semantics (deliberate, A7): whole-row autoregressive CE; PAD targets
excluded; prompt tokens supervised. Receipt records real/capacity/supervised
token counts. Training loop audit (A-ordering): mark_step → forward → loss →
backward → mark_step → clip → optimizer_step → mark_step → zero_grad.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


RECEIPT_SCHEMA = "citadel-tpu-calculator-checkpoint/v1"
LADDER = (5, 20, 100, 200)
IMPROVEMENT_MARGIN = 0.10


def _encode_batch(rows: list[str], *, length: int):
    """Char-level byte encoding into a fixed [B, L] int tensor (host-side).

    Deliberately tokenizer-independent for the canary: the gate is learnability
    through the real TPU step, not the frozen BPE contract (which governs the
    later 5B corpus). Content ids are always >= 12: provably no collision with
    PAD 0 / UNK 1 / BOS 2 / EOS 3. Encoding runs on host; XLA sees fixed ints.
    """
    import torch

    from citadel_tpu import calculator_eval as cev

    ids = [cev.encode(r[:length]) for r in rows]
    real = sum(len(r) for r in ids)
    padded = [r + [cev.PAD_ID] * (length - len(r)) for r in ids]
    return torch.tensor(padded, dtype=torch.long), real


def _mean_ce(model, rows, *, device, torch_mod, batch_rows, length) -> float:
    from v5_model.core import packed_layout
    from v5_objectives.causal_lm import causal_lm_loss
    from citadel_tpu import xla_backend as xb

    torch = torch_mod
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(rows), batch_rows):
            b, _ = _encode_batch(rows[i:i + batch_rows], length=length)
            seg = torch.zeros_like(b)
            pos, mask = packed_layout(seg, torch_module=torch)  # host (audit A8)
            logits = model(b.to(device), pos.to(device), mask.to(device))
            xb.mark_step()
            loss, _ = causal_lm_loss(logits, b.to(device), seg.to(device), torch_module=torch)
            tot += float(loss.detach().to("cpu").item())
            n += 1
    model.train()
    return tot / max(n, 1)


def _train_block(model, optimizer, train_rows, *, start: int, n_updates: int,
                 device, torch_mod, batch_rows, length) -> dict[str, Any]:
    """Run n_updates sequential updates; return loss/token ledger for the block."""
    from v5_model.core import packed_layout
    from v5_objectives.causal_lm import causal_lm_loss
    from citadel_tpu import xla_backend as xb

    torch = torch_mod
    first_loss, last_loss = None, None
    cap_tokens, real_tokens, supervised = 0, 0, 0
    first_update_s = None
    for u in range(n_updates):
        idx = (start + u) % (len(train_rows) // batch_rows)
        rows = train_rows[idx * batch_rows:(idx + 1) * batch_rows]
        b, real = _encode_batch(rows, length=length)
        seg = torch.zeros_like(b)
        pos, mask = packed_layout(seg, torch_module=torch)  # host (audit A8)
        t_up = time.time()
        logits = model(b.to(device), pos.to(device), mask.to(device))
        loss, sup = causal_lm_loss(logits, b.to(device), seg.to(device), torch_module=torch)
        if not bool(torch.isfinite(loss).item()):
            raise RuntimeError(f"abort NONFINITE_LOSS at cumulative update {start + u}")
        loss.backward()
        xb.mark_step()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        xb.optimizer_step(optimizer)
        xb.mark_step()
        optimizer.zero_grad()
        if first_update_s is None:
            first_update_s = time.time() - t_up
        v = float(loss.detach().to("cpu").item())
        first_loss = v if first_loss is None else first_loss
        last_loss = v
        cap_tokens += batch_rows * length
        real_tokens += real
        supervised += int(sup)
    return {"first_loss": first_loss, "last_loss": last_loss,
            "capacity_tokens": cap_tokens, "real_tokens": real_tokens,
            "supervised_tokens": supervised, "first_update_seconds": first_update_s}


def train(*, out: str = "docs/citadel/tpu_receipts/TPU_CALCULATOR_CHECKPOINT.json",
          ladder: tuple = LADDER, batch_rows: int = 32, length: int = 32,
          lr: float = 3e-4, seed: int = 20260904, early_stop: bool = True,
          train_sample_n: int = 200,
          ckpt_name: str = "tpu_calculator_mini.pt") -> dict[str, Any]:
    from citadel_tpu import calculator_data as calc
    from citadel_tpu import calculator_eval as cev
    from citadel_tpu import checkpoint as ckpt_mod
    from citadel_tpu import environment as env_mod
    from citadel_tpu import runtime_bootstrap as rb
    from citadel_tpu import xla_backend as xb

    t0 = time.time()
    rt_root, rt_sha = rb.ensure_cymek_runtime()  # PRECHECK_IMPORT_FAILURE before any device use
    env = env_mod.probe(require_tpu=True)
    if not env.get("probe_pass"):
        raise env_mod.NoTpuError("ABORT_NO_TPU: environment probe did not pass; refusing CPU fallback.")
    n_devices = xb.assert_tpu_active(min_devices=1)
    import torch

    from anra_v5.miniature_run import MINI_SPEC
    from v5_model.core import initialize
    from v5_training.optimizer import build_adamw_optimizer

    data_receipt = cev.build_data_receipt()
    if any(v != 0 for v in data_receipt["overlap"].values()):
        raise RuntimeError(f"abort SPLIT_OVERLAP: {data_receipt['overlap']}")
    train_rows = calc.generate(split="train")
    dev_rows = calc.generate(split="development")
    test_rows = calc.generate(split="test")
    test_targets = [cev.split_prompt_target(r)[1] for r in test_rows]

    torch.manual_seed(seed)
    model = initialize(MINI_SPEC, seed)
    device = xb.get_device()
    model = model.to(device)
    param_count = sum(int(p.numel()) for p in model.parameters())
    optimizer = build_adamw_optimizer(model, torch_module=torch)
    for g in optimizer.param_groups:
        g["lr"] = lr

    dev_targets = [cev.split_prompt_target(r)[1] for r in dev_rows]
    untrained_dev_recs = cev.generate(dev_rows, model, xb, device=device, torch_mod=torch)
    untrained_dev = cev.summarize([r["prediction"] for r in untrained_dev_recs], dev_targets)
    untrained_dev_ce = _mean_ce(model, dev_rows, device=device, torch_mod=torch,
                                batch_rows=batch_rows, length=length)
    untrained_test_recs = cev.generate(test_rows, model, xb, device=device, torch_mod=torch)
    untrained_test = cev.summarize([r["prediction"] for r in untrained_test_recs], test_targets)
    untrained_test_ce = _mean_ce(model, test_rows, device=device, torch_mod=torch,
                                 batch_rows=batch_rows, length=length)
    untrained_answer_ce = cev.answer_token_ce(model, test_rows, device=device, xb=xb,
                                              torch_mod=torch)
    untrained_stop_hist = cev.stop_histogram(untrained_test_recs)
    import random as _random
    train_sample = _random.Random(seed + 1).sample(train_rows, min(train_sample_n, len(train_rows)))
    train_sample_targets = [cev.split_prompt_target(r)[1] for r in train_sample]
    untrained_train_recs = cev.generate(train_sample, model, xb, device=device, torch_mod=torch)
    untrained_train = cev.summarize([r["prediction"] for r in untrained_train_recs],
                                    train_sample_targets)

    # Dev-gated ladder (A11). TEST is never consulted for escalation.
    done_updates, endpoint = 0, 0
    rung_evals: list[dict[str, Any]] = []
    first_loss, last_loss = None, None
    cap_total, real_total, sup_total = 0, 0, 0
    first_update_s = None
    best_dev_exact, best_dev_ce = untrained_dev["accuracy"], untrained_dev_ce
    first_eval_rung = True
    t_train0 = time.time()
    for rung in ladder:
        n_new = rung - done_updates
        if n_new <= 0:
            continue
        blk = _train_block(model, optimizer, train_rows, start=done_updates, n_updates=n_new,
                           device=device, torch_mod=torch, batch_rows=batch_rows, length=length)
        done_updates = rung
        endpoint = rung
        first_loss = blk["first_loss"] if first_loss is None else first_loss
        last_loss = blk["last_loss"]
        cap_total += blk["capacity_tokens"]
        real_total += blk["real_tokens"]
        sup_total += blk["supervised_tokens"]
        if first_update_s is None:
            first_update_s = blk["first_update_seconds"]
        if rung <= 5 and early_stop:
            rung_evals.append({"rung": rung, "plumbing": "finite-loss-mutation-checked"})
            continue  # R5: plumbing only, always continue (cheap)
        dev_recs = cev.generate(dev_rows, model, xb, device=device, torch_mod=torch)
        dev = cev.summarize([r["prediction"] for r in dev_recs], dev_targets)
        dev_ce = _mean_ce(model, dev_rows, device=device, torch_mod=torch,
                          batch_rows=batch_rows, length=length)
        rung_evals.append({"rung": rung, "dev_exact": dev["accuracy"], "dev_ce": dev_ce})
        if early_stop:
            if first_eval_rung:
                if not dev_ce < untrained_dev_ce:
                    break  # no learning signal: stop, TEST runs once at endpoint below
            elif not (dev["accuracy"] > best_dev_exact or dev_ce < best_dev_ce - 1e-4):
                break
        first_eval_rung = False
        best_dev_exact = max(best_dev_exact, dev["accuracy"])
        best_dev_ce = min(best_dev_ce, dev_ce)
    train_wall = time.time() - t_train0

    trained_test_recs = cev.generate(test_rows, model, xb, device=device, torch_mod=torch)
    trained_preds = [r["prediction"] for r in trained_test_recs]
    trained_test = cev.summarize(trained_preds, test_targets)
    trained_test_ce = _mean_ce(model, test_rows, device=device, torch_mod=torch,
                               batch_rows=batch_rows, length=length)
    trained_answer_ce = cev.answer_token_ce(model, test_rows, device=device, xb=xb,
                                            torch_mod=torch)
    trained_stop_hist = cev.stop_histogram(trained_test_recs)
    trained_train_recs = cev.generate(train_sample, model, xb, device=device, torch_mod=torch)
    trained_train = cev.summarize([r["prediction"] for r in trained_train_recs],
                                  train_sample_targets)
    memorization_flag = bool(trained_train["accuracy"] - trained_test["accuracy"] >= 0.30)
    pre_sha = cev.sha_predictions(trained_preds)

    ckpt_path = str(Path(out).parent / ckpt_name)
    ckpt_hash = ckpt_mod.save(model, ckpt_path,
                              {"seed": seed, "spec": "MINI_SPEC", "updates": done_updates})
    del model
    model2 = initialize(MINI_SPEC, seed).to(device)
    ckpt_mod.load_into(model2, ckpt_path)
    xb.mark_step()
    reload_recs = cev.generate(test_rows, model2, xb, device=device, torch_mod=torch)
    reload_preds = [r["prediction"] for r in reload_recs]
    reload_test = cev.summarize(reload_preds, test_targets)
    reload_test_ce = _mean_ce(model2, test_rows, device=device, torch_mod=torch,
                              batch_rows=batch_rows, length=length)
    post_sha = cev.sha_predictions(reload_preds)
    reload_identical = bool(pre_sha == post_sha)

    nulls = cev.heuristic_nulls(test_rows, train_rows)
    null_summaries = {k: cev.summarize(v, test_targets) for k, v in nulls.items()}
    null_name, null_best = cev.strongest_null_accuracy(null_summaries)
    rules = {
        "nonoverlap": bool(trained_test["accuracy"] > untrained_test["accuracy"]
                           and trained_test["wilson_lcb"] > untrained_test["wilson_ucb"]),
        "beats_null": bool(trained_test["wilson_lcb"] > null_best),
        "margin": bool(trained_test["accuracy"] - untrained_test["accuracy"] >= IMPROVEMENT_MARGIN),
        "loss": bool(last_loss is not None and first_loss is not None
                     and last_loss < first_loss and trained_test_ce < untrained_test_ce),
        "reload": bool(reload_identical),
    }
    status = "PASS" if all(rules.values()) else "FAIL"
    wall = time.time() - t0
    receipt: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "citadel_sha": rb.citadel_sha(),
        "cymek_runtime_sha": rt_sha,
        "environment": env,
        "model": {"spec": "MINI_SPEC", "parameter_count": param_count},
        "data": data_receipt,
        "training": {"ladder": list(ladder), "endpoint_updates": done_updates,
                      "rung_evals": rung_evals, "batch_rows": batch_rows,
                      "sequence_length": length, "optimizer": "AdamW",
                      "learning_rate": lr, "seed": seed,
                      "tokens_consumed_capacity": cap_total,
                      "tokens_real": real_total,
                      "tokens_supervised": sup_total,
                      "first_loss": first_loss, "last_loss": last_loss,
                      "first_update_seconds": first_update_s,
                      "train_wall_seconds": train_wall,
                      "steady_tokens_per_second": (cap_total / train_wall) if train_wall > 0 else 0.0},
        "eval": {"untrained_dev": untrained_dev, "untrained_dev_ce": untrained_dev_ce,
                 "untrained_test": untrained_test, "untrained_test_ce": untrained_test_ce,
                 "untrained_train_sample": untrained_train,
                 "trained_test": trained_test, "trained_test_ce": trained_test_ce,
                 "trained_train_sample": trained_train,
                 "reload_test": reload_test, "reload_test_ce": reload_test_ce},
        "diagnostics": {"untrained_stop_histogram": untrained_stop_hist,
                        "trained_stop_histogram": trained_stop_hist,
                        "trained_samples": cev.sample_records(trained_test_recs, 5, seed),
                        "untrained_answer_ce": untrained_answer_ce,
                        "trained_answer_ce": trained_answer_ce,
                        "memorization_flag": memorization_flag},
        "heuristic_nulls": null_summaries,
        "strongest_heuristic_null": {"name": null_name, "accuracy": null_best},
        "gate_rules": rules,
        "pre_reload_prediction_sha256": pre_sha,
        "post_reload_prediction_sha256": post_sha,
        "reload_identical": reload_identical,
        "checkpoint": {"path": ckpt_path, "sha256": ckpt_hash,
                       "load_command": ckpt_mod.load_command(ckpt_path)},
        "device_count": n_devices,
        "wall_seconds": wall,
        "status": status,
        "interpretation": ("learning-system certification: tiny Cymek checkpoint learned "
                           "this held-out canary under this run" if status == "PASS"
                           else "no certified learning under the preregistered gate"),
    }
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
    return receipt


if __name__ == "__main__":  # device entry only; never run without a TPU
    print(json.dumps(train(), indent=2)[:500], "...")


__all__ = ["IMPROVEMENT_MARGIN", "LADDER", "train"]
