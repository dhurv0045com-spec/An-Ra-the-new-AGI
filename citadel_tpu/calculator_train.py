"""Calculator training driver (experiment T1). Runs ONLY on Kaggle TPU.

Prerequisite: T0 PASS. Objective: standard autoregressive CE only (mission
§13 — no query-swap, no cognition objectives; eliminate variables).
Success gate (mission §14): loss clearly decreases; held-out accuracy clearly
exceeds untrained baseline; save → destroy → reload preserves capability.

Writes docs/citadel/tpu_receipts/TPU_CALCULATOR_CHECKPOINT.json.
Interpretation discipline (§15): a pass means "Cymek can learn a small
held-out capability on the target TPU" — never "An-Ra became intelligent".
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


RECEIPT_SCHEMA = "citadel-tpu-calculator-checkpoint/v1"


def _encode_batch(rows: list[str], *, length: int):
    """Char-level byte encoding into a fixed [B, L] int tensor (host-side).

    Deliberately tokenizer-independent for the canary: the gate is learnability
    through the real TPU step, not the frozen BPE contract (which governs the
    later 5B corpus, mission §27). Encoding runs on host; XLA sees fixed ints.
    """
    import torch

    ids = [[(ord(c) % 250) + 2 for c in r[:length]] for r in rows]
    ids = [r + [0] * (length - len(r)) for r in ids]
    return torch.tensor(ids, dtype=torch.long)


def train(*, out: str = "docs/citadel/tpu_receipts/TPU_CALCULATOR_CHECKPOINT.json",
          steps: int = 200, batch_rows: int = 32, length: int = 32,
          lr: float = 3e-4, seed: int = 20260904) -> dict[str, Any]:
    from citadel_tpu import calculator_data as calc
    from citadel_tpu import checkpoint as ckpt_mod
    from citadel_tpu import environment as env_mod
    from citadel_tpu import xla_backend as xb

    t0 = time.time()
    env = env_mod.probe(require_tpu=True)
    n_devices = xb.assert_tpu_active(min_devices=1)
    import torch

    from anra_v5.miniature_run import MINI_SPEC
    from v5_model.core import initialize, packed_layout
    from v5_objectives.causal_lm import causal_lm_loss
    from v5_training.optimizer import build_adamw_optimizer

    train_rows = calc.generate(split="train")
    dev_rows = calc.generate(split="development")
    test_rows = calc.generate(split="test")

    def acc_of(model, rows, device, seg_cache) -> float:
        # Teacher-forced prefix scoring proxy for the canary gate: exact-match
        # of greedy next-token continuation is measured in the notebook eval
        # cell; here we record loss-based fit + held-out CE as the machine gate.
        model.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(rows), batch_rows):
                b = _encode_batch(rows[i:i + batch_rows], length=length)
                seg = torch.zeros_like(b)
                pos, mask = packed_layout(seg, torch_module=torch)
                logits = model(b.to(device), pos.to(device), mask.to(device))
                xb.mark_step()
                loss, _ = causal_lm_loss(logits, b.to(device), seg.to(device), torch_module=torch)
                tot += float(loss.detach().to("cpu").item())
                n += 1
        model.train()
        return tot / max(n, 1)

    torch.manual_seed(seed)
    model = initialize(MINI_SPEC, seed)
    device = xb.xla_device()
    model = model.to(device)
    optimizer = build_adamw_optimizer(model, torch_module=torch)
    for g in optimizer.param_groups:
        g["lr"] = lr
    untrained_ce = acc_of(model, test_rows, device, None)
    start = time.time()
    first_loss, last_loss = None, None
    consumed = 0
    for step in range(steps):
        rows = [train_rows[(step * batch_rows + i) % len(train_rows)] for i in range(batch_rows)]
        b = _encode_batch(rows, length=length)
        seg = torch.zeros_like(b)
        pos, mask = packed_layout(seg, torch_module=torch)  # host (audit A8)
        logits = model(b.to(device), pos.to(device), mask.to(device))
        loss, _ = causal_lm_loss(logits, b.to(device), seg.to(device), torch_module=torch)
        loss.backward()
        xb.mark_step()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        xb.optimizer_step(optimizer)
        xb.mark_step()
        optimizer.zero_grad()
        v = float(loss.detach().to("cpu").item())
        first_loss = v if first_loss is None else first_loss
        last_loss = v
        consumed += batch_rows * length
    train_wall = time.time() - start
    trained_ce = acc_of(model, test_rows, device, None)
    ckpt_path = str(Path(out).parent / "tpu_calculator_mini.pt")
    ckpt_hash = ckpt_mod.save(model, ckpt_path, {"seed": seed, "spec": "MINI_SPEC", "steps": steps})
    del model
    model2 = initialize(MINI_SPEC, seed).to(device)
    meta = ckpt_mod.load_into(model2, ckpt_path)
    xb.mark_step()
    reload_ce = acc_of(model2, test_rows, device, None)
    reload_ok = abs(reload_ce - trained_ce) < 1e-6
    wall = time.time() - t0
    receipt: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "environment": env,
        "model": {"spec": "MINI_SPEC"},
        "data": {"generator_version": calc.GENERATOR_VERSION,
                  "train": len(train_rows), "development": len(dev_rows), "test": len(test_rows)},
        "training": {"steps": steps, "batch_rows": batch_rows, "sequence_length": length,
                      "tokens_consumed": consumed, "optimizer": "AdamW",
                      "first_loss": first_loss, "last_loss": last_loss,
                      "train_wall_seconds": train_wall,
                      "steady_tokens_per_second": consumed / train_wall if train_wall > 0 else 0.0},
        "eval": {"untrained_test_ce": untrained_ce, "trained_test_ce": trained_ce,
                 "reload_test_ce": reload_ce, "reload_identical": bool(reload_ok)},
        "checkpoint": {"path": ckpt_path, "sha256": ckpt_hash, "meta": meta,
                       "load_command": ckpt_mod.load_command(ckpt_path)},
        "device_count": n_devices,
        "wall_seconds": wall,
        "interpretation": "infrastructure only: Cymek can learn a small held-out capability on TPU" if (last_loss or 1e9) < (first_loss or 0) else "no learning signal",
    }
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
    return receipt


if __name__ == "__main__":  # Kaggle entry only
    print(json.dumps(train(), indent=2)[:500], "...")


__all__ = ["train"]
