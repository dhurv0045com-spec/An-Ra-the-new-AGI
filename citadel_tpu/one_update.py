"""TPU one-update certification driver (experiment T0). Runs ONLY on Kaggle TPU.

Chain (tiny model, fixed XLA batch, bucket 512):
  env probe → MINI_SPEC init on CPU → move to XLA → tiny token batch
  → packed_layout on HOST → forward → CE loss → backward → finite grads
  → xm.optimizer_step + xm.mark_step → params-changed verify
  → checkpoint save → reload → identical inference.

Writes docs/citadel/tpu_receipts/TPU_ONE_UPDATE.json. Any gate failure raises
IMPLEMENTATION_FAILURE. CUDA/CPU execution is refused, not downgraded.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any


BUCKET = 512
RECEIPT_SCHEMA = "citadel-tpu-one-update/v1"


def _sha256_tensor(t) -> str:
    import hashlib as _hl

    h = _hl.sha256()
    h.update(str(tuple(t.shape)).encode("ascii"))
    h.update(bytes(t.detach().to("cpu").float().contiguous().numpy().tobytes()))
    return h.hexdigest()


def model_param_count(model) -> int:
    return sum(int(p.numel()) for p in model.parameters())


def model_sha256(model) -> str:
    import hashlib as _hl

    h = _hl.sha256()
    for name, p in sorted(model.named_parameters()):
        h.update(name.encode())
        h.update(bytes(p.detach().to("cpu").float().contiguous().numpy().tobytes()))
    return h.hexdigest()


def run(*, out: str = "docs/citadel/tpu_receipts/TPU_ONE_UPDATE.json", seed: int = 20260904) -> dict[str, Any]:
    from citadel_tpu import environment as env_mod
    from citadel_tpu import xla_backend as xb

    t0 = time.time()
    env = env_mod.probe(require_tpu=True)  # raises ABORT_NO_TPU on CPU fallback
    n_devices = xb.assert_tpu_active(min_devices=1)
    import torch

    from anra_v5.miniature_run import MINI_SPEC
    from v5_model.config import from_spec
    from v5_contracts.model_spec import QK_NORM_EPSILON
    from v5_model.core import initialize, packed_layout
    from v5_objectives.causal_lm import causal_lm_loss
    from v5_training.optimizer import build_adamw_optimizer

    config = from_spec(MINI_SPEC, qk_norm_epsilon=QK_NORM_EPSILON)
    torch.manual_seed(seed)
    model = initialize(MINI_SPEC, seed)  # CPU init (host-only per audit A15)
    device = xb.xla_device()
    model = model.to(device)
    param_count = model_param_count(model)
    before_sha = model_sha256(model)

    # Tiny fixed batch, bucket 512, host-side layout (audit A8: never on XLA).
    batch, length = 2, BUCKET
    tokens_cpu = torch.randint(0, MINI_SPEC.vocabulary_size, (batch, length))
    seg_cpu = torch.zeros((batch, length), dtype=torch.int64)
    positions_cpu, mask_cpu = packed_layout(seg_cpu, torch_module=torch)
    input_hash = hashlib.sha256(bytes(tokens_cpu.contiguous().numpy().tobytes())).hexdigest()
    tokens, positions, mask = tokens_cpu.to(device), positions_cpu.to(device), mask_cpu.to(device)

    optimizer = build_adamw_optimizer(model, torch_module=torch)
    steps_before = [int(g.get("step", 0)) if isinstance(g, dict) else 0 for g in optimizer.param_groups]
    xb.mark_step()
    logits = model(tokens, positions, mask)
    loss, supervised = causal_lm_loss(logits, tokens, seg_cpu.to(device), torch_module=torch)
    if not bool(torch.isfinite(loss).item()):
        raise RuntimeError("abort NONFINITE_LOSS")
    loss.backward()
    xb.mark_step()
    total_sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_sq += float(p.grad.detach().float().pow(2).sum().to("cpu").item())
    grad_norm = total_sq**0.5
    if not (grad_norm == grad_norm and grad_norm < float("inf")):
        raise RuntimeError("abort NONFINITE_GRADIENT")
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    lr = float(optimizer.param_groups[0]["lr"])
    xb.optimizer_step(optimizer)  # xm.optimizer_step, single step
    xb.mark_step()
    after_sha = model_sha256(model)
    if after_sha == before_sha:
        raise RuntimeError("abort NO_PARAM_CHANGE: optimizer step did not mutate parameters")

    # Checkpoint save → destroy → reload → identical inference (host-side format).
    from citadel_tpu import checkpoint as ckpt_mod

    ckpt_path = str(Path(out).parent / "tpu_one_update_mini.pt")
    ckpt_hash = ckpt_mod.save(model, ckpt_path, {"seed": seed, "spec": "MINI_SPEC"})
    with torch.no_grad():
        ref_logits = model(tokens, positions, mask).detach().to("cpu")
    del model
    model2 = initialize(MINI_SPEC, seed).to(device)
    ckpt_mod.load_into(model2, ckpt_path)
    xb.mark_step()
    with torch.no_grad():
        new_logits = model2(tokens, positions, mask).detach().to("cpu")
    reload_identical = bool(torch.equal(ref_logits, new_logits))
    if not reload_identical:
        raise RuntimeError("abort RELOAD_MISMATCH")
    wall = time.time() - t0
    real_tokens = batch * length
    receipt: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "environment": env,
        "model": {"spec": "MINI_SPEC", "parameter_count": param_count},
        "initial_parameter_sha256": before_sha,
        "final_parameter_sha256": after_sha,
        "input_token_sha256": input_hash,
        "loss": float(loss.detach().to("cpu").item()),
        "supervised_tokens": int(supervised),
        "grad_norm_pre_clip": grad_norm,
        "learning_rate": lr,
        "device_count": n_devices,
        "wall_seconds": wall,
        "tokens_processed": real_tokens,
        "tokens_per_second": real_tokens / wall if wall > 0 else 0.0,
        "checkpoint_sha256": ckpt_hash,
        "reload_identical": reload_identical,
        "certification": "PASS",
    }
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
    return receipt


if __name__ == "__main__":  # Kaggle entry only; never run locally without TPU
    print(json.dumps(run(), indent=2)[:500], "...")


__all__ = ["BUCKET", "model_sha256", "run"]
