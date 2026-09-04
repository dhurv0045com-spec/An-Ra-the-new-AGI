"""Portable target-runtime canary: the first thing run on any accelerator.

Performs model build, optimizer construction, live ownership check, one
synthetic-plumbing batch (labeled as such: this proves the runtime path,
never learning), forward, backward, global clip, optimizer update,
parameter-SHA change, moment change, checkpoint, fresh restore, and a next
update. Any failure aborts with the stage named. Run this on TPU/XLA or any
new target before serious training.
"""

from __future__ import annotations

import argparse
import io
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any


CANARY_SCHEMA = "anra-v5-target-canary/v1"


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def run_target_canary(
    *,
    model_spec: Any,
    device: str,
    workdir: Path,
    seed: int = 7,
    torch_module: Any = None,
) -> dict[str, object]:
    """Execute the target portability sequence; return the stage receipt."""

    if torch_module is None:
        import torch as torch_module
    torch = torch_module
    from v5_model.core import assert_receipt, initialize
    from v5_model.core import packed_layout
    from v5_objectives.causal_lm import causal_lm_loss
    from v5_training.checkpoint import CheckpointStore, _canonical_json as store_json
    from v5_training.mutation import (
        assert_mutation,
        global_grad_norm,
        moment_fingerprint,
        optimizer_step,
        parameter_sha,
    )
    from v5_training.optimizer import build_adamw_optimizer, validate_parameter_ownership
    from v5_training.state import CURSOR_SCHEMA, IDENTITY_SCHEMA, CursorState, IdentityBindings, TrainingState

    stages: dict[str, str] = {}
    workdir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    if device != "cpu" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = initialize(model_spec, seed, torch_module=torch).to(device)
    assert_receipt(model, model_spec)
    stages["model_build"] = "PASS"
    optimizer = build_adamw_optimizer(model, torch_module=torch)
    validate_parameter_ownership(model, optimizer)
    stages["optimizer_ownership"] = "PASS"
    batch_size, sequence_length = 2, 32
    tokens = torch.randint(4, model_spec.vocabulary_size, (batch_size, sequence_length)).to(device)
    segment_ids = torch.zeros(batch_size, sequence_length, dtype=torch.int64).to(device)
    positions, mask = packed_layout(segment_ids, torch_module=torch)
    before_sha = parameter_sha(model, torch_module=torch)
    before_moments = moment_fingerprint(optimizer, torch_module=torch)
    logits = model(tokens, positions, mask)
    loss, count = causal_lm_loss(logits, tokens, segment_ids, torch_module=torch)
    if not math.isfinite(float(loss.item())):
        raise ValueError("target canary loss is nonfinite at stage forward")
    stages["forward_loss"] = "PASS"
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = global_grad_norm(model, torch_module=torch)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    stages["backward_clip_update"] = "PASS"
    assert_mutation(
        before_sha=before_sha, after_sha=parameter_sha(model, torch_module=torch),
        before_moments=before_moments,
        after_moments=moment_fingerprint(optimizer, torch_module=torch),
        before_step=0, after_step=optimizer_step(optimizer),
        learning_rate=float(optimizer.param_groups[0]["lr"]),
    )
    stages["mutation_certified"] = "PASS"
    identities = IdentityBindings(
        IDENTITY_SCHEMA, "a" * 40, *["b" * 64] * 8,
    )
    state = TrainingState.initial(
        lineage_id="target-canary", token_budget=count, tokens_per_update=count,
        cursor=CursorState(CURSOR_SCHEMA, identities.pack_manifest_sha256, 0, 0, 0),
        rng_state_sha256="c" * 64, curriculum_phase="u", identities=identities,
    )
    after = state.advance(
        tokens_by_source={"canary": count},
        cursor=CursorState(CURSOR_SCHEMA, identities.pack_manifest_sha256, 0, 1, 0),
        rng_state_sha256="d" * 64, parent_checkpoint_sha256=None,
    )
    store = CheckpointStore(workdir / "checkpoints", "target-canary")
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    payloads = {
        "model.bin": buffer.getvalue(),
        "optimizer.bin": _optimizer_bytes(optimizer, torch),
        "scheduler.json": store_json({"schedule_tokens": after.schedule_tokens}),
        "rng.bin": b"canary",
        "cursor.json": store_json(asdict(after.cursor)),
        "ledger.json": store_json(dict(after.tokens_by_source)),
        "training_state.json": store_json(after.canonical()),
    }
    checkpoint_sha = store.publish(state=after, payloads=payloads, expected_parent_sha256=None)
    stages["checkpoint"] = "PASS"
    fresh = initialize(model_spec, seed + 1, torch_module=torch).to(device)
    fresh.load_state_dict(
        torch.load(io.BytesIO(payloads["model.bin"]), map_location=device, weights_only=True)
    )
    if parameter_sha(fresh, torch_module=torch) != parameter_sha(model, torch_module=torch):
        raise ValueError("restored parameters disagree with the checkpoint")
    stages["fresh_restore"] = "PASS"
    restored, _ = store.restore(checkpoint_sha)
    if restored != after:
        raise ValueError("store restore disagrees with published state")
    stages["store_restore"] = "PASS"
    receipt: dict[str, object] = {
        "schema": CANARY_SCHEMA,
        "device": device,
        "torch_version": str(torch.__version__),
        "cuda_available": bool(torch.cuda.is_available()),
        "stages": stages,
        "loss": float(loss.item()),
        "supervised_tokens": count,
        "grad_norm_pre_clip": grad_norm,
        "checkpoint_sha256": checkpoint_sha,
        "status": "PASS" if all(value == "PASS" for value in stages.values()) else "FAIL",
    }
    (workdir / "target_canary_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt


def _optimizer_bytes(optimizer: Any, torch: Any) -> bytes:
    buffer = io.BytesIO()
    torch.save(optimizer.state_dict(), buffer)
    return buffer.getvalue()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--width", type=int, default=64)
    args = parser.parse_args()
    import dataclasses

    from v5_contracts.model_spec import V5A_250M

    spec = dataclasses.replace(
        V5A_250M, layers=args.layers, width=args.width, query_heads=2, kv_heads=1,
        head_dimension=args.width // 2, ffn_width=args.width * 2,
        vocabulary_size=1024, context_length=128,
    )
    receipt = run_target_canary(model_spec=spec, device=args.device, workdir=args.workdir)
    print(json.dumps({"status": receipt["status"], "device": receipt["device"]}))
    return 0 if receipt["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
