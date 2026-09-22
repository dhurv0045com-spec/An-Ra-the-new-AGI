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
import time
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
    batch_size: int = 2,
    sequence_length: int = 32,
    verify_checkpoint: bool = True,
    verify_continuation: bool = True,
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

    if batch_size <= 0 or sequence_length < 2 or sequence_length > model_spec.context_length:
        raise ValueError("canary batch/sequence dimensions exceed the model contract")
    if not isinstance(verify_continuation, bool):
        raise ValueError("verify_continuation must be boolean")
    if not isinstance(verify_checkpoint, bool):
        raise ValueError("verify_checkpoint must be boolean")
    if verify_continuation and not verify_checkpoint:
        raise ValueError("continuation verification requires checkpoint verification")

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
    tokens = torch.randint(4, model_spec.vocabulary_size, (batch_size, sequence_length)).to(device)
    segment_ids = torch.zeros(batch_size, sequence_length, dtype=torch.int64).to(device)
    positions, mask = packed_layout(segment_ids, torch_module=torch)
    model.train()
    before_sha = parameter_sha(model, torch_module=torch)
    before_moments = moment_fingerprint(optimizer, torch_module=torch)
    update_started = time.perf_counter()
    logits = model(tokens, positions, mask, use_activation_checkpointing=True)
    loss, count = causal_lm_loss(logits, tokens, segment_ids, torch_module=torch)
    if not math.isfinite(float(loss.item())):
        raise ValueError("target canary loss is nonfinite at stage forward")
    stages["forward_loss"] = "PASS"
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = global_grad_norm(model, torch_module=torch)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    runtime = str(getattr(device, "type", device))
    if runtime == "xla":
        try:
            import torch_xla.core.xla_model as xm
        except ImportError as exc:
            raise RuntimeError("XLA target canary requires torch_xla") from exc
        xm.mark_step()
    update_seconds = time.perf_counter() - update_started
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
    checkpoint_sha = None
    after = None
    if verify_checkpoint:
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
        portable_model_state = _to_cpu(model.state_dict())
        torch.save(portable_model_state, buffer)
        rng_state: dict[str, Any] = {"cpu": torch.get_rng_state()}
        if torch.cuda.is_available():
            rng_state["cuda"] = torch.cuda.get_rng_state_all()
        payloads = {
            "model.bin": buffer.getvalue(),
            "optimizer.bin": _optimizer_bytes(optimizer, torch),
            "scheduler.json": store_json({"schedule_tokens": after.schedule_tokens}),
            "rng.bin": _torch_bytes(rng_state, torch),
            "cursor.json": store_json(asdict(after.cursor)),
            "ledger.json": store_json(dict(after.tokens_by_source)),
            "training_state.json": store_json(after.canonical()),
        }
        checkpoint_sha = store.publish(state=after, payloads=payloads, expected_parent_sha256=None)
        stages["checkpoint"] = "PASS"
        fresh = initialize(model_spec, seed + 1, torch_module=torch).to(device)
        fresh_optimizer = build_adamw_optimizer(fresh, torch_module=torch)
        fresh.load_state_dict(
            torch.load(io.BytesIO(payloads["model.bin"]), map_location="cpu", weights_only=True)
        )
        fresh_optimizer.load_state_dict(
            torch.load(io.BytesIO(payloads["optimizer.bin"]), map_location="cpu", weights_only=True)
        )
        restored_rng = torch.load(
            io.BytesIO(payloads["rng.bin"]), map_location="cpu", weights_only=True
        )
        torch.set_rng_state(restored_rng["cpu"].detach().cpu())
        if torch.cuda.is_available() and "cuda" in restored_rng:
            torch.cuda.set_rng_state_all([state.detach().cpu() for state in restored_rng["cuda"]])
        validate_parameter_ownership(fresh, fresh_optimizer)
        if parameter_sha(fresh, torch_module=torch) != parameter_sha(model, torch_module=torch):
            raise ValueError("restored parameters disagree with the checkpoint")
        if moment_fingerprint(fresh_optimizer, torch_module=torch) != moment_fingerprint(
                optimizer, torch_module=torch):
            raise ValueError("restored optimizer moments disagree with the checkpoint")
        if optimizer_step(fresh_optimizer) != optimizer_step(optimizer):
            raise ValueError("restored optimizer step disagrees with the checkpoint")
        stages["fresh_restore"] = "PASS"

        # A fixed deterministic continuation batch isolates checkpoint state
        # from sampler behavior. This is not a full campaign-resume proof.
        if verify_continuation:
            next_tokens = torch.roll(tokens, shifts=1, dims=1)
            for continuation_model, continuation_optimizer in (
                    (model, optimizer), (fresh, fresh_optimizer)):
                continuation_model.train()
                continuation_optimizer.zero_grad(set_to_none=True)
                continuation_logits = continuation_model(
                    next_tokens, positions, mask, use_activation_checkpointing=True)
                continuation_loss, _ = causal_lm_loss(
                    continuation_logits, next_tokens, segment_ids, torch_module=torch)
                continuation_loss.backward()
                torch.nn.utils.clip_grad_norm_(continuation_model.parameters(), 1.0)
                continuation_optimizer.step()
                if runtime == "xla":
                    xm.mark_step()
                continuation_optimizer.zero_grad(set_to_none=True)
            if parameter_sha(model, torch_module=torch) != parameter_sha(fresh, torch_module=torch):
                raise ValueError("restored continuation diverged in model parameters")
            if moment_fingerprint(optimizer, torch_module=torch) != moment_fingerprint(
                    fresh_optimizer, torch_module=torch):
                raise ValueError("restored continuation diverged in Adam moments")
            if optimizer_step(optimizer) != optimizer_step(fresh_optimizer):
                raise ValueError("restored continuation diverged in optimizer step")
            stages["identical_next_update"] = "PASS"
        restored, _ = store.restore(checkpoint_sha)
        if restored != after:
            raise ValueError("store restore disagrees with published state")
        stages["store_restore"] = "PASS"
    receipt: dict[str, object] = {
        "schema": CANARY_SCHEMA,
        "device": str(device),
        "torch_version": str(torch.__version__),
        "cuda_available": bool(torch.cuda.is_available()),
        "model_spec_sha256": model_spec.sha256(),
        "parameter_count": model_spec.parameter_receipt().total,
        "stages": stages,
        "loss": float(loss.item()),
        "supervised_tokens": count,
        "grad_norm_pre_clip": grad_norm,
        "first_update_seconds_including_compile": update_seconds,
        "first_update_real_tokens_per_second": count / update_seconds if update_seconds else None,
        "checkpoint_sha256": checkpoint_sha,
        "status": "PASS" if all(value == "PASS" for value in stages.values()) else "FAIL",
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "activation_checkpointing": True,
        "synthetic_input": True,
        "resume_scope": ("model and Adam state plus one fixed deterministic continuation batch; sampler and per-rank XLA RNG are not exercised"
                         if verify_continuation else ("model/optimizer serialization and one bounded update; continuation comparison skipped"
                                                      if verify_checkpoint else "single bounded model/optimizer update; checkpoint not exercised")),
    }
    (workdir / "target_canary_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt


def _optimizer_bytes(optimizer: Any, torch: Any) -> bytes:
    buffer = io.BytesIO()
    torch.save(_to_cpu(optimizer.state_dict()), buffer)
    return buffer.getvalue()


def _torch_bytes(value: Any, torch: Any) -> bytes:
    buffer = io.BytesIO()
    torch.save(value, buffer)
    return buffer.getvalue()


def _to_cpu(value: Any) -> Any:
    """Materialize checkpoint tensors on CPU before portable serialization."""

    if hasattr(value, "detach") and hasattr(value, "device"):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu(item) for item in value)
    return value


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
