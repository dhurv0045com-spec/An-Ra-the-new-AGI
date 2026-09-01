"""Bounded real-P35 checkpoint transaction canary.

This is the first framework-specific bridge to the framework-neutral
``CheckpointStore``.  It performs two synthetic AdamW updates on the exact
middle P35 constructor, publishes the first update through the content-
addressed transaction, restores from a clean local copy, and proves that the
next update matches an uninterrupted continuation.  It is intentionally not
a trainer, distributed implementation, or remote-durability claim.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import platform
import shutil
import subprocess
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

from e2_architecture.block_benchmark import _build_model, shape_arms

from .checkpoint import CheckpointStore
from .optimizer import build_adamw_optimizer, optimizer_group_receipt
from .state import CURSOR_SCHEMA, IDENTITY_SCHEMA, CursorState, IdentityBindings, TrainingState


SCHEMA = "esoes-v5-p35-checkpoint-canary/v1"
SEED = 47_031
SEQUENCE_LENGTH = 8
BATCH_SIZE = 1
UPDATES = 2
TOKENS_PER_UPDATE = SEQUENCE_LENGTH * BATCH_SIZE


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def _torch_dump(torch: Any, value: object) -> bytes:
    buffer = io.BytesIO()
    torch.save(value, buffer)
    return buffer.getvalue()


def _torch_load(torch: Any, payload: bytes, device: Any) -> object:
    return torch.load(io.BytesIO(payload), map_location=device, weights_only=False)


def _parameter_hash(model: Any) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode("utf-8"))
        digest.update(parameter.detach().float().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _optimizer_hash(torch: Any, model: Any, optimizer: Any) -> str:
    digest = hashlib.sha256()
    for index, parameter in enumerate(model.parameters()):
        digest.update(str(index).encode("ascii"))
        state = optimizer.state.get(parameter, {})
        for key in sorted(state):
            value = state[key]
            digest.update(key.encode("utf-8"))
            if torch.is_tensor(value):
                digest.update(str(value.dtype).encode("ascii"))
                digest.update(value.detach().float().cpu().contiguous().numpy().tobytes())
            else:
                digest.update(repr(value).encode("utf-8"))
    return digest.hexdigest()


def _max_optimizer_error(torch: Any, left: Any, right: Any) -> float:
    maximum = 0.0
    left_parameters = [
        parameter for group in left.param_groups for parameter in group["params"]
    ]
    right_parameters = [
        parameter for group in right.param_groups for parameter in group["params"]
    ]
    left_states = [left.state.get(parameter, {}) for parameter in left_parameters]
    right_states = [right.state.get(parameter, {}) for parameter in right_parameters]
    if len(left_states) != len(right_states):
        raise ValueError("optimizer parameter inventories differ after restore")
    for left_state, right_state in zip(left_states, right_states):
        if set(left_state) != set(right_state):
            raise ValueError("optimizer state keys differ after restore")
        for key in left_state:
            left_value, right_value = left_state[key], right_state[key]
            if torch.is_tensor(left_value) and torch.is_tensor(right_value):
                maximum = max(
                    maximum,
                    float((left_value.detach().float().cpu() - right_value.detach().float().cpu()).abs().max().item()),
                )
            elif left_value != right_value:
                raise ValueError(f"optimizer scalar state differs: {key}")
    return maximum


def _source_commit() -> str:
    try:
        value = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        value = "0" * 40
    return value if len(value) == 40 and all(character in "0123456789abcdef" for character in value) else "0" * 40


def _identities(arm: Any) -> IdentityBindings:
    def identity(label: str) -> str:
        return _sha256(f"anra-v5-p35-canary:{label}".encode("ascii"))

    return IdentityBindings(
        schema=IDENTITY_SCHEMA,
        source_commit=_source_commit(),
        model_spec_sha256=arm.model.sha256(),
        tokenizer_sha256=identity("tokenizer-24576"),
        data_manifest_sha256=identity("synthetic-data"),
        pack_manifest_sha256=identity("synthetic-pack"),
        run_spec_sha256=identity("run"),
        optimizer_spec_sha256=identity("adamw-fp32-master"),
        schedule_spec_sha256=identity("constant-lr"),
        curriculum_spec_sha256=identity("canary"),
    )


def _initial_state(arm: Any) -> TrainingState:
    identities = _identities(arm)
    return TrainingState.initial(
        lineage_id="p35-canary",
        token_budget=TOKENS_PER_UPDATE * UPDATES,
        tokens_per_update=TOKENS_PER_UPDATE,
        cursor=CursorState(
            schema=CURSOR_SCHEMA,
            pack_manifest_sha256=identities.pack_manifest_sha256,
            shard_ordinal=0,
            sequence_ordinal=0,
            token_offset=0,
        ),
        rng_state_sha256="0" * 64,
        curriculum_phase="canary",
        identities=identities,
    )


def _batch(torch: Any, *, step: int, vocabulary: int, device: Any) -> tuple[Any, Any]:
    generator = torch.Generator(device="cpu").manual_seed(SEED + step)
    tokens = torch.randint(
        0, vocabulary, (BATCH_SIZE, SEQUENCE_LENGTH), generator=generator
    ).to(device=device)
    targets = torch.randint(
        0, vocabulary, (BATCH_SIZE, SEQUENCE_LENGTH), generator=generator
    ).to(device=device)
    return tokens, targets


def _one_update(torch: Any, model: Any, optimizer: Any, scheduler: Any, *, step: int, device: Any) -> dict[str, float]:
    import torch.nn.functional as functional

    tokens, targets = _batch(
        torch,
        step=step,
        vocabulary=next(arm for arm in shape_arms() if arm.name == "middle").model.vocabulary_size,
        device=device,
    )
    optimizer.zero_grad(set_to_none=True)
    logits = model(tokens)
    loss = functional.cross_entropy(logits.float().view(-1, logits.shape[-1]), targets.view(-1))
    loss.backward()
    gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item())
    optimizer.step()
    scheduler.step()
    if not torch.isfinite(loss).item() or not torch.isfinite(torch.tensor(gradient_norm)).item():
        raise ValueError("non-finite P35 canary update")
    return {"loss": float(loss.detach().item()), "gradient_norm": gradient_norm}


def _rng_payload(torch: Any) -> bytes:
    state: dict[str, object] = {"cpu": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return _torch_dump(torch, state)


def _restore_rng(torch: Any, payload: bytes, device: Any) -> None:
    state = _torch_load(torch, payload, device)
    torch.set_rng_state(state["cpu"].detach().cpu())
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all([value.detach().cpu() for value in state["cuda"]])


def _payloads(torch: Any, model: Any, optimizer: Any, scheduler: Any, state: TrainingState) -> dict[str, bytes]:
    return {
        "model.bin": _torch_dump(torch, model.state_dict()),
        "optimizer.bin": _torch_dump(torch, optimizer.state_dict()),
        "scheduler.json": _canonical_json(scheduler.state_dict()),
        "rng.bin": _rng_payload(torch),
        "cursor.json": _canonical_json(asdict(state.cursor)),
        "ledger.json": _canonical_json(state.tokens_by_source),
        "training_state.json": _canonical_json(state.canonical()),
    }


def run_canary(*, device_name: str = "cuda") -> dict[str, object]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment-dependent
        return {"schema": SCHEMA, "status": "BLOCKED_TORCH", "reason": str(exc)}
    if device_name not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    if device_name == "cuda" and not torch.cuda.is_available():
        return {"schema": SCHEMA, "status": "BLOCKED_CUDA", "torch_version": torch.__version__}

    device = torch.device(device_name)
    torch.manual_seed(SEED)
    if device_name == "cuda":
        torch.cuda.manual_seed_all(SEED)
    arm = next(candidate for candidate in shape_arms() if candidate.name == "middle")
    model = _build_model(torch, arm, maximum_sequence_length=SEQUENCE_LENGTH).to(device=device, dtype=torch.float32)
    optimizer = build_adamw_optimizer(model, torch_module=torch)
    optimizer_receipt = optimizer_group_receipt(model, optimizer)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    state = _initial_state(arm)
    first_metrics = _one_update(torch, model, optimizer, scheduler, step=0, device=device)
    rng_payload = _rng_payload(torch)
    state = state.advance(
        tokens_by_source={"synthetic": TOKENS_PER_UPDATE},
        cursor=CursorState(CURSOR_SCHEMA, state.cursor.pack_manifest_sha256, 0, 1, SEQUENCE_LENGTH),
        rng_state_sha256=_sha256(rng_payload),
        parent_checkpoint_sha256=None,
    )
    with tempfile.TemporaryDirectory(prefix="esoes-p35-canary-") as directory:
        root = Path(directory) / "store"
        store = CheckpointStore(root, state.lineage_id)
        checkpoint_payloads = _payloads(torch, model, optimizer, scheduler, state)
        first_checkpoint = store.publish(
            state=state, payloads=checkpoint_payloads, expected_parent_sha256=None
        )
        parent_parameter_hash = _parameter_hash(model)
        parent_optimizer_hash = _optimizer_hash(torch, model, optimizer)
        reference_metrics = _one_update(torch, model, optimizer, scheduler, step=1, device=device)
        reference_parameter_hash = _parameter_hash(model)
        reference_optimizer_hash = _optimizer_hash(torch, model, optimizer)
        reference_model_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
        reference_optimizer_state = optimizer
        del model, optimizer, scheduler
        gc.collect()
        if device_name == "cuda":
            torch.cuda.empty_cache()

        clean_root = Path(directory) / "clean-copy"
        shutil.copytree(root, clean_root)
        clean_store = CheckpointStore(clean_root, state.lineage_id)
        restored_state, restored_payloads = clean_store.restore(first_checkpoint)
        _restore_rng(torch, restored_payloads["rng.bin"], device)
        restored_model = _build_model(torch, arm, maximum_sequence_length=SEQUENCE_LENGTH).to(device=device, dtype=torch.float32)
        restored_model.load_state_dict(_torch_load(torch, restored_payloads["model.bin"], device))
        restored_optimizer = build_adamw_optimizer(restored_model, torch_module=torch)
        restored_optimizer.load_state_dict(_torch_load(torch, restored_payloads["optimizer.bin"], device))
        restored_scheduler = torch.optim.lr_scheduler.LambdaLR(restored_optimizer, lambda _: 1.0)
        restored_scheduler.load_state_dict(json.loads(restored_payloads["scheduler.json"]))
        resumed_metrics = _one_update(torch, restored_model, restored_optimizer, restored_scheduler, step=1, device=device)
        parameter_errors = [
            float((reference_model_state[name].float().cpu() - value.detach().float().cpu()).abs().max().item())
            for name, value in restored_model.state_dict().items()
        ]
        optimizer_error = _max_optimizer_error(torch, reference_optimizer_state, restored_optimizer)
        final_state = restored_state.advance(
            tokens_by_source={"synthetic": TOKENS_PER_UPDATE},
            cursor=CursorState(CURSOR_SCHEMA, restored_state.cursor.pack_manifest_sha256, 0, 2, SEQUENCE_LENGTH * 2),
            rng_state_sha256=_sha256(_rng_payload(torch)),
            parent_checkpoint_sha256=first_checkpoint,
        )
        final_checkpoint = clean_store.publish(
            state=final_state,
            payloads=_payloads(torch, restored_model, restored_optimizer, restored_scheduler, final_state),
            expected_parent_sha256=first_checkpoint,
        )
        final_restored_state, _ = clean_store.restore(final_checkpoint)
        if final_restored_state != final_state:
            raise ValueError("final checkpoint did not restore its committed training state")
        resumed_parameter_hash = _parameter_hash(restored_model)
        resumed_optimizer_hash = _optimizer_hash(torch, restored_model, restored_optimizer)
        if reference_optimizer_hash != resumed_optimizer_hash:
            raise ValueError("optimizer state hash changed across clean-copy resume")
        if reference_parameter_hash != resumed_parameter_hash:
            raise ValueError("parameter hash changed across clean-copy resume")
        return {
            "schema": SCHEMA,
            "status": "PASS",
            "scope": "exact middle P35; two synthetic AdamW updates; first publish and clean-copy resume",
            "implementation_sha256": _sha256_file(Path(__file__)),
            "model_constructor_sha256": _sha256_file(Path(__file__).parents[1] / "e2_architecture/block_benchmark.py"),
            "device": device_name,
            "device_name": torch.cuda.get_device_name(0) if device_name == "cuda" else platform.processor(),
            "torch_version": torch.__version__,
            "config": {"seed": SEED, "sequence_length": SEQUENCE_LENGTH, "batch_size": BATCH_SIZE, "updates": UPDATES, "arm": arm.name},
            "first_checkpoint_sha256": first_checkpoint,
            "final_checkpoint_sha256": final_checkpoint,
            "global_update": final_state.global_update,
            "cumulative_tokens": final_state.cumulative_tokens,
            "optimizer_step_max": final_state.optimizer_step_max,
            "parameter_hash": resumed_parameter_hash,
            "optimizer_hash": resumed_optimizer_hash,
            "optimizer_group_receipt": optimizer_receipt,
            "metrics": {"first": first_metrics, "reference_second": reference_metrics, "resumed_second": resumed_metrics},
            "resume": {"parameter_max_abs_error": max(parameter_errors), "optimizer_state_max_abs_error": optimizer_error, "parameter_hash_equal": True, "optimizer_hash_equal": True, "clean_copy_restore": True},
            "checks": {"actual_p35_parameter_count": sum(parameter.numel() for parameter in restored_model.parameters()) == arm.model.parameter_receipt().total, "two_updates_reached": final_state.global_update == UPDATES, "exact_token_ledger": final_state.cumulative_tokens == TOKENS_PER_UPDATE * UPDATES, "model_changed": parent_parameter_hash != reference_parameter_hash, "optimizer_changed": parent_optimizer_hash != reference_optimizer_hash, "resume_within_tolerance": max(parameter_errors) <= 1e-5 and optimizer_error <= 1e-4, "clean_copy_restore": True, "final_transaction_published": bool(final_checkpoint), "final_state_restore": final_restored_state == final_state},
            "limitations": ["Synthetic tokens and two updates prove wiring and continuation, not optimizer quality or cognition.", "The canary uses local filesystem copy plus fsync/atomic publication; remote object-store durability, distributed sharding, TPU/XLA, and failure-state recovery remain open.", "No checkpoint weights or optimizer tensors are committed; only this receipt is committed."],
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_canary(device_name=args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "status": result["status"]}, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
