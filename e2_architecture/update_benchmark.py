"""Bounded real-update and exact-resume canary for the executable P35 stack.

This is deliberately a tiny deterministic experiment, not model training.  It
checks that AdamW owns the live parameters, that parameters/moments/steps really
change, and that a save/load boundary produces the same continuation as an
uninterrupted run.  FP32 and BF16 are run from identical initial weights; the
receipt reports the native optimizer-state dtypes rather than assuming them.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import platform
import tempfile
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .block_benchmark import _build_model, shape_arms


RESUME_TOLERANCES = {
    "float32": {
        # Calibrated from the strict 1k CUDA receipt, then widened by >3x.
        "parameter_max_abs": 1e-5,
        "parameter_relative_rms": 1e-6,
        "optimizer_state_max_abs": 1e-4,
    },
    "bfloat16": {
        # BF16 round-off is bounded separately from FP32 master-state drift.
        "parameter_max_abs": 2e-3,
        "parameter_relative_rms": 5e-4,
        "optimizer_state_max_abs": 1e-3,
    },
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(tensor: Any) -> str:
    import torch

    value = tensor.detach().float().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(value).hexdigest()


@dataclass(frozen=True, slots=True)
class UpdateConfig:
    device: str
    sequence_length: int
    batch_size: int
    steps: int
    seed: int
    parameter_storage: str = "master"
    arm: str = "middle"

    def assert_valid(self) -> None:
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("device must be cpu or cuda")
        if self.parameter_storage not in {"native", "master"}:
            raise ValueError("parameter_storage must be native or master")
        if self.arm not in {arm.name for arm in shape_arms()}:
            raise ValueError("arm must be one of the E2 shape arms")
        if self.sequence_length <= 0 or self.batch_size <= 0 or self.steps < 2:
            raise ValueError("invalid update canary dimensions")
        if self.seed < 0:
            raise ValueError("seed must be nonnegative")


def _make_model(torch: Any, *, device: Any, dtype: Any, sequence_length: int, arm: str = "middle") -> Any:
    selected = {candidate.name: candidate for candidate in shape_arms()}[arm]
    model = _build_model(torch, selected, maximum_sequence_length=sequence_length)
    return model.to(device=device, dtype=dtype)


def _new_optimizer(torch: Any, model: Any) -> Any:
    return torch.optim.AdamW(
        model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1
    )


def _batch(torch: Any, *, config: UpdateConfig, step: int, vocabulary: int, device: Any) -> tuple[Any, Any]:
    generator = torch.Generator(device="cpu").manual_seed(config.seed + step)
    tokens = torch.randint(
        0, vocabulary, (config.batch_size, config.sequence_length), generator=generator
    ).to(device=device)
    targets = torch.randint(
        0, vocabulary, (config.batch_size, config.sequence_length), generator=generator
    ).to(device=device)
    return tokens, targets


def _one_update(
    torch: Any,
    model: Any,
    optimizer: Any,
    tokens: Any,
    targets: Any,
    *,
    autocast_dtype: Any = None,
    device_type: str,
) -> dict[str, float]:
    import torch.nn.functional as functional

    optimizer.zero_grad(set_to_none=True)
    context = (
        torch.autocast(device_type=device_type, dtype=autocast_dtype)
        if autocast_dtype is not None
        else nullcontext()
    )
    with context:
        logits = model(tokens)
    loss = functional.cross_entropy(logits.float().view(-1, logits.shape[-1]), targets.view(-1))
    loss.backward()
    gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item())
    clipped_norm = float(
        torch.sqrt(
            sum(
                gradient.detach().float().square().sum()
                for gradient in (parameter.grad for parameter in model.parameters())
                if gradient is not None
            )
        ).item()
    )
    optimizer.step()
    return {
        "loss": float(loss.detach().item()),
        "gradient_norm": gradient_norm,
        "clipped_gradient_norm": clipped_norm,
    }


def _optimizer_summary(torch: Any, model: Any, optimizer: Any) -> dict[str, Any]:
    states = list(optimizer.state.values())
    steps = [float(state["step"].item()) for state in states if "step" in state]
    moments = [state["exp_avg"] for state in states if "exp_avg" in state]
    optimizer_parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    parameter_ids = {id(parameter) for parameter in optimizer_parameters}
    model_ids = {id(parameter) for parameter in model.parameters()}
    return {
        "optimizer_owns_all_live_parameters": parameter_ids == model_ids,
        "optimizer_has_no_duplicate_parameters": len(optimizer_parameters) == len(parameter_ids),
        "state_entries": len(states),
        "maximum_step": max(steps) if steps else 0.0,
        "moment_nonzero": bool(moments) and any(bool(torch.count_nonzero(moment).item()) for moment in moments),
        "state_dtypes": sorted({str(value.dtype).removeprefix("torch.") for state in states for value in state.values() if torch.is_tensor(value)}),
        "moments_fp32": bool(moments) and all(moment.dtype == torch.float32 for moment in moments),
    }


def _run_dtype(torch: Any, *, config: UpdateConfig, dtype_name: str) -> dict[str, Any]:
    compute_dtype = torch.float32 if dtype_name == "float32" else torch.bfloat16
    storage_dtype = (
        torch.float32
        if dtype_name == "float32" or config.parameter_storage == "master"
        else torch.bfloat16
    )
    device = torch.device(config.device)
    torch.manual_seed(config.seed)
    if config.device == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    model = _make_model(torch, device=device, dtype=storage_dtype, sequence_length=config.sequence_length, arm=config.arm)
    optimizer = _new_optimizer(torch, model)
    initial_hash = _tensor_sha256(model.embedding.weight)
    losses: list[float] = []
    gradients: list[float] = []
    clipped_gradients: list[float] = []
    for step in range(config.steps):
        tokens, targets = _batch(
            torch,
            config=config,
            step=step,
            vocabulary=shape_arms()[1].model.vocabulary_size,
            device=device,
        )
        result = _one_update(
            torch,
            model,
            optimizer,
            tokens,
            targets,
            autocast_dtype=compute_dtype if dtype_name == "bfloat16" and config.parameter_storage == "master" else None,
            device_type=config.device,
        )
        losses.append(result["loss"])
        gradients.append(result["gradient_norm"])
        clipped_gradients.append(result["clipped_gradient_norm"])
    final_hash = _tensor_sha256(model.embedding.weight)
    summary = _optimizer_summary(torch, model, optimizer)
    return {
        "dtype": dtype_name,
        "compute_dtype": str(compute_dtype).removeprefix("torch."),
        "parameter_storage_dtype": str(storage_dtype).removeprefix("torch."),
        "losses": losses,
        "gradient_norms": gradients,
        "clipped_gradient_norms": clipped_gradients,
        "initial_parameter_hash": initial_hash,
        "final_parameter_hash": final_hash,
        "parameter_changed": initial_hash != final_hash,
        "optimizer": summary,
        "finite": all(torch.isfinite(torch.tensor(losses + gradients)).tolist()),
    }


def _resume_equivalence(torch: Any, *, config: UpdateConfig, dtype_name: str) -> dict[str, Any]:
    compute_dtype = torch.float32 if dtype_name == "float32" else torch.bfloat16
    storage_dtype = (
        torch.float32
        if dtype_name == "float32" or config.parameter_storage == "master"
        else torch.bfloat16
    )
    device = torch.device(config.device)
    torch.manual_seed(config.seed + 10_000)
    if config.device == "cuda":
        torch.cuda.manual_seed_all(config.seed + 10_000)
    left = _make_model(torch, device=device, dtype=storage_dtype, sequence_length=config.sequence_length, arm=config.arm)
    right = _make_model(torch, device=device, dtype=storage_dtype, sequence_length=config.sequence_length, arm=config.arm)
    right.load_state_dict(copy.deepcopy(left.state_dict()))
    left_optimizer = _new_optimizer(torch, left)
    right_optimizer = _new_optimizer(torch, right)
    vocabulary = shape_arms()[1].model.vocabulary_size
    for step in range(config.steps):
        tokens, targets = _batch(torch, config=config, step=step + 100, vocabulary=vocabulary, device=device)
        _one_update(torch, left, left_optimizer, tokens, targets, autocast_dtype=compute_dtype if dtype_name == "bfloat16" and config.parameter_storage == "master" else None, device_type=config.device)
    for step in range(1):
        tokens, targets = _batch(torch, config=config, step=step + 100, vocabulary=vocabulary, device=device)
        _one_update(torch, right, right_optimizer, tokens, targets, autocast_dtype=compute_dtype if dtype_name == "bfloat16" and config.parameter_storage == "master" else None, device_type=config.device)
    with tempfile.TemporaryDirectory(prefix="esoes-update-") as directory:
        checkpoint_path = Path(directory) / "resume.pt"
        torch.save(
            {
                "model": right.state_dict(),
                "optimizer": right_optimizer.state_dict(),
                "global_update": 1,
                "dtype": dtype_name,
            },
            checkpoint_path,
        )
        serialized_bytes = checkpoint_path.stat().st_size
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    # Release the pre-save right-hand model and its Adam state before creating
    # the resumed copy.  Keeping all three full P35 models live can exceed
    # host RAM on a local CPU probe and is not part of the resume contract.
    del right, right_optimizer
    gc.collect()
    resumed = _make_model(torch, device=device, dtype=storage_dtype, sequence_length=config.sequence_length, arm=config.arm)
    resumed.load_state_dict(checkpoint["model"])
    resumed_optimizer = _new_optimizer(torch, resumed)
    resumed_optimizer.load_state_dict(checkpoint["optimizer"])
    for step in range(1, config.steps):
        tokens, targets = _batch(torch, config=config, step=step + 100, vocabulary=vocabulary, device=device)
        _one_update(torch, resumed, resumed_optimizer, tokens, targets, autocast_dtype=compute_dtype if dtype_name == "bfloat16" and config.parameter_storage == "master" else None, device_type=config.device)
    left_vector = torch.cat([parameter.detach().float().reshape(-1).cpu() for parameter in left.parameters()])
    resumed_vector = torch.cat([parameter.detach().float().reshape(-1).cpu() for parameter in resumed.parameters()])
    difference = (left_vector - resumed_vector).abs()
    optimizer_max_error = 0.0
    optimizer_exact = True
    left_states = list(left_optimizer.state.values())
    resumed_states = list(resumed_optimizer.state.values())
    if len(left_states) != len(resumed_states):
        optimizer_exact = False
    for left_state, resumed_state in zip(left_states, resumed_states):
        if set(left_state) != set(resumed_state):
            optimizer_exact = False
            continue
        for key in left_state:
            left_value, resumed_value = left_state[key], resumed_state[key]
            if torch.is_tensor(left_value) and torch.is_tensor(resumed_value):
                state_difference = (left_value.detach().float().cpu() - resumed_value.detach().float().cpu()).abs()
                optimizer_max_error = max(optimizer_max_error, float(state_difference.max().item()))
                optimizer_exact = optimizer_exact and bool(
                    torch.equal(left_value.detach().cpu(), resumed_value.detach().cpu())
                )
            else:
                optimizer_exact = optimizer_exact and left_value == resumed_value
    tolerance = RESUME_TOLERANCES[dtype_name]
    parameter_relative_rms = float(
        (torch.sqrt(difference.square().mean()) / torch.sqrt(left_vector.square().mean()).clamp_min(1e-20)).item()
    )
    return {
        "parameter_max_abs_error": float(difference.max().item()),
        "parameter_relative_rms_error": parameter_relative_rms,
        "exact_within_dtype": bool(torch.equal(left_vector, resumed_vector)),
        "parameter_within_tolerance": bool(
            difference.max().item() <= tolerance["parameter_max_abs"]
            and parameter_relative_rms <= tolerance["parameter_relative_rms"]
        ),
        "optimizer_state_max_abs_error": optimizer_max_error,
        "optimizer_state_exact": optimizer_exact,
        "optimizer_state_within_tolerance": optimizer_max_error <= tolerance["optimizer_state_max_abs"],
        "serialized_checkpoint_bytes": serialized_bytes,
        "tolerances": tolerance,
        "resumed_optimizer": _optimizer_summary(torch, resumed, resumed_optimizer),
    }


def benchmark(config: UpdateConfig) -> dict[str, Any]:
    config.assert_valid()
    try:
        import torch
    except ImportError as exc:
        return {"schema": "esoes-e2-real-update/v1", "status": "BLOCKED_TORCH", "reason": str(exc)}
    if config.device == "cuda" and not torch.cuda.is_available():
        return {"schema": "esoes-e2-real-update/v1", "status": "BLOCKED_CUDA", "torch_version": torch.__version__}
    rows: list[dict[str, Any]] = []
    for dtype_name in ("float32", "bfloat16"):
        rows.append(_run_dtype(torch, config=config, dtype_name=dtype_name))
        # The P35 optimizer state is intentionally large relative to the tiny
        # canary.  Explicit collection keeps sequential dtype/arm probes from
        # being killed by host RSS before the receipt is written.
        gc.collect()
        if config.device == "cuda":
            torch.cuda.empty_cache()
    resumes = {row["dtype"]: _resume_equivalence(torch, config=config, dtype_name=row["dtype"]) for row in rows}
    checks = {
        "all_finite": all(row["finite"] for row in rows),
        "global_gradient_clip_effective": all(
            max(row["clipped_gradient_norms"]) <= 1.00001 for row in rows
        ),
        "all_parameters_changed": all(row["parameter_changed"] for row in rows),
        "optimizer_owns_live_parameters": all(row["optimizer"]["optimizer_owns_all_live_parameters"] for row in rows),
        "optimizer_has_no_duplicate_parameters": all(row["optimizer"]["optimizer_has_no_duplicate_parameters"] for row in rows),
        "optimizer_steps_reached_target": all(row["optimizer"]["maximum_step"] == config.steps for row in rows),
        "moments_changed": all(row["optimizer"]["moment_nonzero"] for row in rows),
        "bf16_optimizer_moments_fp32": all(
            row["optimizer"]["moments_fp32"] for row in rows if row["dtype"] == "bfloat16"
        ),
        "resume_equivalence_within_dtype": all(value["parameter_within_tolerance"] for value in resumes.values()),
        "resume_optimizer_equivalence": all(value["optimizer_state_within_tolerance"] for value in resumes.values()),
    }
    return {
        "schema": "esoes-e2-real-update/v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "scope": "exact P35 middle stack; bounded AdamW updates and save/resume equivalence; no training run",
        "implementation_sha256": _sha256_file(Path(__file__)),
        "model_implementation_sha256": _sha256_file(Path(__file__).with_name("block_benchmark.py")),
        "config": {
            "device": config.device,
            "sequence_length": config.sequence_length,
            "batch_size": config.batch_size,
            "steps": config.steps,
            "seed": config.seed,
            "parameter_storage": config.parameter_storage,
            "arm": config.arm,
        },
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(0) if config.device == "cuda" else platform.processor(),
        "checks": checks,
        "resume_tolerances": RESUME_TOLERANCES,
        "rows": rows,
        "resume": resumes,
        "limitations": [
            "Native AdamW state dtype is reported; this canary does not claim FP32 master-state semantics unless the receipt shows them.",
            "Three updates on synthetic tokens prove wiring and continuity, not optimizer quality or cognition.",
            "Target TPU/XLA and distributed all-reduce exact-resume remain open.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=37001)
    parser.add_argument("--parameter-storage", choices=("native", "master"), default="master")
    parser.add_argument("--arm", choices=tuple(arm.name for arm in shape_arms()), default="middle")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = benchmark(UpdateConfig(args.device, args.sequence_length, args.batch_size, args.steps, args.seed, args.parameter_storage, args.arm))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(args.output)}, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
