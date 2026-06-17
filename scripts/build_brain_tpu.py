#!/usr/bin/env python3
"""Dedicated PyTorch/XLA trainer for AN-RA iterate500 on Colab TPU runtimes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from anra.anra_paths import DATASET, ROOT
from evaluation.intelligence_telemetry import create_intelligence_session
from training.anra_optimizer import build_optimizer
from training.tpu_runtime import (
    TPUUnavailableError,
    freeze_parametrized_spectral_norms_for_xla,
    load_checkpoint_cpu_first,
    require_torch_xla,
    restore_checkpoint_from_drive,
    xla_save_checkpoint,
)
from training.v2_config import (
    CHECKPOINT_SCHEMA_VERSION,
    EXPECTED_SPECIAL_TOKEN_IDS,
    EXPECTED_TOKENIZER_VOCAB_SIZE,
    TOKENIZER_SCHEMA_VERSION,
    V2_FRONTIER,
    V2_FRONTIER_PARAMETER_COUNT,
    V2_FRONTIER_TRANSFORMER_PARAMETER_COUNT,
    V2_FRONTIER_TRAINING,
    resolve_model_profile,
)
from training.v2_data_mix import (
    TrainingDataMixController,
    V2ConversationDataset,
    build_v2_training_examples,
)
from training.v2_runtime import (
    build_frontier_model,
    ensure_tied_lm_head,
    get_hal_module,
    hal_state_dict,
    load_or_build_v2_tokenizer,
    model_summary,
    update_hal_from_training,
    v2_report_path,
    write_json,
)
from training.wsd_scheduler import get_wsd_schedule, phase_for_step
from runtime.hal_telemetry import publish_hal_state


MODEL_PARAM_COUNT = V2_FRONTIER_PARAMETER_COUNT
MIN_500M_CLASS_PARAMS = 450_000_000
MAX_500M_CLASS_PARAMS = 600_000_000
TRANSFORMER_PARAM_COUNT = V2_FRONTIER_TRANSFORMER_PARAMETER_COUNT


def _source_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _weighted_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    *,
    pad_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    bsz, seq_len, channels = logits.shape
    per_token = F.cross_entropy(
        logits.view(bsz * seq_len, channels),
        targets.view(bsz * seq_len),
        reduction="none",
    ).view(bsz, seq_len)
    effective_weights = weights * (targets != pad_id).float()
    sample_losses = (per_token * effective_weights).sum(dim=1) / effective_weights.sum(dim=1).clamp_min(1.0)
    return sample_losses.mean(), sample_losses


def _make_loader(
    dataset: V2ConversationDataset,
    *,
    batch_size: int,
    active_weights: dict[str, float] | None,
) -> DataLoader:
    if not active_weights:
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    buckets = [dataset.bucket_for_window(index) for index in range(len(dataset))]
    bucket_counts: dict[str, int] = {}
    for bucket in buckets:
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
    sample_weights = [
        float(active_weights.get("owner" if bucket == "own" else bucket, 0.0))
        / max(1, bucket_counts.get(bucket, 0))
        for bucket in buckets
    ]
    sampler = WeightedRandomSampler(
        sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, drop_last=False)


def _checkpoint_payload(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    global_step: int,
    epoch: int,
    best_loss: float,
    sessions_completed: int,
    mix_report: Any,
    tokenizer_hash: str,
    migration: dict[str, object] | None,
) -> dict[str, Any]:
    data_manifests: dict[str, str] = {}
    manifest_root = ROOT / "output" / "v2" / "data_manifests"
    if manifest_root.exists():
        for path in sorted(manifest_root.glob("*.json")):
            data_manifests[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "tokenizer_schema_version": TOKENIZER_SCHEMA_VERSION,
        "tokenizer_contract": {
            "vocab_size": EXPECTED_TOKENIZER_VOCAB_SIZE,
            "special_token_ids": EXPECTED_SPECIAL_TOKEN_IDS,
            "tokenizer_sha256": tokenizer_hash,
        },
        "runtime": "pytorch_xla_tpu",
        "migration_provenance": migration,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": {},
        "step": global_step,
        "global_step": global_step,
        "epoch": epoch,
        "best_loss": best_loss,
        "sessions_completed": sessions_completed,
        "model_config": model.model_config(),
        "hal_state": hal_state_dict(model),
        "mix_report": mix_report.to_dict(),
        "source_commit": _source_commit(),
        "dataset_manifest_hashes": data_manifests,
        "cognitive_extension_release": "cognition-v1",
    }


def _resolve_path(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else ROOT / path


def train_anra_tpu(
    *,
    data_path: str,
    checkpoint_path: str,
    batch_size: int,
    block_size: int,
    max_minutes: int,
    grad_accum_steps: int,
    max_examples: int | None,
    answer_loss_weight: float,
    optimizer_name: str,
    log_every: int,
    model_size: str,
) -> dict[str, Any]:
    if model_size != "frontier":
        raise ValueError("iterate500 TPU training supports only --model-size frontier")
    if torch.cuda.is_available():
        raise RuntimeError(
            "This is the TPU trainer, but CUDA is visible. Use scripts/build_brain.py for T4/CUDA."
        )

    xm, pl = require_torch_xla()
    device = xm.xla_device()
    xla_devices = getattr(xm, "get_xla_supported_devices", lambda *_args, **_kwargs: [])()
    print("[TPU] PyTorch/XLA runtime active", flush=True)
    print(f"[TPU] device={device} supported_devices={xla_devices}", flush=True)

    model_cfg, training_cfg = resolve_model_profile(model_size)
    if model_cfg != V2_FRONTIER:
        raise AssertionError("TPU route must use the 500M frontier config.")
    max_examples = max_examples or V2_FRONTIER_TRAINING.max_mixture_examples

    dataset_path = _resolve_path(data_path)
    tokenizer = load_or_build_v2_tokenizer(dataset_path=dataset_path)
    tokenizer_file = ROOT / "tokenizer" / "tokenizer_v3.json"
    tokenizer_hash = hashlib.sha256(tokenizer_file.read_bytes()).hexdigest() if tokenizer_file.exists() else "missing"

    examples, mix_report = build_v2_training_examples(
        dataset_path=dataset_path,
        max_examples=max_examples,
        own_ratio=V2_FRONTIER_TRAINING.own_ratio,
        identity_ratio=V2_FRONTIER_TRAINING.identity_ratio,
        teacher_ratio=V2_FRONTIER_TRAINING.teacher_ratio,
        symbolic_ratio=V2_FRONTIER_TRAINING.symbolic_ratio,
        replay_ratio=V2_FRONTIER_TRAINING.replay_ratio,
        model_params=MODEL_PARAM_COUNT,
    )
    write_json(v2_report_path("mix_report"), mix_report.to_dict())
    mix_controller = TrainingDataMixController(MODEL_PARAM_COUNT)
    if mix_report.active_weights:
        mix_controller.weights = dict(mix_report.active_weights)

    dataset = V2ConversationDataset(
        examples,
        tokenizer,
        block_size,
        answer_loss_weight=answer_loss_weight,
    )
    if len(dataset) == 0:
        raise RuntimeError("V2ConversationDataset produced zero training windows.")
    loader = _make_loader(dataset, batch_size=batch_size, active_weights=mix_controller.weights)
    device_loader = pl.MpDeviceLoader(loader, device)

    model = build_frontier_model()
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
        print(
            "[TPU] PyTorch gradient checkpointing disabled: torch.utils.checkpoint "
            "does not support xla device type in this Colab runtime.",
            flush=True,
        )
    if hasattr(model, "disable_kv_cache"):
        model.disable_kv_cache()
    model = model.to(device)
    frozen_parametrizations = freeze_parametrized_spectral_norms_for_xla(model)
    if frozen_parametrizations:
        print(
            f"[TPU] Frozen {len(frozen_parametrizations)} spectral-norm parametrization(s) "
            "for XLA memory compatibility.",
            flush=True,
        )
    tied_lm_head = ensure_tied_lm_head(model)
    summary = model_summary(model)
    if not tied_lm_head:
        raise AssertionError("Frontier model must keep token embeddings and LM head tied on TPU.")
    if not MIN_500M_CLASS_PARAMS <= int(summary["parameters"]) <= MAX_500M_CLASS_PARAMS:
        raise AssertionError(
            f"Unexpected 500M-class frontier parameter count: {summary['parameters']:,}"
        )

    learning_rate = float(getattr(training_cfg, "learning_rate", 3e-4))
    optimizer = build_optimizer(
        model,
        lr=learning_rate,
        weight_decay=float(getattr(training_cfg, "weight_decay", 0.1)),
        optimizer_name=optimizer_name,
    )
    optimizer_report = getattr(optimizer, "_anra_optimizer_report", {"selected": {"actual": optimizer_name}})
    write_json(v2_report_path("optimizer_bakeoff"), optimizer_report)

    total_steps = int(getattr(training_cfg, "max_steps", 50_000))
    warmup_steps = int(getattr(training_cfg, "warmup_steps", 100))
    min_lr_ratio = float(getattr(training_cfg, "min_lr", learning_rate * 0.1)) / learning_rate
    scheduler = get_wsd_schedule(
        optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        min_lr_ratio=min_lr_ratio,
    )

    ckpt_path = _resolve_path(checkpoint_path)
    restore_checkpoint_from_drive(ckpt_path)
    global_step = 0
    epoch = 0
    best_loss = float("inf")
    sessions_completed = 0
    checkpoint_migration: dict[str, object] | None = None
    if ckpt_path.exists():
        print(f"[TPU Resume] Found checkpoint: {ckpt_path}", flush=True)
        state = load_checkpoint_cpu_first(model, optimizer, scheduler, ckpt_path, device=device, strict=False)
        if state["loaded"]:
            global_step = int(state["global_step"])
            epoch = int(state["epoch"])
            best_loss = float(state["best_loss"])
            sessions_completed = int(state.get("sessions_completed", 0))
            checkpoint_migration = state.get("migration")
            print(f"[TPU Resume] step={global_step} best_loss={best_loss:.4f}", flush=True)
    else:
        print("[TPU Resume] No checkpoint found - starting from scratch", flush=True)

    intelligence_session = create_intelligence_session(model)
    if intelligence_session is not None:
        print("[ThirdEye] TPU intelligence telemetry active", flush=True)

    stop_requested = False

    def _handle_sigterm(sig_num: int, _frame: object) -> None:
        nonlocal stop_requested
        stop_requested = True
        print(f"[TPU] received signal={sig_num}; saving at next safe point.", flush=True)

    signal.signal(signal.SIGTERM, _handle_sigterm)

    print("", flush=True)
    print("=" * 66, flush=True)
    print("  AN-RA ITERATE500 TPU TRAINING SESSION", flush=True)
    print("=" * 66, flush=True)
    print(f"  Device              : {device}", flush=True)
    print(f"  Parameters          : {summary['parameters']:,}", flush=True)
    print(f"  Tied LM head        : {tied_lm_head}", flush=True)
    print(f"  Transformer params  : {TRANSFORMER_PARAM_COUNT:,}", flush=True)
    print(f"  Hidden/layers/heads : {V2_FRONTIER.n_embd}/{V2_FRONTIER.n_layer}/{V2_FRONTIER.n_head}", flush=True)
    print(f"  Context             : {block_size}", flush=True)
    print(f"  Micro batch         : {batch_size}", flush=True)
    print(f"  Grad accumulation   : {grad_accum_steps}", flush=True)
    print(f"  Grad checkpointing  : disabled on TPU/XLA", flush=True)
    print(f"  Frozen SN params    : {len(frozen_parametrizations)}", flush=True)
    print(f"  Optimizer           : {optimizer_report.get('selected', {}).get('actual', optimizer_name)}", flush=True)
    print(f"  Examples/windows    : {len(examples):,}/{len(dataset):,}", flush=True)
    print(f"  Session minutes     : {max_minutes}", flush=True)
    print("=" * 66, flush=True)
    print("[TPU] First step can take several minutes because XLA compiles the graph once.", flush=True)

    start_time = time.time()
    end_at = start_time + max_minutes * 60
    checkpoint_every_seconds = max(
        300,
        int(float(os.environ.get("ANRA_CHECKPOINT_EVERY_MIN", "25")) * 60),
    )
    next_checkpoint_at = time.time() + checkpoint_every_seconds
    optimizer.zero_grad(set_to_none=True)
    rolling_loss = 0.0
    rolling_count = 0
    accum_micro_steps = 0
    session_optimizer_steps = 0
    last_loss = float("nan")
    last_loss_tensor: torch.Tensor | None = None

    def save_checkpoint(reason: str) -> None:
        payload = _checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            global_step=global_step,
            epoch=epoch,
            best_loss=best_loss,
            sessions_completed=sessions_completed,
            mix_report=mix_report,
            tokenizer_hash=tokenizer_hash,
            migration=checkpoint_migration,
        )
        print(f"[TPU Checkpoint] saving reason={reason} step={global_step}", flush=True)
        xla_save_checkpoint(payload, ckpt_path, xm=xm, mirror_to_drive=True)
        try:
            hal = get_hal_module(model)
            if hal is not None:
                publish_hal_state(hal, source=f"training_tpu:{reason}")
        except Exception as exc:
            print(f"[HAL] TPU publish skipped: {exc}", flush=True)

    while time.time() < end_at and not stop_requested:
        for batch in device_loader:
            if time.time() >= end_at or stop_requested:
                break
            x, y, weights, _sample_idx = batch
            if intelligence_session is not None:
                intelligence_session.begin_step(global_step + 1)
            logits, _ = model(x)
            loss, _sample_losses = _weighted_loss(
                logits,
                y,
                weights,
                pad_id=tokenizer.pad_token_id,
            )
            (loss / max(1, grad_accum_steps)).backward()
            accum_micro_steps += 1
            last_loss_tensor = loss.detach()

            if accum_micro_steps >= grad_accum_steps:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                xm.optimizer_step(optimizer, barrier=True)
                xm.mark_step()
                last_loss = float(last_loss_tensor.cpu().item()) if last_loss_tensor is not None else float("nan")
                try:
                    grad_norm_value = float(grad_norm.detach().cpu().item()) if torch.is_tensor(grad_norm) else float(grad_norm)
                except Exception:
                    grad_norm_value = float("nan")
                if not math.isfinite(last_loss):
                    print("[TPU] non-finite loss detected after optimizer step; stopping to protect checkpoint.", flush=True)
                    stop_requested = True
                    break
                rolling_loss += last_loss
                rolling_count += 1
                global_step += 1
                session_optimizer_steps += 1
                accum_micro_steps = 0
                best_loss = min(best_loss, last_loss)
                update_hal_from_training(
                    model,
                    loss=last_loss,
                    best_loss=best_loss,
                    gradient_norm=grad_norm_value,
                    step=global_step,
                )
                if intelligence_session is not None:
                    hal = get_hal_module(model)
                    if hal is not None:
                        intelligence_session.record_hal_step(step=global_step, hal_state=hal.state)
                    try:
                        intelligence_session.record_optimizer_step(
                            step=global_step,
                            loss=last_loss,
                            learning_rate=float(optimizer.param_groups[0]["lr"]),
                            gradient_norm=grad_norm_value,
                            tokens=batch_size * block_size * grad_accum_steps,
                        )
                    except Exception as exc:
                        print(f"[ThirdEye] TPU telemetry step skipped: {exc}", flush=True)
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                if global_step <= 3 or global_step % max(1, log_every) == 0:
                    avg_loss = rolling_loss / max(1, rolling_count)
                    phase = phase_for_step(global_step, warmup_steps=warmup_steps, total_steps=total_steps)
                    elapsed_min = (time.time() - start_time) / 60.0
                    print(
                        f"[TPU step {global_step:>7}] "
                        f"loss={last_loss:.4f} avg={avg_loss:.4f} best={best_loss:.4f} "
                        f"lr={optimizer.param_groups[0]['lr']:.2e} phase={phase.name} "
                        f"elapsed={elapsed_min:.1f}m",
                        flush=True,
                    )
                    rolling_loss = 0.0
                    rolling_count = 0

                if time.time() >= next_checkpoint_at:
                    save_checkpoint("interval")
                    next_checkpoint_at = time.time() + checkpoint_every_seconds

            if stop_requested:
                break
        epoch += 1

    sessions_completed += 1
    save_checkpoint("session_end")
    telemetry_report = None
    if intelligence_session is not None:
        try:
            telemetry_report = intelligence_session.finalize(
                checkpoint_id=f"{ckpt_path.name}:step-{global_step:012d}",
                capability_score=None,
            )
        except Exception as exc:
            print(f"[ThirdEye] TPU telemetry finalize skipped: {exc}", flush=True)

    report = {
        "runtime": "pytorch_xla_tpu",
        "checkpoint_path": str(ckpt_path),
        "global_step": global_step,
        "session_optimizer_steps": session_optimizer_steps,
        "best_loss": best_loss,
        "last_loss": last_loss,
        "epochs_completed": epoch,
        "sessions_completed": sessions_completed,
        "batch_size": batch_size,
        "grad_accum_steps": grad_accum_steps,
        "block_size": block_size,
        "model_parameters": summary["parameters"],
        "expected_tied_parameters": MODEL_PARAM_COUNT,
        "tied_lm_head": tied_lm_head,
        "frozen_spectral_norm_parametrizations": frozen_parametrizations,
        "transformer_parameters": TRANSFORMER_PARAM_COUNT,
        "third_eye": telemetry_report,
    }
    write_json(v2_report_path("tpu_training_report"), report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="AN-RA iterate500 TPU trainer")
    parser.add_argument("--data_path", default=str(DATASET))
    parser.add_argument("--checkpoint_path", default="anra_frontier_500m.pt")
    parser.add_argument("--model-size", default="frontier", choices=["frontier"])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--block_size", type=int, default=V2_FRONTIER.block_size)
    parser.add_argument("--max_minutes", type=int, default=V2_FRONTIER_TRAINING.session_minutes)
    parser.add_argument("--grad_accum_steps", type=int, default=V2_FRONTIER_TRAINING.grad_accum_steps)
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--answer_loss_weight", type=float, default=V2_FRONTIER_TRAINING.answer_loss_weight)
    parser.add_argument(
        "--optimizer",
        default="adafactor",
        choices=["adafactor", "adamw", "auto", "muon", "galore", "adam8bit", "scale", "qgalore"],
        help="Adafactor is the TPU default because it is much lighter than AdamW.",
    )
    parser.add_argument("--log_every", type=int, default=1)
    args = parser.parse_args()

    try:
        train_anra_tpu(
            data_path=args.data_path,
            checkpoint_path=args.checkpoint_path,
            batch_size=args.batch_size,
            block_size=args.block_size,
            max_minutes=args.max_minutes,
            grad_accum_steps=args.grad_accum_steps,
            max_examples=args.max_examples,
            answer_loss_weight=args.answer_loss_weight,
            optimizer_name=args.optimizer,
            log_every=args.log_every,
            model_size=args.model_size,
        )
    except TPUUnavailableError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
