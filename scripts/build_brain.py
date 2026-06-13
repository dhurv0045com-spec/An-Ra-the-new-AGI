# NOTE: scripts/train.py is the canonical training script for local runs.
# This file handles tokenizer building and frontier data preparation.
from __future__ import annotations

import argparse
import heapq
import json
import math
import os
import shutil
import signal
import sys
import threading
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from anra.anra_paths import (
    CDR_REPORT,
    DRIVE_V2_CHECKPOINTS,
    FAILURE_REPLAY_DATASET,
    IBS_LATEST,
    MODEL_GROWTH_REPORT,
    QUARANTINE_DIR,
    REGRET_STATE,
    ROOT,
    SOVEREIGNTY_EVENTS,
    V2_TOKENIZER_FILE,
    V2_BRAIN_CHECKPOINT,
    V3_3B_CHECKPOINT,
)
from engine.eval_harness import EvalHarness, EvalResult
from engine.feature_flags import is_enabled
from runtime.safe_load import safe_torch_load
from training.anra_optimizer import (
    IDENTITY_PARAMETER_PATTERNS,
    build_optimizer,
    is_identity_parameter,
)
from training.cdr import CorrectedFailureCurriculum
from training.continual import assess_continual_readiness
from training.eval_v2 import quick_eval_loss, run_compact_eval
from training.mixed_precision import MixedPrecisionTrainer
from training.pcgrad import PCGradAccumulator
from training.v2_config import (
    CHECKPOINT_SCHEMA_VERSION,
    EXPECTED_SPECIAL_TOKEN_IDS,
    EXPECTED_TOKENIZER_VOCAB_SIZE,
    TOKENIZER_SCHEMA_VERSION,
    V2_1B_FRONTIER,
    V2_1B_TRAINING,
    V2_3B,
    V2_3B_TRAINING,
    V2_MODEL,
    V2_TRAINING,
    resolve_model_profile,
)
from training.v2_data_mix import (
    TrainingDataMixController,
    V2ConversationDataset,
    build_v2_training_examples,
)
from scripts.session_dashboard import print_session_dashboard
from training.dynamic_regret import DynamicRegretScheduler
from training.v2_runtime import (
    atomic_save,
    build_frontier_model,
    build_3b_model,
    build_v2_model,
    canonical_v2_checkpoint,
    load_checkpoint,
    load_or_build_v2_tokenizer,
    model_summary,
    sync_to_drive,
    DRIVE_SESSION_MANAGER,
    sync_v2_artifacts,
    v2_report_path,
    write_json,
)
from training.wsd_scheduler import get_wsd_schedule, phase_for_step


EARLY_STATUS_STEPS = {1, 2, 5, 10, 20, 50, 100}
HARD_EXAMPLE_KEEP = 16


EMERGENCY_SAVE_TIMEOUT_SECONDS = 20.0
_SAVE_COMPONENT_ORDER = ("model", "optimizer", "scheduler", "scaler")


def _utc_iso(ts: float | None = None) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() if ts is None else ts))


def _read_json(path: Path) -> dict[str, object] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _build_checkpoint_payload(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: object,
    mp: MixedPrecisionTrainer,
    global_step: int,
    epoch: int,
    best_loss: float,
    sessions_completed: int,
    mix_report: object,
    migration: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "tokenizer_schema_version": TOKENIZER_SCHEMA_VERSION,
        "tokenizer_contract": {
            "vocab_size": EXPECTED_TOKENIZER_VOCAB_SIZE,
            "special_token_ids": EXPECTED_SPECIAL_TOKEN_IDS,
        },
        "migration_provenance": migration,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": mp.state_dict(),
        "step": global_step,
        "global_step": global_step,
        "epoch": epoch,
        "best_loss": best_loss,
        "sessions_completed": sessions_completed,
        "model_config": model.model_config(),
        "mix_report": mix_report.to_dict(),
    }


def _emergency_save_with_timeout(payload: dict[str, object], ckpt_path: Path) -> bool:
    status: dict[str, object] = {"ok": False, "error": None}

    def _save() -> None:
        try:
            ordered_payload = {key: payload[key] for key in _SAVE_COMPONENT_ORDER}
            ordered_payload.update({k: v for k, v in payload.items() if k not in ordered_payload})
            atomic_save(ordered_payload, ckpt_path, drive_dir=None)
            status["ok"] = True
        except Exception as exc:
            status["error"] = repr(exc)

    worker = threading.Thread(target=_save, name="anra-emergency-save", daemon=True)
    worker.start()
    worker.join(timeout=EMERGENCY_SAVE_TIMEOUT_SECONDS)
    if worker.is_alive():
        print(
            f"[build_brain] emergency save timeout after {EMERGENCY_SAVE_TIMEOUT_SECONDS:.1f}s; process exit continues",
            flush=True,
        )
        return False
    if not bool(status["ok"]):
        print(f"[build_brain] emergency save failed: {status['error']}", flush=True)
        return False
    print("[build_brain] emergency save completed", flush=True)
    return True



def _resolve_checkpoint_path(checkpoint_path: str) -> Path:
    path = Path(checkpoint_path)
    raw = checkpoint_path.replace("\\", "/")
    if os.name == "nt" and raw.startswith("/tmp/"):
        local_tmp = ROOT / "output" / "tmp" / path.name
        print(f"[build_brain] remapping temporary checkpoint path to {local_tmp}", flush=True)
        return local_tmp
    return path if path.is_absolute() else (ROOT / path)


def _prepare_resume_target(checkpoint_path: Path, resume_from: str | None) -> None:
    if checkpoint_path.exists():
        return
    candidate = None
    if resume_from:
        candidate = _resolve_checkpoint_path(resume_from)
        if not candidate.exists():
            drive_copy = DRIVE_V2_CHECKPOINTS / candidate.name
            candidate = drive_copy if drive_copy.exists() else None
    if candidate is None:
        drive_copy = DRIVE_V2_CHECKPOINTS / checkpoint_path.name
        candidate = drive_copy if drive_copy.exists() else None
    if candidate is not None and candidate.exists():
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidate, checkpoint_path)
        print(f"[build_brain] restored checkpoint: {candidate} -> {checkpoint_path}", flush=True)


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


def _quick_eval_loss_value(result: float | dict[str, object]) -> float:
    return float(result["loss"]) if isinstance(result, dict) else float(result)


def _compact_eval_to_result(summary: dict[str, object], *, component: str = "training") -> EvalResult:
    score = float(summary.get("overall_score", 0.0) or 0.0)
    return EvalResult(
        component=component,
        mode=str(summary.get("mode", "compact_eval")),
        task_success_rate=score,
        avg_latency_ms=0.0,
        error_rate=0.0,
        notes="compact eval overall_score mapped to task_success_rate",
        raw=list(summary.get("results", [])) if isinstance(summary.get("results", []), list) else [],
    )


def train_anra_v2(
    *,
    data_path: str,
    checkpoint_path: str = "anra_v2_brain.pt",
    resume_from: str | None = None,
    batch_size: int = V2_TRAINING.batch_size,
    block_size: int = V2_MODEL.block_size,
    max_minutes: int = V2_TRAINING.session_minutes,
    answer_loss_weight: float = V2_TRAINING.answer_loss_weight,
    max_examples: int | None = None,
    own_ratio: float | None = None,
    identity_ratio: float | None = None,
    teacher_ratio: float | None = None,
    symbolic_ratio: float | None = None,
    replay_ratio: float | None = None,
    use_ouroboros: bool = False,
    model_size: str = "25m",
    optimizer_name: str = "auto",
) -> dict[str, object]:
    for required_component in ("training_loop", "data_mix", "evaluation"):
        if not is_enabled(required_component):
            raise RuntimeError(
                f"Required component is disabled at its call site: {required_component}"
            )
    print_session_dashboard()
    model_cfg, training_cfg = resolve_model_profile(model_size)
    is_frontier = model_size in {"1b", "frontier", "904m"}
    growth_teacher = None
    growth_alignment = None
    if model_size == "3b" and Path(checkpoint_path).name == V2_BRAIN_CHECKPOINT.name:
        checkpoint_path = str(V3_3B_CHECKPOINT)
    if model_size == "3b":
        from training.ssg import SovereignScalingGovernor

        target_exists = _resolve_checkpoint_path(checkpoint_path).exists()
        gate = SovereignScalingGovernor().check(
            phase="training" if target_exists else "growth"
        )
        print(f"[SSG] Checking {gate.phase} criteria...", flush=True)
        if not gate.allowed:
            for blocker in gate.blockers:
                print(f"[SSG] BLOCKED: {blocker}", flush=True)
            raise SystemExit(3)
    if is_frontier:
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            vram_gb = props.total_memory / 1024 ** 3
            print(f"[Trainer] GPU: {props.name}  VRAM: {vram_gb:.1f}GB")
            if vram_gb < 20:
                print(
                    f"[Trainer] WARNING: {vram_gb:.1f}GB VRAM is below the 20GB minimum.\n"
                    f"          1B training needs RTX 6000 Ada (48GB) or A100 (40-80GB).\n"
                    f"          Continuing; reduce batch_size if it OOMs."
                )
        if batch_size == V2_TRAINING.batch_size:
            batch_size = V2_1B_TRAINING.batch_size
        if block_size == V2_MODEL.block_size:
            block_size = V2_1B_FRONTIER.block_size
        if max_minutes == V2_TRAINING.session_minutes:
            max_minutes = V2_1B_TRAINING.session_minutes
        max_examples = max_examples or V2_1B_TRAINING.max_mixture_examples
        own_ratio = own_ratio if own_ratio is not None else V2_1B_TRAINING.own_ratio
        identity_ratio = identity_ratio if identity_ratio is not None else V2_1B_TRAINING.identity_ratio
        teacher_ratio = teacher_ratio if teacher_ratio is not None else V2_1B_TRAINING.teacher_ratio
        symbolic_ratio = symbolic_ratio if symbolic_ratio is not None else V2_1B_TRAINING.symbolic_ratio
        replay_ratio = replay_ratio if replay_ratio is not None else V2_1B_TRAINING.replay_ratio
        print(
            f"[Trainer] 1B FRONTIER MODE  "
            f"batch={training_cfg.batch_size}  grad_accum={training_cfg.grad_accum_steps}"
        )
    elif model_size == "3b":
        batch_size = 1 if batch_size == V2_TRAINING.batch_size else batch_size
        block_size = V2_3B.block_size if block_size == V2_MODEL.block_size else block_size
        max_examples = max_examples or V2_3B_TRAINING.max_mixture_examples
        print(
            f"[Trainer] 3B GROWTH MODE batch={batch_size} "
            f"grad_accum={training_cfg.grad_accum_steps}"
        )
    else:
        print("[Trainer] 25M BASE MODE")
    dataset_path = Path(data_path)
    tokenizer = load_or_build_v2_tokenizer(dataset_path=dataset_path)
    examples, mix_report = build_v2_training_examples(
        dataset_path=dataset_path,
        max_examples=max_examples,
        own_ratio=own_ratio,
        identity_ratio=identity_ratio,
        teacher_ratio=teacher_ratio,
        symbolic_ratio=symbolic_ratio,
        replay_ratio=replay_ratio,
        model_params=(
            2_918_251_520
            if model_size == "3b"
            else 904_535_040
            if is_frontier
            else 25_000_000
        ),
    )
    training_mix_controller = TrainingDataMixController(
        2_918_251_520
        if model_size == "3b"
        else 904_535_040
        if is_frontier
        else 25_000_000
    )
    if mix_report.active_weights:
        training_mix_controller.weights = dict(mix_report.active_weights)
    write_json(v2_report_path("mix_report"), mix_report.to_dict())
    ds = V2ConversationDataset(
        examples,
        tokenizer,
        block_size,
        answer_loss_weight=answer_loss_weight,
    )
    if len(ds) == 0:
        raise RuntimeError("V2ConversationDataset produced zero training windows.")
    def make_loader(active_weights: dict[str, float] | None = None) -> DataLoader:
        if active_weights is None:
            return DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=True,
                drop_last=False,
            )
        bucket_counts: dict[str, int] = {}
        buckets = [ds.bucket_for_sample(index) for index in range(len(ds))]
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
        return DataLoader(
            ds,
            batch_size=batch_size,
            sampler=sampler,
            drop_last=False,
        )

    loader = make_loader()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if is_frontier:
        hal_module = None
        if V2_1B_FRONTIER.use_hal:
            try:
                from anra.anra_paths import HAL_STATE_FILE
                from identity.hal import HALModule

                if HAL_STATE_FILE.exists():
                    hal_module = HALModule.load(str(HAL_STATE_FILE))
                    print("[Trainer] HAL state loaded from disk")
                else:
                    hal_module = HALModule()
                    print("[Trainer] HAL initialized fresh")
            except Exception as exc:
                print(f"[Trainer] HAL init failed: {exc}; training without HAL")
                hal_module = None
        model = build_frontier_model(hal_module=hal_module)
    elif model_size == "3b":
        model = build_3b_model()
        if not _resolve_checkpoint_path(checkpoint_path).exists():
            parent_path = canonical_v2_checkpoint("brain")
            if not parent_path.exists():
                raise RuntimeError("3B growth requires a promoted frontier checkpoint.")
            parent = build_frontier_model()
            load_checkpoint(parent, None, None, None, parent_path, device=torch.device("cpu"), strict=False)
            from training.csii import (
                CrossScaleIdentityInheritance,
                GrowthAlignmentController,
            )

            growth = CrossScaleIdentityInheritance.grow(
                parent, model, source_checkpoint=parent_path
            )
            frozen_tokens = ds[0][0][: min(16, block_size)].unsqueeze(0)
            parity = CrossScaleIdentityInheritance.verify_parity(
                parent,
                model,
                frozen_tokens,
            )
            growth_payload = {
                **growth.__dict__,
                **parity,
                "frozen_corpus_hash": __import__("hashlib").sha256(
                    frozen_tokens.numpy().tobytes()
                ).hexdigest(),
            }
            CrossScaleIdentityInheritance.write_report(
                growth_payload,
                MODEL_GROWTH_REPORT,
            )
            growth_candidate = _resolve_checkpoint_path(checkpoint_path).with_suffix(
                ".growth-candidate.pt"
            )
            atomic_save(
                {
                    "schema_version": 3,
                    "model_state_dict": model.state_dict(),
                    "model_config": model.model_config(),
                    "growth_report": growth_payload,
                },
                growth_candidate,
                drive_dir=None,
            )
            final_gate = SovereignScalingGovernor().check(phase="training")
            if not final_gate.allowed:
                QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)
                quarantined = QUARANTINE_DIR / growth_candidate.name
                shutil.move(str(growth_candidate), quarantined)
                for blocker in final_gate.blockers:
                    print(f"[SSG] BLOCKED: {blocker}", flush=True)
                raise SystemExit(3)
            growth_teacher = parent
            growth_alignment = GrowthAlignmentController(
                parent,
                model,
                identity_layers=growth.identity_layers,
            )
            print(
                f"[CSII] Grew frontier {growth.source_width}x{growth.source_layers} "
                f"to {growth.target_width}x{growth.target_layers}",
                flush=True,
            )
    else:
        model = build_v2_model(vocab_size=tokenizer.vocab_size, block_size=block_size)
    if (is_frontier or model_size == "3b") and getattr(training_cfg, "gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
        print("[build_brain] Gradient checkpointing enabled for 1B model", flush=True)
    if hasattr(model, "disable_kv_cache"):
        model.disable_kv_cache()
    if use_ouroboros:
        from ouroboros import OuroborosDecoder

        model = OuroborosDecoder(model, n_passes=3)
    model = model.to(device)
    if growth_teacher is not None:
        growth_teacher = growth_teacher.to(device)
    mp = MixedPrecisionTrainer(device=device)
    learning_rate = float(getattr(training_cfg, "learning_rate", 3e-4))
    optimizer = build_optimizer(
        model,
        lr=learning_rate,
        weight_decay=float(getattr(training_cfg, "weight_decay", 0.1)),
        optimizer_name=optimizer_name,
    )
    if growth_alignment is not None:
        growth_alignment.configure_trainable_parameters(0)
    optimizer_report = getattr(optimizer, "_anra_optimizer_report", {"selected": {"actual": optimizer_name}})
    write_json(v2_report_path("optimizer_bakeoff"), optimizer_report)
    total_steps = int(getattr(training_cfg, "max_steps", 50_000))
    warmup_steps = int(getattr(training_cfg, "warmup_steps", 100))
    scheduler = get_wsd_schedule(
        optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        min_lr_ratio=float(getattr(training_cfg, "min_lr", learning_rate * 0.1)) / learning_rate,
    )
    regret_scheduler = DynamicRegretScheduler(None, eta_base=learning_rate)
    regret_scheduler.load(REGRET_STATE)
    cdr = CorrectedFailureCurriculum(FAILURE_REPLAY_DATASET)
    protected_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and is_identity_parameter(name, parameter)
    ]
    pcgrad = PCGradAccumulator(protected_parameters)
    pcgrad_reports = []
    annealing_started = False

    requested_checkpoint = Path(checkpoint_path)
    ckpt_path = requested_checkpoint if requested_checkpoint.is_absolute() else ROOT / requested_checkpoint
    resume_path = Path(resume_from) if resume_from else ckpt_path
    if not resume_path.is_absolute():
        resume_path = ROOT / resume_path
    ckpt: dict[str, object] = {}
    global_step = 0
    epoch = 0
    best_loss = float("inf")
    checkpoint_migration: dict[str, object] | None = None

    registration_ts = time.time()
    signal_state: dict[str, object] = {
        "registered_at": registration_ts,
        "registered_at_iso": _utc_iso(registration_ts),
        "triggered": False,
        "signal": None,
        "emergency_save_completed": None,
    }

    def _handle_sigterm(sig_num: int, _frame: object) -> None:
        signal_state["triggered"] = True
        signal_state["signal"] = sig_num
        print(
            f"[build_brain] SIGTERM handler invoked (signal={sig_num}) at {_utc_iso()}.",
            flush=True,
        )
        sessions_completed = int(ckpt.get("sessions_completed", 0) + 1) if "ckpt" in locals() else 1
        payload = _build_checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            mp=mp,
            global_step=global_step,
            epoch=epoch,
            best_loss=best_loss,
            sessions_completed=sessions_completed,
            mix_report=mix_report,
            migration=checkpoint_migration,
        )
        ok = _emergency_save_with_timeout(payload, ckpt_path)
        signal_state["emergency_save_completed"] = ok
        print(f"[build_brain] SIGTERM emergency save status={ok}", flush=True)
        raise SystemExit(128 + sig_num)

    signal.signal(signal.SIGTERM, _handle_sigterm)
    print(
        f"[build_brain] SIGTERM handler registered at {signal_state['registered_at_iso']} (pre-training).",
        flush=True,
    )

    start_step = 0
    best_loss = float("inf")
    session_start_loss = float("inf")

    # ── AUTO-RESUME ──────────────────────────────────────────────────────────────
    load_path = ckpt_path if ckpt_path.exists() else resume_path
    if load_path.exists():
        print(f"[Resume] Found checkpoint: {load_path}", flush=True)
        ckpt = safe_torch_load(load_path, map_location=device)
        resume_state = load_checkpoint(model, optimizer, scheduler, mp, load_path, device=device, strict=False)
        if resume_state["loaded"]:
            checkpoint_migration = dict(resume_state.get("migration", {}))
            start_step = int(resume_state["global_step"])
            best_loss = float(resume_state["best_loss"])
            session_start_loss = best_loss
            print(f"[Resume] Resuming from step={start_step}  best_loss={best_loss:.4f}", flush=True)
        else:
            print("[Resume] Checkpoint not loaded — starting from scratch", flush=True)
    else:
        print("[Resume] No checkpoint found — starting from scratch", flush=True)
    # ─────────────────────────────────────────────────────────────────────────────

    try:
        session_start_result = quick_eval_loss(model, ds, device=device, max_examples=100, batch_size=batch_size, pad_id=tokenizer.pad_token_id)
        session_start_loss = _quick_eval_loss_value(session_start_result)
    except Exception as exc:
        print(f"[build_brain] quick eval at session_start failed: {exc}", flush=True)
        session_start_loss = best_loss
    if math.isfinite(session_start_loss):
        regret_scheduler.session_start(session_start_loss)

    global_step = start_step
    epoch = 0

    start = time.time()
    end_at = start + max_minutes * 60
    initial_step = start_step
    session_step = 0
    optimizer.zero_grad(set_to_none=True)
    rolling_loss = 0.0
    rolling_count = 0
    accum_micro_steps = 0
    last_avg_loss = best_loss if math.isfinite(best_loss) else 0.0
    first_batch_wall = None
    hard_examples: list[tuple[float, int]] = []
    answer_weighted_tokens = 0.0
    total_target_tokens = 0.0

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0.0
    summary = model_summary(model)
    eff_batch = batch_size * training_cfg.grad_accum_steps

    print("", flush=True)
    print("=" * 62, flush=True)
    print("  AN-RA V2 TRAINING SESSION", flush=True)
    print("=" * 62, flush=True)
    print(f"  GPU          : {gpu_name} ({gpu_mem:.1f} GB)", flush=True)
    print(f"  Parameters   : {summary['parameters']:,}", flush=True)
    print(
        f"  Micro batch  : {batch_size}  |  Grad accum : {training_cfg.grad_accum_steps}  |  Eff batch : {eff_batch}",
        flush=True,
    )
    print(f"  Session time : {max_minutes} minutes", flush=True)
    print(
        f"  Resuming     : step {global_step:,}  |  best loss {best_loss if math.isfinite(best_loss) else float('inf'):.4f}",
        flush=True,
    )
    print(f"  Checkpoint   : {ckpt_path}", flush=True)
    print(f"  Data mix     : {mix_report.realized_counts}", flush=True)
    print("=" * 62, flush=True)
    print("", flush=True)

    def _autosave() -> None:
        sync_to_drive("brain")
        sync_to_drive("tokenizer")

    DRIVE_SESSION_MANAGER.start_autosave(_autosave)
    DRIVE_SESSION_MANAGER.register_sigterm_hook(_autosave)

    while time.time() < end_at:
        epoch += 1
        for xb, yb, wb, sample_idx in loader:
            if first_batch_wall is None:
                first_batch_wall = time.time()
            xb = xb.to(device)
            yb = yb.to(device)
            wb = wb.to(device)
            with mp.autocast():
                logits, _ = model(xb)
                batch_loss, sample_losses = _weighted_loss(
                    logits,
                    yb,
                    wb,
                    pad_id=tokenizer.pad_token_id,
                )
                if growth_alignment is not None:
                    alignment_step = max(0, global_step - initial_step)
                    alignment_penalty = growth_alignment.alignment_loss(
                        xb,
                        step=alignment_step,
                        target_logits=logits,
                    )
                    batch_loss = batch_loss + alignment_penalty
                loss = batch_loss / training_cfg.grad_accum_steps

            if not torch.isfinite(batch_loss):
                cdr.capture_step_failure(
                    input_tokens=xb,
                    target_tokens=yb,
                    predicted_tokens=torch.nan_to_num(logits).argmax(dim=-1),
                    loss=float("inf"),
                    step=global_step,
                    tokenizer=tokenizer,
                    category="execution",
                )
                optimizer.zero_grad(set_to_none=True)
                pcgrad.clear()
                if ckpt_path.exists():
                    load_checkpoint(
                        model,
                        optimizer,
                        scheduler,
                        mp,
                        ckpt_path,
                        device=device,
                        strict=False,
                    )
                print(
                    "[Recovery] Non-finite batch quarantined; last-good checkpoint reloaded.",
                    flush=True,
                )
                continue

            owner_flags = [
                ds.bucket_for_sample(index) in {"own", "identity"}
                for index in sample_idx.tolist()
            ]
            owner_positions = [i for i, flag in enumerate(owner_flags) if flag]
            other_positions = [i for i, flag in enumerate(owner_flags) if not flag]
            owner_loss = (
                sample_losses[owner_positions].mean() / training_cfg.grad_accum_steps
                if owner_positions
                else None
            )
            other_loss = (
                sample_losses[other_positions].mean() / training_cfg.grad_accum_steps
                if other_positions
                else None
            )
            if owner_loss is not None or other_loss is not None:
                pcgrad.accumulate(
                    owner_loss=owner_loss,
                    other_loss=other_loss,
                    grad_scale=mp.scale,
                )

            mp.backward(loss)
            rolling_loss += float(loss.item() * training_cfg.grad_accum_steps)
            rolling_count += 1
            accum_micro_steps += 1
            answer_weighted_tokens += float((wb > 1.0).sum().item())
            total_target_tokens += float((yb != tokenizer.pad_token_id).sum().item())

            for sample_loss, example_index in zip(sample_losses.detach().cpu().tolist(), sample_idx.tolist()):
                entry = (float(sample_loss), int(example_index))
                if len(hard_examples) < HARD_EXAMPLE_KEEP:
                    heapq.heappush(hard_examples, entry)
                elif entry[0] > hard_examples[0][0]:
                    heapq.heapreplace(hard_examples, entry)

            if accum_micro_steps >= training_cfg.grad_accum_steps:
                pcgrad_reports.extend(pcgrad.materialize())
                if growth_alignment is not None:
                    growth_alignment.mask_inactive_gradients()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                mp.step(optimizer)
                mp.update()
                scheduler.step()
                regret_lr = regret_scheduler.update(
                    reward=max(0.0, 1.0 - float(batch_loss.item()))
                )
                multiplier = max(0.5, min(1.5, regret_lr / max(learning_rate, 1e-12)))
                scheduled_lrs = scheduler.get_last_lr()
                for group, scheduled_lr in zip(optimizer.param_groups, scheduled_lrs):
                    group["lr"] = scheduled_lr * multiplier
                optimizer.zero_grad(set_to_none=True)
                pcgrad.clear()
                global_step += 1
                if growth_alignment is not None:
                    growth_alignment.configure_trainable_parameters(
                        global_step - initial_step
                    )
                session_step += 1
                accum_micro_steps = 0

                avg_loss = rolling_loss / max(1, rolling_count)
                loss_val = avg_loss
                last_avg_loss = avg_loss
                best_loss = min(best_loss, avg_loss) if math.isfinite(best_loss) else avg_loss
                phase = phase_for_step(
                    global_step,
                    warmup_steps=warmup_steps,
                    total_steps=total_steps,
                )
                if phase.annealing_started and not annealing_started:
                    annealing_started = True
                    annealed_weights = training_mix_controller.enter_annealing_phase()
                    loader = make_loader(annealed_weights)
                    write_json(
                        v2_report_path("mix_control"),
                        {
                            "generated_at": time.time(),
                            "weights": annealed_weights,
                            "source": "wsd_owner_annealing",
                            "step": global_step,
                        },
                    )
                    print("[WSD] Entered decay phase; owner annealing is active.", flush=True)

                running_mean = rolling_loss / max(1, rolling_count)
                if float(batch_loss.item()) > max(3.0 * running_mean, running_mean + 2.0):
                    with torch.no_grad():
                        cdr.capture_step_failure(
                            input_tokens=xb,
                            target_tokens=yb,
                            predicted_tokens=logits.argmax(dim=-1),
                            loss=float(batch_loss.item()),
                            step=global_step,
                            tokenizer=tokenizer,
                        )
                if global_step % 1000 == 0:
                    flushed = cdr.flush_to_dataset(FAILURE_REPLAY_DATASET)
                    if flushed:
                        added = ds.reload_replay_bucket()
                        print(
                            f"[CDR] Flushed {flushed} verified corrections; "
                            f"reloaded {added} replay examples.",
                            flush=True,
                        )

                elapsed_min = (time.time() - start) / 60.0
                if session_step % 10 == 0:
                    print(
                        f"  step={global_step:6d}"
                        f"  loss={loss_val:.4f}"
                        f"  best={best_loss:.4f}"
                        f"  elapsed={elapsed_min:.1f}m",
                        flush=True,
                    )

                if global_step in EARLY_STATUS_STEPS or global_step % 200 == 0:
                    elapsed_min = (time.time() - start) / 60.0
                    remaining_min = max(0.0, (end_at - time.time()) / 60.0)
                    startup_note = ""
                    if global_step in EARLY_STATUS_STEPS and first_batch_wall is not None:
                        startup_note = f"  startup={(first_batch_wall - start):.1f}s"
                    print(
                        f"  step={global_step:6d}  loss={avg_loss:.4f}  best={best_loss:.4f}  "
                        f"elapsed={elapsed_min:.1f}m  remaining={remaining_min:.1f}m{startup_note}",
                        flush=True,
                    )

            if time.time() >= end_at:
                break

    if accum_micro_steps > 0:
        pcgrad_reports.extend(pcgrad.materialize())
        if growth_alignment is not None:
            growth_alignment.mask_inactive_gradients()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        mp.step(optimizer)
        mp.update()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        pcgrad.clear()
        global_step += 1
        session_step += 1
        avg_loss = rolling_loss / max(1, rolling_count)
        last_avg_loss = avg_loss
        best_loss = min(best_loss, avg_loss) if math.isfinite(best_loss) else avg_loss
        print(
            f"  step={global_step:6d}  loss={avg_loss:.4f}  best={best_loss:.4f}  "
            f"elapsed={(time.time() - start) / 60.0:.1f}m  remaining={max(0.0, (end_at - time.time()) / 60.0):.1f}m"
            f"  partial_accum={accum_micro_steps}/{training_cfg.grad_accum_steps}",
            flush=True,
        )

    if global_step > initial_step and global_step % 200 != 0:
        elapsed_min = (time.time() - start) / 60.0
        remaining_min = max(0.0, (end_at - time.time()) / 60.0)
        print(
            f"  step={global_step:6d}  loss={last_avg_loss:.4f}  best={best_loss:.4f}  "
            f"elapsed={elapsed_min:.1f}m  remaining={remaining_min:.1f}m",
            flush=True,
        )

    payload = _build_checkpoint_payload(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        mp=mp,
        global_step=global_step,
        epoch=epoch,
        best_loss=best_loss,
        sessions_completed=(int(ckpt.get("sessions_completed", 0) + 1) if "ckpt" in locals() else 1),
        mix_report=mix_report,
        migration=checkpoint_migration,
    )
    atomic_save(payload, ckpt_path, drive_dir=None)

    metrics = {
        "generated_at": time.time(),
        "elapsed_minutes": round((time.time() - start) / 60.0, 2),
        "session_minutes_target": max_minutes,
        "global_step": global_step,
        "epoch": epoch,
        "best_loss": round(best_loss, 4),
        "last_avg_loss": round(last_avg_loss, 4),
        "effective_batch_size": eff_batch,
        "grad_accum_steps": training_cfg.grad_accum_steps,
        "answer_loss_weight": answer_loss_weight,
        "model_size": model_size,
        "optimizer": optimizer_report,
        "answer_supervision_ratio": round(ds.answer_supervision_ratio, 4),
        "reply_token_ratio_seen": round(answer_weighted_tokens / max(1.0, total_target_tokens), 4),
        "target_tokens_seen": int(total_target_tokens),
        "model_config": model.model_config(),
        "checkpoint_path": str(ckpt_path),
        "mix_report": mix_report.to_dict(),
        "signal_handler": signal_state,
        "scheduler": {
            "name": "wsd",
            "warmup_steps": warmup_steps,
            "total_steps": total_steps,
            "annealing_started": annealing_started,
        },
        "pcgrad": {
            "comparisons": len(pcgrad_reports),
            "conflict_rate": (
                sum(report.conflict for report in pcgrad_reports) / len(pcgrad_reports)
                if pcgrad_reports
                else 0.0
            ),
            "mean_cosine": (
                sum(report.cosine for report in pcgrad_reports) / len(pcgrad_reports)
                if pcgrad_reports
                else 0.0
            ),
            "protected_parameters": len(protected_parameters),
            "patterns": list(IDENTITY_PARAMETER_PATTERNS),
        },
        "cdr": cdr.report(),
        "continual_learning": assess_continual_readiness(
            int(mix_report.replay_available) + int(cdr.report()["verified"])
        ),
    }
    write_json(v2_report_path("metrics"), metrics)
    write_json(CDR_REPORT, cdr.report())

    hard_examples_report = [
        {
            "loss": round(loss_value, 4),
            "sample_index": sample_index,
            "preview": ds.snippet(sample_index),
        }
        for loss_value, sample_index in sorted(hard_examples, key=lambda item: item[0], reverse=True)
    ]
    write_json(
        v2_report_path("hard_examples"),
        {
            "generated_at": time.time(),
            "answer_loss_weight": answer_loss_weight,
            "examples": hard_examples_report,
        },
    )

    prev_eval_summary = None
    eval_path = v2_report_path("eval_summary")
    if eval_path.exists():
        try:
            prev_eval_summary = json.loads(eval_path.read_text(encoding="utf-8"))
        except Exception:
            prev_eval_summary = None

    eval_summary = run_compact_eval(model, tokenizer, device=device, output=True)
    civ_similarity = float(
        eval_summary.get(
            "civ_similarity",
            eval_summary.get("category_scores", {}).get("identity", 0.0),
        )
    )
    try:
        adjusted_weights = training_mix_controller.update_from_civ(civ_similarity)
        write_json(
            v2_report_path("mix_control"),
            {
                "generated_at": time.time(),
                "civ_similarity": civ_similarity,
                "weights": adjusted_weights,
                "source": "post_eval_ogrs",
            },
        )
        print(
            f"[OGRS] CIV similarity: {civ_similarity:.3f}; "
            f"owner weight: {adjusted_weights['owner']:.3f}",
            flush=True,
        )
    except RuntimeError as exc:
        SOVEREIGNTY_EVENTS.parent.mkdir(parents=True, exist_ok=True)
        with SOVEREIGNTY_EVENTS.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {
                        "timestamp": time.time(),
                        "event": "CIV_TRAINING_PAUSE",
                        "civ_similarity": civ_similarity,
                        "message": str(exc),
                    }
                )
                + "\n"
            )
        print(f"[OGRS] {exc}; checkpoint quarantined and training paused.", flush=True)
        metrics["training_paused"] = True
        metrics["pause_reason"] = str(exc)
    from engine.trajectories import TrajectoryStore
    from evaluation.metrics import build_snapshot, persist_snapshot

    cdr_report = cdr.report()
    ibs_report = _read_json(IBS_LATEST) or {}
    ibs_dimensions = (
        dict(ibs_report.get("dimensions", {}))
        if isinstance(ibs_report.get("dimensions"), dict)
        else {}
    )
    rlvr_report_path = v2_report_path("rlvr_report")
    rlvr_report = _read_json(rlvr_report_path) or {}
    memory_report_path = v2_report_path("memory_benchmark")
    memory_report = _read_json(memory_report_path) or {}
    combined_memory = (
        dict(memory_report.get("combined", {}))
        if isinstance(memory_report.get("combined"), dict)
        else {}
    )
    improvement_report_path = v2_report_path("improvement_report")
    improvement_report = _read_json(improvement_report_path) or {}
    snapshot = build_snapshot(
        checkpoint=str(ckpt_path),
        measurements={
            "M-01": float(ibs_dimensions.get("owner_task", 0.0)),
            "M-02": float(total_target_tokens),
            "M-03": civ_similarity,
            "M-04": float(TrajectoryStore().verified_count()),
            "M-05": float(rlvr_report.get("verifier_pass_rate", 0.0)),
            "M-06": float(rlvr_report.get("truth_checking_coverage", 0.0)),
            "M-07": float(cdr_report["closure_rate"]),
            "M-08": float(combined_memory.get("recall_at_3", 0.0)),
            "M-09": float(bool(improvement_report.get("promotion_allowed", False))),
            "M-10": 0.0,
            "M-11": 0.0,
            "M-12": float(ibs_report.get("overall", 0.0)),
        },
        targets={
            "M-01": 0.60,
            "M-02": 21_000_000_000.0,
            "M-03": 0.88,
            "M-04": 1_000.0,
            "M-05": 0.70,
            "M-06": 0.95,
            "M-07": 0.90,
            "M-08": 0.85,
            "M-09": 1.0,
            "M-10": 0.88,
            "M-11": 0.99,
            "M-12": 0.60,
        },
        evidence={
            "M-01": str(IBS_LATEST),
            "M-02": str(v2_report_path("metrics")),
            "M-03": str(v2_report_path("eval_summary")),
            "M-04": "trajectory_store.jsonl",
            "M-05": str(rlvr_report_path),
            "M-06": str(rlvr_report_path),
            "M-07": str(CDR_REPORT),
            "M-08": str(memory_report_path),
            "M-09": str(improvement_report_path),
            "M-10": "no sovereignty-accuracy evidence produced",
            "M-11": "service telemetry",
            "M-12": str(IBS_LATEST),
        },
    )
    persist_snapshot(snapshot)
    metrics["metric_snapshot"] = snapshot.to_dict()
    write_json(v2_report_path("metrics"), metrics)
    if isinstance(prev_eval_summary, dict):
        try:
            harness = EvalHarness()
            regression_report = harness.compare(
                _compact_eval_to_result(prev_eval_summary),
                _compact_eval_to_result(eval_summary),
            )
            if regression_report.regressed:
                rd = regression_report.to_dict()
                harness.save_report(regression_report)
                severity = rd.get("severity", "low")
                print(f"[REGRESSION] severity={severity}: {rd}", flush=True)
                if severity in ("critical", "high"):
                    print("[ABORT] Restoring last good checkpoint to protect model.", flush=True)
                    prev_ckpt = canonical_v2_checkpoint("brain")
                    if prev_ckpt.exists():
                        load_checkpoint(model, optimizer, scheduler, mp, prev_ckpt, device=device, strict=False)
                        print("[ABORT] Checkpoint restored. Stopping session.", flush=True)
                    return {
                        "checkpoint_path": str(ckpt_path),
                        "global_step": global_step,
                        "best_loss": best_loss,
                        "eval_summary": eval_summary,
                        "mix_report": mix_report.to_dict(),
                        "aborted": True,
                        "regression": rd,
                    }
                print("[WARN] Low severity regression - continuing with caution.", flush=True)
        except Exception as exc:
            print(f"[build_brain] regression check skipped: {exc}", flush=True)

    try:
        session_end_result = quick_eval_loss(model, ds, device=device, max_examples=100, batch_size=batch_size, pad_id=tokenizer.pad_token_id)
        session_end_loss = _quick_eval_loss_value(session_end_result)
        regret_lr = regret_scheduler.session_end(session_end_loss, global_step - initial_step)
        regret_scheduler.save(REGRET_STATE)
        print(f"  Dynamic regret lr : {regret_lr:.8f}", flush=True)
    except Exception as exc:
        print(f"[build_brain] quick eval at session_end failed: {exc}", flush=True)
    sync_v2_artifacts(
        ckpt_path,
        tokenizer_path=V2_TOKENIZER_FILE,
        extra_paths=[
            v2_report_path("metrics"),
            v2_report_path("hard_examples"),
            v2_report_path("eval_summary"),
            v2_report_path("mix_report"),
        ],
    )
    sync_to_drive("brain")
    sync_to_drive("tokenizer")
    sync_to_drive("eval_summary")

    elapsed_total = time.time() - start
    print("", flush=True)
    print("=" * 62, flush=True)
    print("  V2 SESSION COMPLETE", flush=True)
    print("=" * 62, flush=True)
    print(f"  Steps this session : {global_step - initial_step:,}", flush=True)
    print(f"  Total steps        : {global_step:,}", flush=True)
    print(f"  Best loss          : {best_loss:.4f}", flush=True)
    print(f"  Eval score         : {float(eval_summary.get('overall_score', 0.0)):.4f}", flush=True)
    print(f"  Time elapsed       : {elapsed_total / 60:.1f} minutes", flush=True)
    print(f"  Checkpoint saved   : {ckpt_path}", flush=True)
    print("  Drive synced       : yes", flush=True)
    print("=" * 62, flush=True)
    print("", flush=True)

    # ── SESSION SUMMARY ──────────────────────────────────────────────────────────
    print("\n" + "=" * 50, flush=True)
    print("SESSION COMPLETE", flush=True)
    print(f"  Steps this session : {session_step}", flush=True)
    print(f"  Total steps ever   : {global_step}", flush=True)
    print(f"  Loss at start      : {session_start_loss:.4f}", flush=True)
    print(f"  Best loss achieved : {best_loss:.4f}", flush=True)
    if session_start_loss != float("inf"):
        improvement = session_start_loss - best_loss
        direction = "improved" if improvement > 0 else "no improvement"
        print(f"  Improvement        : {improvement:+.4f}  ({direction})", flush=True)
    print("=" * 50 + "\n", flush=True)
    # ─────────────────────────────────────────────────────────────────────────────

    return {
        "checkpoint_path": str(ckpt_path),
        "global_step": global_step,
        "best_loss": best_loss,
        "eval_summary": eval_summary,
        "mix_report": mix_report.to_dict(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Canonical An-Ra base trainer")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--checkpoint_path", default="anra_v2_brain.pt")
    parser.add_argument("--resume_from", default=None)
    parser.add_argument("--batch_size", type=int, default=V2_TRAINING.batch_size)
    parser.add_argument("--block_size", type=int, default=V2_MODEL.block_size)
    parser.add_argument("--max_minutes", type=int, default=V2_TRAINING.session_minutes)
    parser.add_argument(
        "--model-size",
        choices=["25m", "1b", "frontier", "904m", "3b"],
        default="25m",
    )
    parser.add_argument("--answer_loss_weight", type=float, default=V2_TRAINING.answer_loss_weight)
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--own_ratio", type=float, default=None)
    parser.add_argument("--identity_ratio", type=float, default=None)
    parser.add_argument("--teacher_ratio", type=float, default=None)
    parser.add_argument("--symbolic_ratio", type=float, default=None)
    parser.add_argument("--replay_ratio", type=float, default=None)
    parser.add_argument("--optimizer", choices=["auto", "adamw", "muon", "scale", "galore"], default="auto")
    args = parser.parse_args()
    result = train_anra_v2(
        data_path=args.data_path,
        checkpoint_path=args.checkpoint_path,
        resume_from=args.resume_from,
        batch_size=args.batch_size,
        block_size=args.block_size,
        max_minutes=args.max_minutes,
        answer_loss_weight=args.answer_loss_weight,
        max_examples=args.max_examples,
        own_ratio=args.own_ratio,
        identity_ratio=args.identity_ratio,
        teacher_ratio=args.teacher_ratio,
        symbolic_ratio=args.symbolic_ratio,
        replay_ratio=args.replay_ratio,
        model_size=args.model_size,
        optimizer_name=args.optimizer,
    )
    print(result, flush=True)


if __name__ == "__main__":
    main()
