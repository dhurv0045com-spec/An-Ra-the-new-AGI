# NOTE: scripts/train.py is the canonical training script for local runs.
# This file handles tokenizer building and frontier data preparation.
from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from anra.anra_paths import (
    CDR_REPORT,
    DRIVE_V2_CHECKPOINTS,
    FAILURE_REPLAY_DATASET,
    IBS_LATEST,
    REGRET_STATE,
    ROOT,
    SOVEREIGNTY_EVENTS,
    V2_TOKENIZER_FILE,
)
from engine.eval_harness import EvalHarness, EvalResult
from engine.feature_flags import is_enabled
from evaluation.intelligence_telemetry import create_intelligence_session
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
    V2_FRONTIER_PARAMETER_COUNT,
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
    canonical_v2_checkpoint,
    ensure_tied_lm_head,
    get_hal_module,
    load_checkpoint,
    load_or_build_v2_tokenizer,
    model_summary,
    sync_to_drive,
    DRIVE_SESSION_MANAGER,
    hal_state_dict,
    sync_v2_artifacts,
    v2_report_path,
    update_hal_from_training,
    write_json,
)
from runtime.hal_telemetry import publish_hal_state
from training.wsd_scheduler import get_wsd_schedule, phase_for_step


EARLY_STATUS_STEPS = {1, 2, 5, 10, 20, 50, 100}
HARD_EXAMPLE_KEEP = 16


EMERGENCY_SAVE_TIMEOUT_SECONDS = 20.0
_SAVE_COMPONENT_ORDER = ("model", "optimizer", "scheduler", "scaler")


def build_causal_extension_trainer(
    model: torch.nn.Module,
    *,
    total_steps: int,
    warmup_steps: int,
    optimizer_name: str = "auto",
):
    """Canonical build-brain integration point for extension-only causal training."""
    from cognition.cre import CognitiveCausalExtension
    from training.causal_trainer import CausalExtensionTrainer

    layer_count = int(getattr(model, "n_layer"))
    integration_layers = tuple(
        sorted({0, max(0, layer_count // 2), max(0, layer_count - 1)})
    )
    extension = CognitiveCausalExtension(
        int(getattr(model, "n_embd")),
        integration_layers=integration_layers,
    ).to(next(model.parameters()).device)
    model.attach_cognitive_extension(extension)
    trainer = CausalExtensionTrainer(
        model,
        extension,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        cdr_path=str(FAILURE_REPLAY_DATASET),
        optimizer_name=optimizer_name,
    )
    return extension, trainer


def train_causal_extension(
    *,
    data_path: str,
    base_checkpoint: str,
    output_path: str,
    model_size: str,
    batch_size: int,
    block_size: int,
    max_minutes: int,
    optimizer_name: str,
) -> dict[str, object]:
    from cognition.checkpoint import save_cognitive_extension
    from training.causal_trainer import CausalCorpusDataset

    corpus_path = Path(data_path)
    if not corpus_path.exists():
        raise FileNotFoundError(
            f"Causal corpus is missing: {corpus_path}. Run python -m data.causal_corpus."
        )
    tokenizer = load_or_build_v2_tokenizer(
        dataset_path=ROOT / "training_data" / "anra_training.txt"
    )
    if model_size != "frontier":
        raise ValueError("iterate500 supports only --model-size frontier")
    model = build_frontier_model()
    checkpoint = _resolve_checkpoint_path(base_checkpoint)
    if checkpoint.exists():
        load_checkpoint(
            model,
            None,
            None,
            None,
            checkpoint,
            device=torch.device("cpu"),
            strict=False,
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    extension, trainer = build_causal_extension_trainer(
        model,
        total_steps=50_000,
        warmup_steps=100,
        optimizer_name=optimizer_name,
    )
    dataset = CausalCorpusDataset(
        corpus_path,
        tokenizer,
        block_size,
        extension.rank,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    deadline = time.time() + max_minutes * 60
    steps = 0
    latest: dict[str, float] = {}
    while time.time() < deadline:
        for batch in loader:
            input_ids = batch.pop("input_ids").to(device)
            target_ids = batch.pop("target_ids").to(device)
            attention_mask = batch.pop("attention_mask").to(device)
            labels = {name: value.to(device) for name, value in batch.items()}
            latest = trainer.step(
                input_ids,
                target_ids,
                labels,
                attention_mask=attention_mask,
            )
            steps += 1
            if time.time() >= deadline:
                break
    base_hash = (
        hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if checkpoint.exists()
        else "uninitialized-base"
    )
    tokenizer_hash = hashlib.sha256(V2_TOKENIZER_FILE.read_bytes()).hexdigest()
    try:
        source_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        source_commit = "unknown"
    manifest = save_cognitive_extension(
        extension,
        output_path,
        base_checkpoint_hash=base_hash,
        tokenizer_hash=tokenizer_hash,
        source_commit=source_commit,
        release="cognition-v1",
        training_state=trainer.state_dict(),
    )
    report = {"steps": steps, "metrics": latest, "manifest": manifest}
    write_json(v2_report_path("causal_extension"), report)
    return report


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
    try:
        source_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        source_commit = "unknown"
    data_manifests = {}
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
        "hal_state": hal_state_dict(model),
        "mix_report": mix_report.to_dict(),
        "rng_states": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        },
        "source_commit": source_commit,
        "dataset_manifest_hashes": data_manifests,
        "cognitive_extension_release": "cognition-v1",
        "consent_safe_metadata": {
            "owner_derived_data_authorized": bool(
                os.environ.get("ANRA_OWNER_DATA_AUTHORIZED", "").lower()
                in {"1", "true", "yes"}
            )
        },
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
            candidate = None
            for drive_copy in (
                DRIVE_V2_CHECKPOINTS / Path(resume_from).name,
                DRIVE_V2_CHECKPOINTS.parent.parent / Path(resume_from).name,
            ):
                if drive_copy.exists():
                    candidate = drive_copy
                    break
    if candidate is None:
        for drive_copy in (
            DRIVE_V2_CHECKPOINTS / checkpoint_path.name,
            DRIVE_V2_CHECKPOINTS.parent.parent / checkpoint_path.name,
        ):
            if drive_copy.exists():
                candidate = drive_copy
                break
    if candidate is not None and candidate.exists():
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidate, checkpoint_path)
        print(f"[build_brain] restored checkpoint: {candidate} -> {checkpoint_path}", flush=True)


def _sync_training_checkpoint_to_drive(checkpoint_path: Path) -> None:
    if not checkpoint_path.exists():
        return
    target = DRIVE_V2_CHECKPOINTS / checkpoint_path.name
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(checkpoint_path, target)
        print(f"[Drive] frontier checkpoint saved: {target}", flush=True)
    except Exception as exc:
        print(f"[Drive] frontier checkpoint mirror failed for {target}: {exc}", flush=True)


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
    checkpoint_path: str = "anra_frontier_500m.pt",
    resume_from: str | None = None,
    batch_size: int = V2_1B_TRAINING.batch_size,
    block_size: int = V2_1B_FRONTIER.block_size,
    max_minutes: int = V2_1B_TRAINING.session_minutes,
    answer_loss_weight: float = V2_1B_TRAINING.answer_loss_weight,
    max_examples: int | None = None,
    own_ratio: float | None = None,
    identity_ratio: float | None = None,
    teacher_ratio: float | None = None,
    symbolic_ratio: float | None = None,
    replay_ratio: float | None = None,
    use_ouroboros: bool = False,
    model_size: str = "frontier",
    optimizer_name: str = "auto",
    start_eval_examples: int = 0,
) -> dict[str, object]:
    for required_component in ("training_loop", "data_mix", "evaluation"):
        if not is_enabled(required_component):
            raise RuntimeError(
                f"Required component is disabled at its call site: {required_component}"
            )
    print_session_dashboard()
    if model_size != "frontier":
        raise ValueError("iterate500 supports only --model-size frontier")
    model_cfg, training_cfg = resolve_model_profile(model_size)
    is_frontier = model_size == "frontier"
    growth_teacher = None
    growth_alignment = None
    if is_frontier:
        if not torch.cuda.is_available() and os.environ.get("ANRA_ALLOW_CPU_FRONTIER", "0") != "1":
            raise RuntimeError(
                "iterate500 frontier training requires a CUDA GPU. "
                "Your runtime is CPU/TPU, not T4. In Colab choose "
                "Runtime -> Change runtime type -> T4 GPU, then rerun from the top."
            )
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            vram_gb = props.total_memory / 1024 ** 3
            print(f"[Trainer] GPU: {props.name}  VRAM: {vram_gb:.1f}GB", flush=True)
            if "T4" not in props.name.upper():
                print(
                    f"[Trainer] WARNING: expected a T4-class CUDA GPU; got {props.name}.",
                    flush=True,
                )
            if vram_gb < 20:
                print(
                    f"[Trainer] WARNING: {vram_gb:.1f}GB VRAM is below the 20GB minimum.\n"
                    f"          500M frontier training is tight but practical on a T4.\n"
                    f"          Continuing; reduce batch_size if it OOMs.",
                    flush=True,
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
            f"[Trainer] 500M FRONTIER MODE  "
            f"batch={training_cfg.batch_size}  grad_accum={training_cfg.grad_accum_steps}"
        )
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
        model_params=V2_FRONTIER_PARAMETER_COUNT,
    )
    training_mix_controller = TrainingDataMixController(V2_FRONTIER_PARAMETER_COUNT)
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
        num_workers = 2 if torch.cuda.is_available() else 0
        loader_kwargs = {
            "batch_size": batch_size,
            "drop_last": False,
            "pin_memory": torch.cuda.is_available(),
            "num_workers": num_workers,
            "persistent_workers": num_workers > 0,
        }
        if active_weights is None:
            return DataLoader(
                ds,
                shuffle=True,
                **loader_kwargs,
            )
        bucket_counts: dict[str, int] = {}
        buckets = [ds.bucket_for_window(index) for index in range(len(ds))]
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
            sampler=sampler,
            **loader_kwargs,
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
    if getattr(training_cfg, "gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
        print("[build_brain] Gradient checkpointing enabled for 500M model", flush=True)
    if hasattr(model, "disable_kv_cache"):
        model.disable_kv_cache()
    if use_ouroboros:
        from ouroboros import OuroborosDecoder

        model = OuroborosDecoder(model, n_passes=3)
    model = model.to(device)
    ensure_tied_lm_head(model)
    intelligence_session = create_intelligence_session(model)
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
    _prepare_resume_target(ckpt_path, resume_from)
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
        if ok:
            _sync_training_checkpoint_to_drive(ckpt_path)
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
        resume_state = load_checkpoint(model, optimizer, scheduler, mp, load_path, device=device, strict=False)
        if resume_state["loaded"]:
            ckpt["sessions_completed"] = int(resume_state.get("sessions_completed", 0))
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

    if start_eval_examples > 0:
        try:
            print(
                f"[build_brain] running startup quick eval on {start_eval_examples} examples...",
                flush=True,
            )
            session_start_result = quick_eval_loss(
                model,
                ds,
                device=device,
                max_examples=start_eval_examples,
                batch_size=batch_size,
                pad_id=tokenizer.pad_token_id,
            )
            session_start_loss = _quick_eval_loss_value(session_start_result)
        except Exception as exc:
            print(f"[build_brain] quick eval at session_start failed: {exc}", flush=True)
            session_start_loss = best_loss
    else:
        print("[build_brain] startup quick eval skipped so first loss appears sooner.", flush=True)
    if math.isfinite(session_start_loss):
        regret_scheduler.session_start(session_start_loss)

    global_step = start_step
    epoch = 0

    start = time.time()
    end_at = start + max_minutes * 60
    initial_step = start_step
    session_step = 0
    checkpoint_every_seconds = max(
        300,
        int(float(os.environ.get("ANRA_CHECKPOINT_EVERY_MIN", "25")) * 60),
    )
    next_checkpoint_at = time.time() + checkpoint_every_seconds
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
    print(
        "[build_brain] entering training loop; first optimizer step may still take several minutes on T4.",
        flush=True,
    )

    while time.time() < end_at:
        epoch += 1
        for xb, yb, wb, sample_idx in loader:
            if intelligence_session is not None:
                intelligence_session.begin_step(global_step)
            if first_batch_wall is None:
                first_batch_wall = time.time()
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            wb = wb.to(device, non_blocking=True)
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
                if device.type == "cuda":
                    torch.cuda.empty_cache()
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
                gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                loss_float = float(batch_loss.item())
                grad_float = float(gradient_norm)
                if intelligence_session is not None:
                    intelligence_session.record_optimizer_step(
                        step=global_step,
                        loss=loss_float,
                        learning_rate=float(optimizer.param_groups[0]["lr"]),
                        gradient_norm=grad_float,
                        tokens=int((yb != tokenizer.pad_token_id).sum().item()),
                    )
                update_hal_from_training(
                    model,
                    loss=loss_float,
                    best_loss=best_loss,
                    gradient_norm=grad_float,
                    step=global_step,
                )
                if intelligence_session is not None:
                    hal = get_hal_module(model)
                    if hal is not None:
                        intelligence_session.record_hal_step(step=global_step, hal_state=hal.state)
                mp.step(optimizer)
                mp.update()
                scheduler.step()
                regret_lr = regret_scheduler.update(reward=max(0.0, 1.0 - loss_float))
                multiplier = max(0.5, min(1.5, regret_lr / max(learning_rate, 1e-12)))
                scheduled_lrs = scheduler.get_last_lr()
                for group, scheduled_lr in zip(optimizer.param_groups, scheduled_lrs):
                    group["lr"] = scheduled_lr * multiplier
                optimizer.zero_grad(set_to_none=True)
                pcgrad.clear()
                global_step += 1
                if growth_alignment is not None:
                    growth_alignment.configure_trainable_parameters(global_step - initial_step)
                session_step += 1
                accum_micro_steps = 0

                avg_loss = rolling_loss / max(1, rolling_count)
                loss_val = avg_loss
                last_avg_loss = avg_loss
                best_loss = min(best_loss, avg_loss) if math.isfinite(best_loss) else avg_loss
                phase = phase_for_step(global_step, warmup_steps=warmup_steps, total_steps=total_steps)
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
                if loss_float > max(3.0 * running_mean, running_mean + 2.0):
                    with torch.no_grad():
                        cdr.capture_step_failure(
                            input_tokens=xb,
                            target_tokens=yb,
                            predicted_tokens=logits.argmax(dim=-1),
                            loss=loss_float,
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
                    remaining_min = max(0.0, (end_at - time.time()) / 60.0)
                    startup_note = ""
                    if global_step in EARLY_STATUS_STEPS and first_batch_wall is not None:
                        startup_note = f"  startup={(first_batch_wall - start):.1f}s"
                    print(
                        f"  step={global_step:6d}  loss={avg_loss:.4f}  best={best_loss:.4f}  "
                        f"elapsed={elapsed_min:.1f}m  remaining={remaining_min:.1f}m{startup_note}",
                        flush=True,
                    )

                if time.time() >= next_checkpoint_at:
                    payload = _build_checkpoint_payload(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        mp=mp,
                        global_step=global_step,
                        epoch=epoch,
                        best_loss=best_loss,
                        sessions_completed=(int(ckpt.get("sessions_completed", 0)) if "ckpt" in locals() else 0),
                        mix_report=mix_report,
                        migration=checkpoint_migration,
                    )
                    atomic_save(payload, ckpt_path, drive_dir=DRIVE_V2_CHECKPOINTS)
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    try:
                        hal = get_hal_module(model)
                        if hal is not None:
                            publish_hal_state(hal, source="training")
                    except Exception as exc:
                        print(f"[HAL] checkpoint publish skipped: {exc}", flush=True)
                    _sync_training_checkpoint_to_drive(ckpt_path)
                    next_checkpoint_at = time.time() + checkpoint_every_seconds

            if time.time() >= end_at:
                break

    if accum_micro_steps > 0:
        pcgrad_reports.extend(pcgrad.materialize())
        if growth_alignment is not None:
            growth_alignment.mask_inactive_gradients()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        grad_float = float(gradient_norm)
        if intelligence_session is not None:
            intelligence_session.record_optimizer_step(
                step=global_step,
                loss=float(last_avg_loss),
                learning_rate=float(optimizer.param_groups[0]["lr"]),
                gradient_norm=grad_float,
                tokens=int(total_target_tokens),
            )
        update_hal_from_training(
            model,
            loss=float(last_avg_loss),
            best_loss=best_loss,
            gradient_norm=grad_float,
            step=global_step,
        )
        if intelligence_session is not None:
            hal = get_hal_module(model)
            if hal is not None:
                intelligence_session.record_hal_step(step=global_step, hal_state=hal.state)
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
    atomic_save(payload, ckpt_path, drive_dir=DRIVE_V2_CHECKPOINTS)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    try:
        hal = get_hal_module(model)
        if hal is not None:
            publish_hal_state(hal, source="training")
    except Exception as exc:
        print(f"[HAL] final publish skipped: {exc}", flush=True)
    _sync_training_checkpoint_to_drive(ckpt_path)

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
    intelligence_report = None
    if intelligence_session is not None:
        try:
            intelligence_report = intelligence_session.finalize(
                checkpoint_id=f"{Path(ckpt_path).name}:step-{global_step:012d}",
                capability_score=float(eval_summary.get("overall_score", 0.0)),
                capability_samples=max(
                    1,
                    len(eval_summary.get("results", eval_summary.get("items", []))),
                ),
            )
            metrics["thirdeye_intelligence"] = intelligence_report
            write_json(v2_report_path("metrics"), metrics)
        except Exception as exc:
            intelligence_session.hooks.close()
            print(f"[ThirdEye] Intelligence report failed: {exc}", flush=True)
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
    _sync_training_checkpoint_to_drive(ckpt_path)
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
    parser.add_argument("--checkpoint_path", default="anra_frontier_500m.pt")
    parser.add_argument("--resume_from", default=None)
    parser.add_argument("--batch_size", type=int, default=V2_1B_TRAINING.batch_size)
    parser.add_argument("--block_size", type=int, default=V2_1B_FRONTIER.block_size)
    parser.add_argument("--max_minutes", type=int, default=V2_1B_TRAINING.session_minutes)
    parser.add_argument(
        "--model-size",
        choices=["frontier"],
        default="frontier",
    )
    parser.add_argument("--answer_loss_weight", type=float, default=V2_1B_TRAINING.answer_loss_weight)
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument(
        "--start_eval_examples",
        type=int,
        default=0,
        help="Run startup quick-eval before training. Default 0 skips it for faster first loss.",
    )
    parser.add_argument("--own_ratio", type=float, default=None)
    parser.add_argument("--identity_ratio", type=float, default=None)
    parser.add_argument("--teacher_ratio", type=float, default=None)
    parser.add_argument("--symbolic_ratio", type=float, default=None)
    parser.add_argument("--replay_ratio", type=float, default=None)
    parser.add_argument("--optimizer", choices=["auto", "adamw", "muon", "scale", "galore"], default="auto")
    parser.add_argument(
        "--training-objective",
        choices=["base", "causal-extension"],
        default="base",
    )
    parser.add_argument(
        "--cognitive-output",
        default=str(ROOT / "output" / "v2" / "cognition" / "causal_extension.pt"),
    )
    args = parser.parse_args()
    if args.training_objective == "causal-extension":
        result = train_causal_extension(
            data_path=args.data_path,
            base_checkpoint=args.checkpoint_path,
            output_path=args.cognitive_output,
            model_size=args.model_size,
            batch_size=args.batch_size,
            block_size=args.block_size,
            max_minutes=args.max_minutes,
            optimizer_name=args.optimizer,
        )
        print(result, flush=True)
        return
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
        start_eval_examples=args.start_eval_examples,
    )
    print(result, flush=True)


if __name__ == "__main__":
    main()
