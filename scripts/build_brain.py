# NOTE: scripts/train.py is the canonical training script for local runs.
# This file handles tokenizer building and frontier data preparation.
from __future__ import annotations

# Direct execution bootstraps repository imports after resolving REPO_ROOT.
# ruff: noqa: E402
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
import torch.nn.functional as F  # noqa: N812 - canonical PyTorch alias
from anra.anra_paths import (
    CDR_REPORT,
    FAILURE_REPLAY_DATASET,
    IBS_LATEST,
    OUTPUT_V2_DIR,
    REGRET_STATE,
    ROOT,
    SOVEREIGNTY_EVENTS,
    TOKENIZER_MANIFEST,
    V2_TOKENIZER_FILE,
)
from engine.eval_harness import EvalHarness, EvalResult
from engine.feature_flags import is_enabled
from evaluation.intelligence_telemetry import create_intelligence_session
from runtime.hal_telemetry import publish_hal_state
from runtime.safe_load import safe_torch_load
from torch.utils.data import DataLoader, WeightedRandomSampler
from training.anra_optimizer import (
    IDENTITY_PARAMETER_PATTERNS,
    build_append_only_row_learning_rate,
    build_optimizer,
    is_identity_parameter,
)
from training.cdr import CorrectedFailureCurriculum
from training.continual import assess_continual_readiness, ewc_penalty
from training.data_routing import build_data_route_report
from training.dynamic_regret import DynamicRegretScheduler
from training.eval_v2 import quick_eval_loss, run_compact_eval
from training.mixed_precision import MixedPrecisionTrainer
from training.pcgrad import PCGradAccumulator
from training.scheduler import get_cosine_schedule_with_warmup
from training.shared_checkpoint import (
    record_filesystem_checkpoint_origin,
    restore_shared_checkpoint,
    sync_checkpoint_to_origin,
)
from training.v2_config import (
    CHECKPOINT_SCHEMA_VERSION,
    EXPECTED_SPECIAL_TOKEN_IDS,
    EXPECTED_TOKENIZER_VOCAB_SIZE,
    TOKENIZER_SCHEMA_VERSION,
    V2_FRONTIER,
    V2_FRONTIER_TRAINING,
    V2_MODEL,
    V2_TRAINING,
    frontier_parameter_count,
    resolve_model_profile,
)
from training.v2_data_mix import (
    MixReport,
    RawCausalShardDataset,
    TrainingDataMixController,
    V2ConversationDataset,
    WindowConsumptionTracker,
    build_v2_training_examples,
)
from training.v2_runtime import (
    active_tokenizer_path,
    atomic_save,
    build_frontier_model,
    canonical_v2_checkpoint,
    ensure_tied_lm_head,
    get_hal_module,
    hal_state_dict,
    load_checkpoint,
    load_or_build_v2_tokenizer,
    model_summary,
    v2_report_path,
    write_json,
)

from scripts.session_dashboard import print_session_dashboard

EARLY_STATUS_STEPS = {1, 2, 5, 10, 20, 50, 100}
HARD_EXAMPLE_KEEP = 16
CONTINUATION_PHASE_TOKEN_TARGETS = {
    "A": 1_000_000_000,
    "B": 1_000_000_000,
    "C": 200_000_000,
    "D": 100_000_000,
    "E": 10_000_000,
}


EMERGENCY_SAVE_TIMEOUT_SECONDS = 20.0
_SAVE_COMPONENT_ORDER = ("model", "optimizer", "scheduler", "scaler")


def build_causal_extension_trainer(
    model: torch.nn.Module,
    *,
    total_steps: int,
    warmup_steps: int,
    optimizer_name: str = "adafactor",
) -> object:
    """Canonical build-brain integration point for extension-only causal training."""
    from cognition.cre import CognitiveCausalExtension
    from training.causal_trainer import CausalExtensionTrainer

    layer_count = int(model.n_layer)
    integration_layers = tuple(sorted({0, max(0, layer_count // 2), max(0, layer_count - 1)}))
    extension = CognitiveCausalExtension(
        int(model.n_embd),
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


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _freeze_training_lineage(
    *,
    checkpoint_path: Path,
    tokenizer_path: Path,
    model_config: dict[str, object],
    data_manifests: list[Path],
) -> dict[str, object]:
    """Preserve the exact pre-training artifacts before any optimizer step."""
    source_checkpoint = checkpoint_path.resolve()
    source_tokenizer = tokenizer_path.resolve()
    if not source_tokenizer.is_file():
        raise FileNotFoundError(f"Cannot freeze missing tokenizer: {source_tokenizer}")
    checkpoint_hash = (
        _sha256_path(source_checkpoint) if source_checkpoint.is_file() else "uninitialized"
    )
    tokenizer_hash = _sha256_path(source_tokenizer)
    archive_root = source_checkpoint.parent / ".anra_lineage"
    archive_root.mkdir(parents=True, exist_ok=True)

    def preserve(source: Path, target: Path) -> None:
        if target.exists():
            if _sha256_path(target) != _sha256_path(source):
                raise RuntimeError(f"Lineage archive hash collision: {target}")
            return
        shutil.copy2(source, target)

    checkpoint_archive: Path | None = None
    if source_checkpoint.is_file():
        checkpoint_archive = archive_root / f"checkpoint-{checkpoint_hash}.pt"
        preserve(source_checkpoint, checkpoint_archive)
    tokenizer_archive = archive_root / f"tokenizer-{tokenizer_hash}.json"
    preserve(source_tokenizer, tokenizer_archive)
    tokenizer_meta = source_tokenizer.with_suffix(source_tokenizer.suffix + ".meta.json")
    tokenizer_meta_archive = None
    if tokenizer_meta.is_file():
        tokenizer_meta_hash = _sha256_path(tokenizer_meta)
        tokenizer_meta_archive = archive_root / f"tokenizer-meta-{tokenizer_meta_hash}.json"
        preserve(tokenizer_meta, tokenizer_meta_archive)

    manifest_hashes = {
        str(path.resolve()): _sha256_path(path.resolve())
        for path in data_manifests
        if path.resolve().is_file()
    }
    try:
        source_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        source_commit = "unknown"
    payload: dict[str, object] = {
        "schema_version": 1,
        "created_at": time.time(),
        "checkpoint_source": str(source_checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_archive": str(checkpoint_archive) if checkpoint_archive else None,
        "tokenizer_source": str(source_tokenizer),
        "tokenizer_sha256": tokenizer_hash,
        "tokenizer_archive": str(tokenizer_archive),
        "tokenizer_meta_archive": (str(tokenizer_meta_archive) if tokenizer_meta_archive else None),
        "model_config": dict(model_config),
        "data_manifest_sha256": manifest_hashes,
        "source_commit": source_commit,
    }
    identity = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    immutable = archive_root / f"lineage-{identity}.json"
    if not immutable.exists():
        temporary = immutable.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(immutable)
    current = OUTPUT_V2_DIR / "lineage_freeze.json"
    current.parent.mkdir(parents=True, exist_ok=True)
    current_tmp = current.with_suffix(".tmp")
    current_tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    current_tmp.replace(current)
    return payload


def _session_data_mix_seed(base_seed: int = 1337) -> int:
    """Rotate the deterministic training sample after each completed session."""
    state = _read_json(REGRET_STATE) or {}
    try:
        completed_sessions = max(0, int(state.get("session_count", 0)))
    except (TypeError, ValueError):
        completed_sessions = 0
    return int(base_seed) + completed_sessions


def _tokenizer_checkpoint_contract() -> dict[str, object]:
    tokenizer_path = active_tokenizer_path()
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Canonical tokenizer is missing: {tokenizer_path}")
    raw = tokenizer_path.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    vocabulary = payload.get("token_to_id", {}) if isinstance(payload, dict) else {}
    vocabulary_bytes = json.dumps(
        vocabulary,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    manifest = _read_json(TOKENIZER_MANIFEST) or {}
    return {
        "schema_version": int(manifest.get("schema_version", TOKENIZER_SCHEMA_VERSION)),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "vocabulary_sha256": hashlib.sha256(vocabulary_bytes).hexdigest(),
        "vocab_size": int(manifest.get("vocab_size", EXPECTED_TOKENIZER_VOCAB_SIZE)),
        "special_token_ids": EXPECTED_SPECIAL_TOKEN_IDS,
        "probe_count": int(manifest.get("probe_count", 0)),
        "probe_sha256": str(manifest.get("probe_sha256", "")),
    }


def _collect_data_manifest_payloads(
    manifest_root: Path,
) -> tuple[dict[str, str], dict[str, bytes]]:
    hashes: dict[str, str] = {}
    payloads: dict[str, bytes] = {}
    if manifest_root.exists():
        for path in sorted(manifest_root.rglob("*.json")):
            relative = path.relative_to(manifest_root).as_posix()
            payload = path.read_bytes()
            hashes[relative] = hashlib.sha256(payload).hexdigest()
            payloads[relative] = payload
    return hashes, payloads


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
    tokens_seen: int = 0,
    unique_token_ids_seen: set[int] | None = None,
    continuation_token_counts: dict[str, int] | None = None,
    best_validation_loss: float = float("inf"),
    validation_history: list[dict[str, object]] | None = None,
    appended_row_optimizer_steps: int = 0,
    raw_window_consumption: dict[str, object] | None = None,
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
    manifest_root = ROOT / "output" / "v2" / "data_manifests"
    data_manifests, data_manifest_payloads = _collect_data_manifest_payloads(manifest_root)
    tokenizer_contract = _tokenizer_checkpoint_contract()
    return {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "tokenizer_schema_version": int(tokenizer_contract["schema_version"]),
        "tokenizer_contract": tokenizer_contract,
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
        "tokens_seen": int(tokens_seen),
        "completed_optimizer_boundary": True,
        "accum_micro_steps": 0,
        "unique_token_ids_seen": sorted(unique_token_ids_seen or set()),
        "unique_tokens_seen": len(unique_token_ids_seen or set()),
        "continuation_token_counts": dict(continuation_token_counts or {}),
        "best_validation_loss": float(best_validation_loss),
        "validation_history": list(validation_history or []),
        "appended_row_optimizer_steps": int(appended_row_optimizer_steps),
        "raw_window_consumption": dict(raw_window_consumption or {}),
        "model_config": model.model_config(),
        "hal_state": hal_state_dict(model),
        "mix_report": mix_report.to_dict(),
        "rng_states": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        },
        "source_commit": source_commit,
        # A resumed optimizer state is only comparable when it sees the same
        # prepared corpus profile. Keep this explicit rather than inferring it
        # from whichever files happen to be present in a fresh Colab runtime.
        "data_profile": os.environ.get("ANRA_DATA_PROFILE", "unknown"),
        "training_data_layout": _active_training_data_layout(),
        "data_manifests": data_manifests,
        "dataset_manifest_hashes": data_manifests,
        "data_manifest_payloads": data_manifest_payloads,
        "cognitive_extension_release": "cognition-v1",
        "consent_safe_metadata": {
            "owner_derived_data_authorized": bool(
                os.environ.get("ANRA_OWNER_DATA_AUTHORIZED", "").lower() in {"1", "true", "yes"}
            )
        },
    }


def _active_training_data_layout() -> str:
    """Return the explicit dataset layout recorded in every checkpoint."""
    configured = os.environ.get("ANRA_TRAINING_DATA_LAYOUT", "").strip()
    allowed = {
        V2ConversationDataset.PACKING_LAYOUT,
        RawCausalShardDataset.PACKING_LAYOUT,
    }
    if configured and configured not in allowed:
        raise RuntimeError(
            "This trainer only supports "
            f"{sorted(allowed)}; got ANRA_TRAINING_DATA_LAYOUT={configured}."
        )
    return configured or V2ConversationDataset.PACKING_LAYOUT


def _assert_resume_data_profile_compatible(
    checkpoint_profile: object,
    active_profile: str,
) -> None:
    """Prevent a checkpoint from silently continuing on another corpus profile."""
    saved = str(checkpoint_profile or "unknown").strip()
    active = active_profile.strip() or "unknown"
    if saved in {"unknown", "unlabelled", "unlabeled"}:
        print(
            "[Resume] Checkpoint predates data-profile tracking; continuity cannot be verified. "
            f"Training with active profile={active}.",
            flush=True,
        )
        return
    if active in {"unknown", "unlabelled", "unlabeled"}:
        print(
            f"[Resume] Active data profile is not declared; checkpoint profile={saved}.",
            flush=True,
        )
        return
    if saved == active:
        print(f"[Resume] Data profile verified: {active}", flush=True)
        return
    if os.environ.get("ANRA_ALLOW_DATA_PROFILE_CHANGE", "0") == "1":
        print(
            f"[Resume] WARNING: data-profile change explicitly allowed: {saved} -> {active}.",
            flush=True,
        )
        return
    raise RuntimeError(
        "Refusing to resume with a different data profile: "
        f"checkpoint={saved}, active={active}. Restore the original prepared corpus, "
        "or set ANRA_ALLOW_DATA_PROFILE_CHANGE=1 only for an intentional new experiment."
    )


def _assert_resume_data_layout_compatible(
    checkpoint_layout: object,
    active_layout: str,
    continuation_phase: str = "D",
) -> None:
    """Allow only the explicit raw/chat layout transitions in the curriculum."""
    saved = str(checkpoint_layout or "unknown").strip()
    active = active_layout.strip() or "unknown"
    if saved in {"unknown", "unlabelled", "unlabeled"}:
        print(
            "[Resume] Checkpoint predates data-layout tracking; the configured "
            "layout transition is explicit: "
            f"{active}.",
            flush=True,
        )
        return
    if saved == active:
        print(f"[Resume] Data layout verified: {active}", flush=True)
        return
    phase = continuation_phase.upper()
    planned_transition = (
        saved == V2ConversationDataset.PACKING_LAYOUT
        and active == RawCausalShardDataset.PACKING_LAYOUT
        and phase in {"A", "B", "C"}
    ) or (
        saved == RawCausalShardDataset.PACKING_LAYOUT
        and active == V2ConversationDataset.PACKING_LAYOUT
        and phase in {"D", "E"}
    )
    if planned_transition:
        print(
            f"[Resume] Planned curriculum layout transition: {saved} -> {active} "
            f"for phase {phase}.",
            flush=True,
        )
        return
    raise RuntimeError(
        "Refusing to resume with a different training data layout: "
        f"checkpoint={saved}, active={active}. Start a separate experiment for this change."
    )


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
            "[build_brain] emergency save timeout after "
            f"{EMERGENCY_SAVE_TIMEOUT_SECONDS:.1f}s; process exit continues",
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
    if resume_from:
        candidate = _resolve_checkpoint_path(resume_from)
        if candidate.exists():
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(candidate, checkpoint_path)
            record_filesystem_checkpoint_origin(checkpoint_path.name, candidate)
            print(
                f"[build_brain] restored checkpoint: {candidate} -> {checkpoint_path}", flush=True
            )
            return
    shared_name = Path(resume_from).name if resume_from else checkpoint_path.name
    restored = restore_shared_checkpoint(checkpoint_path, shared_name)
    if restored is not None:
        print(
            f"[build_brain] restored shared checkpoint: {restored} -> {checkpoint_path}", flush=True
        )


def _sync_training_checkpoint_to_drive(checkpoint_path: Path) -> None:
    if not checkpoint_path.exists():
        return
    try:
        sync_checkpoint_to_origin(checkpoint_path)
    except Exception as exc:
        print(f"[Drive] checkpoint publish failed: {exc}", flush=True)
        raise


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
    sample_losses = (per_token * effective_weights).sum(dim=1) / effective_weights.sum(
        dim=1
    ).clamp_min(1.0)
    return sample_losses.mean(), sample_losses


def _quick_eval_loss_value(result: float | dict[str, object]) -> float:
    return float(result["loss"]) if isinstance(result, dict) else float(result)


def _compact_eval_to_result(
    summary: dict[str, object], *, component: str = "training"
) -> EvalResult:
    score = float(summary.get("overall_score", 0.0) or 0.0)
    return EvalResult(
        component=component,
        mode=str(summary.get("mode", "compact_eval")),
        task_success_rate=score,
        avg_latency_ms=0.0,
        error_rate=0.0,
        notes="compact eval overall_score mapped to task_success_rate",
        raw=list(summary.get("results", []))
        if isinstance(summary.get("results", []), list)
        else [],
    )


def _configure_continuation_phase(
    model: torch.nn.Module,
    phase: str,
) -> dict[str, object]:
    native_model = getattr(model, "model", model)
    phase_name = phase.upper()
    subsystem_patterns = (
        "mod_routers.",
        "rim_modules.",
        "esv_module.",
        "residual_depth_logits",
        "dstp_temperature_log",
        "hal_module.",
    )
    phase_b_target = os.environ.get("ANRA_PHASE_B_SUBSYSTEM", "mod").strip().lower()
    target_patterns = {
        "mod": ("mod_routers.",),
        "rim": ("rim_modules.",),
        "dstp": ("residual_depth_logits", "dstp_temperature_log"),
        "esv": ("esv_module.",),
    }
    if phase_b_target not in target_patterns:
        raise ValueError("ANRA_PHASE_B_SUBSYSTEM must be mod, rim, dstp, or esv")
    frozen: list[str] = []
    active: list[str] = []
    for name, parameter in native_model.named_parameters():
        if not name.startswith(subsystem_patterns):
            continue
        trainable = phase_name not in {"A", "B"}
        if phase_name == "B":
            trainable = name.startswith(target_patterns[phase_b_target])
        parameter.requires_grad_(trainable)
        (active if trainable else frozen).append(name)
    capacity = 1.0 if phase_name in {"A", "B"} else 0.75
    parity = os.environ.get("ANRA_SUBSYSTEM_VALIDATION_PARITY", "0").lower()
    if phase_name in {"D", "E"} and parity in {"1", "true", "yes"}:
        capacity = 0.5
    if hasattr(native_model, "set_mod_capacity"):
        native_model.set_mod_capacity(capacity)
    report = {
        "phase": phase_name,
        "phase_b_subsystem": phase_b_target if phase_name == "B" else None,
        "mod_capacity": capacity,
        "active_subsystem_parameters": active,
        "frozen_subsystem_parameters": frozen,
    }
    print(
        f"[Continuation] phase={phase_name} mod_capacity={capacity:.2f} "
        f"active_native={len(active)} frozen_native={len(frozen)}",
        flush=True,
    )
    return report


def train_anra_v2(
    *,
    data_path: str,
    checkpoint_path: str = "anra_frontier_500m.pt",
    resume_from: str | None = None,
    batch_size: int = V2_FRONTIER_TRAINING.batch_size,
    block_size: int = V2_FRONTIER.block_size,
    max_minutes: int = V2_FRONTIER_TRAINING.session_minutes,
    answer_loss_weight: float = V2_FRONTIER_TRAINING.answer_loss_weight,
    max_examples: int | None = None,
    own_ratio: float | None = None,
    identity_ratio: float | None = None,
    teacher_ratio: float | None = None,
    symbolic_ratio: float | None = None,
    replay_ratio: float | None = None,
    use_ouroboros: bool = False,
    model_size: str = "frontier",
    optimizer_name: str = "adafactor",
    start_eval_examples: int = 0,
    training_layout: str = V2ConversationDataset.PACKING_LAYOUT,
    token_shard_manifest: str | None = None,
    validation_shard_manifest: str | None = None,
    continuation_phase: str = "D",
    max_phase_tokens: int | None = None,
) -> dict[str, object]:
    for required_component in ("training_loop", "data_mix", "evaluation"):
        if not is_enabled(required_component):
            raise RuntimeError(
                f"Required component is disabled at its call site: {required_component}"
            )
    print_session_dashboard()
    if model_size != "frontier":
        raise ValueError("iterate500 supports only --model-size frontier")
    if training_layout not in {
        V2ConversationDataset.PACKING_LAYOUT,
        RawCausalShardDataset.PACKING_LAYOUT,
    }:
        raise ValueError(f"unsupported training layout: {training_layout}")
    os.environ["ANRA_TRAINING_DATA_LAYOUT"] = training_layout
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
            vram_gb = props.total_memory / 1024**3
            print(f"[Trainer] GPU: {props.name}  VRAM: {vram_gb:.1f}GB", flush=True)
            if "T4" not in props.name.upper():
                print(
                    f"[Trainer] WARNING: expected a T4-class CUDA GPU; got {props.name}.",
                    flush=True,
                )
            if vram_gb < 14:
                print(
                    f"[Trainer] WARNING: {vram_gb:.1f}GB VRAM is below the 14GB practical floor.\n"
                    f"          500M frontier training may still OOM on this runtime.\n"
                    f"          Continuing; reduce batch_size if it OOMs.",
                    flush=True,
                )
        if batch_size == V2_TRAINING.batch_size:
            batch_size = V2_FRONTIER_TRAINING.batch_size
        if block_size == V2_MODEL.block_size:
            block_size = V2_FRONTIER.block_size
        if max_minutes == V2_TRAINING.session_minutes:
            max_minutes = V2_FRONTIER_TRAINING.session_minutes
        if max_examples is None:
            max_examples = (
                300_000
                if continuation_phase.upper() in {"D", "E"}
                else V2_FRONTIER_TRAINING.max_mixture_examples
            )
        own_ratio = own_ratio if own_ratio is not None else V2_FRONTIER_TRAINING.own_ratio
        identity_ratio = (
            identity_ratio if identity_ratio is not None else V2_FRONTIER_TRAINING.identity_ratio
        )
        teacher_ratio = (
            teacher_ratio if teacher_ratio is not None else V2_FRONTIER_TRAINING.teacher_ratio
        )
        symbolic_ratio = (
            symbolic_ratio if symbolic_ratio is not None else V2_FRONTIER_TRAINING.symbolic_ratio
        )
        replay_ratio = (
            replay_ratio if replay_ratio is not None else V2_FRONTIER_TRAINING.replay_ratio
        )
        print(
            f"[Trainer] 500M FRONTIER MODE  "
            f"batch={training_cfg.batch_size}  grad_accum={training_cfg.grad_accum_steps}"
        )
    dataset_path = Path(data_path)
    tokenizer = load_or_build_v2_tokenizer(dataset_path=dataset_path)
    model_parameter_contract = frontier_parameter_count(tokenizer.vocab_size)
    data_mix_seed = _session_data_mix_seed()
    training_mix_controller = TrainingDataMixController(model_parameter_contract)
    print(f"[Trainer] Data mix sampling seed: {data_mix_seed}", flush=True)
    if training_layout == RawCausalShardDataset.PACKING_LAYOUT:
        if not token_shard_manifest:
            raise ValueError("raw causal training requires --token-shard-manifest")
        manifest_path = Path(token_shard_manifest)
        if not manifest_path.is_absolute():
            manifest_path = ROOT / manifest_path
        ds = RawCausalShardDataset(
            manifest_path,
            tokenizer,
            block_size,
            rotation_seed=data_mix_seed,
            verify_hashes=True,
            expected_tokenizer_sha256=str(
                (_read_json(TOKENIZER_MANIFEST) or {}).get("tokenizer_sha256", "")
            ),
        )
        validation_manifest_path = (
            Path(validation_shard_manifest)
            if validation_shard_manifest
            else manifest_path.parent / "validation" / "manifest.json"
        )
        if not validation_manifest_path.is_absolute():
            validation_manifest_path = ROOT / validation_manifest_path
        if not validation_manifest_path.is_file():
            raise FileNotFoundError(
                f"Immutable validation manifest is missing: {validation_manifest_path}"
            )
        eval_ds = RawCausalShardDataset(
            validation_manifest_path,
            tokenizer,
            block_size,
            rotation_seed=0,
            verify_hashes=True,
            expected_tokenizer_sha256=str(
                (_read_json(TOKENIZER_MANIFEST) or {}).get("tokenizer_sha256", "")
            ),
        )
        mix_report = MixReport(
            total_examples=len(ds),
            requested_counts={"foundation": len(ds)},
            realized_counts={"foundation": len(ds)},
            source_counts={str(manifest_path): len(ds)},
            teacher_external_used=0,
            replay_available=0,
            active_weights={"foundation": 1.0},
            sampling_seed=data_mix_seed,
        )
    else:
        examples, mix_report = build_v2_training_examples(
            dataset_path=dataset_path,
            seed=data_mix_seed,
            max_examples=max_examples,
            own_ratio=own_ratio,
            identity_ratio=identity_ratio,
            teacher_ratio=teacher_ratio,
            symbolic_ratio=symbolic_ratio,
            replay_ratio=replay_ratio,
            model_params=model_parameter_contract,
            post_training_mix=continuation_phase.upper() in {"D", "E"},
        )
        if set(mix_report.active_weights) == set(training_mix_controller.weights):
            training_mix_controller.weights = dict(mix_report.active_weights)
        ds = V2ConversationDataset(
            examples,
            tokenizer,
            block_size,
            answer_loss_weight=answer_loss_weight,
        )
        eval_ds = ds
    if training_layout == RawCausalShardDataset.PACKING_LAYOUT:
        manifest_payload = _read_json(manifest_path) or {}
        source_mix = manifest_payload.get("source_mix", {})
        source_classes = list(source_mix) if isinstance(source_mix, dict) else []
        if not source_classes:
            source_classes = sorted(
                {
                    str(shard.get("source_class", shard.get("source", "")))
                    for shard in manifest_payload.get("shards", [])
                    if isinstance(shard, dict)
                    and shard.get("source_class", shard.get("source"))
                }
            )
    else:
        source_classes = sorted({example.bucket for example in examples})
    if not source_classes:
        raise RuntimeError("Training data does not declare any routable source classes")
    data_route_report = build_data_route_report(source_classes)
    data_route_report.update(
        {
            "training_layout": training_layout,
            "tokenizer_path": str(active_tokenizer_path()),
            "tokenizer_sha256": hashlib.sha256(active_tokenizer_path().read_bytes()).hexdigest(),
        }
    )
    write_json(OUTPUT_V2_DIR / "data_route_report.json", data_route_report)
    write_json(v2_report_path("mix_report"), mix_report.to_dict())
    if len(ds) == 0:
        raise RuntimeError(f"{training_layout} produced zero training windows.")
    window_consumption = (
        WindowConsumptionTracker(len(ds), block_size)
        if training_layout == RawCausalShardDataset.PACKING_LAYOUT
        else None
    )

    def make_loader(active_weights: dict[str, float] | None = None) -> DataLoader:
        num_workers = 2 if torch.cuda.is_available() else 0
        loader_kwargs = {
            "batch_size": batch_size,
            "drop_last": False,
            "pin_memory": torch.cuda.is_available(),
            "num_workers": num_workers,
            "persistent_workers": num_workers > 0,
        }
        if active_weights is None or training_layout == RawCausalShardDataset.PACKING_LAYOUT:
            return DataLoader(
                eval_ds,
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
    if device.type == "cuda":
        # T4 has fixed training shapes here. Let cuDNN select its fastest
        # convolution kernels without changing numerical semantics.
        torch.backends.cudnn.benchmark = True
    if is_frontier:
        hal_module = None
        if V2_FRONTIER.use_hal:
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
        if block_size > V2_FRONTIER.block_size:
            growth_evidence = _read_json(OUTPUT_V2_DIR / "context_growth_evidence.json") or {}
            if (
                float(growth_evidence.get("coherence_rate", 0.0)) < 0.90
                or float(growth_evidence.get("short_context_regression", 1.0)) >= 0.02
                or not bool(growth_evidence.get("retrieval_accuracy_improved", False))
            ):
                raise RuntimeError(
                    "Context growth is blocked until coherence >= 0.90, short-context "
                    "regression < 2%, and retrieval accuracy improves."
                )
        model = build_frontier_model(
            hal_module=hal_module,
            block_size=block_size,
            vocab_size=tokenizer.vocab_size,
        )
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
    continuation_report = _configure_continuation_phase(model, continuation_phase)
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
    optimizer_report = getattr(
        optimizer, "_anra_optimizer_report", {"selected": {"actual": optimizer_name}}
    )
    appended_row_lr = build_append_only_row_learning_rate(
        model,
        base_rows=EXPECTED_TOKENIZER_VOCAB_SIZE,
        multiplier=3.0,
        max_steps=2_000,
    )
    if appended_row_lr is not None:
        optimizer_report["append_only_rows"] = appended_row_lr.report()
    write_json(v2_report_path("optimizer_bakeoff"), optimizer_report)
    total_steps = int(getattr(training_cfg, "max_steps", 50_000))
    warmup_steps = max(1, int(total_steps * 0.02))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        min_lr_ratio=float(getattr(training_cfg, "min_lr", learning_rate * 0.1)) / learning_rate,
    )
    regret_scheduler = DynamicRegretScheduler(None, eta_base=learning_rate)
    regret_scheduler.load(REGRET_STATE)
    cdr = CorrectedFailureCurriculum(FAILURE_REPLAY_DATASET)
    pcgrad_enabled = (
        training_layout == V2ConversationDataset.PACKING_LAYOUT
        and continuation_phase.upper() in {"D", "E"}
    )
    protected_parameters = (
        [
            parameter
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and is_identity_parameter(name, parameter)
        ]
        if pcgrad_enabled
        else []
    )
    pcgrad = PCGradAccumulator(protected_parameters)
    pcgrad_reports = []
    annealing_started = False

    requested_checkpoint = Path(checkpoint_path)
    ckpt_path = (
        requested_checkpoint if requested_checkpoint.is_absolute() else ROOT / requested_checkpoint
    )
    _prepare_resume_target(ckpt_path, resume_from)
    if os.environ.get("ANRA_REQUIRE_RESUME", "0") == "1" and not ckpt_path.exists():
        raise RuntimeError(
            "ANRA_REQUIRE_RESUME=1, but no checkpoint was restored. "
            "Refusing to start from scratch and overwrite the intended experiment."
        )
    resume_path = Path(resume_from) if resume_from else ckpt_path
    if not resume_path.is_absolute():
        resume_path = ROOT / resume_path
    lineage_manifest_paths: list[Path] = []
    for raw_manifest in (token_shard_manifest, validation_shard_manifest):
        if not raw_manifest:
            continue
        manifest_path = Path(raw_manifest)
        if not manifest_path.is_absolute():
            manifest_path = ROOT / manifest_path
        lineage_manifest_paths.append(manifest_path)
    _freeze_training_lineage(
        checkpoint_path=ckpt_path if ckpt_path.exists() else resume_path,
        tokenizer_path=active_tokenizer_path(),
        model_config=model.model_config(),
        data_manifests=lineage_manifest_paths,
    )
    ckpt: dict[str, object] = {}
    global_step = 0
    epoch = 0
    best_loss = float("inf")
    checkpoint_migration: dict[str, object] | None = None
    campaign_tokens_seen = 0
    known_token_ids: set[int] = set()
    continuation_token_counts: dict[str, int] = {}
    best_validation_loss = float("inf")
    validation_history: list[dict[str, object]] = []

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
            tokens_seen=campaign_tokens_seen,
            unique_token_ids_seen=known_token_ids,
            continuation_token_counts=continuation_token_counts,
            best_validation_loss=best_validation_loss,
            validation_history=validation_history,
            appended_row_optimizer_steps=(
                appended_row_lr.steps_completed if appended_row_lr is not None else 0
            ),
            raw_window_consumption=(
                window_consumption.state_dict() if window_consumption is not None else None
            ),
        )
        ok = _emergency_save_with_timeout(payload, ckpt_path)
        if ok:
            _sync_training_checkpoint_to_drive(ckpt_path)
        signal_state["emergency_save_completed"] = ok
        print(f"[build_brain] SIGTERM emergency save status={ok}", flush=True)
        raise SystemExit(128 + sig_num)

    signal.signal(signal.SIGTERM, _handle_sigterm)
    print(
        "[build_brain] SIGTERM handler registered at "
        f"{signal_state['registered_at_iso']} (pre-training).",
        flush=True,
    )

    start_step = 0
    best_loss = float("inf")
    session_start_loss = float("inf")

    # ── AUTO-RESUME ──────────────────────────────────────────────────────────────
    load_path = ckpt_path if ckpt_path.exists() else resume_path
    if load_path.exists():
        print(f"[Resume] Found checkpoint: {load_path}", flush=True)
        resume_state = load_checkpoint(
            model, optimizer, scheduler, mp, load_path, device=device, strict=False
        )
        if resume_state["loaded"]:
            load_report = resume_state.get("load_report", {})
            if (
                not isinstance(load_report, dict)
                or not load_report.get("exact_core_load", False)
                or not load_report.get("exact_native_load", False)
                or not load_report.get("all_target_tensors_accounted", False)
            ):
                raise RuntimeError(
                    "Checkpoint failed exact core-tensor accounting; refusing to continue "
                    f"training: {load_report}"
                )
            _assert_resume_data_profile_compatible(
                resume_state.get("data_profile"),
                os.environ.get("ANRA_DATA_PROFILE", "unknown"),
            )
            _assert_resume_data_layout_compatible(
                resume_state.get("training_data_layout"),
                _active_training_data_layout(),
                continuation_phase,
            )
            ckpt["sessions_completed"] = int(resume_state.get("sessions_completed", 0))
            campaign_tokens_seen = int(resume_state.get("tokens_seen", 0))
            known_token_ids.update(
                int(value) for value in resume_state.get("unique_token_ids_seen", [])
            )
            continuation_token_counts.update(
                {
                    str(name): int(value)
                    for name, value in resume_state.get("continuation_token_counts", {}).items()
                }
            )
            best_validation_loss = float(resume_state.get("best_validation_loss", float("inf")))
            validation_history = list(resume_state.get("validation_history", []))
            checkpoint_migration = dict(resume_state.get("migration", {}))
            if appended_row_lr is not None:
                appended_row_lr.steps_completed = int(
                    resume_state.get("appended_row_optimizer_steps", 0)
                )
            if window_consumption is not None:
                raw_consumption_state = resume_state.get("raw_window_consumption", {})
                if isinstance(raw_consumption_state, dict) and raw_consumption_state:
                    window_consumption.load_state_dict(raw_consumption_state)
            start_step = int(resume_state["global_step"])
            best_loss = float(resume_state["best_loss"])
            session_start_loss = best_loss
            print(
                f"[Resume] Resuming from step={start_step}  best_loss={best_loss:.4f}", flush=True
            )
        else:
            print("[Resume] Checkpoint not loaded — starting from scratch", flush=True)
    else:
        print("[Resume] No checkpoint found — starting from scratch", flush=True)
    # ─────────────────────────────────────────────────────────────────────────────

    ewc_weight = max(0.0, float(os.environ.get("ANRA_EWC_WEIGHT", "0")))
    ewc_reference: dict[str, torch.Tensor] = {}
    ewc_fisher: dict[str, torch.Tensor] = {}
    ewc_state_value = os.environ.get("ANRA_EWC_STATE", "").strip()
    if ewc_weight > 0.0:
        if not ewc_state_value:
            raise RuntimeError("ANRA_EWC_WEIGHT requires ANRA_EWC_STATE")
        ewc_payload = safe_torch_load(Path(ewc_state_value), map_location="cpu")
        if not isinstance(ewc_payload, dict):
            raise TypeError("EWC state must be a dictionary")
        raw_reference = ewc_payload.get("reference", {})
        raw_fisher = ewc_payload.get("fisher", {})
        if not isinstance(raw_reference, dict) or not isinstance(raw_fisher, dict):
            raise TypeError("EWC state requires reference and fisher dictionaries")
        ewc_reference = {
            str(name): value.to(device=device)
            for name, value in raw_reference.items()
            if isinstance(value, torch.Tensor)
        }
        ewc_fisher = {
            str(name): value.to(device=device)
            for name, value in raw_fisher.items()
            if isinstance(value, torch.Tensor)
        }
        if not ewc_reference or not ewc_fisher:
            raise RuntimeError("EWC state has no usable tensors")
        print(
            f"[Trainer] EWC active weight={ewc_weight} tensors={len(ewc_fisher)}",
            flush=True,
        )

    if window_consumption is not None:
        phase_target = CONTINUATION_PHASE_TOKEN_TARGETS.get(continuation_phase.upper())
        consumption_report = window_consumption.report(phase_target_tokens=phase_target)
        print(
            "[Campaign] "
            f"unique_tokens={consumption_report['unique_tokens_consumed']:,} "
            f"repeated={consumption_report['repeated_token_percentage']:.3f}% "
            f"remaining_phase_tokens={consumption_report['remaining_phase_tokens']:,}",
            flush=True,
        )
        print(
            f"[Campaign] source_mix={getattr(ds, 'manifest', {}).get('campaign_mix_realized', {})}",
            flush=True,
        )
        print(
            f"[Campaign] best_validation_loss={best_validation_loss}",
            flush=True,
        )

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
    durable_checkpoint_steps = max(
        1,
        int(os.environ.get("ANRA_DURABLE_CHECKPOINT_STEPS", "100")),
    )
    next_checkpoint_at = time.time() + checkpoint_every_seconds
    optimizer.zero_grad(set_to_none=True)
    rolling_loss = 0.0
    rolling_count = 0
    accumulated_step_loss = 0.0
    accumulated_ewc_loss = 0.0
    accum_micro_steps = 0
    pending_trained_tokens = 0
    pending_token_ids: set[int] = set()
    pending_window_indices: list[int] = []
    last_avg_loss = best_loss if math.isfinite(best_loss) else 0.0
    loss_ema: float | None = None
    first_batch_wall = None
    hard_examples: list[tuple[float, int]] = []
    answer_weighted_tokens = 0.0
    total_target_tokens = 0.0

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    gpu_mem = (
        torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0.0
    )
    summary = model_summary(model)
    eff_batch = batch_size * training_cfg.grad_accum_steps
    pcgrad_fast_path = pcgrad_enabled and batch_size == 1

    print("", flush=True)
    print("=" * 62, flush=True)
    print("  AN-RA V2 TRAINING SESSION", flush=True)
    print("=" * 62, flush=True)
    print(f"  GPU          : {gpu_name} ({gpu_mem:.1f} GB)", flush=True)
    print(f"  Parameters   : {summary['parameters']:,}", flush=True)
    print(
        f"  Micro batch  : {batch_size}  |  Grad accum : "
        f"{training_cfg.grad_accum_steps}  |  Eff batch : {eff_batch}",
        flush=True,
    )
    print(f"  Session time : {max_minutes} minutes", flush=True)
    print(
        f"  Resuming     : step {global_step:,}  |  best loss "
        f"{best_loss if math.isfinite(best_loss) else float('inf'):.4f}",
        flush=True,
    )
    print(f"  Checkpoint   : {ckpt_path}", flush=True)
    print(f"  Data mix     : {mix_report.realized_counts}", flush=True)
    print(
        f"  Data layout  : {ds.PACKING_LAYOUT} | token utilization {ds.token_utilization:.1%}",
        flush=True,
    )
    pcgrad_status = (
        "disabled for foundation continuation"
        if not pcgrad_enabled
        else "normal-backward fast path"
        if pcgrad_fast_path
        else "mixed-batch gradient separation"
    )
    print(f"  PCGrad       : {pcgrad_status}", flush=True)
    print("=" * 62, flush=True)
    print("", flush=True)

    print(
        "[build_brain] entering training loop; SIGTERM and timed frontier checkpoints are active.",
        flush=True,
    )

    while time.time() < end_at and (
        max_phase_tokens is None
        or continuation_token_counts.get(continuation_phase.upper(), 0) < max_phase_tokens
    ):
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
                native_model = getattr(model, "model", model)
                if hasattr(ds, "verified_esv_targets") and hasattr(
                    native_model,
                    "_last_esv_prediction",
                ):
                    esv_targets, esv_mask = ds.verified_esv_targets(
                        sample_idx.tolist(),
                        device=logits.device,
                        dtype=native_model._last_esv_prediction.dtype,
                    )
                    if bool(esv_mask.any()):
                        esv_prediction = native_model._last_esv_prediction
                        batch_loss = batch_loss + 0.01 * torch.nn.functional.mse_loss(
                            esv_prediction[esv_mask],
                            esv_targets[esv_mask],
                        )
                if hasattr(native_model, "native_regularization_loss"):
                    batch_loss = batch_loss + native_model.native_regularization_loss()
                current_ewc_loss = (
                    ewc_penalty(native_model, ewc_reference, ewc_fisher, ewc_weight)
                    if ewc_weight > 0.0
                    else torch.zeros((), device=batch_loss.device, dtype=batch_loss.dtype)
                )
                batch_loss = batch_loss + current_ewc_loss
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
                accum_micro_steps = 0
                accumulated_step_loss = 0.0
                accumulated_ewc_loss = 0.0
                pending_trained_tokens = 0
                pending_token_ids.clear()
                pending_window_indices.clear()
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
                ds.bucket_for_sample(index) in {"own", "identity"} for index in sample_idx.tolist()
            ]
            owner_positions = [i for i, flag in enumerate(owner_flags) if flag]
            other_positions = [i for i, flag in enumerate(owner_flags) if not flag]
            # A single-example microbatch belongs to one data source. Its
            # normal backward pass is exactly the gradient PCGrad needs, so
            # avoid an additional graph traversal. Multi-example CLI runs
            # retain the explicit source-gradient calculation.
            if pcgrad_enabled and not pcgrad_fast_path:
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
            if pcgrad_fast_path:
                pcgrad.accumulate_existing_gradients(owner=owner_flags[0])
            microbatch_loss = float(batch_loss.item())
            rolling_loss += microbatch_loss
            rolling_count += 1
            accumulated_step_loss += microbatch_loss
            accumulated_ewc_loss += float(current_ewc_loss.detach().item())
            accum_micro_steps += 1
            answer_weighted_tokens += float((wb > 1.0).sum().item())
            total_target_tokens += float((yb != tokenizer.pad_token_id).sum().item())
            target_ids = yb[yb != tokenizer.pad_token_id]
            pending_trained_tokens += int(target_ids.numel())
            pending_token_ids.update(int(value) for value in target_ids.unique().tolist())
            if window_consumption is not None:
                pending_window_indices.extend(int(value) for value in sample_idx.tolist())

            for sample_loss, example_index in zip(
                sample_losses.detach().cpu().tolist(),
                sample_idx.tolist(),
                strict=True,
            ):
                entry = (float(sample_loss), int(example_index))
                if len(hard_examples) < HARD_EXAMPLE_KEEP:
                    heapq.heappush(hard_examples, entry)
                elif entry[0] > hard_examples[0][0]:
                    heapq.heapreplace(hard_examples, entry)

            if accum_micro_steps >= training_cfg.grad_accum_steps:
                if pcgrad_enabled:
                    pcgrad_reports.extend(pcgrad.materialize())
                if growth_alignment is not None:
                    growth_alignment.mask_inactive_gradients()
                gradient_norm = mp.clip_gradients(model, optimizer, 1.0)
                # The optimizer step represents all accumulation microbatches,
                # not merely the final one. The old final-microbatch value made
                # HAL and adaptive LR react to random hard examples.
                loss_float = accumulated_step_loss / accum_micro_steps
                last_ewc_loss = accumulated_ewc_loss / accum_micro_steps
                grad_float = float(gradient_norm)
                if intelligence_session is not None:
                    intelligence_session.record_optimizer_step(
                        step=global_step,
                        loss=loss_float,
                        learning_rate=float(optimizer.param_groups[0]["lr"]),
                        gradient_norm=grad_float,
                        tokens=int((yb != tokenizer.pad_token_id).sum().item()),
                    )
                if intelligence_session is not None:
                    hal = get_hal_module(model)
                    if hal is not None:
                        intelligence_session.record_hal_step(step=global_step, hal_state=hal.state)
                appended_rows_before = (
                    appended_row_lr.capture() if appended_row_lr is not None else None
                )
                mp.step(optimizer)
                if appended_row_lr is not None:
                    appended_row_lr.apply(appended_rows_before)
                mp.update()
                scheduler.step()
                campaign_tokens_seen += pending_trained_tokens
                phase_key = continuation_phase.upper()
                continuation_token_counts[phase_key] = (
                    continuation_token_counts.get(phase_key, 0) + pending_trained_tokens
                )
                known_token_ids.update(pending_token_ids)
                if window_consumption is not None:
                    window_consumption.mark(pending_window_indices)
                pending_trained_tokens = 0
                pending_token_ids.clear()
                pending_window_indices.clear()
                regret_lr = regret_scheduler.update(reward=max(0.0, 1.0 - loss_float))
                multiplier = max(0.5, min(1.5, regret_lr / max(learning_rate, 1e-12)))
                scheduled_lrs = scheduler.get_last_lr()
                for group, scheduled_lr in zip(
                    optimizer.param_groups,
                    scheduled_lrs,
                    strict=True,
                ):
                    group["lr"] = scheduled_lr * multiplier
                optimizer.zero_grad(set_to_none=True)
                pcgrad.clear()
                global_step += 1
                if growth_alignment is not None:
                    growth_alignment.configure_trainable_parameters(global_step - initial_step)
                session_step += 1
                accum_micro_steps = 0
                accumulated_step_loss = 0.0
                accumulated_ewc_loss = 0.0
                write_json(
                    OUTPUT_V2_DIR / "training_progress_journal.json",
                    {
                        "schema_version": 1,
                        "updated_at": time.time(),
                        "global_step": global_step,
                        "completed_optimizer_boundary": True,
                        "accumulation_step": 0,
                        "tokens_seen": campaign_tokens_seen,
                        "phase": continuation_phase.upper(),
                        "phase_tokens_seen": continuation_token_counts.get(
                            continuation_phase.upper(), 0
                        ),
                        "checkpoint_path": str(ckpt_path),
                    },
                )

                avg_loss = rolling_loss / max(1, rolling_count)
                last_avg_loss = avg_loss
                loss_ema = loss_float if loss_ema is None else 0.9 * loss_ema + 0.1 * loss_float
                best_loss = min(best_loss, loss_ema) if math.isfinite(best_loss) else loss_ema
                if (
                    training_layout == V2ConversationDataset.PACKING_LAYOUT
                    and global_step >= int(total_steps * 0.90)
                    and not annealing_started
                ):
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
                elapsed_seconds = max(1e-6, time.time() - start)
                tokens_per_second = total_target_tokens / elapsed_seconds
                optimizer_steps_per_hour = session_step * 3600.0 / elapsed_seconds
                if session_step % 10 == 0:
                    print(
                        f"  step={global_step:6d}"
                        f"  step_loss={loss_float:.4f}"
                        f"  ema_loss={loss_ema:.4f}"
                        f"  session_avg={avg_loss:.4f}"
                        f"  best_train={best_loss:.4f}"
                        f"  lr={optimizer.param_groups[0]['lr']:.2e}"
                        f"  tok/s={tokens_per_second:.0f}"
                        f"  steps/h={optimizer_steps_per_hour:.1f}"
                        f"  elapsed={elapsed_min:.1f}m",
                        flush=True,
                    )

                if global_step in EARLY_STATUS_STEPS or global_step % 200 == 0:
                    remaining_min = max(0.0, (end_at - time.time()) / 60.0)
                    startup_note = ""
                    if global_step in EARLY_STATUS_STEPS and first_batch_wall is not None:
                        startup_note = f"  startup={(first_batch_wall - start):.1f}s"
                    print(
                        f"  step={global_step:6d}  step_loss={loss_float:.4f}  "
                        f"ema_loss={loss_ema:.4f}  session_avg={avg_loss:.4f}  "
                        f"best_train={best_loss:.4f}  "
                        f"tok/s={tokens_per_second:.0f}  steps/h={optimizer_steps_per_hour:.1f}  "
                        f"elapsed={elapsed_min:.1f}m  remaining={remaining_min:.1f}m{startup_note}",
                        flush=True,
                    )

                if global_step % 250 == 0:
                    was_training = model.training
                    model.eval()
                    try:
                        validation_result = quick_eval_loss(
                            model,
                            eval_ds,
                            device=device,
                            max_examples=50,
                            batch_size=batch_size,
                            pad_id=tokenizer.pad_token_id,
                        )
                        validation_loss = _quick_eval_loss_value(validation_result)
                        best_validation_loss = min(
                            best_validation_loss,
                            validation_loss,
                        )
                        validation_history.append(
                            {
                                "step": global_step,
                                "loss": validation_loss,
                                "best_loss": best_validation_loss,
                            }
                        )
                        write_json(
                            v2_report_path("validation_history"),
                            {
                                "generated_at": time.time(),
                                "layout": eval_ds.PACKING_LAYOUT,
                                "history": validation_history,
                            },
                        )
                        print(
                            f"  validation step={global_step} loss={validation_loss:.4f} "
                            f"best={best_validation_loss:.4f}",
                            flush=True,
                        )
                    finally:
                        model.train(was_training)

                if (
                    global_step % durable_checkpoint_steps == 0
                    or time.time() >= next_checkpoint_at
                ):
                    payload = _build_checkpoint_payload(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        mp=mp,
                        global_step=global_step,
                        epoch=epoch,
                        best_loss=best_loss,
                        sessions_completed=(
                            int(ckpt.get("sessions_completed", 0)) if "ckpt" in locals() else 0
                        ),
                        mix_report=mix_report,
                        migration=checkpoint_migration,
                        tokens_seen=campaign_tokens_seen,
                        unique_token_ids_seen=known_token_ids,
                        continuation_token_counts=continuation_token_counts,
                        best_validation_loss=best_validation_loss,
                        validation_history=validation_history,
                        appended_row_optimizer_steps=(
                            appended_row_lr.steps_completed if appended_row_lr is not None else 0
                        ),
                        raw_window_consumption=(
                            window_consumption.state_dict()
                            if window_consumption is not None
                            else None
                        ),
                    )
                    atomic_save(payload, ckpt_path, drive_dir=None)
                    _sync_training_checkpoint_to_drive(ckpt_path)
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    try:
                        hal = get_hal_module(model)
                        if hal is not None:
                            publish_hal_state(hal, source="training")
                    except Exception as exc:
                        print(f"[HAL] checkpoint publish skipped: {exc}", flush=True)
                    next_checkpoint_at = time.time() + checkpoint_every_seconds

            if time.time() >= end_at:
                break
            if (
                max_phase_tokens is not None
                and continuation_token_counts.get(continuation_phase.upper(), 0)
                >= max_phase_tokens
            ):
                print(
                    f"[Campaign] phase token cap reached: {max_phase_tokens:,}",
                    flush=True,
                )
                break

    if accum_micro_steps > 0:
        optimizer.zero_grad(set_to_none=True)
        pcgrad.clear()
        pending_trained_tokens = 0
        pending_token_ids.clear()
        pending_window_indices.clear()
        print(
            "  discarded_partial_accum="
            f"{accum_micro_steps}/{training_cfg.grad_accum_steps}; "
            "checkpoint remains on the last complete optimizer boundary.",
            flush=True,
        )

    if global_step > initial_step and global_step % 200 != 0:
        elapsed_min = (time.time() - start) / 60.0
        remaining_min = max(0.0, (end_at - time.time()) / 60.0)
        print(
            f"  step={global_step:6d}  session_avg={last_avg_loss:.4f}  "
            f"best_train={best_loss:.4f}  "
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
        sessions_completed=(
            int(ckpt.get("sessions_completed", 0) + 1) if "ckpt" in locals() else 1
        ),
        mix_report=mix_report,
        migration=checkpoint_migration,
        tokens_seen=campaign_tokens_seen,
        unique_token_ids_seen=known_token_ids,
        continuation_token_counts=continuation_token_counts,
        best_validation_loss=best_validation_loss,
        validation_history=validation_history,
        appended_row_optimizer_steps=(
            appended_row_lr.steps_completed if appended_row_lr is not None else 0
        ),
        raw_window_consumption=(
            window_consumption.state_dict() if window_consumption is not None else None
        ),
    )
    atomic_save(payload, ckpt_path, drive_dir=None)
    _sync_training_checkpoint_to_drive(ckpt_path)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    try:
        hal = get_hal_module(model)
        if hal is not None:
            publish_hal_state(hal, source="training")
    except Exception as exc:
        print(f"[HAL] final publish skipped: {exc}", flush=True)

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
        "append_only_row_learning": (
            appended_row_lr.report() if appended_row_lr is not None else None
        ),
        "answer_supervision_ratio": round(ds.answer_supervision_ratio, 4),
        "data_layout": ds.PACKING_LAYOUT,
        "token_utilization": round(ds.token_utilization, 4),
        "reply_token_ratio_seen": round(answer_weighted_tokens / max(1.0, total_target_tokens), 4),
        "target_tokens_seen": int(total_target_tokens),
        "campaign_tokens_seen": campaign_tokens_seen,
        "phase_tokens_seen": continuation_token_counts.get(continuation_phase.upper(), 0),
        "raw_window_consumption": (
            window_consumption.report(
                phase_target_tokens=CONTINUATION_PHASE_TOKEN_TARGETS.get(continuation_phase.upper())
            )
            if window_consumption is not None
            else None
        ),
        "model_config": model.model_config(),
        "continuation": continuation_report,
        "best_validation_loss": best_validation_loss,
        "validation_history": validation_history,
        "checkpoint_path": str(ckpt_path),
        "mix_report": mix_report.to_dict(),
        "signal_handler": signal_state,
        "scheduler": {
            "name": "cosine_with_warmup",
            "warmup_steps": warmup_steps,
            "warmup_fraction": 0.02,
            "total_steps": total_steps,
            "min_lr": float(getattr(training_cfg, "min_lr", learning_rate * 0.1)),
            "annealing_started": annealing_started,
        },
        "ewc": {
            "active": ewc_weight > 0.0,
            "weight": ewc_weight,
            "last_optimizer_step_loss": locals().get("last_ewc_loss", 0.0),
            "state_path": ewc_state_value or None,
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
        for loss_value, sample_index in sorted(
            hard_examples, key=lambda item: item[0], reverse=True
        )
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

    try:
        eval_summary = run_compact_eval(model, tokenizer, device=device, output=True, seed=0)
    except Exception as exc:
        # The frontier checkpoint has already been persisted above. Evaluation
        # must report its own failure without converting a successful training
        # session into an apparent training failure.
        print(f"[Eval] compact evaluation failed after checkpoint save: {exc}", flush=True)
        eval_summary = {
            "overall_score": 0.0,
            "results": [],
            "error": f"{type(exc).__name__}: {exc}",
        }
        write_json(v2_report_path("eval_summary"), eval_summary)
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
        if ds.PACKING_LAYOUT == RawCausalShardDataset.PACKING_LAYOUT:
            raise LookupError("raw foundation continuation does not use identity mix control")
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
    except LookupError as exc:
        print(f"[OGRS] skipped: {exc}.", flush=True)
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
                        load_checkpoint(
                            model, optimizer, scheduler, mp, prev_ckpt, device=device, strict=False
                        )
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
        session_end_result = quick_eval_loss(
            model,
            eval_ds,
            device=device,
            max_examples=100,
            batch_size=batch_size,
            pad_id=tokenizer.pad_token_id,
        )
        session_end_loss = _quick_eval_loss_value(session_end_result)
        regret_lr = regret_scheduler.session_end(session_end_loss, global_step - initial_step)
        regret_scheduler.save(REGRET_STATE)
        print(f"  Dynamic regret lr : {regret_lr:.8f}", flush=True)
    except Exception as exc:
        print(f"[build_brain] quick eval at session_end failed: {exc}", flush=True)
    # The frontier checkpoint has exactly one Drive destination: the shared
    # master that was restored at session start. Do not invoke legacy V2
    # artifact mirroring here; it creates duplicate multi-gigabyte brain files.
    _sync_training_checkpoint_to_drive(ckpt_path)

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
    parser.add_argument("--batch_size", type=int, default=V2_FRONTIER_TRAINING.batch_size)
    parser.add_argument("--block_size", type=int, default=V2_FRONTIER.block_size)
    parser.add_argument("--max_minutes", type=int, default=V2_FRONTIER_TRAINING.session_minutes)
    parser.add_argument(
        "--model-size",
        choices=["frontier"],
        default="frontier",
    )
    parser.add_argument(
        "--answer_loss_weight", type=float, default=V2_FRONTIER_TRAINING.answer_loss_weight
    )
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument(
        "--training-layout",
        choices=[
            V2ConversationDataset.PACKING_LAYOUT,
            RawCausalShardDataset.PACKING_LAYOUT,
        ],
        default=V2ConversationDataset.PACKING_LAYOUT,
    )
    parser.add_argument("--token-shard-manifest", default=None)
    parser.add_argument("--validation-shard-manifest", default=None)
    parser.add_argument(
        "--continuation-phase",
        choices=["A", "B", "C", "D", "E"],
        default="D",
    )
    parser.add_argument(
        "--max-phase-tokens",
        type=int,
        default=None,
        help="Stop at the first complete optimizer boundary reaching this phase token count.",
    )
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
    parser.add_argument(
        "--optimizer",
        choices=["auto", "adamw", "adam8bit", "adafactor", "muon", "scale", "galore", "qgalore"],
        default="adafactor",
    )
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
        training_layout=args.training_layout,
        token_shard_manifest=args.token_shard_manifest,
        validation_shard_manifest=args.validation_shard_manifest,
        continuation_phase=args.continuation_phase,
        max_phase_tokens=args.max_phase_tokens,
    )
    print(result, flush=True)


if __name__ == "__main__":
    main()
