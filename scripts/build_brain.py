# Canonical phase trainer. The legacy scripts/train.py path is fail-closed.
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
    FAILURE_REPLAY_DATASET,
    IBS_LATEST,
    OUTPUT_V2_DIR,
    ROOT,
    SOVEREIGNTY_EVENTS,
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
from training.checkpoint_durability import (
    CheckpointDurabilitySession,
    build_checkpoint_lineage,
)
from training.continual import assess_continual_readiness, ewc_penalty
from training.curriculum_sampler import (
    CURRICULUMS,
    PERMUTATION_SAMPLER_ALGORITHM,
    SAMPLER_ALGORITHM,
    DeterministicPermutationSampler,
    ScheduledCurriculumSampler,
    source_replay_budget_violations,
    validate_sampler_resume_contract,
)
from training.data_routing import build_data_route_report
from training.eval_v2 import quick_eval_loss, run_compact_eval
from training.growth_runtime import load_growth_teacher, load_growth_training_pair
from training.mixed_precision import MixedPrecisionTrainer
from training.pcgrad import PCGradAccumulator
from training.reproducibility import (
    DETERMINISM_MODE,
    capture_rng_states,
    make_data_generator,
    seed_everything,
    seed_worker,
)
from training.scheduler import get_cosine_schedule_with_warmup
from training.shared_checkpoint import (
    restore_shared_checkpoint,
    sync_checkpoint_to_origin,
)
from training.v2_config import (
    ANRA_V4_GROWTH_MODEL_PROFILE,
    ANRA_V4_MODEL,
    ANRA_V4_TRAINING,
    CANONICAL_FOUNDATION_OPTIMIZER,
    CANONICAL_FOUNDATION_SCHEDULE,
    CANONICAL_MODEL_PROFILE,
    CANONICAL_TRAINING_SEED,
    CHECKPOINT_SCHEMA_VERSION,
    EXPECTED_TOKENIZER_VOCAB_SIZE,
    model_parameter_count,
    resolve_model_profile,
)
from training.v2_data_mix import (
    MixReport,
    RawCausalShardDataset,
    TrainingDataMixController,
    V2ConversationDataset,
    WindowConsumptionTracker,
    build_v2_training_examples,
    split_conversation_validation,
)
from training.v2_runtime import (
    active_tokenizer_identity,
    active_tokenizer_path,
    atomic_save,
    build_model_for_profile,
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
from training.verified_process import VERIFIED_PROCESS_OBJECTIVE

from scripts.session_dashboard import print_session_dashboard

EARLY_STATUS_STEPS = {1, 2, 5, 10, 20, 50, 100}
HARD_EXAMPLE_KEEP = 16
CONTINUATION_PHASE_TOKEN_TARGETS = {
    # Historical phase letters remain only as the on-disk checkpoint counter
    # key.  The active dense V4 foundation is one cumulative lineage ending at
    # 3.6B tokens; reporting a 1B ceiling here made later windows look complete.
    "A": 3_600_000_000,
    # B is reserved for bounded, paired architecture pilots.
    "B": 20_000_000,
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
    optimizer_name: str = CANONICAL_FOUNDATION_OPTIMIZER,
) -> object:
    """Canonical build-brain integration point for extension-only causal training."""
    if optimizer_name != CANONICAL_FOUNDATION_OPTIMIZER:
        raise ValueError("Operational V4 causal-extension training requires AdamW")
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
    if model_size != CANONICAL_MODEL_PROFILE:
        raise ValueError(
            f"causal-extension training requires --model-size {CANONICAL_MODEL_PROFILE}"
        )
    model = build_model_for_profile(model_size, vocab_size=tokenizer.vocab_size)
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


def _tokenizer_checkpoint_contract() -> dict[str, object]:
    identity = active_tokenizer_identity()
    if identity.get("available") is not True:
        raise FileNotFoundError(
            f"Canonical tokenizer is missing: {active_tokenizer_path()}"
        )
    return {
        "schema_version": int(identity["schema_version"]),
        "sha256": str(identity["sha256"]),
        "vocabulary_sha256": str(identity["vocabulary_sha256"]),
        "vocab_size": int(identity["vocab_size"]),
        "special_token_ids": dict(identity["special_token_ids"]),
        "probe_count": int(identity["probe_count"]),
        "probe_sha256": str(identity["probe_sha256"]),
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
    best_answer_validation_loss: float = float("inf"),
    validation_history: list[dict[str, object]] | None = None,
    appended_row_optimizer_steps: int = 0,
    raw_window_consumption: dict[str, object] | None = None,
    data_sampler_state: dict[str, object] | None = None,
    data_generator: torch.Generator | None = None,
    seed_contract: dict[str, object] | None = None,
    rng_states_override: dict[str, object] | None = None,
    token_window: dict[str, object] | None = None,
    growth_provenance: dict[str, object] | None = None,
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
    native_model = getattr(model, "model", model)
    payload: dict[str, object] = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "checkpoint_artifact_class": "full_resume",
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
        # Retained for legacy readers only. Never use it as a promotion metric.
        "best_loss": best_loss,
        "best_training_loss": best_loss,
        "loss_semantics": {
            "best_loss": "legacy alias of best_training_loss",
            "best_training_loss": "minimum exponential-moving-average weighted training loss",
            "best_validation_loss": "minimum loss on the immutable validation dataset",
            "best_answer_validation_loss": (
                "minimum answer-token-only loss on immutable conversational validation"
            ),
            "promotion_metric": "best_validation_loss plus behavioral and verifier gates",
        },
        "sessions_completed": sessions_completed,
        "tokens_seen": int(tokens_seen),
        "completed_optimizer_boundary": True,
        "accum_micro_steps": 0,
        "unique_token_ids_seen": sorted(unique_token_ids_seen or set()),
        "unique_tokens_seen": len(unique_token_ids_seen or set()),
        "continuation_token_counts": dict(continuation_token_counts or {}),
        "best_validation_loss": float(best_validation_loss),
        "best_answer_validation_loss": float(best_answer_validation_loss),
        "validation_history": list(validation_history or []),
        "appended_row_optimizer_steps": int(appended_row_optimizer_steps),
        "raw_window_consumption": dict(raw_window_consumption or {}),
        "data_sampler_state": dict(data_sampler_state or {}),
        "token_window": dict(token_window or {}),
        "growth_provenance": dict(growth_provenance or {}),
        "model_config": model.model_config(),
        "training_recipe": dict(getattr(native_model, "training_recipe", {})),
        "seed_contract": dict(seed_contract or {}),
        "hal_state": hal_state_dict(model),
        "mix_report": mix_report.to_dict(),
        "rng_states": (
            dict(rng_states_override)
            if rng_states_override is not None
            else capture_rng_states(data_generator=data_generator)
        ),
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
    payload["checkpoint_lineage"] = build_checkpoint_lineage(payload)
    return payload


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
) -> bool:
    """Prevent a checkpoint from silently continuing on another corpus profile."""
    saved = str(checkpoint_profile or "unknown").strip()
    active = active_profile.strip() or "unknown"
    if saved in {"unknown", "unlabelled", "unlabeled"}:
        print(
            "[Resume] Checkpoint predates data-profile tracking; continuity cannot be verified. "
            f"Training with active profile={active}.",
            flush=True,
        )
        return False
    if active in {"unknown", "unlabelled", "unlabeled"}:
        print(
            f"[Resume] Active data profile is not declared; checkpoint profile={saved}.",
            flush=True,
        )
        return False
    if saved == active:
        print(f"[Resume] Data profile verified: {active}", flush=True)
        return False
    if os.environ.get("ANRA_ALLOW_DATA_PROFILE_CHANGE", "0") == "1":
        print(
            f"[Resume] WARNING: data-profile change explicitly allowed: {saved} -> {active}.",
            flush=True,
        )
        return True
    raise RuntimeError(
        "Refusing to resume with a different data profile: "
        f"checkpoint={saved}, active={active}. Restore the original prepared corpus, "
        "or set ANRA_ALLOW_DATA_PROFILE_CHANGE=1 only for an intentional new experiment."
    )


def _assert_resume_data_layout_compatible(
    checkpoint_layout: object,
    active_layout: str,
    continuation_phase: str = "A",
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
    if resume_from:
        candidate = _resolve_checkpoint_path(resume_from)
        if not candidate.is_file():
            raise FileNotFoundError(f"Signed resume checkpoint is missing: {candidate}")
        if candidate.resolve() == checkpoint_path.resolve():
            raise RuntimeError("Resume source and mutable output checkpoint must differ")
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = checkpoint_path.with_suffix(checkpoint_path.suffix + ".resume.tmp")
        try:
            shutil.copy2(candidate, temporary)
            os.replace(temporary, checkpoint_path)
        finally:
            temporary.unlink(missing_ok=True)
        print(
            f"[build_brain] restored signed checkpoint: {candidate} -> {checkpoint_path}",
            flush=True,
        )
        return
    if checkpoint_path.exists():
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
        if os.environ.get("ANRA_REQUIRE_SHARED_MASTER", "0") == "1":
            raise
        print(
            "[Drive] continuing with the durable local checkpoint; "
            "shared publication is not required for this run.",
            flush=True,
        )


def _weighted_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    answer_mask: torch.Tensor,
    *,
    pad_id: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
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
    nonpad = targets != pad_id
    answer = answer_mask.bool() & nonpad
    scaffold = (~answer_mask.bool()) & nonpad
    breakdown = {
        "answer_nll_sum": per_token[answer].sum(),
        "answer_tokens": answer.sum(),
        "scaffold_nll_sum": per_token[scaffold].sum(),
        "scaffold_tokens": scaffold.sum(),
    }
    return sample_losses.mean(), sample_losses, breakdown


def _resolve_token_window_contract(
    window_id: str | None,
    start_token: int | None,
    end_token: int | None,
) -> dict[str, object] | None:
    """Resolve a launch-bound token window from CLI values or verified worker env."""
    resolved_id = (
        window_id
        if window_id is not None
        else os.environ.get("ANRA_TOKEN_WINDOW_ID", "").strip() or None
    )
    raw_start = (
        start_token
        if start_token is not None
        else os.environ.get("ANRA_TOKEN_WINDOW_START", "").strip() or None
    )
    raw_end = (
        end_token
        if end_token is not None
        else os.environ.get("ANRA_TOKEN_WINDOW_END", "").strip() or None
    )
    if resolved_id is None and raw_start is None and raw_end is None:
        return None
    if resolved_id is None or raw_start is None or raw_end is None:
        raise ValueError("Token-window id, start, and end must be supplied together")
    normalized_id = str(resolved_id).lower()
    if len(normalized_id) != 64 or any(
        character not in "0123456789abcdef" for character in normalized_id
    ):
        raise ValueError("Token-window id must be a SHA-256 hex digest")
    start = int(raw_start)
    end = int(raw_end)
    if start < 0 or end <= start:
        raise ValueError("Token window requires 0 <= start_token < end_token")
    return {
        "window_id": normalized_id,
        "start_token": start,
        "end_token": end,
    }


def _assert_token_window_start(
    token_window: dict[str, object] | None,
    *,
    phase_tokens_seen: int,
    scratch_run: bool,
) -> None:
    if token_window is None:
        return
    start = int(token_window["start_token"])
    if scratch_run and start != 0:
        raise RuntimeError(
            f"A scratch launch requires token-window start 0; received {start:,}"
        )
    if phase_tokens_seen != start:
        raise RuntimeError(
            "Checkpoint/token-window boundary mismatch: "
            f"checkpoint phase tokens={phase_tokens_seen:,}, signed start={start:,}"
        )


def _cap_batch_to_token_budget(
    xb: torch.Tensor,
    yb: torch.Tensor,
    wb: torch.Tensor,
    sample_idx: torch.Tensor,
    answer_mask: torch.Tensor,
    *,
    remaining_tokens: int,
    pad_id: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    int,
]:
    """Cap the final microbatch so a signed token window is never exceeded."""
    if remaining_tokens <= 0:
        raise ValueError("remaining_tokens must be positive")
    valid_per_sample = (yb != pad_id).sum(dim=1).tolist()
    total_valid = int(sum(valid_per_sample))
    if total_valid <= remaining_tokens:
        return xb, yb, wb, sample_idx, answer_mask, total_valid

    kept_samples = 0
    running = 0
    for count in valid_per_sample:
        kept_samples += 1
        running += int(count)
        if running >= remaining_tokens:
            break
    xb = xb[:kept_samples]
    yb = yb[:kept_samples].clone()
    wb = wb[:kept_samples].clone()
    sample_idx = sample_idx[:kept_samples]
    answer_mask = answer_mask[:kept_samples].clone()
    valid_positions = torch.nonzero(yb != pad_id, as_tuple=False)
    for row, column in valid_positions[remaining_tokens:].tolist():
        yb[row, column] = pad_id
        wb[row, column] = 0
        answer_mask[row, column] = False
    return xb, yb, wb, sample_idx, answer_mask, remaining_tokens


def _masked_logit_z_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    pad_id: int,
    weight: float,
) -> torch.Tensor:
    """Penalize runaway logit scale over real target positions only."""
    if weight < 0:
        raise ValueError("logit z-loss weight must be non-negative")
    if weight == 0:
        return logits.sum() * 0.0
    valid = (targets != pad_id).to(dtype=torch.float32)
    log_partition = torch.logsumexp(logits.float(), dim=-1)
    return float(weight) * (
        (log_partition.square() * valid).sum() / valid.sum().clamp_min(1.0)
    )


def _quick_eval_loss_value(result: float | dict[str, object]) -> float:
    return float(result["loss"]) if isinstance(result, dict) else float(result)


def _assert_training_loader_dataset(
    loader: DataLoader,
    training_dataset: object,
    validation_dataset: object,
) -> None:
    """Fail closed if a training loader crosses into the validation boundary."""
    if loader.dataset is not training_dataset:
        selected_validation = (
            training_dataset is not validation_dataset and loader.dataset is validation_dataset
        )
        reason = (
            "validation dataset selected"
            if selected_validation
            else "unknown dataset selected"
        )
        raise RuntimeError(f"training loader boundary violation: {reason}")


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
    if phase_name not in {"A", "B", "C", "D", "E"}:
        raise ValueError(f"unknown continuation phase: {phase}")
    subsystem_patterns = (
        "mod_routers.",
        "rim_modules.",
        "esv_module.",
        "residual_depth_logits",
        "dstp_temperature_log",
        "layer_temperature_bias_log",
    )
    phase_b_target = os.environ.get("ANRA_PHASE_B_SUBSYSTEM", "mod").strip().lower()
    target_patterns = {
        "mod": ("mod_routers.",),
        "rim": ("rim_modules.",),
        "dstp": (
            "residual_depth_logits",
            "dstp_temperature_log",
            "layer_temperature_bias_log",
        ),
        "esv": ("esv_module.",),
    }
    if phase_b_target not in target_patterns:
        raise ValueError("ANRA_PHASE_B_SUBSYSTEM must be mod, rim, dstp, or esv")

    known_subsystems = set(target_patterns)
    if phase_name == "A":
        enabled_subsystems: set[str] = set()
        policy_source = "dense_foundation_contract"
    elif phase_name == "B":
        enabled_subsystems = {phase_b_target}
        policy_source = "isolated_phase_b_ablation"
    else:
        declared = os.environ.get("ANRA_ENABLED_SUBSYSTEMS", "").strip()
        if not declared:
            raise RuntimeError(
                f"Phase {phase_name} requires an explicit ANRA_ENABLED_SUBSYSTEMS "
                "recipe; refusing an implicit all-on architecture"
            )
        enabled_subsystems = {
            value.strip().lower() for value in declared.split(",") if value.strip()
        }
        unknown = enabled_subsystems - known_subsystems
        if unknown:
            raise ValueError(f"ANRA_ENABLED_SUBSYSTEMS contains unknown values: {sorted(unknown)}")
        policy_source = "declared_architecture_recipe"

    if not hasattr(native_model, "configure_subsystems"):
        raise TypeError("Training model does not implement explicit subsystem policies")
    activation = native_model.configure_subsystems(enabled_subsystems)
    frozen: list[str] = []
    active: list[str] = []
    for name, parameter in native_model.named_parameters():
        if not name.startswith(subsystem_patterns):
            continue
        trainable = any(
            subsystem in enabled_subsystems and name.startswith(patterns)
            for subsystem, patterns in target_patterns.items()
        )
        parameter.requires_grad_(trainable)
        (active if trainable else frozen).append(name)
    capacity = 0.5 if "mod" in enabled_subsystems else 1.0
    parity = os.environ.get("ANRA_SUBSYSTEM_VALIDATION_PARITY", "0").lower()
    if phase_name in {"D", "E"} and parity in {"1", "true", "yes"}:
        capacity = 0.5
    if hasattr(native_model, "set_mod_capacity"):
        native_model.set_mod_capacity(capacity)
    report = {
        "phase": phase_name,
        "phase_b_subsystem": phase_b_target if phase_name == "B" else None,
        "mod_capacity": capacity,
        "enabled_subsystems": sorted(enabled_subsystems),
        "subsystem_activation": activation,
        "policy_source": policy_source,
        "active_subsystem_parameters": active,
        "frozen_subsystem_parameters": frozen,
    }
    print(
        f"[Continuation] phase={phase_name} enabled={sorted(enabled_subsystems)} "
        f"mod_capacity={capacity:.2f} "
        f"active_native={len(active)} frozen_native={len(frozen)}",
        flush=True,
    )
    return report


def train_anra_v2(
    *,
    data_path: str,
    checkpoint_path: str = "anra_v4_180m.pt",
    resume_from: str | None = None,
    batch_size: int = ANRA_V4_TRAINING.batch_size,
    accumulation: int = ANRA_V4_TRAINING.grad_accum_steps,
    block_size: int = ANRA_V4_MODEL.block_size,
    max_minutes: int = ANRA_V4_TRAINING.session_minutes,
    answer_loss_weight: float = ANRA_V4_TRAINING.answer_loss_weight,
    max_examples: int | None = None,
    own_ratio: float | None = None,
    identity_ratio: float | None = None,
    teacher_ratio: float | None = None,
    symbolic_ratio: float | None = None,
    replay_ratio: float | None = None,
    use_ouroboros: bool = False,
    model_size: str = CANONICAL_MODEL_PROFILE,
    optimizer_name: str = CANONICAL_FOUNDATION_OPTIMIZER,
    start_eval_examples: int = 0,
    training_layout: str = V2ConversationDataset.PACKING_LAYOUT,
    token_shard_manifest: str | None = None,
    validation_shard_manifest: str | None = None,
    continuation_phase: str = "A",
    max_phase_tokens: int | None = None,
    use_qk_norm: bool | None = None,
    attention_pattern: str | None = None,
    use_mtp: bool = False,
    use_moe: bool = False,
    curriculum: str = "none",
    seed: int = CANONICAL_TRAINING_SEED,
    post_session_eval: bool = True,
    rehearsal_interrupt_after_microsteps: int | None = None,
    token_window_id: str | None = None,
    token_window_start: int | None = None,
    token_window_end: int | None = None,
    growth_initialization: str | None = None,
    growth_manifest: str | None = None,
    growth_parent_checkpoint: str | None = None,
) -> dict[str, object]:
    for required_component in ("training_loop", "data_mix", "evaluation"):
        if not is_enabled(required_component):
            raise RuntimeError(
                f"Required component is disabled at its call site: {required_component}"
            )
    print_session_dashboard()
    growth_run = model_size == ANRA_V4_GROWTH_MODEL_PROFILE
    if model_size not in {CANONICAL_MODEL_PROFILE, ANRA_V4_GROWTH_MODEL_PROFILE}:
        raise ValueError("model size is not registered in the operational V4 lineage")
    if optimizer_name != CANONICAL_FOUNDATION_OPTIMIZER:
        raise ValueError(
            "Operational V4 foundation and growth training require AdamW; "
            "optimizer alternatives belong in isolated pilot code, not the canonical trainer"
        )
    growth_paths = (growth_initialization, growth_manifest, growth_parent_checkpoint)
    if growth_run and (not growth_manifest or not growth_parent_checkpoint):
        raise ValueError("The 500M child requires its growth manifest and parent checkpoint")
    if not growth_run and any(growth_paths):
        raise ValueError("The 181M foundation cannot bind model-growth artifacts")
    if growth_initialization and resume_from:
        raise ValueError("Growth initialization and exact resume are mutually exclusive")
    if accumulation < 1:
        raise ValueError("gradient accumulation must be positive")
    if training_layout not in {
        V2ConversationDataset.PACKING_LAYOUT,
        RawCausalShardDataset.PACKING_LAYOUT,
    }:
        raise ValueError(f"unsupported training layout: {training_layout}")
    if curriculum not in CURRICULUMS:
        raise ValueError(f"unsupported curriculum: {curriculum}")
    token_window = _resolve_token_window_contract(
        token_window_id,
        token_window_start,
        token_window_end,
    )
    if token_window is not None:
        signed_end = int(token_window["end_token"])
        if max_phase_tokens is not None and int(max_phase_tokens) != signed_end:
            raise ValueError(
                "--max-phase-tokens must equal the signed token-window end: "
                f"{max_phase_tokens:,} != {signed_end:,}"
            )
        max_phase_tokens = signed_end
    if curriculum != "none" and training_layout != RawCausalShardDataset.PACKING_LAYOUT:
        raise ValueError("pilot curricula require immutable raw causal shards")
    if rehearsal_interrupt_after_microsteps is not None:
        if post_session_eval:
            raise ValueError(
                "rehearsal interruption requires --post-session-eval none"
            )
        if rehearsal_interrupt_after_microsteps < 1:
            raise ValueError("rehearsal interruption microsteps must be positive")
    seed_report = seed_everything(seed)
    seed = seed_report.seed
    os.environ["ANRA_TRAINING_DATA_LAYOUT"] = training_layout
    model_cfg, training_cfg = resolve_model_profile(
        model_size,
        allow_experimental=growth_run,
    )
    growth_teacher = None
    growth_alignment = None
    growth_provenance: dict[str, object] | None = None
    if not torch.cuda.is_available() and os.environ.get("ANRA_ALLOW_CPU_PILOT", "0") != "1":
        raise RuntimeError(
            f"{model_size} training requires CUDA; set ANRA_ALLOW_CPU_PILOT=1 "
            "only for a deliberately tiny trainer smoke test"
        )
    if max_examples is None:
        max_examples = training_cfg.max_mixture_examples
    own_ratio = own_ratio if own_ratio is not None else training_cfg.own_ratio
    identity_ratio = identity_ratio if identity_ratio is not None else training_cfg.identity_ratio
    teacher_ratio = teacher_ratio if teacher_ratio is not None else training_cfg.teacher_ratio
    symbolic_ratio = symbolic_ratio if symbolic_ratio is not None else training_cfg.symbolic_ratio
    replay_ratio = replay_ratio if replay_ratio is not None else training_cfg.replay_ratio
    print(
        f"[Trainer] {model_size.upper()} SCRATCH TRAINING  "
        f"batch={batch_size} grad_accum={accumulation}",
        flush=True,
    )
    dataset_path = Path(data_path)
    tokenizer = load_or_build_v2_tokenizer(dataset_path=dataset_path)
    tokenizer_identity = active_tokenizer_identity()
    if tokenizer_identity.get("available") is not True:
        raise RuntimeError("Active tokenizer identity could not be established")
    model_parameter_contract = model_parameter_count(
        model_cfg,
        tokenizer.vocab_size,
        mtp_depth=2 if use_mtp else 0,
        moe_routed_experts=8 if use_moe else 0,
    )
    data_mix_seed = seed
    training_mix_controller = TrainingDataMixController(model_parameter_contract)
    print(f"[Trainer] Data mix sampling seed: {data_mix_seed}", flush=True)
    print(
        "[Trainer] Seed contract: "
        f"{seed_report.determinism_mode} seed={seed_report.seed} "
        f"python_hash_seed_matches={seed_report.python_hash_seed_matches}",
        flush=True,
    )
    raw_sampling_policy = "source_weighted_replacement_v1"
    active_sampler_algorithm = "torch_random_sampler_v1"
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
            expected_tokenizer_sha256=str(tokenizer_identity["sha256"]),
            verified_process_multiplier=training_cfg.verified_process_multiplier,
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
            expected_tokenizer_sha256=str(tokenizer_identity["sha256"]),
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
        training_examples, validation_examples, conversation_split = (
            split_conversation_validation(examples)
        )
        write_json(
            OUTPUT_V2_DIR / "data_manifests" / "conversation_validation_split.json",
            conversation_split,
        )
        ds = V2ConversationDataset(
            training_examples,
            tokenizer,
            block_size,
            answer_loss_weight=answer_loss_weight,
        )
        eval_ds = V2ConversationDataset(
            validation_examples,
            tokenizer,
            block_size,
            answer_loss_weight=answer_loss_weight,
            validation_identity=str(conversation_split["split_sha256"]),
        )
        if ds is eval_ds:
            raise RuntimeError("conversation training and validation datasets must be distinct")
    if training_layout == RawCausalShardDataset.PACKING_LAYOUT:
        manifest_payload = _read_json(manifest_path) or {}
        raw_sampling_policy = str(
            manifest_payload.get("sampling_policy", "source_weighted_replacement_v1")
        )
        if raw_sampling_policy not in {
            "source_weighted_replacement_v1",
            PERMUTATION_SAMPLER_ALGORITHM,
        }:
            raise RuntimeError(f"unsupported raw sampling policy: {raw_sampling_policy}")
        active_sampler_algorithm = (
            PERMUTATION_SAMPLER_ALGORITHM
            if raw_sampling_policy == PERMUTATION_SAMPLER_ALGORITHM
            else SAMPLER_ALGORITHM
        )
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
    write_json(v2_report_path("data_route_report.json"), data_route_report)
    write_json(v2_report_path("mix_report"), mix_report.to_dict())
    if len(ds) == 0:
        raise RuntimeError(f"{training_layout} produced zero training windows.")
    window_consumption = (
        WindowConsumptionTracker(len(ds), block_size)
        if training_layout == RawCausalShardDataset.PACKING_LAYOUT
        else None
    )

    data_generator = make_data_generator(seed)
    raw_sample_budget: int | None = None
    if training_layout == RawCausalShardDataset.PACKING_LAYOUT:
        signed_profile_reset = (
            os.environ.get("ANRA_RESET_DATA_SAMPLER_ON_PROFILE_CHANGE", "0") == "1"
        )
        target_windows = (
            len(ds)
            if signed_profile_reset
            else math.ceil(max_phase_tokens / block_size)
            if max_phase_tokens is not None
            else len(ds)
        )
        optimizer_windows = batch_size * accumulation
        raw_sample_budget = (
            math.ceil(target_windows / optimizer_windows) * optimizer_windows
        )
    data_sampler_position = 0

    def current_data_sampler_state() -> dict[str, object]:
        if raw_sample_budget is None:
            return {}
        state: dict[str, object] = {
            "schema_version": 1,
            "algorithm": active_sampler_algorithm,
            "seed": seed,
            "position": data_sampler_position,
            "num_samples": raw_sample_budget,
            "curriculum": curriculum,
        }
        if active_sampler_algorithm == PERMUTATION_SAMPLER_ALGORITHM:
            state["dataset_size"] = len(ds)
        return state

    def make_loader(
        active_weights: dict[str, float] | None = None,
        *,
        sample_offset: int = 0,
    ) -> DataLoader:
        num_workers = 2 if torch.cuda.is_available() else 0
        loader_kwargs = {
            "batch_size": batch_size,
            "drop_last": False,
            "pin_memory": torch.cuda.is_available(),
            "num_workers": num_workers,
            "persistent_workers": num_workers > 0,
            "generator": data_generator,
            "worker_init_fn": seed_worker,
        }
        if training_layout == RawCausalShardDataset.PACKING_LAYOUT:
            if raw_sampling_policy == PERMUTATION_SAMPLER_ALGORITHM:
                if curriculum != "none":
                    raise RuntimeError(
                        "compact permutation packs support only the dense foundation curriculum"
                    )
                assert raw_sample_budget is not None
                if raw_sample_budget > len(ds):
                    raise RuntimeError(
                        "compact permutation pack has fewer unique windows than its token budget: "
                        f"windows={len(ds)} requested={raw_sample_budget}"
                    )
                sampler = DeterministicPermutationSampler(
                    len(ds),
                    num_samples=raw_sample_budget,
                    seed=data_mix_seed,
                    start_position=sample_offset,
                )
                return DataLoader(ds, sampler=sampler, **loader_kwargs)
            ranges = ds.source_window_ranges()
            if curriculum != "none":
                required_source = {
                    "code-before-prose": "permissive_code",
                    "math-density-ramp": "finemath",
                    "identity-mix-late": "identity_replay",
                }[curriculum]
                if required_source not in ranges:
                    raise RuntimeError(
                        f"curriculum {curriculum} requires source class {required_source}"
                    )
            target_mix = ds.manifest.get("campaign_mix_target", {})
            if target_mix and ds.manifest.get("campaign_mix_verified") is not True:
                raise RuntimeError("raw campaign manifest has an unverified source-mix recipe")
            if not isinstance(target_mix, dict):
                raise RuntimeError("raw campaign source-mix recipe must be an object")
            assert raw_sample_budget is not None
            replay_violations = source_replay_budget_violations(
                {
                    name: sum(stop - start for start, stop in source_ranges)
                    for name, source_ranges in ranges.items()
                },
                {str(key): float(value) for key, value in target_mix.items()},
                num_samples=raw_sample_budget,
            )
            if replay_violations:
                raise RuntimeError(
                    "raw foundation source mix exceeds the unique-data replay budget; "
                    "move small supervised sources to structured continuation instead: "
                    f"{replay_violations}"
                )
            sampler = ScheduledCurriculumSampler(
                ranges,
                curriculum=curriculum,
                num_samples=raw_sample_budget,
                seed=data_mix_seed,
                start_position=sample_offset,
                target_mass={str(key): float(value) for key, value in target_mix.items()}
                if target_mix
                else None,
            )
            return DataLoader(ds, sampler=sampler, **loader_kwargs)
        if active_weights is None or training_layout == RawCausalShardDataset.PACKING_LAYOUT:
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
    _assert_training_loader_dataset(loader, ds, eval_ds)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        # ``seed_everything`` establishes the reproducible-same-stack
        # contract. Do not silently undo it here: cuDNN benchmarking can
        # select a different kernel after a restart even when the shapes are
        # fixed. This transformer is dominated by matmul/attention kernels,
        # so enabling convolution autotuning has no justified foundation
        # benefit anyway.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    model = build_model_for_profile(
        model_size,
        block_size=block_size,
        vocab_size=tokenizer.vocab_size,
        use_qk_norm=use_qk_norm,
        attention_pattern=attention_pattern,
        use_mtp=use_mtp,
        use_moe=use_moe,
        allow_experimental=growth_run,
    )
    actual_parameters = sum(parameter.numel() for parameter in model.parameters())
    if actual_parameters != model_parameter_contract:
        raise AssertionError(
            f"{model_size} parameter accounting mismatch: "
            f"{actual_parameters:,} != {model_parameter_contract:,}"
        )
    if getattr(training_cfg, "gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
        print("[build_brain] Gradient checkpointing enabled for V4 model", flush=True)
    if hasattr(model, "disable_kv_cache"):
        model.disable_kv_cache()
    if use_ouroboros:
        from ouroboros import OuroborosDecoder

        model = OuroborosDecoder(model, n_passes=3)
    if growth_run:
        if use_ouroboros or use_mtp or use_moe:
            raise ValueError("Growth stabilization must use the dense registered child")
        if growth_initialization:
            growth_teacher, growth_alignment, growth_provenance = load_growth_training_pair(
                model,
                initialization_path=growth_initialization,
                growth_manifest_path=str(growth_manifest),
                parent_checkpoint_path=str(growth_parent_checkpoint),
            )
        else:
            growth_teacher, growth_alignment, growth_provenance = load_growth_teacher(
                model,
                growth_manifest_path=str(growth_manifest),
                parent_checkpoint_path=str(growth_parent_checkpoint),
            )
    model = model.to(device)
    ensure_tied_lm_head(model)
    native_model = getattr(model, "model", model)
    growth_recipe = (
        {
            key: growth_provenance[key]
            for key in (
                "schema",
                "growth_manifest_sha256",
                "parent_checkpoint_sha256",
                "identity_layers",
                "optimizer_restart_required",
                "alignment_steps",
                "new_only_steps",
            )
        }
        if growth_provenance is not None
        else None
    )
    native_model.training_recipe = {
        "model_profile": model_size,
        "training_layout": training_layout,
        "curriculum": curriculum,
        "max_phase_tokens": max_phase_tokens,
        "optimizer": optimizer_name,
        "seed": seed,
        "schedule": CANONICAL_FOUNDATION_SCHEDULE,
        "gradient_clip_norm": training_cfg.max_grad_norm,
        "verified_process_objective": VERIFIED_PROCESS_OBJECTIVE,
        "verified_process_multiplier": training_cfg.verified_process_multiplier,
        "micro_batch_size": batch_size,
        "gradient_accumulation": accumulation,
        "sampler_algorithm": active_sampler_algorithm,
        "determinism_mode": DETERMINISM_MODE,
        "growth": growth_recipe,
    }
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
    actual_optimizer = str(optimizer_report.get("selected", {}).get("actual", ""))
    if actual_optimizer != optimizer_name:
        raise RuntimeError(
            f"Pilot requested optimizer={optimizer_name}, but backend selected "
            f"{actual_optimizer or 'unknown'}; causal pilot cells may not use fallbacks"
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
    if growth_initialization and ckpt_path.exists():
        raise RuntimeError(
            "Growth output already exists; use a new artifact path or an explicit "
            "full-resume launch"
        )
    _prepare_resume_target(ckpt_path, resume_from)
    if (
        os.environ.get("ANRA_REQUIRE_RESUME", "0") == "1"
        and not ckpt_path.exists()
        and not growth_initialization
    ):
        raise RuntimeError(
            "ANRA_REQUIRE_RESUME=1, but no checkpoint was restored. "
            "Refusing to start from scratch and overwrite the intended experiment."
        )
    scratch_run = not ckpt_path.exists()
    durability = CheckpointDurabilitySession.from_environment(
        OUTPUT_V2_DIR / "durability" / "outbox",
        scratch_run=scratch_run,
    )
    if durability.required and token_window is None:
        raise RuntimeError(
            "Required cluster durability also requires a signed token window. "
            "Set ANRA_TOKEN_WINDOW_ID/START/END or pass the matching CLI arguments."
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
    lineage_source = (
        Path(growth_initialization)
        if growth_initialization
        else ckpt_path
        if ckpt_path.exists()
        else resume_path
    )
    _freeze_training_lineage(
        checkpoint_path=lineage_source,
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
    best_answer_validation_loss = float("inf")
    validation_history: list[dict[str, object]] = []

    def _publish_training_checkpoint(
        payload: dict[str, object],
        *,
        final: bool = False,
    ) -> None:
        if durability.enabled:
            ref = durability.publish_checkpoint(ckpt_path, payload, final=final)
            if ref is not None:
                status = json.loads(
                    durability.outbox.status_path(ref.snapshot_id).read_text(
                        encoding="utf-8"
                    )
                )
                print(
                    f"[Durability] snapshot={ref.snapshot_id} state={status.get('state')}",
                    flush=True,
                )
            return
        _sync_training_checkpoint_to_drive(ckpt_path)

    registration_ts = time.time()
    signal_state: dict[str, object] = {
        "registered_at": registration_ts,
        "registered_at_iso": _utc_iso(registration_ts),
        "triggered": False,
        "signal": None,
        "emergency_save_completed": None,
    }
    termination_request: dict[str, int | None] = {"signal": None}
    boundary_rng_states = capture_rng_states(data_generator=data_generator)
    boundary_epoch = epoch

    def _handle_sigterm(sig_num: int, _frame: object) -> None:
        signal_state["triggered"] = True
        signal_state["signal"] = sig_num
        termination_request["signal"] = sig_num
        print(
            f"[build_brain] termination requested (signal={sig_num}) at {_utc_iso()}; "
            "deferring save until an optimizer-safe boundary.",
            flush=True,
        )

    def _save_interrupted_boundary(
        sig_num: int,
        *,
        discarded_micro_steps: int,
    ) -> None:
        sessions_completed = int(ckpt.get("sessions_completed", 0) + 1)
        payload = _build_checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            mp=mp,
            global_step=global_step,
            epoch=boundary_epoch,
            best_loss=best_loss,
            sessions_completed=sessions_completed,
            mix_report=mix_report,
            migration=checkpoint_migration,
            tokens_seen=campaign_tokens_seen,
            unique_token_ids_seen=known_token_ids,
            continuation_token_counts=continuation_token_counts,
            best_validation_loss=best_validation_loss,
            best_answer_validation_loss=best_answer_validation_loss,
            validation_history=validation_history,
            appended_row_optimizer_steps=(
                appended_row_lr.steps_completed if appended_row_lr is not None else 0
            ),
            raw_window_consumption=(
                window_consumption.state_dict() if window_consumption is not None else None
            ),
            data_sampler_state=current_data_sampler_state(),
            data_generator=data_generator,
            seed_contract=seed_report.to_dict(),
            rng_states_override=boundary_rng_states,
            token_window=token_window,
            growth_provenance=growth_provenance,
        )
        payload["interruption"] = {
            "signal": int(sig_num),
            "safe_optimizer_boundary": True,
            "discarded_micro_steps": int(discarded_micro_steps),
            "global_step": int(global_step),
            "sampler_position": int(data_sampler_position),
            "requested_at": _utc_iso(),
        }
        ok = _emergency_save_with_timeout(payload, ckpt_path)
        if ok:
            _publish_training_checkpoint(payload, final=True)
        signal_state["emergency_save_completed"] = ok
        print(f"[build_brain] deferred termination save status={ok}", flush=True)
        raise SystemExit(128 + sig_num)

    signal.signal(signal.SIGTERM, _handle_sigterm)
    if os.name == "nt" and hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, _handle_sigterm)
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
            model,
            optimizer,
            scheduler,
            mp,
            load_path,
            device=device,
            strict=False,
            resume_training=True,
            data_generator=data_generator,
            sampler_reset_token=(
                token_window_start
                if os.environ.get("ANRA_ALLOW_DATA_PROFILE_CHANGE", "0") == "1"
                and os.environ.get(
                    "ANRA_RESET_DATA_SAMPLER_ON_PROFILE_CHANGE", "0"
                )
                == "1"
                else None
            ),
            continuation_phase=continuation_phase,
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
            data_profile_changed = _assert_resume_data_profile_compatible(
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
            best_answer_validation_loss = float(
                resume_state.get("best_answer_validation_loss", float("inf"))
            )
            validation_history = list(resume_state.get("validation_history", []))
            saved_growth_provenance = resume_state.get("growth_provenance", {})
            if growth_run:
                if not isinstance(saved_growth_provenance, dict) or not saved_growth_provenance:
                    raise RuntimeError("Growth resume checkpoint is missing its growth provenance")
                stable_saved = {
                    key: saved_growth_provenance.get(key)
                    for key in dict(growth_recipe or {})
                }
                if stable_saved != growth_recipe:
                    raise RuntimeError("Growth resume lineage differs from the active manifest")
                growth_provenance = dict(saved_growth_provenance)
            checkpoint_migration = dict(resume_state.get("migration", {}))
            if appended_row_lr is not None:
                appended_row_lr.steps_completed = int(
                    resume_state.get("appended_row_optimizer_steps", 0)
                )
            if window_consumption is not None:
                reset_sampler = (
                    data_profile_changed
                    and os.environ.get(
                        "ANRA_RESET_DATA_SAMPLER_ON_PROFILE_CHANGE", "0"
                    )
                    == "1"
                )
                if reset_sampler:
                    data_sampler_position = 0
                    print(
                        "[Resume] Signed data-profile transition reset the sampler "
                        "and window-consumption evidence for the new corpus.",
                        flush=True,
                    )
                else:
                    raw_consumption_state = resume_state.get(
                        "raw_window_consumption", {}
                    )
                    if isinstance(raw_consumption_state, dict) and raw_consumption_state:
                        window_consumption.load_state_dict(raw_consumption_state)
                    sampler_state = resume_state.get("data_sampler_state", {})
                    if not isinstance(sampler_state, dict) or not sampler_state:
                        raise RuntimeError("Raw V4 resume is missing its sampler cursor")
                    data_sampler_position = validate_sampler_resume_contract(
                        sampler_state,
                        seed=seed,
                        curriculum=curriculum,
                        active_num_samples=int(raw_sample_budget or 0),
                        algorithm=active_sampler_algorithm,
                        dataset_size=(
                            len(ds)
                            if active_sampler_algorithm == PERMUTATION_SAMPLER_ALGORITHM
                            else None
                        ),
                    )
                    visits = (
                        window_consumption.unique_windows
                        + window_consumption.repeated_windows
                    )
                    if visits != data_sampler_position:
                        raise RuntimeError(
                            "Raw V4 sampler cursor disagrees with window-consumption "
                            f"evidence: cursor={data_sampler_position}, visits={visits}"
                        )
                loader = make_loader(sample_offset=data_sampler_position)
                _assert_training_loader_dataset(loader, ds, eval_ds)
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

    if growth_initialization and not load_path.exists():
        if growth_provenance is None:
            raise RuntimeError("Growth initialization is missing its verified provenance")
        parent_progress = growth_provenance.get("parent_progress", {})
        if not isinstance(parent_progress, dict):
            raise RuntimeError("Growth initialization is missing its parent cursor")
        data_profile_changed = _assert_resume_data_profile_compatible(
            parent_progress.get("data_profile"),
            os.environ.get("ANRA_DATA_PROFILE", "unknown"),
        )
        reset_growth_sampler = (
            data_profile_changed
            and os.environ.get("ANRA_RESET_DATA_SAMPLER_ON_PROFILE_CHANGE", "0") == "1"
        )
        if data_profile_changed and not reset_growth_sampler:
            raise RuntimeError(
                "A growth data-window transition must explicitly reset its pack-local sampler"
            )
        _assert_resume_data_layout_compatible(
            parent_progress.get("training_data_layout"),
            _active_training_data_layout(),
            continuation_phase,
        )
        campaign_tokens_seen = int(parent_progress.get("tokens_seen", 0))
        continuation_token_counts.update(
            {
                str(name): int(value)
                for name, value in dict(
                    parent_progress.get("continuation_token_counts", {})
                ).items()
            }
        )
        best_validation_loss = float(
            parent_progress.get("best_validation_loss", float("inf"))
        )
        best_answer_validation_loss = float(
            parent_progress.get("best_answer_validation_loss", float("inf"))
        )
        validation_history = list(parent_progress.get("validation_history", []))
        if window_consumption is not None:
            if reset_growth_sampler:
                data_sampler_position = 0
                print(
                    "[Growth] Signed child window starts with a fresh pack-local "
                    "cursor while preserving the parent's cumulative token boundary.",
                    flush=True,
                )
            else:
                raw_consumption = parent_progress.get("raw_window_consumption", {})
                sampler_state = parent_progress.get("data_sampler_state", {})
                if not isinstance(raw_consumption, dict) or not raw_consumption:
                    raise RuntimeError("Growth parent is missing window-consumption evidence")
                if not isinstance(sampler_state, dict) or not sampler_state:
                    raise RuntimeError("Growth parent is missing its exact sampler cursor")
                window_consumption.load_state_dict(raw_consumption)
                data_sampler_position = validate_sampler_resume_contract(
                    sampler_state,
                    seed=seed,
                    curriculum=curriculum,
                    active_num_samples=int(raw_sample_budget or 0),
                    algorithm=active_sampler_algorithm,
                    dataset_size=(
                        len(ds)
                        if active_sampler_algorithm == PERMUTATION_SAMPLER_ALGORITHM
                        else None
                    ),
                )
                visits = (
                    window_consumption.unique_windows + window_consumption.repeated_windows
                )
                if visits != data_sampler_position:
                    raise RuntimeError(
                        "Growth parent sampler cursor disagrees with consumption evidence"
                    )
            loader = make_loader(sample_offset=data_sampler_position)
            _assert_training_loader_dataset(loader, ds, eval_ds)
        print(
            "[Growth] Fresh AdamW initialized; inherited model, corpus cursor, "
            "and cumulative token lineage verified.",
            flush=True,
        )

    phase_key = continuation_phase.upper()
    _assert_token_window_start(
        token_window,
        phase_tokens_seen=continuation_token_counts.get(phase_key, 0),
        scratch_run=scratch_run and not bool(growth_initialization),
    )
    if growth_alignment is not None:
        growth_alignment.configure_trainable_parameters(start_step)
    if token_window is not None:
        print(
            "[Token Window] "
            f"id={token_window['window_id']} "
            f"range=[{int(token_window['start_token']):,}, "
            f"{int(token_window['end_token']):,})",
            flush=True,
        )

    boundary_rng_states = capture_rng_states(data_generator=data_generator)
    boundary_epoch = epoch
    if termination_request["signal"] is not None:
        _save_interrupted_boundary(int(termination_request["signal"]), discarded_micro_steps=0)

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
                eval_ds,
                device=device,
                max_examples=start_eval_examples,
                batch_size=batch_size,
                pad_id=tokenizer.pad_token_id,
            )
            session_start_loss = _quick_eval_loss_value(session_start_result)
            answer_start_loss = session_start_result.get("answer_loss")
            best_validation_loss = min(best_validation_loss, session_start_loss)
            if answer_start_loss is not None:
                best_answer_validation_loss = min(
                    best_answer_validation_loss,
                    float(answer_start_loss),
                )
            validation_history.append(
                {
                    "step": global_step,
                    "kind": "preflight",
                    **session_start_result,
                    "best_validation_loss": best_validation_loss,
                    "best_answer_validation_loss": best_answer_validation_loss,
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
        except Exception as exc:
            print(f"[build_brain] quick eval at session_start failed: {exc}", flush=True)
            session_start_loss = best_loss
    else:
        print("[build_brain] startup quick eval skipped so first loss appears sooner.", flush=True)
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
    if durability.enabled:
        # A durable run checkpoints at whichever boundary arrives first.  The
        # caps cannot be relaxed by a stale notebook environment.
        checkpoint_every_seconds = min(checkpoint_every_seconds, 15 * 60)
        durable_checkpoint_steps = min(durable_checkpoint_steps, 100)
    next_checkpoint_at = time.time() + checkpoint_every_seconds
    optimizer.zero_grad(set_to_none=True)
    rolling_loss = 0.0
    rolling_count = 0
    accumulated_step_loss = 0.0
    accumulated_ewc_loss = 0.0
    accumulated_logit_z_loss = 0.0
    accumulated_answer_nll = 0.0
    accumulated_answer_tokens = 0
    accumulated_scaffold_nll = 0.0
    accumulated_scaffold_tokens = 0
    accum_micro_steps = 0
    session_micro_steps = 0
    pending_trained_tokens = 0
    pending_token_ids: set[int] = set()
    pending_window_indices: list[int] = []
    last_avg_loss = best_loss if math.isfinite(best_loss) else 0.0
    loss_ema: float | None = None
    first_batch_wall = None
    hard_examples: list[tuple[float, int]] = []
    answer_weighted_tokens = 0.0
    verified_process_weighted_tokens = 0
    total_target_tokens = 0.0

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    gpu_mem = (
        torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0.0
    )
    summary = model_summary(model)
    eff_batch = batch_size * accumulation
    pcgrad_fast_path = pcgrad_enabled and batch_size == 1

    print("", flush=True)
    print("=" * 62, flush=True)
    print("  AN-RA V2 TRAINING SESSION", flush=True)
    print("=" * 62, flush=True)
    print(f"  GPU          : {gpu_name} ({gpu_mem:.1f} GB)", flush=True)
    print(f"  Parameters   : {summary['parameters']:,}", flush=True)
    print(
        f"  Micro batch  : {batch_size}  |  Grad accum : "
        f"{accumulation}  |  Eff batch : {eff_batch}",
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
        if raw_sample_budget is not None:
            if data_sampler_position >= raw_sample_budget:
                break
            loader = make_loader(sample_offset=data_sampler_position)
            _assert_training_loader_dataset(loader, ds, eval_ds)
        epoch += 1
        for xb, yb, wb, sample_idx, answer_mask in loader:
            signed_window_boundary = False
            if token_window is not None:
                remaining_window_tokens = int(token_window["end_token"]) - (
                    continuation_token_counts.get(phase_key, 0)
                    + pending_trained_tokens
                )
                if remaining_window_tokens <= 0:
                    break
                (
                    xb,
                    yb,
                    wb,
                    sample_idx,
                    answer_mask,
                    accepted_batch_tokens,
                ) = _cap_batch_to_token_budget(
                    xb,
                    yb,
                    wb,
                    sample_idx,
                    answer_mask,
                    remaining_tokens=remaining_window_tokens,
                    pad_id=tokenizer.pad_token_id,
                )
                signed_window_boundary = accepted_batch_tokens == remaining_window_tokens
            if intelligence_session is not None:
                intelligence_session.begin_step(global_step)
            if first_batch_wall is None:
                first_batch_wall = time.time()
            verified_process_weighted_tokens += int((wb > 1.0).sum().item())
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            wb = wb.to(device, non_blocking=True)
            answer_mask = answer_mask.to(device, non_blocking=True)
            with mp.autocast():
                logits, _ = model(xb)
                batch_loss, sample_losses, loss_breakdown = _weighted_loss(
                    logits,
                    yb,
                    wb,
                    answer_mask,
                    pad_id=tokenizer.pad_token_id,
                )
                current_logit_z_loss = _masked_logit_z_loss(
                    logits,
                    yb,
                    pad_id=tokenizer.pad_token_id,
                    weight=training_cfg.logit_z_loss_weight,
                )
                batch_loss = batch_loss + current_logit_z_loss
                native_model = getattr(model, "model", model)
                current_mtp_loss = (
                    native_model.multi_token_prediction_loss(yb)
                    if use_mtp and hasattr(native_model, "multi_token_prediction_loss")
                    else torch.zeros((), device=batch_loss.device, dtype=batch_loss.dtype)
                )
                batch_loss = batch_loss + current_mtp_loss
                if growth_alignment is not None:
                    alignment_step = max(0, global_step)
                    alignment_penalty = growth_alignment.alignment_loss(
                        xb,
                        step=alignment_step,
                        target_logits=logits,
                    )
                    batch_loss = batch_loss + alignment_penalty
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
                loss = batch_loss / accumulation

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
                accumulated_logit_z_loss = 0.0
                accumulated_answer_nll = 0.0
                accumulated_answer_tokens = 0
                accumulated_scaffold_nll = 0.0
                accumulated_scaffold_tokens = 0
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
                        resume_training=True,
                        data_generator=data_generator,
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
                    sample_losses[owner_positions].mean() / accumulation
                    if owner_positions
                    else None
                )
                other_loss = (
                    sample_losses[other_positions].mean() / accumulation
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
            accumulated_logit_z_loss += float(current_logit_z_loss.detach().item())
            accumulated_answer_nll += float(loss_breakdown["answer_nll_sum"].detach().item())
            accumulated_answer_tokens += int(loss_breakdown["answer_tokens"].detach().item())
            accumulated_scaffold_nll += float(
                loss_breakdown["scaffold_nll_sum"].detach().item()
            )
            accumulated_scaffold_tokens += int(
                loss_breakdown["scaffold_tokens"].detach().item()
            )
            accum_micro_steps += 1
            session_micro_steps += 1
            if (
                rehearsal_interrupt_after_microsteps is not None
                and session_micro_steps == rehearsal_interrupt_after_microsteps
            ):
                print(
                    "[build_brain] rehearsal fault injection requested after "
                    f"{session_micro_steps} microsteps.",
                    flush=True,
                )
                _handle_sigterm(signal.SIGTERM, None)
            answer_weighted_tokens += float(answer_mask.sum().item())
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

            if accum_micro_steps >= accumulation or signed_window_boundary:
                if pcgrad_enabled:
                    pcgrad_reports.extend(pcgrad.materialize())
                if growth_alignment is not None:
                    growth_alignment.mask_inactive_gradients()
                if signed_window_boundary and accum_micro_steps < accumulation:
                    correction = accumulation / accum_micro_steps
                    for parameter in model.parameters():
                        if parameter.grad is not None:
                            parameter.grad.mul_(correction)
                gradient_norm = mp.clip_gradients(
                    model, optimizer, training_cfg.max_grad_norm
                )
                # The optimizer step represents all accumulation microbatches,
                # not merely the final one. The old final-microbatch value made
                # HAL and adaptive LR react to random hard examples.
                loss_float = accumulated_step_loss / accum_micro_steps
                answer_loss_float = (
                    accumulated_answer_nll / accumulated_answer_tokens
                    if accumulated_answer_tokens
                    else None
                )
                scaffold_loss_float = (
                    accumulated_scaffold_nll / accumulated_scaffold_tokens
                    if accumulated_scaffold_tokens
                    else None
                )
                last_ewc_loss = accumulated_ewc_loss / accum_micro_steps
                last_logit_z_loss = accumulated_logit_z_loss / accum_micro_steps
                grad_float = float(gradient_norm)
                appended_rows_before = (
                    appended_row_lr.capture() if appended_row_lr is not None else None
                )
                mp.step(optimizer)
                step_succeeded = mp.update()
                if not step_succeeded:
                    optimizer.zero_grad(set_to_none=True)
                    pcgrad.clear()
                    accum_micro_steps = 0
                    accumulated_step_loss = 0.0
                    accumulated_ewc_loss = 0.0
                    accumulated_logit_z_loss = 0.0
                    accumulated_answer_nll = 0.0
                    accumulated_answer_tokens = 0
                    accumulated_scaffold_nll = 0.0
                    accumulated_scaffold_tokens = 0
                    pending_trained_tokens = 0
                    pending_token_ids.clear()
                    pending_window_indices.clear()
                    print(
                        "[AMP] Non-finite gradients skipped; optimizer, scheduler, "
                        "token counters, and sampler cursor were not advanced.",
                        flush=True,
                    )
                    break
                if hasattr(native_model, "update_moe_balance"):
                    native_model.update_moe_balance()
                if appended_row_lr is not None:
                    appended_row_lr.apply(appended_rows_before)
                if intelligence_session is not None:
                    intelligence_session.record_optimizer_step(
                        step=global_step,
                        loss=loss_float,
                        learning_rate=float(optimizer.param_groups[0]["lr"]),
                        gradient_norm=grad_float,
                        tokens=int((yb != tokenizer.pad_token_id).sum().item()),
                    )
                    hal = get_hal_module(model)
                    if hal is not None:
                        intelligence_session.record_hal_step(
                            step=global_step,
                            hal_state=hal.state,
                        )
                scheduler.step()
                campaign_tokens_seen += pending_trained_tokens
                phase_key = continuation_phase.upper()
                continuation_token_counts[phase_key] = (
                    continuation_token_counts.get(phase_key, 0) + pending_trained_tokens
                )
                known_token_ids.update(pending_token_ids)
                if window_consumption is not None:
                    window_consumption.mark(pending_window_indices)
                    data_sampler_position += len(pending_window_indices)
                pending_trained_tokens = 0
                pending_token_ids.clear()
                pending_window_indices.clear()
                optimizer.zero_grad(set_to_none=True)
                pcgrad.clear()
                global_step += 1
                if growth_alignment is not None:
                    growth_alignment.configure_trainable_parameters(global_step)
                session_step += 1
                accum_micro_steps = 0
                accumulated_step_loss = 0.0
                accumulated_ewc_loss = 0.0
                accumulated_logit_z_loss = 0.0
                accumulated_answer_nll = 0.0
                accumulated_answer_tokens = 0
                accumulated_scaffold_nll = 0.0
                accumulated_scaffold_tokens = 0
                write_json(
                    v2_report_path("training_progress_journal.json"),
                    {
                        "schema_version": 2,
                        "updated_at": time.time(),
                        "global_step": global_step,
                        "completed_optimizer_boundary": True,
                        "accumulation_step": 0,
                        "tokens_seen": campaign_tokens_seen,
                        "weighted_training_loss": loss_float,
                        "answer_training_loss": answer_loss_float,
                        "scaffold_training_loss": scaffold_loss_float,
                        "logit_z_loss": last_logit_z_loss,
                        "phase": continuation_phase.upper(),
                        "phase_tokens_seen": continuation_token_counts.get(
                            continuation_phase.upper(), 0
                        ),
                        "token_window": dict(token_window or {}),
                        "checkpoint_path": str(ckpt_path),
                    },
                )

                avg_loss = rolling_loss / max(1, rolling_count)
                last_avg_loss = avg_loss
                loss_ema = loss_float if loss_ema is None else 0.9 * loss_ema + 0.1 * loss_float
                best_loss = min(best_loss, loss_ema) if math.isfinite(best_loss) else loss_ema
                boundary_rng_states = capture_rng_states(data_generator=data_generator)
                boundary_epoch = epoch
                if termination_request["signal"] is not None:
                    _save_interrupted_boundary(
                        int(termination_request["signal"]),
                        discarded_micro_steps=0,
                    )
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
                        answer_validation_loss = validation_result.get("answer_loss")
                        best_validation_loss = min(
                            best_validation_loss,
                            validation_loss,
                        )
                        if answer_validation_loss is not None:
                            best_answer_validation_loss = min(
                                best_answer_validation_loss,
                                float(answer_validation_loss),
                            )
                        validation_history.append(
                            {
                                "step": global_step,
                                **validation_result,
                                "best_validation_loss": best_validation_loss,
                                "best_answer_validation_loss": best_answer_validation_loss,
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
                            f"answer={answer_validation_loss} "
                            f"best={best_validation_loss:.4f} "
                            f"best_answer={best_answer_validation_loss}",
                            flush=True,
                        )
                    finally:
                        model.train(was_training)

                if (
                    durability.requires_initial_boundary
                    or global_step % durable_checkpoint_steps == 0
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
                        best_answer_validation_loss=best_answer_validation_loss,
                        validation_history=validation_history,
                        appended_row_optimizer_steps=(
                            appended_row_lr.steps_completed if appended_row_lr is not None else 0
                        ),
                        raw_window_consumption=(
                            window_consumption.state_dict()
                            if window_consumption is not None
                            else None
                        ),
                        data_sampler_state=current_data_sampler_state(),
                        data_generator=data_generator,
                        seed_contract=seed_report.to_dict(),
                        token_window=token_window,
                        growth_provenance=growth_provenance,
                    )
                    atomic_save(payload, ckpt_path, drive_dir=None)
                    _publish_training_checkpoint(payload)
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    try:
                        hal = get_hal_module(model)
                        if hal is not None:
                            publish_hal_state(hal, source="training")
                    except Exception as exc:
                        print(f"[HAL] checkpoint publish skipped: {exc}", flush=True)
                    next_checkpoint_at = time.time() + checkpoint_every_seconds

            if termination_request["signal"] is not None:
                discarded_micro_steps = accum_micro_steps
                optimizer.zero_grad(set_to_none=True)
                pcgrad.clear()
                accum_micro_steps = 0
                pending_trained_tokens = 0
                pending_token_ids.clear()
                pending_window_indices.clear()
                _save_interrupted_boundary(
                    int(termination_request["signal"]),
                    discarded_micro_steps=discarded_micro_steps,
                )

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
            f"{accum_micro_steps}/{accumulation}; "
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
        best_answer_validation_loss=best_answer_validation_loss,
        validation_history=validation_history,
        appended_row_optimizer_steps=(
            appended_row_lr.steps_completed if appended_row_lr is not None else 0
        ),
        raw_window_consumption=(
            window_consumption.state_dict() if window_consumption is not None else None
        ),
        data_sampler_state=current_data_sampler_state(),
        data_generator=data_generator,
        seed_contract=seed_report.to_dict(),
        token_window=token_window,
        growth_provenance=growth_provenance,
    )
    atomic_save(payload, ckpt_path, drive_dir=None)
    _publish_training_checkpoint(payload, final=True)
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
        "grad_accum_steps": accumulation,
        "answer_loss_weight": answer_loss_weight,
        "model_size": model_size,
        "optimizer": optimizer_report,
        "training_algorithm": {
            "optimizer": optimizer_name,
            "schedule": CANONICAL_FOUNDATION_SCHEDULE,
            "adaptive_lr_overlay": False,
            "gradient_clip_norm": training_cfg.max_grad_norm,
            "verified_process_objective": VERIFIED_PROCESS_OBJECTIVE,
            "verified_process_multiplier": training_cfg.verified_process_multiplier,
            "determinism": seed_report.to_dict(),
        },
        "append_only_row_learning": (
            appended_row_lr.report() if appended_row_lr is not None else None
        ),
        "answer_supervision_ratio": round(ds.answer_supervision_ratio, 4),
        "data_layout": ds.PACKING_LAYOUT,
        "token_utilization": round(ds.token_utilization, 4),
        "reply_token_ratio_seen": round(answer_weighted_tokens / max(1.0, total_target_tokens), 4),
        "target_tokens_seen": int(total_target_tokens),
        "verified_process_weighted_tokens": verified_process_weighted_tokens,
        "campaign_tokens_seen": campaign_tokens_seen,
        "phase_tokens_seen": continuation_token_counts.get(continuation_phase.upper(), 0),
        "token_window": dict(token_window or {}),
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
        "best_answer_validation_loss": best_answer_validation_loss,
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
        "logit_scale_control": {
            "z_loss_weight": training_cfg.logit_z_loss_weight,
            "last_optimizer_step_loss": locals().get("last_logit_z_loss", 0.0),
        },
        "multi_token_prediction": {
            "enabled": use_mtp,
            "depth": 2 if use_mtp else 0,
            "loss_weight": 0.2 if use_mtp else 0.0,
            "last_microbatch_loss": float(
                locals().get("current_mtp_loss", torch.zeros(())).detach().cpu()
            ),
        },
        "curriculum": {
            "name": curriculum,
            "expected_token_budget": max_phase_tokens,
            "sampling_basis": "immutable_source_window_share",
            "sampler_state": current_data_sampler_state(),
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
    cdr_report_path = v2_report_path("cdr_report.json")
    write_json(cdr_report_path, cdr.report())

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
        eval_summary = (
            run_compact_eval(model, tokenizer, device=device, output=True, seed=0)
            if post_session_eval
            else {
                "overall_score": 0.0,
                "results": [],
                "skipped": True,
                "reason": "bounded_training_rehearsal",
            }
        )
        if not post_session_eval:
            print("[Eval] post-session compact generation skipped for rehearsal.", flush=True)
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
        if not post_session_eval:
            intelligence_session.hooks.close()
            print("[ThirdEye] evaluation/calibration skipped for rehearsal.", flush=True)
        else:
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
            "M-07": str(cdr_report_path),
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
    if post_session_eval and isinstance(prev_eval_summary, dict):
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
                            model,
                            optimizer,
                            scheduler,
                            mp,
                            prev_ckpt,
                            device=device,
                            strict=False,
                            resume_training=True,
                            data_generator=data_generator,
                        )
                        print("[ABORT] Checkpoint restored. Stopping session.", flush=True)
                    durability.close()
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

    if post_session_eval:
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
            print(f"  Session-end validation loss : {session_end_loss:.6f}", flush=True)
        except Exception as exc:
            print(f"[build_brain] quick eval at session_end failed: {exc}", flush=True)
    else:
        print("[Eval] session-end validation skipped for rehearsal.", flush=True)
    # The frontier checkpoint has exactly one Drive destination: the shared
    # master that was restored at session start. Do not invoke legacy V2
    # artifact mirroring here; it creates duplicate multi-gigabyte brain files.
    if durability.enabled:
        durability.close()
    else:
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
    print(
        "  Durability         : "
        + ("verified outbox" if durability.enabled else "legacy checkpoint mirror"),
        flush=True,
    )
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
        "token_window": dict(token_window or {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Canonical An-Ra base trainer")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--checkpoint_path", default="anra_v4_180m.pt")
    parser.add_argument("--resume_from", default=None)
    parser.add_argument("--batch_size", type=int, default=ANRA_V4_TRAINING.batch_size)
    parser.add_argument(
        "--accumulation", type=int, default=ANRA_V4_TRAINING.grad_accum_steps
    )
    parser.add_argument("--block_size", type=int, default=ANRA_V4_MODEL.block_size)
    parser.add_argument("--max_minutes", type=int, default=ANRA_V4_TRAINING.session_minutes)
    parser.add_argument(
        "--model-size",
        choices=[CANONICAL_MODEL_PROFILE, ANRA_V4_GROWTH_MODEL_PROFILE],
        default=CANONICAL_MODEL_PROFILE,
    )
    parser.add_argument("--growth-initialization", default=None)
    parser.add_argument("--growth-manifest", default=None)
    parser.add_argument("--growth-parent-checkpoint", default=None)
    parser.add_argument(
        "--answer_loss_weight", type=float, default=ANRA_V4_TRAINING.answer_loss_weight
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
    parser.add_argument("--qk-norm", choices=["on", "off"], default=None)
    parser.add_argument("--mtp", choices=["on", "off"], default="off")
    parser.add_argument(
        "--moe", choices=["off", "upcycle-8r1s-top2"], default="off"
    )
    parser.add_argument("--curriculum", choices=sorted(CURRICULUMS), default="none")
    parser.add_argument("--seed", type=int, default=CANONICAL_TRAINING_SEED)
    parser.add_argument(
        "--attention-pattern", choices=["hybrid", "full-only"], default=None
    )
    parser.add_argument(
        "--continuation-phase",
        choices=["A", "B", "C", "D", "E"],
        default="A",
    )
    parser.add_argument(
        "--max-phase-tokens",
        type=int,
        default=None,
        help="Stop at the first complete optimizer boundary reaching this phase token count.",
    )
    parser.add_argument("--token-window-id", default=None)
    parser.add_argument("--token-window-start", type=int, default=None)
    parser.add_argument("--token-window-end", type=int, default=None)
    parser.add_argument(
        "--start_eval_examples",
        type=int,
        default=0,
        help="Run startup quick-eval before training. Default 0 skips it for faster first loss.",
    )
    parser.add_argument(
        "--post-session-eval",
        choices=["full", "none"],
        default="full",
        help="Use 'none' only for bounded execution/restart rehearsals.",
    )
    parser.add_argument(
        "--rehearsal-interrupt-after-microsteps",
        type=int,
        default=None,
        help=(
            "Rehearsal-only deterministic fault injection. It requests the normal "
            "deferred SIGTERM checkpoint path after N session microsteps."
        ),
    )
    parser.add_argument("--own_ratio", type=float, default=None)
    parser.add_argument("--identity_ratio", type=float, default=None)
    parser.add_argument("--teacher_ratio", type=float, default=None)
    parser.add_argument("--symbolic_ratio", type=float, default=None)
    parser.add_argument("--replay_ratio", type=float, default=None)
    parser.add_argument(
        "--optimizer",
        choices=[CANONICAL_FOUNDATION_OPTIMIZER],
        default=CANONICAL_FOUNDATION_OPTIMIZER,
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
        accumulation=args.accumulation,
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
        use_qk_norm=(None if args.qk_norm is None else args.qk_norm == "on"),
        attention_pattern=args.attention_pattern,
        use_mtp=args.mtp == "on",
        use_moe=args.moe != "off",
        curriculum=args.curriculum,
        seed=args.seed,
        post_session_eval=args.post_session_eval == "full",
        rehearsal_interrupt_after_microsteps=(
            args.rehearsal_interrupt_after_microsteps
        ),
        token_window_id=args.token_window_id,
        token_window_start=args.token_window_start,
        token_window_end=args.token_window_end,
        growth_initialization=args.growth_initialization,
        growth_manifest=args.growth_manifest,
        growth_parent_checkpoint=args.growth_parent_checkpoint,
    )
    print(result, flush=True)


if __name__ == "__main__":
    main()
