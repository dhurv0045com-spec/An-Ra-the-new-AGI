# NOTE: scripts/train.py is the canonical training script for local runs.
# This file handles Colab and Google Drive data ingestion.
from __future__ import annotations

# Direct script execution must bootstrap the repository before package imports.
# ruff: noqa: E402
import argparse
import importlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from anra.anra_paths import ANRA_V4_CHECKPOINT, DATASET, ROOT, ensure_dirs, get_dataset_file
from anra.startup_checks import assert_flash_sdp_ready
from runtime.training_readiness import assess_training_readiness

from training.data_ingestion import mount_google_drive_if_available, prepare_training_corpus
from training.eval_v2 import run_compact_eval
from training.v2_config import (
    ANRA_V4_MODEL,
    ANRA_V4_TRAINING,
    CANONICAL_FOUNDATION_OPTIMIZER,
    CANONICAL_MODEL_PROFILE,
    CANONICAL_TRAINING_SEED,
    resolve_model_profile,
)
from training.v2_runtime import (
    active_tokenizer_identity,
    build_model_for_profile,
    canonical_v2_checkpoint,
    load_checkpoint,
    load_or_build_v2_tokenizer,
    load_session_state,
    update_session_state,
    v2_report_path,
    write_json,
)

ensure_dirs()

_CANONICAL_DATASET = DATASET


def _valid_text_dataset(path: Path) -> bool:
    if not path.exists() or not path.is_file():
        return False
    if path.stat().st_size < 100_000:
        return False
    try:
        sample = path.read_text(encoding="utf-8", errors="replace")[:2000]
    except Exception:
        return False
    return "H:" in sample and "ANRA:" in sample


def resolve_dataset_path(explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit)
        if not path.is_absolute():
            path = (ROOT / path).resolve()
        if not _valid_text_dataset(path):
            raise FileNotFoundError(
                "Dataset invalid or too small (must be anra_training.txt with "
                f"H:/ANRA: format, >100 KB): {path}"
            )
        return path
    if _valid_text_dataset(_CANONICAL_DATASET):
        return _CANONICAL_DATASET
    drive_dataset = get_dataset_file()
    if drive_dataset != _CANONICAL_DATASET and _valid_text_dataset(drive_dataset):
        return drive_dataset
    raise FileNotFoundError(
        "\n\n[FATAL] Training dataset not found.\n"
        f"  Expected locally: {_CANONICAL_DATASET}\n"
        f"  Or on Drive:      {drive_dataset}\n"
    )


def _module_health(module_name: str) -> str:
    try:
        mod = importlib.import_module(module_name)
        fn = getattr(mod, "health_check", None)
        if callable(fn):
            result = fn()
            if isinstance(result, dict):
                return str(result.get("status", "ok"))
            return "ok"
        return "ok (no health_check)"
    except Exception as exc:
        return f"degraded ({type(exc).__name__})"


def print_system_health() -> None:
    subsystems = {
        "identity": "identity_injector",
        "ouroboros": "ouroboros_numpy",
        "symbolic_bridge": "symbolic_bridge",
        "sovereignty": "sovereignty_bridge",
        "turboquant": "turboquant",
    }
    print("\n[Unified Trainer] Subsystem health:")
    for label, mod_name in subsystems.items():
        status = _module_health(mod_name)
        icon = "OK" if "ok" in status.lower() else "WARN"
        print(f"  {icon:<4} {label:<16}: {status}")
    print()


def run_report_path(report: dict[str, object]) -> Path:
    """Keep signed worker evidence beside its unique artifact."""
    manifest = report.get("launch_manifest")
    if isinstance(manifest, dict):
        artifact = str(manifest.get("artifact_path", "")).strip()
        if artifact:
            path = Path(artifact)
            if not path.is_absolute():
                path = (ROOT / path).resolve()
            return path.with_suffix(".run.json")
    return v2_report_path("run_report")


def _write_run_report(report: dict[str, object]) -> None:
    target = run_report_path(report)
    report["run_report_path"] = str(target)
    write_json(target, report)


def stage_plan_for_mode(mode: str) -> list[str]:
    if mode in {"session", "resume"}:
        return ["base"]
    if mode == "eval":
        return ["eval"]
    if mode in {"status", "preflight"}:
        return ["status"]
    if mode in {"train", "production"}:
        return ["base", "evaluation", "sovereignty_audit", "tests"]
    raise ValueError(f"Unknown training mode: {mode}")


def _load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def checkpoint_resume_path(checkpoint_source: object) -> str | None:
    """Resolve a signed checkpoint source without treating scratch as a path."""
    value = str(checkpoint_source or "").strip()
    if not value or value.lower() == "scratch":
        return None
    return value


def resolve_campaign_inventory(
    launch_manifest: dict[str, object] | None,
    model_size: str,
    default_inventory_path: Path,
) -> dict | None:
    """Resolve tokens from the exact manifest bound to canonical V4 training."""
    if not launch_manifest or model_size != CANONICAL_MODEL_PROFILE:
        return _load_json(default_inventory_path)
    manifests = launch_manifest.get("data_manifests")
    if not isinstance(manifests, list) or not manifests:
        raise RuntimeError("Signed pilot launch manifest has no training data manifest")
    roles = launch_manifest.get("data_manifest_roles")
    if not isinstance(roles, dict):
        raise RuntimeError("Signed pilot launch manifest has no data-manifest roles")
    train_manifests = [
        str(entry) for entry in manifests if str(roles.get(str(entry), "")) == "train"
    ]
    validation_manifests = [
        str(entry)
        for entry in manifests
        if str(roles.get(str(entry), "")) == "validation"
    ]
    if len(train_manifests) != 1:
        raise RuntimeError("Signed pilot launch manifest must bind exactly one train manifest")
    if len(validation_manifests) != 1:
        raise RuntimeError(
            "Signed pilot launch manifest must bind exactly one validation manifest"
        )
    signed_train_manifest = Path(train_manifests[0])
    if not signed_train_manifest.is_absolute():
        signed_train_manifest = (ROOT / signed_train_manifest).resolve()
    inventory = _load_json(signed_train_manifest)
    if inventory is None:
        raise RuntimeError(
            f"Signed pilot training manifest is unreadable: {signed_train_manifest}"
        )
    total_tokens = int(inventory.get("total_tokens", 0))
    if total_tokens <= 0:
        raise RuntimeError(
            f"Signed pilot training manifest has no tokens: {signed_train_manifest}"
        )
    return {
        **inventory,
        "licensed_tokens": total_tokens,
        "manifest": str(signed_train_manifest),
        "validation_manifest": str(Path(validation_manifests[0]).resolve()),
    }


def launch_data_profile(launch_manifest: dict[str, object]) -> str:
    """Return the exact signed training-manifest identity for checkpoint lineage."""
    roles = {
        str(key): str(value)
        for key, value in dict(launch_manifest["data_manifest_roles"]).items()
    }
    hashes = {
        str(key): str(value)
        for key, value in dict(launch_manifest["data_manifest_hashes"]).items()
    }
    train_manifests = [path for path, role in roles.items() if role == "train"]
    if len(train_manifests) != 1:
        raise RuntimeError("Signed launch must bind exactly one training manifest")
    train_manifest = train_manifests[0]
    if train_manifest not in hashes or not hashes[train_manifest]:
        raise RuntimeError("Signed launch has no hash for its training manifest")
    return "manifest-sha256:" + hashes[train_manifest]


def _milestone_due(training_cfg: object | None = None) -> dict[str, object]:
    """Check if a milestone eval is due. Uses the active training config."""
    cfg = training_cfg if training_cfg is not None else ANRA_V4_TRAINING
    state = load_session_state()
    successful = int(state.get("successful_sessions", 0) or 0)
    entries = state.get("eval_scores", [])
    scores = [float(item.get("score", 0.0)) for item in entries if isinstance(item, dict)]
    plateau = False
    if len(scores) >= cfg.plateau_window:
        recent = scores[-cfg.plateau_window :]
        plateau = max(recent) - min(recent) <= cfg.plateau_delta
    due = successful > 0 and successful % cfg.milestone_every_sessions == 0
    return {
        "successful_sessions": successful,
        "plateau_detected": plateau,
        "milestone_due": due or plateau,
    }


def _run_innovation_if_due(training_cfg: object, session_n: int) -> None:
    """Run innovation pipeline every N sessions using MetricBus deltas as input."""
    innovation_every = getattr(training_cfg, "milestone_every_sessions", 5)
    if session_n <= 0 or session_n % innovation_every != 0:
        return
    try:
        from anra.anra_paths import OUTPUT_V2_DIR
        from engine.metric_bus import get_metric_bus
        from innovation.action_queue import queue_actions
        from innovation.gap_scanner import scan
        from innovation.hypothesis import gap_to_hypothesis
        from innovation.scoreboard import score_hypothesis, write_report

        print("[Innovation] Running gap scan from MetricBus deltas...", flush=True)
        mbus = get_metric_bus()
        deltas = getattr(mbus, "_last_deltas", {})

        gaps = scan(deltas)
        hypotheses = [gap_to_hypothesis(gap) for gap in gaps]
        scores = {hyp.hyp_id: score_hypothesis(hyp) for hyp in hypotheses}
        approved = queue_actions(hypotheses, scores)

        report_path = OUTPUT_V2_DIR / f"innovation_{int(time.time())}.json"
        write_report(list(scores.values()), report_path)
        print(
            f"[Innovation] {len(gaps)} gaps, {len(approved)} queued. Report: {report_path}",
            flush=True,
        )
    except Exception as exc:
        print(f"[Innovation] Skipped (error): {exc}", flush=True)


def _start_supervisor(args: object) -> object | None:
    try:
        from agents.supervisor import SupervisorAgent

        _supervisor = SupervisorAgent(
            model_size=getattr(args, "model_size", CANONICAL_MODEL_PROFILE)
        )
        _supervisor.start_session()
        _session_run_id = _supervisor._bus.run_id
        print(f"[Unified Trainer] Session tracked — run_id: {_session_run_id}")
        return _supervisor
    except Exception as _sup_err:
        print(f"[Unified Trainer] Supervisor init failed: {_sup_err}")
        return None


def _end_supervisor(_supervisor: object | None) -> None:
    if _supervisor is None:
        return
    if getattr(_supervisor, "_unified_trainer_closed", False):
        return
    _supervisor._unified_trainer_closed = True
    try:
        _summary = _supervisor.end_session()
        _supervisor.push_scorecard_to_drive(_summary)
        print(f"[Unified Trainer] Scorecard saved — run_id: {_summary.run_id}")
    except Exception as _sup_err:
        print(f"[Unified Trainer] Supervisor end failed: {_sup_err}")


def _write_daily_curriculum() -> dict[str, object]:
    eval_summary = _load_json(v2_report_path("eval_summary")) or {}
    hard_blob = _load_json(v2_report_path("hard_examples")) or {}
    mix_report = _load_json(v2_report_path("mix_report")) or {}
    recommendations: list[str] = []
    category_scores = (
        eval_summary.get("category_scores", {}) if isinstance(eval_summary, dict) else {}
    )
    if float(category_scores.get("identity", 0.0) or 0.0) < 0.7:
        recommendations.append(
            "Increase identity-heavy turns next session to keep An-Ra's voice anchored."
        )
    if float(category_scores.get("symbolic", 0.0) or 0.0) < 0.6:
        recommendations.append("Increase verified symbolic/code samples next session.")
    if float(category_scores.get("reasoning", 0.0) or 0.0) < 0.6:
        recommendations.append(
            "Feed more teacher-style reasoning traces through the teacher bucket."
        )
    if not recommendations:
        recommendations.append("Keep the current training mix; no category is lagging badly.")
    report = {
        "generated_at": time.time(),
        "eval_summary_path": str(v2_report_path("eval_summary")),
        "hard_examples_path": str(v2_report_path("hard_examples")),
        "mix_report_path": str(v2_report_path("mix_report")),
        "top_hard_examples": (
            hard_blob.get("examples", [])[:6] if isinstance(hard_blob, dict) else []
        ),
        "category_scores": category_scores,
        "recommendations": recommendations,
        "mix_report": mix_report if isinstance(mix_report, dict) else {},
    }
    write_json(v2_report_path("curriculum"), report)
    return report


def run_cmd(cmd: list[str], *, cwd: Path | None = None) -> int:
    print("\n[Unified Trainer] Running:", " ".join(cmd), flush=True)
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd or ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
    proc.wait()
    return int(proc.returncode)


def _restore_core_artifacts() -> None:
    # V4 owns one checkpoint lineage. The caller's checkpoint path controls
    # resume; no independent V2 identity or ouroboros weights are restored.
    load_or_build_v2_tokenizer(dataset_path=resolve_dataset_path(None))


def _run_eval_only() -> dict[str, object]:
    tokenizer = load_or_build_v2_tokenizer(dataset_path=resolve_dataset_path(None))
    model = build_model_for_profile(
        CANONICAL_MODEL_PROFILE, vocab_size=tokenizer.vocab_size
    )
    if hasattr(model, "disable_kv_cache"):
        model.disable_kv_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    checkpoint = ANRA_V4_CHECKPOINT
    load_checkpoint(model, None, None, None, checkpoint, device=device, strict=False)
    return run_compact_eval(model, tokenizer, device=device, output=True, golden=True)


def main() -> None:
    assert_flash_sdp_ready("training.train_unified")
    ap = argparse.ArgumentParser(description="An-Ra unified training dispatcher")
    ap.add_argument(
        "--mode",
        default="session",
        choices=["session", "train", "resume", "production", "eval", "status", "preflight"],
    )
    ap.add_argument("--data_path", default=None)
    ap.add_argument("--data_files", nargs="*", default=[])
    ap.add_argument("--prepare_data", default="auto", choices=["auto", "always", "never"])
    ap.add_argument("--no_drive_scan", action="store_true")
    ap.add_argument("--max_source_mb", type=int, default=64)
    ap.add_argument("--checkpoint_path", default="anra_v4_180m.pt")
    ap.add_argument("--batch_size", type=int, default=ANRA_V4_TRAINING.batch_size)
    ap.add_argument("--block_size", type=int, default=ANRA_V4_MODEL.block_size)
    ap.add_argument(
        "--answer_loss_weight", type=float, default=ANRA_V4_TRAINING.answer_loss_weight
    )
    ap.add_argument(
        "--optimizer",
        choices=["auto", "adamw", "adam8bit", "adafactor", "muon", "scale", "galore", "qgalore"],
        default=CANONICAL_FOUNDATION_OPTIMIZER,
    )
    ap.add_argument(
        "--session_minutes",
        "--session-minutes",
        type=int,
        default=ANRA_V4_TRAINING.session_minutes,
    )
    ap.add_argument(
        "--model-size",
        choices=[CANONICAL_MODEL_PROFILE],
        default=CANONICAL_MODEL_PROFILE,
        help="The sole active canonical V4 model profile.",
    )
    ap.add_argument(
        "--campaign",
        choices=["frontier_full", "stage_a", "stage_b", "stage_c", "stage_d", "stage_e"],
        default=None,
    )
    ap.add_argument("--identity_minutes", type=int, default=12)
    ap.add_argument("--ouroboros_minutes", type=int, default=10)
    ap.add_argument("--max_examples", type=int, default=None)
    ap.add_argument("--launch-manifest", default=None)
    ap.add_argument(
        "--post-session-eval",
        choices=["full", "none"],
        default="full",
        help="Use 'none' only for bounded execution/restart rehearsals.",
    )
    ap.add_argument("--seed", type=int, default=CANONICAL_TRAINING_SEED)
    ap.add_argument(
        "--training-objective",
        choices=["base", "causal-extension"],
        default="base",
    )
    ap.add_argument(
        "--runtime-class",
        choices=["t4_v4_session"],
        default="t4_v4_session",
    )
    args = ap.parse_args()
    launch_manifest: dict[str, object] | None = None
    manifest_resume_from: str | None = None
    if args.launch_manifest:
        from training.launch_manifest import load_and_validate_manifest

        launch_manifest = load_and_validate_manifest(args.launch_manifest)
        os.environ["ANRA_TOKENIZER_PATH"] = str(launch_manifest["tokenizer_path"])
        args.model_size = str(launch_manifest["model_profile"])
        args.optimizer = str(launch_manifest["optimizer"])
        args.batch_size = int(launch_manifest["batch_size"])
        args.seed = int(launch_manifest["seed"])
        os.environ["ANRA_DATA_PROFILE"] = launch_data_profile(launch_manifest)
        checkpoint_source = str(launch_manifest["checkpoint_source"])
        artifact_path = str(launch_manifest.get("artifact_path", "")).strip()
        manifest_resume_from = checkpoint_resume_path(checkpoint_source)
        if artifact_path:
            args.checkpoint_path = artifact_path
            report_root = Path(artifact_path)
            if not report_root.is_absolute():
                report_root = (ROOT / report_root).resolve()
            os.environ["ANRA_RUN_REPORT_DIR"] = str(report_root.with_suffix(".reports"))
        stage = str(launch_manifest["stage"])
        if stage in {
            "frontier_full",
            "stage_a",
            "stage_b",
            "stage_c",
            "stage_d",
            "stage_e",
        }:
            args.campaign = stage
    model_cfg, training_cfg = resolve_model_profile(args.model_size)
    # The child phase trainer inherits this at interpreter startup, making
    # hash-based Python containers part of the signed seed contract.
    os.environ["PYTHONHASHSEED"] = str(args.seed)

    mount_google_drive_if_available()

    data_ingestion_report: dict[str, object] | None = None
    if args.mode not in {"status", "preflight", "eval"} and args.prepare_data != "never":
        should_prepare = (
            args.prepare_data == "always" or bool(args.data_files) or args.data_path is None
        )
        if should_prepare:
            report = prepare_training_corpus(
                explicit_sources=args.data_files,
                include_drive=not args.no_drive_scan,
                max_source_mb=args.max_source_mb,
                mount_drive=False,
            )
            data_ingestion_report = report.to_dict()
            if args.data_path is None:
                args.data_path = report.output_dataset
            print(
                "[Unified Trainer] data prepared: "
                f"{report.total_examples} examples, {report.teacher_records} teacher records, "
                f"{report.output_bytes / 1024**2:.2f} MB",
                flush=True,
            )

    _restore_core_artifacts()
    stage_plan = stage_plan_for_mode(args.mode)

    if args.mode == "preflight":
        from training.preflight import run_preflight

        decision = run_preflight(args.model_size, runtime_class=args.runtime_class)
        print(json.dumps(decision.to_dict(), indent=2, sort_keys=True))
        raise SystemExit(0 if decision.allowed else 2)

    if args.mode == "status":
        readiness = assess_training_readiness()
        print_system_health()
        print(f"[Unified Trainer] model_size={args.model_size}")
        print(f"[Unified Trainer] dataset={resolve_dataset_path(args.data_path)}")
        print(f"[Unified Trainer] checkpoint={canonical_v2_checkpoint('brain')}")
        print(f"[Unified Trainer] tokenizer={ROOT / 'tokenizer' / 'tokenizer_v4_32k.json'}")
        print(f"[Unified Trainer] milestone={_milestone_due(training_cfg)}")
        print(
            "[Unified Trainer] readiness="
            f"{readiness.score}/{readiness.out_of} "
            f"session={readiness.ready_for_session} milestone={readiness.ready_for_milestone}"
        )
        for blocker in readiness.blockers:
            print(f"  BLOCKER {blocker}")
        for warning in readiness.warnings[:8]:
            print(f"  WARN    {warning}")
        return

    if args.mode == "eval":
        print(json.dumps(_run_eval_only(), indent=2))
        return

    dataset = resolve_dataset_path(args.data_path)
    readiness = assess_training_readiness(dataset)
    if not readiness.ready_for_session:
        print("[Unified Trainer] readiness blockers:", flush=True)
        for blocker in readiness.blockers:
            print(f"  - {blocker}", flush=True)
        raise SystemExit(2)
    if readiness.warnings:
        print(
            f"[Unified Trainer] readiness {readiness.score}/{readiness.out_of}; "
            f"{len(readiness.warnings)} warning(s)",
            flush=True,
        )
        for warning in readiness.warnings[:6]:
            print(f"  WARN {warning}", flush=True)
    print(f"[Unified Trainer] dataset={dataset}", flush=True)
    print_system_health()

    run_report: dict[str, object] = {
        "started_at": time.time(),
        "mode": args.mode,
        "stage_plan": stage_plan,
        "readiness": readiness.to_dict(),
        "dataset": str(dataset),
        "model_line": "v4",
        "model_size": args.model_size,
        "seed": args.seed,
        "data_ingestion": data_ingestion_report,
        "launch_manifest": launch_manifest,
        "stages": {},
    }
    _supervisor = _start_supervisor(args)

    base_cmd = [
        sys.executable,
        str(ROOT / "scripts" / "build_brain.py"),
        "--data_path",
        str(dataset),
        "--checkpoint_path",
        args.checkpoint_path,
        "--batch_size",
        str(args.batch_size),
        "--block_size",
        str(model_cfg.block_size),
        "--answer_loss_weight",
        str(args.answer_loss_weight),
        "--optimizer",
        args.optimizer,
        "--max_minutes",
        str(args.session_minutes),
        "--model-size",
        args.model_size,
        "--seed",
        str(args.seed),
        "--training-objective",
        args.training_objective,
        "--post-session-eval",
        args.post_session_eval,
    ]
    if args.max_examples is not None:
        base_cmd.extend(["--max_examples", str(args.max_examples)])
    if manifest_resume_from:
        base_cmd.extend(["--resume_from", manifest_resume_from])
    if launch_manifest and args.model_size == CANONICAL_MODEL_PROFILE:
        signed_data = [str(value) for value in launch_manifest["data_manifests"]]
        roles = {
            str(key): str(value)
            for key, value in dict(launch_manifest["data_manifest_roles"]).items()
        }
        training_manifests = [value for value in signed_data if roles.get(value) == "train"]
        validation_manifests = [
            value for value in signed_data if roles.get(value) == "validation"
        ]
        if len(training_manifests) != 1 or len(validation_manifests) != 1:
            raise RuntimeError(
                "Pilot launch manifest must bind exactly one train and one "
                "validation shard manifest"
            )
        base_cmd.extend(
            [
                "--training-layout",
                "raw_causal_shards_v1",
                "--token-shard-manifest",
                training_manifests[0],
                "--validation-shard-manifest",
                validation_manifests[0],
            ]
        )
        base_cmd.extend(
            ["--max-phase-tokens", str(int(launch_manifest["expected_tokens"]))]
        )
        pilot_axes = dict(launch_manifest.get("pilot_axes", {}))
        qk_norm = str(pilot_axes.get("qk_norm", "on"))
        attention_pattern = str(pilot_axes.get("attention", "hybrid"))
        mtp = str(pilot_axes.get("mtp", "off"))
        moe = str(pilot_axes.get("moe", "off"))
        curriculum = str(pilot_axes.get("curriculum", "none"))
        if qk_norm not in {"on", "off"}:
            raise RuntimeError(f"Unsupported signed qk_norm pilot axis: {qk_norm}")
        if attention_pattern not in {"hybrid", "full-only"}:
            raise RuntimeError(
                f"Unsupported signed attention pilot axis: {attention_pattern}"
            )
        if mtp not in {"on", "off"}:
            raise RuntimeError(f"Unsupported signed mtp pilot axis: {mtp}")
        if moe not in {"off", "upcycle-8r1s-top2"}:
            raise RuntimeError(f"Unsupported signed moe pilot axis: {moe}")
        if curriculum not in {
            "none",
            "code-before-prose",
            "math-density-ramp",
            "identity-mix-late",
        }:
            raise RuntimeError(f"Unsupported signed curriculum pilot axis: {curriculum}")
        base_cmd.extend(["--qk-norm", qk_norm])
        base_cmd.extend(["--attention-pattern", attention_pattern])
        base_cmd.extend(["--mtp", mtp])
        base_cmd.extend(["--moe", moe])
        base_cmd.extend(["--curriculum", curriculum])

    if args.campaign:
        from math import exp

        from anra.anra_paths import (
            OUTPUT_V2_DIR,
            TOKEN_INVENTORY_MANIFEST,
            TRAJECTORY_STORE,
        )

        from training.stages import CampaignConfig, StagedTrainingCampaign

        inventory = resolve_campaign_inventory(
            launch_manifest,
            args.model_size,
            TOKEN_INVENTORY_MANIFEST,
        )
        if inventory is None or int(inventory.get("licensed_tokens", 0)) <= 0:
            raise RuntimeError(
                "Campaign training requires a published offline licensed-token inventory. "
                "Run scripts/download_training_data.py --bucket base --publish-token-shards."
            )

        campaign = StagedTrainingCampaign(
            CampaignConfig(
                model_size=args.model_size,
                data_path=str(dataset),
                output_dir=str(OUTPUT_V2_DIR / "campaigns"),
            )
        )
        names = (
            [
                "foundation",
                "owner_adaptation",
                "agency",
                "verified_reasoning",
                "verifier_replay",
            ]
            if args.campaign == "frontier_full"
            else [args.campaign]
        )
        stage_validation_offsets: dict[str, int] = {}

        def validation_history() -> list[dict[str, object]]:
            payload = _load_json(v2_report_path("validation_history")) or {}
            history = payload.get("history", [])
            return [dict(row) for row in history if isinstance(row, dict)]

        def execute_stage(config: object) -> tuple[int, str]:
            stage_validation_offsets[str(config.stage.value)] = len(validation_history())
            command = list(base_cmd)
            command.extend(
                [
                    "--training-layout",
                    config.training_layout,
                    "--continuation-phase",
                    config.continuation_phase,
                ]
            )
            if config.training_layout == "raw_causal_shards_v1":
                manifest = str(inventory.get("manifest", ""))
                if not manifest:
                    raise RuntimeError("Token inventory does not name its immutable manifest")
                command.extend(["--token-shard-manifest", manifest])
                validation_manifest = str(inventory.get("validation_manifest", ""))
                if not validation_manifest:
                    raise RuntimeError("Token inventory does not name its validation manifest")
                command.extend(["--validation-shard-manifest", validation_manifest])
            else:
                command.extend(["--own_ratio", str(config.owner_ratio)])
            previous_phase = os.environ.get("ANRA_CONTINUATION_PHASE")
            os.environ["ANRA_CONTINUATION_PHASE"] = config.continuation_phase
            try:
                rc = run_cmd(command)
            finally:
                if previous_phase is None:
                    os.environ.pop("ANRA_CONTINUATION_PHASE", None)
                else:
                    os.environ["ANRA_CONTINUATION_PHASE"] = previous_phase
            return rc, str(canonical_v2_checkpoint("brain"))

        def load_stage_metrics(_config: object) -> dict[str, object]:
            eval_report = _load_json(v2_report_path("ibs_latest")) or {}
            compact = _load_json(v2_report_path("eval_summary")) or {}
            train_metrics = _load_json(v2_report_path("metrics")) or {}
            trajectory_count = 0
            if TRAJECTORY_STORE.exists():
                trajectory_count = sum(
                    1
                    for line in TRAJECTORY_STORE.read_text(encoding="utf-8").splitlines()
                    if line.strip() and '"verified": true' in line.lower()
                )
            loss = float(train_metrics.get("last_avg_loss", float("inf")))
            history = validation_history()
            offset = stage_validation_offsets.get(str(_config.stage.value), len(history))
            new_validation = history[offset:]
            validation_baseline = new_validation[0] if new_validation else {}
            validation_candidate = new_validation[-1] if len(new_validation) >= 2 else {}
            tokenizer_identity = active_tokenizer_identity()
            return {
                "perplexity": exp(min(loss, 20.0)),
                "numerically_stable": bool(loss < float("inf")),
                "training_tokens": int(train_metrics.get("phase_tokens_seen", 0)),
                "tokenizer_schema_valid": bool(
                    tokenizer_identity.get("available") is True
                    and int(tokenizer_identity.get("schema_version", 0)) >= 3
                ),
                "civ_similarity": float(compact.get("civ_similarity", 0.0)),
                "coherence_rate": float(compact.get("coherence_rate", 0.0)),
                "format_compliance": float(
                    (compact.get("category_scores", {}) or {}).get("format", 0.0)
                ),
                "validation_baseline": validation_baseline,
                "validation_candidate": validation_candidate,
                "ibs": eval_report,
                "verified_trajectories": trajectory_count,
                "star_verification_rate": float(
                    (_load_json(v2_report_path("star_report")) or {}).get("verification_rate", 0.0)
                ),
                "truth_checking_coverage": float(
                    (_load_json(v2_report_path("rlvr_report")) or {}).get(
                        "truth_checking_coverage", 0.0
                    )
                ),
            }

        for stage_name in names:
            result = campaign.run_stage(
                stage_name,
                execute=execute_stage,
                load_metrics=load_stage_metrics,
            )
            print(json.dumps(result.__dict__, indent=2, default=str))
            if not result.passed_gate:
                raise SystemExit(4)
        return

    run_base_first = "base" in stage_plan
    mode = "session" if args.mode == "resume" else args.mode
    if mode == "session":
        rc = run_cmd(base_cmd)
        run_report["stages"] = {"base": {"exit_code": rc}}
        if rc != 0:
            run_report["ended_at"] = time.time()
            _write_run_report(run_report)
            _end_supervisor(_supervisor)
            raise SystemExit(rc)
        eval_summary = _load_json(v2_report_path("eval_summary")) or {}
        curriculum = _write_daily_curriculum()
        state = update_session_state(
            eval_score=float(eval_summary.get("overall_score", 0.0) or 0.0)
        )
        _run_innovation_if_due(training_cfg, int(state.get("successful_sessions", 0) or 0))
        run_report["post_session"] = {
            "eval_summary_path": str(v2_report_path("eval_summary")),
            "hard_examples_path": str(v2_report_path("hard_examples")),
            "curriculum_path": str(v2_report_path("curriculum")),
            "curriculum_recommendations": curriculum.get("recommendations", []),
            "session_state": state,
            "milestone": _milestone_due(training_cfg),
        }
        run_report["ended_at"] = time.time()
        _write_run_report(run_report)
        _end_supervisor(_supervisor)
        return

    if run_base_first:
        rc = run_cmd(base_cmd)
        run_report["stages"]["base"] = {"exit_code": rc}
        if rc != 0:
            run_report["ended_at"] = time.time()
            _write_run_report(run_report)
            _end_supervisor(_supervisor)
            raise SystemExit(rc)

    # V4 is trained as one continuous checkpoint. Identity, verified reasoning,
    # and self-improvement examples are curriculum/data phases, not separately
    # fine-tuned model files.
    rc = run_cmd([sys.executable, str(ROOT / "scripts" / "run_sovereignty_audit.py")])
    run_report["stages"]["sovereignty_audit"] = {"exit_code": rc}
    if rc != 0:
        run_report["ended_at"] = time.time()
        _write_run_report(run_report)
        _end_supervisor(_supervisor)
        raise SystemExit(rc)

    rc = run_cmd(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_v2_stack.py",
            "-q",
            "--tb=short",
            "--no-header",
        ]
    )
    run_report["stages"]["tests"] = {"exit_code": rc}
    if rc != 0:
        run_report["ended_at"] = time.time()
        _write_run_report(run_report)
        _end_supervisor(_supervisor)
        raise SystemExit(rc)

    eval_summary = _load_json(v2_report_path("eval_summary")) or {}
    state = update_session_state(eval_score=float(eval_summary.get("overall_score", 0.0) or 0.0))
    _run_innovation_if_due(training_cfg, int(state.get("successful_sessions", 0) or 0))
    run_report["post_session"] = {
        "eval_summary_path": str(v2_report_path("eval_summary")),
        "improvement_report_path": str(v2_report_path("improvement_report")),
        "audit_report_path": str(v2_report_path("audit_report")),
        "session_state": state,
        "milestone": _milestone_due(training_cfg),
    }
    run_report["ended_at"] = time.time()
    _write_run_report(run_report)
    _end_supervisor(_supervisor)


class UnifiedTrainer:
    def __init__(
        self,
        data_path: str | None = None,
        checkpoint_path: str = "anra_v4_180m.pt",
        batch_size: int = ANRA_V4_TRAINING.batch_size,
        block_size: int = ANRA_V4_MODEL.block_size,
        answer_loss_weight: float = ANRA_V4_TRAINING.answer_loss_weight,
        session_minutes: int = ANRA_V4_TRAINING.session_minutes,
        identity_minutes: int = 12,
        ouroboros_minutes: int = 10,
        model_size: str = CANONICAL_MODEL_PROFILE,
        optimizer: str = CANONICAL_FOUNDATION_OPTIMIZER,
    ) -> None:
        self.data_path = data_path
        self.checkpoint_path = checkpoint_path
        self.batch_size = batch_size
        self.block_size = block_size
        self.answer_loss_weight = answer_loss_weight
        self.session_minutes = session_minutes
        self.identity_minutes = identity_minutes
        self.ouroboros_minutes = ouroboros_minutes
        self.model_size = model_size
        self.optimizer = optimizer
        self._dataset: Path | None = None

    def resolve_dataset(self) -> Path:
        if self._dataset is None:
            self._dataset = resolve_dataset_path(self.data_path)
        return self._dataset

    def health_check(self) -> None:
        print_system_health()

    def train(self, mode: str = "session", **kwargs: object) -> int:
        cmd = [
            sys.executable,
            "-m",
            "training.train_unified",
            "--mode",
            mode,
            "--checkpoint_path",
            self.checkpoint_path,
            "--batch_size",
            str(self.batch_size),
            "--block_size",
            str(self.block_size),
            "--answer_loss_weight",
            str(self.answer_loss_weight),
            "--optimizer",
            self.optimizer,
            "--session_minutes",
            str(self.session_minutes),
            "--model-size",
            self.model_size,
            "--identity_minutes",
            str(self.identity_minutes),
            "--ouroboros_minutes",
            str(self.ouroboros_minutes),
        ]
        if self.data_path:
            cmd.extend(["--data_path", self.data_path])
        if kwargs.get("data_files"):
            cmd.append("--data_files")
            cmd.extend(str(path) for path in kwargs["data_files"])
        if kwargs.get("prepare_data") is not None:
            cmd.extend(["--prepare_data", str(kwargs["prepare_data"])])
        if kwargs.get("no_drive_scan"):
            cmd.append("--no_drive_scan")
        if kwargs.get("max_source_mb") is not None:
            cmd.extend(["--max_source_mb", str(kwargs["max_source_mb"])])
        if kwargs.get("max_examples") is not None:
            cmd.extend(["--max_examples", str(kwargs["max_examples"])])
        return run_cmd(cmd)

    def status(self) -> None:
        print_system_health()
        try:
            print(f"dataset: {self.resolve_dataset()}")
        except FileNotFoundError as exc:
            print(f"dataset: {exc}")

    def run_session(self, minutes: int | None = None) -> int:
        if minutes is not None:
            self.session_minutes = minutes
        return self.train(mode="session")


AnRaTrainer = UnifiedTrainer


if __name__ == "__main__":
    main()
