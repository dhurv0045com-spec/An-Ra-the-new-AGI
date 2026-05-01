from __future__ import annotations

import argparse
import importlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from anra_paths import DATASET, DATASET_LEGACY, ROOT, TRAINING_DATA_DIR, ensure_dirs, inject_all_paths
from training.eval_v2 import run_compact_eval
from training.v2_config import V2_MODEL, V2_TRAINING
from training.v2_runtime import (
    build_v2_model,
    canonical_v2_checkpoint,
    load_checkpoint,
    load_or_build_v2_tokenizer,
    load_session_state,
    restore_v2_artifact,
    update_session_state,
    v2_report_path,
    write_json,
)

inject_all_paths()
ensure_dirs()

_CANONICAL_DATASET = DATASET
_LEGACY_DATASET = DATASET_LEGACY


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
                f"Dataset invalid or too small (must be anra_dataset_v6_1.txt with H:/ANRA: format, >100 KB): {path}"
            )
        return path
    if _valid_text_dataset(_CANONICAL_DATASET):
        return _CANONICAL_DATASET
    if _valid_text_dataset(_LEGACY_DATASET):
        print(
            f"[Unified Trainer][WARN] Using legacy dataset fallback: {_LEGACY_DATASET}",
            flush=True,
        )
        TRAINING_DATA_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(_LEGACY_DATASET, _CANONICAL_DATASET)
        print(f"[Unified Trainer] dataset restored from Drive: {_CANONICAL_DATASET}", flush=True)
        return _CANONICAL_DATASET
    raise FileNotFoundError(
        "\n\n[FATAL] Training dataset not found.\n"
        f"  Expected locally: {_CANONICAL_DATASET}\n"
        f"  Or on Drive:      {_LEGACY_DATASET}\n"
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
        "ghost_memory": "ghost_memory",
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


def _write_run_report(report: dict) -> None:
    write_json(v2_report_path("run_report"), report)


def _load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _milestone_due() -> dict[str, object]:
    state = load_session_state()
    successful = int(state.get("successful_sessions", 0) or 0)
    entries = state.get("eval_scores", [])
    scores = [float(item.get("score", 0.0)) for item in entries if isinstance(item, dict)]
    plateau = False
    if len(scores) >= V2_TRAINING.plateau_window:
        recent = scores[-V2_TRAINING.plateau_window :]
        plateau = max(recent) - min(recent) <= V2_TRAINING.plateau_delta
    due = successful > 0 and successful % V2_TRAINING.milestone_every_sessions == 0
    return {
        "successful_sessions": successful,
        "plateau_detected": plateau,
        "milestone_due": due or plateau,
    }


def _write_daily_curriculum() -> dict[str, object]:
    eval_summary = _load_json(v2_report_path("eval_summary")) or {}
    hard_blob = _load_json(v2_report_path("hard_examples")) or {}
    mix_report = _load_json(v2_report_path("mix_report")) or {}
    recommendations: list[str] = []
    category_scores = eval_summary.get("category_scores", {}) if isinstance(eval_summary, dict) else {}
    if float(category_scores.get("identity", 0.0) or 0.0) < 0.7:
        recommendations.append("Increase identity-heavy turns next session to keep An-Ra's voice anchored.")
    if float(category_scores.get("symbolic", 0.0) or 0.0) < 0.6:
        recommendations.append("Increase verified symbolic/code samples next session.")
    if float(category_scores.get("reasoning", 0.0) or 0.0) < 0.6:
        recommendations.append("Feed more teacher-style reasoning traces through the teacher bucket.")
    if not recommendations:
        recommendations.append("Keep the current training mix; no category is lagging badly.")
    report = {
        "generated_at": time.time(),
        "eval_summary_path": str(v2_report_path("eval_summary")),
        "hard_examples_path": str(v2_report_path("hard_examples")),
        "mix_report_path": str(v2_report_path("mix_report")),
        "top_hard_examples": (hard_blob.get("examples", [])[:6] if isinstance(hard_blob, dict) else []),
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
    restore_v2_artifact("brain")
    restore_v2_artifact("identity")
    restore_v2_artifact("ouroboros")
    restore_v2_artifact("tokenizer")


def _run_eval_only() -> dict[str, object]:
    tokenizer = load_or_build_v2_tokenizer(dataset_path=resolve_dataset_path(None))
    model = build_v2_model(vocab_size=tokenizer.vocab_size, block_size=V2_MODEL.block_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    checkpoint = canonical_v2_checkpoint("ouroboros")
    if not checkpoint.exists():
        checkpoint = canonical_v2_checkpoint("identity")
    if not checkpoint.exists():
        checkpoint = canonical_v2_checkpoint("brain")
    load_checkpoint(model, None, None, None, checkpoint, device=device, strict=False)
    return run_compact_eval(model, tokenizer, device=device, output=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="An-Ra unified training dispatcher")
    ap.add_argument("--mode", default="session", choices=["session", "train", "resume", "eval", "status"])
    ap.add_argument("--data_path", default=None)
    ap.add_argument("--checkpoint_path", default=str(canonical_v2_checkpoint("brain").name))
    ap.add_argument("--batch_size", type=int, default=V2_TRAINING.batch_size)
    ap.add_argument("--block_size", type=int, default=V2_MODEL.block_size)
    ap.add_argument("--answer_loss_weight", type=float, default=V2_TRAINING.answer_loss_weight)
    ap.add_argument("--session_minutes", type=int, default=V2_TRAINING.session_minutes)
    ap.add_argument("--identity_minutes", type=int, default=12)
    ap.add_argument("--ouroboros_minutes", type=int, default=10)
    ap.add_argument("--max_examples", type=int, default=None)
    args = ap.parse_args()

    _restore_core_artifacts()

    if args.mode == "status":
        print_system_health()
        print(f"[Unified Trainer] dataset={resolve_dataset_path(args.data_path)}")
        print(f"[Unified Trainer] brain_ckpt={canonical_v2_checkpoint('brain')}")
        print(f"[Unified Trainer] identity_ckpt={canonical_v2_checkpoint('identity')}")
        print(f"[Unified Trainer] ouroboros_ckpt={canonical_v2_checkpoint('ouroboros')}")
        print(f"[Unified Trainer] tokenizer={ROOT / 'tokenizer' / 'tokenizer_v2.json'}")
        print(f"[Unified Trainer] milestone={_milestone_due()}")
        return

    if args.mode == "eval":
        print(json.dumps(_run_eval_only(), indent=2))
        return

    dataset = resolve_dataset_path(args.data_path)
    print(f"[Unified Trainer] dataset={dataset}", flush=True)
    print_system_health()

    run_report: dict[str, object] = {
        "started_at": time.time(),
        "mode": args.mode,
        "dataset": str(dataset),
        "model_line": "v2",
        "stages": {},
    }

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
        str(args.block_size),
        "--answer_loss_weight",
        str(args.answer_loss_weight),
        "--max_minutes",
        str(args.session_minutes),
    ]
    if args.max_examples is not None:
        base_cmd.extend(["--max_examples", str(args.max_examples)])

    mode = "session" if args.mode == "resume" else args.mode
    if mode == "session":
        rc = run_cmd(base_cmd)
        run_report["stages"] = {"base": {"exit_code": rc}}
        if rc != 0:
            run_report["ended_at"] = time.time()
            _write_run_report(run_report)
            raise SystemExit(rc)
        eval_summary = _load_json(v2_report_path("eval_summary")) or {}
        curriculum = _write_daily_curriculum()
        state = update_session_state(eval_score=float(eval_summary.get("overall_score", 0.0) or 0.0))
        run_report["post_session"] = {
            "eval_summary_path": str(v2_report_path("eval_summary")),
            "hard_examples_path": str(v2_report_path("hard_examples")),
            "curriculum_path": str(v2_report_path("curriculum")),
            "curriculum_recommendations": curriculum.get("recommendations", []),
            "session_state": state,
            "milestone": _milestone_due(),
        }
        run_report["ended_at"] = time.time()
        _write_run_report(run_report)
        return

    # Auto-merge identity files before identity fine-tune
    merge_script = ROOT / "scripts" / "merge_identity.py"
    if merge_script.exists():
        print("[Unified Trainer] Running merge_identity.py ...", flush=True)
        run_cmd([sys.executable, str(merge_script)])
    else:
        print("[Unified Trainer] WARN: merge_identity.py not found — skipping", flush=True)

    session_timeout_minutes = max(1, args.session_minutes - V2_TRAINING.unified_trainer_overhead_minutes)
    print(f"[Unified Trainer] Calculated finetuning duration: {session_timeout_minutes} minutes", flush=True)

    identity_cmd = [
        sys.executable,
        "-m",
        "training.finetune_anra",
        "--data_path",
        str(dataset),
        "--max_minutes",
        str(session_timeout_minutes),
    ]
    if args.max_examples is not None:
        identity_cmd.extend(["--max_examples", str(args.max_examples)])
    rc = run_cmd(identity_cmd)
    run_report["stages"]["identity"] = {"exit_code": rc}
    if rc != 0:
        run_report["ended_at"] = time.time()
        _write_run_report(run_report)
        raise SystemExit(rc)

    ouro_cmd = [
        sys.executable,
        str(ROOT / "scripts" / "train_ouroboros.py"),
        "--data_path",
        str(dataset),
        "--max_minutes",
        str(args.ouroboros_minutes),
    ]
    if args.max_examples is not None:
        ouro_cmd.extend(["--max_examples", str(args.max_examples)])
    rc = run_cmd(ouro_cmd)
    run_report["stages"]["ouroboros"] = {"exit_code": rc}
    if rc != 0:
        run_report["ended_at"] = time.time()
        _write_run_report(run_report)
        raise SystemExit(rc)

    rc = run_cmd([sys.executable, str(ROOT / "scripts" / "run_self_improvement.py")])
    run_report["stages"]["self_improvement"] = {"exit_code": rc}
    if rc != 0:
        run_report["ended_at"] = time.time()
        _write_run_report(run_report)
        raise SystemExit(rc)

    rc = run_cmd([sys.executable, str(ROOT / "scripts" / "run_sovereignty_audit.py")])
    run_report["stages"]["sovereignty_audit"] = {"exit_code": rc}
    if rc != 0:
        run_report["ended_at"] = time.time()
        _write_run_report(run_report)
        raise SystemExit(rc)

    rc = run_cmd([sys.executable, "-m", "pytest", "tests/test_v2_stack.py", "-q", "--tb=short", "--no-header"])
    run_report["stages"]["tests"] = {"exit_code": rc}
    if rc != 0:
        run_report["ended_at"] = time.time()
        _write_run_report(run_report)
        raise SystemExit(rc)

    eval_summary = _load_json(v2_report_path("eval_summary")) or {}
    state = update_session_state(eval_score=float(eval_summary.get("overall_score", 0.0) or 0.0))
    run_report["post_session"] = {
        "eval_summary_path": str(v2_report_path("eval_summary")),
        "improvement_report_path": str(v2_report_path("improvement_report")),
        "audit_report_path": str(v2_report_path("audit_report")),
        "session_state": state,
        "milestone": _milestone_due(),
    }
    run_report["ended_at"] = time.time()
    _write_run_report(run_report)


class UnifiedTrainer:
    def __init__(
        self,
        data_path: str | None = None,
        checkpoint_path: str = "anra_v2_brain.pt",
        batch_size: int = V2_TRAINING.batch_size,
        block_size: int = V2_MODEL.block_size,
        answer_loss_weight: float = V2_TRAINING.answer_loss_weight,
        session_minutes: int = V2_TRAINING.session_minutes,
        identity_minutes: int = 12,
        ouroboros_minutes: int = 10,
    ):
        self.data_path = data_path
        self.checkpoint_path = checkpoint_path
        self.batch_size = batch_size
        self.block_size = block_size
        self.answer_loss_weight = answer_loss_weight
        self.session_minutes = session_minutes
        self.identity_minutes = identity_minutes
        self.ouroboros_minutes = ouroboros_minutes
        self._dataset: Path | None = None

    def resolve_dataset(self) -> Path:
        if self._dataset is None:
            self._dataset = resolve_dataset_path(self.data_path)
        return self._dataset

    def health_check(self) -> None:
        print_system_health()

    def train(self, mode: str = "session", **kwargs) -> int:
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
            "--session_minutes",
            str(self.session_minutes),
            "--identity_minutes",
            str(self.identity_minutes),
            "--ouroboros_minutes",
            str(self.ouroboros_minutes),
        ]
        if self.data_path:
            cmd.extend(["--data_path", self.data_path])
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
             
