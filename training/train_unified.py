from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from anra_paths import DRIVE_CHECKPOINTS, ROOT, TRAINING_DATA_DIR, ensure_dirs, inject_all_paths

inject_all_paths()
ensure_dirs()

_CANONICAL_DATASET = TRAINING_DATA_DIR / "anra_dataset_v6_1.txt"
_DRIVE_DATASET = DRIVE_CHECKPOINTS.parent / "anra_dataset_v6_1.txt"


def _valid_text_dataset(path: Path) -> bool:
    """Return True only for the canonical dataset file with valid content."""
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
                f"Dataset invalid or too small (must be anra_dataset_v6_1.txt "
                f"with H:/ANRA: format, >100 KB): {path}"
            )
        return path

    if _valid_text_dataset(_CANONICAL_DATASET):
        return _CANONICAL_DATASET

    if _valid_text_dataset(_DRIVE_DATASET):
        TRAINING_DATA_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(_DRIVE_DATASET, _CANONICAL_DATASET)
        print(f"[Unified Trainer] Dataset restored from Drive: {_CANONICAL_DATASET}", flush=True)
        return _CANONICAL_DATASET

    raise FileNotFoundError(
        "\n\n[FATAL] Training dataset not found.\n"
        f"  Expected locally:  {_CANONICAL_DATASET}\n"
        f"  Or on Drive:       {_DRIVE_DATASET}\n\n"
        "  Action: Upload 'anra_dataset_v6_1.txt' to Google Drive at:\n"
        "    MyDrive/AnRa/anra_dataset_v6_1.txt\n"
        "  Then re-run Cell 4 in AnRa_Master.ipynb.\n"
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


def _module_health_report(module_name: str) -> dict:
    try:
        mod = importlib.import_module(module_name)
        fn = getattr(mod, "health_check", None)
        if callable(fn):
            result = fn()
            if isinstance(result, dict):
                return result
            return {"status": "ok", "module": module_name}
        return {"status": "ok", "module": module_name, "detail": "no health_check"}
    except Exception as exc:
        return {"status": "degraded", "module": module_name, "reason": str(exc)}


def print_system_health() -> None:
    subsystems = {
        "identity    (45N)": "identity_injector",
        "ouroboros   (45O)": "ouroboros_numpy",
        "ghost_mem   (45P)": "ghost_memory",
        "symbolic    (45Q)": "symbolic_bridge",
        "sovereignty (45R)": "sovereignty_bridge",
        "turboquant       ": "turboquant",
    }
    print("\n[Unified Trainer] Subsystem health:")
    for label, mod_name in subsystems.items():
        status = _module_health(mod_name)
        icon = "OK" if "ok" in status.lower() else "WARN"
        print(f"  {icon:<4} {label}: {status}")
    print()


def _restore_checkpoint_if_available(checkpoint_name: str) -> Path | None:
    local = ROOT / checkpoint_name
    if local.exists():
        return local
    remote = DRIVE_CHECKPOINTS / checkpoint_name
    if remote.exists():
        local.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(remote, local)
            print(f"[Unified Trainer] restored checkpoint from drive: {remote}", flush=True)
            return local
        except Exception:
            return None
    return None


def _ensure_turboquant_runtime_config() -> Path:
    cfg_path = ROOT / "config" / "optimization_config.json"
    data: dict[str, object] = {}
    if cfg_path.exists():
        try:
            data = json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    data.setdefault("adaptive_scheduler", True)
    data.setdefault("hard_sample_routing", True)
    data.setdefault("turboquant_enabled", True)
    data.setdefault("turboquant_bits", 4)
    data.setdefault("ghost_memory_enabled", True)
    data.setdefault("symbolic_bridge_enabled", True)
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return cfg_path


def _write_run_report(report: dict) -> None:
    (ROOT / "output").mkdir(parents=True, exist_ok=True)
    (ROOT / "output" / "unified_training_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )


def _load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _keyword_hits(text: str, keywords: list[str]) -> int:
    lowered = text.lower()
    return sum(1 for keyword in keywords if keyword in lowered)


def _build_curriculum_recommendations(hard_examples: list[dict], metrics: dict | None) -> list[str]:
    combined = " ".join(str(example.get("preview", "")) for example in hard_examples)
    recommendations: list[str] = []
    if _keyword_hits(combined, ["why", "because", "explain", "reason", "compare"]) >= 2:
        recommendations.append("Add more multi-step reasoning and explanation exchanges to the next session.")
    if _keyword_hits(combined, ["differentiate", "derivative", "solve", "equation", "logic", "proof"]) >= 2:
        recommendations.append("Upsample verified symbolic math and logic samples generated through symbolic_bridge.")
    if _keyword_hits(combined, ["remember", "memory", "before", "earlier", "context"]) >= 1:
        recommendations.append("Replay continuity-heavy conversations to strengthen long-horizon memory behavior.")
    if _keyword_hits(combined, ["who are you", "anra", "identity", "sovereign"]) >= 1:
        recommendations.append("Mix in identity-preservation turns so An-Ra stays stable under pressure.")
    if metrics and float(metrics.get("best_loss", 0.0) or 0.0) > 3.8:
        recommendations.append("Keep session length the same, but bias the next curriculum toward the hardest windows instead of widening the model.")
    if not recommendations:
        recommendations.append("Continue the current curriculum; no dominant weak area stood out in the latest hard examples.")
    return recommendations


def _post_session_review(run_report: dict) -> dict:
    output_dir = ROOT / "output"
    metrics = _load_json(output_dir / "session_train_metrics.json")
    hard_examples_blob = _load_json(output_dir / "hard_examples.json") or {}
    hard_examples = hard_examples_blob.get("examples", []) if isinstance(hard_examples_blob, dict) else []

    subsystems = {
        "identity": _module_health_report("identity_injector"),
        "ouroboros": _module_health_report("ouroboros_numpy"),
        "ghost_memory": _module_health_report("ghost_memory"),
        "symbolic_bridge": _module_health_report("symbolic_bridge"),
        "sovereignty": _module_health_report("sovereignty_bridge"),
        "turboquant": _module_health_report("turboquant"),
    }

    symbolic_checks: dict[str, object] = {}
    try:
        symbolic = importlib.import_module("symbolic_bridge")
        query = getattr(symbolic, "query", None)
        if callable(query):
            derivative = query("differentiate x^3 + 2*x")
            logic = query("Is (A->B) and (B->C) -> (A->C) a tautology?")
            symbolic_checks = {
                "derivative": getattr(derivative, "answer_text", str(derivative)),
                "derivative_confidence": float(getattr(derivative, "confidence", 0.0)),
                "logic": getattr(logic, "answer_text", str(logic)),
                "logic_confidence": float(getattr(logic, "confidence", 0.0)),
            }
    except Exception as exc:
        symbolic_checks = {"status": "degraded", "reason": str(exc)}

    review = {
        "generated_at": time.time(),
        "mode": run_report.get("mode"),
        "subsystems": subsystems,
        "symbolic_checks": symbolic_checks,
        "metrics_snapshot": metrics or {},
        "hard_examples": hard_examples[:8],
        "recommendations": _build_curriculum_recommendations(hard_examples, metrics),
    }
    (output_dir / "next_session_curriculum.json").write_text(
        json.dumps(review, indent=2),
        encoding="utf-8",
    )
    return review


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


def main() -> None:
    ap = argparse.ArgumentParser(description="An-Ra unified training dispatcher")
    ap.add_argument("--model_line", default=os.environ.get("ANRA_MODEL_LINE", "v2"), choices=["v1", "v2"])
    ap.add_argument("--mode", default="session", choices=["session", "train", "resume", "eval", "status"])
    ap.add_argument("--data_path", default=None)
    ap.add_argument("--checkpoint_path", default="anra_brain.pt")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--block_size", type=int, default=256)
    ap.add_argument("--answer_loss_weight", type=float, default=1.75)
    ap.add_argument("--ouroboros_steps", type=int, default=5000)
    ap.add_argument("--session_minutes", type=int, default=30)
    ap.add_argument("--skip_ouroboros", action="store_true")
    ap.add_argument("--skip_finetune", action="store_true")
    ap.add_argument("--skip_self_improvement", action="store_true")
    ap.add_argument("--skip_sovereignty", action="store_true")
    args = ap.parse_args()

    if args.model_line == "v2":
        mode_map = {
            "session": "session",
            "resume": "session",
            "train": "milestone",
            "eval": "eval",
            "status": "status",
        }
        delegated = [
            sys.executable,
            "-m",
            "training.train_v2",
            "--mode",
            mode_map[args.mode],
        ]
        if args.data_path:
            delegated.extend(["--data_path", args.data_path])
        delegated.extend(
            [
                "--checkpoint_path",
                "anra_v2_brain.pt",
                "--batch_size",
                str(min(args.batch_size, 32)),
                "--block_size",
                str(args.block_size),
                "--answer_loss_weight",
                str(args.answer_loss_weight),
                "--session_minutes",
                str(args.session_minutes),
            ]
        )
        raise SystemExit(run_cmd(delegated))

    if args.mode == "status":
        print_system_health()
        print(f"dataset: {resolve_dataset_path(args.data_path)}")
        return

    if args.mode == "eval":
        raise SystemExit(run_cmd([sys.executable, "-m", "tests.test_suite"]))

    dataset = resolve_dataset_path(args.data_path)
    print(f"[Unified Trainer] dataset={dataset}", flush=True)
    print_system_health()
    turbo_cfg = _ensure_turboquant_runtime_config()
    print(f"[Unified Trainer] turboquant config synced: {turbo_cfg}", flush=True)
    _restore_checkpoint_if_available(args.checkpoint_path)
    _restore_checkpoint_if_available("anra_brain_identity.pt")

    run_report = {
        "started_at": time.time(),
        "dataset": str(dataset),
        "checkpoint_path": args.checkpoint_path,
        "mode": args.mode,
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
    rc = run_cmd(base_cmd)
    run_report["stages"]["base"] = {"exit_code": rc}
    if rc != 0:
        run_report["ended_at"] = time.time()
        _write_run_report(run_report)
        raise SystemExit(rc)

    if args.mode == "session":
        review = _post_session_review(run_report)
        run_report["post_session_review"] = {
            "path": str(ROOT / "output" / "next_session_curriculum.json"),
            "recommendations": review.get("recommendations", []),
        }
        run_report["ended_at"] = time.time()
        _write_run_report(run_report)
        return

    if not args.skip_finetune:
        rc = run_cmd([sys.executable, "-m", "training.finetune_anra"])
        run_report["stages"]["finetune"] = {"exit_code": rc}
        if rc != 0:
            run_report["ended_at"] = time.time()
            _write_run_report(run_report)
            raise SystemExit(rc)

    if not args.skip_ouroboros:
        base_model = "anra_brain_identity.pt" if (ROOT / "anra_brain_identity.pt").exists() else args.checkpoint_path
        rc = run_cmd(
            [
                sys.executable,
                str(ROOT / "scripts" / "train_ouroboros.py"),
                "--base_model",
                base_model,
                "--steps",
                str(args.ouroboros_steps),
            ]
        )
        run_report["stages"]["ouroboros"] = {"exit_code": rc}
        if rc != 0:
            run_report["ended_at"] = time.time()
            _write_run_report(run_report)
            raise SystemExit(rc)

    if not args.skip_self_improvement:
        rc = run_cmd([sys.executable, "run_self_improvement.py"])
        run_report["stages"]["self_improvement"] = {"exit_code": rc}

    if not args.skip_sovereignty:
        rc = run_cmd([sys.executable, "run_sovereignty_audit.py"])
        run_report["stages"]["sovereignty"] = {"exit_code": rc}

    if args.mode == "train":
        rc = run_cmd([sys.executable, "-m", "tests.test_suite"])
        run_report["stages"]["test_suite"] = {"exit_code": rc}
        review = _post_session_review(run_report)
        run_report["post_session_review"] = {
            "path": str(ROOT / "output" / "next_session_curriculum.json"),
            "recommendations": review.get("recommendations", []),
        }
        run_report["ended_at"] = time.time()
        _write_run_report(run_report)
        raise SystemExit(rc)

    review = _post_session_review(run_report)
    run_report["post_session_review"] = {
        "path": str(ROOT / "output" / "next_session_curriculum.json"),
        "recommendations": review.get("recommendations", []),
    }
    run_report["ended_at"] = time.time()
    _write_run_report(run_report)


if __name__ == "__main__":
    main()


class UnifiedTrainer:
    """Compatibility shim that wraps the functional API as a class."""

    def __init__(
        self,
        data_path: str | None = None,
        checkpoint_path: str = "anra_brain.pt",
        batch_size: int = 64,
        block_size: int = 256,
        answer_loss_weight: float = 1.75,
        session_minutes: int = 30,
        ouroboros_steps: int = 5000,
    ):
        self.data_path = data_path
        self.checkpoint_path = checkpoint_path
        self.batch_size = batch_size
        self.block_size = block_size
        self.answer_loss_weight = answer_loss_weight
        self.session_minutes = session_minutes
        self.ouroboros_steps = ouroboros_steps
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
            "--ouroboros_steps",
            str(self.ouroboros_steps),
        ]
        if self.data_path:
            cmd.extend(["--data_path", self.data_path])
        for flag_name in ("skip_ouroboros", "skip_finetune", "skip_self_improvement", "skip_sovereignty"):
            if kwargs.get(flag_name, False):
                cmd.append(f"--{flag_name}")
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
