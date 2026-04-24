from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from anra_paths import (
    ROOT,
    TRAINING_DATA_DIR,
    DRIVE_CHECKPOINTS,
    ensure_dirs,
    inject_all_paths,
    get_dataset_file,
)

inject_all_paths()
ensure_dirs()


def _valid_text_dataset(path: Path) -> bool:
    if not path.exists() or not path.is_file() or path.suffix.lower() not in {".txt", ".jsonl", ".md"}:
        return False
    if path.stat().st_size < 1024:
        return False
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return False
    return "H:" in text and "ANRA:" in text


def _iter_training_files() -> Iterable[Path]:
    candidates: list[Path] = []
    for base in [TRAINING_DATA_DIR, ROOT / "training"]:
        if not base.exists():
            continue
        for p in base.glob("*"):
            if _valid_text_dataset(p):
                candidates.append(p)
    return sorted(candidates, key=lambda p: p.stat().st_size, reverse=True)


def resolve_dataset_path(explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.is_absolute():
            p = (ROOT / p).resolve()
        if not _valid_text_dataset(p):
            raise FileNotFoundError(f"Invalid dataset path: {p}")
        return p

    canonical = get_dataset_file()
    if _valid_text_dataset(canonical):
        return canonical

    files = list(_iter_training_files())
    if files:
        return files[0]
    raise FileNotFoundError(f"No valid dataset found in {TRAINING_DATA_DIR}")


def _module_health(module_name: str) -> str:
    try:
        mod = __import__(module_name)
        fn = getattr(mod, "health_check", None)
        if callable(fn):
            report = fn()
            return str(report.get("status", "ok"))
        return "ok"
    except Exception as exc:
        return f"degraded ({exc})"


def print_system_health() -> None:
    modules = {
        "identity": "identity_injector",
        "ouroboros": "ouroboros_numpy",
        "symbolic": "symbolic_bridge",
        "sovereignty": "sovereignty_bridge",
    }
    print("\n[Unified Trainer] Subsystem health")
    for label, mod in modules.items():
        print(f"- {label}: {_module_health(mod)}")


def _restore_checkpoint_if_available(checkpoint_name: str) -> Path | None:
    local = ROOT / checkpoint_name
    if local.exists():
        return local
    remote = DRIVE_CHECKPOINTS / checkpoint_name
    if remote.exists():
        local.parent.mkdir(parents=True, exist_ok=True)
        try:
            import shutil
            shutil.copy2(remote, local)
            print(f"[Unified Trainer] restored checkpoint from drive: {remote}")
            return local
        except Exception:
            return None
    return None


def _ensure_turboquant_runtime_config() -> Path:
    cfg_path = ROOT / "config" / "optimization_config.json"
    data = {}
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
    cfg_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return cfg_path


def run_cmd(cmd: list[str], *, cwd: Path | None = None) -> int:
    print("\n[Unified Trainer] Running:", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(cwd or ROOT), check=False)
    return int(result.returncode)


def main() -> None:
    ap = argparse.ArgumentParser(description="An-Ra unified training dispatcher")
    ap.add_argument("--mode", default="session", choices=["session", "train", "resume", "eval", "status"])
    ap.add_argument("--data_path", default=None)
    ap.add_argument("--checkpoint_path", default="anra_brain.pt")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--block_size", type=int, default=128)
    ap.add_argument("--ouroboros_steps", type=int, default=5000)
    ap.add_argument("--session_minutes", type=int, default=30)
    ap.add_argument("--skip_ouroboros", action="store_true")
    ap.add_argument("--skip_finetune", action="store_true")
    ap.add_argument("--skip_self_improvement", action="store_true")
    ap.add_argument("--skip_sovereignty", action="store_true")
    args = ap.parse_args()

    if args.mode == "status":
        print_system_health()
        print(f"dataset: {resolve_dataset_path(args.data_path)}")
        return

    if args.mode == "eval":
        raise SystemExit(run_cmd([sys.executable, "-m", "tests.test_suite"]))

    dataset = resolve_dataset_path(args.data_path)
    print(f"[Unified Trainer] dataset={dataset}")
    print_system_health()
    turbo_cfg = _ensure_turboquant_runtime_config()
    print(f"[Unified Trainer] turboquant config synced: {turbo_cfg}")
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
        "--max_minutes",
        str(args.session_minutes),
    ]
    rc = run_cmd(base_cmd)
    run_report["stages"]["base"] = {"exit_code": rc}
    if rc != 0:
        run_report["ended_at"] = time.time()
        (ROOT / "output").mkdir(parents=True, exist_ok=True)
        (ROOT / "output" / "unified_training_report.json").write_text(json.dumps(run_report, indent=2), encoding="utf-8")
        raise SystemExit(rc)

    if args.mode == "session":
        # Fast resumable 30-minute session mode for Colab: base training only.
        run_report["ended_at"] = time.time()
        (ROOT / "output").mkdir(parents=True, exist_ok=True)
        (ROOT / "output" / "unified_training_report.json").write_text(json.dumps(run_report, indent=2), encoding="utf-8")
        return

    if not args.skip_finetune:
        rc = run_cmd([sys.executable, "-m", "training.finetune_anra"])
        run_report["stages"]["finetune"] = {"exit_code": rc}
        if rc != 0:
            run_report["ended_at"] = time.time()
            (ROOT / "output").mkdir(parents=True, exist_ok=True)
            (ROOT / "output" / "unified_training_report.json").write_text(json.dumps(run_report, indent=2), encoding="utf-8")
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
            (ROOT / "output").mkdir(parents=True, exist_ok=True)
            (ROOT / "output" / "unified_training_report.json").write_text(json.dumps(run_report, indent=2), encoding="utf-8")
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
        run_report["ended_at"] = time.time()
        (ROOT / "output").mkdir(parents=True, exist_ok=True)
        (ROOT / "output" / "unified_training_report.json").write_text(json.dumps(run_report, indent=2), encoding="utf-8")
        raise SystemExit(rc)
    run_report["ended_at"] = time.time()
    (ROOT / "output").mkdir(parents=True, exist_ok=True)
    (ROOT / "output" / "unified_training_report.json").write_text(json.dumps(run_report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()


# ── Backward-compatibility: UnifiedTrainer class ─────────────────────────────
# Some legacy notebooks import `from training.train_unified import UnifiedTrainer`
# or `from training.train_unified import AnRaTrainer`.
# This class wraps the existing functional API so those imports succeed.

class UnifiedTrainer:
    """Compatibility shim — wraps the functional training API as a class."""

    def __init__(
        self,
        data_path: str | None = None,
        checkpoint_path: str = "anra_brain.pt",
        batch_size: int = 64,
        block_size: int = 128,
        session_minutes: int = 30,
        ouroboros_steps: int = 5000,
    ):
        self.data_path = data_path
        self.checkpoint_path = checkpoint_path
        self.batch_size = batch_size
        self.block_size = block_size
        self.session_minutes = session_minutes
        self.ouroboros_steps = ouroboros_steps
        self._dataset: Path | None = None

    def resolve_dataset(self) -> Path:
        """Locate the training dataset."""
        if self._dataset is None:
            self._dataset = resolve_dataset_path(self.data_path)
        return self._dataset

    def health_check(self) -> None:
        """Print subsystem health status."""
        print_system_health()

    def train(self, mode: str = "session", **kwargs) -> int:
        """Run training via subprocess (same as CLI)."""
        cmd = [
            sys.executable, "-m", "training.train_unified",
            "--mode", mode,
            "--checkpoint_path", self.checkpoint_path,
            "--batch_size", str(self.batch_size),
            "--block_size", str(self.block_size),
            "--session_minutes", str(self.session_minutes),
            "--ouroboros_steps", str(self.ouroboros_steps),
        ]
        if self.data_path:
            cmd.extend(["--data_path", self.data_path])
        for flag_name in ("skip_ouroboros", "skip_finetune", "skip_self_improvement", "skip_sovereignty"):
            if kwargs.get(flag_name, False):
                cmd.append(f"--{flag_name}")
        return run_cmd(cmd)

    def status(self) -> None:
        """Print dataset + subsystem status."""
        print_system_health()
        try:
            ds = self.resolve_dataset()
            print(f"dataset: {ds}")
        except FileNotFoundError as e:
            print(f"dataset: {e}")

    def run_session(self, minutes: int | None = None) -> int:
        """Quick 30-minute (or custom) training session."""
        if minutes is not None:
            self.session_minutes = minutes
        return self.train(mode="session")


# Legacy alias — some old notebooks use AnRaTrainer
AnRaTrainer = UnifiedTrainer

