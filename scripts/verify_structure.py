from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from anra_paths import (
    AGENT_LOOP_DIR,
    IDENTITY_DIR,
    MEMORY_DIR,
    OUROBOROS_DIR,
    ROOT,
    SOVEREIGNTY_DIR,
    SYMBOLIC_BRIDGE_DIR,
    TRAINING_DATA_DIR,
    inject_all_paths,
)
from training.v2_runtime import load_or_build_v2_tokenizer

inject_all_paths()


def _ensure_v2_tokenizer() -> None:
    dataset_path = TRAINING_DATA_DIR / "anra_dataset_v6_1.txt"
    if not dataset_path.exists():
        return
    load_or_build_v2_tokenizer(dataset_path=dataset_path)


def main() -> int:
    _ensure_v2_tokenizer()
    required = [
        ROOT / "app.py",
        ROOT / "generate.py",
        ROOT / "anra_paths.py",
        ROOT / "anra_brain.py",
        ROOT / "tokenizer" / "subword_tokenizer.py",
        ROOT / "training" / "v2_config.py",
        ROOT / "training" / "v2_data_mix.py",
        ROOT / "training" / "v2_runtime.py",
        ROOT / "training" / "eval_v2.py",
        ROOT / "training" / "train_unified.py",
        ROOT / "training" / "finetune_anra.py",
        ROOT / "scripts" / "build_brain.py",
        ROOT / "scripts" / "train_ouroboros.py",
        ROOT / "scripts" / "run_self_improvement.py",
        ROOT / "scripts" / "run_sovereignty_audit.py",
        TRAINING_DATA_DIR / "anra_dataset_v6_1.txt",
        MEMORY_DIR / "memory_manager.py",
        AGENT_LOOP_DIR / "agent_main.py",
        IDENTITY_DIR / "identity_injector.py",
        OUROBOROS_DIR / "ouroboros_numpy.py",
        SYMBOLIC_BRIDGE_DIR / "symbolic_bridge.py",
        SOVEREIGNTY_DIR / "sovereignty_bridge.py",
        ROOT / "inference" / "optimize_context_window.py",
        ROOT / "inference" / "full_system_connector.py",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        print("Missing required files:")
        for item in missing:
            print(f"- {item}")
        return 1
    print(f"Structure OK - {len(required)} files verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
