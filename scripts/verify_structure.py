from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from anra_paths import (
    AGENT_LOOP_DIR,
    CORE_DIR,
    IDENTITY_DIR,
    MEMORY_DIR,
    OUROBOROS_DIR,
    ROOT,
    SOVEREIGNTY_DIR,
    SYMBOLIC_BRIDGE_DIR,
    TRAINING_DATA_DIR,
    TOKENIZER_DIR,
)


def main() -> int:
    required = [
        ROOT / "app.py",
        ROOT / "generate.py",
        ROOT / "anra_paths.py",
        ROOT / "anra_brain.py",
        TRAINING_DATA_DIR / "anra_dataset_v6_1.txt",
        TOKENIZER_DIR / "tokenizer.pkl",
        ROOT / "training" / "curriculum.py",
        ROOT / "config" / "optimization_config.json",
        CORE_DIR / "model.py",
        MEMORY_DIR / "memory_manager.py",
        AGENT_LOOP_DIR / "agent_main.py",
        IDENTITY_DIR / "identity_injector.py",
        OUROBOROS_DIR / "ouroboros_numpy.py",
        SYMBOLIC_BRIDGE_DIR / "symbolic_bridge.py",
        SOVEREIGNTY_DIR / "sovereignty_bridge.py",
        ROOT / "training" / "finetune_anra.py",
        ROOT / "training" / "train_unified.py",
        ROOT / "training" / "optimizations.py",
        ROOT / "inference" / "optimize_context_window.py",
        ROOT / "inference" / "full_system_connector.py",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        print("Missing required files:")
        for m in missing:
            print(f"- {m}")
        return 1
    print(f"Structure OK — {len(required)} files verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
