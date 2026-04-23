from __future__ import annotations

from pathlib import Path

from anra_paths import (
    AGENT_LOOP_DIR,
    CORE_DIR,
    IDENTITY_DIR,
    MEMORY_DIR,
    OUROBOROS_DIR,
    ROOT,
    SOVEREIGNTY_DIR,
    SYMBOLIC_BRIDGE_DIR,
)


def main() -> int:
    required = [
        ROOT / "app.py",
        ROOT / "generate.py",
        ROOT / "finetune_anra.py",
        ROOT / "anra_paths.py",
        ROOT / "training" / "curriculum.py",
        ROOT / "AnRa" / "optimization_config.json",
        CORE_DIR / "model.py",
        MEMORY_DIR / "memory_manager.py",
        AGENT_LOOP_DIR / "agent_main.py",
        IDENTITY_DIR / "identity_injector.py",
        OUROBOROS_DIR / "ouroboros_numpy.py",
        SYMBOLIC_BRIDGE_DIR / "symbolic_bridge.py",
        SOVEREIGNTY_DIR / "sovereignty_bridge.py",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        print("Missing required files:")
        for m in missing:
            print(f"- {m}")
        return 1
    print("Structure OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
