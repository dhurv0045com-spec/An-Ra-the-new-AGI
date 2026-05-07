from __future__ import annotations

import json
from pathlib import Path

from anra_paths import STATE_DIR

FLAGS_FILE = STATE_DIR / "feature_flags.json"

_DEFAULTS: dict[str, bool] = {
    "brain": True,
    "tokenizer": True,
    "data_mix": True,
    "training_loop": True,
    "evaluation": True,
    "runtime": True,
    "api_web": True,
    "identity": True,
    "memory": True,
    "phase2_memory": True,
    "goals": True,
    "agent_loop": True,
    "master_system": True,
    "self_improvement": True,
    "self_modification": True,
    "ouroboros": True,
    "ghost_memory": True,
    "symbolic_bridge": True,
    "sovereignty": True,
}


def load_flags() -> dict[str, bool]:
    if FLAGS_FILE.exists():
        try:
            overrides = json.loads(FLAGS_FILE.read_text(encoding="utf-8"))
            return {**_DEFAULTS, **overrides}
        except Exception:
            pass
    return dict(_DEFAULTS)


def is_enabled(component_name: str) -> bool:
    return load_flags().get(component_name, True)


def set_flag(component_name: str, enabled: bool) -> None:
    flags = load_flags()
    flags[component_name] = bool(enabled)
    FLAGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    FLAGS_FILE.write_text(json.dumps(flags, indent=2), encoding="utf-8")


def enabled_components() -> list[str]:
    return [name for name, on in load_flags().items() if on]


def disabled_components() -> list[str]:
    return [name for name, on in load_flags().items() if not on]
