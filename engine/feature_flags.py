from __future__ import annotations

import json
from contextlib import contextmanager
from typing import Iterator

from anra.anra_paths import STATE_DIR

FLAGS_FILE = STATE_DIR / "feature_flags.json"

_DEFAULTS: dict[str, bool] = {
    "brain": True,
    "tokenizer": True,
    "data_mix": True,
    "training_loop": True,
    "evaluation": True,
    "runtime": True,
    "api_web": True,
    "identity": False,
    "memory": True,
    "goals": True,
    "agent_loop": False,
    "self_modification": False,
    "ouroboros": False,
    "symbolic_bridge": True,
    "sovereignty": True,
    "cognition": True,
    "causal_reasoning": False,
    "epistemic_tracker": False,
    "human_model": False,
    "ssie": False,
    "cdse": False,
    "cec": False,
    "self_debate": False,
    "inference_efficiency": False,
    "intelligence": False,
    "multimodal": False,
    "robotics": False,
    "v4_training": True,
}


def load_flags() -> dict[str, bool]:
    if FLAGS_FILE.exists():
        try:
            overrides = json.loads(FLAGS_FILE.read_text(encoding="utf-8"))
            if not isinstance(overrides, dict):
                return dict(_DEFAULTS)
            known = {
                str(name): bool(value)
                for name, value in overrides.items()
                if name in _DEFAULTS and isinstance(value, bool)
            }
            return {**_DEFAULTS, **known}
        except Exception:
            pass
    return dict(_DEFAULTS)


def is_enabled(component_name: str) -> bool:
    # An unknown component name is not a capability: returning True for any
    # string meant a typo or unregistered feature silently passed every gate.
    return load_flags().get(component_name, False)


def set_flag(component_name: str, enabled: bool) -> None:
    if component_name not in _DEFAULTS:
        raise KeyError(f"unknown feature flag: {component_name}")
    flags = load_flags()
    flags[component_name] = bool(enabled)
    FLAGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    FLAGS_FILE.write_text(json.dumps(flags, indent=2), encoding="utf-8")


def enabled_components() -> list[str]:
    return [name for name, on in load_flags().items() if on]


def disabled_components() -> list[str]:
    return [name for name, on in load_flags().items() if not on]


@contextmanager
def enable_feature(component_name: str | list[str]) -> Iterator[None]:
    names = [component_name] if isinstance(component_name, str) else list(component_name)
    previous = {name: load_flags().get(name, False) for name in names}
    for name in names:
        set_flag(name, True)
    try:
        yield
    finally:
        for name, was_on in previous.items():
            set_flag(name, was_on)
