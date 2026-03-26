"""
================================================================================
FILE: config_loader.py
PROJECT: Transformer Language Model — 45H Final Phase
PURPOSE: YAML configuration loading, validation, and CLI override system
================================================================================

Design:
  1. Load base.yaml defaults
  2. Overlay preset YAML (tiny/small/medium/large or custom)
  3. Apply CLI overrides (dot-notation: "train.learning_rate=1e-3")
  4. Validate all required fields and types
  5. Return a typed Config object with attribute access

Validation fails immediately with a clear message — no silent bad configs.
================================================================================
"""

import os
import sys
import yaml
import copy
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG CONTAINER
# ──────────────────────────────────────────────────────────────────────────────

class Config:
    """
    Nested config container with dot-notation access and dict-like iteration.

    Converts nested dicts into Config objects recursively so you can write:
      cfg.model.d_model  instead of  cfg["model"]["d_model"]

    Also supports:
      cfg.get("model.d_model", default=512)
      cfg.to_dict()
      cfg.set("train.learning_rate", 1e-4)
    """

    def __init__(self, data: Dict[str, Any]):
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

    def get(self, dotpath: str, default: Any = None) -> Any:
        """Get a nested value by dot-path, returning default if not found."""
        parts = dotpath.split(".")
        obj = self
        for part in parts:
            if not hasattr(obj, part):
                return default
            obj = getattr(obj, part)
        return obj

    def set(self, dotpath: str, value: Any) -> None:
        """Set a nested value by dot-path, creating intermediate Config objects if needed."""
        parts = dotpath.split(".")
        obj = self
        for part in parts[:-1]:
            if not hasattr(obj, part):
                setattr(obj, part, Config({}))
            obj = getattr(obj, part)
        setattr(obj, parts[-1], value)

    def to_dict(self) -> Dict[str, Any]:
        """Convert back to a plain nested dict."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result

    def __repr__(self) -> str:
        import json
        return json.dumps(self.to_dict(), indent=2, default=str)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)


# ──────────────────────────────────────────────────────────────────────────────
# DEEP MERGE
# ──────────────────────────────────────────────────────────────────────────────

def _deep_merge(base: Dict, override: Dict) -> Dict:
    """
    Recursively merge override into base.

    Only dict values are merged recursively — all other types (including lists)
    are replaced entirely by the override value.
    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


# ──────────────────────────────────────────────────────────────────────────────
# VALIDATION
# ──────────────────────────────────────────────────────────────────────────────

# (field_dotpath, expected_type_or_types, required, min_val, max_val)
_VALIDATION_RULES = [
    # Model
    ("model.vocab_size",   int,   True,  1,      None),
    ("model.d_model",      int,   True,  1,      None),
    ("model.num_layers",   int,   True,  1,      None),
    ("model.num_heads",    int,   True,  1,      None),
    ("model.num_kv_heads", int,   True,  1,      None),
    ("model.max_seq_len",  int,   True,  1,      None),
    ("model.dropout_rate", float, True,  0.0,    1.0),
    ("model.ffn_type",     str,   True,  None,   None),
    ("model.tie_weights",  bool,  True,  None,   None),
    ("model.rope_base",    float, False, 1.0,    None),

    # Train
    ("train.learning_rate",  float, False, 0.0, None),
    ("train.weight_decay",   float, False, 0.0, None),
    ("train.batch_size",     int,   False, 1,   None),
    ("train.seq_len",        int,   False, 1,   None),
    ("train.warmup_steps",   int,   False, 0,   None),
    ("train.max_steps",      int,   False, 1,   None),
    ("train.grad_clip",      float, False, 0.0, None),

    # Inference
    ("inference.temperature",  float, False, 0.0, None),
    ("inference.top_k",        int,   False, 0,   None),
    ("inference.top_p",        float, False, 0.0, 1.0),
]

_VALID_FFN_TYPES     = {"swiglu", "gelu"}
_VALID_LR_SCHEDULES  = {"cosine", "linear", "constant"}
_VALID_STRATEGIES    = {"greedy", "temperature", "top_k", "top_p"}
_VALID_LOG_LEVELS    = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


def _validate_config(cfg: Config) -> None:
    """
    Validate config fields.  Raises ConfigError with a clear message on failure.
    Checks types, ranges, and cross-field constraints.
    """
    errors = []

    for dotpath, expected_type, required, min_val, max_val in _VALIDATION_RULES:
        value = cfg.get(dotpath)

        if value is None:
            if required:
                errors.append(f"  MISSING required field: {dotpath}")
            continue

        # Type check — coerce ints for fields that accept float
        if expected_type is float and isinstance(value, int):
            cfg.set(dotpath, float(value))
            value = float(value)

        if not isinstance(value, expected_type):
            errors.append(
                f"  TYPE ERROR {dotpath}: expected {expected_type.__name__}, "
                f"got {type(value).__name__} ({value!r})"
            )
            continue

        if min_val is not None and value < min_val:
            errors.append(f"  RANGE ERROR {dotpath}: {value} < minimum {min_val}")
        if max_val is not None and value > max_val:
            errors.append(f"  RANGE ERROR {dotpath}: {value} > maximum {max_val}")

    # Cross-field constraints
    d_model   = cfg.get("model.d_model")
    num_heads = cfg.get("model.num_heads")
    kv_heads  = cfg.get("model.num_kv_heads")
    seq_len   = cfg.get("train.seq_len")
    max_seq   = cfg.get("model.max_seq_len")

    if d_model and num_heads and d_model % num_heads != 0:
        errors.append(
            f"  CONSTRAINT: model.d_model ({d_model}) must be divisible by "
            f"model.num_heads ({num_heads})"
        )

    if num_heads and kv_heads and num_heads % kv_heads != 0:
        errors.append(
            f"  CONSTRAINT: model.num_heads ({num_heads}) must be divisible by "
            f"model.num_kv_heads ({kv_heads})"
        )

    if seq_len and max_seq and seq_len > max_seq:
        errors.append(
            f"  CONSTRAINT: train.seq_len ({seq_len}) must be ≤ "
            f"model.max_seq_len ({max_seq})"
        )

    # Enum fields
    ffn_type = cfg.get("model.ffn_type")
    if ffn_type and ffn_type not in _VALID_FFN_TYPES:
        errors.append(f"  INVALID model.ffn_type: '{ffn_type}'. Must be one of {_VALID_FFN_TYPES}")

    strategy = cfg.get("inference.strategy")
    if strategy and strategy not in _VALID_STRATEGIES:
        errors.append(f"  INVALID inference.strategy: '{strategy}'. Must be one of {_VALID_STRATEGIES}")

    log_level = cfg.get("logging.level")
    if log_level and log_level not in _VALID_LOG_LEVELS:
        errors.append(f"  INVALID logging.level: '{log_level}'. Must be one of {_VALID_LOG_LEVELS}")

    if errors:
        raise ConfigError(
            "Configuration validation failed:\n" + "\n".join(errors) +
            "\n\nFix the above errors in your config file or CLI overrides."
        )


class ConfigError(ValueError):
    """Raised when configuration is invalid. Message includes fix instructions."""
    pass


# ──────────────────────────────────────────────────────────────────────────────
# LOADER
# ──────────────────────────────────────────────────────────────────────────────

_BASE_CONFIG_PATH = Path(__file__).parent / "config" / "base.yaml"


def load_config(
    config_path:  Optional[str]       = None,
    overrides:    Optional[List[str]] = None,
    validate:     bool                = True,
) -> Config:
    """
    Load, merge, and validate configuration.

    Merge order (later overrides earlier):
      1. config/base.yaml    (all defaults)
      2. config_path         (preset or custom YAML)
      3. overrides           (CLI key=value pairs)

    Args:
        config_path: Path to a YAML config file (preset or custom).
                     None = use only base.yaml defaults.
        overrides:   List of "section.field=value" strings from CLI.
                     Example: ["train.learning_rate=1e-3", "model.d_model=256"]
        validate:    Run validation after loading (default: True).

    Returns:
        Config object with attribute-style access.

    Raises:
        ConfigError: If validation fails.
        FileNotFoundError: If config_path does not exist.
    """
    # Step 1: Load base defaults
    if not _BASE_CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Base config not found at {_BASE_CONFIG_PATH}. "
            "Make sure config/base.yaml exists alongside this script."
        )
    with open(_BASE_CONFIG_PATH) as f:
        data = yaml.safe_load(f) or {}
    logger.debug(f"Loaded base config from {_BASE_CONFIG_PATH}")

    # Step 2: Overlay preset/custom config
    if config_path is not None:
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Config file not found: {config_path}\n"
                f"Available presets: config/tiny.yaml, small.yaml, medium.yaml, large.yaml"
            )
        with open(path) as f:
            preset = yaml.safe_load(f) or {}
        data = _deep_merge(data, preset)
        logger.info(f"Loaded config preset: {config_path}")

    # Step 3: Apply CLI overrides
    if overrides:
        for override in overrides:
            if "=" not in override:
                raise ConfigError(
                    f"Invalid CLI override: '{override}'\n"
                    "Format must be: section.field=value  (e.g. train.learning_rate=1e-3)"
                )
            key, _, raw_value = override.partition("=")
            # Parse value: try int, float, bool, then keep as string
            parsed = _parse_scalar(raw_value)
            # Navigate to the parent section and set the field
            _dict_set(data, key.strip(), parsed)
            logger.debug(f"CLI override: {key} = {parsed!r}")

    cfg = Config(data)

    if validate:
        _validate_config(cfg)
        logger.info("Config validated successfully")

    return cfg


def _parse_scalar(s: str) -> Any:
    """Parse a string CLI value into the appropriate Python type."""
    s = s.strip()
    if s.lower() in ("true", "yes"):
        return True
    if s.lower() in ("false", "no"):
        return False
    if s.lower() in ("null", "none", "~"):
        return None
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s   # leave as string


def _dict_set(d: Dict, dotpath: str, value: Any) -> None:
    """Set a value in a nested dict using a dot-separated path."""
    keys = dotpath.split(".")
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value


# ──────────────────────────────────────────────────────────────────────────────
# SELF-TEST
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.DEBUG)

    print("=" * 60)
    print("  config_loader.py — Self-Test")
    print("=" * 60)

    script_dir = Path(__file__).parent
    os.chdir(script_dir)

    # Test 1: Load base config
    print("\n[1] Load base config")
    cfg = load_config(validate=False)
    print(f"  d_model={cfg.model.d_model}  num_heads={cfg.model.num_heads}")

    # Test 2: Load tiny preset
    print("\n[2] Load tiny preset")
    cfg_tiny = load_config("config/tiny.yaml")
    print(f"  d_model={cfg_tiny.model.d_model}  batch={cfg_tiny.train.batch_size}")
    assert cfg_tiny.model.d_model == 128

    # Test 3: CLI overrides
    print("\n[3] CLI overrides")
    cfg_ov = load_config("config/tiny.yaml", overrides=["train.learning_rate=1e-2", "model.d_model=64"])
    assert cfg_ov.train.learning_rate == 1e-2
    assert cfg_ov.model.d_model == 64
    print(f"  lr={cfg_ov.train.learning_rate}  d_model={cfg_ov.model.d_model}")

    # Test 4: Invalid config is caught
    print("\n[4] Validation catches errors")
    try:
        bad_cfg = load_config(overrides=["model.d_model=7", "model.num_heads=8"])  # 7 % 8 != 0
        print("  ERROR: should have raised ConfigError!")
        sys.exit(1)
    except ConfigError as e:
        print(f"  ConfigError caught correctly: {str(e)[:80]}...")

    # Test 5: to_dict and round-trip
    print("\n[5] Config serialization")
    d = cfg_tiny.to_dict()
    assert "model" in d and "train" in d
    print(f"  Keys: {list(d.keys())}")

    print("\n  ✓ All config tests passed")
    print("=" * 60)
