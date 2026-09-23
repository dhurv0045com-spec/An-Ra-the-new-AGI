"""Strict build configuration for the BRAMASTRA integrated build (B01).

The configuration resolves one of the declared profiles into concrete model
geometry, validates feature flags and translates into the accepted base
decoder's ``ModelConfig`` without changing declared semantics. Unknown keys,
invalid head geometry and unsafe feature combinations reject loudly.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import math
from typing import Any, Mapping

from bramastra_lab.research.contracts.core import content_identity


class ConfigError(ValueError):
    """A configuration value does not satisfy the build contract."""


# Declared in engineering/build_20260912/build_contract.json. The numbers are
# the contract's; duplicating them here is the translation step B01 owns.
PROFILE_SPECS: Mapping[str, Mapping[str, int]] = {
    "tiny": {"vocab": 260, "layers": 2, "width": 64, "heads": 4, "ffn": 176, "max_seq": 128},
    "development": {"vocab": 260, "layers": 8, "width": 256, "heads": 4, "ffn": 704, "max_seq": 256},
    "future_capacity": {"vocab": 260, "layers": 12, "width": 512, "heads": 8, "ffn": 1408, "max_seq": 512},
    # Configuration-only Kaggle TPU candidate. Exact integrated count is
    # 100,334,720 (decoder + action/value heads); runtime fit is not implied.
    "tpu_100m": {"vocab": 260, "layers": 15, "width": 640, "heads": 10,
                  "ffn": 2624, "max_seq": 512},
}

LOGIT_TREATMENTS = frozenset({"full", "participating_mask", "inactive_offset"})
PAIR_SCORE_NORMALIZATIONS = frozenset({"eligible_token_mean"})
CONTROLLER_MODES = frozenset({"disabled", "fixed_schedule", "evidence_driven"})
PLANNER_MODES = frozenset({"none", "bounded"})
PRIMARY_GENERATION_MODES = frozenset({"full_vocabulary"})

# The byte tokenizer is the compatibility starting point: 256 byte values plus
# four declared structural tokens. Larger physical vocabularies remain allowed
# as controlled configurations; the tokenizer itself does not grow silently.
BYTE_TOKENIZER_NAME = "bramastra-byte-260/v1"
BYTE_TOKENIZER_SPECIALS = {"pad": 256, "eos": 257, "end_of_event": 258, "boundary": 259}
MIN_INTEGRATED_VOCAB = 260


def tokenizer_identity() -> str:
    """Stable identity of the declared byte tokenizer mapping."""
    return content_identity({"name": BYTE_TOKENIZER_NAME, "specials": BYTE_TOKENIZER_SPECIALS})


def _strict_section(raw: Mapping[str, Any], allowed: frozenset[str], name: str) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ConfigError(f"{name} must be an object")
    unknown = set(raw) - set(allowed)
    if unknown:
        raise ConfigError(f"{name} has unknown fields: {sorted(unknown)}")
    return dict(raw)


def _positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ConfigError(f"{name} must be a positive integer")
    return value


def _finite_number(value: Any, name: str, *, minimum: float | None = None,
                   strictly_positive: bool = False) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise ConfigError(f"{name} must be a finite number")
    if strictly_positive and value <= 0:
        raise ConfigError(f"{name} must be strictly positive")
    if minimum is not None and value < minimum:
        raise ConfigError(f"{name} must be at least {minimum}")
    return float(value)


@dataclass(frozen=True)
class ModelSection:
    """Concrete decoder geometry plus the declared tokenizer."""

    profile: str
    vocab: int
    layers: int
    width: int
    heads: int
    ffn: int
    max_seq: int
    tokenizer: str = BYTE_TOKENIZER_NAME

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ModelSection":
        values = _strict_section(
            raw, frozenset({"profile", "vocab", "layers", "width", "heads", "ffn", "max_seq", "tokenizer"}),
            "model section")
        if "profile" not in values:
            raise ConfigError("model section must declare a profile")
        profile = values["profile"]
        if not isinstance(profile, str) or profile not in PROFILE_SPECS:
            raise ConfigError(f"unknown profile: {profile!r}; expected one of {sorted(PROFILE_SPECS)}")
        resolved = dict(PROFILE_SPECS[profile])
        for key in ("vocab", "layers", "width", "heads", "ffn", "max_seq"):
            if key in values:
                resolved[key] = _positive_int(values[key], f"model.{key}")
        for key, value in resolved.items():
            resolved[key] = _positive_int(value, f"model.{key}")
        tokenizer = values.get("tokenizer", BYTE_TOKENIZER_NAME)
        if tokenizer != BYTE_TOKENIZER_NAME:
            raise ConfigError(f"unsupported tokenizer {tokenizer!r}; expected {BYTE_TOKENIZER_NAME!r}")
        section = cls(profile=profile, tokenizer=tokenizer, **resolved)
        section.validate()
        return section

    def validate(self) -> None:
        if self.vocab < MIN_INTEGRATED_VOCAB:
            raise ConfigError(
                f"model.vocab must be at least {MIN_INTEGRATED_VOCAB} for the byte tokenizer")
        if self.width % self.heads:
            raise ConfigError("model.width must be divisible by model.heads")
        if (self.width // self.heads) % 2:
            raise ConfigError("per-head width must be even for RoPE")
        if self.ffn < self.width:
            raise ConfigError("model.ffn must be at least model.width")
        if self.max_seq < 8:
            raise ConfigError("model.max_seq must be at least 8")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TrainingSection:
    """Objective and optimizer controls. Defaults are the full-loss controls."""

    seed: int = 0
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    grad_accum_steps: int = 1
    max_updates: int | None = None
    warmup_updates: int = 0
    clip_norm: float = 1.0
    logit_treatment: str = "full"
    effective_vocab: int | None = None
    pair_loss_weight: float = 0.0
    pair_margin: float = 1.0
    pair_score_normalization: str = "eligible_token_mean"
    length_bucketing: bool = False

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "TrainingSection":
        values = _strict_section(
            raw,
            frozenset({"seed", "learning_rate", "weight_decay", "grad_accum_steps", "max_updates",
                       "warmup_updates", "clip_norm", "logit_treatment", "effective_vocab",
                       "pair_loss_weight", "pair_margin", "pair_score_normalization",
                       "length_bucketing"}),
            "training section")
        kwargs: dict[str, Any] = {}
        if "seed" in values:
            seed = values["seed"]
            if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
                raise ConfigError("training.seed must be a nonnegative integer")
            kwargs["seed"] = seed
        if "learning_rate" in values:
            kwargs["learning_rate"] = _finite_number(values["learning_rate"], "training.learning_rate",
                                                     strictly_positive=True)
        if "weight_decay" in values:
            kwargs["weight_decay"] = _finite_number(values["weight_decay"], "training.weight_decay",
                                                    minimum=0.0)
        if "grad_accum_steps" in values:
            kwargs["grad_accum_steps"] = _positive_int(values["grad_accum_steps"], "training.grad_accum_steps")
        if "max_updates" in values and values["max_updates"] is not None:
            kwargs["max_updates"] = _positive_int(values["max_updates"], "training.max_updates")
        if "warmup_updates" in values:
            warmup = values["warmup_updates"]
            if not isinstance(warmup, int) or isinstance(warmup, bool) or warmup < 0:
                raise ConfigError("training.warmup_updates must be a nonnegative integer")
            kwargs["warmup_updates"] = warmup
        if "clip_norm" in values:
            kwargs["clip_norm"] = _finite_number(values["clip_norm"], "training.clip_norm",
                                                 strictly_positive=True)
        if "logit_treatment" in values:
            if values["logit_treatment"] not in LOGIT_TREATMENTS:
                raise ConfigError(
                    f"training.logit_treatment must be one of {sorted(LOGIT_TREATMENTS)}")
            kwargs["logit_treatment"] = values["logit_treatment"]
        if "effective_vocab" in values and values["effective_vocab"] is not None:
            kwargs["effective_vocab"] = _positive_int(values["effective_vocab"], "training.effective_vocab")
        if "pair_loss_weight" in values:
            kwargs["pair_loss_weight"] = _finite_number(values["pair_loss_weight"],
                                                        "training.pair_loss_weight", minimum=0.0)
        if "pair_margin" in values:
            kwargs["pair_margin"] = _finite_number(values["pair_margin"], "training.pair_margin",
                                                   strictly_positive=True)
        if "pair_score_normalization" in values:
            if values["pair_score_normalization"] not in PAIR_SCORE_NORMALIZATIONS:
                raise ConfigError("training.pair_score_normalization must be one of "
                                  f"{sorted(PAIR_SCORE_NORMALIZATIONS)}")
            kwargs["pair_score_normalization"] = values["pair_score_normalization"]
        if "length_bucketing" in values:
            if not isinstance(values["length_bucketing"], bool):
                raise ConfigError("training.length_bucketing must be a boolean")
            kwargs["length_bucketing"] = values["length_bucketing"]
        section = cls(**kwargs)
        section.validate(vocab=None)
        return section

    def validate(self, vocab: int | None) -> None:
        if self.logit_treatment == "inactive_offset":
            if self.effective_vocab is None:
                raise ConfigError(
                    "training.effective_vocab is required when logit_treatment is inactive_offset")
            if self.effective_vocab < 2:
                raise ConfigError("training.effective_vocab must be at least 2")
            if vocab is not None and self.effective_vocab > vocab:
                raise ConfigError("training.effective_vocab must not exceed model.vocab")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ControllerSection:
    """Plasticity-controller configuration (B06). Default is disabled."""

    mode: str = "disabled"
    controller_pool_id: str | None = None
    formation_threshold: float = 0.9
    reacquire_threshold: float = 0.6
    stabilize_window: int = 8
    stabilize_lr_multiplier: float = 0.25
    expand_lr_multiplier: float = 1.0
    reacquire_lr_multiplier: float = 1.0
    cooldown_updates: int = 16
    max_transitions: int = 32
    max_metrics_age_updates: int = 8
    exit_hysteresis: float = 0.05
    controller_eval_every: int = 4
    collapse_confirmation_evaluations: int = 2
    recovery_confirmation_evaluations: int = 2

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ControllerSection":
        values = _strict_section(
            raw,
            frozenset({"mode", "controller_pool_id", "formation_threshold", "reacquire_threshold",
                       "stabilize_window", "stabilize_lr_multiplier", "expand_lr_multiplier",
                       "reacquire_lr_multiplier", "cooldown_updates", "max_transitions",
                       "max_metrics_age_updates", "exit_hysteresis", "controller_eval_every",
                       "collapse_confirmation_evaluations", "recovery_confirmation_evaluations"}),
            "controller section")
        kwargs: dict[str, Any] = {}
        if "mode" in values:
            if values["mode"] not in CONTROLLER_MODES:
                raise ConfigError(f"controller.mode must be one of {sorted(CONTROLLER_MODES)}")
            kwargs["mode"] = values["mode"]
        if "controller_pool_id" in values and values["controller_pool_id"] is not None:
            pool_id = values["controller_pool_id"]
            if not isinstance(pool_id, str) or not pool_id.strip():
                raise ConfigError("controller.controller_pool_id must be a nonempty string")
            kwargs["controller_pool_id"] = pool_id
        for key in ("formation_threshold", "reacquire_threshold"):
            if key in values:
                kwargs[key] = _finite_number(values[key], f"controller.{key}", minimum=0.0, )
                if kwargs[key] > 1.0:
                    raise ConfigError(f"controller.{key} must be at most 1.0")
        for key in ("stabilize_lr_multiplier", "expand_lr_multiplier", "reacquire_lr_multiplier"):
            if key in values:
                kwargs[key] = _finite_number(values[key], f"controller.{key}", strictly_positive=True)
        for key in ("stabilize_window", "cooldown_updates", "max_transitions",
                    "max_metrics_age_updates", "controller_eval_every",
                    "collapse_confirmation_evaluations", "recovery_confirmation_evaluations"):
            if key in values:
                kwargs[key] = _positive_int(values[key], f"controller.{key}")
        section = cls(**kwargs)
        section.validate()
        return section

    def validate(self) -> None:
        if self.mode == "evidence_driven" and not self.controller_pool_id:
            raise ConfigError(
                "controller.controller_pool_id is required when controller.mode is evidence_driven")
        if self.reacquire_threshold >= self.formation_threshold:
            raise ConfigError(
                "controller.reacquire_threshold must be below controller.formation_threshold "
                "so enter/exit thresholds differ")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FeatureSection:
    """Optional-mechanism switches. Defaults keep every optional feature off."""

    retrieval_enabled: bool = False
    planner: str = "none"
    primary_generation: str = "full_vocabulary"
    require_eos: bool = True
    automatic_training_launch: bool = False

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "FeatureSection":
        values = _strict_section(
            raw,
            frozenset({"retrieval_enabled", "planner", "primary_generation", "require_eos",
                       "automatic_training_launch"}),
            "features section")
        kwargs: dict[str, Any] = {}
        if "retrieval_enabled" in values:
            if not isinstance(values["retrieval_enabled"], bool):
                raise ConfigError("features.retrieval_enabled must be a boolean")
            kwargs["retrieval_enabled"] = values["retrieval_enabled"]
        if "planner" in values:
            if values["planner"] not in PLANNER_MODES:
                raise ConfigError(f"features.planner must be one of {sorted(PLANNER_MODES)}")
            kwargs["planner"] = values["planner"]
        if "primary_generation" in values:
            if values["primary_generation"] not in PRIMARY_GENERATION_MODES:
                raise ConfigError("features.primary_generation must be one of "
                                  f"{sorted(PRIMARY_GENERATION_MODES)}")
            kwargs["primary_generation"] = values["primary_generation"]
        if "require_eos" in values:
            if not isinstance(values["require_eos"], bool):
                raise ConfigError("features.require_eos must be a boolean")
            kwargs["require_eos"] = values["require_eos"]
        if "automatic_training_launch" in values:
            if not isinstance(values["automatic_training_launch"], bool):
                raise ConfigError("features.automatic_training_launch must be a boolean")
            if values["automatic_training_launch"]:
                raise ConfigError(
                    "features.automatic_training_launch=true is not supported by this build; "
                    "training starts only through an explicit 'train' command")
            kwargs["automatic_training_launch"] = values["automatic_training_launch"]
        section = cls(**kwargs)
        section.validate()
        return section

    def validate(self) -> None:
        return

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReplaySection:
    """Experience-replay configuration (B2.1). Default is disabled.

    ``ledger_path`` is resolved against the prepared-data directory. When
    enabled, training mixes in replay batches drawn from the experience
    ledger at the declared proportion; the exact consumed counters and any
    shortfall are recorded in the run report.
    """

    enabled: bool = False
    proportion: float = 0.25
    family_weights: Mapping[str, float] = field(default_factory=dict)
    ledger_path: str = "episodes.jsonl"
    on_empty: str = "refuse"

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ReplaySection":
        values = _strict_section(
            raw,
            frozenset({"enabled", "proportion", "family_weights", "ledger_path",
                       "on_empty"}),
            "replay section")
        kwargs: dict[str, Any] = {}
        if "enabled" in values:
            if not isinstance(values["enabled"], bool):
                raise ConfigError("replay.enabled must be a boolean")
            kwargs["enabled"] = values["enabled"]
        if "proportion" in values:
            kwargs["proportion"] = _finite_number(values["proportion"], "replay.proportion",
                                                  minimum=0.0)
        if "family_weights" in values:
            weights = values["family_weights"]
            if not isinstance(weights, Mapping):
                raise ConfigError("replay.family_weights must be an object")
            for family, weight in weights.items():
                if not isinstance(family, str) or not family:
                    raise ConfigError("replay.family_weights keys must be nonempty strings")
                _finite_number(weight, f"replay.family_weights.{family}", strictly_positive=True)
            kwargs["family_weights"] = dict(weights)
        if "ledger_path" in values:
            if not isinstance(values["ledger_path"], str) or not values["ledger_path"]:
                raise ConfigError("replay.ledger_path must be a nonempty string")
            kwargs["ledger_path"] = values["ledger_path"]
        if "on_empty" in values:
            if values["on_empty"] not in ("refuse", "skip"):
                raise ConfigError("replay.on_empty must be 'refuse' or 'skip'")
            kwargs["on_empty"] = values["on_empty"]
        section = cls(**kwargs)
        section.validate()
        return section

    def validate(self) -> None:
        if self.enabled and not 0.0 < self.proportion <= 1.0:
            raise ConfigError("replay.proportion must lie in (0, 1] when replay is enabled")

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled, "proportion": self.proportion,
            "family_weights": dict(self.family_weights),
            "ledger_path": self.ledger_path, "on_empty": self.on_empty,
        }


@dataclass(frozen=True)
class BuildConfig:
    """Resolved, validated integrated-build configuration."""

    schema: str = "bramastra-build-config/v1"
    model: ModelSection | None = None
    training: TrainingSection | None = None
    controller: ControllerSection | None = None
    features: FeatureSection | None = None
    replay: ReplaySection | None = None

    def __post_init__(self) -> None:
        if self.schema != "bramastra-build-config/v1":
            raise ConfigError("unsupported build-config schema")
        if not isinstance(self.model, ModelSection):
            raise ConfigError("model section is required")
        self.model.validate()
        if self.training is None:
            object.__setattr__(self, "training", TrainingSection())
        self.training.validate(vocab=self.model.vocab)
        if self.controller is None:
            object.__setattr__(self, "controller", ControllerSection())
        self.controller.validate()
        if self.features is None:
            object.__setattr__(self, "features", FeatureSection())
        self.features.validate()
        if self.replay is None:
            object.__setattr__(self, "replay", ReplaySection())
        self.replay.validate()

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "BuildConfig":
        if not isinstance(raw, Mapping):
            raise ConfigError("build config must be an object")
        unknown = set(raw) - {"schema", "model", "training", "controller", "features",
                              "replay"}
        if unknown:
            raise ConfigError(f"build config has unknown fields: {sorted(unknown)}")
        schema = raw.get("schema", "bramastra-build-config/v1")
        if "model" not in raw:
            raise ConfigError("build config is missing the model section")
        return cls(
            schema=schema,
            model=ModelSection.from_dict(raw["model"]),
            training=TrainingSection.from_dict(raw.get("training", {})),
            controller=ControllerSection.from_dict(raw.get("controller", {})),
            features=FeatureSection.from_dict(raw.get("features", {})),
            replay=ReplaySection.from_dict(raw.get("replay", {})),
        )

    @classmethod
    def for_profile(cls, profile: str, **section_overrides: Mapping[str, Any]) -> "BuildConfig":
        return cls.from_dict({
            "model": {"profile": profile, **section_overrides.get("model", {})},
            "training": section_overrides.get("training", {}),
            "controller": section_overrides.get("controller", {}),
            "features": section_overrides.get("features", {}),
            "replay": section_overrides.get("replay", {}),
        })

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "model": self.model.to_dict(),
            "training": self.training.to_dict(),
            "controller": self.controller.to_dict(),
            "features": self.features.to_dict(),
            "replay": self.replay.to_dict(),
        }

    def identity(self) -> str:
        """Content identity over the fully resolved configuration."""
        return content_identity(self.to_dict())

    # -- translations into the accepted decoder's configuration ---------------

    def model_config(self):
        from bramastra_lab.model import ModelConfig

        return ModelConfig(
            vocab=self.model.vocab,
            width=self.model.width,
            layers=self.model.layers,
            heads=self.model.heads,
            ffn=self.model.ffn,
            max_seq=self.model.max_seq,
        )

    def parameter_count(self) -> int:
        """Exact analytic parameter count of the integrated model.

        The base count is the accepted decoder's exact formula (tied output
        projection). The action and value heads are bias-free linear probes on
        decoder hidden states, adding ``width`` parameters each; their
        parameters are counted separately and are never advertised as part of
        the base decoder's count.
        """
        from bramastra_lab.model import parameter_count as base_parameter_count

        base = base_parameter_count(self.model_config())
        heads = 2 * self.model.width  # action head + value head, both width->1
        return base + heads

    def base_decoder_parameter_count(self) -> int:
        return int(
            __import__("bramastra_lab.model", fromlist=["parameter_count"])
            .parameter_count(self.model_config())
        )


def load_config(raw: Mapping[str, Any]) -> BuildConfig:
    """Parse and validate a raw configuration mapping."""
    return BuildConfig.from_dict(raw)


def seed_everything(seed: int) -> None:
    """Initialize every RNG stream from one declared seed, deterministically.

    Call order is fixed (Python ``random``, NumPy when installed, then torch)
    so repeated initialization from the same seed reproduces identical
    streams. This only initializes generators; it trains nothing.
    """
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ConfigError("seed must be a nonnegative integer")
    import random

    random.seed(seed)
    try:
        import numpy as np
    except ImportError:
        np = None
    if np is not None:
        np.random.seed(seed % (2**32))
    import torch

    torch.manual_seed(seed)
