from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from pathlib import Path
from typing import Literal, Self, TypeVar, cast

import yaml
from pydantic import (
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_validator,
)
from pydantic.dataclasses import dataclass

ConfigMap = MutableMapping[str, object]
ConfigSource = str | Path | Mapping[str, object]
T = TypeVar("T")


def _as_plain_dict(value: Mapping[str, object]) -> ConfigMap:
    return {str(key): item for key, item in value.items()}


def _rename_keys(data: ConfigMap, aliases: Mapping[str, str]) -> ConfigMap:
    normalized = dict(data)
    for old_key, new_key in aliases.items():
        if old_key in normalized and new_key not in normalized:
            normalized[new_key] = normalized.pop(old_key)
    return normalized


def _validate_dataclass(cls: type[T], data: object) -> T:
    return TypeAdapter(cls).validate_python(data)


def _dump_dataclass(value: object) -> dict[str, object]:
    dumped = TypeAdapter(type(value)).dump_python(value, exclude_none=False)
    if not isinstance(dumped, dict):
        raise TypeError(f"Expected dataclass dump to be a dict, got {type(dumped).__name__}")
    return cast(dict[str, object], dumped)


@dataclass(frozen=True, slots=True, config=ConfigDict(extra="forbid", validate_assignment=True))
class ModelConfig:
    """Validated model architecture configuration."""

    type: str = "causal_transformer_v4"
    architecture_version: str = "anra_v4_rope_interleaved_v1"
    vocab_size: int = Field(default=8192, gt=0)
    n_embd: int = Field(default=512, gt=0)
    n_layer: int = Field(default=8, gt=0)
    n_head: int = Field(default=8, gt=0)
    n_kv_head: int | None = Field(default=2, gt=0)
    d_ff: int | None = Field(default=None, gt=0)
    block_size: int = Field(default=1024, gt=0)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    ffn_type: Literal["swiglu", "gelu"] = "swiglu"
    rope_base: float = Field(default=10_000.0, gt=0.0)
    rms_norm_eps: float = Field(default=1.0e-5, gt=0.0)
    tie_weights: bool = True
    pad_token_id: int = Field(default=0, ge=0)
    eos_token_id: int | None = Field(default=None, ge=0)
    use_mod: bool = False
    mod_layers: tuple[int, ...] = ()
    mod_capacity: float = Field(default=0.5, gt=0.0, le=1.0)
    use_hal: bool = False
    use_rim: bool = True
    use_dstp: bool = True
    use_qk_norm: bool = True
    sliding_window: int | None = Field(default=1024, gt=0)
    full_attention_every: int = Field(default=4, ge=0)
    use_mtp: bool = False
    use_moe: bool = False
    initialization_scheme: str = "depth_scaled_residual_v1"
    base_seq_len: int = Field(default=512, gt=0)
    target_seq_len: int = Field(default=2048, gt=0)
    gradient_checkpointing: bool = False

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> Self:
        normalized = _rename_keys(
            _as_plain_dict(data),
            {
                "d_model": "n_embd",
                "num_layers": "n_layer",
                "num_heads": "n_head",
                "num_kv_heads": "n_kv_head",
                "max_seq_len": "block_size",
                "dropout_rate": "dropout",
            },
        )
        return _validate_dataclass(cls, normalized)

    @model_validator(mode="after")
    def validate_architecture(self) -> Self:
        n_kv_head = self.n_kv_head or self.n_head
        if self.n_embd % self.n_head != 0:
            raise ValueError("model.n_embd must be divisible by model.n_head")
        if self.n_head % n_kv_head != 0:
            raise ValueError("model.n_head must be divisible by model.n_kv_head")
        if self.pad_token_id >= self.vocab_size:
            raise ValueError("model.pad_token_id must be smaller than model.vocab_size")
        if self.eos_token_id is not None and self.eos_token_id >= self.vocab_size:
            raise ValueError("model.eos_token_id must be smaller than model.vocab_size")
        if len(set(self.mod_layers)) != len(self.mod_layers):
            raise ValueError("model.mod_layers cannot contain duplicates")
        if any(layer < 0 or layer >= self.n_layer for layer in self.mod_layers):
            raise ValueError("model.mod_layers must reference existing zero-based layers")
        if self.d_ff is not None and self.d_ff % 64 != 0:
            raise ValueError("model.d_ff must be divisible by 64")
        return self

    def dict(self) -> dict[str, object]:
        return _dump_dataclass(self)


@dataclass(frozen=True, slots=True, config=ConfigDict(extra="forbid", validate_assignment=True))
class TrainingConfig:
    """Validated training configuration."""

    type: str = "base_trainer"
    dataset_path: Path = Path("data/train.txt")
    val_dataset_path: Path = Path("data/val.txt")
    tokenizer_path: Path = Path("tokenizer.json")
    learning_rate: float = Field(default=3.0e-4, gt=0.0)
    min_lr: float = Field(default=3.0e-5, ge=0.0)
    weight_decay: float = Field(default=0.1, ge=0.0)
    beta1: float = Field(default=0.9, gt=0.0, lt=1.0)
    beta2: float = Field(default=0.95, gt=0.0, lt=1.0)
    grad_clip: float = Field(default=1.0, ge=0.0)
    max_grad_norm: float | None = Field(default=None, ge=0.0)
    warmup_steps: int = Field(default=2000, ge=0)
    max_steps: int = Field(default=100_000, gt=0)
    lr_schedule: Literal["cosine", "linear", "constant"] = "cosine"
    batch_size: int = Field(default=16, gt=0)
    gradient_accumulation: int = Field(default=1, gt=0)
    seq_len: int = Field(default=512, gt=0)
    checkpoint_dir: Path = Path("checkpoints")
    checkpoint_every: int = Field(default=1000, gt=0)
    keep_last_n_checkpoints: int = Field(default=3, ge=0)
    resume_from: Path | None = None
    eval_every: int = Field(default=500, gt=0)
    eval_steps: int = Field(default=50, gt=0)
    log_every: int = Field(default=10, gt=0)
    log_dir: Path = Path("logs")
    seed: int = 42
    num_workers: int = Field(default=0, ge=0)
    objective: str = "cross_entropy"
    rlvr_enabled: bool = False
    star_enabled: bool = False
    entropy_coef: float = Field(default=0.01, ge=0.0)

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> Self:
        return _validate_dataclass(cls, _as_plain_dict(data))

    @model_validator(mode="after")
    def validate_schedule(self) -> Self:
        if self.min_lr > self.learning_rate:
            raise ValueError("training.min_lr must be less than or equal to training.learning_rate")
        return self

    def dict(self) -> dict[str, object]:
        return _dump_dataclass(self)


@dataclass(frozen=True, slots=True, config=ConfigDict(extra="forbid", validate_assignment=True))
class InferenceConfig:
    """Validated inference configuration."""

    strategy: str = "top_p"
    checkpoint_path: Path | None = None
    max_new_tokens: int = Field(default=200, gt=0)
    temperature: float = Field(default=0.8, ge=0.0)
    top_k: int = Field(default=50, ge=0)
    top_p: float = Field(default=0.95, gt=0.0, le=1.0)
    repetition_penalty: float = Field(default=1.1, ge=1.0)
    batch_size: int = Field(default=1, gt=0)
    turboquant: bool = False
    turboquant_bits: Literal[4, 8] = 4

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> Self:
        return _validate_dataclass(cls, _as_plain_dict(data))

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, value: float) -> float:
        if value == 0.0:
            return value
        if value < 0.05:
            raise ValueError("inference.temperature must be 0.0 or at least 0.05")
        return value

    def dict(self) -> dict[str, object]:
        return _dump_dataclass(self)


@dataclass(frozen=True, slots=True, config=ConfigDict(extra="forbid", validate_assignment=True))
class LoggingConfig:
    """Validated logging configuration."""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_to_file: bool = True
    log_to_console: bool = True
    use_wandb: bool = False
    wandb_project: str = "anra"
    wandb_entity: str | None = None
    wandb_run_name: str | None = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> Self:
        return _validate_dataclass(cls, _as_plain_dict(data))

    def dict(self) -> dict[str, object]:
        return _dump_dataclass(self)


@dataclass(frozen=True, slots=True, config=ConfigDict(extra="forbid", validate_assignment=True))
class HardwareConfig:
    """Validated hardware and performance configuration."""

    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    mixed_precision: bool = False
    compile_model: bool = False
    gradient_checkpointing: bool = False

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> Self:
        return _validate_dataclass(cls, _as_plain_dict(data))

    def dict(self) -> dict[str, object]:
        return _dump_dataclass(self)


@dataclass(frozen=True, slots=True, config=ConfigDict(extra="forbid", validate_assignment=True))
class PathsConfig:
    """Validated filesystem path configuration."""

    output_dir: Path = Path("output")
    model_dir: Path = Path("output/model")
    log_dir: Path = Path("output/logs")
    checkpoint_dir: Path = Path("output/checkpoints")
    state_dir: Path = Path("state")
    session_db: Path = Path("state/sessions.sqlite")
    metrics_path: Path = Path("output/metrics/events.jsonl")

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> Self:
        return _validate_dataclass(cls, _as_plain_dict(data))

    @model_validator(mode="after")
    def validate_paths(self) -> Self:
        for name in ("output_dir", "model_dir", "log_dir", "checkpoint_dir", "state_dir"):
            path = getattr(self, name)
            if path == Path("/content") or str(path).startswith("/content/drive"):
                raise ValueError(f"paths.{name} must not hardcode Colab or Google Drive paths")
        return self

    def dict(self) -> dict[str, object]:
        return _dump_dataclass(self)


@dataclass(frozen=True, slots=True, config=ConfigDict(extra="forbid", validate_assignment=True))
class AnRaConfig:
    """Validated top-level AN-RA configuration."""

    experiment_name: str = "base"
    seed: int = 42
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Self:
        config_path = Path(path)
        try:
            raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Configuration file not found: {config_path}") from exc
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid YAML in configuration file {config_path}: {exc}") from exc

        if raw is None:
            raw = {}
        if not isinstance(raw, Mapping):
            raise TypeError(f"Configuration root must be a mapping, got {type(raw).__name__}")
        return cls.from_mapping(cast(Mapping[str, object], raw))

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> Self:
        normalized = _rename_keys(_as_plain_dict(data), {"train": "training"})

        model_data = normalized.get("model", {})
        training_data = normalized.get("training", {})
        inference_data = normalized.get("inference", {})
        paths_data = normalized.get("paths", {})
        logging_data = normalized.get("logging", {})
        hardware_data = normalized.get("hardware", {})

        if not isinstance(model_data, Mapping):
            raise TypeError("config.model must be a mapping")
        if not isinstance(training_data, Mapping):
            raise TypeError("config.training must be a mapping")
        if not isinstance(inference_data, Mapping):
            raise TypeError("config.inference must be a mapping")
        if not isinstance(paths_data, Mapping):
            raise TypeError("config.paths must be a mapping")
        if not isinstance(logging_data, Mapping):
            raise TypeError("config.logging must be a mapping")
        if not isinstance(hardware_data, Mapping):
            raise TypeError("config.hardware must be a mapping")

        known_sections = {
            "experiment_name",
            "seed",
            "model",
            "training",
            "inference",
            "paths",
            "logging",
            "hardware",
        }
        unknown_sections = sorted(set(normalized) - known_sections)
        if unknown_sections:
            joined = ", ".join(unknown_sections)
            raise ValueError(f"Unknown top-level configuration section(s): {joined}")

        normalized_training = _as_plain_dict(cast(Mapping[str, object], training_data))
        if "seq_len" not in normalized_training:
            normalized_training["seq_len"] = ModelConfig.from_mapping(
                cast(Mapping[str, object], model_data)
            ).block_size

        try:
            return _validate_dataclass(
                cls,
                {
                    "experiment_name": str(normalized.get("experiment_name", "base")),
                    "seed": TypeAdapter(int).validate_python(normalized.get("seed", 42)),
                    "model": ModelConfig.from_mapping(cast(Mapping[str, object], model_data)),
                    "training": TrainingConfig.from_mapping(normalized_training),
                    "inference": InferenceConfig.from_mapping(
                        cast(Mapping[str, object], inference_data)
                    ),
                    "paths": PathsConfig.from_mapping(cast(Mapping[str, object], paths_data)),
                    "logging": LoggingConfig.from_mapping(cast(Mapping[str, object], logging_data)),
                    "hardware": HardwareConfig.from_mapping(
                        cast(Mapping[str, object], hardware_data)
                    ),
                },
            )
        except ValidationError as exc:
            raise ValueError(f"Invalid AN-RA configuration: {exc}") from exc

    @model_validator(mode="after")
    def validate_cross_section(self) -> Self:
        if self.training.seq_len > self.model.block_size:
            raise ValueError("training.seq_len must be less than or equal to model.block_size")
        return self

    def dict(self) -> dict[str, object]:
        return _dump_dataclass(self)

    def model_dump(self) -> Mapping[str, object]:
        """Pydantic-style serialization used by checkpoint writers."""
        return self.dict()
