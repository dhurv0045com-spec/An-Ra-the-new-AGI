from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Literal, TypeVar, cast, get_args, get_origin, get_type_hints

import yaml


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


def _field_names(cls: type[object]) -> set[str]:
    return {field.name for field in fields(cls)}


def _coerce_path(value: object) -> Path | None:
    if value is None:
        return None
    return value if isinstance(value, Path) else Path(str(value))


def _coerce_value(value: object, annotation: object) -> object:
    origin = get_origin(annotation)
    args = get_args(annotation)
    if annotation is Path:
        return _coerce_path(value)
    if origin is tuple:
        if value is None:
            return ()
        return tuple(cast(Any, value))
    if origin is Literal and value not in args:
        allowed = ", ".join(repr(arg) for arg in args)
        raise ValueError(f"Expected one of {allowed}, got {value!r}")
    if origin in (type(None),):
        return value
    if origin is None and annotation in (int, float, str, bool):
        if value is None:
            return value
        return annotation(value)
    if origin is None:
        return value
    if type(None) in args and value is None:
        return None
    non_none = [arg for arg in args if arg is not type(None)]
    if len(non_none) == 1:
        return _coerce_value(value, non_none[0])
    return value


def _build_dataclass(cls: type[T], data: Mapping[str, object]) -> T:
    normalized = _as_plain_dict(data)
    known = _field_names(cls)
    unknown = sorted(set(normalized) - known)
    if unknown:
        joined = ", ".join(unknown)
        raise ValueError(f"Unknown {cls.__name__} field(s): {joined}")
    annotations = get_type_hints(cls)
    kwargs = {
        key: _coerce_value(value, annotations.get(key))
        for key, value in normalized.items()
    }
    return cls(**kwargs)  # type: ignore[arg-type]


def _dump_dataclass(value: object) -> dict[str, object]:
    return cast(dict[str, object], asdict(value))


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Validated model architecture configuration."""

    type: str = "causal_transformer_v2"
    vocab_size: int = 8192
    n_embd: int = 512
    n_layer: int = 8
    n_head: int = 8
    n_kv_head: int | None = 2
    d_ff: int | None = None
    block_size: int = 1024
    dropout: float = 0.1
    ffn_type: Literal["swiglu", "gelu"] = "swiglu"
    rope_base: float = 10_000.0
    tie_weights: bool = True
    pad_token_id: int = 0
    eos_token_id: int | None = None
    use_mod: bool = False
    mod_capacity: float = 0.5
    gradient_checkpointing: bool = False
    mod_layers: tuple[int, ...] = ()
    use_hal: bool = False

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> ModelConfig:
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
        return _build_dataclass(cls, normalized)

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("model.vocab_size must be positive")
        if self.n_embd <= 0 or self.n_layer <= 0 or self.n_head <= 0:
            raise ValueError("model dimensions must be positive")
        n_kv_head = self.n_kv_head or self.n_head
        if n_kv_head <= 0:
            raise ValueError("model.n_kv_head must be positive")
        if self.n_embd % self.n_head != 0:
            raise ValueError("model.n_embd must be divisible by model.n_head")
        if self.n_head % n_kv_head != 0:
            raise ValueError("model.n_head must be divisible by model.n_kv_head")
        if self.d_ff is not None and self.d_ff <= 0:
            raise ValueError("model.d_ff must be positive")
        if self.block_size <= 0:
            raise ValueError("model.block_size must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("model.dropout must be in [0, 1)")
        if self.rope_base <= 0:
            raise ValueError("model.rope_base must be positive")
        if self.pad_token_id < 0 or self.pad_token_id >= self.vocab_size:
            raise ValueError("model.pad_token_id must be smaller than model.vocab_size")
        if self.eos_token_id is not None and (
            self.eos_token_id < 0 or self.eos_token_id >= self.vocab_size
        ):
            raise ValueError("model.eos_token_id must be smaller than model.vocab_size")
        if not 0.0 < self.mod_capacity <= 1.0:
            raise ValueError("model.mod_capacity must be in (0, 1]")

    def dict(self) -> dict[str, object]:
        return _dump_dataclass(self)


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Validated training configuration."""

    type: str = "base_trainer"
    dataset_path: Path = Path("data/train.txt")
    val_dataset_path: Path = Path("data/val.txt")
    tokenizer_path: Path = Path("tokenizer.json")
    learning_rate: float = 3.0e-4
    min_lr: float = 3.0e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    max_grad_norm: float | None = None
    warmup_steps: int = 2000
    max_steps: int = 100_000
    lr_schedule: Literal["cosine", "linear", "constant"] = "cosine"
    batch_size: int = 16
    gradient_accumulation: int = 1
    seq_len: int = 512
    checkpoint_dir: Path = Path("checkpoints")
    checkpoint_every: int = 1000
    keep_last_n_checkpoints: int = 3
    resume_from: Path | None = None
    eval_every: int = 500
    eval_steps: int = 50
    log_every: int = 10
    log_dir: Path = Path("logs")
    seed: int = 42
    num_workers: int = 0
    objective: str = "cross_entropy"
    rlvr_enabled: bool = False
    star_enabled: bool = False

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> TrainingConfig:
        return _build_dataclass(cls, data)

    def __post_init__(self) -> None:
        for name in ("dataset_path", "val_dataset_path", "tokenizer_path", "checkpoint_dir", "log_dir"):
            object.__setattr__(self, name, _coerce_path(getattr(self, name)))
        object.__setattr__(self, "resume_from", _coerce_path(self.resume_from))
        if self.learning_rate <= 0.0:
            raise ValueError("training.learning_rate must be positive")
        if self.min_lr < 0.0 or self.min_lr > self.learning_rate:
            raise ValueError("training.min_lr must be less than or equal to training.learning_rate")
        if self.weight_decay < 0.0:
            raise ValueError("training.weight_decay must be non-negative")
        if not 0.0 < self.beta1 < 1.0 or not 0.0 < self.beta2 < 1.0:
            raise ValueError("training beta values must be in (0, 1)")
        if self.grad_clip < 0.0:
            raise ValueError("training.grad_clip must be non-negative")
        if self.max_grad_norm is not None and self.max_grad_norm < 0.0:
            raise ValueError("training.max_grad_norm must be non-negative")
        if self.warmup_steps < 0:
            raise ValueError("training.warmup_steps must be non-negative")
        if self.max_steps <= 0:
            raise ValueError("training.max_steps must be positive")
        for name in (
            "batch_size",
            "gradient_accumulation",
            "seq_len",
            "checkpoint_every",
            "eval_every",
            "eval_steps",
            "log_every",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"training.{name} must be positive")
        if self.keep_last_n_checkpoints < 0:
            raise ValueError("training.keep_last_n_checkpoints must be non-negative")
        if self.num_workers < 0:
            raise ValueError("training.num_workers must be non-negative")

    def dict(self) -> dict[str, object]:
        return _dump_dataclass(self)


@dataclass(frozen=True, slots=True)
class InferenceConfig:
    """Validated inference configuration."""

    strategy: str = "top_p"
    checkpoint_path: Path | None = None
    max_new_tokens: int = 200
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.95
    repetition_penalty: float = 1.1
    batch_size: int = 1
    turboquant: bool = False
    turboquant_bits: Literal[2, 4, 8] = 4

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> InferenceConfig:
        return _build_dataclass(cls, data)

    def __post_init__(self) -> None:
        object.__setattr__(self, "checkpoint_path", _coerce_path(self.checkpoint_path))
        if self.max_new_tokens <= 0:
            raise ValueError("inference.max_new_tokens must be positive")
        if self.temperature != 0.0 and self.temperature < 0.05:
            raise ValueError("inference.temperature must be 0.0 or at least 0.05")
        if self.top_k < 0:
            raise ValueError("inference.top_k must be non-negative")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("inference.top_p must be in (0, 1]")
        if self.repetition_penalty < 1.0:
            raise ValueError("inference.repetition_penalty must be at least 1.0")
        if self.batch_size <= 0:
            raise ValueError("inference.batch_size must be positive")

    def dict(self) -> dict[str, object]:
        return _dump_dataclass(self)


@dataclass(frozen=True, slots=True)
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
    def from_mapping(cls, data: Mapping[str, object]) -> LoggingConfig:
        return _build_dataclass(cls, data)

    def dict(self) -> dict[str, object]:
        return _dump_dataclass(self)


@dataclass(frozen=True, slots=True)
class HardwareConfig:
    """Validated hardware and performance configuration."""

    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    mixed_precision: bool = False
    compile_model: bool = False
    gradient_checkpointing: bool = False

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> HardwareConfig:
        return _build_dataclass(cls, data)

    def dict(self) -> dict[str, object]:
        return _dump_dataclass(self)


@dataclass(frozen=True, slots=True)
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
    def from_mapping(cls, data: Mapping[str, object]) -> PathsConfig:
        return _build_dataclass(cls, data)

    def __post_init__(self) -> None:
        for name in (
            "output_dir",
            "model_dir",
            "log_dir",
            "checkpoint_dir",
            "state_dir",
            "session_db",
            "metrics_path",
        ):
            object.__setattr__(self, name, _coerce_path(getattr(self, name)))
        for name in ("output_dir", "model_dir", "log_dir", "checkpoint_dir", "state_dir"):
            path = getattr(self, name)
            if path == Path("/content") or str(path).startswith("/content/drive"):
                raise ValueError(f"paths.{name} must not hardcode Colab or Google Drive paths")

    def dict(self) -> dict[str, object]:
        return _dump_dataclass(self)


@dataclass(frozen=True, slots=True)
class AnRaConfig:
    """Validated top-level AN-RA configuration."""

    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    experiment_name: str = "base"
    seed: int = 42

    @classmethod
    def from_yaml(cls, path: str | Path) -> AnRaConfig:
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
    def from_mapping(cls, data: Mapping[str, object]) -> AnRaConfig:
        normalized = _rename_keys(_as_plain_dict(data), {"train": "training"})

        known_sections = {
            "experiment_name",
            "hardware",
            "inference",
            "logging",
            "model",
            "paths",
            "seed",
            "training",
        }
        unknown_sections = sorted(set(normalized) - known_sections)
        if unknown_sections:
            joined = ", ".join(unknown_sections)
            raise ValueError(f"Unknown top-level configuration section(s): {joined}")

        def section(name: str) -> Mapping[str, object]:
            value = normalized.get(name, {})
            if not isinstance(value, Mapping):
                raise TypeError(f"config.{name} must be a mapping")
            return cast(Mapping[str, object], value)

        return cls(
            model=ModelConfig.from_mapping(section("model")),
            training=TrainingConfig.from_mapping(section("training")),
            inference=InferenceConfig.from_mapping(section("inference")),
            paths=PathsConfig.from_mapping(section("paths")),
            logging=LoggingConfig.from_mapping(section("logging")),
            hardware=HardwareConfig.from_mapping(section("hardware")),
            experiment_name=str(normalized.get("experiment_name", "base")),
            seed=int(cast(int | str, normalized.get("seed", 42))),
        )

    def __post_init__(self) -> None:
        if self.training.seq_len > self.model.block_size:
            raise ValueError("training.seq_len must be less than or equal to model.block_size")

    def dict(self) -> dict[str, object]:
        return _dump_dataclass(self)
