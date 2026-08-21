"""Validated runtime execution for the exact dense An-Ra V4 neural model."""

from __future__ import annotations

import uuid
from importlib.resources import files
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn.functional as F

from .checkpoint import load_core_checkpoint
from .config import CANONICAL_CONFIG, CoreConfig
from .contracts import (
    ArchitectureIdentity,
    CapabilitySet,
    CheckpointIdentity,
    ExecutionProfile,
    PredictionResult,
    RepresentationIdentity,
    RuntimeIdentity,
)
from .errors import (
    CoreError,
    RepresentationIncompatibleError,
    ResourceExhaustionError,
    StateIncompatibleError,
    UnexpectedExecutionFault,
    UnsupportedCapabilityError,
    UnsupportedProfileError,
)
from .model import AnRaCore
from .state import CoreState
from .tokenizer import V4Tokenizer

_DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}
_PROFILE_CATEGORIES = frozenset({"exact", "optimized", "approximate"})


class CoreExecutor:
    """Inference executor owning device, precision, and opaque incremental state.

    The neural model remains a pure differentiable function. This class validates
    state ownership and commits logical state only after a complete call succeeds.
    """

    def __init__(
        self,
        model: AnRaCore,
        *,
        tokenizer: V4Tokenizer | None = None,
        checkpoint_identity: CheckpointIdentity | None = None,
        device: str = "cpu",
        dtype: str = "float32",
        profile_category: Literal["exact", "optimized", "approximate"] = "exact",
        enable_telemetry: bool = False,
    ) -> None:
        if dtype not in _DTYPES:
            raise UnsupportedProfileError(
                f"Unsupported dtype {dtype!r}", details={"supported": sorted(_DTYPES)}
            )
        if profile_category not in _PROFILE_CATEGORIES:
            raise UnsupportedProfileError(
                f"Unsupported profile category {profile_category!r}",
                details={"supported": sorted(_PROFILE_CATEGORIES)},
            )
        try:
            resolved_device = torch.device(device)
        except (RuntimeError, ValueError) as exc:
            raise UnsupportedProfileError(
                f"Invalid device {device!r}", details={"device": device}
            ) from exc
        if resolved_device.type == "cuda" and not torch.cuda.is_available():
            raise UnsupportedProfileError("CUDA was requested but is unavailable")
        if resolved_device.type not in {"cpu", "cuda"}:
            raise UnsupportedProfileError(
                f"Unsupported device type {resolved_device.type!r}",
                details={"supported": ["cpu", "cuda"]},
            )
        if resolved_device.type == "cpu" and dtype == "float16":
            raise UnsupportedProfileError("float16 execution is not supported on CPU")
        if profile_category == "exact" and dtype != "float32":
            raise UnsupportedProfileError(
                "The exact profile requires float32; lower precision is approximate"
            )

        self.model = model.eval()
        self.tokenizer = tokenizer
        self.checkpoint_identity = checkpoint_identity or CheckpointIdentity(
            checkpoint_sha256=None,
            source_path=None,
            global_step=None,
            training_stage=None,
            source_commit=None,
            tokenizer_contract_valid=False,
        )
        self.device = resolved_device
        self.dtype_str = dtype
        self.torch_dtype = _DTYPES[dtype]
        self.enable_telemetry = enable_telemetry
        self._owner_id = str(uuid.uuid4())
        self._ephemeral_parameter_id = f"ephemeral:{uuid.uuid4()}"

        try:
            self.model.to(device=self.device, dtype=self.torch_dtype)
        except (RuntimeError, TypeError) as exc:
            raise UnsupportedProfileError(
                "Model cannot be materialized under the requested execution profile",
                details={"device": str(self.device), "dtype": dtype},
            ) from exc
        self.model.lm_head.weight = self.model.token_embedding_table.weight

        deterministic = (
            profile_category == "exact"
            and self.device.type == "cpu"
            and self.torch_dtype == torch.float32
        )
        self.execution_profile = ExecutionProfile(
            profile_id=(
                f"anra-v4-executor-v2:{profile_category}:{self.device}:{dtype}"
            ),
            category=profile_category,
            device=str(self.device),
            dtype=dtype,
            deterministic=deterministic,
            sliding_window_enabled=True,
        )
        self.runtime_identity = RuntimeIdentity()
        self.capabilities = CapabilitySet(
            supports_full_forward=True,
            supports_incremental_decode=True,
            supports_state_fork=True,
            supports_state_reset=True,
            supports_state_serialization=False,
            supports_quantization=False,
            supports_multi_device_sharding=False,
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        tokenizer_path: str | Path | None = None,
        config: CoreConfig = CANONICAL_CONFIG,
        device: str = "cpu",
        dtype: str = "float32",
        profile_category: Literal["exact", "optimized", "approximate"] = "exact",
        enable_telemetry: bool = False,
        allow_legacy_unverified: bool = False,
    ) -> CoreExecutor:
        model, metadata, identity = load_core_checkpoint(
            checkpoint_path,
            config=config,
            legacy_unverified=allow_legacy_unverified,
        )
        if tokenizer_path is None:
            tokenizer_path = files("anra_core.assets").joinpath("tokenizer_v4_32k.json")
        tokenizer = V4Tokenizer.load(tokenizer_path)
        contract = metadata.get("tokenizer_contract")
        if contract is None and not allow_legacy_unverified:
            raise RepresentationIncompatibleError(
                "Checkpoint has no tokenizer contract; strict execution cannot bind representation"
            )
        if contract is not None:
            tokenizer.assert_checkpoint_contract(contract)
        return cls(
            model,
            tokenizer=tokenizer,
            checkpoint_identity=identity,
            device=device,
            dtype=dtype,
            profile_category=profile_category,
            enable_telemetry=enable_telemetry,
        )

    def architecture_identity(self) -> ArchitectureIdentity:
        cfg = self.model.config
        return ArchitectureIdentity(
            architecture_version=cfg.architecture_version,
            vocab_size=cfg.vocab_size,
            d_model=cfg.d_model,
            n_layers=cfg.n_layers,
            n_heads=cfg.n_heads,
            n_kv_heads=cfg.n_kv_heads,
            head_dim=cfg.head_dim,
            d_ff=cfg.d_ff,
            block_size=cfg.block_size,
            rope_base=cfg.rope_base,
            sliding_window=cfg.sliding_window,
            full_attention_every=cfg.full_attention_every,
            qk_norm=cfg.qk_norm,
            dense_parameter_count=cfg.dense_parameter_count,
            architecture_sha256=cfg.architecture_sha256,
        )

    def representation_identity(self) -> RepresentationIdentity | None:
        if self.tokenizer is None:
            return None
        identity = self.tokenizer.identity()
        return RepresentationIdentity(
            schema_version=int(identity["schema_version"]),
            vocab_size=int(identity["vocab_size"]),
            vocabulary_sha256=str(identity["vocabulary_sha256"]),
            probe_count=int(identity["probe_count"]),
            probe_sha256=str(identity["probe_sha256"]),
        )

    def _representation_id(self) -> str | None:
        identity = self.representation_identity()
        if identity is None:
            return None
        return f"v{identity.schema_version}:{identity.vocabulary_sha256}:{identity.probe_sha256}"

    def _parameter_id(self) -> str:
        parameter_sha = getattr(self.checkpoint_identity, "parameter_sha256", None)
        return parameter_sha or self.checkpoint_identity.checkpoint_sha256 or self._ephemeral_parameter_id

    def create_state(self, *, batch_size: int = 1, capacity: int | None = None) -> CoreState:
        if batch_size <= 0:
            raise StateIncompatibleError(
                "State batch size must be positive", details={"batch_size": batch_size}
            )
        requested_capacity = capacity or self.model.config.block_size
        if not 1 <= requested_capacity <= self.model.config.block_size:
            raise StateIncompatibleError(
                "State capacity must be within the model context limit",
                details={
                    "capacity": requested_capacity,
                    "model_limit": self.model.config.block_size,
                },
            )
        return CoreState(
            _owner_id=self._owner_id,
            _architecture_id=self.model.config.architecture_sha256,
            _parameter_id=self._parameter_id(),
            _representation_id=self._representation_id(),
            _execution_profile_id=self.execution_profile.profile_id,
            _batch_size=batch_size,
            _capacity=requested_capacity,
            _n_layers=self.model.config.n_layers,
            _n_kv_heads=self.model.config.n_kv_heads,
            _head_dim=self.model.config.head_dim,
        )

    def _validate_state_compatibility(self, state: CoreState) -> None:
        state._assert_active()
        mismatches: dict[str, dict[str, object]] = {}
        expected = {
            "owner_id": self._owner_id,
            "architecture_id": self.model.config.architecture_sha256,
            "parameter_id": self._parameter_id(),
            "representation_id": self._representation_id(),
            "execution_profile_id": self.execution_profile.profile_id,
        }
        actual = {
            "owner_id": state._owner_id,
            "architecture_id": state.architecture_id,
            "parameter_id": state.parameter_id,
            "representation_id": state.representation_id,
            "execution_profile_id": state.execution_profile_id,
        }
        for name, expected_value in expected.items():
            if actual[name] != expected_value:
                mismatches[name] = {"expected": expected_value, "actual": actual[name]}
        if state.capacity > self.model.config.block_size:
            mismatches["capacity"] = {
                "expected_max": self.model.config.block_size,
                "actual": state.capacity,
            }
        if mismatches:
            raise StateIncompatibleError(
                "State does not belong to this executor/model/profile",
                details={"mismatches": mismatches, "state_id": state.state_id},
            )

    def _validate_ids(self, token_ids: torch.Tensor, *, state: CoreState | None) -> torch.Tensor:
        if not isinstance(token_ids, torch.Tensor) or token_ids.ndim != 2:
            raise RepresentationIncompatibleError("token_ids must have shape [batch, sequence]")
        if token_ids.shape[1] <= 0:
            raise RepresentationIncompatibleError("token_ids cannot contain an empty sequence")
        if token_ids.dtype not in {
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }:
            raise RepresentationIncompatibleError(
                "token_ids must use an integer dtype", details={"dtype": str(token_ids.dtype)}
            )
        if state is not None and token_ids.shape[0] != state.batch_size:
            raise StateIncompatibleError(
                "Input batch size does not match state batch size",
                details={"input_batch": token_ids.shape[0], "state_batch": state.batch_size},
            )
        if token_ids.numel():
            minimum = int(token_ids.min().item())
            maximum = int(token_ids.max().item())
            if minimum < 0 or maximum >= self.model.config.vocab_size:
                raise RepresentationIncompatibleError(
                    "token_ids contain values outside the active vocabulary",
                    details={"minimum": minimum, "maximum": maximum},
                )
        return token_ids.to(device=self.device, dtype=torch.long)

    def _compute_telemetry(self, logits: torch.Tensor) -> dict[str, float]:
        last_logits = logits[:, -1, :].float()
        probabilities = F.softmax(last_logits, dim=-1)
        log_probabilities = F.log_softmax(last_logits, dim=-1)
        entropy = -(probabilities * log_probabilities).sum(dim=-1).mean().item()
        top_values = last_logits.topk(k=2, dim=-1).values
        return {
            "logit_entropy": float(entropy),
            "peak_logit": float(top_values[:, 0].mean().item()),
            "top2_margin": float((top_values[:, 0] - top_values[:, 1]).mean().item()),
            "min_logit": float(last_logits.min(dim=-1).values.mean().item()),
        }

    def _prediction(self, logits: torch.Tensor, *, sequence_length: int) -> PredictionResult:
        metadata: dict[str, Any] = {}
        if self.enable_telemetry:
            metadata["telemetry"] = self._compute_telemetry(logits)
        return PredictionResult(
            logits=logits,
            sequence_length=sequence_length,
            execution_profile_id=self.execution_profile.profile_id,
            metadata=metadata,
        )

    def _translate_fault(self, operation: str, exc: Exception) -> CoreError:
        if isinstance(exc, CoreError):
            return exc
        if isinstance(exc, torch.OutOfMemoryError) or "out of memory" in str(exc).lower():
            return ResourceExhaustionError(
                f"Core {operation} exhausted execution memory",
                details={"device": str(self.device), "dtype": self.dtype_str},
            )
        return UnexpectedExecutionFault(
            f"Core {operation} failed unexpectedly",
            details={"exception_type": type(exc).__name__, "reason": str(exc)},
        )

    @torch.inference_mode()
    def forward(self, token_ids: torch.Tensor, state: CoreState | None = None) -> PredictionResult:
        if state is None:
            ids = self._validate_ids(token_ids, state=None)
            try:
                logits = self.model(ids)
                return self._prediction(logits, sequence_length=ids.shape[1])
            except Exception as exc:
                raise self._translate_fault("forward", exc) from exc
        self._validate_state_compatibility(state)
        ids = self._validate_ids(token_ids, state=state)
        return self._incremental(ids, state=state, chunk_size=None)

    @torch.inference_mode()
    def _incremental(
        self,
        ids: torch.Tensor,
        *,
        state: CoreState,
        chunk_size: int | None,
    ) -> PredictionResult:
        state._check_capacity(int(ids.shape[1]))
        caches: list[tuple[torch.Tensor, torch.Tensor]]
        try:
            # Buffer allocation is part of the fault-translated boundary: an
            # OOM during lazy state storage must surface as a typed Core
            # resource error, not a raw torch.OutOfMemoryError.
            state._ensure_buffers(device=self.device, dtype=self.torch_dtype)
            caches = state._cache_buffers()
        except Exception as exc:
            raise self._translate_fault("state allocation", exc) from exc
        size = int(ids.shape[1]) if chunk_size is None else int(chunk_size)
        if size <= 0:
            raise StateIncompatibleError("chunk_size must be positive")
        pieces: list[torch.Tensor] = []
        start_position = state.current_length
        try:
            for offset in range(0, ids.shape[1], size):
                chunk = ids[:, offset : offset + size]
                logits = self.model.forward_incremental(
                    chunk,
                    cache_buffers=caches,
                    start_pos=start_position + offset,
                )
                pieces.append(logits)
            combined = torch.cat(pieces, dim=1)
            result = self._prediction(
                combined, sequence_length=state.current_length + ids.shape[1]
            )
            state._commit(ids)
            return result
        except Exception as exc:
            # Cache writes beyond current_length are uncommitted scratch. The
            # prior logical prefix remains valid and a retry overwrites them.
            raise self._translate_fault("incremental execution", exc) from exc

    @torch.inference_mode()
    def prefill(
        self,
        token_ids: torch.Tensor,
        state: CoreState,
        *,
        chunk_size: int | None = None,
    ) -> PredictionResult:
        self._validate_state_compatibility(state)
        if state.current_length:
            raise StateIncompatibleError(
                "Cannot prefill a non-empty state; reset or create a new state",
                details={"current_length": state.current_length},
            )
        ids = self._validate_ids(token_ids, state=state)
        return self._incremental(ids, state=state, chunk_size=chunk_size)

    @torch.inference_mode()
    def forward_step(self, token_id: int | torch.Tensor, state: CoreState) -> PredictionResult:
        self._validate_state_compatibility(state)
        if isinstance(token_id, int):
            ids = torch.tensor([[token_id]], dtype=torch.long)
        elif isinstance(token_id, torch.Tensor):
            if token_id.ndim == 1:
                ids = token_id[:, None]
            elif token_id.ndim == 2:
                ids = token_id
            else:
                raise RepresentationIncompatibleError(
                    "forward_step token tensor must be rank 1 or 2"
                )
        else:
            raise RepresentationIncompatibleError("forward_step requires an integer token")
        if ids.shape[1] != 1:
            raise RepresentationIncompatibleError("forward_step accepts exactly one token per row")
        ids = self._validate_ids(ids, state=state)
        return self._incremental(ids, state=state, chunk_size=1)

    def reset_state(self, state: CoreState) -> None:
        self._validate_state_compatibility(state)
        state._reset()

    def rollback_state(self, state: CoreState, target_length: int) -> None:
        self._validate_state_compatibility(state)
        try:
            state._truncate(target_length)
        except ValueError as exc:
            raise StateIncompatibleError(
                f"Invalid rollback target length: {exc}",
                details={
                    "state_id": state.state_id,
                    "current_length": state.current_length,
                    "target_length": target_length,
                },
            ) from exc

    def fork_state(self, state: CoreState) -> CoreState:
        self._validate_state_compatibility(state)
        try:
            return state._fork()
        except torch.OutOfMemoryError as exc:
            raise ResourceExhaustionError(
                "Core state fork exhausted execution memory",
                details={"device": str(self.device), "dtype": self.dtype_str},
            ) from exc

    def release_state(self, state: CoreState) -> None:
        self._validate_state_compatibility(state)
        state._release()

    def serialize_state(self, state: CoreState) -> bytes:
        self._validate_state_compatibility(state)
        raise UnsupportedCapabilityError(
            "Portable state serialization is not supported by state schema v2"
        )

    def describe(self) -> dict[str, Any]:
        return {
            "runtime": self.runtime_identity.to_dict(),
            "architecture": self.architecture_identity().to_dict(),
            "checkpoint": self.checkpoint_identity.to_dict(),
            "representation": (
                self.representation_identity().to_dict() if self.tokenizer else None
            ),
            "execution_profile": self.execution_profile.to_dict(),
            "capabilities": self.capabilities.to_dict(),
            "batching": {"mode": "homogeneous", "ragged": False},
        }
