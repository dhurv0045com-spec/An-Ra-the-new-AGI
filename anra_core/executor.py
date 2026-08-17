"""Core Executor: Device placement, precision, lifecycle, and execution execution profiles."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import torch

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
    ContextOverflowError,
    StateIncompatibleError,
    StateReleasedError,
    UnsupportedProfileError,
)
from .model import AnRaCore
from .state import CoreState
from .tokenizer import V4Tokenizer


class CoreExecutor:
    """The runtime engine executing the An-Ra neural model.

    Decouples device management, execution profiles, and incremental KV state
    from the mathematical neural definition.
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
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.checkpoint_identity = checkpoint_identity or CheckpointIdentity(
            checkpoint_sha256=None,
            source_path=None,
            global_step=None,
            training_stage=None,
            source_commit=None,
            tokenizer_contract_valid=False,
        )
        self.device = torch.device(device)
        self.dtype_str = dtype
        self.torch_dtype = getattr(torch, dtype, torch.float32)

        self.model.to(device=self.device, dtype=self.torch_dtype)
        # Ensure tied weights pointer is intact after device transfer
        self.model.lm_head.weight = self.model.token_embedding_table.weight

        self.execution_profile = ExecutionProfile(
            profile_id=f"{profile_category}-{device}-{dtype}",
            category=profile_category,
            device=str(device),
            dtype=dtype,
            deterministic=True,
            sliding_window_enabled=True,
        )
        self.runtime_identity = RuntimeIdentity()
        self.capabilities = CapabilitySet()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        tokenizer_path: str | Path | None = None,
        config: CoreConfig = CANONICAL_CONFIG,
        device: str = "cpu",
        dtype: str = "float32",
    ) -> CoreExecutor:
        model, metadata, ckpt_identity = load_core_checkpoint(checkpoint_path, config=config)
        tokenizer = None
        if tokenizer_path is not None:
            tokenizer = V4Tokenizer.load(tokenizer_path)
            contract = metadata.get("tokenizer_contract")
            if contract:
                tokenizer.assert_checkpoint_contract(contract)
        return cls(
            model,
            tokenizer=tokenizer,
            checkpoint_identity=ckpt_identity,
            device=device,
            dtype=dtype,
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
        )

    def representation_identity(self) -> RepresentationIdentity | None:
        if self.tokenizer is None:
            return None
        ident = self.tokenizer.identity()
        return RepresentationIdentity(
            schema_version=int(ident["schema_version"]),
            vocab_size=int(ident["vocab_size"]),
            vocabulary_sha256=str(ident["vocabulary_sha256"]),
            probe_count=int(ident["probe_count"]),
            probe_sha256=str(ident["probe_sha256"]),
        )

    def create_state(self, *, capacity: int | None = None) -> CoreState:
        cap = capacity if capacity is not None else self.model.config.block_size
        return CoreState(
            architecture_version=self.model.config.architecture_version,
            checkpoint_id=self.checkpoint_identity.checkpoint_sha256 or "unpinned-weights",
            execution_profile_id=self.execution_profile.profile_id,
            capacity=cap,
            current_length=0,
            _kv_cache=[None] * self.model.config.n_layers,
        )

    def _validate_state_compatibility(self, state: CoreState) -> None:
        state.assert_active()
        if state.architecture_version != self.model.config.architecture_version:
            raise StateIncompatibleError(
                f"State architecture {state.architecture_version} does not match model {self.model.config.architecture_version}",
                details={"state_arch": state.architecture_version, "model_arch": self.model.config.architecture_version},
            )

    @torch.inference_mode()
    def forward(
        self,
        token_ids: torch.Tensor,
        state: CoreState | None = None,
    ) -> PredictionResult:
        """Execute forward pass for full sequence or incremental step."""
        if state is not None:
            self._validate_state_compatibility(state)

        ids = token_ids.to(device=self.device, dtype=torch.long)
        logits = self.model(ids, state=state)
        current_len = state.current_length if state is not None else ids.shape[1]

        return PredictionResult(
            logits=logits,
            sequence_length=current_len,
            execution_profile_id=self.execution_profile.profile_id,
        )

    @torch.inference_mode()
    def prefill(self, token_ids: torch.Tensor, state: CoreState) -> PredictionResult:
        """Process an initial prompt into the state cache and return the final logits."""
        self._validate_state_compatibility(state)
        if state.current_length > 0:
            raise StateIncompatibleError(
                "Cannot prefill into a state that already contains tokens. Call reset() first.",
                details={"current_length": state.current_length},
            )
        return self.forward(token_ids, state=state)

    @torch.inference_mode()
    def forward_step(self, token_id: int | torch.Tensor, state: CoreState) -> PredictionResult:
        """Process a single token step using the existing incremental state cache."""
        self._validate_state_compatibility(state)
        if isinstance(token_id, int):
            ids = torch.tensor([[token_id]], device=self.device, dtype=torch.long)
        elif isinstance(token_id, torch.Tensor):
            ids = token_id if token_id.ndim == 2 else token_id.view(1, -1)
            ids = ids.to(device=self.device, dtype=torch.long)
        else:
            raise ValueError("token_id must be an int or a 1D/2D Tensor")

        return self.forward(ids, state=state)

    def reset_state(self, state: CoreState) -> None:
        self._validate_state_compatibility(state)
        state.reset()

    def fork_state(self, state: CoreState) -> CoreState:
        self._validate_state_compatibility(state)
        return state.fork()

    def release_state(self, state: CoreState) -> None:
        state.release()

    def describe(self) -> dict[str, Any]:
        return {
            "runtime": self.runtime_identity.to_dict(),
            "architecture": self.architecture_identity().to_dict(),
            "checkpoint": self.checkpoint_identity.to_dict(),
            "representation": self.representation_identity().to_dict() if self.tokenizer else None,
            "execution_profile": self.execution_profile.to_dict(),
            "capabilities": self.capabilities.to_dict(),
        }
