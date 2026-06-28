"""Cross-scale inheritance and deterministic progressive model growth."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.nn import functional as F  # noqa: N812 - canonical PyTorch alias


@dataclass(frozen=True)
class GrowthReport:
    schema_version: int
    generated_at: float
    source_layers: int
    target_layers: int
    source_width: int
    target_width: int
    copied_tensors: int
    identity_layers: tuple[int, ...]
    parity_cosine: float | None = None
    parity_max_error: float | None = None
    source_checkpoint_sha256: str = ""


class GrowthAlignmentController:
    """Teacher alignment and progressive unfreezing for an expanded child."""

    def __init__(
        self,
        source: object,
        target: object,
        *,
        identity_layers: Iterable[int],
        new_only_steps: int = 1_000,
        alignment_steps: int = 5_000,
    ) -> None:
        self.source = source
        self.target = target
        self.identity_layers = tuple(int(value) for value in identity_layers)
        self.new_only_steps = int(new_only_steps)
        self.alignment_steps = int(alignment_steps)
        for parameter in self.source.parameters():
            parameter.requires_grad_(False)
        self.source.eval()
        self._active_names = {name for name, _ in self.target.named_parameters()}

    def configure_trainable_parameters(self, step: int) -> dict[str, int]:
        step = max(0, int(step))
        if step >= self.alignment_steps:
            for parameter in self.target.parameters():
                parameter.requires_grad_(True)
            self._active_names = {name for name, _ in self.target.named_parameters()}
            return {"trainable": sum(p.numel() for p in self.target.parameters()), "phase": 2}
        active_names: set[str] = set()
        for parameter in self.target.parameters():
            parameter.requires_grad_(True)
        for name, _parameter in self.target.named_parameters():
            inserted = any(
                name.startswith(f"blocks.{layer}.") or name.startswith(f"rim_modules.{layer}.")
                for layer in self.identity_layers
            )
            identity_surface = any(
                token in name
                for token in (
                    "token_embedding",
                    "esv_module",
                    "residual_depth_logits",
                )
            )
            if inserted or identity_surface:
                active_names.add(name)
        phase = 0
        if step >= self.new_only_steps:
            phase = 1
            inherited_fraction = (step - self.new_only_steps + 1) / max(
                1,
                self.alignment_steps - self.new_only_steps,
            )
            inherited_layers = max(
                1,
                round(self.target.n_layer * min(1.0, inherited_fraction)),
            )
            for name, _parameter in self.target.named_parameters():
                for layer in range(inherited_layers):
                    if name.startswith(f"blocks.{layer}.") or name.startswith(
                        f"rim_modules.{layer}."
                    ):
                        active_names.add(name)
                        break
        self._active_names = active_names
        return {
            "trainable": sum(
                parameter.numel()
                for name, parameter in self.target.named_parameters()
                if name in active_names
            ),
            "phase": phase,
        }

    def mask_inactive_gradients(self) -> None:
        for name, parameter in self.target.named_parameters():
            if name not in self._active_names and parameter.grad is not None:
                parameter.grad.zero_()

    def alignment_loss(
        self,
        token_ids: torch.Tensor,
        *,
        step: int,
        target_logits: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if step >= self.alignment_steps:
            return torch.zeros((), device=token_ids.device)
        with torch.no_grad():
            source_logits, _ = self.source(token_ids)
        if target_logits is None:
            target_logits, _ = self.target(token_ids)
        weight = max(0.0, 1.0 - float(step) / max(1, self.alignment_steps))
        return weight * F.mse_loss(target_logits.float(), source_logits.float())


class CrossScaleIdentityInheritance:
    @staticmethod
    def _mapping(source: int, target: int) -> tuple[torch.Tensor, torch.Tensor]:
        index = torch.arange(target, dtype=torch.long) % source
        counts = torch.bincount(index, minlength=source).float()
        scale = counts[index].rsqrt()
        return index, scale

    @staticmethod
    def _hidden_mapping(
        source: int,
        target: int,
        *,
        preserved_tail: int = 64,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if target < source:
            raise ValueError("Model growth cannot shrink the residual width.")
        tail = min(preserved_tail, source, target)
        body = max(1, source - tail)
        index = torch.arange(target, dtype=torch.long) % body
        index[:source] = torch.arange(source)
        if tail:
            index[-tail:] = torch.arange(source - tail, source)
        counts = torch.bincount(index, minlength=source).float()
        return index, counts

    @classmethod
    def _expand_vector(cls, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        index, _ = cls._mapping(source.numel(), target.numel())
        return source.reshape(-1)[index].reshape_as(target).to(dtype=target.dtype)

    @classmethod
    def _expand_matrix(cls, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        row_index, _ = cls._mapping(source.shape[0], target.shape[0])
        col_index, _ = cls._mapping(source.shape[1], target.shape[1])
        col_counts = torch.bincount(col_index, minlength=source.shape[1]).float()
        expanded = source[row_index][:, col_index]
        expanded = expanded / col_counts[col_index][None, :]
        return expanded.to(dtype=target.dtype)

    @classmethod
    def _expand_model_matrix(
        cls,
        source: torch.Tensor,
        target: torch.Tensor,
        *,
        source_width: int,
        target_width: int,
    ) -> torch.Tensor:
        hidden_index, hidden_counts = cls._hidden_mapping(
            source_width,
            target_width,
        )
        if source.shape[0] == source_width and target.shape[0] == target_width:
            row_index = hidden_index
        else:
            row_index, _ = cls._mapping(source.shape[0], target.shape[0])
        if source.shape[1] == source_width and target.shape[1] == target_width:
            col_index = hidden_index
            col_counts = hidden_counts
        else:
            col_index, _ = cls._mapping(source.shape[1], target.shape[1])
            col_counts = torch.bincount(
                col_index,
                minlength=source.shape[1],
            ).float()
        return (source[row_index][:, col_index] / col_counts[col_index][None, :]).to(
            dtype=target.dtype
        )

    @staticmethod
    def _attention_mapping(
        *,
        source_heads: int,
        target_heads: int,
        source_head_dim: int,
        target_head_dim: int,
        source_kv_heads: int,
        target_kv_heads: int,
        kv_projection: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        source_count = source_kv_heads if kv_projection else source_heads
        target_count = target_kv_heads if kv_projection else target_heads
        source_per_group = 1 if kv_projection else source_heads // source_kv_heads
        target_per_group = 1 if kv_projection else target_heads // target_kv_heads
        mapped: list[int] = []
        for target_head in range(target_count):
            target_group = target_head // target_per_group
            within_group = target_head % target_per_group
            source_group = target_group % source_kv_heads
            source_head = (
                source_group
                if kv_projection
                else source_group * source_per_group + within_group % source_per_group
            )
            mapped.extend(
                source_head * source_head_dim + dim % source_head_dim
                for dim in range(target_head_dim)
            )
        index = torch.tensor(mapped, dtype=torch.long)
        counts = torch.bincount(
            index,
            minlength=source_count * source_head_dim,
        ).float()
        return index, counts

    @classmethod
    def _expand_attention_weight(
        cls,
        source_tensor: torch.Tensor,
        target_tensor: torch.Tensor,
        *,
        source_model: object,
        target_model: object,
        kind: str,
    ) -> torch.Tensor:
        source_head_dim = source_model.n_embd // source_model.n_head
        target_head_dim = target_model.n_embd // target_model.n_head
        hidden_index, hidden_counts = cls._hidden_mapping(
            source_model.n_embd,
            target_model.n_embd,
        )
        if kind in {"q", "k", "v"}:
            kv_projection = kind in {"k", "v"}
            row_index, row_counts = cls._attention_mapping(
                source_heads=source_model.n_head,
                target_heads=target_model.n_head,
                source_head_dim=source_head_dim,
                target_head_dim=target_head_dim,
                source_kv_heads=source_model.n_kv_head,
                target_kv_heads=target_model.n_kv_head,
                kv_projection=kv_projection,
            )
            expanded = (
                source_tensor[row_index][:, hidden_index] / hidden_counts[hidden_index][None, :]
            )
            if kind in {"q", "k"}:
                scale = (target_head_dim / source_head_dim) ** 0.25
                expanded = expanded * scale / row_counts[row_index].sqrt()[:, None]
            return expanded.to(dtype=target_tensor.dtype)
        if kind == "out":
            col_index, col_counts = cls._attention_mapping(
                source_heads=source_model.n_head,
                target_heads=target_model.n_head,
                source_head_dim=source_head_dim,
                target_head_dim=target_head_dim,
                source_kv_heads=source_model.n_kv_head,
                target_kv_heads=target_model.n_kv_head,
                kv_projection=False,
            )
            return (source_tensor[hidden_index][:, col_index] / col_counts[col_index][None, :]).to(
                dtype=target_tensor.dtype
            )
        raise ValueError(f"Unknown attention projection kind: {kind}")

    @classmethod
    def _expand_tensor(cls, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if source.shape == target.shape:
            return source.to(dtype=target.dtype)
        if source.ndim == target.ndim == 1:
            return cls._expand_vector(source, target)
        if source.ndim == target.ndim == 2:
            return cls._expand_matrix(source, target)
        return target

    @staticmethod
    def _layer_map(
        source_layers: int, target_layers: int
    ) -> tuple[dict[int, int], tuple[int, ...]]:
        inserted = tuple(
            sorted(
                {
                    round((i + 1) * target_layers / (target_layers - source_layers + 1)) - 1
                    for i in range(target_layers - source_layers)
                }
            )
        )
        inserted = tuple(i for i in inserted if 0 <= i < target_layers)[
            : target_layers - source_layers
        ]
        mapping: dict[int, int] = {}
        src = 0
        for target in range(target_layers):
            if target in inserted:
                continue
            mapping[target] = min(src, source_layers - 1)
            src += 1
        return mapping, inserted

    @staticmethod
    def _identity_initialize_block(block: object) -> None:
        torch.nn.init.zeros_(block.attn.out_proj.weight)
        torch.nn.init.zeros_(block.mlp.down_proj.weight)

    @classmethod
    def grow(
        cls,
        source: object,
        target: object,
        *,
        source_checkpoint: str | Path | None = None,
    ) -> GrowthReport:
        source_state = source.state_dict()
        target_state = target.state_dict()
        layer_map, inserted = cls._layer_map(source.n_layer, target.n_layer)
        copied = 0

        for target_key, target_tensor in list(target_state.items()):
            source_key = target_key
            if target_key.startswith("blocks.") or target_key.startswith("rim_modules."):
                parts = target_key.split(".")
                target_layer = int(parts[1])
                if target_layer in inserted:
                    continue
                parts[1] = str(layer_map[target_layer])
                source_key = ".".join(parts)
            source_tensor = source_state.get(source_key)
            if source_tensor is None:
                continue
            attention_kind = next(
                (
                    kind
                    for suffix, kind in (
                        (".attn.q_proj.weight", "q"),
                        (".attn.k_proj.weight", "k"),
                        (".attn.v_proj.weight", "v"),
                        (".attn.out_proj.weight", "out"),
                    )
                    if target_key.endswith(suffix)
                ),
                None,
            )
            if attention_kind is not None:
                target_state[target_key] = cls._expand_attention_weight(
                    source_tensor,
                    target_tensor,
                    source_model=source,
                    target_model=target,
                    kind=attention_kind,
                )
            elif (
                target_key.endswith("token_embedding_table.weight")
                or target_key.endswith("lm_head.weight")
                or target_key.endswith("token_embedding.weight")
            ) and source_tensor.ndim == target_tensor.ndim == 2:
                hidden_index, hidden_counts = cls._hidden_mapping(
                    source.n_embd,
                    target.n_embd,
                )
                target_state[target_key] = (
                    source_tensor[:, hidden_index] / hidden_counts[hidden_index][None, :]
                ).to(dtype=target_tensor.dtype)
            elif source_tensor.ndim == target_tensor.ndim == 2:
                target_state[target_key] = cls._expand_model_matrix(
                    source_tensor,
                    target_tensor,
                    source_width=source.n_embd,
                    target_width=target.n_embd,
                )
            elif (
                source_tensor.ndim == target_tensor.ndim == 1
                and source_tensor.numel() == source.n_embd
                and target_tensor.numel() == target.n_embd
            ):
                hidden_index, _ = cls._hidden_mapping(
                    source.n_embd,
                    target.n_embd,
                )
                target_state[target_key] = source_tensor[hidden_index].to(dtype=target_tensor.dtype)
            else:
                target_state[target_key] = cls._expand_tensor(source_tensor, target_tensor)
            copied += 1

        target.load_state_dict(target_state, strict=False)
        hidden_index, hidden_counts = cls._hidden_mapping(
            source.n_embd,
            target.n_embd,
        )
        with torch.no_grad():
            target.embedding_input_scale.copy_(
                hidden_counts[hidden_index].to(target.embedding_input_scale)
            )
            norm_multiplicity = target.n_embd / source.n_embd / hidden_counts[hidden_index]
            for module in target.modules():
                if hasattr(module, "multiplicity_weight"):
                    module.multiplicity_weight.copy_(
                        norm_multiplicity.to(module.multiplicity_weight)
                    )
        for layer in inserted:
            cls._identity_initialize_block(target.blocks[layer])
        target.esv_module.predictor.load_state_dict(source.esv_module.predictor.state_dict())
        with torch.no_grad():
            if hasattr(source, "dstp_temperature_log") and hasattr(target, "dstp_temperature_log"):
                target.dstp_temperature_log.copy_(
                    F.interpolate(
                        source.dstp_temperature_log.detach().view(1, 1, -1),
                        size=target.n_layer,
                        mode="linear",
                        align_corners=True,
                    ).view_as(target.dstp_temperature_log)
                )
            target.residual_depth_logits.copy_(
                F.interpolate(
                    source.residual_depth_logits.detach().view(1, 1, -1),
                    size=target.n_layer,
                    mode="linear",
                    align_corners=True,
                ).view_as(target.residual_depth_logits)
            )

        digest = ""
        if source_checkpoint is not None and Path(source_checkpoint).exists():
            digest = hashlib.sha256(Path(source_checkpoint).read_bytes()).hexdigest()
        return GrowthReport(
            schema_version=1,
            generated_at=time.time(),
            source_layers=source.n_layer,
            target_layers=target.n_layer,
            source_width=source.n_embd,
            target_width=target.n_embd,
            copied_tensors=copied,
            identity_layers=inserted,
            source_checkpoint_sha256=digest,
        )

    @staticmethod
    @torch.no_grad()
    def verify_parity(
        source: object,
        target: object,
        token_ids: torch.Tensor,
    ) -> dict[str, float]:
        source.eval()
        target.eval()
        source_logits, _ = source(token_ids.to(next(source.parameters()).device))
        target_logits, _ = target(token_ids.to(next(target.parameters()).device))
        source_logits = source_logits.float().cpu()
        target_logits = target_logits.float().cpu()
        cosine = F.cosine_similarity(
            source_logits.reshape(-1, source_logits.shape[-1]),
            target_logits.reshape(-1, target_logits.shape[-1]),
            dim=-1,
        ).mean()
        error = (source_logits - target_logits).abs().max()
        return {"parity_cosine": float(cosine), "parity_max_error": float(error)}

    @staticmethod
    def write_report(report: GrowthReport | dict[str, object], path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(report) if isinstance(report, GrowthReport) else dict(report)
        target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return target

    @staticmethod
    def transfer(source: object, target: object) -> dict[str, object]:
        copied: list[str] = []
        target.esv_module.predictor.load_state_dict(source.esv_module.predictor.state_dict())
        copied.append("esv_predictor")
        if hasattr(source, "dstp_temperature_log") and hasattr(target, "dstp_temperature_log"):
            values = source.dstp_temperature_log.detach().view(1, 1, -1)
            interpolated = F.interpolate(
                values,
                size=target.dstp_temperature_log.numel(),
                mode="linear",
                align_corners=True,
            ).view_as(target.dstp_temperature_log)
            target.dstp_temperature_log.data.copy_(interpolated)
            copied.append("dstp")
        return {"copied": copied, "source_layers": source.n_layer, "target_layers": target.n_layer}

    @staticmethod
    def alignment_loss(
        target_state: torch.Tensor,
        reference_state: torch.Tensor,
        *,
        step: int,
        warmup_steps: int = 5000,
    ) -> torch.Tensor:
        weight = max(0.0, 1.0 - float(step) / max(1, warmup_steps))
        return weight * F.mse_loss(target_state, reference_state.detach())
