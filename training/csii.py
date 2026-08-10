"""Cross-scale inheritance and deterministic progressive model growth."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F  # noqa: N812 - canonical PyTorch alias


@dataclass(frozen=True)
class GrowthReport:
    schema_version: int
    generated_at: float
    source_profile: str
    target_profile: str
    source_layers: int
    target_layers: int
    source_width: int
    target_width: int
    copied_tensors: int
    identity_layers: tuple[int, ...]
    layer_mapping: tuple[tuple[int, int], ...]
    attention_mode_mapping: tuple[tuple[int, int | None, str, int | None], ...]
    source_architecture_sha256: str
    target_architecture_sha256: str
    source_checkpoint_sha256: str
    optimizer_restart_required: bool = True
    optimizer_state_inherited: bool = False
    parity_semantics: str = "real_distribution_valid_tokens_v2"
    parity_token_ids_sha256: str = ""
    parity_cosine: float | None = None
    parity_max_error: float | None = None
    parity_mean_absolute_error: float | None = None
    parity_mean_kl: float | None = None
    parity_top1_agreement: float | None = None
    parity_minimum_cosine: float = 0.99
    parity_maximum_mean_kl: float = 0.001
    parity_minimum_top1_agreement: float = 0.99
    parity_passed: bool | None = None


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _attention_mode(block: Any) -> tuple[str, int | None]:
    window = getattr(block.attn, "sliding_window", None)
    return ("full", None) if window is None else ("sliding", int(window))


def model_architecture_payload(model: Any) -> dict[str, object]:
    """Describe effective structure, including per-layer attention behavior."""
    blocks = list(model.blocks)
    return {
        "schema_version": 1,
        "architecture_version": str(getattr(model, "architecture_version", "unknown")),
        "vocab_size": int(model.vocab_size),
        "d_model": int(model.n_embd),
        "n_layers": int(model.n_layer),
        "n_query_heads": int(model.n_head),
        "n_kv_heads": int(model.n_kv_head),
        "head_dim": int(model.n_embd) // int(model.n_head),
        "d_ff": int(model.d_ff),
        "context_length": int(model.block_size),
        "rope_base": int(model.rope_base),
        "use_qk_norm": bool(model.use_qk_norm),
        "mod_layers": list(getattr(model, "mod_layers", ())),
        "attention_modes": [
            {"kind": kind, "window": window}
            for kind, window in (_attention_mode(block) for block in blocks)
        ],
        "use_mtp": bool(getattr(model, "use_mtp", False)),
        "use_moe": bool(getattr(model, "use_moe", False)),
        "approved_subsystems": list(getattr(model, "approved_subsystems", ())),
        "initialization_scheme": str(getattr(model, "initialization_scheme", "unknown")),
    }


def model_architecture_sha256(model: Any) -> str:
    material = json.dumps(
        model_architecture_payload(model), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(material).hexdigest()


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
        for name, parameter in self.target.named_parameters():
            parameter.requires_grad_(name in active_names)
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
                # AdamW updates a zero-gradient tensor through decoupled weight
                # decay and existing moments. ``None`` is the only true skip.
                parameter.grad = None

    def alignment_loss(
        self,
        token_ids: torch.Tensor,
        *,
        step: int,
        target_logits: torch.Tensor | None = None,
        max_tokens: int = 64,
    ) -> torch.Tensor:
        if step >= self.alignment_steps:
            return torch.zeros((), device=token_ids.device)
        if max_tokens < 1:
            raise ValueError("Growth alignment max_tokens must be positive")
        aligned_tokens = token_ids[:, :max_tokens]
        with torch.no_grad():
            source_logits, _ = self.source(aligned_tokens)
        if target_logits is None:
            target_logits, _ = self.target(aligned_tokens)
        else:
            target_logits = target_logits[:, : aligned_tokens.shape[1]]
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
    def _preserve_attention_modes(
        cls,
        source: object,
        target: object,
        *,
        layer_map: dict[int, int],
        inserted: tuple[int, ...],
    ) -> tuple[tuple[int, int | None, str, int | None], ...]:
        """Copy effective attention mode rather than re-deriving it by child index."""
        result: list[tuple[int, int | None, str, int | None]] = []
        for target_layer, target_block in enumerate(target.blocks):
            source_layer: int | None = None
            if target_layer not in inserted:
                source_layer = layer_map[target_layer]
                kind, window = _attention_mode(source.blocks[source_layer])
                target_block.attn.sliding_window = window if kind == "sliding" else None
            kind, window = _attention_mode(target_block)
            result.append((target_layer, source_layer, kind, window))
        return tuple(result)

    @staticmethod
    def apply_attention_mode_mapping(
        target: object,
        report: GrowthReport | dict[str, object],
    ) -> None:
        """Restore the non-periodic attention layout recorded by a growth artifact."""
        payload = asdict(report) if isinstance(report, GrowthReport) else dict(report)
        raw_mapping = payload.get("attention_mode_mapping", ())
        if not isinstance(raw_mapping, (list, tuple)) or len(raw_mapping) != target.n_layer:
            raise ValueError("Growth manifest must bind one attention mode per target layer")
        seen: set[int] = set()
        for entry in raw_mapping:
            if not isinstance(entry, (list, tuple)) or len(entry) != 4:
                raise ValueError("Invalid growth attention-mode mapping entry")
            target_layer = int(entry[0])
            kind = str(entry[2])
            window = entry[3]
            if target_layer in seen or not 0 <= target_layer < target.n_layer:
                raise ValueError("Growth attention mapping has duplicate or invalid layers")
            if kind == "full":
                if window is not None:
                    raise ValueError("Full attention mapping cannot declare a window")
                target.blocks[target_layer].attn.sliding_window = None
            elif kind == "sliding":
                if window is None or int(window) <= 0:
                    raise ValueError("Sliding attention mapping requires a positive window")
                target.blocks[target_layer].attn.sliding_window = int(window)
            else:
                raise ValueError(f"Unsupported attention mode in growth manifest: {kind}")
            seen.add(target_layer)
        expected = str(payload.get("target_architecture_sha256", ""))
        actual = model_architecture_sha256(target)
        if expected and expected != actual:
            raise ValueError(
                "Growth attention mapping does not reconstruct the bound target architecture: "
                f"{actual} != {expected}"
            )

    @classmethod
    def grow(
        cls,
        source: object,
        target: object,
        *,
        source_checkpoint: str | Path | None = None,
        source_profile: str = "",
        target_profile: str = "",
    ) -> GrowthReport:
        if target.n_layer < source.n_layer:
            raise ValueError("Model growth cannot shrink transformer depth")
        if target.vocab_size != source.vocab_size:
            raise ValueError("Cross-scale growth requires one unchanged tokenizer vocabulary")
        source_architecture_sha256 = model_architecture_sha256(source)
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
        attention_mode_mapping = cls._preserve_attention_modes(
            source,
            target,
            layer_map=layer_map,
            inserted=inserted,
        )
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
            digest = _sha256_file(source_checkpoint)
        return GrowthReport(
            schema_version=3,
            generated_at=time.time(),
            source_profile=str(source_profile),
            target_profile=str(target_profile),
            source_layers=source.n_layer,
            target_layers=target.n_layer,
            source_width=source.n_embd,
            target_width=target.n_embd,
            copied_tensors=copied,
            identity_layers=inserted,
            layer_mapping=tuple(sorted(layer_map.items())),
            attention_mode_mapping=attention_mode_mapping,
            source_architecture_sha256=source_architecture_sha256,
            target_architecture_sha256=model_architecture_sha256(target),
            source_checkpoint_sha256=digest,
        )

    @staticmethod
    @torch.no_grad()
    def verify_parity(
        source: object,
        target: object,
        token_ids: torch.Tensor,
        *,
        valid_token_mask: torch.Tensor | None = None,
    ) -> dict[str, float | str]:
        if token_ids.ndim != 2 or token_ids.numel() == 0:
            raise ValueError("Parity verification requires non-empty [batch, sequence] token IDs")
        if token_ids.dtype not in {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        }:
            raise TypeError("Parity token IDs must be an integer tensor")
        if source.vocab_size != target.vocab_size:
            raise ValueError("Real-logits parity requires identical source/target vocabularies")
        source.eval()
        target.eval()
        source_logits, _ = source(token_ids.to(next(source.parameters()).device))
        target_logits, _ = target(token_ids.to(next(target.parameters()).device))
        source_logits = source_logits.float().cpu()
        target_logits = target_logits.float().cpu()
        if source_logits.shape != target_logits.shape:
            raise ValueError(
                "Real-logits parity requires identical source/target logit shapes; "
                f"got {tuple(source_logits.shape)} and {tuple(target_logits.shape)}"
            )
        if not torch.isfinite(source_logits).all() or not torch.isfinite(target_logits).all():
            raise ValueError("Real-logits parity cannot be computed from NaN/Inf logits")
        if valid_token_mask is None:
            mask = torch.ones(source_logits.shape[:-1], dtype=torch.bool)
        else:
            mask = valid_token_mask.detach().cpu().to(torch.bool)
            if tuple(mask.shape) != tuple(source_logits.shape[:-1]):
                raise ValueError("Parity mask must match the batch/sequence logit dimensions")
        if not bool(mask.any()):
            raise ValueError("Parity mask selects no real tokens")
        source_valid = source_logits[mask]
        target_valid = target_logits[mask]
        cosine = F.cosine_similarity(
            source_valid,
            target_valid,
            dim=-1,
        ).mean()
        absolute_error = (source_valid - target_valid).abs()
        source_log_probs = F.log_softmax(source_valid, dim=-1)
        target_log_probs = F.log_softmax(target_valid, dim=-1)
        source_probs = source_log_probs.exp()
        mean_kl = (source_probs * (source_log_probs - target_log_probs)).sum(dim=-1).mean()
        top1_agreement = (
            source_valid.argmax(dim=-1) == target_valid.argmax(dim=-1)
        ).float().mean()
        token_material = json.dumps(
            {
                "token_ids": token_ids.detach().cpu().to(torch.int64).tolist(),
                "valid_token_mask": mask.tolist(),
            },
            separators=(",", ":"),
        ).encode("utf-8")
        return {
            "parity_semantics": "real_distribution_valid_tokens_v2",
            "parity_token_ids_sha256": hashlib.sha256(token_material).hexdigest(),
            "parity_cosine": float(cosine),
            "parity_max_error": float(absolute_error.max()),
            "parity_mean_absolute_error": float(absolute_error.mean()),
            "parity_mean_kl": float(mean_kl),
            "parity_top1_agreement": float(top1_agreement),
        }

    @staticmethod
    def bind_parity(
        report: GrowthReport,
        parity: dict[str, float | str],
        *,
        minimum_cosine: float = 0.99,
        maximum_mean_kl: float = 0.001,
        minimum_top1_agreement: float = 0.99,
    ) -> GrowthReport:
        cosine = float(parity.get("parity_cosine", float("nan")))
        maximum_error = float(parity.get("parity_max_error", float("nan")))
        mean_error = float(parity.get("parity_mean_absolute_error", float("nan")))
        mean_kl = float(parity.get("parity_mean_kl", float("nan")))
        top1_agreement = float(parity.get("parity_top1_agreement", float("nan")))
        semantics = str(parity.get("parity_semantics", ""))
        token_hash = str(parity.get("parity_token_ids_sha256", ""))
        if semantics != "real_distribution_valid_tokens_v2":
            raise ValueError("Growth parity must compare output distributions on real tokens")
        if len(token_hash) != 64:
            raise ValueError("Growth parity is missing its token-ID hash")
        if not all(
            math.isfinite(value)
            for value in (cosine, maximum_error, mean_error, mean_kl, top1_agreement)
        ):
            raise ValueError("Growth parity metrics must be finite")
        threshold = float(minimum_cosine)
        kl_threshold = float(maximum_mean_kl)
        top1_threshold = float(minimum_top1_agreement)
        if not 0.0 < threshold <= 1.0:
            raise ValueError("minimum_cosine must be in (0, 1]")
        if kl_threshold < 0.0 or not 0.0 <= top1_threshold <= 1.0:
            raise ValueError("Growth distribution thresholds are invalid")
        passed = (
            cosine >= threshold
            and mean_kl <= kl_threshold
            and top1_agreement >= top1_threshold
        )
        return replace(
            report,
            parity_semantics=semantics,
            parity_token_ids_sha256=token_hash,
            parity_cosine=cosine,
            parity_max_error=maximum_error,
            parity_mean_absolute_error=mean_error,
            parity_mean_kl=mean_kl,
            parity_top1_agreement=top1_agreement,
            parity_minimum_cosine=threshold,
            parity_maximum_mean_kl=kl_threshold,
            parity_minimum_top1_agreement=top1_threshold,
            parity_passed=passed,
        )

    @staticmethod
    def validate_growth_report(
        report: GrowthReport | dict[str, object],
        *,
        require_passed_parity: bool = True,
    ) -> dict[str, object]:
        payload = asdict(report) if isinstance(report, GrowthReport) else dict(report)
        if int(payload.get("schema_version", 0)) != 3:
            raise ValueError("Unsupported growth-manifest schema version")
        source_profile = str(payload.get("source_profile", "")).strip()
        target_profile = str(payload.get("target_profile", "")).strip()
        if not source_profile or not target_profile or source_profile == target_profile:
            raise ValueError("Growth manifest requires distinct source and target profiles")
        for field in (
            "source_architecture_sha256",
            "target_architecture_sha256",
            "source_checkpoint_sha256",
            "parity_token_ids_sha256",
        ):
            value = str(payload.get(field, ""))
            if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                raise ValueError(f"Growth manifest has invalid {field}")
        source_layers = int(payload.get("source_layers", 0))
        target_layers = int(payload.get("target_layers", 0))
        source_width = int(payload.get("source_width", 0))
        target_width = int(payload.get("target_width", 0))
        if source_layers <= 0 or target_layers < source_layers:
            raise ValueError("Growth manifest has invalid depth expansion")
        if source_width <= 0 or target_width < source_width:
            raise ValueError("Growth manifest has invalid width expansion")
        raw_layer_mapping = payload.get("layer_mapping", ())
        raw_identity_layers = payload.get("identity_layers", ())
        if not isinstance(raw_layer_mapping, (list, tuple)) or not isinstance(
            raw_identity_layers, (list, tuple)
        ):
            raise ValueError("Growth manifest layer mapping must be an array")
        layer_mapping = {int(entry[0]): int(entry[1]) for entry in raw_layer_mapping}
        identities = {int(value) for value in raw_identity_layers}
        if len(layer_mapping) != source_layers or len(identities) != target_layers - source_layers:
            raise ValueError("Growth manifest does not account for every source/inserted layer")
        if set(layer_mapping) & identities or set(layer_mapping) | identities != set(
            range(target_layers)
        ):
            raise ValueError("Growth manifest target-layer accounting is incomplete")
        if set(layer_mapping.values()) != set(range(source_layers)):
            raise ValueError("Growth manifest source-layer mapping is incomplete")
        raw_attention = payload.get("attention_mode_mapping", ())
        if not isinstance(raw_attention, (list, tuple)) or len(raw_attention) != target_layers:
            raise ValueError("Growth manifest must bind every target attention mode")
        attention_targets: set[int] = set()
        for entry in raw_attention:
            if not isinstance(entry, (list, tuple)) or len(entry) != 4:
                raise ValueError("Growth manifest has an invalid attention mapping entry")
            target_layer = int(entry[0])
            source_layer = None if entry[1] is None else int(entry[1])
            if target_layer in attention_targets or target_layer not in range(target_layers):
                raise ValueError("Growth manifest attention targets are invalid or duplicated")
            expected_source = layer_mapping.get(target_layer)
            if source_layer != expected_source:
                raise ValueError("Growth attention mapping disagrees with the depth mapping")
            kind = str(entry[2])
            window = entry[3]
            if (kind == "full" and window is not None) or (
                kind == "sliding" and (window is None or int(window) <= 0)
            ):
                raise ValueError("Growth manifest has an invalid effective attention mode")
            if kind not in {"full", "sliding"}:
                raise ValueError("Growth manifest has an unsupported effective attention mode")
            attention_targets.add(target_layer)
        if payload.get("optimizer_restart_required") is not True:
            raise ValueError("A growth child must restart its optimizer")
        if payload.get("optimizer_state_inherited") is not False:
            raise ValueError("A growth child must not inherit shape-incompatible optimizer state")
        if str(payload.get("parity_semantics", "")) != "real_distribution_valid_tokens_v2":
            raise ValueError("Growth manifest is not bound to masked distribution parity")
        cosine = float(payload.get("parity_cosine", float("nan")))
        maximum_error = float(payload.get("parity_max_error", float("nan")))
        mean_error = float(payload.get("parity_mean_absolute_error", float("nan")))
        mean_kl = float(payload.get("parity_mean_kl", float("nan")))
        top1_agreement = float(payload.get("parity_top1_agreement", float("nan")))
        threshold = float(payload.get("parity_minimum_cosine", float("nan")))
        kl_threshold = float(payload.get("parity_maximum_mean_kl", float("nan")))
        top1_threshold = float(
            payload.get("parity_minimum_top1_agreement", float("nan"))
        )
        if not all(
            math.isfinite(value)
            for value in (
                cosine,
                maximum_error,
                mean_error,
                mean_kl,
                top1_agreement,
                threshold,
                kl_threshold,
                top1_threshold,
            )
        ):
            raise ValueError("Growth manifest parity metrics must be finite")
        if (
            not 0.0 < threshold <= 1.0
            or maximum_error < 0.0
            or mean_error < 0.0
            or mean_kl < 0.0
            or kl_threshold < 0.0
            or not 0.0 <= top1_agreement <= 1.0
            or not 0.0 <= top1_threshold <= 1.0
        ):
            raise ValueError("Growth manifest parity metrics are outside their valid ranges")
        passed = (
            cosine >= threshold
            and mean_kl <= kl_threshold
            and top1_agreement >= top1_threshold
        )
        if payload.get("parity_passed") is not passed:
            raise ValueError("Growth manifest parity decision disagrees with its metrics")
        if require_passed_parity and not passed:
            raise ValueError("Growth manifest did not pass its real-logits parity gate")
        return payload

    @staticmethod
    def write_report(report: GrowthReport | dict[str, object], path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(report) if isinstance(report, GrowthReport) else dict(report)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(target)
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
