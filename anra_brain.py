"""An-Ra mainline model and tokenizer definitions.

Canonical implementation: the An-Ra V4 decoder used by the sole active model
profile. Historical Python aliases remain import-compatible but are not
selectable model profiles.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Iterable, Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass
from functools import partial

import torch
import torch.nn as nn
from anra.core.registry import MODEL_REGISTRY
from torch.nn import functional as F  # noqa: N812 - conventional torch alias
from torch.nn.utils.parametrizations import spectral_norm
from torch.utils.checkpoint import checkpoint as _torch_checkpoint

from identity.esv import ESVModule

try:
    from identity.hal import HALModule
except Exception:  # pragma: no cover - HAL is optional for old runtimes.
    HALModule = None
from tokenizer.char_tokenizer import CharTokenizer  # noqa: F401 - compatibility export

ANRA_V4_ARCHITECTURE_VERSION = "anra_v4_rope_interleaved_v1"


@contextmanager
def _freeze_rim_spectral_norm_updates(
    rim: ResidualIdentityModulator | None,
) -> Iterator[None]:
    """Keep checkpoint recomputation from advancing spectral-norm state twice."""
    if rim is None:
        yield
        return
    projection = rim.projection
    was_training = projection.training
    projection.eval()
    try:
        yield
    finally:
        projection.train(was_training)


def _checkpoint_contexts(
    rim: ResidualIdentityModulator | None,
) -> tuple[AbstractContextManager[None], AbstractContextManager[None]]:
    return nullcontext(), _freeze_rim_spectral_norm_updates(rim)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.register_buffer("multiplicity_weight", torch.ones(dim), persistent=True)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean_square = (x.pow(2) * self.multiplicity_weight).mean(
            dim=-1,
            keepdim=True,
        )
        scale = torch.rsqrt(mean_square + self.eps)
        return x * scale * self.weight


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


class RotaryEmbedding(nn.Module):
    def __init__(
        self, dim: int, base: int = 10000, base_seq_len: int = 512, target_seq_len: int = 2048
    ) -> None:
        super().__init__()
        if dim <= 0 or dim % 2:
            raise ValueError("RoPE head dimension must be a positive even integer")
        if base <= 1 or base_seq_len <= 0 or target_seq_len <= 0:
            raise ValueError("RoPE base and sequence lengths must be positive")
        self.dim = dim
        self.base = base
        self.base_seq_len = base_seq_len
        self.target_seq_len = target_seq_len
        inv_freq = self._yarn_inv_freq()
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cached_seq_len = 0
        self._cached_cos: torch.Tensor | None = None
        self._cached_sin: torch.Tensor | None = None

    def _yarn_inv_freq(self) -> torch.Tensor:
        import math

        scale = self.target_seq_len / self.base_seq_len
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        dim_threshold = (
            self.dim
            * math.log(scale)
            / (2 * math.log(self.base * self.base_seq_len / (2 * math.pi)))
        )
        dim_threshold = max(0, min(self.dim // 2 - 1, int(dim_threshold)))
        scaling = torch.ones(self.dim // 2)
        scaling[:dim_threshold] = 1.0 / scale
        self._attn_scale = 0.1 * math.log(scale) + 1.0
        return inv_freq * scaling

    def _build_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> None:
        if (
            self._cached_cos is not None
            and self._cached_seq_len >= seq_len
            and self._cached_cos.dtype == dtype
            and self._cached_cos.device == device
        ):
            return
        positions = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(positions, self.inv_freq.to(device))
        # _rotate_half rotates adjacent pairs (0,1), (2,3), ...; therefore
        # each pair must receive the same phase. Concatenating the frequency
        # vector would pair different phases and corrupt the rotary geometry.
        emb = torch.repeat_interleave(freqs, 2, dim=-1)
        self._cached_cos = emb.cos()[None, None, :, :].to(dtype=dtype)
        self._cached_sin = emb.sin()[None, None, :, :].to(dtype=dtype)
        self._cached_seq_len = seq_len

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        *,
        position_offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = q.size(-2)
        end = int(position_offset) + seq_len
        self._build_cache(end, q.device, q.dtype)
        cos = self._cached_cos[..., position_offset:end, :]
        sin = self._cached_sin[..., position_offset:end, :]
        q = (q * cos) + (_rotate_half(q) * sin)
        k = (k * cos) + (_rotate_half(k) * sin)
        return q, k


class MultiHeadAttentionV2(nn.Module):
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        n_kv_head: int | None = None,
        dropout: float = 0.0,
        base_seq_len: int = 512,
        target_seq_len: int = 2048,
        rope_base: int = 10_000,
        use_qk_norm: bool = False,
        sliding_window: int | None = None,
    ) -> None:
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError(f"n_embd={n_embd} must be divisible by n_head={n_head}")
        self.n_head = n_head
        self.n_kv_head = n_kv_head if n_kv_head is not None else n_head
        if self.n_kv_head <= 0 or self.n_head % self.n_kv_head != 0:
            raise ValueError(
                f"n_head={n_head} must be divisible by positive n_kv_head={self.n_kv_head}"
            )
        self.head_dim = n_embd // n_head
        self.groups = self.n_head // self.n_kv_head
        self.use_qk_norm = bool(use_qk_norm)
        self.sliding_window = int(sliding_window) if sliding_window else None
        if self.sliding_window is not None and self.sliding_window < 1:
            raise ValueError("sliding_window must be positive when enabled")

        self.q_proj = nn.Linear(n_embd, n_head * self.head_dim, bias=False)
        self.k_proj = nn.Linear(n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.out_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.rope = RotaryEmbedding(
            self.head_dim,
            base=rope_base,
            base_seq_len=base_seq_len,
            target_seq_len=target_seq_len,
        )
        self.dropout = dropout
        # Either the exact tensor dictionary or the evidence-gated,
        # device-resident TurboQuant pilot. Training always leaves this None.
        self._kv_cache: object | None = None
        self._layer_idx: int = 0
        self.max_cache_len = int(target_seq_len)
        self.lba_bound = 0.8 * (65504.0 / self.head_dim) ** 0.5

    def forward(
        self, x: torch.Tensor, *, attention_temperature: torch.Tensor | float | None = None
    ) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        q = self.q_proj(x).view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.n_kv_head, self.head_dim).transpose(1, 2)
        position_offset = 0
        if self._kv_cache is not None:
            if isinstance(self._kv_cache, dict):
                position_offset = int(self._kv_cache.get("position", 0))
            else:
                position_offset = int(self._kv_cache.position)
        if self.use_qk_norm:
            # Parameter-free per-head QK-Norm keeps the checkpoint tensor
            # schema stable while preventing query/key norm drift. RoPE is an
            # orthogonal transform, so applying normalization before it
            # preserves the intended unit-RMS geometry.
            q = F.rms_norm(q, (self.head_dim,))
            k = F.rms_norm(k, (self.head_dim,))
        q, k = self.rope(q, k, position_offset=position_offset)

        if attention_temperature is not None:
            temperature = torch.as_tensor(
                attention_temperature, dtype=q.dtype, device=q.device
            ).clamp(0.5, 2.0)
            q = q / temperature

        bound = torch.as_tensor(self.lba_bound, dtype=q.dtype, device=q.device)
        q = bound * torch.tanh(q / bound)
        k = bound * torch.tanh(k / bound)

        if self._kv_cache is not None:
            if isinstance(self._kv_cache, dict):
                cache_k = self._kv_cache.get("k")
                cache_v = self._kv_cache.get("v")
                if cache_k is not None and cache_v is not None:
                    k = torch.cat([cache_k, k], dim=2)
                    v = torch.cat([cache_v, v], dim=2)
                cache_limit = self.max_cache_len
                if self.sliding_window is not None:
                    cache_limit = min(cache_limit, self.sliding_window)
                if k.size(2) > cache_limit:
                    k = k[:, :, -cache_limit:, :]
                    v = v[:, :, -cache_limit:, :]
                self._kv_cache["k"] = k.detach()
                self._kv_cache["v"] = v.detach()
                self._kv_cache["position"] = position_offset + seq_len
            else:
                k, v = self._kv_cache.update(k.detach(), v.detach())

        is_causal = not (self._kv_cache is not None and q.size(2) == 1)
        attention_mask = None
        if self.sliding_window is not None and k.size(2) > self.sliding_window:
            # SDPA boolean masks use True for positions that may participate.
            # Express positions relative to the retained key sequence so this
            # works for both full-sequence training and multi-token cache use.
            query_positions = torch.arange(
                k.size(2) - q.size(2), k.size(2), device=q.device
            ).unsqueeze(1)
            key_positions = torch.arange(k.size(2), device=q.device).unsqueeze(0)
            attention_mask = (
                (key_positions <= query_positions)
                & (key_positions > query_positions - self.sliding_window)
            )[None, None, :, :]
            is_causal = False
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
            enable_gqa=self.groups > 1,
        )
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.out_proj(out)


class SwiGLU(nn.Module):
    def __init__(self, n_embd: int, hidden_dim: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(n_embd, hidden_dim, bias=False)
        self.up_proj = nn.Linear(n_embd, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class SparseUpcycledMoE(nn.Module):
    """Eight routed SwiGLU experts plus one always-on shared expert.

    Routed experts are cloned from the shared dense MLP after initialization.
    Consequently ``0.5 * (shared + routed)`` is exactly the original dense
    function at step zero even though each token selects only two routed
    experts. Load balancing uses a persistent score bias updated only after a
    completed optimizer boundary; no auxiliary loss competes with language
    modeling and checkpoint recomputation cannot advance balancing state.
    """

    def __init__(
        self,
        dense: SwiGLU,
        *,
        routed_experts: int = 8,
        top_k: int = 2,
        balance_rate: float = 1e-3,
    ) -> None:
        super().__init__()
        if routed_experts < 2 or not 1 <= top_k <= routed_experts:
            raise ValueError("MoE requires at least two experts and a valid top_k")
        self.routed_experts = int(routed_experts)
        self.top_k = int(top_k)
        self.balance_rate = float(balance_rate)
        width = dense.gate_proj.in_features
        self.shared_expert = dense
        self.experts = nn.ModuleList(copy.deepcopy(dense) for _ in range(routed_experts))
        self.router = nn.Linear(width, routed_experts, bias=False)
        self.register_buffer("expert_bias", torch.zeros(routed_experts), persistent=True)
        self.register_buffer(
            "expert_load_ema",
            torch.full((routed_experts,), 1.0 / routed_experts),
            persistent=True,
        )
        self._last_load: torch.Tensor | None = None
        self._load_batches = 0

    def reset_from_shared(self) -> None:
        state = self.shared_expert.state_dict()
        for expert in self.experts:
            expert.load_state_dict(state)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        route_scores = self.router(flat).float() + self.expert_bias
        top_values, top_indices = route_scores.topk(self.top_k, dim=-1)
        route_weights = torch.softmax(top_values, dim=-1).to(dtype=x.dtype)
        routed = torch.zeros_like(flat)
        for expert_index, expert in enumerate(self.experts):
            assignments = (top_indices == expert_index).nonzero(as_tuple=False)
            if assignments.numel() == 0:
                continue
            token_indices = assignments[:, 0]
            slots = assignments[:, 1]
            expert_output = expert(flat.index_select(0, token_indices))
            weighted = expert_output * route_weights[token_indices, slots].unsqueeze(-1)
            routed.index_add_(0, token_indices, weighted)
        if self.training:
            with torch.no_grad():
                load = torch.bincount(
                    top_indices.reshape(-1), minlength=self.routed_experts
                ).float()
                normalized_load = load / max(1, top_indices.numel())
                self._last_load = (
                    normalized_load
                    if self._last_load is None
                    else self._last_load + normalized_load
                )
                self._load_batches += 1
        output = 0.5 * (self.shared_expert(flat) + routed)
        return output.reshape(shape)

    @torch.no_grad()
    def update_balance(self) -> None:
        if self._last_load is None:
            return
        observed = self._last_load / max(1, self._load_batches)
        self.expert_load_ema.mul_(0.9).add_(observed.to(self.expert_load_ema), alpha=0.1)
        target = 1.0 / self.routed_experts
        self.expert_bias.add_(self.balance_rate * (target - self.expert_load_ema))
        self.expert_bias.clamp_(-0.1, 0.1)
        self._last_load = None
        self._load_batches = 0


@dataclass(frozen=True)
class RouterContext:
    esv_arousal: torch.Tensor | float = 0.0
    token_entropy: torch.Tensor | float = 0.0
    civ_similarity: torch.Tensor | float = 1.0


class MoDRouter(nn.Module):
    def __init__(self, d_model: int, capacity: float = 0.5) -> None:
        super().__init__()
        self.capacity = capacity
        self.gate = nn.Linear(d_model, 1, bias=False)
        self.capacity_control = nn.Parameter(torch.zeros(()))
        self.context_weights = nn.Parameter(torch.zeros(3))
        nn.init.normal_(self.gate.weight, std=0.02)
        self.last_telemetry: dict[str, float] = {}
        self.telemetry_enabled = False
        self._telemetry_tensors: dict[str, torch.Tensor] = {}
        self._last_balance_loss: torch.Tensor | None = None
        self._last_z_loss: torch.Tensor | None = None
        # AN: Small normal init avoids tied top-k routing during early training.

    def forward(
        self,
        x: torch.Tensor,
        ffn: nn.Module,
        ctx: RouterContext | None = None,
    ) -> torch.Tensor:
        _batch_size, n, d = x.shape
        # Discrete token count is an explicit experiment setting. The learned
        # capacity_control remains a differentiable gate bias; deriving ``k``
        # from it would require a CUDA-synchronizing .item() on every layer.
        capacity = max(0.05, min(1.0, float(self.capacity)))
        k = max(1, min(n, int(n * capacity)))
        scores = self.gate(x).squeeze(-1)
        if ctx is not None:
            context_values = []
            for value in (ctx.esv_arousal, ctx.token_entropy, ctx.civ_similarity):
                tensor = torch.as_tensor(value, device=x.device, dtype=x.dtype)
                while tensor.ndim < scores.ndim:
                    tensor = tensor.unsqueeze(-1)
                context_values.append(tensor.expand_as(scores))
            context_stack = torch.stack(context_values, dim=-1)
            scores = scores + torch.sum(
                context_stack * self.context_weights.to(dtype=x.dtype),
                dim=-1,
            )
        topk_vals, topk_idx = scores.topk(k, dim=-1)
        gate_probabilities = torch.sigmoid(scores + self.capacity_control)
        hard_mask = torch.zeros_like(gate_probabilities)
        hard_mask.scatter_(1, topk_idx, 1.0)
        if self.training:
            # Forward uses the exact hard top-k mask. Backward uses sigmoid gates
            # so routing decisions receive a straight-through learning signal.
            straight_through_mask = hard_mask + gate_probabilities - gate_probabilities.detach()
            routed = (straight_through_mask * gate_probabilities).unsqueeze(-1) * ffn(x)
        else:
            selected = x.gather(1, topk_idx.unsqueeze(-1).expand(-1, -1, d))
            selected_out = ffn(selected)
            routing_weights = torch.sigmoid(topk_vals + self.capacity_control).unsqueeze(-1)
            routed = torch.zeros_like(x)
            routed.scatter_add_(
                1,
                topk_idx.unsqueeze(-1).expand(-1, -1, d),
                routing_weights * selected_out,
            )
        self._last_balance_loss = (gate_probabilities.mean() - capacity).pow(2)
        self._last_z_loss = torch.logaddexp(torch.zeros_like(scores), scores).pow(2).mean()
        if self.telemetry_enabled:
            with torch.no_grad():
                gate_entropy = -(
                    gate_probabilities * gate_probabilities.clamp_min(1e-8).log()
                    + (1.0 - gate_probabilities)
                    * (1.0 - gate_probabilities).clamp_min(1e-8).log()
                ).mean()
                self._telemetry_tensors = {
                    "capacity": torch.as_tensor(capacity, device=x.device),
                    "selected_fraction": torch.as_tensor(k / max(1, n), device=x.device),
                    "gate_mean": gate_probabilities.mean().detach(),
                    "gate_entropy": gate_entropy.detach(),
                    "gate_saturation_fraction": (
                        (gate_probabilities < 0.01) | (gate_probabilities > 0.99)
                    )
                    .float()
                    .mean()
                    .detach(),
                    "score_std": scores.float().std(unbiased=False).detach(),
                    "routed_update_norm": routed.norm(dim=-1).mean().detach(),
                }
        return x + routed

    def telemetry(self) -> dict[str, float]:
        """Materialize the latest trace once, outside the hot forward path."""
        if self._telemetry_tensors:
            self.last_telemetry = {
                key: float(value.float().cpu().item())
                for key, value in self._telemetry_tensors.items()
            }
        return dict(self.last_telemetry)


class ResidualIdentityModulator(nn.Module):
    """Bounded additive identity path from the reserved ESV residual channels."""

    def __init__(self, d_model: int, d_esv: int = 64) -> None:
        super().__init__()
        self.projection = spectral_norm(nn.Linear(d_esv, d_model, bias=False))
        self.raw_alpha = nn.Parameter(torch.zeros(()))

    def forward(self, x: torch.Tensor, esv_channel: torch.Tensor) -> torch.Tensor:
        alpha = 0.25 * torch.tanh(self.raw_alpha)
        normalized = F.layer_norm(esv_channel, (esv_channel.shape[-1],))
        identity_delta = self.projection(normalized).unsqueeze(1)
        return x + alpha * identity_delta


class BlockV2(nn.Module):
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        n_kv_head: int | None = None,
        *,
        eps: float = 1e-5,
        dropout: float = 0.0,
        base_seq_len: int = 512,
        target_seq_len: int = 2048,
        rope_base: int = 10_000,
        d_ff: int | None = None,
        use_qk_norm: bool = False,
        sliding_window: int | None = None,
        use_moe: bool = False,
    ) -> None:
        super().__init__()
        hidden_dim = int(d_ff) if d_ff is not None else int(8 / 3 * n_embd)
        hidden_dim = (hidden_dim + 63) // 64 * 64
        if hidden_dim <= 0:
            raise ValueError("SwiGLU hidden dimension must be positive")
        self.norm_1 = RMSNorm(n_embd, eps=eps)
        self.attn = MultiHeadAttentionV2(
            n_embd,
            n_head,
            n_kv_head=n_kv_head,
            dropout=dropout,
            base_seq_len=base_seq_len,
            target_seq_len=target_seq_len,
            rope_base=rope_base,
            use_qk_norm=use_qk_norm,
            sliding_window=sliding_window,
        )
        self.norm_2 = RMSNorm(n_embd, eps=eps)
        dense_mlp = SwiGLU(n_embd, hidden_dim)
        self.mlp: nn.Module = SparseUpcycledMoE(dense_mlp) if use_moe else dense_mlp
        self._normed_mlp = nn.Sequential(*[self.norm_2, self.mlp])

    def forward(
        self,
        x: torch.Tensor,
        *,
        attention_temperature: torch.Tensor | float | None = None,
        mod_router: MoDRouter | None = None,
        router_context: RouterContext | None = None,
        residual_scale: torch.Tensor | float = 1.0,
    ) -> torch.Tensor:
        x = x + residual_scale * self.attn(
            self.norm_1(x), attention_temperature=attention_temperature
        )
        if mod_router is not None:
            routed = mod_router(x, self._normed_mlp, router_context)
            return x + residual_scale * (routed - x)
        return x + residual_scale * self.mlp(self.norm_2(x))


@MODEL_REGISTRY.register(
    "causal_transformer_v4",
    aliases=("causal_transformer_v2",),
)
class CausalTransformerV2(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_embd: int,
        n_head: int,
        n_layer: int,
        block_size: int,
        *,
        n_kv_head: int | None = None,
        rms_norm_eps: float = 1e-5,
        dropout: float = 0.0,
        d_ff: int | None = None,
        rope_base: int = 10_000,
        mod_layers: Iterable[int] = (),
        base_seq_len: int = 512,
        target_seq_len: int = 2048,
        pad_token_id: int = 0,
        use_layer_temperature_bias: bool = True,
        use_hal: bool = False,
        hal_module: object | None = None,
        use_rim: bool = True,
        use_dstp: bool = True,
        use_qk_norm: bool = False,
        sliding_window: int | None = None,
        full_attention_every: int = 0,
        use_mtp: bool = False,
        mtp_depth: int = 2,
        mtp_loss_weight: float = 0.2,
        use_moe: bool = False,
    ) -> None:
        super().__init__()
        if not 0 <= pad_token_id < vocab_size:
            raise ValueError(f"pad_token_id={pad_token_id} must be within vocab_size={vocab_size}")
        self.vocab_size = vocab_size
        self.architecture_version = ANRA_V4_ARCHITECTURE_VERSION
        self.pad_token_id = int(pad_token_id)
        self.n_embd = n_embd
        self.d_model = n_embd
        self.n_head = n_head
        self.n_kv_head = n_kv_head if n_kv_head is not None else n_head
        self.n_layer = n_layer
        self.block_size = block_size
        self.rms_norm_eps = float(rms_norm_eps)
        self.dropout = float(dropout)
        derived_d_ff = (int(8 / 3 * n_embd) + 63) // 64 * 64
        self.d_ff = int(d_ff) if d_ff is not None else derived_d_ff
        if self.d_ff <= 0 or self.d_ff % 64 != 0:
            raise ValueError("d_ff must be positive and divisible by 64")
        self.rope_base = int(rope_base)
        self.mod_layers = tuple(sorted(mod_layers))
        self.base_seq_len = base_seq_len
        self.target_seq_len = target_seq_len
        self.use_layer_temperature_bias = bool(use_layer_temperature_bias)
        self.use_hal = bool(use_hal)
        self.use_rim = bool(use_rim)
        self.use_dstp = bool(use_dstp)
        self.use_qk_norm = bool(use_qk_norm)
        self.sliding_window = int(sliding_window) if sliding_window else None
        self.full_attention_every = int(full_attention_every)
        if self.full_attention_every < 0:
            raise ValueError("full_attention_every cannot be negative")
        self.use_mtp = bool(use_mtp)
        self.mtp_depth = int(mtp_depth) if self.use_mtp else 0
        self.mtp_loss_weight = float(mtp_loss_weight)
        if self.mtp_depth < 0 or self.mtp_loss_weight < 0.0:
            raise ValueError("MTP depth and loss weight cannot be negative")
        self.use_moe = bool(use_moe)
        self.use_mod = True
        self.use_esv_control = True
        self.use_residual_depth = True
        # Native modules are pilot-capable, not implicitly approved. Training
        # records the exact recipe that a checkpoint has actually optimized.
        self.approved_subsystems: tuple[str, ...] = ()
        self._runtime_civ_similarity = 1.0
        self._subsystem_execution = {
            "mod": 0,
            "rim": 0,
            "dstp": 0,
            "esv": 0,
            "esv_features": 0,
            "hal": 0,
        }
        self.cognitive_extension = None
        self._last_cognitive_evidence: list[dict[str, torch.Tensor]] = []
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.token_embedding = self.token_embedding_table
        self.register_buffer(
            "embedding_input_scale",
            torch.ones(n_embd),
            persistent=True,
        )
        self.blocks = nn.ModuleList(
            [
                BlockV2(
                    n_embd,
                    n_head,
                    n_kv_head=self.n_kv_head,
                    eps=rms_norm_eps,
                    dropout=dropout,
                    base_seq_len=base_seq_len,
                    target_seq_len=target_seq_len,
                    rope_base=self.rope_base,
                    d_ff=self.d_ff,
                    use_qk_norm=self.use_qk_norm,
                    sliding_window=(
                        None
                        if self.sliding_window is None
                        or (
                            self.full_attention_every > 0
                            and (layer_index + 1) % self.full_attention_every == 0
                        )
                        else self.sliding_window
                    ),
                    use_moe=self.use_moe,
                )
                for layer_index in range(n_layer)
            ]
        )
        for block in self.blocks:
            block.attn.max_cache_len = int(block_size)
        self.mod_routers = nn.ModuleDict({str(i): MoDRouter(n_embd) for i in mod_layers})
        self.norm_f = RMSNorm(n_embd, eps=rms_norm_eps)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding_table.weight
        self.mtp_norms = nn.ModuleList(
            [RMSNorm(n_embd, eps=rms_norm_eps) for _ in range(self.mtp_depth)]
        )
        self.mtp_projections = nn.ModuleList(
            [nn.Linear(n_embd, n_embd, bias=False) for _ in range(self.mtp_depth)]
        )
        self._last_hidden: torch.Tensor | None = None
        self.use_gradient_checkpointing: bool = False
        self.apply(self._init_weights)
        # Deep pre-norm transformers need depth-scaled residual projections.
        # Initializing every projection at 0.02 causes residual variance to
        # accumulate across depth before training can correct it.
        residual_std = 0.02 / math.sqrt(2 * max(1, n_layer))
        for block in self.blocks:
            nn.init.normal_(block.attn.out_proj.weight, mean=0.0, std=residual_std)
            if isinstance(block.mlp, SparseUpcycledMoE):
                nn.init.normal_(
                    block.mlp.shared_expert.down_proj.weight,
                    mean=0.0,
                    std=residual_std,
                )
                block.mlp.reset_from_shared()
            else:
                nn.init.normal_(block.mlp.down_proj.weight, mean=0.0, std=residual_std)
        self.initialization_scheme = "depth_scaled_residual_v1"
        self.esv_module = ESVModule(d_model=n_embd, d_esv=min(64, n_embd))
        esv_dim = min(64, n_embd)
        self.rim_modules = nn.ModuleList(
            [ResidualIdentityModulator(n_embd, esv_dim) for _ in range(n_layer)]
            if self.use_rim
            else []
        )
        self.residual_depth_logits = nn.Parameter(torch.zeros(n_layer))
        if self.use_dstp:
            # Native controls enter a trained dense backbone without changing
            # its function. Per-layer temperature structure must be learned,
            # not imposed as an unverified depth schedule at activation time.
            initial_temperatures = torch.ones(n_layer, dtype=torch.float32)
            self.dstp_temperature_log = nn.Parameter(initial_temperatures.log())
            self.register_buffer(
                "dstp_temperature_initial",
                initial_temperatures.log().clone(),
                persistent=False,
            )
        if self.use_hal:
            if hal_module is not None:
                self.hal_module = hal_module
            elif HALModule is not None:
                # AN: HAL is optional so existing V2 checkpoints keep loading unchanged.
                self.hal_module = HALModule()
            else:
                self.use_hal = False
        if self.use_layer_temperature_bias:
            # Positive, bounded per-layer multiplier. Log-space keeps the neutral
            # initialization exact and prevents sign flips in attention temperature.
            self.layer_temperature_bias_log = nn.Parameter(torch.zeros(n_layer))

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def model_config(self) -> dict[str, object]:
        return {
            "architecture_version": self.architecture_version,
            "vocab_size": self.vocab_size,
            "pad_token_id": self.pad_token_id,
            "n_embd": self.n_embd,
            "n_head": self.n_head,
            "n_layer": self.n_layer,
            "block_size": self.block_size,
            "rms_norm_eps": self.rms_norm_eps,
            "dropout": self.dropout,
            "d_ff": self.d_ff,
            "rope_base": self.rope_base,
            "n_kv_head": self.n_kv_head,
            "mod_layers": self.mod_layers,
            "base_seq_len": self.base_seq_len,
            "target_seq_len": self.target_seq_len,
            "use_layer_temperature_bias": self.use_layer_temperature_bias,
            "use_hal": self.use_hal,
            "use_rim": self.use_rim,
            "use_dstp": self.use_dstp,
            "use_mod": self.use_mod,
            "use_esv_control": self.use_esv_control,
            "use_residual_depth": self.use_residual_depth,
            "use_qk_norm": self.use_qk_norm,
            "sliding_window": self.sliding_window or 0,
            "full_attention_every": self.full_attention_every,
            "use_mtp": self.use_mtp,
            "mtp_depth": self.mtp_depth,
            "mtp_loss_weight": self.mtp_loss_weight,
            "use_moe": self.use_moe,
            "moe_routed_experts": 8 if self.use_moe else 0,
            "moe_top_k": 2 if self.use_moe else 0,
            "approved_subsystems": list(self.approved_subsystems),
            "initialization_scheme": self.initialization_scheme,
        }

    def configure_attention(
        self,
        *,
        use_qk_norm: bool,
        sliding_window: int | None,
        full_attention_every: int,
    ) -> None:
        """Apply an explicit, checkpoint-recorded attention experiment."""
        self.use_qk_norm = bool(use_qk_norm)
        self.sliding_window = int(sliding_window) if sliding_window else None
        self.full_attention_every = int(full_attention_every)
        for index, block in enumerate(self.blocks):
            block.attn.use_qk_norm = self.use_qk_norm
            block.attn.sliding_window = (
                None
                if self.sliding_window is None
                or (
                    self.full_attention_every > 0
                    and (index + 1) % self.full_attention_every == 0
                )
                else self.sliding_window
            )

    def configure_runtime_mode(self, mode: str) -> dict[str, bool]:
        if mode not in {"diagnostic", "native", "full_system"}:
            raise ValueError(f"unknown runtime mode: {mode}")
        previous = {
            "use_mod": self.use_mod,
            "use_rim": self.use_rim,
            "use_dstp": self.use_dstp,
            "use_esv_control": self.use_esv_control,
            "use_residual_depth": self.use_residual_depth,
            "use_hal": self.use_hal,
            "use_layer_temperature_bias": self.use_layer_temperature_bias,
        }
        enabled = set(self.approved_subsystems) if mode != "diagnostic" else set()
        self._set_subsystem_activation(enabled)
        # HAL remains an external, evidence-driven runtime controller. It may
        # influence bounded sampling/memory policy, but an untrained heuristic
        # must never perturb the transformer's attention activations.
        self.use_hal = False
        return previous

    def restore_runtime_mode(self, state: dict[str, bool]) -> None:
        for name, value in state.items():
            setattr(self, name, bool(value))

    def begin_subsystem_trace(self, *, civ_similarity: float | None = None) -> None:
        """Reset execution counters for one request and bind its CIV evidence."""
        for name in self._subsystem_execution:
            self._subsystem_execution[name] = 0
        self._runtime_civ_similarity = (
            1.0 if civ_similarity is None else max(0.0, min(1.0, float(civ_similarity)))
        )
        for router in self.mod_routers.values():
            router.telemetry_enabled = True

    def end_subsystem_trace(self) -> None:
        """Disable optional router instrumentation after a request completes."""
        for router in self.mod_routers.values():
            router.telemetry_enabled = False

    def neutralize_subsystem(self, name: str) -> None:
        controls = {
            "mod": ("use_mod",),
            "rim": ("use_rim",),
            "dstp": (
                "use_dstp",
                "use_residual_depth",
                "use_layer_temperature_bias",
            ),
            "esv": ("use_esv_control",),
            "hal": ("use_hal",),
        }
        if name not in controls:
            raise ValueError(f"unknown subsystem ablation: {name}")
        for attribute in controls[name]:
            setattr(self, attribute, False)

    def _set_subsystem_activation(self, requested: set[str]) -> None:
        self.use_mod = "mod" in requested
        self.use_rim = "rim" in requested
        self.use_dstp = "dstp" in requested
        self.use_residual_depth = "dstp" in requested
        self.use_layer_temperature_bias = "dstp" in requested
        self.use_esv_control = "esv" in requested
        self.use_hal = False

    def configure_subsystems(
        self,
        enabled: Iterable[str],
        *,
        approve: bool = True,
    ) -> dict[str, bool]:
        """Apply and optionally record an explicit trained subsystem recipe."""
        requested = {str(name).strip().lower() for name in enabled}
        known = {"mod", "rim", "dstp", "esv"}
        unknown = requested - known
        if unknown:
            raise ValueError(f"unknown native subsystems: {sorted(unknown)}")
        self._set_subsystem_activation(requested)
        if approve:
            self.approved_subsystems = tuple(sorted(requested))
        return {
            "mod": self.use_mod,
            "rim": self.use_rim,
            "dstp": self.use_dstp,
            "esv": self.use_esv_control,
            "hal": False,
        }

    def _residual_scale(self, layer_idx: int) -> torch.Tensor | float:
        if not self.use_residual_depth:
            return 1.0
        return 2.0 * torch.sigmoid(self.residual_depth_logits[layer_idx])

    def _dstp_temperature(self, layer_idx: int) -> torch.Tensor | float:
        if not self.use_dstp:
            return 1.0
        return self.dstp_temperature_log[layer_idx].exp().clamp(0.5, 2.0)

    def _layer_temperature_bias(self, layer_idx: int) -> torch.Tensor | float:
        if not self.use_layer_temperature_bias:
            return 1.0
        return self.layer_temperature_bias_log[layer_idx].exp().clamp(0.5, 2.0)

    def _router_context(self, x: torch.Tensor, esv_state: torch.Tensor) -> RouterContext:
        probabilities = torch.softmax(x.float(), dim=-1)
        entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(dim=-1)
        entropy = entropy / max(1.0, math.log(x.shape[-1]))
        arousal = (
            esv_state[..., 1].unsqueeze(-1)
            if self.use_esv_control
            else torch.zeros(
                (x.shape[0], 1),
                device=x.device,
                dtype=x.dtype,
            )
        )
        return RouterContext(
            esv_arousal=arousal,
            token_entropy=entropy.to(dtype=x.dtype),
            civ_similarity=self._runtime_civ_similarity,
        )

    def set_mod_capacity(self, capacity: float) -> None:
        bounded = max(0.05, min(1.0, float(capacity)))
        for router in self.mod_routers.values():
            router.capacity = bounded

    def native_regularization_loss(self) -> torch.Tensor:
        device = self.token_embedding_table.weight.device
        loss = torch.zeros((), device=device)
        for router in self.mod_routers.values():
            if router._last_balance_loss is not None:
                loss = loss + 0.01 * router._last_balance_loss
            if router._last_z_loss is not None:
                loss = loss + 0.001 * router._last_z_loss
        residual_scales = 2.0 * torch.sigmoid(self.residual_depth_logits)
        loss = loss + 0.001 * (residual_scales - 1.0).pow(2).mean()
        if self.use_dstp:
            loss = (
                loss
                + 0.001 * (self.dstp_temperature_log - self.dstp_temperature_initial).pow(2).mean()
            )
        if self.use_layer_temperature_bias:
            loss = loss + 0.001 * self.layer_temperature_bias_log.pow(2).mean()
        if self.use_esv_control:
            loss = loss + 0.001 * self.esv_module.temporal_consistency_loss()
        return loss

    def subsystem_telemetry(self) -> dict[str, object]:
        return {
            "execution": dict(self._subsystem_execution),
            "router_civ_similarity": float(self._runtime_civ_similarity),
            "mod": {key: router.telemetry() for key, router in self.mod_routers.items()},
            "dstp_temperatures": [
                float(value) for value in self.dstp_temperature_log.detach().exp().cpu()
            ]
            if self.use_dstp
            else [],
            "layer_temperature_biases": [
                float(value)
                for value in self.layer_temperature_bias_log.detach().exp().clamp(0.5, 2.0).cpu()
            ]
            if self.use_layer_temperature_bias
            else [],
            "residual_scales": [
                float(value)
                for value in (2.0 * torch.sigmoid(self.residual_depth_logits.detach())).cpu()
            ],
        }

    def enable_kv_cache(
        self,
        *,
        backend: str = "float",
        turboquant_bits: int = 4,
    ) -> None:
        """Enable an exact or compressed inference cache.

        The compressed backend is a reversible serving pilot. It never affects
        training weights or checkpoint compatibility.
        """
        if backend not in {"float", "turboquant"}:
            raise ValueError("KV cache backend must be 'float' or 'turboquant'")
        if self.training:
            raise RuntimeError("KV cache must not be enabled while the model is training")
        compressed_cache = None
        if backend == "turboquant":
            from inference.turboquant import TorchTurboQuantCache, TurboQuantConfig

            compressed_cache = (TorchTurboQuantCache, TurboQuantConfig(bits=turboquant_bits))
        for i, block in enumerate(self.blocks):
            if compressed_cache is None:
                block.attn._kv_cache = {}
            else:
                cache_type, config = compressed_cache
                cache_limit = block.attn.max_cache_len
                if block.attn.sliding_window is not None:
                    cache_limit = min(cache_limit, block.attn.sliding_window)
                block.attn._kv_cache = cache_type(
                    num_kv_heads=block.attn.n_kv_head,
                    max_seq_len=cache_limit,
                    d_head=block.attn.head_dim,
                    config=config,
                )
            block.attn._layer_idx = i

    def disable_kv_cache(self) -> None:
        for block in self.blocks:
            block.attn._kv_cache = None

    def clear_kv_cache(self) -> None:
        """Call between independent generation calls."""
        for block in self.blocks:
            if block.attn._kv_cache is not None:
                if isinstance(block.attn._kv_cache, dict):
                    block.attn._kv_cache.clear()
                else:
                    block.attn._kv_cache.reset()

    def kv_cache_telemetry(self) -> dict[str, object]:
        """Aggregate physical cache storage and distortion across layers."""

        reports = []
        for block in self.blocks:
            cache = block.attn._kv_cache
            if cache is not None and not isinstance(cache, dict):
                reports.append(cache.memory_report())
        if not reports:
            return {
                "backend": "float",
                "compressed_bytes": 0,
                "uncompressed_bytes": 0,
                "memory_saved_bytes": 0,
                "compression_ratio": 1.0,
                "layers": 0,
            }
        compressed = sum(int(report["compressed_bytes"]) for report in reports)
        uncompressed = sum(int(report["uncompressed_bytes"]) for report in reports)
        return {
            "backend": "turboquant",
            "algorithm": reports[0]["algorithm"],
            "paper_complete": False,
            "qjl_fused": False,
            "bits": reports[0]["bits"],
            "compressed_bytes": compressed,
            "uncompressed_bytes": uncompressed,
            "memory_saved_bytes": max(0, uncompressed - compressed),
            "compression_ratio": uncompressed / compressed if compressed else 0.0,
            "max_relative_mse": max(float(report["max_relative_mse"]) for report in reports),
            "layers": len(reports),
        }

    def get_hidden_states(self, idx: torch.Tensor) -> list[torch.Tensor]:
        """Return the residual stream after each transformer block."""
        was_training = self.training
        self.eval()
        hidden_states: list[torch.Tensor] = []
        try:
            with torch.no_grad():
                x = self.embed(idx)
                for i, block in enumerate(self.blocks):
                    key = str(i)
                    mod_router = (
                        self.mod_routers[key] if self.use_mod and key in self.mod_routers else None
                    )
                    native_context = self.use_esv_control or self.use_rim or mod_router is not None
                    if native_context:
                        esv_state = self.esv_module(x)
                        esv_channel = self.esv_module.extract_channel(x)
                        self._subsystem_execution["esv"] += 1
                    else:
                        esv_state = torch.zeros((x.shape[0], 3), device=x.device, dtype=x.dtype)
                        esv_channel = None
                    if self.use_rim:
                        assert esv_channel is not None
                        x = self.rim_modules[i](x, esv_channel)
                        self._subsystem_execution["rim"] += 1
                    if self.use_hal and hasattr(self, "hal_module"):
                        attention_temperature = self.hal_module.attention_temperature_tensor(
                            device=x.device,
                            dtype=x.dtype,
                        )
                        self._subsystem_execution["hal"] += 1
                    elif self.use_esv_control:
                        attention_temperature = self.esv_module.attention_temperature_tensor(
                            esv_state
                        )
                    else:
                        attention_temperature = torch.ones((), device=x.device, dtype=x.dtype)
                    attention_temperature = attention_temperature * self._layer_temperature_bias(i)
                    if self.use_dstp:
                        self._subsystem_execution["dstp"] += 1
                    attention_temperature = attention_temperature * self._dstp_temperature(i)
                    if mod_router is not None:
                        self._subsystem_execution["mod"] += 1
                    router_context = self._router_context(x, esv_state)
                    x = block(
                        x,
                        attention_temperature=attention_temperature,
                        mod_router=mod_router,
                        router_context=router_context,
                        residual_scale=self._residual_scale(i),
                    )
                    hidden_states.append(x.detach().cpu())
        finally:
            self.train(was_training)
        return hidden_states

    def layer_norms(self) -> list[float]:
        """Return the mean L2 residual norm at each layer for a dummy input."""
        device = self.token_embedding_table.weight.device
        dummy = torch.zeros(1, min(8, self.block_size), dtype=torch.long, device=device)
        return [state.norm(dim=-1).mean().item() for state in self.get_hidden_states(dummy)]

    def gradient_checkpointing_enable(self) -> None:
        """Enable gradient checkpointing to trade compute for VRAM."""
        self.use_gradient_checkpointing = True

    def gradient_checkpointing_disable(self) -> None:
        self.use_gradient_checkpointing = False

    def embed(self, idx: torch.Tensor) -> torch.Tensor:
        """Expose canonical token embedding for milestone reasoning wrappers."""
        return self.token_embedding_table(idx) * self.embedding_input_scale

    def attach_cognitive_extension(self, extension: nn.Module) -> None:
        """Attach a separately counted cognitive extension to the base model."""
        extension_width = int(getattr(extension, "d_model", -1))
        if extension_width != self.n_embd:
            raise ValueError(
                f"cognitive extension width {extension_width} != model width {self.n_embd}"
            )
        self.cognitive_extension = extension

    def detach_cognitive_extension(self) -> nn.Module | None:
        extension = self.cognitive_extension
        self.cognitive_extension = None
        self._last_cognitive_evidence = []
        return extension

    def base_parameter_count(self) -> int:
        """Count transformer parameters without separately packaged extensions."""
        return sum(
            parameter.numel()
            for name, parameter in self.named_parameters()
            if not name.startswith("cognitive_extension.")
        )

    def cognitive_parameter_count(self) -> int:
        extension = self.cognitive_extension
        return 0 if extension is None else sum(p.numel() for p in extension.parameters())

    def _apply_cognitive_extension(
        self,
        x: torch.Tensor,
        layer_index: int,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        extension = self.cognitive_extension
        if extension is None:
            return x
        x, evidence = extension.apply_layer(
            x,
            layer_index,
            attention_mask=attention_mask,
        )
        if evidence:
            self._last_cognitive_evidence.append(evidence)
        return x

    def run_all_layers(
        self,
        x: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run residual stream with optional gradient checkpointing."""
        self._last_cognitive_evidence = []
        for i, block in enumerate(self.blocks):
            key = str(i)
            mod_router = self.mod_routers[key] if self.use_mod and key in self.mod_routers else None
            native_context = self.use_esv_control or self.use_rim or mod_router is not None

            use_checkpoint = (
                self.use_gradient_checkpointing and self.training and x.device.type != "xla"
            )
            if use_checkpoint:
                # Recompute temperature from the checkpointed residual stream during backward.
                use_hal = self.use_hal and hasattr(self, "hal_module")
                hal_mod = self.hal_module if use_hal else None
                esv_mod = self.esv_module
                bias_i = (
                    self._layer_temperature_bias(i) if self.use_layer_temperature_bias else None
                )
                rim_i = self.rim_modules[i] if self.use_rim else None
                scale_i = self._residual_scale(i)

                def _block_fn(
                    x_: torch.Tensor,
                    b_: BlockV2 = block,
                    esv_: ESVModule = esv_mod,
                    hal_: object = hal_mod,
                    bias_: torch.Tensor | None = bias_i,
                    mr_: MoDRouter | None = mod_router,
                    rim_: ResidualIdentityModulator | None = rim_i,
                    scale_: torch.Tensor | float = scale_i,
                    layer_i_: int = i,
                    native_context_: bool = native_context,
                ) -> torch.Tensor:
                    if native_context_:
                        esv_state_ = esv_(x_)
                        esv_channel_ = esv_.extract_channel(x_)
                    else:
                        esv_state_ = torch.zeros((x_.shape[0], 3), device=x_.device, dtype=x_.dtype)
                        esv_channel_ = None
                    if rim_ is not None:
                        assert esv_channel_ is not None
                        x_ = rim_(x_, esv_channel_)
                    if hal_ is not None:
                        at_ = hal_.attention_temperature_tensor(
                            device=x_.device,
                            dtype=x_.dtype,
                        )
                    elif self.use_esv_control:
                        at_ = esv_.attention_temperature_tensor(esv_state_)
                    else:
                        at_ = torch.ones((), device=x_.device, dtype=x_.dtype)
                    if bias_ is not None:
                        at_ = at_ * bias_
                    at_ = at_ * self._dstp_temperature(layer_i_)
                    router_context_ = self._router_context(x_, esv_state_)
                    return b_(
                        x_,
                        attention_temperature=at_,
                        mod_router=mr_,
                        router_context=router_context_,
                        residual_scale=scale_,
                    )

                if self.use_esv_control:
                    self._subsystem_execution["esv"] += 1
                if native_context:
                    self._subsystem_execution["esv_features"] += 1
                if rim_i is not None:
                    self._subsystem_execution["rim"] += 1
                if use_hal:
                    self._subsystem_execution["hal"] += 1
                if self.use_dstp:
                    self._subsystem_execution["dstp"] += 1
                if mod_router is not None:
                    self._subsystem_execution["mod"] += 1
                x = _torch_checkpoint(
                    _block_fn,
                    x,
                    use_reentrant=False,
                    context_fn=partial(_checkpoint_contexts, rim_i),
                )
            else:
                if native_context:
                    esv_state = self.esv_module(x)
                    esv_channel = self.esv_module.extract_channel(x)
                    self._subsystem_execution["esv_features"] += 1
                    if self.use_esv_control:
                        self._subsystem_execution["esv"] += 1
                else:
                    esv_state = torch.zeros((x.shape[0], 3), device=x.device, dtype=x.dtype)
                    esv_channel = None
                if self.use_rim:
                    assert esv_channel is not None
                    x = self.rim_modules[i](x, esv_channel)
                    self._subsystem_execution["rim"] += 1
                if self.use_hal and hasattr(self, "hal_module"):
                    attention_temperature = self.hal_module.attention_temperature_tensor(
                        device=x.device,
                        dtype=x.dtype,
                    )
                    self._subsystem_execution["hal"] += 1
                elif self.use_esv_control:
                    attention_temperature = self.esv_module.attention_temperature_tensor(esv_state)
                else:
                    attention_temperature = torch.ones((), device=x.device, dtype=x.dtype)
                attention_temperature = attention_temperature * self._layer_temperature_bias(i)
                if self.use_dstp:
                    self._subsystem_execution["dstp"] += 1
                attention_temperature = attention_temperature * self._dstp_temperature(i)
                if mod_router is not None:
                    self._subsystem_execution["mod"] += 1
                router_context = self._router_context(x, esv_state)
                x = block(
                    x,
                    attention_temperature=attention_temperature,
                    mod_router=mod_router,
                    router_context=router_context,
                    residual_scale=self._residual_scale(i),
                )
            x = self._apply_cognitive_extension(x, i, attention_mask)

        x = self.norm_f(x)
        if self.use_esv_control:
            self._last_esv_prediction = self.esv_module(x)
            self._subsystem_execution["esv"] += 1
            if not self.training:
                self._pending_esv_state = self._last_esv_prediction.detach()
        else:
            self._last_esv_prediction = torch.zeros((x.shape[0], 3), device=x.device, dtype=x.dtype)
            self._pending_esv_state = None
        return x

    @torch.no_grad()
    def commit_pending_esv_state(self) -> bool:
        state = getattr(self, "_pending_esv_state", None)
        if state is None:
            return False
        self.esv_module.commit_state(state)
        self._pending_esv_state = None
        return True

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, seq_len = idx.shape
        if seq_len > self.block_size:
            raise ValueError(f"sequence length {seq_len} exceeds block size {self.block_size}")
        x = self.run_all_layers(self.embed(idx))
        self._last_hidden = x
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            bsz, time_steps, channels = logits.shape
            loss = F.cross_entropy(
                logits.view(bsz * time_steps, channels),
                targets.view(bsz * time_steps),
                ignore_index=self.pad_token_id,
            )
        return logits, loss

    def multi_token_prediction_loss(self, targets: torch.Tensor) -> torch.Tensor:
        """Predict horizons +2..+(depth+1) from the shared final residual.

        ``targets[:, i]`` is already the next token for input position ``i``;
        head zero therefore aligns with ``targets[:, 1:]`` (+2), head one
        with ``targets[:, 2:]`` (+3), and so on. Heads share the canonical
        embedding matrix as their vocabulary projection.
        """
        hidden = self._last_hidden
        if not self.use_mtp or self.mtp_depth == 0:
            return torch.zeros(
                (), device=targets.device, dtype=self.token_embedding_table.weight.dtype
            )
        if hidden is None or hidden.shape[:2] != targets.shape:
            raise RuntimeError("MTP loss requires targets from the immediately preceding forward")
        losses: list[torch.Tensor] = []
        for offset, (norm, projection) in enumerate(
            zip(self.mtp_norms, self.mtp_projections, strict=True), start=1
        ):
            if targets.size(1) <= offset:
                continue
            future_targets = targets[:, offset:]
            if not bool((future_targets != self.pad_token_id).any()):
                continue
            future_hidden = projection(norm(hidden[:, :-offset, :]))
            future_logits = F.linear(future_hidden, self.token_embedding_table.weight)
            losses.append(
                F.cross_entropy(
                    future_logits.reshape(-1, self.vocab_size),
                    future_targets.reshape(-1),
                    ignore_index=self.pad_token_id,
                )
            )
        if not losses:
            return hidden.sum() * 0.0
        return self.mtp_loss_weight * torch.stack(losses).mean()

    @torch.no_grad()
    def update_moe_balance(self) -> None:
        """Advance aux-loss-free expert balancing at an optimizer boundary."""
        for block in self.blocks:
            if isinstance(block.mlp, SparseUpcycledMoE):
                block.mlp.update_balance()

    def forward_cognitive(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        *,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        tuple[dict[str, torch.Tensor], ...],
    ]:
        """Forward with typed extension evidence while preserving ``forward``."""
        _, seq_len = idx.shape
        if seq_len > self.block_size:
            raise ValueError(f"sequence length {seq_len} exceeds block size {self.block_size}")
        x = self.run_all_layers(self.embed(idx), attention_mask=attention_mask)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            bsz, time_steps, channels = logits.shape
            loss = F.cross_entropy(
                logits.view(bsz * time_steps, channels),
                targets.view(bsz * time_steps),
                ignore_index=self.pad_token_id,
            )
        return logits, loss, tuple(self._last_cognitive_evidence)

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        *,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Autoregressively sample tokens from the model."""
        for _ in range(int(max_new_tokens)):
            idx_cond = idx[:, -self.block_size :]
            logits, _ = self(idx_cond)
            self.commit_pending_esv_state()
            logits = logits[:, -1, :]
            if temperature <= 0:
                next_id = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / float(temperature)
                if top_k is not None:
                    k = min(int(top_k), logits.size(-1))
                    values, _ = torch.topk(logits, k)
                    logits = logits.masked_fill(logits < values[:, [-1]], float("-inf"))
                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)
        return idx


CausalTransformer = CausalTransformerV2
CausalTransformerV4 = CausalTransformerV2
MetacognitiveRouter = MoDRouter
