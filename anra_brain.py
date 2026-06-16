"""An-Ra mainline model and tokenizer definitions.

Canonical exports:
- CausalTransformer          -> V2 mainline decoder
- CausalTransformerV2        -> explicit V2 class name
- CharTokenizer             -> legacy char tokenizer (kept for compatibility)
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import spectral_norm
from torch.utils.checkpoint import checkpoint as _torch_checkpoint

from anra.core.registry import MODEL_REGISTRY
from identity.esv import ESVModule
try:
    from identity.hal import HALModule
except Exception:  # pragma: no cover - HAL is optional for old runtimes.
    HALModule = None
from tokenizer.char_tokenizer import CharTokenizer


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
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
    def __init__(self, dim: int, base: int = 10000,
                 base_seq_len: int = 512, target_seq_len: int = 2048):
        super().__init__()
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
        dim_threshold = self.dim * math.log(scale) / (2 * math.log(self.base * self.base_seq_len / (2 * math.pi)))
        dim_threshold = max(0, min(self.dim // 2 - 1, int(dim_threshold)))
        scaling = torch.ones(self.dim // 2)
        scaling[:dim_threshold] = 1.0 / scale
        self._attn_scale = 0.1 * math.log(scale) + 1.0
        return inv_freq * scaling

    def _build_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> None:
        if (self._cached_cos is not None
                and self._cached_seq_len >= seq_len
                and self._cached_cos.dtype == dtype
                and self._cached_cos.device == device):
            return
        positions = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(positions, self.inv_freq.to(device))
        emb = torch.cat([freqs, freqs], dim=-1)
        self._cached_cos = emb.cos()[None, None, :, :].to(dtype=dtype)
        self._cached_sin = emb.sin()[None, None, :, :].to(dtype=dtype)
        self._cached_seq_len = seq_len

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = q.size(-2)
        self._build_cache(seq_len, q.device, q.dtype)
        cos = self._cached_cos[..., :seq_len, :]
        sin = self._cached_sin[..., :seq_len, :]
        q = (q * cos) + (_rotate_half(q) * sin)
        k = (k * cos) + (_rotate_half(k) * sin)
        return q, k


class MultiHeadAttentionV2(nn.Module):
    def __init__(self, n_embd: int, n_head: int, n_kv_head: int | None = None,
                 dropout: float = 0.0,
                 base_seq_len: int = 512, target_seq_len: int = 2048):
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError(f"n_embd={n_embd} must be divisible by n_head={n_head}")
        self.n_head = n_head
        self.n_kv_head = n_kv_head if n_kv_head is not None else n_head
        assert self.n_head % self.n_kv_head == 0, f"n_head={n_head} must be divisible by n_kv_head={self.n_kv_head}"
        self.head_dim = n_embd // n_head
        self.groups = self.n_head // self.n_kv_head

        self.q_proj = nn.Linear(n_embd, n_head * self.head_dim, bias=False)
        self.k_proj = nn.Linear(n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.out_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.rope = RotaryEmbedding(self.head_dim, base_seq_len=base_seq_len, target_seq_len=target_seq_len)
        self.dropout = dropout
        self._kv_cache: dict[str, torch.Tensor] | None = None
        self._layer_idx: int = 0
        self.lba_bound = 0.8 * (65504.0 / self.head_dim) ** 0.5

    def forward(self, x: torch.Tensor, *, attention_temperature: torch.Tensor | float | None = None) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        q = self.q_proj(x).view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.n_kv_head, self.head_dim).transpose(1, 2)
        q, k = self.rope(q, k)

        if attention_temperature is not None:
            temperature = torch.as_tensor(attention_temperature, dtype=q.dtype, device=q.device).clamp_min(0.25)
            q = q / temperature

        bound = torch.as_tensor(self.lba_bound, dtype=q.dtype, device=q.device)
        q = bound * torch.tanh(q / bound)
        k = bound * torch.tanh(k / bound)

        if self._kv_cache is not None:
            cache_k = self._kv_cache.get("k")
            cache_v = self._kv_cache.get("v")
            if cache_k is not None and cache_v is not None:
                k = torch.cat([cache_k, k], dim=2)
                v = torch.cat([cache_v, v], dim=2)
            self._kv_cache["k"] = k.detach()
            self._kv_cache["v"] = v.detach()

        is_causal = not (self._kv_cache is not None and q.size(2) == 1)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
            enable_gqa=self.groups > 1,
        )
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.out_proj(out)


class SwiGLU(nn.Module):
    def __init__(self, n_embd: int, hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(n_embd, hidden_dim, bias=False)
        self.up_proj = nn.Linear(n_embd, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


@dataclass(frozen=True)
class RouterContext:
    esv_arousal: torch.Tensor | float = 0.0
    token_entropy: torch.Tensor | float = 0.0
    civ_similarity: torch.Tensor | float = 1.0


class MoDRouter(nn.Module):
    def __init__(self, d_model: int, capacity: float = 0.5):
        super().__init__()
        self.capacity = capacity
        self.gate = nn.Linear(d_model, 1, bias=False)
        self.capacity_control = nn.Parameter(torch.zeros(()))
        self.context_weights = nn.Parameter(torch.zeros(3))
        nn.init.normal_(self.gate.weight, std=0.02)
        # AN: Small normal init avoids tied top-k routing during early training.

    def forward(
        self,
        x: torch.Tensor,
        ffn: nn.Module,
        ctx: RouterContext | None = None,
    ) -> torch.Tensor:
        B, n, d = x.shape
        capacity = float(
            torch.clamp(
                torch.as_tensor(self.capacity, device=x.device)
                + 0.25 * torch.tanh(self.capacity_control.detach()),
                0.05,
                1.0,
            ).item()
        )
        k = max(1, min(n, int(n * capacity)))
        scores = self.gate(x).squeeze(-1)
        topk_vals, topk_idx = scores.topk(k, dim=-1)
        selected = x.gather(1, topk_idx.unsqueeze(-1).expand(-1, -1, d))
        selected_out = ffn(selected)
        routing_weights = torch.softmax(topk_vals, dim=-1).unsqueeze(-1)
        routed = torch.zeros_like(x)
        context_signal = torch.zeros((), device=x.device, dtype=x.dtype)
        if ctx is not None:
            context_values = torch.stack(
                [
                    torch.as_tensor(ctx.esv_arousal, device=x.device, dtype=x.dtype),
                    torch.as_tensor(ctx.token_entropy, device=x.device, dtype=x.dtype),
                    torch.as_tensor(ctx.civ_similarity, device=x.device, dtype=x.dtype),
                ]
            )
            context_signal = torch.dot(self.context_weights.to(dtype=x.dtype), context_values)
        route_strength = 2.0 * torch.sigmoid(self.capacity_control + context_signal)
        routed.scatter_add_(
            1,
            topk_idx.unsqueeze(-1).expand(-1, -1, d),
            route_strength * routing_weights * selected_out,
        )
        return x + routed


class ResidualIdentityModulator(nn.Module):
    """Bounded additive identity path from the reserved ESV residual channels."""

    def __init__(self, d_model: int, d_esv: int = 64) -> None:
        super().__init__()
        self.projection = spectral_norm(nn.Linear(d_esv, d_model, bias=False))
        self.raw_alpha = nn.Parameter(torch.zeros(()))

    def forward(self, x: torch.Tensor, esv_channel: torch.Tensor) -> torch.Tensor:
        alpha = 0.25 * torch.tanh(self.raw_alpha)
        identity_delta = self.projection(esv_channel).view(1, 1, -1)
        return x + alpha * identity_delta


class BlockV2(nn.Module):
    def __init__(self, n_embd: int, n_head: int, n_kv_head: int | None = None, *, eps: float = 1e-5, dropout: float = 0.0, base_seq_len: int = 512, target_seq_len: int = 2048):
        super().__init__()
        hidden_dim = int(8 / 3 * n_embd)
        hidden_dim = (hidden_dim + 63) // 64 * 64
        self.norm_1 = RMSNorm(n_embd, eps=eps)
        self.attn = MultiHeadAttentionV2(n_embd, n_head, n_kv_head=n_kv_head, dropout=dropout, base_seq_len=base_seq_len, target_seq_len=target_seq_len)
        self.norm_2 = RMSNorm(n_embd, eps=eps)
        self.mlp = SwiGLU(n_embd, hidden_dim)
        self._normed_mlp = nn.Sequential(*[self.norm_2, self.mlp])

    def forward(
        self,
        x: torch.Tensor,
        *,
        attention_temperature: torch.Tensor | float | None = None,
        mod_router: MoDRouter | None = None,
        residual_scale: torch.Tensor | float = 1.0,
    ) -> torch.Tensor:
        x = x + residual_scale * self.attn(
            self.norm_1(x), attention_temperature=attention_temperature
        )
        if mod_router is not None:
            routed = mod_router(x, self._normed_mlp)
            x = x + residual_scale * (routed - x)
            return x
        x = x + residual_scale * self.mlp(self.norm_2(x))
        return x


@MODEL_REGISTRY.register("causal_transformer_v3")
@MODEL_REGISTRY.register("causal_transformer_v2")
class CausalTransformerV2(nn.Module):
    def __init__(self, vocab_size: int, n_embd: int, n_head: int, n_layer: int, block_size: int, *, n_kv_head: int | None = None, rms_norm_eps: float = 1e-5, dropout: float = 0.0, mod_layers=(), base_seq_len: int = 512, target_seq_len: int = 2048, pad_token_id: int = 0, use_layer_temperature_bias: bool = True, use_hal: bool = False, hal_module=None, use_rim: bool = True, use_dstp: bool = True):
        super().__init__()
        if not 0 <= pad_token_id < vocab_size:
            raise ValueError(f"pad_token_id={pad_token_id} must be within vocab_size={vocab_size}")
        self.vocab_size = vocab_size
        self.pad_token_id = int(pad_token_id)
        self.n_embd = n_embd
        self.d_model = n_embd
        self.n_head = n_head
        self.n_kv_head = n_kv_head if n_kv_head is not None else n_head
        self.n_layer = n_layer
        self.block_size = block_size
        self.mod_layers = tuple(sorted(mod_layers))
        self.base_seq_len = base_seq_len
        self.target_seq_len = target_seq_len
        self.use_layer_temperature_bias = bool(use_layer_temperature_bias)
        self.use_hal = bool(use_hal)
        self.use_rim = bool(use_rim)
        self.use_dstp = bool(use_dstp)
        self.cognitive_extension = None
        self._last_cognitive_evidence: list[dict[str, torch.Tensor]] = []
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.token_embedding = self.token_embedding_table
        self.register_buffer(
            "embedding_input_scale",
            torch.ones(n_embd),
            persistent=True,
        )
        self.blocks = nn.ModuleList([BlockV2(n_embd, n_head, n_kv_head=self.n_kv_head, eps=rms_norm_eps, dropout=dropout, base_seq_len=base_seq_len, target_seq_len=target_seq_len) for _ in range(n_layer)])
        self.mod_routers = nn.ModuleDict({str(i): MoDRouter(n_embd) for i in mod_layers})
        self.norm_f = RMSNorm(n_embd, eps=rms_norm_eps)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding_table.weight
        self.use_gradient_checkpointing: bool = False
        self.apply(self._init_weights)
        self.esv_module = ESVModule(d_model=n_embd, d_esv=min(64, n_embd))
        esv_dim = min(64, n_embd)
        self.rim_modules = nn.ModuleList(
            [ResidualIdentityModulator(n_embd, esv_dim) for _ in range(n_layer)]
            if self.use_rim
            else []
        )
        self.residual_depth_logits = nn.Parameter(torch.zeros(n_layer))
        if self.use_dstp:
            layer = torch.arange(n_layer, dtype=torch.float32)
            denom = max(1, n_layer - 1)
            initial_temperatures = 0.65 + (1.35 - 0.65) * (
                1.0 + torch.cos(math.pi * layer / denom)
            ) / 2.0
            self.register_buffer(
                "dstp_temperature_log",
                initial_temperatures.log(),
                persistent=True,
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
            # AN: let each block learn how strongly shared ESV arousal should shape its attention.
            self.register_buffer("layer_temperature_bias", torch.ones(n_layer))

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def model_config(self) -> dict[str, int | bool]:
        return {"vocab_size": self.vocab_size, "pad_token_id": self.pad_token_id, "n_embd": self.n_embd, "n_head": self.n_head, "n_layer": self.n_layer, "block_size": self.block_size, "n_kv_head": self.n_kv_head, "base_seq_len": self.base_seq_len, "target_seq_len": self.target_seq_len, "use_layer_temperature_bias": self.use_layer_temperature_bias, "use_hal": self.use_hal, "use_rim": self.use_rim, "use_dstp": self.use_dstp}

    def _residual_scale(self, layer_idx: int) -> torch.Tensor | float:
        return 2.0 * torch.sigmoid(self.residual_depth_logits[layer_idx])

    def _dstp_temperature(self, layer_idx: int) -> torch.Tensor | float:
        if not self.use_dstp:
            return 1.0
        return self.dstp_temperature_log[layer_idx].exp().clamp(0.25, 4.0)

    def enable_kv_cache(self) -> None:
        """Call before inference. Never call during training."""
        for i, block in enumerate(self.blocks):
            block.attn._kv_cache = {}
            block.attn._layer_idx = i

    def disable_kv_cache(self) -> None:
        for block in self.blocks:
            block.attn._kv_cache = None

    def clear_kv_cache(self) -> None:
        """Call between independent generation calls."""
        for block in self.blocks:
            if block.attn._kv_cache is not None:
                block.attn._kv_cache.clear()

    def get_hidden_states(self, idx: torch.Tensor) -> list[torch.Tensor]:
        """Return the residual stream after each transformer block."""
        was_training = self.training
        self.eval()
        hidden_states: list[torch.Tensor] = []
        try:
            with torch.no_grad():
                x = self.embed(idx)
                for i, block in enumerate(self.blocks):
                    esv_state = self.esv_module(x)
                    esv_channel = self.esv_module.extract_channel(x)
                    if self.use_rim:
                        x = self.rim_modules[i](x, esv_channel)
                    if self.use_hal and hasattr(self, "hal_module"):
                        attention_temperature = self.hal_module.attention_temperature_tensor(
                            device=x.device,
                            dtype=x.dtype,
                        )
                    else:
                        attention_temperature = self.esv_module.attention_temperature_tensor(
                            esv_state
                        )
                    if self.use_layer_temperature_bias:
                        attention_temperature = (
                            attention_temperature * self.layer_temperature_bias[i]
                        )
                    attention_temperature = attention_temperature * self._dstp_temperature(i)
                    key = str(i)
                    mod_router = self.mod_routers[key] if key in self.mod_routers else None
                    x = block(
                        x,
                        attention_temperature=attention_temperature,
                        mod_router=mod_router,
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
            mod_router = self.mod_routers[key] if key in self.mod_routers else None

            use_checkpoint = (
                self.use_gradient_checkpointing
                and self.training
                and x.device.type != "xla"
            )
            if use_checkpoint:
                # Recompute temperature from the checkpointed residual stream during backward.
                use_hal = self.use_hal and hasattr(self, "hal_module")
                hal_mod = self.hal_module if use_hal else None
                esv_mod = self.esv_module
                bias_i = (
                    self.layer_temperature_bias[i]
                    if self.use_layer_temperature_bias
                    else None
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
                ) -> torch.Tensor:
                    esv_state_ = esv_(x_)
                    esv_channel_ = esv_.extract_channel(x_)
                    if rim_ is not None:
                        x_ = rim_(x_, esv_channel_)
                    if hal_ is not None:
                        at_ = hal_.attention_temperature_tensor(
                            device=x_.device,
                            dtype=x_.dtype,
                        )
                    else:
                        at_ = esv_.attention_temperature_tensor(esv_state_)
                    if bias_ is not None:
                        at_ = at_ * bias_
                    at_ = at_ * self._dstp_temperature(i)
                    return b_(
                        x_,
                        attention_temperature=at_,
                        mod_router=mr_,
                        residual_scale=scale_,
                    )

                x = _torch_checkpoint(_block_fn, x, use_reentrant=False)
            else:
                esv_state = self.esv_module(x)
                esv_channel = self.esv_module.extract_channel(x)
                if self.use_rim:
                    x = self.rim_modules[i](x, esv_channel)
                if self.use_hal and hasattr(self, "hal_module"):
                    attention_temperature = self.hal_module.attention_temperature_tensor(
                        device=x.device,
                        dtype=x.dtype,
                    )
                else:
                    attention_temperature = self.esv_module.attention_temperature_tensor(esv_state)
                if self.use_layer_temperature_bias:
                    attention_temperature = attention_temperature * self.layer_temperature_bias[i]
                attention_temperature = attention_temperature * self._dstp_temperature(i)
                x = block(
                    x,
                    attention_temperature=attention_temperature,
                    mod_router=mod_router,
                    residual_scale=self._residual_scale(i),
                )
            x = self._apply_cognitive_extension(x, i, attention_mask)

        if not self.training and "esv_state" in locals():
            self._pending_esv_state = esv_state.detach()
        x = self.norm_f(x)
        return x

    @torch.no_grad()
    def commit_pending_esv_state(self) -> bool:
        state = getattr(self, "_pending_esv_state", None)
        if state is None:
            return False
        self.esv_module.commit_state(state)
        self._pending_esv_state = None
        return True

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        _, seq_len = idx.shape
        if seq_len > self.block_size:
            raise ValueError(f"sequence length {seq_len} exceeds block size {self.block_size}")
        x = self.run_all_layers(self.embed(idx))
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            bsz, time_steps, channels = logits.shape
            loss = F.cross_entropy(logits.view(bsz * time_steps, channels), targets.view(bsz * time_steps), ignore_index=self.pad_token_id)
        return logits, loss

    def forward_cognitive(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        *,
        attention_mask: torch.Tensor | None = None,
    ):
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
CausalTransformerV3 = CausalTransformerV2
BlockV3 = BlockV2
MultiHeadAttentionV3 = MultiHeadAttentionV2
MetacognitiveRouter = MoDRouter
