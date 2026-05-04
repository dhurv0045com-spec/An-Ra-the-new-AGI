"""An-Ra mainline model and tokenizer definitions.

Canonical exports:
- CausalTransformer          -> V2 mainline decoder
- CausalTransformerV2        -> explicit V2 class name
- CharTokenizer             -> legacy char tokenizer (kept for compatibility)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import functional as F

from identity.esv import ESVModule
from tokenizer.char_tokenizer import CharTokenizer


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
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
                and self._cached_cos.dtype == dtype):
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

    def forward(self, x: torch.Tensor, *, attention_temperature: torch.Tensor | float | None = None) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        q = self.q_proj(x).view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.n_kv_head, self.head_dim).transpose(1, 2)
        q, k = self.rope(q, k)
        if attention_temperature is not None:
            temperature = torch.as_tensor(attention_temperature, dtype=q.dtype, device=q.device).clamp_min(0.25)
            q = q / temperature
        if self.groups > 1:
            k = k.repeat_interleave(self.groups, dim=1)
            v = v.repeat_interleave(self.groups, dim=1)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
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


class MoDRouter(nn.Module):
    def __init__(self, d_model: int, capacity: float = 0.5):
        super().__init__()
        self.capacity = capacity
        self.gate = nn.Linear(d_model, 1, bias=False)
        nn.init.zeros_(self.gate.weight)

    def forward(self, x: torch.Tensor, ffn: nn.Module) -> torch.Tensor:
        B, n, d = x.shape
        k = max(1, int(n * self.capacity))
        scores = self.gate(x).squeeze(-1)
        topk = scores.topk(k, dim=-1).indices
        idx_exp = topk.unsqueeze(-1).expand(-1, -1, d)
        x_sel = x.gather(1, idx_exp)
        x_proc = x_sel + ffn(x_sel)
        out = x.clone()
        out.scatter_(1, idx_exp, x_proc)
        return out


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
    ) -> torch.Tensor:
        x = x + self.attn(self.norm_1(x), attention_temperature=attention_temperature)
        if mod_router is not None:
            x = mod_router(x, self._normed_mlp)
            return x
        x = x + self.mlp(self.norm_2(x))
        return x


class CausalTransformerV2(nn.Module):
    def __init__(self, vocab_size: int, n_embd: int, n_head: int, n_layer: int, block_size: int, *, n_kv_head: int | None = None, rms_norm_eps: float = 1e-5, dropout: float = 0.0, mod_layers=(), base_seq_len: int = 512, target_seq_len: int = 2048, pad_token_id: int = 0):
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
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.blocks = nn.ModuleList([BlockV2(n_embd, n_head, n_kv_head=self.n_kv_head, eps=rms_norm_eps, dropout=dropout, base_seq_len=base_seq_len, target_seq_len=target_seq_len) for _ in range(n_layer)])
        self.mod_routers = nn.ModuleDict({str(i): MoDRouter(n_embd) for i in mod_layers})
        self.norm_f = RMSNorm(n_embd, eps=rms_norm_eps)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding_table.weight
        self.apply(self._init_weights)
        self.esv_module = ESVModule(d_model=n_embd, d_esv=min(64, n_embd))

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def model_config(self) -> dict[str, int]:
        return {"vocab_size": self.vocab_size, "pad_token_id": self.pad_token_id, "n_embd": self.n_embd, "n_head": self.n_head, "n_layer": self.n_layer, "block_size": self.block_size, "n_kv_head": self.n_kv_head, "base_seq_len": self.base_seq_len, "target_seq_len": self.target_seq_len}

    def embed(self, idx: torch.Tensor) -> torch.Tensor:
        """Expose canonical token embedding for milestone reasoning wrappers."""
        return self.token_embedding_table(idx)

    def run_all_layers(self, x: torch.Tensor) -> torch.Tensor:
        """Run residual stream. ESV updates per block for layer-wise temperature."""
        for i, block in enumerate(self.blocks):
            esv_state = self.esv_module(x)
            attention_temperature = self.esv_module.attention_temperature_tensor(esv_state)
            key = str(i)
            mod_router = self.mod_routers[key] if key in self.mod_routers else None
            x = block(x, attention_temperature=attention_temperature, mod_router=mod_router)
        x = self.norm_f(x)
        return x

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


CausalTransformer = CausalTransformerV2
