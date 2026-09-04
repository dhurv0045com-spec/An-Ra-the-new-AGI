# XLA_SAFE_AUDIT.md

Static audit only. No execution. Source pinned: `origin/cymek@4abeaeb`.
Method: `git show origin/cymek:<path>` reads of `v5_model/`, `v5_objectives/`,
`v5_training/`, `v5_data/pack.py`. Labels per `RESEARCH_PROTOCOL.md`.

**FACT.** No `torch_xla` import exists anywhere under `v5_*` at the pinned SHA.
TPU support is a greenfield shim, not a port. The `torch_module`-injection
pattern (`build_attention(config, torch_module=...)`, etc.) is the correct seam:
pass an XLA-bound module rather than rewriting the model. No JAX/TF rewrite.

**FACT.** `production_backend.py` docstring already records the "TPU/XLA
device-movement lesson" (stale-ownership boundary). The lesson is documented;
the XLA execution path is not implemented.

## Classification

| # | Op / site | Verdict | Reason |
|---|---|---|---|
| A1 | RMSNorm (`v5_model/block.py:build_rmsnorm`) — rsqrt/square/mean/float-cast | XLA_NATIVE | Elementwise only, no host sync. |
| A2 | SwiGLU FFN (`block.py:build_block`) — `silu(gate)*up`, `down` | XLA_NATIVE | Elementwise + linear; standard XLA lowering. |
| A3 | RoPE (`v5_model/attention.py:rope`) — arange/phase/cos/sin/stack/flatten | XLA_COMPILES | Compiles. INFERENCE: per-step cos/sin recompute + `arange(device=...)` is recompile-sensitive; precompute or cache per bucket. |
| A4 | QK-norm (`attention.py:normalize`) — float/rsqrt/affine scale | XLA_COMPILES | Fine. Keep bf32/f32 dtype discipline consistent on TPU. |
| A5 | GQA expand (`attention.py:forward`) — `repeat_interleave` | XLA_SLOW | Compilable but known-slow on XLA. Prefer expand/view before SDPA. |
| A6 | SDPA (`attention.py:forward`) — `scaled_dot_product_attention` + 4D bool `attn_mask` (`mask[:, None, :, :]` from `packed_layout`) | XLA_SLOW (verify on device) | Availability/lowering depends on torch-xla version; bool-mask path may go unfused. Needs version-pinned probe on Kaggle + math-fallback path. |
| A7 | Tied embedding + `functional.linear(hidden, embedding.weight)` (`v5_model/core.py:forward`) | XLA_COMPILES | Compiles. Drop any `data_ptr()` identity check on the XLA path (pointers meaningless on XLA). |
| A8 | `packed_layout` (`v5_model/core.py`) — `cummax/where/eye` + `.any().item()` | XLA_GRAPH_BREAK if on-device; HOST_OK if on CPU | `.item()` forces host sync. Resolution (binding): keep `packed_layout` as host preprocessing on CPU; ship only `positions` + `mask` to XLA. Never place inside the compiled step. |
| A9 | `causal_lm_loss` (`v5_objectives/causal_lm.py`) — `keep.sum().item()`, masked CE | XLA_GRAPH_BREAK as written | Loss math is XLA-native; the `.item()` count is the break. Compute `count` on host or via XLA-safe count outside the compiled region. |
| A10 | `query_swap_loss` (`v5_objectives/query_swap.py`) — clamp/max/mean | XLA_NATIVE | Moot: frozen-disabled by design. Do not enable for TPU bring-up. |
| A11 | AdamW (`v5_training/optimizer.py:build_adamw_optimizer` → vanilla `torch.optim.AdamW`; `id()`-based ownership) | UNSUPPORTED on XLA as-is | Requires `torch_xla.amp.syncfree.AdamW` (or XLA-wrapped opt) + `xm.optimizer_step` + `xm.mark_step`. `id()` ownership is invalid across XLA device moves. |
| A12 | Evidence/hashing (`v5_training/production_backend.py:_tensor_sha256` `.cpu().numpy().tobytes()`; `_rng_state_sha256` `torch.cuda.get_rng_state_all`) | UNSUPPORTED on XLA as-is | Per-step `.cpu()` transfers + CUDA RNG branch break graph and perf. Fix: hash on host once per checkpoint, not per step; XLA-safe RNG path; drop `embedding_data_ptr` on XLA. |
| A13 | Checkpoint tx (`v5_training/checkpoint.py`) — file write/fsync/manifest | XLA_COMPILES (host-side) | Host-side, fine. Canonical state from one replica only. |
| A14 | Distributed metadata (`v5_training/distributed.py`) — pure dataclass ledger | XLA_NATIVE (host) | Design already correct: per-rank `token_contribution`, barrier hash, global-sum check. Wire to XLA data-parallel sharding; this is where the no-8x-duplication invariant lives. |
| A15 | Init (`v5_model/initialize.py` — `nn.init.normal_`, `fork_rng`, `manual_seed`) | XLA_SLOW / host-only | Run init on CPU, then move to XLA; seed per-replica explicitly. |

## Binding resolutions (INFERENCE → plan constraints)

1. Host/device split: `packed_layout`, loss `count`, hashing, checkpoint I/O stay on CPU host. Only forward/loss-math/backward/optimizer-step compile on XLA.
2. `torch_module` injection is the port seam. No architecture change for bring-up.
3. Single-bucket (512) fixed-batch bring-up first; 1024/2048/4096 only after `TPU_8_DEVICE.json` passes (see `TPU_MILESTONES.md`).
4. CUDA receipts (`ONE_UPDATE.json` etc.) remain reference only; they certify nothing about XLA.
