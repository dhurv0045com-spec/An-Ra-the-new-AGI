# XLA_TAXONOMY_MAP.md

Maps the mission §7 taxonomy onto the committed static audit
(`tpu/XLA_SAFE_AUDIT.md`, source `origin/cymek@4abeaeb`). No execution;
this file only aligns vocabulary so later device receipts can use one scheme.

| Mission §7 label | Audit label | Members |
|---|---|---|
| XLA_OK | XLA_NATIVE | A1 RMSNorm, A2 SwiGLU, A10 query-swap math, A14 distributed metadata (host) |
| XLA_RECOMPILES | XLA_COMPILES | A3 RoPE, A4 QK-norm, A7 tied embedding, A13 checkpoint tx (host) |
| XLA_SLOW | XLA_SLOW | A5 GQA `repeat_interleave`, A6 SDPA bool-mask, A15 init (host-only) |
| XLA_UNSUPPORTED | UNSUPPORTED | A11 vanilla AdamW + `id()` ownership, A12 per-step `.cpu()` hash + CUDA RNG + `data_ptr()` |
| NOT_TESTED | — | All device behavior: SDPA lowering per torch-xla version, XLA optimizer numerics, 8-device collectives, bucket recomp counts, host→device feeding |

Unresolved device unknowns (require Kaggle execution, never local simulation):
SDPA fused-vs-fallback path + version pin; XLA AdamW numerics vs CUDA
reference; 512-vs-1024/2048/4096 recomp and HBM cost; steady-state tok/s.
