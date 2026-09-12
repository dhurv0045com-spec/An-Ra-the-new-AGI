# CURRENT V5 AUDIT (executable architecture, from code)

**Method:** reconstructed from the live source on `cymek-500m-readiness` @ `f2c27a6` (identical content on this branch), not from prose. Every claim cites a file.

## 1. Mechanical inventory

| Item | Value (code) | Source | Code/doc agree? |
|---|---|---|---|
| family | dense causal decoder-only Transformer, block-diagonal packed-segment attention with per-segment RoPE resets | `v5_model/core.py` (`packed_layout`) | ✓ |
| parameter count (V5-A) | **250,216,960** exactly; asserted at construction against the spec receipt | `v5_contracts/model_spec.py:75-96` formula; `v5_model/core.py:assert_receipt`; reproduced independently by `tools/next_core_compute_model.py` | ✓ (docs, receipts, and this audit agree; an earlier Phase-2 hand-calc discrepancy was the auditor's arithmetic, not the code's) |
| layers | 26 | `v5_contracts/model_spec.py:113` | ✓ |
| d_model (width) | 896 | `:112` | ✓ |
| Q heads / KV heads | 14 / 7 (GQA 2:1, repeat_interleave) | `:114-115`; `v5_model/attention.py:67-69` | ✓ |
| head dimension | 64 (width == q_heads × head_dim enforced) | `:116`, `:64` | ✓ |
| FFN | SwiGLU 3×896×2,368 = 6,365,184/layer, no bias | `v5_model/block.py:42-50` | ✓ |
| activation | SiLU gate × up, then down | `block.py:49-50` | ✓ |
| normalization | pre-RMSNorm (fp32 statistics) ×2/layer + final RMSNorm | `block.py:10-24,37-41`; `core.py:40-41` | ✓ |
| Q/K normalization | affine per-head RMS, eps 1e-6 (repo constant `QK_NORM_EPSILON`), scales shape [heads, 64] | `attention.py:23-42`; `model_spec.py:9` | ✓ |
| position encoding | pairwise RoPE, base 10,000, applied post-QK-norm | `attention.py:44-55,65-66` | ✓ |
| context length | 4,096 native (forward hard-checks) | `core.py:45` | ✓ |
| embedding/output tying | exactly one embedding table; output = `linear(final_norm(h), embedding.weight)`; separate heads rejected by name | `core.py:54,65-76` | ✓ |
| vocabulary | 24,576 (frozen BPE artifact, `artifacts/e1/local_tournament/tokenizer-24576.json.gz`) | spec + evidence ledger | ✓ |
| initialization | `Normal(0, 0.02)` embeddings/QKV/gate/up; residual outputs `Normal(0, 0.02/√(2L))` | `v5_model/initialize.py` | ✓ |
| precision | one persistent FP32 parameter set, BF16 autocast compute, FP32 logits/loss/grad-norm reductions, FP32 Adam moments | `v5_training/production_backend.py`, `step.py` (D-026/D-028) | ✓ |
| optimizer | AdamW β(0.9,0.95), eps 1e-8, wd 0.1 on ndim≥2, no decay on norms/QK scales, ownership-verified | `v5_training/optimizer.py` | ✓ |
| schedule | token-indexed WSD (0→3e-4/50M, decay→3e-5/5B) unit-tested; **every executed run to date used bounded warmup** (citadel audit) | `v5_training/schedule.py` | ✓ with caveat |
| gradient clipping | global L2 1.0, post-clip certification, fp32-reduction tolerance 1e-4 (R1C repair, single-sourced `v5_training/step.py`) | `step.py`; RUN_READINESS_V4 | ✓ |
| objective | causal CE over eligible targets; BOS/PAD/segment-boundary excluded; **EOS included**; fp32 reduction | `v5_objectives/` + evidence ledger OBJECTIVE component | ✓ |
| EOS contract | answer+EOS supervised (EVIDENCE-LOCKED cross-program) | ledger B20 | ✓ |
| checkpoint representation | content-addressed store, parent-fenced, hash+structural validation, exact-resume (restore ≡ uninterrupted), RNG captured | `v5_training/checkpoint.py`, `state.py`; canaries | ✓ |
| training-state contract | FP32 params, FP32 moments, step, token ledger, sampler cursor, RNG, topology, model/tokenizer/data/pack/source/code identities | `state.py`; durability canaries | ✓ |

## 2. Formula (authoritative, `v5_contracts/model_spec.py:75-96`)

```
embedding       = V × w
attention/layer = w·q_w + 2·w·kv_w + q_w·w          (q_w = Q·hd, kv_w = KV·hd)
ffn/layer       = 3·w·f                             (SwiGLU gate+up+down)
norms/layer     = 2·w  +  (q_w + kv_w if affine QK)
total           = embedding + L·block + w           (tied; no separate head)
```
V5-A: 22,020,096 + 26×(2,408,448 + 6,365,184 + 1,792 + 1,344) + 896 = **250,216,960**.

## 3. Audit findings (documentation vs code, and gaps)

1. **Parameter accounting is exactly consistent** across contract, constructed model, receipts, and docs — verified mechanically, not by trust.
2. **The canonical WSD schedule has never trained a token** — every executed run used `bounded_warmup` (citadel 500M audit; still true at `f2c27a6`). This is a launch-blocking execution gap, not an architecture defect.
3. **The production entry point and 5B corpus remain MISSING** (citadel audit; unchanged).
4. **No separate output-head path exists** — any untied/factorized output would be NEW code (relevant to Candidate B/C).
5. **No output-space treatment exists in V5** — the R1C MASK/OFFSET arms live only in `anra_v5/cyr_gpu014_r1c_run*.py` experiment code (frozen); `v5_next` therefore adds the treatment as EXPERIMENT_ONLY, default-off, hash-visible.
6. **Train-side receipt meta-checks are STALE-BY-DESIGN** after the R1C constant consolidation (documented in EXPERIMENT_LOG); a full-suite refresh is pending.
7. **Config surface is already minimal and fail-closed** (`from_spec` rejects bias/dropout/untied) — V5 structurally resists architecture soup. Keep.
