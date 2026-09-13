# CANARY SPEC (human summary)

**Frozen preregistration (authoritative):** [`experiments/V5_1_CANARY/PREREGISTRATION.json`](../../../experiments/V5_1_CANARY/PREREGISTRATION.json) — committed BEFORE the final canary result. Thresholds were chosen from design reasoning and the shortcut-screen mathematics, never moved after observing performance.

## The two rungs (exact)

| | Rung A (micro/integration, CPU) | Rung B (development GPU, T4) |
|---|---|---|
| geometry | V=24,576, w=256, L=4, Q=4, KV=2, hd=64, ffn=1,024, ctx=1,024 | V=24,576, w=512, L=10, Q=8, KV=4, hd=64, ffn=1,408, ctx=2,048 |
| parameters | **10,227,456** | **42,092,544** |
| preserved semantics | head_dim 64, GQA 2:1, V5 norms, SwiGLU, pairwise RoPE 10k, tied output, canonical class space | same |
| role | repeated mechanical qualification | production-path CUDA rehearsal |

## Data instrument (not production data)

Deterministic generator `anra_v5/v51_canary_data.py` (version `v51-canary-data/v1`), seed 20260913, 300 latent worlds per family across six families (identity/copy, query-conditioned binding, state/order, transitive composition, termination/counting, missing-information abstention). Every family has a reference solver; the solver is tested against every rendering. Splits cut at the latent-world level (sealed generated first and hashed). Per-world counterbalancing (independent hash domain) pins positional/constant baselines: worst shortcut baseline 0.267 < 0.35 across seeds. Contamination screen: zero exact/normalized/group collisions, fail closed.

## Training contract

4096 real tokens/update, 120 updates (491,520-token WSD budget: warmup 10% → stable → linear decay to 0.1×peak 3e-4) — executed through `ProductionTrainingBackend` with a canary-scaled schedule bound to the frozen 5B schedule's shape; the frozen `lr_at` itself is verified on its real domain. Checkpoint every 16 updates. Clip 1.0 with the audited 1e-4 certificate tolerance.

## Pass gates (preregistered)

Mechanical: exact parameter accounting; clean split audit; no shortcut ≥ 0.35; production path only; finite training with real updates; valid clip certificate; exact token accounting; WSD trace expected==actual with zero rewarm events; durable checkpoints; fresh-process bitwise resume (model/optimizer/ledger/cursor/schedule bytes); corrupted/incompatible restore rejected; EOS contract valid; evaluation receipts valid; no EXPERIMENT_ONLY output mode in the canonical path.
Formation: identity dev ≥ 0.30, binding dev ≥ 0.25, dev overall ≥ 0.15, best-family formation ≥ 0.30, EOS-correct everywhere — train acquisition, dev transfer, and the sealed result reported separately.
