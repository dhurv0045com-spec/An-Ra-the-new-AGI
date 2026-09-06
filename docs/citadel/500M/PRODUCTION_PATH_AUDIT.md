# PRODUCTION PATH AUDIT — 500M campaign readiness (Cymek @ 28bf57a)

Audit date: 2026-09-06. Pin: `28bf57a0d299a2c13a99fe0046616c00a1b8530c`
(== origin/cymek == Citadel `PINNED_CYMEK_SHA`; the divergent local lineage
4abeaeb is UNPUSHED WIP and is not production — see
`docs/citadel/CROSS_BRANCH_INGESTION.md`).

Classification per §4: **CONNECTED** (executed inside the real production
path with receipts) / **BYPASSED** (exists but the production trainer skips
it) / **AMBIGUOUS** (present, correctness unclear) / **MISSING** (nothing
exists).

| # | Component | Class | Evidence |
|---|---|---|---|
| 1 | dataset (production corpus) | **MISSING** | No materialized corpus, no manifest-bound bytes. Mixture is DECLARED only (`run_spec` 65/20/15). PRE50M data gate: DATA_NOT_READY. |
| 2 | tokenizer | **AMBIGUOUS** | Adapter CONNECTED (identity receipts, zero-unknown invariant; used by T1D/PRE50M runs), but the artifact is `PROVISIONAL_CANDIDATE_E1_IDENTITY_REQUIRED` — no frozen production artifact hash. |
| 3 | packing | **CONNECTED** | `v5_data.pack` true multi-segment stream-fill; T1D + PRE50M executed it; ledger triple-checked. |
| 4 | sampler/cursor | **CONNECTED** | `sampler_order` + `cursor.advance` + `microbatch` executed in Citadel's certification run (previously tests-only — closed 2026-09-03). |
| 5 | batch assembly | **CONNECTED** | `assemble_batch`/`microbatch` executed end-to-end in the certified production path (one-update + ten-update runs). |
| 6 | model | **CONNECTED** | P35 (3 updates, CUDA) + V5-A bounded canary (exact 250,216,960 params) through the certified backend on TPU (PRE50M smoke 2026-09-06). |
| 7 | causal objective | **CONNECTED** | `causal_lm_loss` with segment/eligible semantics, ELIGIBLE_MISMATCH abort executed in every certified update. |
| 8 | backward/optimizer | **CONNECTED** | Real mutation certification (per-tensor SHA), clip 1.0, AdamW tied-embedding ownership; adversarial stale-optimizer tests. |
| 9 | scheduler | **AMBIGUOUS** | Canonical WSD `lr_at` is token-based and unit-tested, but every EXECUTED run (including PRE50M smoke) used `bounded_warmup_schedule` (constant LR). The 500M campaign must bind the canonical token-space schedule and execute it at least once before launch. |
| 10 | TrainingState | **CONNECTED** | Exact-update ledger, reserved-final-update resume verified (PRE50M smoke: 10,240 → 12,288 tokens). |
| 11 | checkpoint transaction | **CONNECTED** | Genesis/update publish, single-writer fence negative probe, exact restore, hash-verified — executed on TPU in the PRE50M smoke. |
| 12 | restore/continuation | **CONNECTED** | Resume-proof update after reload: moments preserved, continued update OK. |
| 13 | evaluation | **AMBIGUOUS** | `CheckpointBackedV5Adapter` + firewall CONNECTED and executed (miniature + T1D arms), but there is no wired production evaluation loop over frozen eval sets at token milestones. |
| 14 | top-level production entry point | **MISSING** | No single command wires corpus → manifest → tokenizer → packing → sampler → trainer. T1D/PRE50M each hand-wired their inputs. CYMEK_REQUIRED_CHANGES BLOCKING #2. |

## Verdict

**500M is BLOCKED.** Critical MISSING/BYPASSED items: the production corpus
(#1) and the top-level entry point (#14); tokenizer artifact (#2) and
canonical-schedule execution (#9) must also be resolved before
`ready_for_500m_training` can be true. Every other component is connected
and certified. The required Cymek-side work is enumerated in
[`CYMEK_REQUIRED_CHANGES.md`](CYMEK_REQUIRED_CHANGES.md).
