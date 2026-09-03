# CYMEK_BOTTLENECK_RANKING.md

Cymek-first bottleneck ranking (supersedes the ESOES-heavy ordering in `BOTTLENECK_RANKING.md`,
which remains historical record). Derived from the `26a61f6` delta audit, the data/shortcut
audits, and Citadel's own certification executions of 2026-09-03. Every bottleneck is classified:

- **ENGINEERING** — the intended experiment cannot run correctly.
- **MEASUREMENT** — it can run, but we cannot reliably determine whether it worked.
- **SCIENTIFIC** — it runs, measurement is valid, but capability does not appear.

Solved roughly in that order (central operating rule of the continuation instruction).

| # | Bottleneck | Kind | Evidence | Cost to resolve | Info gain | Priority |
|---|---|---|---|---|---|---|
| E1 | **No cognition-data producer wired to the pipeline.** 0 cognition tokens ever consumed; 2 of 9 planned families have no generator; the one training generator is 3 shortcut-saturated templates (no shuffle, recency-fixated, never baseline-audited); no `TrainingExample`→`Document` adapter (split/family/difficulty discarded). | ENGINEERING (data) | CYMEK_DATA_AUDIT §0/§3; CYMEK_DATA_SHORTCUTS A1–A4 | Medium: write cognition training generators per family (shuffle-after-graph-build, attested candidates), adapter, baseline audit | Very high — unblocks every learning experiment on the intended data | **1** |
| E2 | **Executed path bypasses its own sampler/microbatch/mixture modules.** Tests-only until Citadel's certification executed microbatch+sampler once (mini spec, CPU); mixture enforcement never exercised anywhere. | ENGINEERING (wiring) | miniature_run.py:151-185 hand-slicing; `microbatch` tests-only; cert receipts | Low: route a run through `sampler_order`+`microbatch`+mixture allocation (the cert script already does 2 of 3) | High — makes executed runs representative of the contract | **2** |
| E3 | **Device-narrow certification gate.** `certify_real_update`'s post-clip tolerance (1e-6, calibrated on CUDA bf16) fails closed on CPU fp32 when clipping activates (measured breach 1.00000286 on P35; measured pass 0.99999934 on mini — outcome is noise-dependent). Blocks CPU certification/CI of the backend. | ENGINEERING (portability) | Citadel cert attempts 2026-09-03 (logged in receipts' environment + report) | Low: device-calibrated tolerance or float64 norm recompute — but it is a production-contract change → promotion proposal, not a silent edit | Medium — unblocks CPU CI + local certification at P35 | **3** |
| M1 | **No cognition evaluation harness in the production path.** Executed evaluation = 2 hardcoded tasks; e0 generators not wired to `CheckpointBackedV5Adapter`; no baseline/null gate over trained-model outputs; candidate-selection scorer unsanctioned (`production_scoring_mode = null` legacy). | MEASUREMENT | miniature receipt eval section; CYMEK_DATA_SHORTCUTS B1/B2 | Medium: task supplier from eval generators + candidate-free generation metrics + heuristic-null gating | Very high — without it, no training result is interpretable | **4** |
| M2 | **Evidence-integrity debt.** Cymek's own `evidence_ledger.json` carries stale numbers contradicting its committed receipts (47 vs 48 docs; 94.5% vs 96.67% pack efficiency; 68.7/31.3 vs 49.7/50.3 mixture) and one code-contradicted claim (cursor-microbatch "feeds updates"); miniature receipt lacks device/wall-time/timestamp; CI reproduces contract receipts but no execution receipts. | MEASUREMENT (evidence integrity) | delta audit §6/§7; CYMEK_SYNC stale-green note | Low: regenerate ledger from receipts, extend receipt fields, CI job reproducing the miniature | Medium — prevents the stale-green class triquetra demonstrated | **5** |
| S1 | **Does cognition-targeted data produce capability per token at feasible micro scale?** Never tested at any scale; founding negative (loss-without-cognition) is n=1 on V4. | SCIENTIFIC | EVIDENCE_LEDGER E1/N1; CYMEK_DATA_AUDIT §2 | 2 × 30M-token arms ≈ hours (certified CPU throughput 2,360 tok/s ⇒ ~3.5 h/arm; faster on CUDA) | Very high — first causal data-vs-capability measurement; preregistered as **C1** | **6** |
| S2 | **Objective pressure (CE-only vs query-conditioned contrast).** Query-swap implemented, weight exactly 0 everywhere. | SCIENTIFIC | `v5_objectives/query_swap.py`; D-041 | Same scale as S1 after it | High, downstream of S1's baseline | **7** |
| B1 | External 5B corpus, TPU/XLA topology, remote credentials, sealed custody | BLOCKED (external) | Cymek's own ledger `BLOCKED_BY_*` entries | Out of Citadel's hands | — | recorded, not ranked |
| B2 | Canonical 5B WSD schedule never executed; distributed all-reduce CONTRACT_ONLY | ENGINEERING (deferred) | CYMEK_EXECUTION_GRAPH stages 12/16 | Only matters at production scale | — | deferred |

## Top three (actionable now)

1. **E1 — build the cognition data path** (generators that create the intended pressure without
   the A-class shortcuts, `Document` adapter preserving family/split/difficulty, baseline audit
   over emitted data). Nothing else can produce a cognition result.
2. **E2 — wire the contract path into executed runs** (sampler + microbatch + mixture at
   consumption time). Citadel's certification already proves the modules work in isolation.
3. **E3 — calibrate the update-certification gate per device** (CPU fp32 verdicts are
   noise-dependent; one measured breach, one measured pass). Blocks repeatable local
   certification and CI.
