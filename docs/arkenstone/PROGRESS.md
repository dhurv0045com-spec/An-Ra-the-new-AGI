# PROGRESS — all experiments, dated, attributed, with results

| Date | Agent | Experiment | Result |
|------|-------|------------|--------|
| 2026-09-06 | arkenstone-agent | **Branch created** — base `28bf57a`, isolated worktree, evidence map + bottleneck graph + ledgers established | Infrastructure ready |
| 2026-09-06 | arkenstone-agent | **ARK-001 lift-off mapping** | Lift-off at 200–400 steps; vocab/width insensitive; memorize-then-grok decomposition discovered |
| 2026-09-06 | arkenstone-agent | **Binding-v2 red-team** | **CONSISTENT** with cymek claim: independent qualification survives stronger attack |
| 2026-09-06 | arkenstone-agent | **ARK-002a T2 saturation** | OOD reaches 1.0; grokking transition is real on this task family |
| 2026-09-06 | arkenstone-agent | **ARK-002B T2 replication** | Qualitative transition replicates on both seeds; G90 dose seed-variable (~9k–18k steps) |
| 2026-09-06 | arkenstone-agent | **ARK-003 acceleration screen** | **NULL** — no arm beats flat; curriculum delays memorization; teacher compute-handicapped |
| 2026-09-06 | arkenstone-agent | **ARK-004A transition mapping** | G90 on all 4 seeds; original tens-selectivity precursor claim later inverted; post-G90 instability discovered |
| 2026-09-06 | arkenstone-agent | **ARK-004A-R reanalysis** | **VERDICT B+C**: precursor direction inverted; selectivity is a marker not precursor; ARK-004B cancelled |
| 2026-09-06 | arkenstone-agent | **ARK-005 retention study** | Acquisition ≠ retention; LR reduction delays collapse on a decaying seed; wd-removal/EMA null |
| 2026-09-06 | arkenstone-agent | **ARK-006 LR dose-response** | Threshold candidate between 1e-5 and 1e-4 on seed 606 |
| 2026-09-06 | arkenstone-agent | **ARK-007 paired continuation** | Low-LR protection supported; high-LR instability continuation-dependent |
| 2026-09-08 | user Colab T4 + ChatGPT audit | **MASTER GPU V5 receipt validation** | 9/9 uploaded JSON receipt hashes revalidated; bundle SHA256 `e40355e0e0212406985c2445d2a8388b5d704ad722fd4c4b4c3455a5e64f0c7f`; runtime 96.47 min |
| 2026-09-08 | user Colab T4 | **ARK-007R fresh-checkpoint retention replication** | **REPLICATED MICRO-TASK RETENTION EFFECT**: HIGH collapse 9/12, LOW 0/12, risk diff -0.75 |
| 2026-09-08 | user Colab T4 | **ARK-009 non-arithmetic transfer gate** | Ordinary held-out exact reached 1.0; composite query/order robustness failed; transfer remained NOT_DEMONSTRATED |
| 2026-09-08 | user Colab T4 | **ARK-010 recovery after instability** | Immediate LOW recovery hypothesis falsified: HIGH recovered G90 8/9 vs LOW 2/9 |
| 2026-09-08 | ChatGPT live-repo audit | **Cymek 500M alignment review** | `cymek-500m-readiness` hardens production/resume/mixture/eval boundaries; no 500M cognition result exists |
| 2026-09-08 | user Colab T4 + ChatGPT audit | **Discovery V6 integrity validation** | 179.09 min; GPU smoke PASS; uploaded bundle SHA256 `1ec3224075de49b080e229111e4c4f897430c3eca888c487116aff07a65c8d15`; **14/14 receipt hashes PASS**, no failure receipt |
| 2026-09-08 | user Colab T4 | **ARK-011 adaptive recovery→retention controller** | **SUPPORTED_ADAPTIVE_PROTECTION**: 6 sealed-qualified forks across 3 fresh acquisitions; HIGH recurrent instability 3/6 vs SWITCH_LOW 0/6, risk diff -0.50 |
| 2026-09-08 | user Colab T4 | **ARK-012 recovery threshold screen** | **TIME_NOT_STATE_SCREEN**: 2 selected sources; 0.85/0.90 best mean sealed area but threshold aliasing prevents an exact state-threshold law |
| 2026-09-08 | user Colab T4 | **ARK-013 new-skill stability–plasticity screen** | **INCONCLUSIVE_NEW_SKILL_NOT_ACQUIRED**: four triplets, HIGH never T3-G90, adaptive never switched; no-replay T3 training erased sustained T2 under all arms |
| 2026-09-08 | user Colab T4 | **ARK-014 repaired non-arithmetic binding** | **ORDER_ROBUSTNESS_REPAIRED**: canonical training remained order-brittle (~0.33), order augmentation qualified at 1.8k with sealed order robustness ~0.987; LR transfer still inconclusive because HIGH/LOW failures were both 0/3 |

## Current high-confidence program regularities

| # | Regularity | Status | Evidence |
|---|------------|--------|----------|
| 1 | **Lift-off**: simple symbolic instance-fitting emerges rapidly under flat answer-only CE once the task is sufficiently simple | REPLICATED at micro | ARK-001/002B |
| 2 | **Memorization ≠ generalization**: T2 memorizes early, then reaches structural OOD after a long seed-variable delay; M99 does not predict G90 timing | REPLICATED | ARK-002/002B/004A |
| 3 | **Retention is state-dependent**: a generalized solution can enter instability episodes under continued high-LR training | REPLICATED phenomenon | ARK-004A/005/007/007R |
| 4 | **Low LR protects an already-generalized T2 state under same-task continuation** | REPLICATED at Micro T2 | ARK-007R: HIGH 9/12 vs LOW 0/12 |
| 5 | **Recovery and retention require different dynamics on Micro T2**: HIGH usually reacquires better after instability, while LOW after confirmed recovery reduces recurrent instability | DIRECTLY SUPPORTED at Micro T2 | ARK-010 + ARK-011 |
| 6 | **LOW protection is not a general cross-task interference solution**: under 12k no-replay T3 updates, old T2 retention was lost under every arm | EXECUTED boundary, Micro arithmetic only | ARK-013 |
| 7 | **Binding order brittleness is repairable by order augmentation** on the ARK-014 subject | SUPPORTED SCREEN, one acquisition seed | ARK-014 |

## Falsified / superseded claims

| Claim | How it died | Evidence |
|-------|-------------|----------|
| "Higher tens-selectivity -> earlier generalization" | Direction inverted; marker not precursor | ARK-004A-R |
| "Curriculum accelerates the transition" | Delays memorization, zero OOD at box end | ARK-003 arm B |
| "Weight-decay removal prevents post-G90 decay" | Null | ARK-005 arm C |
| "EMA consolidation prevents post-G90 decay" | Null | ARK-005 arm D |
| "10x LR reduction is sufficient" | 1e-4 still unstable | ARK-006 |
| "A collapse90 event means irreversible forgetting" | Most high-LR continuations reacquired G90 | ARK-010 |
| "Immediate low LR is the best recovery intervention after collapse" | HIGH recovered 8/9 vs LOW 2/9 | ARK-010 |
| "The exact best switch threshold is identified" | 0.75/0.85/0.90/0.95 often alias at the same 200-step evaluation resolution; no monotonic ordering | ARK-012 |
