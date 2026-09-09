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
| 2026-09-08 | user Colab T4 + ChatGPT audit | **MASTER GPU V5 receipt validation** | 9/9 uploaded JSON receipt hashes revalidated; runtime 96.47 min |
| 2026-09-08 | user Colab T4 | **ARK-007R fresh-checkpoint retention replication** | **REPLICATED MICRO-TASK RETENTION EFFECT**: HIGH 9/12, LOW 0/12, risk diff -0.75 |
| 2026-09-08 | user Colab T4 | **ARK-009 non-arithmetic transfer gate** | Ordinary held-out exact reached 1.0; composite query/order robustness failed; transfer remained NOT_DEMONSTRATED |
| 2026-09-08 | user Colab T4 | **ARK-010 recovery after instability** | Immediate LOW recovery hypothesis falsified: HIGH recovered G90 8/9 vs LOW 2/9 |
| 2026-09-08 | ChatGPT live-repo audit | **Cymek alignment review** | production/readiness discipline inspected read-only; no Arkenstone result auto-promoted |
| 2026-09-08 | user Colab T4 + ChatGPT audit | **Discovery V6 integrity validation** | 179.09 min; GPU smoke PASS; 14/14 receipt hashes PASS |
| 2026-09-08 | user Colab T4 | **ARK-011 adaptive recovery→retention controller** | **SUPPORTED_ADAPTIVE_PROTECTION**: HIGH recurrent instability 3/6 vs SWITCH_LOW 0/6 |
| 2026-09-08 | user Colab T4 | **ARK-012 recovery threshold screen** | **TIME_NOT_STATE_SCREEN**: no exact threshold law |
| 2026-09-08 | user Colab T4 | **ARK-013 stability–plasticity screen** | **INCONCLUSIVE_NEW_SKILL_NOT_ACQUIRED**; LOW did not solve no-replay cross-task forgetting |
| 2026-09-08 | user Colab T4 | **ARK-014 repaired non-arithmetic binding** | **ORDER_ROBUSTNESS_REPAIRED**; LR transfer still zero-event/inconclusive |
| 2026-09-09 | user Colab T4 + ChatGPT audit | **Discovery V7 integrity validation** | 166.24 min; GPU smoke PASS; bundle SHA256 `25c4ac0aa01478cb2067147a240a12b4e30516810dad465e619391ff8a6f7faf`; **11/11 receipt hashes PASS**, no failure receipt |
| 2026-09-09 | user Colab T4 | **ARK-015 non-arithmetic invariance retention** | **SUPPORTED_NONARITHMETIC_INVARIANCE_PROTECTION**: 3 fresh parents, 8 matched pairs; NARROW_HIGH 8/8 failures, NARROW_LOW 0/8, AUGMENTED_HIGH_REFERENCE 0/8; canonical exact stayed 1.0 while order robustness eroded under narrowed HIGH |
| 2026-09-09 | user Colab T4 | **ARK-016 applied-update mechanism screen** | **INCONCLUSIVE_LOW_EVENT_RATE**: 3/3 T2 parents acquired but only 1/12 opportunities yielded a qualifying collapse→recovery fork; all four arms stable there |
| 2026-09-09 | ChatGPT | **Discovery V8 preregistration** | ARK-017 mechanism factorial, ARK-018 ~1GB real-data substrate bridge, ARK-019 Capability Guardian closed-loop controller preregistered before implementation |

## Current high-confidence program regularities

| # | Regularity | Status | Evidence |
|---|------------|--------|----------|
| 1 | **Lift-off:** simple symbolic instance-fitting emerges rapidly when task is sufficiently simple | REPLICATED at Micro | ARK-001/002B |
| 2 | **Memorization ≠ generalization:** T2 memorizes early, then reaches structural OOD after a delayed seed-variable transition | REPLICATED | ARK-002/002B/004A |
| 3 | **Retention is state-dependent:** a generalized solution can enter instability episodes under continued HIGH training | REPLICATED phenomenon | ARK-004A/005/007/007R |
| 4 | **LOW protects an already-generalized T2 state under same-task continuation** | REPLICATED at Micro T2 | ARK-007R |
| 5 | **Recovery and retention require different dynamics on Micro T2:** HIGH usually reacquires better after instability; LOW after confirmed recovery reduces recurrence | DIRECTLY SUPPORTED | ARK-010/011 |
| 6 | **LOW is not a general no-replay cross-task forgetting solution** | EXECUTED boundary | ARK-013 |
| 7 | **Binding order brittleness is repairable by order augmentation** | SUPPORTED SCREEN | ARK-014 |
| 8 | **Capability can narrow while ordinary accuracy stays perfect:** HIGH canonical-only continuation preserved canonical exact but destroyed order invariance; LOW or continued augmentation protected it | DEMONSTRATED at controlled Micro non-arithmetic scale | ARK-015 |
| 9 | **Large parameter movement alone is not sufficient to explain invariant erosion** | DEMONSTRATED boundary | ARK-015 augmented-HIGH moved farther than narrowed-HIGH while retaining invariance |

## Falsified / superseded claims

| Claim | How it died | Evidence |
|-------|-------------|----------|
| "Higher tens-selectivity -> earlier generalization" | Direction inverted; marker not precursor | ARK-004A-R |
| "Curriculum accelerates the transition" | Delays memorization, zero OOD at box end | ARK-003 |
| "Weight-decay removal prevents post-G90 decay" | Null | ARK-005 |
| "EMA consolidation prevents post-G90 decay" | Null | ARK-005 |
| "10x LR reduction is sufficient" | 1e-4 still unstable | ARK-006 |
| "A collapse90 event means irreversible forgetting" | Most high-LR continuations reacquired G90 | ARK-010 |
| "Immediate low LR is the best recovery intervention after collapse" | HIGH recovered 8/9 vs LOW 2/9 | ARK-010 |
| "The exact best switch threshold is identified" | threshold aliasing/non-monotonicity | ARK-012 |
| "Large parameter movement by itself explains capability erosion" | augmented HIGH moved farther than narrowed HIGH while staying robust | ARK-015 |
