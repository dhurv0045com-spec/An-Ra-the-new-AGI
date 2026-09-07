# PROGRESS — all experiments, dated, attributed, with results

| Date | Agent | Experiment | Result |
|------|-------|------------|--------|
| 2026-09-06 | arkenstone-agent | **Branch created** — base `28bf57a`, isolated worktree, evidence map + bottleneck graph + ledgers established | Infrastructure ready |
| 2026-09-06 | arkenstone-agent | **ARK-001 lift-off mapping** (5 arms: T1/T2/T0/T1-LARGE/T1-BYTE, 24k steps each, T4 GPU) | Lift-off at 200–400 steps; vocab/width insensitive; memorize-then-grok decomposition discovered |
| 2026-09-06 | arkenstone-agent | **Binding-v2 red-team** (3 seeds + trained logistic escalation) | **CONSISTENT** with cymek claim: independent qualification survives stronger attack |
| 2026-09-06 | arkenstone-agent | **ARK-002a T2 saturation** (seed 13, 10176 steps) | OOD reaches 1.0; grokking transition is real on this task family |
| 2026-09-06 | arkenstone-agent | **ARK-002B T2 replication** (seeds 29, 47, commutation-free manifest) | Qualitative transition replicates on both seeds; G90 dose seed-variable (~9k–18k steps) |
| 2026-09-06 | arkenstone-agent | **ARK-003 acceleration screen** (4 arms: flat/curriculum/aligned-teacher/unaligned, seed 29) | **NULL** — no arm beats flat; curriculum delays memorization; teacher compute-handicapped |
| 2026-09-06 | arkenstone-agent | **ARK-004A transition mapping** (4 fresh seeds: 101/202/303/404, 24k steps, GPU, probes) | G90 on ALL 4 seeds; tens-selectivity claimed as precursor (later inverted); post-G90 instability discovered |
| 2026-09-06 | arkenstone-agent | **ARK-004A-R reanalysis** | **VERDICT B+C**: precursor direction INVERTED in prose; selectivity is a MARKER not precursor; ARK-004B cancelled |
| 2026-09-06 | arkenstone-agent | **ARK-005 retention study** | Acquisition ≠ Retention; LR reduction delays collapse on a decaying seed; wd-removal/EMA null |
| 2026-09-06 | arkenstone-agent | **ARK-006 LR dose-response** | Threshold candidate between 1e-5 and 1e-4 on seed 606 |
| 2026-09-06 | arkenstone-agent | **ARK-007 paired continuation** | Low-LR protection supported; high-LR instability continuation-dependent |
| 2026-09-08 | user Colab T4 + ChatGPT audit | **MASTER GPU V5 receipt validation** | 9/9 uploaded JSON receipt hashes independently revalidated; bundle SHA256 `e40355e0e0212406985c2445d2a8388b5d704ad722fd4c4b4c3455a5e64f0c7f`; runtime 96.47 min |
| 2026-09-08 | user Colab T4 | **ARK-007R fresh-checkpoint retention replication** (seeds 909/1010/1111 × continuation 2701..2704) | **REPLICATED MICRO-TASK RETENTION EFFECT**: HIGH collapse 9/12, LOW 0/12, risk diff -0.75; direction holds on all 3 independent acquisitions |
| 2026-09-08 | user Colab T4 | **ARK-009 non-arithmetic transfer gate** | Ordinary fact-set-disjoint held-out exact reached 1.0 on both seeds; strict composite query/order diagnostic failed, so no retention forks; transfer remains NOT_DEMONSTRATED |
| 2026-09-08 | user Colab T4 | **ARK-010 recovery after instability** (9 prospective collapse states) | Immediate LOW recovery hypothesis falsified: HIGH recovered sustained G90 8/9 vs LOW 2/9; opens state-dependent HIGH-recover -> LOW-retain controller hypothesis |

## Current high-confidence program regularities

| # | Regularity | Status | Evidence |
|---|---|---|---|
| 1 | **Lift-off**: simple symbolic instance-fitting emerges rapidly under flat answer-only CE once the task is sufficiently simple | REPLICATED at micro | ARK-001/002B |
| 2 | **Memorization ≠ generalization**: T2 memorizes early, then reaches structural OOD after a long seed-variable delay; M99 does not predict G90 timing | REPLICATED | ARK-002/002B/004A |
| 3 | **Retention is state-dependent**: a generalized solution can enter instability episodes under continued high-LR training | REPLICATED phenomenon | ARK-004A/005/007/007R |
| 4 | **Low LR protects an already-generalized T2 state**: under matched continuations, 1e-5 produces far fewer instability events than 1e-3 | REPLICATED at Micro T2 | ARK-007R: 9/12 vs 0/12 |
| 5 | **Recovery and retention require different dynamics**: after an instability episode, high LR usually reacquires G90 better than immediate low LR | SUPPORTED pattern, controller untested | ARK-010: 8/9 vs 2/9 |

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
