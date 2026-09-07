# PROGRESS — all experiments, dated, attributed, with results

| Date | Agent | Experiment | Result |
|------|-------|------------|--------|
| 2026-09-06 | arkenstone-agent | **Branch created** — base `28bf57a`, isolated worktree, evidence map + bottleneck graph + ledgers established | Infrastructure ready |
| 2026-09-06 | arkenstone-agent | **ARK-001 lift-off mapping** (5 arms: T1/T2/T0/T1-LARGE/T1-BYTE, 24k steps each, T4 GPU) | Lift-off at 200–400 steps; vocab/width insensitive; memorize-then-grok decomposition discovered |
| 2026-09-06 | arkenstone-agent | **Binding-v2 red-team** (3 seeds + trained logistic escalation) | **CONSISTENT** with cymek claim: low-LR collapse 0%, high-LR 62.5%; cymek's test-asserted qualification now independently verified |
| 2026-09-06 | arkenstone-agent | **ARK-002a T2 saturation** (seed 13, 10176 steps) | OOD reaches 1.0; grokking transition is real on this task family |
| 2026-09-06 | arkenstone-agent | **ARK-002B T2 replication** (seeds 29, 47, commutation-free manifest) | Qualitative transition replicates on both seeds; G90 dose seed-variable (~9k–18k steps) |
| 2026-09-06 | arkenstone-agent | **ARK-003 acceleration screen** (4 arms: flat/curriculum/aligned-teacher/unaligned, seed 29) | **NULL** — no arm beats flat; curriculum delays memorization; teacher compute-handicapped |
| 2026-09-06 | arkenstone-agent | **ARK-004A transition mapping** (4 fresh seeds: 101/202/303/404, 24k steps, GPU, probes) | G90 on ALL 4 seeds; tens-selectivity claimed as precursor (later inverted); post-G90 instability discovered (seed 101 collapse) |
| 2026-09-06 | arkenstone-agent | **ARK-004A-R reanalysis** (computational, no training) | **VERDICT B+C**: precursor direction INVERTED in prose; selectivity is a MARKER not precursor; ARK-004B cancelled |
| 2026-09-06 | arkenstone-agent | **ARK-005 retention observation** (from ARK-004A receipts) | Acquisition ≠ Retention: 3/4 seeds decay after sustained G90; retention metrics defined |
| 2026-09-06 | arkenstone-agent | **ARK-006 LR dose-response** (seed 606, lr ×0.001/0.01/0.1) | Threshold between 1e-5 and 1e-4: below → RET90=1.0; above → collapse |
| 2026-09-06 | arkenstone-agent | **Colab T4 full program** (84 min, all 5 experiments) | Transitions replicate across devices; retention arms stable in short fork window (8k steps too short for decay) |
| 2026-09-06 | arkenstone-agent | **ARK-007 paired continuation** (2 seeds × 8 continuation orders × 2 LR arms, T4 GPU, 32 forks × 8k steps) | **REPLICATED_PROTECTION**: high-LR collapse 10/16 (62.5%), low-LR collapse 0/16 (0%), risk diff −0.625, 10 discordant pairs, zero reverse. **First causally demonstrated retention mechanism in An-Ra** |
| 2026-09-06 | arkenstone-agent | **ARK-007 multi-seed threshold replication** (seeds 707/808, LR ×0.001 vs ×1.0) | Low-LR safety universal (5/5 seeds); high-LR decay seed-DEPENDENT (risk factor, not deterministic); law refined |
| 2026-09-06 | arkenstone-agent | **BRAMASTRA ingested** — terminal/EOS contract fix, Citadel T1C stopping confound, capacity accounting correction, AGI blueprint noted | Program knowledge expanded |
| 2026-09-06 | arkenstone-agent | **Gap-review assessment** — verified all claims against live code; BUILTINS already fixed upstream; STATUS drift confirmed; cross-branch map written | GAP 2/3/4/5/6 addressed Arkenstone-side |
| 2026-09-06 | arkenstone-agent | **Ledger verifier redesigned** — content-hash drift detection, dual-scheme legacy receipts, no self-reference; caught real ARK-001 canonicalization drift | Integrity system operational |
| 2026-09-06 | arkenstone-agent | **Colab notebook rebuilt** — device-adaptive TPU/CUDA/CPU, resumable, budgeted, manifest hash asserted, auto-download | GPU (T4) is required runtime; TPU crashes from graph compilation |

## Three laws discovered

| # | Law | Status | Evidence |
|---|-----|--------|----------|
| 1 | **Lift-off law**: symbolic tasks lift off in 200–400 steps with flat answer-only CE; vocab/width insensitive; curriculum/teacher don't accelerate | REPLICATED (7 seeds, 2 devices) | ARK-001/002B/004A |
| 2 | **Grokking decomposition**: memorize → stochastic delay → structural OOD emergence; timing decoupled from memorization speed (ρ=0.00); per-position asymmetry (ones early, tens late) | REPLICATED (qualitative, all seeds, 2 devices) | ARK-002/002B/004A |
| 3 | **LR-retention protection**: dropping LR from 1e-3 to 1e-5 at G90 eliminates collapse (0% vs 62.5%, risk diff −0.625); low LR universally safe (5/5 seeds); high-LR decay is seed-dependent risk factor | REPLICATED_PROTECTION (T4 GPU, 16 paired conditions) | ARK-005/006/007 |

## Falsified claims

| Claim | How it died | Evidence |
|-------|-------------|----------|
| "Higher tens-selectivity → earlier generalization" | Direction inverted in prose; ρ(+0.60) vs G90_step means later | ARK-004A-R |
| "Curriculum accelerates the transition" | Delays memorization, zero OOD at box end | ARK-003 arm B |
| "Aligned teacher accelerates the transition" | No effect at equal wall budget; compute confound documented | ARK-003 arm C |
| "Weight-decay removal prevents post-G90 decay" | Identical trajectory to control | ARK-005 arm C |
| "EMA consolidation prevents post-G90 decay" | Identical trajectory to control | ARK-005 arm D |
| "LR ×0.1 prevents post-G90 decay" | Delays collapse ~2000 steps but does not prevent it | ARK-006 |
