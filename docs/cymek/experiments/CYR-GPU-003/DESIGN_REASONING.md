# CYR-GPU-003 Design Reasoning

## Measured bottlenecks (from executed experiments)

1. **Post-G90 collapse** — 10/16 high-LR forks collapse after sustained G90;
   0/16 low-LR forks collapse (ARK-007, T4 GPU, 2 acquisition checkpoints).
   This is the strongest causal result in the program.
2. **Grokking delay** — memorization and generalization are decoupled;
   the delay between M99 and G90 is 10–20× the memorization dose and
   varies ~2× across seeds (ARK-002B/004A, 7 seeds).
3. **Terminal supervision** — without an answer-terminator token, models
   produce correct prefixes followed by garbage (BRAMASTRA: 0/32 → 32/32).
4. **Order augmentation repairs binding brittleness** — ARK-014 showed that
   order-augmented training qualifies at 1.8k steps vs canonical failure.

## What we do NOT know

- Does the LR-retention protection hold at P35 scale (35M params)?
- Does it hold with the production 24,576 tokenizer?
- What is the exact threshold between 1e-5 and 1e-4?
- Does LR-drop after collapse (not before) recover capability?
- Does the effect transfer to non-arithmetic tasks?

## Rejected interventions (with reasons)

| Intervention | Why rejected |
|---|---|
| Curriculum (easy→hard) | ARK-003: delays memorization, zero OOD at box end |
| Aligned teacher | ARK-003: no acceleration; compute confound |
| EMA consolidation | ARK-005: identical trajectory to control |
| Weight-decay removal | ARK-005: identical trajectory to control |
| Column-selectivity objective | ARK-004A-R: direction inverted; marker not precursor |
| Learned discovery policy | BRAMASTRA: n.s. result, 1 round, entangled seeds |

## Selected experiment

**P35-scale LR-retention replication.**

Why this beats alternatives:
- Tests the strongest causal finding (risk diff −0.625) at the next scale up
- Uses the REAL production tokenizer (24,576) for the first time
- Bridges the gap between micro (0.8M) and production (500M tokens)
- Cheap: 4 arms × 24k steps on T4 ≈ 60–90 min
- Directly actionable: if confirmed, add LR-drop to the 500M schedule

## Why NOT the alternatives

- Binding transfer (ARK-008 retry): binding was too hard for 0.8M; needs
  P35 scale which makes it a 3+ hour experiment on its own. Park for now.
- Recovery-after-collapse: interesting but secondary; the primary question
  is whether PRE-EMPTIVE LR-drop works at scale.
- Threshold mapping: premature; first confirm the effect exists at scale.
- Multi-token prediction: no evidence it addresses a measured bottleneck.
- StableAdamW: interesting but no measured instability in our optimizer.
