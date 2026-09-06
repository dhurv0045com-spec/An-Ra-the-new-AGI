# ARK-003 SCREENING ANALYSIS (seed 29, all arms wall-boxed at 1800s)

| Arm | Steps run | M99 | Final OOD | Locality | Supervised tokens |
|-----|-----------|-----|-----------|----------|--------------------|
| A flat | 9600 | 1400 | **0.421** (mid-transition) | 0.424 | 2,457,600 |
| B curriculum | 10209 | 5400 | 0.00 | 0.00 | 2,437,637 |
| C aligned teacher | 5301 | 1200 | 0.01 | 0.01 | 3,665,282 |
| D unaligned teacher | 5325 | 800 | 0.00 | 0.00 | 3,678,583 |

## Screening verdict (preregistered criterion: >=2x tokens_to_G90/post_mem_delay_90)
**NO arm demonstrates sustained G90 within the box; NO candidate beats the flat
baseline. Racing stage 2 is NOT triggered** (per the preregistered rule: do not
spend replication seeds on arms with no effect).

## Findings
1. **No acceleration demonstrated.** Neither aligned decomposition (C) nor
   curriculum (B) accelerated the OOD transition; C/D consumed MORE supervised
   tokens (suffix cost: ~691 vs ~256 tokens/step) for less progress. This is
   the honest CASE-B outcome of the mission decision tree: simple
   curriculum/component supervision does not accelerate this transition under
   the tested conditions.
2. **Curriculum was actively harmful within budget**: B's single-digit phase
   delayed T2 memorization (M99 at 5400 vs 1400) and left zero OOD at box end.
3. **Compute handicap cuts both ways**: C/D ran ~5300 optimizer steps vs A's
   9600 at equal wall time (their suffixes cost ~2.7x supervised tokens/step).
   The screening therefore cannot distinguish "teacher fails" from "the
   teacher's own token cost consumed the budget" — a step-matched rerun is the
   clean follow-up (ARK-004 candidate), NOT a replication stage.
4. **Locality co-emergence (tentative)**: on the only arm that transitioned (A),
   counterfactual locality tracks OOD exactly (both ~0 through 8800, both jump
   at 9600: 0.421/0.424). Weak single-arm support for H-FACTOR (behavioral
   factorization signature emerges with OOD lift-off). Single snapshot — not a
   claim yet.

## Red-team notes (applied to the screening)
- The wall-box asymmetry (fewer steps for C/D) is the strongest alternative
  explanation for C/D's null — no "teacher fails" claim is made.
- Teacher suffix contains gold digits of its own row (by design, aligned);
  eval never includes suffixes; no eval leakage.
- A's 0.421 at box end is a mid-transition snapshot; sustained G90 was NOT
  demonstrated even for A — consistent with 002B seed 29 (0.924 at box end,
  G90 just missed). The box is too short for flat T2 at 4-way parallel speed.

## Decision
Per the preregistered racing rule, no replication stage. Next highest-value
experiment (ARK-004): step-matched teacher rerun (remove the wall-time
handicap) OR grokking-control sweep (replay ratio / unique-data fraction /
weight decay) — one variable at a time. The tier dose-ratio map (T3 carry)
also remains open from the parked plan.
