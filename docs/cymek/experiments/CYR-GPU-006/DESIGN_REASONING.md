# CYR-GPU-006 — DESIGN REASONING

## Why this experiment still exists

The strongest replicated branch evidence is state-dependent: high LR is better for acquisition/recovery, while very low LR protects already-acquired arithmetic capability. The unresolved question is whether that behavior survives the real Cymek V5 implementation at a materially larger proxy and whether an adaptive policy beats a simple time-based decay.

CYR-GPU-005 finally had the right scientific skeleton but its execution harness invalidated the comparison. CYR-GPU-006 does **not** add another fashionable mechanism. It repairs the experiment so the original uncertainty can actually be measured.

## Primary causal unit

An independent acquired parent is the subject. Each parent is trained exactly once at HIGH until candidate-free G90 or its actual-token ceiling. If qualified, four continuations restore the exact same model+optimizer checkpoint and consume the same frozen future index stream:

1. HIGH_CONTINUE
2. LOW_CONTINUE
3. FIXED_TIME_HIGH_TO_LOW
4. HYSTERETIC_HIGH_LOW

The comparison is paired within parent, then replicated across parents. A single parent can never produce a scientific winner.

## What was deliberately rejected

- More arms: lower information per GPU minute than deeper replicated matched forks.
- A new optimizer: current evidence does not justify confounding LR/state with optimizer family.
- P35 by default: proxy scale is subordinate to enough token dose and replication.
- Teacher-forced capability gates: candidate-free complete generation with valid stop is the authority.
- A result picked from one lucky parent: mechanically impossible in the verdict function.

## Runtime design

Hardware resolution uses only measured training throughput, measured batched-generation throughput, VRAM/fit and the frozen wall budget. It first proves the minimum scientific dose is affordable, then prospectively expands token dose toward the target. Accuracy, loss and treatment outcomes are forbidden resolver inputs.

## Transfer

A replicated retention winner is subjected to a second-family registry/binding plasticity check on two parents. Transfer never rescues a failed retention hypothesis and SEALED_RESERVED is measured only after the decision object is fixed.
