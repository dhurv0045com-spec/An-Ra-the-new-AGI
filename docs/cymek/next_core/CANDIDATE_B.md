# CANDIDATE B — REPRESENTATION-AWARE CORE (conditional: R1C World A/B)

**Claim:** the strongest Core **if** inactive-softmax competition (or a partition-adjacent mechanism) is confirmed as the causal carrier of the class-space effect.

## What changes vs Candidate A

The output head and vocabulary fields unblock; nothing else moves.

| Field | Value | Status in this world |
|---|---|---|
| output head | tied full-softmax **training-time treatment**: participating-set restriction (MASK_K) or inactive-partition offset, selected by R1C's preregistered endpoints | EVIDENCE_LOCKED by R1C *if* it passes; implementation EXPERIMENT_ONLY until then |
| physical vocabulary | retained at 24,576 (World A: mechanism is partition, not matrix size) or reduced to the R1C/transfer-justified class count (World B partial) | BLOCKED until R1C + CS-TRANSFER-001 |
| tokenizer | unchanged byte-BPE for language; numeric/symbol segmentation only if the transfer probe implicates segmentation | BLOCKED |
| embedding/output burden | if untied/low-rank output is required by the mechanism, it arrives as its own preregistered experiment (function-preserving init), never silently | BLOCKED |

## Parameter accounting (unchanged canonical path)

The canonical path stays exactly Candidate A's 250,216,960 (tied, full softmax). Treatments change *training-time logits only*, so parameter counts are identical under MASK/OFFSET; an untied or reduced-vocab variant changes counts and must go through `tools/next_core_compute_model.py` accounting + a function-preservation test before adoption.

## Costs / risks

- compute: none at inference; training-time reduction (MASK) or +1 logit op (OFFSET);
- risk: train/inference mismatch is the entire point — R1C's dual structural/functional endpoints exist precisely to catch output-calibration failure; likelihood semantics change = new scientific intervention (§12), never a refactor;
- falsifier: R1C `SOFTMAX_COMPETITION_NOT_SUFFICIENT` kills this candidate's mechanism (K01).

## Why it must wait

Its entire justification is R1C evidence that does not exist yet. Building it now would be EXPERIMENT_GATED fields pretending to be canonical.
