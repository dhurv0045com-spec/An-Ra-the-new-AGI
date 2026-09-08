# ARK-011S — SMALL SCREEN FOR STATE-DEPENDENT LR CONTROL

Status: PRE-EXECUTION SCREENING PLAN. This is intentionally a small, non-decisive run.

## Question
Can the ARK-011 adaptive controller execute end-to-end on a previously high-event T2 condition, and does switching HIGH -> LOW after confirmed recovery show the expected retention direction before we spend a full 3-seed x 4-order budget?

## Why this is small
This screen reuses a previously observed high-event acquisition/order family from ARK-007R/010. It is **not** an independent replication and may not upgrade any mechanism verdict by itself. Its job is to catch implementation/design errors cheaply and estimate effect direction.

## Frozen substrate
- Canonical T2 commutation-free manifest SHA256: `0dd9305697045b0fbf4e7f268b46a4d7276e4794af5d78b60e999df914ae4236`.
- Micro model: 4 layers, width 128, 4 heads, compact vocab.
- AdamW acquisition/recovery HIGH LR = `1e-3`; switched retention LOW LR = `1e-5`.
- Batch 64; eval every 200 steps.
- G90/collapse/recovery thresholds use 3 consecutive evaluations.

## Evaluation firewall
The frozen canonical OOD set is deterministically split by operand-A tens band into:
- `OOD_CONTROL`: may trigger acquisition, collapse, and recovery state transitions.
- `OOD_SEALED`: never changes training or LR; used only to evaluate the post-recovery fork.
The split algorithm must be identical to ARK-011.

## Screening condition
- Acquisition seed: `909`.
- Continuation orders: `2702`, then `2703` only if the first order fails to produce a complete collapse->recovery->fork event or if budget remains.
These are intentionally replay-style conditions because ARK-010 previously observed prospective high-LR instability for seed 909 on these orders.

## Sequence per order
1. Acquire sustained CONTROL G90 at HIGH LR.
2. Continue HIGH on the frozen continuation order until sustained CONTROL collapse90, max 4000 treated steps.
3. From exact collapse-confirmation snapshot, continue HIGH until sustained CONTROL G90 recovery, max 3000 steps.
4. At exact recovery-confirmation snapshot, measure SEALED once. The fork is primary-interpretable only if SEALED >= 0.90.
5. Fork exact same snapshot and exact same unused future minibatches:
   - `HIGH_CONTINUE = 1e-3`
   - `SWITCH_LOW = 1e-5`
   for 3000 steps, evaluating CONTROL and SEALED every 200.

## Budget
Hard Colab screening budget target: **25 minutes**. The runner must save partial receipts and stop opening new order work when the remaining budget is too small.

## Readout
Primary screen readout is descriptive, not a confirmatory p-value:
- Did a complete collapse->recovery->fork event occur?
- Was SEALED >= 0.90 at the recovery fork?
- Did HIGH re-collapse on SEALED while SWITCH_LOW stayed stable?
- SEALED RET90/AREA/FINAL for both arms.
- Parameter displacement from the recovery fork.

## Interpretation
- `SCREEN_DIRECTION_POSITIVE`: at least one sealed-qualified fork, HIGH re-collapses and SWITCH_LOW does not, with no reverse event among completed sealed-qualified forks.
- `SCREEN_DIRECTION_NEGATIVE`: a sealed-qualified reverse event occurs (HIGH stable, SWITCH_LOW re-collapses) or LOW is materially worse on SEALED retention.
- `SCREEN_NO_EVENT`: no complete collapse->recovery->sealed-qualified fork occurs within the small budget.

None of these outcomes replaces ARK-011. Even a positive screen remains **SCREENING / NOT_REPLICATED**.

## Claim discipline
No AGI, universal optimizer, transfer, scale, or Cymek promotion claim is authorized by ARK-011S alone.
