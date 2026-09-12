# MASTER DISCOVERY V6 — MULTI-EXPERIMENT GPU CAMPAIGN

## Status

**PREREGISTERED BEFORE RUNNER IMPLEMENTATION OR EXECUTION.**

This campaign is designed to spend a long Colab T4 session on multiple high-information questions rather than extending one experiment merely to consume wall time.

## Scientific order

1. **ARK-011** — full state-conditional HIGH→LOW retention controller test using its already-frozen plan.
2. **ARK-012** — selected-event recovery-switch threshold map: when should LOW begin?
3. **ARK-013** — stability–plasticity frontier: does LOW protection prevent learning a new carry skill, and can adaptive HIGH→LOW improve the Pareto frontier?
4. **ARK-014** — repair ARK-009 order robustness and, if robust binding qualifies, screen non-arithmetic transfer of LR protection.

The order is fixed before execution. Later campaigns may be skipped by budget gates, but the runner must never reorder campaigns based on favorable results.

## Runtime policy

- Target environment: Colab CUDA/T4.
- Combined campaign wall budget: **240 minutes**.
- No artificial sleeping or filler work is permitted.
- Compute is spent only on preregistered training/evaluation.
- The orchestrator allocates at most ~105 minutes to ARK-011 and reserves the remaining budget for ARK-012/013/014.
- If ARK-011 ends earlier, unused time becomes available to later campaigns.
- If a campaign cannot begin with its minimum reserve, it is recorded `BUDGET_BLOCKED` rather than started and truncated silently.

Expected runtime is roughly 2–4 hours depending on event frequency and acquisition speed. Finishing earlier with all executable questions answered is scientifically valid; the clock itself is not an endpoint.

## Integrity

- Every experiment's own PLAN.md is authoritative for scientific thresholds.
- All plan commit SHAs, runner commit SHA/source SHA, device, torch version, task-manifest hashes, continuation-order hashes, actual supervised-token counts, and runtime are written to receipts.
- Controller sets and sealed measurement sets remain separated as specified by each plan.
- Partial JSON is written after each completed source/order/arm.
- Any exception writes a failure receipt and a final ZIP containing all completed JSONs.
- The notebook must pin one exact runner commit, delete stale clones, compile the runner and relevant dependencies, run GPU smoke tests, and only then launch the campaign.
- Execution artifacts beat summaries. No result is promoted until receipt hashes and plan binding are audited.

## Campaign-specific budget gates

### ARK-011

Run the existing full runner with a campaign allocation of up to 105 minutes. Its own prospective seed/order design remains unchanged. If the allocation ends before all 12 opportunities, status is partial and the frozen omitted opportunities remain unexecuted, not negative.

### ARK-012

Minimum reserve to start: 35 minutes. Execute frozen source combinations in listed order. Stop starting new sources when <15 minutes remain for this campaign allocation. Completed source schedules remain valid selected-event screens.

### ARK-013

Minimum reserve to start: 50 minutes. Execute acquisition seed 1717 before 1818; order 6801 before 6802. A matched triplet is only counted when all three arms complete the same 12k new-skill horizon. Do not compare a truncated arm to completed arms.

### ARK-014

Minimum reserve to start: 40 minutes. Run CANONICAL_TRAIN and ORDER_AUGMENTED acquisition at seed2201 under the same max-step box. If robust qualification occurs and >=20 minutes remain, run retention continuation seeds in ascending order 7701..7703. Do not inspect BIND_SEALED to choose which acquisition arm proceeds; continuation eligibility is CONTROL-only as specified in PLAN.md.

## Pre-execution checks

The V6 smoke test must verify:
- canonical T2 manifest SHA;
- CONTROL/SEALED partitions are disjoint and exhaustive;
- T3CARRY manifest determinism, forced-carry invariant, train/OOD commutation-overlap=0;
- binding fact-set CONTROL/SEALED split has zero overlap;
- canonical/order-only/query-order diagnostic construction preserves answer semantics;
- deterministic order augmentation reproducibility;
- GPU forward/backward/optimizer step;
- snapshot+optimizer+RNG reload;
- same snapshot + same minibatch + same LR produces identical next parameter hash;
- continuation/order hashes reproduce;
- result packaging in a finally path.

## Program-level decision table

After execution, classify each experiment separately. No aggregate "V6 success" may hide a negative component.

The campaign may justify the following next steps only if evidence supports them:
- ARK-011 positive -> adaptive controller remains a mechanism candidate.
- ARK-012 ordered threshold effect -> design a state estimator rather than a fixed step schedule.
- ARK-013 adaptive Pareto improvement -> low-LR protection is not merely useful because learning stops.
- ARK-014 positive transfer screen -> authorize a fresh multi-seed non-arithmetic replication.

No combination of these Micro-scale experiments authorizes modifying Cymek production scheduling or claiming AGI.