# ARK-017 V2 ADDENDUM — EXECUTION-CORRECT MECHANISM DISSECTION

## Status

**PREREGISTERED BEFORE V2 IMPLEMENTATION / EXECUTION.**

This addendum does not erase or rewrite `PLAN.md` (base preregistration commit `4d145288215c304310253a93881073d5d1a03800`). It tightens execution semantics after a static pre-run review. Primary scientific question, fresh seeds, CONTROL/SEALED firewall, 8k horizon and core mechanism contrasts remain unchanged.

## Why a V2 addendum is necessary

The first implementation reused ARK-014's generic `ORDER_AUGMENTED` renderer for the 1/16 replay arm. ARK-014 samples all six permutations, including the identity permutation. Therefore a nominal replay-selected example could accidentally remain canonical. That violates the intended statement that exactly 4/64 examples carry non-canonical order support.

The correction must happen before any ARK-017 GPU outcome exists.

## Frozen correction 1 — replay means exactly non-canonical support

For `REPLAY_1OF16` arms:

- exactly four unique batch positions are selected deterministically per step;
- the other 60 rows are byte-identical canonical renderings;
- each selected position is rendered with one of the **five non-identity permutations only**;
- permutation choice is deterministic from `(acquisition_seed, continuation_seed, absolute_step, batch_position, semantic_id)`;
- the selected permutation may never equal canonical order;
- implementation records a replay-schedule SHA256 and asserts `noncanonical_replay_examples == 4 * continuation_steps`.

`AUGMENTED_HIGH_REFERENCE` intentionally retains the original six-permutation ARK-014 augmentation distribution; this correction applies only to sparse replay arms.

## Frozen correction 2 — stronger fork/resume identity smoke

Before expensive execution the V2 smoke test must verify:

1. exact binding manifest SHA;
2. exact 4/64 sparse replay count and non-identity permutations;
3. deterministic replay schedule reproduction;
4. CUDA forward/backward and finite update;
5. exact model **and optimizer** snapshot restore;
6. identical next-update parameter hash from two identical forks;
7. cap primitive: requested cap binds applied parameter delta without nonfinite optimizer state;
8. LOW-reference trace can be consumed by a capped HIGH fork for multiple consecutive steps.

Any failure blocks the campaign and produces a failure receipt.

## Frozen correction 3 — efficiency frontier is secondary and cannot change the primary verdict

The primary six-arm experiment in `PLAN.md` remains the sole source of `UPDATE_MAGNITUDE_SUFFICIENT`, `DIVERSITY_SUPPORT_SUFFICIENT`, `JOINT_CONTROL_REQUIRED`, `BOTH_LEVERS_SUFFICIENT`, or unresolved verdicts.

If and only if the core experiment is event-sufficient and budget remains, one **prospectively conditional secondary dose screen** may run on continuation seed `10801` across all available fresh parents:

### If movement rescue qualifies

Run:
- `NARROW_HIGH_CAP4X`
- `NARROW_HIGH_CAP16X`

where each step cap equals 4x or 16x the matched LOW-reference applied-delta trace.

Purpose: determine whether useful mobility can increase materially above near-freezing LOW while preserving the invariant.

### If sparse replay rescue qualifies

Run:
- `NARROW_HIGH_REPLAY_1OF64` — exactly 1 non-canonical row / 64
- `NARROW_HIGH_REPLAY_1OF32` — exactly 2 non-canonical rows / 64

Purpose: estimate the minimum invariant-supporting data floor.

### If both qualify

Run both dose screens if >=45 minutes remain; otherwise movement screen has priority only when its core arm used <=1.5x LOW cumulative path, otherwise replay screen has priority. This priority rule is frozen before execution.

Secondary dose results may tune a later candidate but may **not** retroactively change the core causal verdict.

## Budget

V2 safety budget becomes **240 minutes** on T4. The experiment should stop earlier if all primary matched sets and any eligible secondary screen complete. Runtime is never padded.

## Claim boundary

Same as base plan: controlled Micro causal-credit experiment only. V2 fixes treatment fidelity and optionally maps efficiency after the primary result; it does not authorize Cymek production changes.