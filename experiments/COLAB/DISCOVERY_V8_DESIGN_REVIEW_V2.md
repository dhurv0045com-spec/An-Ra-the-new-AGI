# DISCOVERY V8 — SECOND-PASS DESIGN REVIEW

Date: 2026-09-09

## Review objective

Iterate the V8 program before spending GPU budget. The review asks whether ARK-017/018/019 would actually answer questions useful for the next An-Ra training algorithm, rather than merely extend the toy evidence chain.

## What survived review

The program ordering remains correct:

`controlled causal mechanism -> real-data continuation -> closed-loop controller`

The V7 starting evidence is strong enough to justify this ladder but not strong enough to skip any rung.

## Problems found in V1

### 1. ARK-017 sparse replay was not treatment-exact

V1 reused ARK-014's six-permutation augmentation. Identity order is one of those permutations, so nominally replayed rows could remain canonical. This weakens the data-support treatment.

**Correction:** sparse replay now samples only the five non-identity permutations and asserts all selected replay rows are genuinely non-canonical.

### 2. ARK-017 answered causality but not efficiency

A CAP1X or 1/16 replay rescue would identify a sufficient lever but not tell us whether the intervention can preserve useful plasticity or how much replay is really required.

**Correction:** the primary verdict stays unchanged, but a preregistered conditional secondary screen can test CAP4X/CAP16X or replay 1/32 and 1/64 after the core result, budget permitting.

### 3. ARK-018 V1 used real data mainly to create the substrate

After real-text pretraining, the main retention stress reverted to synthetic canonical-only continuation. That is weaker than the intended question about training on real data.

**Correction:** V2's primary post-capability phase is continued **real-text LM training**. Protection must preserve SKILL_A while the model continues improving/maintaining held-out real-text NLL. A matched distractor-replay arm tests whether any replay benefit is capability-specific rather than generic structured regularization.

### 4. ARK-018 word/token selection was under-specified

“Frequent content-like words” allowed hidden researcher degrees of freedom.

**Correction:** ASCII/length/single-token/frequency/hash rules are now frozen, including a deterministic frequency-threshold fallback.

### 5. ARK-018 fine-tuning LR depended on final pretraining LR

A decayed final LR could accidentally make the supposed HIGH arm too small to acquire or stress the capability.

**Correction:** pretraining schedule and fine-tuning HIGH/LOW values are frozen independently.

### 6. ARK-019 V1 did not keep real-data learning active

Training only SKILL_B would test sequential fine-tuning, not the training environment we ultimately care about.

**Correction:** V2 uses a fixed 7/8 real-text + 1/8 SKILL_B mixture. Protection replay displaces real-text slots, never SKILL_B, making protection cost measurable.

### 7. A reactive controller may detect erosion too late

ARK-015 showed canonical performance can remain perfect while order invariance falls. Waiting for formal capability failure may waste the warning margin.

**Correction:** compare a reactive Guardian with a preregistered anticipatory-margin Guardian. SEALED remains excluded from control.

## What V8 can realistically produce

If all stages are positive, V8 can justify a **research challenger** consisting of:

- capability registry + CONTROL/SEALED probe roles;
- state-aware intervention logic;
- measured update-budget and/or capability-specific data-support floor;
- exact resumable controller state;
- old capability / new capability / base-LM Pareto accounting.

It cannot establish a universal cognition law, production benefit, or authorization for Cymek 500M training.

## Execution order after review

1. Run pinned ARK-017 V2 only.
2. Audit its raw receipts and primary causal verdict.
3. Bind the user's Drive corpus by hash before ARK-018 implementation/execution.
4. Implement ARK-018 against the actual bound data format; run and audit.
5. Only then implement ARK-019 using the prospectively selected upstream mechanism.

This order prevents coding a favored controller before the mechanism and real-data evidence exist.