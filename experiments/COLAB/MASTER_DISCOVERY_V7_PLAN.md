# MASTER DISCOVERY V7 — TRANSFER + MECHANISM CAMPAIGN

## Status

**PREREGISTERED BEFORE MASTER RUNNER / NOTEBOOK IMPLEMENTATION AND BEFORE GPU EXECUTION.**

## Mission

Discovery V6 established a real Micro-T2 state-conditional retention effect but left the two questions that matter most for designing the next training system unresolved:

1. **Transfer:** does the retention effect survive on a robust non-arithmetic capability when the continuation distribution actually creates a retention challenge?
2. **Mechanism:** is LOW-LR protection mainly a consequence of near-freezing parameter motion, or can a more useful update-magnitude controller preserve capability while allowing more learning movement?

Discovery V7 spends one long Colab GPU session on exactly those questions. It does not add filler experiments to consume wall time.

## Frozen experiment plans

- ARK-015 plan commit: `da56dfbbafc6951c8d1c17c86410187cd7472e74`
- ARK-016 plan commit: `459b8cebb080a7e325378f6396a37963d8fc4ac1`

The master implementation must bind these exact preregistration commits into every final summary.

## Read-only external context

The design considered the live `cymek-500m-readiness` branch at HEAD `125b25c19204cce1994deebbfc4957119f2ae31f`, especially its read-only audit of Arkenstone Discovery V6. No Cymek file is modified by this campaign.

The campaign is meant to produce evidence and an explicit handoff decision, not to silently alter Cymek's production scheduler.

## Campaign A — ARK-015

**Non-arithmetic invariance retention under distribution narrowing.**

Primary question:

> after ORDER_AUGMENTED robust binding acquisition, does canonical-only continuation at HIGH LR destroy sealed order robustness more often than the byte-identical LOW-LR continuation?

Key design:

- exact ARK-014 task manifest;
- fresh acquisition seeds 2301/2402/2503;
- continuation semantic orders 8801/8802/8803;
- primary NARROW_HIGH vs NARROW_LOW pairs;
- AUGMENTED_HIGH_REFERENCE stress-specificity control;
- 12k continuation horizon;
- CONTROL/SEALED firewall;
- parameter displacement and path-length telemetry.

Target allocation: up to **135 minutes**.

If ARK-015 finishes earlier, unused budget flows to ARK-016.

## Campaign B — ARK-016

**Update-magnitude mechanism red-team.**

Primary question:

> at the same recovered T2 state, can HIGH-LR optimizer dynamics with an explicit applied-update trust region reproduce LOW retention, and can a 10x-low movement budget preserve capability while moving materially more than LOW?

Key design:

- fresh T2 acquisition seeds 1919/2020/2121;
- orders 9901..9904;
- prospectively acquire -> destabilize -> HIGH recover;
- exact recovery-checkpoint fork;
- LOW_REFERENCE;
- HIGH_UNCAPPED;
- HIGH_CAP_1X using the matched LOW per-step delta norm as cap schedule;
- HIGH_CAP_10X;
- full raw/applied update-norm, cumulative path, displacement, gradient, and retention telemetry;
- CONTROL/SEALED firewall.

Target allocation: all remaining campaign time after ARK-015, with a preferred maximum of **165 minutes**.

## Total runtime budget

Nominal master budget: **300 minutes / 5 hours**.

This is a safety/compute budget, not a requirement to remain busy for five hours.

Rules:

- no sleeping or artificial padding;
- no duplicate run solely to consume remaining minutes;
- partial results saved after every completed acquisition/event/order;
- once a matched pair/set has started, finish the complete matched unit even if this slightly crosses the nominal budget;
- do not start a new expensive matched unit when remaining budget is below that experiment's conservative launch gate;
- package all completed evidence in `finally` on failure.

Expected actual runtime may be materially below 300 minutes if event counts are low or experiments finish early.

## Mandatory preexecution smoke gates

The pinned Colab checkout must fail closed unless all pass:

1. CUDA available.
2. checked-out HEAD equals the notebook's pinned executable commit.
3. Python compilation of master/common/ARK-015/ARK-016 and reused ARK-011/ARK-014 dependencies.
4. canonical T2 manifest SHA matches.
5. ARK-015 binding manifest SHA exactly matches ARK-014 V6 manifest `fbc8605dc1cfc19ac8692d2b2c6338fcb2af3e02b24907f3b5cca3571947fee3`.
6. binding CONTROL/SEALED fact-set overlap = 0.
7. order augmentation deterministic for repeated inputs.
8. one CUDA forward/backward/AdamW update succeeds.
9. exact model/optimizer snapshot reload.
10. same snapshot + same minibatch + same LR produces identical next parameter SHA.
11. post-step global delta-cap primitive is numerically checked: applied delta <= requested cap within tolerance and uncapped optimizer-state moments remain finite.
12. receipt writing + packaging path writable.

A failed smoke gate blocks the full campaign.

## Evidence integrity

Every result JSON must include:

- experiment ID;
- experiment plan commit SHA;
- master plan commit SHA;
- pinned runner commit SHA;
- runner source SHA256;
- task/manifest SHA(s);
- device + torch version;
- runtime minutes;
- frozen acquisition/order seeds;
- continuation-order hashes;
- supervised-token counts;
- receipt SHA256.

Master output must preserve partial experiment receipts rather than replacing them with prose-only summaries.

## Post-campaign training-design decision

The master runner must emit `TRAINING_DESIGN_DECISION.json` using the following preregistered decision logic. This file is a **research handoff recommendation**, not an automatic production action.

### Decision A — `CANDIDATE_STATE_LR_CONTROLLER`

Emit when:

- ARK-015 verdict = `SUPPORTED_NONARITHMETIC_INVARIANCE_PROTECTION`, and
- ARK-016 fires `LOW_LR_SPECIFIC_BEYOND_STEP_NORM`, and
- ARK-016 does not fire `TRUST_REGION_CANDIDATE`.

Recommended next-training prototype:

- capability-state probe registry;
- state machine `ACQUIRE/RECOVER -> CONSOLIDATE`;
- state-triggered LR transition;
- sealed evaluation kept outside the controller;
- update/path telemetry retained for red-team.

### Decision B — `CANDIDATE_UPDATE_TRUST_REGION`

Emit when:

- ARK-015 verdict = `SUPPORTED_NONARITHMETIC_INVARIANCE_PROTECTION`, and
- ARK-016 fires `TRUST_REGION_CANDIDATE`.

Recommended next-training prototype:

- capability-state probe registry;
- optimizer wrapper measuring raw update norm;
- state-conditional applied-update trust-region budget;
- log raw/applied update norm, cumulative path length, and relative displacement;
- compare against existing WSD/LR schedule as challenger, not replacement.

### Decision C — `RETENTION_EFFECT_TRANSFERRED_MECHANISM_UNRESOLVED`

Emit when ARK-015 is supported but ARK-016 is inconclusive/mixed.

Next action: larger proxy mechanism study; do not hard-code an LR or update-norm law.

### Decision D — `ARITHMETIC_OR_STRESS_SPECIFIC_ONLY`

Emit when ARK-015 has sufficient retention events but returns `TRANSFER_NOT_SUPPORTED`, regardless of ARK-016 T2 mechanism result.

Next action: do not promote the Micro-T2 optimizer mechanism to general training infrastructure.

### Decision E — `INSUFFICIENT_EVENT_EVIDENCE`

Emit when ARK-015 or ARK-016 is event-rate inconclusive and no stronger decision above is justified.

Next action: improve event-generating test design or move to the already-designed Cymek GPU proxy only if its independent preregistration/readiness gates justify it.

## Required infrastructure telemetry recommendation

Regardless of scientific verdict, the master decision receipt must state whether the evidence supports instrumenting the next serious training run with these **measurement-only** components:

- capability probe registry with explicit CONTROL vs SEALED roles;
- optimizer raw/applied update-norm logging;
- cumulative parameter path length;
- relative displacement from capability milestones;
- state-transition event log (acquired / unstable / recovered / consolidated);
- data-mixture/presentation regime identifier;
- exact resume identity for controller state;
- old-skill + new-skill metrics when any curriculum/mixture shift occurs.

Measurement instrumentation may be recommended even when an intervention is not.

## Claim boundary

Discovery V7 cannot by itself authorize:

- a universal LR law;
- a universal update-norm law;
- Cymek production WSD modification;
- TPU equivalence;
- PRE500M/500M execution;
- AGI claims.

A positive result can justify a precisely scoped algorithm challenger and the telemetry needed to test it at the next scale.
