# ARK-019 V2 ADDENDUM — REAL-MIXTURE CAPABILITY GUARDIAN

## Status

**PREREGISTERED BEFORE ARK-019 IMPLEMENTATION / EXECUTION.**

This addendum refines base `PLAN.md` (commit `f1da21382ae15d0ab1ae12308af07c53a088a874`). Entry gates and evidence-dependent protection selection remain unchanged. V2 makes the controller test closer to an actual training loop: the model must continue real-text learning while acquiring SKILL_B and preserving SKILL_A.

## Why V2

A controller that only fine-tunes on SKILL_B could look good while forgetting the real-data substrate, or could solve the problem by spending nearly all time in protection. The actual infrastructure question is multi-objective:

> While useful training continues, can the system learn a new capability, retain an old broader capability, and preserve/improve language modeling without simply freezing?

## Frozen training mixture

Every optimizer update uses a fixed-size packed batch with equal total context-token budget across arms.

Default mixture in PLASTIC state:
- 7/8 sequences: real TRAIN text continuation from the exact ARK-018 stream;
- 1/8 sequences: SKILL_B order-augmented examples;
- 0 SKILL_A replay unless the selected protection mechanism calls for it.

All arms receive identical real-text sequence IDs and identical SKILL_B semantic IDs before any protection replacements. If replay protection activates, it replaces real-text slots, **never SKILL_B slots**, so new-skill exposure stays exactly matched. Exact displaced real-text tokens are a cost metric.

## Arms

1. `PLASTIC_HIGH`
   - fixed HIGH LR;
   - base 7/8 real + 1/8 SKILL_B mixture;
   - no SKILL_A protection.

2. `LOW_ALL`
   - HIGH/100 throughout;
   - same base mixture;
   - near-freezing reference.

3. `STATIC_PROTECT`
   - selected upstream protection active from step 1;
   - same SKILL_B exposure as every other arm;
   - any replay replaces real-text slots only.

4. `GUARDIAN_REACTIVE`
   - base mixture/HIGH in PLASTIC;
   - evaluate SKILL_A CONTROL every 200 steps;
   - enter PROTECT immediately on first CONTROL robust-qualification failure;
   - return to PLASTIC after 3 consecutive qualified CONTROL evals;
   - after SKILL_B CONTROL qualifies for 3 consecutive evals, enter CONSOLIDATE using selected protection for the remaining horizon.

5. `GUARDIAN_MARGIN`
   - same as reactive Guardian except it may enter PROTECT **before formal failure** when either SKILL_A CONTROL ORDER_ONLY or QUERY_ORDER drops below 0.90 at two consecutive evaluations;
   - catastrophic trigger remains immediate if formal qualification fails once;
   - exit after 3 consecutive CONTROL evals with both ORDER_ONLY and QUERY_ORDER >=0.90 and formal qualification satisfied.

The margin controller is preregistered because ARK-015 showed canonical accuracy can remain perfect while invariance erodes; waiting for formal failure may be too late. It is evaluated against the reactive controller, not selected post hoc.

SEALED never affects controller state.

## Controller state contract

Checkpoint must persist exactly:
- current state `PLASTIC | PROTECT | CONSOLIDATE`;
- trigger type and trigger metric values;
- consecutive-pass/fail counters;
- selected protection mechanism + dose;
- replay schedule position and RNG;
- update-cap reference state if relevant;
- optimizer/scaler/RNG states;
- real-text stream position;
- SKILL_B semantic stream position;
- capability registry version.

A pre-execution smoke test must fork, checkpoint, resume and prove identical next 10 updates/metrics versus uninterrupted execution.

## Primary comparison

Primary controller candidate is whichever of `GUARDIAN_REACTIVE` and `GUARDIAN_MARGIN` satisfies the full success criterion with lower protection cost. Selection order is frozen:

1. old-skill SEALED failure risk;
2. SKILL_B qualification speed;
3. SEALED real-text NLL;
4. protection duty cycle;
5. displaced real-text tokens.

If neither satisfies success criterion, no Guardian is promoted.

## Revised success criterion

`CAPABILITY_GUARDIAN_PROXY_CANDIDATE` requires >=4 matched sets across >=2 SKILL_A parents and all:

- PLASTIC_HIGH exhibits meaningful interference: >=2 SKILL_A SEALED failures **or** mean robustness-area deficit >=0.15 versus STATIC_PROTECT;
- candidate Guardian old-skill risk <= STATIC_PROTECT +0.10 and, if failures occur, >=0.30 below PLASTIC_HIGH;
- candidate median SKILL_B qualification step <=1.5x PLASTIC_HIGH;
- candidate SKILL_B final SEALED robustness within 0.05 of PLASTIC_HIGH;
- candidate SEALED real-text NLL no more than 5% worse than PLASTIC_HIGH and not materially worse than STATIC_PROTECT;
- candidate pre-SKILL_B-qualification protection duty cycle <60%;
- candidate displaces <10% of real-text tokens before SKILL_B qualification when replay is the mechanism;
- exact-resume test passes.

`ANTICIPATORY_MARGIN_NEEDED` is an additional flag if margin Guardian qualifies but reactive Guardian fails due to late intervention.

`REACTIVE_SUFFICIENT` if both qualify and reactive has equal or lower total cost.

## Output for future training architecture

ARK-019 V2 must emit not only `CAPABILITY_STATE_TRACE.json`, but also `GUARDIAN_POLICY_SPEC.json` containing a hardware-agnostic policy contract:

- probe names and thresholds;
- state transition logic;
- intervention semantics;
- replay/update budget accounting;
- exact resume fields;
- metrics required from the trainer;
- failure modes that disable the controller.

This file is a research interface specification, not production authorization.