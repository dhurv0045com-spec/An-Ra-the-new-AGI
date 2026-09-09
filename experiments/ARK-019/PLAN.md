# ARK-019 — CAPABILITY GUARDIAN: CLOSED-LOOP STABILITY–PLASTICITY PROTOTYPE

## Status

**PREREGISTERED BEFORE IMPLEMENTATION / EXECUTION. DEPENDENT ON ARK-017 AND ARK-018 EVIDENCE.**

## Purpose

ARK-015 showed that a model can stay perfect on the narrow distribution while silently losing a broader invariant. ARK-017 is designed to identify whether update magnitude, invariant-supporting replay, or both cause protection. ARK-018 moves the phenomenon onto a 1GB real-data-pretrained substrate.

ARK-019 asks the engineering question that matters for the next training system:

> Can a closed-loop controller preserve an already-acquired capability while the same model learns a new capability, without permanently freezing training or sacrificing real-data language quality?

This is the first Arkenstone experiment whose primary output is an **algorithm/infrastructure promotion decision**, not merely a mechanism measurement.

## Entry gates

ARK-019 may execute only if all are true before its first model update:

1. ARK-018 produced a valid real-data pretrained checkpoint with exact dataset/tokenizer/model hashes and SEALED real-text evaluation.
2. The old invariant capability `SKILL_A` is robustly qualified on CONTROL and SEALED at the checkpoint.
3. ARK-017 produced a mechanism verdict other than `INCONCLUSIVE_LOW_EVENT_RATE` / `MECHANISM_MIXED_OR_UNRESOLVED`, OR ARK-018 independently demonstrates one protection arm meeting its real-substrate criterion.
4. The selected protection mechanism is chosen by the frozen mapping below and written into `ARK-019_ENTRY_RECEIPT.json` before execution.

If any gate fails, status `BLOCKED_BY_UPSTREAM_EVIDENCE`; do not invent a controller.

## Frozen mechanism-selection mapping

Primary source is ARK-017:

- `UPDATE_MAGNITUDE_SUFFICIENT` -> `PROTECTION = UPDATE_CAP_1X`
- `DIVERSITY_SUPPORT_SUFFICIENT` -> `PROTECTION = REPLAY_1OF16`
- `JOINT_CONTROL_REQUIRED` -> `PROTECTION = UPDATE_CAP_1X_PLUS_REPLAY_1OF16`
- `BOTH_LEVERS_SUFFICIENT` -> choose the single mechanism with lower measured cost in ARK-018: first minimize SEALED real-text NLL degradation, then new-skill acquisition slowdown; ties choose REPLAY_1OF16 because it retains greater parameter mobility.

If ARK-017 is unresolved but ARK-018 has exactly one qualifying protection arm, use that arm. If multiple ARK-018 arms qualify without an ARK-017 causal verdict, ARK-019 is blocked rather than selecting post hoc.

All source receipt hashes and the selected mapping are recorded before training.

## Substrate

Start from the exact ARK-018 real-data checkpoint after:

1. real-text pretraining;
2. robust acquisition of `SKILL_A` (the one-hop temporary binding/invariance task).

No re-pretraining is allowed inside ARK-019. The ARK-018 checkpoint hash is immutable.

## New capability: SKILL_B

Use a disjoint temporary-binding namespace drawn from a different set of frequent single-token words in the same tokenizer.

SKILL_B differs from SKILL_A in both token identities and surface template while keeping difficulty reliably learnable:

- SKILL_A template: declarative `X means Y; ... Query: X means`;
- SKILL_B template: table-like `X -> Y | ... Requested X =>`;
- disjoint key/value token sets;
- 3 bindings per example;
- fact-set split before query expansion;
- deterministic order augmentation during SKILL_B training;
- 400 TRAIN / 50 CONTROL / 50 SEALED fact-sets;
- all queries represented;
- zero SKILL_A/SKILL_B semantic fact-set overlap.

The target is deliberately a second reliably acquirable skill rather than the failed ARK-013 carry task. ARK-019 studies stability–plasticity control, not task difficulty.

## Independent forks

Use at least 2 independent SKILL_A fine-tuning seeds from the same real-data pretrained substrate. If ARK-018 produced only one qualified seed, create a second SKILL_A acquisition from the frozen pretrained checkpoint before ARK-019 and bind it in the entry receipt.

For each parent, use 2 frozen SKILL_B semantic-order seeds, giving a target of 4 matched arm sets.

## Arms

All arms begin from the exact same SKILL_A-qualified parent and consume the same SKILL_B semantic-example IDs at every step. Batch size/update count are identical. When replay is active it **replaces** 1/16 of SKILL_B examples rather than adding tokens, so update count and batch token budget remain comparable.

1. `PLASTIC_HIGH`
   - HIGH fine-tuning LR from ARK-018;
   - no old-skill protection;
   - maximum plasticity reference.

2. `LOW_ALL`
   - HIGH/100 for the entire SKILL_B phase;
   - near-freezing / slow-learning reference.

3. `STATIC_PROTECT`
   - selected protection mechanism active from step 1 through the full SKILL_B phase;
   - HIGH LR unless the selected protection itself is an LR rule.

4. `CAPABILITY_GUARDIAN`
   - default state `PLASTIC` with HIGH LR and no protection;
   - every 200 steps evaluate **SKILL_A CONTROL only**;
   - if CONTROL robust qualification fails at any evaluation, enter `PROTECT` immediately;
   - in PROTECT, activate the selected mechanism while continuing SKILL_B training;
   - return to PLASTIC only after SKILL_A CONTROL is robust-qualified for 3 consecutive evaluations;
   - after SKILL_B CONTROL qualifies for 3 consecutive evaluations, enter `CONSOLIDATE` for the remaining horizon: selected protection remains active, but no further new control rule is introduced.

SEALED never affects state transitions.

Controller state, trigger reason, duty cycle, exact transition step, selected mechanism state (including cap reference or replay RNG), optimizer state and RNG must be checkpointed and exactly resumable.

## Horizon

Maximum SKILL_B training: 12,000 optimizer steps, eval every 200.

No arm may stop early at SKILL_B qualification; all arms continue to the same horizon so post-acquisition retention is measurable.

## Evaluation

At every eval milestone measure:

### Old capability SKILL_A
- CONTROL + SEALED canonical, ORDER_ONLY, QUERY_ORDER;
- robust-qualified flag;
- first failure and recovery times.

### New capability SKILL_B
- CONTROL + SEALED canonical, ORDER_ONLY, QUERY_ORDER;
- qualification onset/confirmation;
- area/final.

### Real-data substrate
- fixed CONTROL and SEALED text NLL/perplexity subsets;
- relative NLL change from ARK-018 entry checkpoint.

### Optimization / controller telemetry
- LR;
- raw/applied step delta norm;
- cumulative path;
- relative displacement from entry;
- gradient norm;
- replay fraction actually applied;
- protection duty cycle;
- state transitions;
- exact supervised/new-skill/old-replay token counts.

## Primary stability–plasticity endpoints

For each arm report:

- probability of SKILL_A SEALED robust failure;
- SKILL_A SEALED robustness area;
- SKILL_B qualification-confirmation step;
- SKILL_B SEALED robustness area/final;
- SEALED real-text NLL delta;
- cumulative movement;
- total old-skill replay tokens;
- Guardian protection duty cycle.

## Preregistered Guardian success criterion

`CAPABILITY_GUARDIAN_PROXY_CANDIDATE` requires all:

1. at least 4 matched arm sets across >=2 independent SKILL_A parent seeds;
2. PLASTIC_HIGH produces >=2 old-skill SEALED failures or a mean SKILL_A robustness-area deficit >= .15 versus STATIC_PROTECT;
3. Guardian old-skill failure risk is <= STATIC_PROTECT + .10 and at least .30 below PLASTIC_HIGH when failure events exist;
4. Guardian median SKILL_B qualification step <= 1.50 × PLASTIC_HIGH median qualification step;
5. Guardian SKILL_B qualification is at least as reliable as LOW_ALL and materially faster if LOW_ALL qualifies;
6. Guardian SEALED real-text NLL is not >5% worse than PLASTIC_HIGH;
7. Guardian spends <70% of training in PROTECT before SKILL_B qualification, demonstrating that it is not simply static protection in disguise.

If static protection wins retention but Guardian cannot preserve plasticity: `STATIC_PROTECTION_ONLY`.

If PLASTIC_HIGH retains old skill and no contrast exists: `INCONCLUSIVE_LOW_INTERFERENCE`.

If no protection preserves old skill while SKILL_B learns: `CONTROLLER_NOT_SUPPORTED`.

## Infrastructure output

Regardless of intervention verdict, ARK-019 must emit a complete `CAPABILITY_STATE_TRACE.json` suitable as a prototype contract for later Cymek instrumentation:

- probe registry and CONTROL/SEALED role;
- capability ID/version;
- state machine transitions;
- optimizer/update telemetry;
- data-mixture/replay state;
- controller resume state;
- checkpoint and dataset identities;
- old/new capability metrics.

## Promotion boundary

A positive ARK-019 result promotes the controller only to **larger-proxy / Cymek research challenger** status. It does not authorize a production WSD scheduler change, PRE500M, TPU semantics, or 500M training. Cymek must independently reproduce the controller on its real V5 implementation before production consideration.
