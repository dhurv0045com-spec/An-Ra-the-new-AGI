# ARK-016 — UPDATE-MAGNITUDE RED TEAM OF STATE-CONDITIONAL LR PROTECTION

## Status

**PREREGISTERED BEFORE IMPLEMENTATION / EXECUTION.**

## Why this experiment exists

ARK-007R and ARK-011 show strong same-task Micro-T2 retention protection after LR is reduced to `1e-5`. But the strongest alternative explanation is still alive: LOW may work mainly because it nearly freezes the parameters.

The Cymek Discovery-V6 audit explicitly keeps this alternative open and requires parameter movement/path diagnostics before treating the effect as a true consolidation mechanism.

ARK-016 asks:

> At the same recovered T2 state, does retention depend on the literal LR value, or primarily on the magnitude of applied parameter updates? Can an update-norm trust region preserve capability while allowing materially more movement than `lr=1e-5`?

This is a causal mechanism red-team. Its purpose is to decide what kind of training controller is worth engineering next: a hysteretic LR state machine, an update-norm/trust-region controller, or neither.

## Frozen task / firewall

Use canonical T2 no-carry task SHA256:

`0dd9305697045b0fbf4e7f268b46a4d7276e4794af5d78b60e999df914ae4236`

Use the ARK-011 deterministic `OOD_CONTROL` / `OOD_SEALED` split algorithm.

- OOD_CONTROL may trigger acquisition, instability, and recovery states.
- OOD_SEALED is measurement-only.
- No sealed value may affect LR, update caps, stopping, arm selection, or event inclusion.

## Subject / optimizer

Historical Micro T2 subject:

- width 128;
- 4 layers;
- 4 heads;
- FFN 512;
- `CompactVocab`;
- AdamW `betas=(0.9,0.95)`, `eps=1e-8`, `weight_decay=0.1`;
- batch 64;
- gradient clip 1.0;
- eval every 200;
- HIGH LR = `1e-3`;
- LOW LR = `1e-5`.

## Fresh event sources

Independent acquisition seeds:

`1919, 2020, 2121`

Frozen continuation-order seeds per acquired checkpoint:

`9901, 9902, 9903, 9904`

Maximum 12 prospective event opportunities. Seeds/orders may not be replaced after outcomes are observed.

## Phase A — acquire, destabilize, recover

Use the ARK-011 state machine:

1. Acquire at HIGH until first 3 consecutive OOD_CONTROL evals >=0.90; max 28k steps.
2. Continue HIGH on the frozen continuation order until first 3 consecutive OOD_CONTROL evals <0.90; max 6k steps.
3. Continue HIGH from that exact instability-confirmation snapshot until first 3 consecutive OOD_CONTROL evals >=0.90; max 4k steps.
4. Snapshot exact recovered model + optimizer + CPU RNG + CUDA RNG at recovery confirmation.
5. Measure OOD_SEALED once for analysis only; it does not gate arm execution.

If an opportunity does not collapse or recover inside its preregistered box, record the corresponding no-event status. No perturbation is allowed to manufacture events.

## Phase B — decisive movement-control fork

From the exact same recovered checkpoint and identical unused continuation minibatches, run 6,000 post-recovery optimizer steps for every arm.

### Arm 1 — `LOW_REFERENCE`

- AdamW LR `1e-5`.
- This is the established post-recovery protection policy.
- For **every optimizer step**, record the global L2 norm of the actual parameter delta after the optimizer step.
- The resulting per-step delta trace is SHA256-bound and becomes the preregistered cap schedule for the movement-control arms on that same event.

The delta trace is a function only of the LOW_REFERENCE model path and matched training minibatches. No evaluation value contributes to it.

### Arm 2 — `HIGH_UNCAPPED`

- AdamW LR `1e-3`.
- No post-step cap.
- This is the high-movement instability reference.

### Arm 3 — `HIGH_CAP_1X`

- AdamW internal LR remains `1e-3`.
- After each optimizer step, compute the raw proposed global parameter-delta L2 norm relative to parameters immediately before that step.
- Let `c_t` be the actual LOW_REFERENCE delta norm at the same matched step `t`.
- If the raw HIGH delta exceeds `c_t`, rescale the **applied parameter delta** globally so its L2 norm equals `c_t`.
- If the raw delta is already <= `c_t`, leave it unchanged.
- AdamW moment state is **not** rescaled or rewritten.

This arm tests whether low-scale applied movement is sufficient for retention even when the optimizer is otherwise running with HIGH-LR state.

### Arm 4 — `HIGH_CAP_10X`

Same as HIGH_CAP_1X, except the cap is `10 * c_t`.

This arm asks whether a larger movement budget can retain capability while avoiding the full instability of uncapped HIGH. It is the key candidate for an actionable update-trust-region controller.

## Important interpretation of capped arms

Post-step delta capping is an experimental intervention, not standard AdamW. It intentionally separates **applied parameter movement** from the nominal optimizer LR while leaving optimizer moments on the HIGH-LR path.

Therefore:

- if HIGH_CAP_1X behaves like LOW_REFERENCE, that supports update magnitude as a major mediator;
- if LOW_REFERENCE protects but HIGH_CAP_1X fails despite matched/lower applied step norms, literal LR/trajectory/optimizer-state effects remain important;
- if HIGH_CAP_10X protects while moving materially more than LOW_REFERENCE, it becomes a stronger algorithm candidate than near-freezing LOW.

## Per-step telemetry

For every arm record:

- clipped gradient global L2 norm before optimizer step;
- raw proposed parameter-delta L2 norm;
- applied parameter-delta L2 norm;
- cap value and whether cap fired;
- cumulative applied path length `sum_t ||delta_t||_2`;
- displacement from recovery checkpoint;
- relative displacement;
- supervised-token count.

For LOW_REFERENCE, raw proposed = applied delta.

Telemetry may be downsampled in the JSON trajectory for storage, but aggregate sums/max/means and the full LOW delta-cap trace hash must be preserved. The implementation may store the LOW cap trace itself if size permits.

## Primary retention endpoint

Among recovery forks where OOD_SEALED >=0.90 at the recovery snapshot, define recurrent instability as the first 3 consecutive post-fork OOD_SEALED evals <0.90.

Report per arm:

- recurrent-instability count;
- RET90;
- sealed area;
- final sealed exact;
- grouped outcomes by acquisition seed;
- paired discordances versus HIGH_UNCAPPED and LOW_REFERENCE.

## Mechanism verdicts

Event sufficiency gate:

- at least 4 SEALED-qualified recovery forks;
- at least 2 independent acquisition seeds;
- HIGH_UNCAPPED must have at least 2 recurrent-instability events.

Otherwise verdict = `INCONCLUSIVE_LOW_EVENT_RATE`.

### `UPDATE_MAGNITUDE_MAJOR_MEDIATOR`

Requires event sufficiency plus:

1. HIGH_CAP_1X recurrent-instability risk is at least 0.30 lower than HIGH_UNCAPPED;
2. HIGH_CAP_1X risk is within 0.20 absolute of LOW_REFERENCE;
3. median cumulative applied path length of HIGH_CAP_1X is within `[0.5x, 1.5x]` of LOW_REFERENCE.

### `TRUST_REGION_CANDIDATE`

Requires event sufficiency plus:

1. HIGH_CAP_10X recurrent-instability risk is at least 0.30 lower than HIGH_UNCAPPED;
2. HIGH_CAP_10X mean/median cumulative applied path length is >=3x LOW_REFERENCE;
3. HIGH_CAP_10X sealed RET90 is no more than 0.10 below LOW_REFERENCE;
4. no majority acquisition-seed group shows HIGH_CAP_10X worse than HIGH_UNCAPPED.

This verdict means an update-norm controller deserves a larger proxy test. It does **not** mean the 10x factor is universally optimal.

### `LOW_LR_SPECIFIC_BEYOND_STEP_NORM`

Requires event sufficiency plus:

1. LOW_REFERENCE recurrent-instability risk is <=0.25;
2. HIGH_CAP_1X risk exceeds LOW_REFERENCE by >=0.30;
3. HIGH_CAP_1X median cumulative applied path length is <=1.5x LOW_REFERENCE.

This would argue that matching small applied update norms is not sufficient; optimizer trajectory/state or the literal LR intervention matters.

If none of the above fire with sufficient events, verdict = `MECHANISM_MIXED_OR_UNRESOLVED`.

Multiple positive flags may be reported if logically compatible; the final summary must state which conditions fired exactly.

## What ARK-016 can change

The experiment is explicitly designed to choose between infrastructure directions:

- **LR-state-machine candidate** if LOW remains special beyond movement matching;
- **update-trust-region candidate** if capped HIGH preserves while allowing useful movement;
- **do not promote either** if events are insufficient or results are mixed.

## Claim limits

Even a clean result remains Micro T2 evidence. It does not establish:

- natural-language pretraining benefit;
- non-arithmetic transfer by itself;
- TPU equivalence;
- a universal update-norm threshold;
- permission to alter Cymek production WSD;
- AGI.
