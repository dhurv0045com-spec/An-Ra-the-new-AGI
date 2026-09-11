# ARK-019 V4 — Science-preserving, acquisition-qualified Capability Guardian

**Status:** prospective design after ARK-019 V3.1 result audit; scientific execution not started.  
**Prior evidence anchor:** ARK-019 V3.1 bundle SHA256 `fcc14c5378318b8c447c2735768b72920334d8c9292b445372fd17d3d2b554d8`.  
**Claim ceiling:** real-text proxy continual-learning controller only.

## Why V4 exists

V3.1 produced a strong interference/recovery signal but did not answer the intended continual-learning question. Three design failures were identified prospectively from the completed V3.1 audit:

1. `SKILL_B` never acquired in any arm, including `PLASTIC_HIGH`; the new-skill channel was underpowered.
2. binding-only `SKILL_A` parent acquisition had already damaged science NLL severely before the matched continuation began.
3. a 100-update control interval was too coarse for a reactive controller; every Guardian arm had already formally failed by the first observation.

V4 repairs those failures before asking whether a Guardian can preserve an old capability while a new capability actually forms.

## Scientific question

Can a controller remain highly plastic during real-text continuation, acquire a prospectively viability-qualified new capability, and preserve or rapidly recover an existing capability with less protection cost than strong static replay?

The experiment separates three outcomes that V3 conflated:

- **prevention:** did old capability avoid formal failure?
- **recovery:** if old capability failed, did it return to a healthy state quickly?
- **continual success:** did old capability remain strong while new capability acquired and science modeling remained competitive?

## Stage A — construct valid science-preserving parents

Start from the two completed ARK-018 `SCIENCE_ONLY` checkpoints (`31801`, `31902`).

Unlike V3, SKILL_A acquisition is mixed with real text on every update:

- total slots/update: 32
- SKILL_A slots: 16
- real-text slots: 16
- LR: `3e-4`
- max: 2600 updates
- evaluate every 50 updates

The parent is accepted only after three consecutive `PARENT_CONTROL` evaluations where:

- SKILL_A canonical >= .90;
- order-only >= .85;
- query-order >= .85;
- science CONTROL NLL is no more than 15% worse than the original ARK-018 source checkpoint.

A separate SKILL_A `VALIDATION` split must then qualify. The future `MAIN_CONTROL` and `SEALED` splits are not used to build the parent.

This is intentionally stricter than V3: a parent that possesses SKILL_A but has already destroyed its real-text substrate is not admissible.

## Split hygiene

For each binding skill, the deterministic factset universe is partitioned before training into:

- 400 train factsets;
- 50 parent/dose-control factsets;
- 50 main-control factsets;
- 50 validation factsets;
- 50 sealed factsets.

The adaptive dose stage is allowed to inspect only the dose-control and validation splits. The main Guardian controls use the disjoint main-control split. SEALED never controls training, dose selection, controller state, or early stopping.

## Stage B — prospective SKILL_B viability gate

V3 used only 4 SKILL_B slots/update and never produced the new capability. V4 does not guess a replacement dose and then interpret another failed campaign.

Before the Guardian comparison, a preregistered pilot selects the **smallest viable dose** from:

`8, 12, 16 SKILL_B slots/update`

For each candidate, one pilot fork is run from each qualified parent, with fixed pilot order seeds `419001`, `419002`, high LR and no SKILL_A protection. Each pilot has at most 2000 updates.

A candidate dose passes only if both parent pilots:

- achieve three consecutive dose-control robust qualifications;
- confirm by update <= 1800;
- also qualify on the separate validation split.

The smallest passing candidate is frozen for the main comparison. If no candidate passes, V4 stops **before** the Guardian comparison with `INCONCLUSIVE_NO_VIABLE_SKILL_B_DOSE`.

This stage is a prospectively frozen adaptive calibration, not post-outcome tuning of Guardian arms.

## Stage C — main matched comparison

Main order seeds: `429001`, `429002`.

Matched sets:

`2 parent seeds × 2 order seeds = 4`

Each arm runs exactly 2000 updates. The selected SKILL_B slot count is fixed across every arm. Replay, when active, displaces a real-text slot only; it never reduces SKILL_B exposure.

Arms:

1. `PLASTIC_HIGH` — no protection.
2. `STATIC_REPLAY_1OF64` — one SKILL_A replay slot every second update.
3. `STATIC_REPLAY_1OF32` — one SKILL_A replay slot every update; strong static reference.
4. `STATIC_CAP16X` — no replay; every applied update is capped to 16× the matched LOW-shadow median full delta.
5. `GUARDIAN_REPLAY` — dynamic replay only.
6. `GUARDIAN_HYBRID` — dynamic replay plus emergency CAP16X after persistent failure under 1/32 replay.

This six-arm design directly tests the real-text transfer of both R2 retention levers and compares a dynamic controller against weak and strong static protection references.

## Guardian V4 controller

CONTROL observation interval is reduced from 100 updates to **25 updates**.

States:

`PLASTIC -> SPARSE64 -> REPLAY32 -> [EMERGENCY_CAP16X] -> SPARSE64 -> PLASTIC`

Rules:

- while PLASTIC, one early margin warning (`robust-min < .95`) activates 1/64 replay;
- formal qualification failure immediately activates 1/32 replay;
- Hybrid only: two consecutive 25-update failures while already in REPLAY32 activate CAP16X + 1/32 replay;
- four consecutive healthy CONTROL observations de-escalate one protection level;
- after confirmed SKILL_B acquisition, an optional 200-update sparse consolidation window applies only if the controller is already in sparse protection; it does not force a healthy PLASTIC controller into protection.

The controller reads SKILL_A MAIN_CONTROL only. It never sees SEALED.

## SKILL_B acquisition definition

SKILL_B CONTROL is evaluated every 50 updates. Qualification requires:

- canonical >= .90;
- order-only >= .85;
- query-order >= .85.

Acquisition confirmation requires three consecutive qualifying observations. Final SEALED qualification is required by the main success rule.

Critically, if `PLASTIC_HIGH` does not acquire and finish SEALED-qualified in at least 3/4 matched sets, the main verdict becomes:

`INCONCLUSIVE_MAIN_B_FORMATION_INSTABILITY`

not `GUARDIAN_NOT_SUPPORTED`.

## Measurement schedule

- SKILL_A MAIN_CONTROL / Guardian decision: every 25 updates.
- SKILL_B MAIN_CONTROL: every 50 updates.
- SEALED SKILL_A + SKILL_B: every 100 updates, measurement only.
- science CONTROL + SEALED NLL: every 100 updates, measurement only.
- exact full checkpoint: every 200 updates.

## Primary continual-success rule

A Guardian arm can be promoted only if all of the following hold prospectively:

1. `PLASTIC_HIGH` demonstrates a viable new-skill channel in >=3/4 sets.
2. meaningful old-skill interference exists under unprotected continuation.
3. Guardian final SKILL_A SEALED qualification in >=3/4 sets.
4. Guardian mean SKILL_A SEALED robustness area is no worse than 0.10 below `STATIC_REPLAY_1OF32`.
5. Guardian SKILL_B acquisition + final SEALED qualification in >=3/4 sets.
6. Guardian median SKILL_B confirmation <=1.5× matched `PLASTIC_HIGH` median.
7. mean final SKILL_B robust-min gap versus `PLASTIC_HIGH` <=.05.
8. each matched final science SEALED NLL <=5% worse than `PLASTIC_HIGH`.
9. mean protection duty <60%.
10. mean replay cost <5% of total sequence slots.

Passing produces only:

`GUARDIAN_CONTINUAL_PROXY_CANDIDATE`

It does not authorize production scheduling, PRE500M, 500M, broad continual-learning claims or AGI claims.

## Prevention and recovery are separate

V4 no longer treats every reactive failure as identical to controller failure.

- `PREVENTION_SIGNAL`: a dynamic Guardian avoids formal SKILL_A CONTROL failure in >=3/4 matched sets.
- `RECOVERY_SIGNAL`: a dynamic Guardian that does fail returns to four consecutive healthy CONTROL observations within 200 updates in >=3/4 matched sets.

These are flags. The main continual-success verdict additionally requires new-skill formation and final SEALED/science criteria.

## CAP16X calibration

For every matched parent/order pair, a 32-step LOW (`3e-6`) shadow is run from the exact same parent state using the selected SKILL_B dose. CAP16X equals 16× the median full applied parameter delta.

The LOW shadow is calibration only and is not an outcome arm.

## Durability / multi-session contract

V4 is deliberately allowed to exceed one Colab T4 session.

- session wall: 225 minutes;
- packaging reserve: 10 minutes;
- runtime calibration estimates session count only;
- runtime may never reduce seeds, arms, horizon, dose candidates, thresholds or evaluation frequency;
- parent, pilot and main phases checkpoint to Drive;
- continuation checkpoints every 200 updates;
- rerunning the same frozen notebook resumes exact checkpoint state.

A 10-update exact-resume smoke must verify identical model, optimizer, scaler and telemetry between uninterrupted `10` and `5 + save/load + 5` before main arms run.

V4 fixes the V3 duplicate-telemetry issue by deduplicating and truncating trajectories to the checkpoint step on resume.

## Output

Drive root:

`/content/drive/MyDrive/genisis-arkenstone/ARK019_GUARDIAN_V4/`

Partial session bundle:

`ARKENSTONE_ARK019_V4_GUARDIAN_PARTIAL.zip`

Final bundle:

`ARKENSTONE_ARK019_V4_GUARDIAN_RESULTS.zip`

## Claim boundary

Even a positive V4 remains a ~21M-parameter real-text proxy continual-learning result. It is not evidence of AGI, consciousness, a universal replay law, or production-scale transfer.
