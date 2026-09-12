# ARK-011 — STATE-CONDITIONAL LR CONTROL AFTER RECOVERY

## Status

**PREREGISTERED BEFORE EXECUTION.**

This experiment tests the mechanism suggested jointly by ARK-007R and ARK-010:

- ARK-007R: after confirmed G90, `lr=1e-5` strongly reduced subsequent post-G90 instability versus `lr=1e-3` under matched continuation order.
- ARK-010: after an instability episode, continuing at `lr=1e-3` usually reacquired sustained G90 whereas immediately dropping to `lr=1e-5` usually did not.

The unresolved causal question is whether **HIGH LR should be used for reacquisition and LOW LR only after capability has been reacquired**.

## Primary question

After a prospectively observed post-G90 instability and a HIGH-LR recovery, does switching from `1e-3` to `1e-5` at the exact recovery-confirmation checkpoint reduce recurrent instability relative to continuing `1e-3`, when both arms consume byte-identical future minibatches?

## Hypothesis

`H-ADAPTIVE-LR`:

> plasticity mode (`1e-3`) is useful for reacquisition; consolidation mode (`1e-5`) is useful after capability is present. A HIGH -> LOW switch at sustained recovery should preserve the recovered capability better than HIGH-only continuation.

This is a **new hypothesis**, not a demonstrated mechanism.

## Cross-branch context, read-only

Before preregistration, Arkenstone inspected:

- `cymek` at `28bf57a0d299a2c13a99fe0046616c00a1b8530c`;
- `cymek-500m-readiness` at `3f92cf8c185379f138f3da4649ebadc6d94c31c4`.

No Cymek files are modified by ARK-011. Relevant engineering lessons only: exact continuation/resume identity, fail-closed receipts, no-rewarm schedule discipline, and strict train/eval boundary separation. Cymek remains the production substrate; Arkenstone remains the discovery lab.

## Frozen task

Canonical T2 no-carry structural-band task from:

`experiments/ARK-002B/TASK_MANIFEST.json`

Required source split SHA256:

`0dd9305697045b0fbf4e7f268b46a4d7276e4794af5d78b60e999df914ae4236`

Train set is unchanged.

### Controller/evaluation firewall

The canonical OOD holdout is deterministically partitioned **without changing membership**:

1. Parse the tens digit of operand A (`6` or `7`).
2. Within each tens band, sort rows by `sha256(prompt + "\\0" + answer)`.
3. Even-ranked rows -> `OOD_CONTROL`.
4. Odd-ranked rows -> `OOD_SEALED`.

`OOD_CONTROL` may influence acquisition/collapse/recovery state transitions.
`OOD_SEALED` must **never** influence optimizer state, LR switching, stopping of an event, or whether a fork is executed. It is measurement-only.

The runner must write the exact derived split membership and SHA256 identities to the result receipt.

## Model and optimizer

Historical Arkenstone Micro T2 subject, unchanged:

- `Micro(vocab=19, width=128, layers=4, heads=4, ffn=512)`
- `CompactVocab`
- AdamW `betas=(0.9, 0.95)`, `eps=1e-8`, `weight_decay=0.1`
- gradient clip global L2 = `1.0`
- batch = `64`
- evaluation every `200` optimizer steps
- HIGH LR = `1e-3`
- LOW LR = `1e-5`

Loss/evaluation semantics must come from the historical `experiments/ARK-001/run_ark001.py` implementation. No architecture or objective change is allowed in this experiment.

## Fresh prospective event sources

Independent acquisition seeds:

`1313, 1414, 1515`

For each acquired checkpoint, frozen continuation-order seeds:

`5701, 5702, 5703, 5704`

Maximum 12 prospective event opportunities.

These seeds/orders are chosen before execution and must not be replaced because an outcome is inconvenient.

## Phase A — acquisition

Train at HIGH LR `1e-3`.

- max steps: `28,000`
- acquisition trigger: first **3 consecutive OOD_CONTROL evals >= 0.90**
- record onset and confirmation separately
- at confirmation, snapshot exact model + optimizer + CPU RNG + CUDA RNG

No OOD_SEALED value may affect this phase.

If a seed does not acquire within the box, record `NO_CONTROL_G90`.

## Phase B — prospective instability generation

From the acquisition-confirmation snapshot, continue at HIGH LR `1e-3` using the frozen order for up to `6,000` steps.

Instability trigger:

- first **3 consecutive OOD_CONTROL evals < 0.90**
- record onset and confirmation separately
- snapshot at confirmation

If no event occurs, record `NO_CONTROL_COLLAPSE` for that acquisition/order opportunity. Do not force a perturbation.

## Phase C — HIGH-LR reacquisition

From the exact instability-confirmation snapshot, continue HIGH LR `1e-3` on the same unused frozen order tail for up to `4,000` steps.

Recovery trigger:

- first **3 consecutive OOD_CONTROL evals >= 0.90**
- snapshot at confirmation

If recovery does not occur, record `NO_CONTROL_RECOVERY`; no retention fork is run for that event.

At the recovery snapshot, measure OOD_SEALED once for analysis only. This measurement does not gate whether the fork executes.

## Phase D — decisive fork at recovery

From the exact same recovery-confirmation snapshot, create two arms:

- `HIGH_CONTINUE`: remain at `lr=1e-3`
- `SWITCH_LOW`: set `lr=1e-5`

Both arms:

- restore identical model/optimizer/RNG state;
- consume byte-identical unused future minibatch indices;
- run exactly `6,000` post-recovery steps;
- evaluate OOD_CONTROL and OOD_SEALED every `200` steps;
- have no further adaptive LR changes.

The first post-recovery optimizer step is recovery confirmation + 1 in the continuation timeline.

## Primary endpoint

Primary scientific readout is on **OOD_SEALED**.

For events where OOD_SEALED >= 0.90 at the recovery fork, define recurrent instability as the first 3 consecutive post-fork OOD_SEALED evals < 0.90.

Report:

- paired recurrent-instability count in HIGH vs LOW;
- risk difference `LOW - HIGH`;
- discordant counts (`HIGH unstable / LOW stable`, reverse);
- result grouped by independent acquisition checkpoint.

All controller-recovered events are still reported, including those whose OOD_SEALED value at the fork is below 0.90; they are not silently dropped.

## Secondary endpoints

For both OOD_CONTROL and OOD_SEALED:

- RET90
- RET50
- mean exact/area
- final exact
- peak exact
- recurrent-instability onset + confirmation

Also record:

- acquisition/collapse/recovery steps;
- actual supervised-token counts from the loss mask;
- L2 and relative parameter displacement from the recovery checkpoint;
- continuation-order SHA256;
- controller/eval split SHA256;
- full plan commit SHA;
- runner commit SHA/source SHA;
- device, torch version, runtime minutes.

## Verdict rules

`SUPPORTED_ADAPTIVE_PROTECTION` requires all of:

1. at least **4** recovery-fork events whose OOD_SEALED exact at the fork is >= 0.90;
2. those events span at least **2 independent acquisition seeds**;
3. HIGH_CONTINUE has at least **2** recurrent-instability events (otherwise protection cannot be distinguished from a low-event regime);
4. paired sealed risk difference `LOW - HIGH <= -0.30`;
5. direction of effect is non-reversed at the acquisition-checkpoint level (LOW is not worse than HIGH in a majority of independent acquisition groups).

`INCONCLUSIVE_LOW_RECOVERY_EVENT_RATE` if condition 1 or 2 fails.

`INCONCLUSIVE_LOW_RECOLLAPSE_RATE` if enough qualified recovery events exist but HIGH_CONTINUE has fewer than 2 recurrent-instability events.

`NOT_SUPPORTED` if event rates are sufficient but the preregistered protection rule does not fire.

Any reverse effect must be reported directly; no threshold tuning after execution.

## Integrity / red-team constraints

- OOD_SEALED is measurement-only and cannot influence the controller.
- Acquisition/order seeds are frozen here and cannot be replaced after seeing outcomes.
- No event may be fabricated by perturbing weights, data, or optimizer state.
- Arms fork from the exact same recovery checkpoint and consume the exact same future batch indices.
- Receipt hashes must be canonical and self-verifying.
- Partial results must be saved after every event opportunity.
- A runtime failure must write a failure receipt and preserve partial evidence.
- 12 order opportunities are clustered within 3 acquisition checkpoints; do not claim 12 independent model replications.

## What ARK-011 cannot prove

Even a positive result would establish only a **micro T2 state-conditional LR control effect**. It would not demonstrate:

- transfer to non-arithmetic cognition;
- benefit on natural language pretraining;
- compatibility with Cymek's production WSD schedule;
- improved plasticity-retention tradeoff on new skills;
- a universal optimizer law;
- AGI.

A positive result may justify a later transfer experiment and an optional Cymek integration proposal; it does **not** authorize modifying Cymek's production scheduler by itself.
