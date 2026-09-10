# ARKENSTONE — CURRENT STATE

**Branch:** `Arkenstone`  
**Updated:** 2026-09-10  
**Purpose:** one-file operational handoff for continuing Arkenstone without reconstructing the program from multiple branches.

## Mission

Discover mechanisms that improve transferable cognition per parameter/token/compute, then promote only prospectively supported mechanisms. Arkenstone is a discovery branch, not production authority.

Canonical discovery loop:

`BUILD → MEASURE → UNDERSTAND → IMPROVE → VERIFY NOVELTY`

Cross-program loop:

`BUILD → MEASURE → UNDERSTAND → DIAGNOSE → PREDICT → INTERVENE → VERIFY → INTERNALIZE → SCALE`

## Evidence that should be treated as current

### Capability formation

- ARK-002B: delayed memorize→generalize transition replicated at Micro T2; memorization does not imply immediate structural generalization.
- Cross-branch CYR-GPU-011 on `cymek-500m-readiness`: same Cymek V5 4L/128w geometry showed a major representation-dependent acquisition divergence. Compact 19-symbol representation reached 56.47% held-out STANDARD at 44.89% of reference exposure, while production 24,576-token representation reached train M99 but remained 0% STANDARD and 0/48 SEALED at 100% reference semantic exposure. This identifies representation/capability formation as a major cross-program bottleneck, but does not isolate whether vocabulary size, segmentation, atomization, tied-output burden or another correlated factor is causal.

### Retention / recovery

- ARK-007R: HIGH `1e-3` failed 9/12 matched forks; LOW `1e-5` failed 0/12 after acquired Micro-T2 capability.
- ARK-010: after collapse, HIGH recovered sustained G90 8/9; immediate LOW recovered 2/9.
- ARK-011: after HIGH recovery, continued HIGH recollapsed 3/6; HIGH→LOW recollapsed 0/6.
- ARK-012: no unique universal numeric switch threshold was identified.
- ARK-013: LOW LR alone did not solve cross-task no-replay interference.

### Invariance / distribution narrowing

- ARK-015: under canonical-only continuation, NARROW_HIGH lost robust order/query invariance 8/8, NARROW_LOW 0/8, AUGMENTED_HIGH_REFERENCE 0/8. Canonical exact stayed 1.0 while broader invariance collapsed under narrowed HIGH training.
- Large parameter movement alone is therefore insufficient as the explanation; continued capability-supporting data can protect under HIGH plasticity.
- ARK-016 did not isolate update-cap/trust-region mechanism because event rate was too low.

### Real-text / plasticity

- ARK-018 V4 is now **EXECUTED and independently audited**. Final bundle SHA256: `cea50622b1725eb12308c54355616e2188b7956761db46289c6da33c09c31f7c`.
- Both seeds × all four arms completed at 8000 updates; 40/40 receipt hashes and 39/39 manifest-covered files independently validated.
- 10% Birth rehearsal strongly reduced Birth NLL but did not meet the preregistered Birth-content internalization threshold versus matched 10% scientific replay: deltas `+0.000` and `+0.067`, where `+0.10` was required in both seeds.
- 10% Birth increased SEALED science NLL by about 3.68% and 3.91% relative to matched 10% scientific replay.
- Secondary temporary-binding acquisition was 1200 / >1500 steps after Birth-10%, versus 300 / 300 after matched 10% science replay. Treat this as a replicated narrow plasticity screen, not a universal law.
- Exact evidence record: `experiments/ARK-018/FINAL_RESULT_AUDIT.md`.

## Current causal model

The best supported picture is not “LOW is best” and not “movement is bad.” It is:

```text
CAPABILITY ABSENT
  → enough plasticity/update strength is required for acquisition or recovery

CAPABILITY PRESENT
  → excessive plasticity under narrowed support can erode broader invariance
  → lower plasticity can protect
  → continued invariant-supporting data can also protect even at HIGH

NEW CAPABILITY ARRIVES
  → old-capability support/replay is probably required
  → protection must not destroy future acquisition speed
```

ARK-018 adds a new warning: optimizing strongly for one repeated corpus can change the substrate in a way that appears to reduce later new-skill acquisition, even when ordinary token prediction on that corpus improves dramatically.

## Immediate Arkenstone execution priority

### 1. ARK-017 V2 — run unchanged

**Status:** PREREGISTERED + IMPLEMENTED + STATICALLY AUDITED / NOT EXECUTED.

Question: is ARK-015 protection caused by applied update magnitude, continued invariant-support data, or their interaction?

Primary arms already frozen:

- HIGH;
- LOW;
- HIGH_CAP1X;
- HIGH + exact-noncanonical 1/16 replay;
- CAP + replay;
- AUGMENTED_HIGH_REFERENCE.

Do not redesign after ARK-018. ARK-018 changes downstream interpretation, not the prospective ARK-017 causal contrast.

Launcher: `experiments/COLAB/arkenstone_ark017_v2.ipynb`  
Pinned scientific runner commit: `377c4743f8017e3455f576eafb75bb8ab9c50284`

### 2. ARK-019 V2 — revise only after ARK-017 evidence

ARK-018 is no longer a blocker. ARK-017 remains the causal-mechanism blocker.

The Guardian trial must now satisfy two conjunctive requirements:

1. preserve SKILL_A under continued real-text + SKILL_B learning;
2. avoid materially slowing SKILL_B/new-skill acquisition.

A controller that protects old skill by effectively freezing the network fails the continual-learning objective.

### 3. Cross-branch R1 representation experiment

R1 lives on `cymek-500m-readiness`, not Arkenstone. Treat it as read-only external evidence. Its purpose is to isolate the CYR-GPU-011 representation divergence before large production training.

Do not merge production R1 code into Arkenstone merely for convenience; preserve causal and branch authority boundaries.

## Claim boundaries

**DEMONSTRATED:** narrow Micro delayed generalization, state-dependent retention/recovery effects, non-arithmetic invariance narrowing/protection, ARK-018 corpus-specific token-level assimilation and science-cost tradeoff.

**SUPPORTED / unresolved mechanism:** stability–plasticity × data-support interaction; ARK-018 heavy-corpus plasticity slowdown.

**NOT DEMONSTRATED:** universal LOW-LR law, universal Guardian, broad continual learning, broad reasoning gain from Birth data, production-scale transfer, AGI, identity or consciousness.

## What not to do next

Do not launch a new architecture soup experiment. Do not silently install LOW LR as the global scheduler. Do not use ARK-018 Birth NLL as evidence of reasoning. Do not alter ARK-017 after seeing later evidence unless explicitly declaring a new experiment. Do not authorize PRE500M/500M from Arkenstone.

## Minimal agent startup sequence

Read, in order:

1. `docs/arkenstone/CURRENT_STATE.md`
2. `experiments/ARK-018/FINAL_RESULT_AUDIT.md`
3. `experiments/ARK-017/PLAN.md`
4. `experiments/ARK-017/PLAN_V2_ADDENDUM.md`
5. `experiments/ARK-017/RUN_READINESS_V2.json`
6. `docs/arkenstone/EXPERIMENT_LOG.md`
7. `docs/arkenstone/MECHANISM_TOURNAMENT.md`
8. `docs/arkenstone/COGNITION_BOTTLENECK_GRAPH.md`

Then either execute ARK-017 V2 or, if execution is unavailable, work only on predeclared analysis/infrastructure that cannot contaminate its prospective outcome.
