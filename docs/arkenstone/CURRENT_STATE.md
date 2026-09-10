# ARKENSTONE — CURRENT STATE

**Branch:** `Arkenstone`  
**Updated:** 2026-09-11  
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
- Cross-branch CYR-GPU-011 on `cymek-500m-readiness`: compact 19-symbol representation reached 56.47% held-out STANDARD at 44.89% reference exposure while production 24,576-token representation reached train M99 but 0% STANDARD and 0/48 SEALED at full reference exposure.
- Cross-branch CYR-GPU-012/R1 is now executed: at the fixed 512k-row endpoint one fresh matched seed produced V19 = 12.94% STANDARD, V4096 = 100%, V24576 = 0%. This demonstrates a large non-monotonic class-space effect in one prospective seed, not a universal 4096-token optimum. R1B is the replication/response-curve follow-up on `cymek-500m-readiness`.

### Retention / recovery

- ARK-007R: HIGH `1e-3` failed 9/12 matched forks; LOW `1e-5` failed 0/12 after acquired Micro-T2 capability.
- ARK-010: after collapse, HIGH recovered sustained G90 8/9; immediate LOW recovered 2/9.
- ARK-011: after HIGH recovery, continued HIGH recollapsed 3/6; HIGH→LOW recollapsed 0/6.
- ARK-012: no unique universal numeric switch threshold was identified.
- ARK-013: LOW LR alone did not solve cross-task no-replay interference.

### Invariance / distribution narrowing

- ARK-015: under canonical-only continuation, NARROW_HIGH lost robust order/query invariance 8/8, NARROW_LOW 0/8, AUGMENTED_HIGH_REFERENCE 0/8. Canonical exact stayed 1.0 while broader invariance collapsed under narrowed HIGH training.
- ARK-017 V2 is now **EXECUTED and audited**. Across 6 matched sets, NARROW_HIGH failed 4/6; LOW, HIGH_CAP1X, HIGH+exact 1/16 noncanonical replay, CAP+replay and augmented-HIGH all failed 0/6. Primary verdict: `BOTH_LEVERS_SUFFICIENT`.
- ARK-017 secondary efficiency screen: CAP4X 0/3, CAP16X 0/3, replay1/32 0/3, replay1/64 0/3. These are strong tuning clues but secondary one-order screens, not universal thresholds.
- The simple hypothesis `retention = small total parameter movement` is falsified: sparse replay preserved the invariant despite cumulative movement larger than failing HIGH. The better supported picture is an interaction between optimization plasticity and whether current data continues to constrain capability-relevant directions.

### Real-text / plasticity

- ARK-018 V4 is **EXECUTED and independently audited**. Final bundle SHA256: `cea50622b1725eb12308c54355616e2188b7956761db46289c6da33c09c31f7c`.
- Both seeds × all four arms completed at 8000 updates; 40/40 receipt hashes and 39/39 manifest-covered files independently validated.
- 10% Birth rehearsal strongly reduced Birth NLL but did not meet the preregistered Birth-content internalization threshold versus matched 10% scientific replay: deltas `+0.000` and `+0.067`, where `+0.10` was required in both seeds.
- 10% Birth increased SEALED science NLL by about 3.68% and 3.91% relative to matched 10% scientific replay.
- Secondary temporary-binding acquisition was 1200 / >1500 steps after Birth-10%, versus 300 / 300 after matched 10% science replay. Treat this as a replicated narrow plasticity screen, not a universal law.

## Current causal model

The best supported picture is now:

```text
CAPABILITY ABSENT
  → enough plasticity/update strength is required for acquisition or recovery
  → representation/output geometry can strongly alter whether capability forms at all

CAPABILITY PRESENT
  → excessive plasticity under narrowed support can erode broader invariance
  → lower effective update magnitude can protect
  → sparse continued invariant-support data can independently protect while permitting large movement

NEW CAPABILITY ARRIVES
  → a useful controller should remain HIGH/plastic by default
  → detect declining old-capability margin on CONTROL
  → inject sparse targeted support before formal failure
  → escalate replay and only then apply a temporary movement cap if failure persists
  → de-escalate after recovery so future learning remains fast
```

ARK-018 adds the warning that heavy repeated specialization can damage later new-skill acquisition even when token prediction on the specialized corpus improves.

## Immediate Arkenstone execution priority

### 1. ARK-019 V3 — run the R2-informed Guardian trial

**Status:** PREREGISTERED + IMPLEMENTED + STATIC AUDIT PASS / READY FOR OPERATOR CUDA PREFLIGHT / NOT EXECUTED.

R3 uses the completed ARK-018 `SCIENCE_ONLY` real-text checkpoints for seeds 31801 and 31902, freshly acquires disjoint SKILL_A parents, then creates 4 mandatory matched sets using two SKILL_B order streams.

Primary arms:

- `PLASTIC_HIGH` — no protection;
- `STATIC_REPLAY_1OF64` — static sparse-support reference;
- `GUARDIAN_REPLAY` — PLASTIC → 1/64 on margin warning → 1/32 on formal failure → de-escalate after recovery;
- `GUARDIAN_HYBRID` — same, with temporary CAP16X only when failure persists despite 1/32 replay.

Fixed horizon is 1000 updates per arm, CONTROL evaluation every 100, 4 matched sets, no post-outcome arm/seed/horizon reduction. CAP16X is calibrated prospectively from a 32-step LOW shadow for each matched set. SEALED never controls the policy.

Launcher: `experiments/COLAB/arkenstone_ark019_v3.ipynb`  
Frozen scientific executable: `63dc47d9bdcd85e004c03aaf9a738c9682fcd8f0`  
Readiness: `experiments/ARK-019/RUN_READINESS_V3.json`.

### 2. Cross-branch R1B representation replication

R1B lives on `cymek-500m-readiness`, not Arkenstone. It maps fresh matched response curves over vocabulary class-space sizes 19/1024/4096/8192/16384/24576 to test whether R1's intermediate-class-space advantage replicates.

Arkenstone should consume that result read-only for the capability-formation causal graph.

## Claim boundaries

**DEMONSTRATED:** narrow Micro delayed generalization; state-dependent retention/recovery effects; non-arithmetic invariance narrowing/protection; two independently sufficient Micro retention levers (applied-update control and sparse invariant support); ARK-018 corpus-specific token-level assimilation and science-cost tradeoff; one-seed R1 non-monotonic class-space effect.

**SUPPORTED / next proxy:** dynamic sparse-support Guardian with emergency movement cap; representation/optimization response curve.

**NOT DEMONSTRATED:** universal 1/64 replay law, universal CAP16X law, a successful Guardian, broad continual learning, production-scale transfer, universal 4096-vocab optimum, AGI, identity or consciousness.

## What not to do next

Do not install LOW LR globally. Do not freeze the network to preserve capabilities. Do not promote 1/64 or CAP16X directly from the secondary R2 screen. Do not use Birth NLL as reasoning evidence. Do not alter R3 thresholds, seeds, arms or horizon after seeing its outcomes. Do not authorize PRE500M/500M from Arkenstone.

## Minimal agent startup sequence

Read, in order:

1. `docs/arkenstone/CURRENT_STATE.md`
2. `experiments/ARK-017/RESULT_V2.md`
3. `experiments/ARK-018/FINAL_RESULT_AUDIT.md`
4. `experiments/ARK-019/PLAN_V3_R2_INFORMED_ADDENDUM.md`
5. `experiments/ARK-019/EXECUTION_V3_CLARIFICATIONS.md`
6. `experiments/ARK-019/PREEXECUTION_V3_HORIZON_AMENDMENT.md`
7. `experiments/ARK-019/PREREGISTRATION_V3.json`
8. `experiments/ARK-019/RUN_READINESS_V3.json`
9. `docs/arkenstone/EXPERIMENT_LOG.md`

Then execute ARK-019 V3 unchanged, or if GPU execution is unavailable work only on analysis/infrastructure that cannot contaminate its prospective outcome.
