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
- Cross-branch CYR-GPU-011: compact 19-symbol representation showed partial held-out structural acquisition while the 24,576-token production-style condition reached training mastery but 0% held-out STANDARD. CYR11 remained confounded by simultaneous representation/output-space changes.
- Cross-branch CYR-GPU-012/R1: one prospective matched seed at 512k semantic rows produced V19 = 12.94%, V4096 = 100%, V24576 = 0% held-out STANDARD. This established a large non-monotonic class-space effect in one seed, not a universal 4096 optimum.
- Cross-branch CYR-GPU-013/R1B: two fresh response curves at 128k rows produced V19 0/0, V1024 0/0, V4096 50.6/49.4, V8192 71.8/0, V16384 64.7/12.9, V24576 1.18/0. Frozen verdict: `MIXED_OR_SEED_SENSITIVE_RESPONSE_CURVE`. Intermediate class-space is strongly supported as a developmental regime; the exact optimum remains seed-sensitive and the mechanism remains unresolved.

### Retention / recovery

- ARK-007R: HIGH `1e-3` failed 9/12 matched forks; LOW `1e-5` failed 0/12 after acquired Micro-T2 capability.
- ARK-010: after collapse, HIGH recovered sustained G90 8/9; immediate LOW recovered 2/9.
- ARK-011: after HIGH recovery, continued HIGH recollapsed 3/6; HIGH→LOW recollapsed 0/6.
- ARK-012: no unique universal numeric switch threshold was identified.
- ARK-013: LOW LR alone did not solve cross-task no-replay interference.

### Invariance / distribution narrowing

- ARK-015: under canonical-only continuation, NARROW_HIGH lost robust order/query invariance 8/8, NARROW_LOW 0/8, AUGMENTED_HIGH_REFERENCE 0/8. Canonical exact stayed 1.0 while broader invariance collapsed under narrowed HIGH training.
- ARK-017 V2/R2 is **EXECUTED and audited**. Across 6 matched sets, NARROW_HIGH failed 4/6; LOW, HIGH_CAP1X, HIGH+exact 1/16 noncanonical replay, CAP+replay and augmented-HIGH all failed 0/6. Primary verdict: `BOTH_LEVERS_SUFFICIENT`.
- ARK-017 secondary efficiency screen: CAP4X 0/3, CAP16X 0/3, replay1/32 0/3, replay1/64 0/3. These are strong tuning clues but secondary one-order screens, not universal thresholds.
- The simple hypothesis `retention = small total parameter movement` is falsified: sparse replay preserved the invariant despite cumulative movement larger than failing HIGH. The better supported picture is an interaction between optimization plasticity and whether current data continues to constrain capability-relevant directions.

### Real-text / plasticity

- ARK-018 V4 is **EXECUTED and independently audited**. Final bundle SHA256: `cea50622b1725eb12308c54355616e2188b7956761db46289c6da33c09c31f7c`.
- Both seeds × all four arms completed at 8000 updates; 40/40 receipt hashes and 39/39 manifest-covered files independently validated.
- 10% Birth rehearsal strongly reduced Birth NLL but did not meet the preregistered Birth-content internalization threshold versus matched 10% scientific replay: improvements `+0.000` and `+0.067`, where `+0.10` was required in both seeds.
- 10% Birth increased SEALED science NLL by about 3.68% and 3.91% relative to matched 10% scientific replay.
- Secondary temporary-binding acquisition was 1200 / >1500 steps after Birth-10%, versus 300 / 300 after matched 10% science replay. Treat this as a replicated narrow plasticity screen, not a universal law.

### ARK-019 V3.1 Guardian result

ARK-019 V3.1 is **EXECUTED and independently audited**. Bundle SHA256:

`fcc14c5378318b8c447c2735768b72920334d8c9292b445372fd17d3d2b554d8`

Integrity: ZIP CRC pass; 45/45 manifest-covered files matched; 46/46 embedded receipt hashes recomputed; all 4 matched sets × 4 arms reached 1000 updates; exact-resume smoke passed. Measured wall time was about 141.83 minutes.

Frozen official verdict: `CONTROLLER_NOT_SUPPORTED`.

The scientific interpretation is narrower because the required new capability never formed in any arm, including the unprotected reference:

`NOT_DEMONSTRATED / NEW-SKILL-FORMATION BOTTLENECK`

What V3.1 did demonstrate in the tested real-text proxy setup:

- unprotected `PLASTIC_HIGH` catastrophically destroyed the previously robust SKILL_A in 4/4 matched sets; mean final robust-min was about 0.005;
- `STATIC_REPLAY_1OF64` ended around 0.882 mean final SKILL_A robust-min;
- both reactive Guardian variants ended around 0.963 mean final SKILL_A robust-min and all four Guardian runs finished SEALED-qualified;
- every Guardian suffered an initial formal failure by the first 100-update observation, then replay escalation reconstructed SKILL_A strongly;
- CAP16X was never triggered, so V3.1 provides no comparative evidence for emergency-cap value inside the Guardian.

Critical V3.1 design limitations:

- SKILL_B received only 4 binding slots/update and never acquired in any arm;
- binding-only SKILL_A parent acquisition damaged science modeling severely before the matched continuation began;
- a 100-update controller interval was too coarse to test prevention fairly;
- two trajectories contained duplicate resume telemetry, although deduplication did not change conclusions.

See `experiments/ARK-019/FINAL_RESULT_AUDIT_V3.md`.

## Current causal model

The best supported picture is now:

```text
CAPABILITY ABSENT
  → enough plasticity/update strength and enough capability-relevant exposure are required
  → representation/output geometry can strongly alter whether capability forms at all

CAPABILITY PRESENT
  → excessive plasticity under narrowed or competing support can erode broader invariance
  → lower effective update magnitude can protect
  → sparse continued capability-support data can independently protect while permitting large movement

NEW CAPABILITY ARRIVES
  → first establish that the new capability can actually form under an unprotected plastic reference
  → maintain real-text competence while constructing the old-capability parent
  → remain HIGH/plastic by default
  → observe old-capability health frequently enough to act before deep collapse
  → inject sparse targeted support on early warning
  → escalate replay on formal failure
  → optionally apply a temporary movement cap only after persistent failure under replay
  → de-escalate after recovery so future learning remains fast
```

## Immediate Arkenstone execution priority

### ARK-019 V4 — science-preserving, acquisition-qualified Guardian

**Status:** PREREGISTERED + IMPLEMENTED + DEDICATED STATIC CONTRACT PASS / READY FOR OPERATOR COLAB CUDA PREFLIGHT / NOT EXECUTED.

**Audit note (2026-09-12, ARK-020 mission):** a mission brief referred to a believed V4
result `GUARDIAN_CONTINUAL_PROXY_CANDIDATE`. Independent repository verification found
**no executed V4 result anywhere on the live branch** — `RUN_READINESS_V4.json` still
records `scientific_result_status: NOT_EXECUTED`. The believed result is treated as
unverified; nothing downstream may cite it until an operator bundle is returned and
independently audited.

V4 is the direct repair of V3.1, not an outcome-driven rerun.

Stage A builds two science-preserving SKILL_A parents from the ARK-018 `SCIENCE_ONLY` checkpoints using a 16 real-text / 16 SKILL_A slot mixture. Parent qualification requires robust A plus science CONTROL NLL no more than 15% worse than the original source, followed by validation qualification.

Stage B prospectively selects the smallest viable SKILL_B dose from 8/12/16 slots per update. Both parent pilots must achieve sustained robust acquisition by the frozen deadline and pass validation. If none does, V4 stops before the Guardian comparison with `INCONCLUSIVE_NO_VIABLE_SKILL_B_DOSE`.

Stage C runs 4 matched sets × 6 arms for 2000 updates/arm:

- `PLASTIC_HIGH`
- `STATIC_REPLAY_1OF64`
- `STATIC_REPLAY_1OF32`
- `STATIC_CAP16X`
- `GUARDIAN_REPLAY`
- `GUARDIAN_HYBRID`

The selected SKILL_B exposure is identical across all arms. Replay replaces real-text only. SKILL_A CONTROL is observed every 25 updates; SKILL_B CONTROL every 50; SEALED/science every 100 for measurement only. Checkpoints occur every 200 updates.

V4 separates prevention from recovery. If the unprotected main reference fails to acquire SKILL_B in at least 3/4 matched sets, the verdict is `INCONCLUSIVE_MAIN_B_FORMATION_INSTABILITY`, not a Guardian failure.

The campaign is intentionally exact-resumable across multiple T4 sessions. Runtime calibration may estimate the number of sessions but cannot reduce seeds, arms, horizons, dose candidates, thresholds, or evaluation cadence.

Frozen scientific executable: `8e858c614d764335100dfd41dda6f8e0d0c877a7`  
Plan: `experiments/ARK-019/PLAN_V4.md`  
Preregistration: `experiments/ARK-019/PREREGISTRATION_V4.json`  
Preexecution audit: `experiments/ARK-019/PREEXECUTION_AUDIT_V4.md`  
Readiness: `experiments/ARK-019/RUN_READINESS_V4.json`  
Launcher: `experiments/COLAB/arkenstone_ark019_v4.ipynb`

Dedicated `ARK-019 V4 contracts` CI passed on the frozen executable. A separate broad ESOES workflow failed because its environment omitted NumPy/PyTorch; that unrelated dependency failure is not represented as green evidence.

### ARK-020 V2 — scientific repair of ARK-020 (designed behind V4)

**Status:** PREREGISTERED (`db44e05`, before implementation) + IMPLEMENTED + 36/36 tests
PASS (including real V4 integration contracts and a full CPU integration smoke) +
red-team audit PASS / READY FOR OPERATOR COLAB CUDA RUN / NOT EXECUTED.

The V1 pre-execution audit confirmed five blocking defects (integration-boundary shapes,
information-theoretically-impossible skill C, unused phase seeds, global-step
confirmation bias, resume overclaim). V2 repairs all five, replaces skill C with a
two-hop composition task (TASK_VALIDITY_ANALYSIS.md), and records the externally-audited
V4 result in FINAL_RESULT_AUDIT_V4.md with explicit provenance. V1 remains immutable.
Launcher: `experiments/COLAB/arkenstone_ark020_v2.ipynb` (cell 0 = read-only resume scan).

### ARK-020 — multi-skill continual cognition generalization (designed behind V4)

**Status:** PREREGISTERED (`e2e6e09`, before implementation) + IMPLEMENTED + 29/29 pure
tests PASS + static pre-execution audit PASS / READY FOR OPERATOR COLAB CUDA RUN /
NOT EXECUTED.

ARK-020 generalizes the Guardian question from A→B to a four-skill sequential battery
(A/B binding families, C successor-rule induction with never-trained sealed keys,
D inverse retrieval) under 7 arms including a preregistered REACTIVE vs PREDICTIVE vs
HYBRID controller comparison, a capability registry, risk-allocated replay (max 2
slots/update) with an efficiency gate against STATIC_REPLAY_1OF32, prevention/recovery/
retention reported separately, a protection-cost slope measurement (k = 1→2→3), and a
formation-first gate set that structurally prevents the V3.1 misattribution. It reuses
V4 parents/dose machinery and inherits V4's DOSE_SELECTION when present, so it can run
immediately after V4 on the same Drive substrate. It does NOT depend on any V4 outcome.

Plan: `experiments/ARK-020/PLAN.md` · Preregistration: `experiments/ARK-020/PREREGISTRATION.json` ·
Audit: `experiments/ARK-020/PREEXECUTION_AUDIT.md` · Readiness: `experiments/ARK-020/RUN_READINESS.json` ·
Core: `experiments/ARK-020/ark020_core.py` · Runner: `experiments/ARK-020/run_ark020.py` ·
Tests: `tests/test_ark020.py` · Launcher: `experiments/COLAB/arkenstone_ark020.ipynb` ·
Novelty: `docs/arkenstone/NOVELTY_AUDIT_ARK020.md` · Synthesis: `docs/arkenstone/CONTROLLER_SYNTHESIS.md`

## Claim boundaries

**DEMONSTRATED:** narrow delayed generalization; state-dependent retention/recovery; non-arithmetic invariance narrowing/protection; two independently sufficient Micro retention levers; ARK-018 corpus-specific assimilation/science-cost tradeoff; R3 real-text-proxy catastrophic old-skill interference; large static sparse-replay retention advantage; reactive replay recovery after collapse.

**SUPPORTED / next proxy:** a faster-observing, science-preserving, new-skill-qualified Guardian; intermediate class-space capability-formation regime.

**NOT DEMONSTRATED:** successful continual-cognition Guardian; simultaneous robust old-skill retention + new-skill acquisition; CAP16X value in the real-text Guardian; universal 1/64 replay law; production-scale continual learning; universal 4096-vocab optimum; PRE500M/500M readiness; AGI; identity or consciousness.

## What not to do next

Do not install LOW LR globally. Do not freeze the network to preserve capabilities. Do not promote 1/64 or CAP16X as universal settings. Do not reinterpret V3.1 as either a successful Guardian or a clean Guardian falsification. Do not weaken V4 after observing outcomes. Do not authorize PRE500M/500M from Arkenstone.

## Minimal agent startup sequence

Read, in order:

1. `docs/arkenstone/CURRENT_STATE.md`
2. `experiments/ARK-017/RESULT_V2.md`
3. `experiments/ARK-018/FINAL_RESULT_AUDIT.md`
4. `experiments/ARK-019/FINAL_RESULT_AUDIT_V3.md`
5. `experiments/ARK-019/PLAN_V4.md`
6. `experiments/ARK-019/PREREGISTRATION_V4.json`
7. `experiments/ARK-019/PREEXECUTION_AUDIT_V4.md`
8. `experiments/ARK-019/RUN_READINESS_V4.json`
9. `docs/arkenstone/EXPERIMENT_LOG.md`

Then execute ARK-019 V4 unchanged through its pinned Colab launcher, or if GPU execution is unavailable work only on analysis/infrastructure that cannot contaminate its prospective outcome.
