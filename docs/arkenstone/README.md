# ARKENSTONE

**Mission:** discover mechanisms that produce transferable cognition per parameter, per token and per compute, then turn only well-supported mechanisms into bounded training challengers.

- **Branch:** `Arkenstone`
- **Base:** `origin/cymek` at `28bf57a0d299a2c13a99fe0046616c00a1b8530c`
- **Central question:** what causes An-Ra to acquire, preserve, recover and transfer useful computation rather than merely lowering token-prediction loss?
- **Start here for current work:** `docs/arkenstone/CURRENT_STATE.md`

## Current evidence state

- T2 memorize→generalize transition is replicated; memorization timing does not predict G90 timing.
- ARK-007R: same-task Micro T2 LOW-LR protection replicated, HIGH `1e-3` failure 9/12 vs LOW `1e-5` 0/12.
- ARK-010: after instability, HIGH reacquired sustained G90 8/9 vs immediate LOW 2/9.
- ARK-011: HIGH-recover→LOW-retain adaptive switch directly supported on Micro T2, HIGH recurrent instability 3/6 vs SWITCH_LOW 0/6.
- ARK-012: no exact universal switch threshold identified.
- ARK-013: LOW is not a general no-replay cross-task forgetting solution.
- ARK-014: deterministic order augmentation repaired robust non-arithmetic binding acquisition.
- **ARK-015: non-arithmetic retention transfer is demonstrated at Micro scale under controlled distribution narrowing.** Across 3 fresh acquisition parents and 8 matched pairs, NARROW_HIGH failed 8/8, NARROW_LOW 0/8, and AUGMENTED_HIGH_REFERENCE 0/8. Canonical exact remained 1.0 while order robustness eroded under narrowed HIGH training.
- **ARK-016: mechanism credit remains unresolved.** Only 1/12 continuation opportunities produced a qualifying collapse→recovery fork.
- **ARK-018 V4 is now EXECUTED and independently audited.** Both seeds × all four arms completed 8000/8000 updates. The final bundle SHA256 is `cea50622b1725eb12308c54355616e2188b7956761db46289c6da33c09c31f7c`; 40/40 receipt hashes and 39/39 manifest-covered files validated. Heavy 10% Birth rehearsal strongly changed Birth-text modeling, but the preregistered Birth-content internalization criterion failed (`+0.000`, `+0.067` versus required `+0.10` in both seeds). A replicated secondary screen showed much slower temporary-binding acquisition after Birth-10% (1200 / >1500 steps) than after matched science replay (300 / 300); mechanism and generality remain unresolved.

Exact ARK-018 result record: `experiments/ARK-018/FINAL_RESULT_AUDIT.md`.

## Current causal picture

ARK-015 shows that continued specialization can narrow a capability without ordinary task accuracy exposing the damage:

`ROBUST INVARIANT CAPABILITY --(HIGH plasticity + narrowed presentation support)--> CANONICAL skill retained, INVARIANT eroded`

ARK-018 adds a second warning:

`STRONG REPEATED CORPUS PRESSURE --> corpus prediction improves strongly, while later new-skill acquisition may slow`

The best current hypothesis is therefore an interaction among **capability state, plasticity/update scale, and continued data support**. Large movement alone is not sufficient because AUGMENTED_HIGH moved farther than NARROW_HIGH while preserving robustness. LOW is not universally superior because HIGH is substantially better for recovery when capability is absent.

## Current execution program

### 1. ARK-017 V2 — immediate Arkenstone priority

Question: is invariant protection caused by update magnitude, continued invariant-support data, or their interaction?

Primary arms remain prospectively frozen: HIGH, LOW, HIGH_CAP1X, HIGH + exact-noncanonical 1/16 replay, CAP+replay, and AUGMENTED_HIGH_REFERENCE. ARK-018 does not justify redesigning ARK-017 after the fact; it changes only the downstream interpretation and the regression requirements for ARK-019.

Status: **PREREGISTERED + IMPLEMENTED + STATICALLY AUDITED; READY FOR PINNED GPU SMOKE; NOT EXECUTED.**

Launcher: `experiments/COLAB/arkenstone_ark017_v2.ipynb`  
Pinned scientific runner commit: `377c4743f8017e3455f576eafb75bb8ab9c50284`

### 2. ARK-018 V4 — completed real-data gate

The operator-bound Common Pile peS2o shard passed the frozen runtime SHA256 gate. The repository Birth Book also matched its frozen identity. The complete two-seed/four-arm study is finished; do not rerun it merely to obtain a different result.

Primary result: **NO_LARGE_BIRTH_SPECIFIC_INTERNALIZATION_EFFECT** under the preregistered criterion. Strong Birth-distribution assimilation and a real scientific-NLL cost were observed. The post-pretraining binding slowdown is a useful replicated screen, not a demonstrated universal plasticity mechanism.

See `experiments/ARK-018/FINAL_RESULT_AUDIT.md`.

### 3. ARK-019 V2 — mechanism-gated Capability Guardian

ARK-018 is no longer a blocker. ARK-017 remains the mechanism-credit blocker.

The Guardian must eventually beat static baselines on two conjunctive goals:

1. preserve SKILL_A while real-text learning and SKILL_B continue;
2. preserve enough plasticity that SKILL_B/new-skill acquisition is not materially slowed.

A controller that preserves old capability only by effectively freezing the model is not a continual-learning solution.

Status: **V2 PREREGISTERED; BLOCKED ON ARK-017; NOT IMPLEMENTED/EXECUTED.**

## Cross-branch representation context

`cymek-500m-readiness` now carries the production-side R1 representation experiment. CYR-GPU-011 showed a major condition-specific divergence: compact 19-symbol representation partially generalized while the production 24,576-token representation memorized but stayed at 0% held-out under the full semantic exposure box. Arkenstone should treat R1 results as read-only external evidence and should not absorb production code just for convenience.

## Training-infrastructure implications already justified

Future serious training proxies should expose capability probes with explicit CONTROL/SEALED roles, robustness/invariance probes, raw and applied update norms, cumulative path and milestone displacement, data-regime/replay identity, acquire/stable/eroding/recovered/protected events, exact resumable controller state, and old-skill/new-skill measurements across distribution shifts.

## Cymek boundary

Arkenstone may inspect Cymek and other branches read-only and hand off research challengers. **No Arkenstone result currently authorizes a Cymek production scheduler change, PRE500M, TPU promotion or 500M training.** Production promotion must independently pass Cymek's gates.

## Program rules

Loss is a diagnostic, never proof of cognition. Execution artifacts beat prose. Failures are preserved. Reproductions are labeled reproductions. Every claim gets a novelty class. Historical preregistration and receipts are immutable. Sealed outcomes cannot become tuning feedback. Branch authority remains explicit.

---

stamped-at-commit: ca08b7fb2361de13969c9b91c6d8ee6356985a6c
