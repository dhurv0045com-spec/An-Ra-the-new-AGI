# ARK-019 V3.1 — FINAL RESULT AUDIT

**Status:** EXECUTED / OPERATOR BUNDLE INDEPENDENTLY AUDITED  
**Audit date:** 2026-09-11  
**Branch:** `Arkenstone`  
**Official frozen verdict:** `CONTROLLER_NOT_SUPPORTED`  
**Scientific interpretation:** R3 does **not** demonstrate a successful Guardian, but it is also **not a clean falsification of the Guardian idea** because the required new capability SKILL_B never acquired in any arm, including the unprotected `PLASTIC_HIGH` reference.

## 1. Bundle integrity

Audited operator artifact:

`ARKENSTONE_ARK019_V3_GUARDIAN_RESULTS.zip`

Observed ZIP SHA256:

`fcc14c5378318b8c447c2735768b72920334d8c9292b445372fd17d3d2b554d8`

Independent checks:

- ZIP CRC: PASS;
- 46 JSON files present;
- 45/45 files covered by `ZIP_MANIFEST.json` matched their raw SHA256;
- 46/46 embedded `receipt_sha256` values recomputed successfully;
- `ARK-019_V3_RESULT.json` status: `COMPLETE`;
- both parent seeds qualified;
- all 4 matched sets × all 4 continuation arms completed to 1000 updates;
- exact-resume smoke: PASS, uninterrupted/resumed model hash identical and telemetry identical;
- runtime calibration: PASS under the V3.1 240-minute wall;
- measured full run wall time: 8509.69 s = 141.83 min.

The archived original 175-minute preexecution gate failure is preserved in `PREEXECUTION_FAILURE_ARCHIVE/RUNTIME_GATE_175M.json` and is not a scientific continuation outcome.

## 2. Parent qualification

Both prospective SKILL_A parents entered continuation robustly qualified on SEALED:

| parent seed | confirmation step | SEALED canonical | order | query-order |
|---|---:|---:|---:|---:|
| 31801 | 400 | 1.00 | 1.00 | 1.00 |
| 31902 | 600 | 1.00 | 1.00 | 1.00 |

This confirms a real old capability existed before the matched continuation arms.

## 3. Official R3 decision

The frozen decision function emitted:

`CONTROLLER_NOT_SUPPORTED`

with:

- interference = true;
- `PLASTIC_HIGH` old-skill failure in 4/4 matched sets;
- `PLASTIC_HIGH` mean old-skill robustness area ≈ 0.02988 (≈0.02967 after removing identical duplicated resume telemetry; see audit caveat below);
- `STATIC_REPLAY_1OF64` mean old-skill robustness area ≈ 0.90067;
- both Guardian arms old-skill any-failure risk = 1.0;
- no Guardian arm met the preregistered promotion rule;
- `GUARDIAN_POLICY_SPEC.json` status = `NOT_PROMOTED`.

The official verdict must remain the frozen preregistered verdict.

## 4. What the continuation trajectories actually show

### 4.1 Unprotected high-plasticity continuation destroys SKILL_A

Across the four matched `PLASTIC_HIGH` arms, final SEALED robust-min scores were approximately:

`0.0067, 0.0000, 0.0133, 0.0000`

Mean final robust-min ≈ **0.005**.

Thus the real-text + new-skill continuation condition creates extremely strong interference with the previously acquired SKILL_A capability.

### 4.2 Sparse replay strongly protects/reconstructs SKILL_A

`STATIC_REPLAY_1OF64` final SEALED robust-min scores:

`0.8133, 0.9733, 1.0000, 0.7400`

Mean final robust-min ≈ **0.8817**.

`GUARDIAN_REPLAY` final SEALED robust-min scores:

`0.9067, 0.9667, 0.9800, 1.0000`

Mean final robust-min ≈ **0.9633**.

`GUARDIAN_HYBRID` produced the same final scores as `GUARDIAN_REPLAY`.

All Guardian runs suffered an initial formal failure by the first 100-update control check, then the controller moved from `PLASTIC` to `REPLAY32`. The old capability subsequently recovered strongly. Some runs had later transient failures and replay escalation/de-escalation dynamics, but all four Guardian runs ended SEALED-qualified.

This is useful **secondary/exploratory evidence of reactive recovery**, not a preregistered successful Guardian result, because the primary any-failure risk criterion counts the initial failure before the reactive controller can act.

### 4.3 Emergency CAP16X was never exercised

`capped_steps = 0` in all four `GUARDIAN_HYBRID` arms.

The final model hashes and counters of `GUARDIAN_HYBRID` equal the corresponding `GUARDIAN_REPLAY` arms in all four matched sets. Therefore R3 provides **no comparative evidence that CAP16X adds or does not add value**. The state machine never encountered the prospective trigger needed to invoke it.

## 5. Critical blocker: SKILL_B never acquired

No arm, including `PLASTIC_HIGH`, achieved SKILL_B qualification or confirmation.

Final SEALED robust-min scores were clustered around ~0.31–0.34 across all arms. Peak scores also stayed in approximately the same range. `skill_b_confirmation_step` is null everywhere.

This is the largest interpretability limitation of R3. The central systems question was whether a controller can preserve an old capability **while still acquiring a new one**. Because the new skill failed to form even in the unprotected reference, the experiment never created the required plasticity-success reference.

The frozen decision taxonomy maps this situation to `CONTROLLER_NOT_SUPPORTED`; scientifically, the narrower interpretation is:

**NOT_DEMONSTRATED / NEW-SKILL-FORMATION BOTTLENECK.**

Do not claim that the Guardian slows SKILL_B acquisition, because SKILL_B did not acquire in any arm.

### Exposure-design diagnosis

The failure is consistent with an underpowered SKILL_B dose. Parent SKILL_A acquisition used 64 binding examples per optimizer step and required 400 / 600 steps in the two parents. R3 gives SKILL_B only 4 examples per mixed update for 1000 updates = 4000 SKILL_B examples total. Matching even 400 full binding steps by raw binding-example count would require 25,600 examples, about **6.4×** the R3 SKILL_B exposure; matching 600 steps would require about **9.6×**.

This is not an exact gradient-equivalence theorem because mixed optimization is nonlinear and includes real-text gradients, but it is a strong design-level explanation for why the new capability never had a realistic formation budget.

## 6. Real-text metric caveat

The ARK-018 `SCIENCE_ONLY` source checkpoints had SEALED science NLL around 4.2607 (seed 31801) and 4.293 (seed 31902). After R3's binding-only SKILL_A parent acquisition, the continuation step-0 science NLL was approximately 14.4511 and 17.7845 respectively.

So SKILL_A parent formation had already severely damaged the science model before the four R3 continuation arms began. During continuation, the 28/32 real-text slots rapidly reacquired science modeling, ending around NLL 4.11–4.15.

The frozen R3 comparison against matched `PLASTIC_HIGH` remains internally useful, but R3 does **not** demonstrate preservation of an intact real-text model while acquiring SKILL_A and SKILL_B. A future version should require a science-NLL parent gate before continuation or acquire SKILL_A with concurrent real-text support.

## 7. Resume-telemetry audit caveat

Two arm trajectories contain duplicated, byte-identical evaluation rows caused by resume bookkeeping:

- `p31801_b319001/PLASTIC_HIGH`: updates 700, 800, 900 duplicated;
- `p31902_b319002/GUARDIAN_REPLAY`: update 400 duplicated.

The duplicated rows are identical to their originals. Deduplicating them changes `PLASTIC_HIGH` mean area from ≈0.0298846 to ≈0.0296667 and `GUARDIAN_REPLAY` mean area from ≈0.84808 to ≈0.84667. It does **not** change any qualitative conclusion or the frozen verdict, but the next runner should truncate/merge trajectory telemetry by update when resuming.

## 8. Claim ledger

**DEMONSTRATED in this real-text proxy setup**

- unprotected HIGH mixed continuation catastrophically destroys previously qualified SKILL_A across 4/4 matched sets;
- static 1/64 support produces a very large retention advantage over unprotected HIGH in trajectory area and final capability;
- the reactive replay Guardian can recover from severe old-skill collapse and ends qualified in all four matched sets;
- the experiment executed reproducibly enough to pass exact-resume smoke and all receipt/manifest integrity checks.

**SUPPORTED / EXPLORATORY**

- dynamic replay escalation can reconstruct old capability after an observed collapse;
- the first 100-update observation interval is too coarse to prevent the initial collapse in this interference regime;
- SKILL_B exposure was likely too weak relative to known binding-acquisition requirements.

**NOT DEMONSTRATED**

- a preregistered successful Guardian;
- preservation without any old-skill failure;
- simultaneous old-skill retention + robust new-skill acquisition;
- value of CAP16X inside the hierarchical Guardian;
- broad continual learning;
- production scheduler readiness;
- PRE500M / 500M readiness;
- AGI.

## 9. Highest-value next experiment

Do **not** simply rerun V3 with another seed. R3 V4 should repair the causal design before replication:

1. establish a viable SKILL_B acquisition dose in a matched `PLASTIC_HIGH` pilot **before** freezing the Guardian comparison;
2. preserve a meaningful fraction of real-text competence during SKILL_A parent acquisition, or require a science-NLL parent gate;
3. shorten CONTROL observation spacing enough that a reactive policy has a chance to intervene before formal failure;
4. retain static sparse replay, dynamic replay and emergency-cap arms;
5. define separate preregistered outcomes for **prevention**, **recovery**, and **new-skill acquisition**, so an initial reactive failure is not conflated with inability to recover;
6. fix resume trajectory deduplication and keep exact checkpoint identity.

Until that repair, ARK-019 V3.1 should be treated as a strong interference/recovery experiment with an underpowered new-skill formation channel, not as a validated continual-cognition controller.
