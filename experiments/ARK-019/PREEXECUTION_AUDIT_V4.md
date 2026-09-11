# ARK-019 V4 — PREEXECUTION AUDIT

**Audit state:** STATIC REVIEW COMPLETE / OPERATOR CUDA PREFLIGHT STILL REQUIRED  
**Scientific result:** NOT EXECUTED  
**Frozen scientific executable:** `8e858c614d764335100dfd41dda6f8e0d0c877a7`

## Why V4 is the correct successor to V3.1

ARK-019 V3.1 completed cleanly but did not create the causal situation required for a continual-learning controller claim. The audited bundle showed: (1) `PLASTIC_HIGH` destroyed SKILL_A in 4/4 matched sets; (2) sparse replay and the reactive replay Guardian strongly recovered/protected SKILL_A; (3) SKILL_B never acquired in any arm, including `PLASTIC_HIGH`; (4) SKILL_A parent construction had already damaged science modeling before the matched comparison; and (5) the 100-update controller observation interval allowed collapse before the first decision.

V4 repairs those exact blockers rather than merely adding more seeds to the same design.

## Static design findings

1. **Science-preserving parent gate.** SKILL_A parent construction now mixes 16 SKILL_A slots with 16 real-text slots per update. A parent must satisfy three consecutive robust SKILL_A CONTROL qualifications while science CONTROL NLL stays within +15% of its own ARK-018 source checkpoint, then also qualify on a separate validation split.
2. **Selection/evaluation namespaces are separated.** Each skill has train, parent/dose-control, main-control, validation and sealed partitions. Main-control and SEALED are not used to build parents or select the SKILL_B dose; SEALED never controls training.
3. **New-skill viability is established before the Guardian comparison.** The smallest dose among 8/12/16 SKILL_B slots that qualifies in both parent pilots by the frozen deadline and passes validation is selected. If none passes, V4 stops before the main matched comparison with `INCONCLUSIVE_NO_VIABLE_SKILL_B_DOSE`.
4. **Main exposure is matched.** The selected SKILL_B dose is frozen across all six main arms. Replay displaces real-text slots only and never reduces SKILL_B exposure.
5. **Both R2 levers receive explicit references.** Main arms include `STATIC_REPLAY_1OF64`, stronger `STATIC_REPLAY_1OF32`, and `STATIC_CAP16X`, in addition to `PLASTIC_HIGH`, `GUARDIAN_REPLAY`, and `GUARDIAN_HYBRID`.
6. **Controller latency is reduced 4×.** SKILL_A main-control is checked every 25 updates rather than every 100. A single sub-.95 robust-min warning while PLASTIC activates 1/64 support; formal failure activates 1/32 support immediately at the next observation.
7. **Prevention and recovery are no longer conflated.** V4 separately records whether failure was prevented and, if failure occurred, whether four consecutive healthy observations were regained within 200 updates.
8. **A failed new-skill reference cannot be mislabelled controller failure.** If `PLASTIC_HIGH` does not acquire and finish SEALED-qualified SKILL_B in at least 3/4 matched sets, the frozen main verdict is `INCONCLUSIVE_MAIN_B_FORMATION_INSTABILITY`.
9. **Science cost remains matched and prospective.** A Guardian candidate must remain within +5% final science SEALED NLL of its matched `PLASTIC_HIGH` reference in every matched set.
10. **Resume bookkeeping is repaired.** Control and measurement trajectories are deduplicated by update and truncated to the durable checkpoint step on resume, preventing V3's duplicate-evaluation rows from affecting area metrics.
11. **Multi-session execution is intentional.** The campaign may span multiple T4 sessions. Runtime calibration estimates session count only and is forbidden from changing seeds, arms, horizon, dose candidates, thresholds, or evaluation frequency.
12. **Exact resume is executable.** Before the main comparison, a 10-update smoke requires equality of model hash, optimizer hash, scaler state and update telemetry between uninterrupted 10 updates and 5 + save/load + 5.

## Frozen main design

- parent seeds: `31801`, `31902`;
- pilot order seeds: `419001`, `419002`;
- main order seeds: `429001`, `429002`;
- matched main sets: 4;
- main arms: 6;
- main horizon: 2000 updates/arm;
- SKILL_A controller interval: 25 updates;
- SKILL_B control interval: 50 updates;
- SEALED/science measurement interval: 100 updates;
- checkpoint interval: 200 updates;
- session wall: 225 minutes with 10-minute packaging reserve.

## Code and contract review

Frozen executable blobs:

- `experiments/ARK-019/ark019_v4_core.py`: `18133fd31e16de20e633ca86f5b88cb817b97371`
- `experiments/ARK-019/run_ark019_v4.py`: `4adb1558e92270f596df8bccdb0b11478b9097d8`
- `tests/test_ark019_v4.py`: `4a972076f187035dfaae1353e3d77c65443afc48`
- inherited V3 runner: `ef41096069714668b3b1e5c0e165904908ebc885`
- inherited ARK-018 common module: `05b1c6a7832749740420b5b540c3214ef4c84492`
- inherited ARK-018 binding module: `7492c697eccc5d87528094acf1b2e6164e47b1e1`

The dedicated `ARK-019 V4 contracts` GitHub Actions workflow completed successfully on the frozen executable commit. It compiled the V4 core/runner and passed the V4 pure contract suite.

An unrelated broad `ESOES research contracts` workflow failed on the same PR because that workflow invokes repository-wide tests without installing dependencies such as NumPy/PyTorch. Its log also fails pre-existing V3/V5 imports for the same missing-package reason. This is **not** counted as V4 scientific or contract evidence, and it is not being hidden as green CI. The dedicated V4 gate is the relevant static contract gate and is green.

## Remaining operator gates

The Colab launcher must still verify at runtime:

- live readiness binds to this frozen scientific commit;
- every frozen Git blob matches;
- V4 and inherited modules `py_compile`;
- V4 pure contracts pass in the operator environment;
- CUDA/T4 is present;
- the ARK-018 prepared receipt, tokenizer, caches and both `SCIENCE_ONLY` checkpoints exist;
- both V4 science-preserving parents qualify;
- a viable SKILL_B dose is prospectively selected or the experiment fails closed before main comparison;
- exact-resume CUDA smoke passes;
- runtime calibration records expected session count without modifying protocol.

## Audit verdict

**IMPLEMENTED + STATIC CONTRACT PASS / READY FOR OPERATOR CUDA PREFLIGHT / NOT EXECUTED.**

No successful Guardian, broad continual learning, production scheduler, PRE500M, 500M or AGI claim follows from this readiness state.
