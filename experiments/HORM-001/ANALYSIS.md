# HORM-001 — Final analysis

## Verdict

**NO_MEASURABLE_EFFECT_AT_THIS_SCALE** under the registered endpoint thresholds. The bounded implementation and four-lane experiment are complete. Neither arithmetic acquisition, useful feedback adaptation, nor cognition was demonstrated. This is not a claim of statistical equivalence or a general rejection of hormonal modulation.

## Measured result

Each lane ran 1,500 updates, batch 16, CPU only. Every final result is correct answer plus EOS on **3/512 = 0.5859375%** of disjoint held-out pairs.

| Seed | OFF exact | ON exact | ON minus OFF | OFF formation AUC | ON formation AUC | ON appraisal events |
|---|---:|---:|---:|---:|---:|---:|
| 424242 | 3/512 | 3/512 | 0.0000 | 0.00953311 | 0.00953311 | 0 |
| 424243 | 3/512 | 3/512 | 0.0000 | 0.01018415 | 0.00990513 | 0 |

All four final training-subset scores: 2/512 = 0.390625%. No lane reached the predefined 10% acquisition threshold. Four measured lanes took **97.7384 seconds summed**, excluding test/preflight overhead. OFF has 81,024 parameters; ON has 81,038. This used randomly initialized tiny real V5 architecture, not a pretrained SequenceGPT checkpoint or the production-size sibling.

## Audit of the negative result

Identical endpoint scores warranted checking the evaluator rather than accepting the number blindly:

- Independently rescored all stored generated token sequences against modular arithmetic and EOS. All counts match lane receipts.
- **Every final prediction in every lane is `[26,100]`: value 23 followed by EOS.** All 512 rows have a valid EOS; only three held-out answers equal 23. Thus the endpoint failure is not an EOS-scoring mismatch. It is constant-answer behavior in the actual model outputs.
- Evaluator tests check correct scripted generation without gold in the prompt, wrong answers, missing EOS, and prediction coverage. The measured final checkpoints reproduce their logits and full held-out generations after reload.
- Initial shared-core hashes, dataset hashes, and initial accuracies match within each seed pair. Training losses and learned temperature values differ, so identical final scores do not mean the ON arm was bypassed.
- Final losses are 2.28850/2.29077 (seed 424242 OFF/ON) and 2.29122/2.28684 (seed 424243 OFF/ON), down from about 4.70–4.73. The objective averages six targets: two operands, two separators, answer, EOS. A descriptive reference with three uniform 97-way targets and three perfectly predicted formatting targets is `3*ln(97)/6 = 2.28735549`. Proximity to that reference, together with constant predictions, is consistent with learning formatting without acquiring addition; it is not a measured per-position entropy decomposition.
- Pre-clipping gradient norms reached 15.20/66.43 and 60.38/30.09 in OFF/ON by seed. Clipping at 1.0 was active. All checked losses, gradient norms, parameters, and evaluation logits remained finite. No claim of universally stable unbounded training follows.

## The mechanism was live, but appraisal was not exercised

Final ON log-temperatures:
- seed 424242: `[-0.160614, -0.082238]`
- seed 424243: `[-0.016950, +0.182307]`

These are finite and within the configured +/-1.5 bound. Exact no-op initialization, live projection gradients, hormone-state effects, and checkpoint snapshot behavior are tested separately.

**Zero loss spikes triggered in either ON lane.** The training experiment therefore exercised learning with nonzero baseline hormone state, not outcome-driven stress adaptation. Its negative endpoint cannot settle whether useful feedback signals would help an already competent model. OFF/ON also differs by extra learnable parameters, so even a positive result would not isolate appraisal causality without a constant-state control. No such extra arm was run after seeing results.

## Engineering changes and corrections

- Experimental `v5_identity` state, projection, explicit wrapper, and config; separate composite sibling specification and accurate parameter receipt.
- Constructor and appraisal finite/bounds checks, integer state steps, bounded decay history, serotonin stress damping, and numeric-state normalization.
- Forward reads the registered projection and live state; checkpoint recomputation uses the captured temperature. Configured temperature bounds reach execution.
- Checkpoint metadata rejects different spec/config identity before tensor copying in the wrapper's direct loader. Checkpoints from this run were actually serialized and reloaded with exact results.
- `pyproject.toml` includes `v5_identity` in package discovery.
- Earlier draft runner incorrectly reintroduced held-out pairs into training and did not implement the described answer/stop evaluator. Tests caught those defects before measured execution. Draft code/results are not evidence. Final protocol revision 3 supersedes earlier descriptions, including text rendering, split semantics, state ordering, and decision partitions. A two-update smoke test preceded the final freeze; its tiny outputs are excluded from scientific results.

## Verification and artifacts

Pre-run focused command: 49 passed, 1 deliberately deselected (full-size model inventory test). This is not a full-repository test claim.

- `PLAN.md`: final protocol, frozen before measured lanes.
- `results/PRERUN.json`: source/test/plan hashes, protected-file hashes, runtime identity.
- `results/lane_{HAL_OFF,HAL_ON}_{424242,424243}.json`: complete training traces, evaluation curves, final raw predictions, identities and timing.
- Corresponding `.pt`: final checkpoint per lane.
- `results/RESULT.json`: aggregate verdict and SHA-256 of preceding artifacts.
- `verify_receipts.py`: separate read-only audit of hashes, pairs, raw scores, traces, temperature bounds and checkpoint receipts.

The independent verifier returned PASS. All **26 protected files** are hash-identical before/after the measured run and have no tracked diff from HEAD: includes canonical ModelSpec, production V5 model files, tracked blueprint files, and `artifacts/v5/launch_readiness.json`. All 24 recorded source/test/plan hashes also remained unchanged during measured execution. No production training, GPU training, frontier launch, commit, or push was performed.

## Remaining limits and next scientific step

This closes the small HORM-001 engineering experiment, not the purpose of the entire AGI project and not a claim of perfection.

- Seven projection columns are stored; the three state-only columns are masked from computation/gradients, not physically absent. Parameter receipts count all seven. This differs from the earlier proposed column-removal design and is explicit in the final protocol.
- `VerifiedOutcome` validates structure/provenance text, not truth: the caller remains responsible for supplying genuinely measured events. Forward does not autonomously verify events or infer emotions.
- Appraisal state history lives outside the module; the final checkpoint preserves model tensors and current hormone levels, not full optimizer/training-resume state. This is a final-model artifact, not a resumable production trainer.
- Tiny toy tokens, two seeds, one schedule, and one operation do not establish general reasoning. No large-scale integration/performance claim is made.

The next justified experiment would first establish task competence and a verified, non-evaluation feedback event that actually fires, then preregister a matched constant-state control. That is future work, not a hidden retry of this negative run. No automatic scale-up is warranted by these results.
