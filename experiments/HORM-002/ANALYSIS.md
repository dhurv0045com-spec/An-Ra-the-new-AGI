# HORM-002 — Completed local experiment

## Verdict

**COMPETENCE_GATE_FAILED.** The single-seed calibration passed at modulus 23 / 12,000 updates, but the second causal OFF seed failed the required 50% endpoint. Per the frozen decision precedence, all treatment differences are descriptive only. This is neither a supported feedback-effect claim nor a finding of no effect.

## Actual local execution

Nine sequential CPU lanes, two threads, batch 16, real V5 tiny geometry (81,024 baseline parameters; wrapped arms add 14). No GPU allocation or production-model construction. 102,000 total optimizer updates; summed lane time 1,522 seconds (25.37 minutes); longest lane 226.65 seconds, below the 600-second cap. No score-driven retries. Calibration restarted each ladder entry from the same seed; causal lanes restarted from scratch. Nothing committed or pushed.

Preflight: 7.02 GiB free RAM, CPU load 24%, RTX 4050 6 GiB idle. Final nvidia-smi: 0 MiB allocated, 0% utilization.

## Calibration

| Modulus | Updates | Held-out exact + EOS | Train-subset exact | Gate |
|---|---:|---:|---:|---|
| 97 | 12,000 | 3/512 (0.586%) | 1.172% | fail |
| 23 | 6,000 | 1/105 (0.952%) | 5.189% | fail |
| 23 | 12,000 | 105/105 (100%) | 99.764% | pass |

This establishes tiny-task competence for one initialization, not reliability across seeds. The 6,000-update lane and first 6,000 updates of the 12,000-update lane are repeated computation, not independent evidence.

## Registered matched lanes (modulus 23, 12,000 updates)

| Seed | Arm | Held-out exact + EOS | Train exact | Trailing-window AUC | First eval >=10% |
|---|---|---:|---:|---:|---:|
| 424242 | OFF | 105/105 (100%) | 99.764% | 0.58242 | 7,250 |
| 424242 | CONST | 66/105 (62.857%) | 72.406% | 0.27546 | 9,250 |
| 424242 | ON | 66/105 (62.857%) | 91.745% | 0.22015 | 9,250 |
| 424243 | OFF | 5/105 (4.762%) | 57.547% | 0.00934 | never |
| 424243 | CONST | 38/105 (36.190%) | 73.113% | 0.22344 | 8,000 |
| 424243 | ON | 58/105 (55.238%) | 85.377% | 0.23919 | 8,000 |

ON minus CONST endpoint: 0.00000 and +0.19048. CONST minus OFF: -0.37143 and +0.31429. Both ON lanes exercised appraisal: seed 424242 had 838 success events and 1 surprise (first event update 1607); seed 424243 had 1109 successes and no surprise (first event 8784). CONST had no decay, appraisal, or state drift. OFF had no hormone state.

## Matched-curve audit (exploratory)

At all 48 matched evaluation updates, ON/CONST wins-ties-losses were 7/21/20 for seed 424242 and 9/35/4 for seed 424243. Mean pointwise differences were -0.02956 and +0.01052. Trailing-window AUC differences were -0.05531 and +0.01575. Equal final score on seed 424242 did NOT mean identical predictions: 64 of 105 generated sequences differed; the second seed differed on 45 sequences. First >=10% acquisition was unchanged by ON relative to CONST in both seeds.

These curves do not show a consistent acceleration or repeatable benefit. Better training accuracy is not evidence of better held-out generalization. The +20 correct answers on one seed is a real descriptive observation, not a replacement endpoint or a license to discard the other seed.

## Verification and evaluator audit

- Focused test invocation: 57 passed, 1 deselected in 10.88s (`-k 'not full'`; explicitly report this filter rather than implying the entire suite ran).
- Real HORM-001 runner trigger test injected known synthetic losses only into a temporary engineering test: expected adrenaline rise and cortisol carryover passed. Actual HORM-001 loss ratios never qualified; no alteration of old frozen sources/results.
- Independent `verify_receipts.py`: PASS for nine lane receipts, all 20 artifact hashes, all 26 source and 26 protected file hashes, full probe coverage/uniqueness, independent exact+EOS rescoring, 102,000 update trace records, finite losses/gradients, 96 eligible targets per update, event counts, success thresholds, frozen CONST state, seed pairing and initial-core/data hashes, calibration order and verdict precedence.
- All nine checkpoints serialized and were reloaded by the runner with exact logits and generated predictions; independent audit confirms checkpoint hashes and those receipts, not a second model execution.
- Protected files were byte-identical to HEAD. `git diff --check` passed. No source changed during measurement. The read-only verifier was authored separately and is not a frozen experimental input.
- RESULT.json SHA-256: `d10f17cdaac7db3b327b31606be758ced96c4a6fd55dcaacc733b3f3345d663f`.

## Limits and next defensible question

The same calibration probe selected the geometry/budget, so all scores are selection-conditioned exploratory measurements. Only two seeds were run. ON versus CONST changes both state decay and appraisal; appraisal-only attribution would require a matched decay-only control. Training-side reward is measured but hand-designed; hormone names are analogies, not biological or subjective states. Final checkpoints are model-only reload artifacts, not full optimizer/state-history resume checkpoints.

The important blocker is now seed robustness: why did identical tiny baseline training generalize at 100% for one initialization but 4.76% for the other? Existing checkpoints and traces support a bounded read-only loss/logit/representation audit before additional training. No further run, scale-up, frontier reopening, or capability/cognition claim is justified by this result alone.
