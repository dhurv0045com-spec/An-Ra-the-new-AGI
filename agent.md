# AGENT.md — Citadel handover (machine-readable status for the operator's agent)

> Convention (binding): rewritten at the END of every Citadel work cycle,
> committed to the `citadel` branch ONLY, then `git push origin citadel`.
> Other branches are read-only audit inputs — never modified, never pushed.
> A CPU/CUDA run is NEVER a TPU result. No fabricated device results.
> Preregistration and results never share a commit. Download ceiling <10 GB;
> this cycle: 0 bytes.
>
> STANDING OPERATOR POLICY: prefer batched preregistered experiment suites
> over repeated round-trips. Tiny experiments are automated internal
> gates/preflight only, never repeated operator-facing work.
> Citadel validates Cymek; it does not replace Cymek.

## STATUS

CURRENT_CITADEL_SHA: see `git log origin/citadel -1` (post-reload closure
hotfix + final_model_ready recovery cycle)
AUDITED_CYMEK_SHA: `28bf57a0d299a2c13a99fe0046616c00a1b8530c`
RUNTIME_PIN_SHA: `28bf57a0d299a2c13a99fe0046616c00a1b8530c` (== live HEAD)

CYMEK_ALIGNMENT = PASS
DEVELOPMENT_CERTIFICATE = PASS (regenerated after this cycle's executable
  changes; code_sha matches the pushed tree)
LOCAL_TESTS = 86 PASS / 4 torch-optional skips across 7 files
  (t1d 38; t1c 10; t1_canary 6; notebooks 2; bootstrap 6; cymek_checkpoint 7;
   one_shot 11 + 4 torch skips — ALL four PASS under the torch environment)
RUN_ARM_FEEDER_STATE_RESTORE = PASS (mid-arm resume restores the exact
  data-plane state before any resumed consumption)
POST_RELOAD_SELF_EVALUATION = PASS (explicit-model evaluators; the
  trained-self probe can no longer reference a deleted model)
STALE_MODEL_CLOSURE_REGRESSION = PASS (source + callable audit: gen/gen_text/
  final_evals take active_model explicitly; every post-`del model` call uses
  model2/model_v)
FINAL_MODEL_READY_RECOVERY = PASS (artifact written after the FINAL
  checkpoint, BEFORE any final TEST/teacher/self/diagnostic evaluation;
  hash-verified load; recovery reruns evaluation+finalization with NO
  retraining — proven by forbidding _train_updates_packed during recovery)
75_PERCENT_MID_RECOVERY = PASS (A/B-style simulation: crash at the last
  training block -> resume from mid -> only the remaining block runs ->
  data schedule byte-identical to an uninterrupted run)

## REAL TPU ATTEMPT RECORD (2026-09-06, run from one-shot notebook)

```text
BOOTSTRAP: PASS        PREFLIGHT: PASS        CANARY: PASS        DATA: PASS
calibration: selected (512, 64) @ ~6990 tok/s on TPU (torch/XLA 2.9.0)
Arm A: trained 61/244 -> 244/244 (100% reached), then
       IMPLEMENTATION_FAILURE in the post-reload self evaluation:
       "cannot access free variable 'model' where it is not associated
       with a value in enclosing scope"
Arm B: 100% training reached, same IMPLEMENTATION_FAILURE
C-F:   not completed
Cause: gen_text closure captured `model`, which run_arm had deleted after
       the final checkpoint (post-reload lifetime bug). DETERMINISTIC.
TPU itself: NOT implicated. The one-shot machinery (preflight, canary,
       calibration, data, failure-bundle export) worked exactly as designed.
A/B:   scientifically INVALID (no completed receipt). The failed TEST
       observations are recorded INVALID_IMPLEMENTATION_RUN and are not used
       for selection or tuning; the recovery rerun re-executes the frozen
       TEST deterministically on the same final state.
Recovery: if the Colab disk is still alive, A/B resume from their valid 75%
       mid checkpoints (mid state restores model+optimizer+feeder+cursor
       state), run the last 25%, and finalize. On a fresh runtime the arms
       retrain — the orphan checkpoints were not downloadable.
Fix:   gen/gen_text/final_evals take the model EXPLICITLY (no captured
       model); final_model_ready boundary written after the final checkpoint
       and BEFORE any final evaluation; recovery hierarchy completed receipt
       -> prefinal -> final_model_ready -> mid -> fresh; failure bundles now
       carry mid/final-model sidecars and checkpoints.
```

## T1D RESULT - FIRST COMPLETED SESSION (2026-09-06, real TPU)

```text
One-shot run: BOOTSTRAP/PREFLIGHT/CANARY/DATA PASS -> arms A-F all executed
-> PRE50M phase ran -> bundle exported and verified. One-shot machinery
worked end-to-end on a fresh Colab TPU (calibration (512,64) @ ~6978 tok/s).

RESULT: all six arms SCIENTIFIC_FAIL; cross-arm INCONCLUSIVE (no contrast
rule fired - correct, nothing lifted anywhere).
  loss: 10.04 -> 0.77-1.52 (A-E), 10.06 -> 5.19 (F, 2M budget)
  TEST exact: 0-6.6% on every tier, every arm - below the 22.5% strongest
  heuristic null (copy-first-operand)
  TRAIN exact: 0-11% - no memorization either
  dev curves: flat ~0-5.5% across all four checkpoints in every arm
  SELF-KNOWLEDGE (Arm F): probe 0.0 trained and untrained - self knowledge
  NOT acquired at 2M tokens (loss only reached 5.19)
Meaning (preregistered interpretation): answer-CE arithmetic learning does
not lift off at 2-8M tokens for 3.7-7.4M models under ANY of the five
variations - the T1/T1C null is now replicated with curriculum, teacher,
scale, output-space, and self-knowledge contrasts, all instrumented,
reload-verified, and receipted. Verdict INCONCLUSIVE per the frozen rules;
the suite ELIMINATES curriculum/teacher/scale/output-space/self-knowledge
at these budgets as sufficient for lift-off.
PRE50M: FAILED on a deterministic smoke bug - the smoke state's token
budget funded `updates` but the resume-proof publishes updates+1 (Cymek:
"a completed run cannot advance"). FIXED this cycle
(token_budget=(updates+1)*tokens_per_update; regression against the real
Cymek TrainingState contract). Second bug fixed: summarize_session labeled
PRE50M PASS from file existence alone - status now derives from the
decision's ready flag/blockers/failure status.
```

## T0 / T1 / T1B / T1C (history)## T0 / T1 / T1B / T1C (history)

```text
T0: PASS (unchanged, still applicable)
T1: FAIL (loss-learned, exact-flat; historical result unchanged)
T1B: SUPERSEDED_BY_T1C (preserved, unexecuted)
T1C: EXECUTED — 4 arms FAIL, cross-arm INCONCLUSIVE (mode collapse, no
memorization; objective/data/2.3x scale moved nothing at 4M)
```

## T1D + PRE50M (preregistered, pending execution)

Arms A/B/C/D/E on the tiered ladder (~130 MB); PRE50M smoke on SCALE2 7.4M
through production transactions; NEXT_50M_DECISION machine gate.
"50M checkpoint" = 50M training TOKENS (Cymek training_spec, re-verified);
no 50M-param spec exists. Notebook: notebooks/citadel_colab_t1d.ipynb.

## DOWNLOADS

```text
ITEM | SOURCE/PURPOSE | BYTES | CUMULATIVE BYTES
(none — no pip installs, no datasets, no checkpoints, no artifacts)
TOTAL_DOWNLOADED_GB = 0.0
```

## QUESTIONS_FOR_OPERATOR

```text
NONE
```

## BIGGEST BLOCKER

NONE REMAINING THAT LOCAL VALIDATION CAN ADDRESS — next is the operator run.

## NEXT ACTION

Run `notebooks/citadel_colab_t1d.ipynb` once from Cell 0 through F in a
fresh/restarted Colab TPU session and return `CITADEL_T1D_RESULTS.zip`.

## CYMEK_REQUIRED_CHANGE

```text
NONE FOUND this cycle (checkpoint, data-interface, and contract surfaces
checked against current Cymek; Citadel adapts at its own layer only)
```

