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

CURRENT_CITADEL_SHA: see `git log origin/citadel -1` (handover commit sits on
the one-shot hardening cycle)
AUDITED_CYMEK_SHA: `28bf57a0d299a2c13a99fe0046616c00a1b8530c`
RUNTIME_PIN_SHA: `28bf57a0d299a2c13a99fe0046616c00a1b8530c` (== live origin/cymek
HEAD, re-verified this cycle)

CYMEK_ALIGNMENT = PASS
RUNTIME_BOOTSTRAP_FRESH_CLONE = PASS (Cell 0 prints + enforces
EXPECTED_CYMEK_SHA; DEVELOPMENT_CERTIFICATION.json is checked at BOOTSTRAP
and any code newer than certification fails closed BEFORE the TPU)
DEVELOPMENT_CERTIFICATE = PASS (7/7 test files, committed receipt,
docs/citadel/experiments/T1D/DEVELOPMENT_CERTIFICATION.json)
LOCAL_TESTS = 78 PASS / 2 torch-optional skips across 7 files
  (t1d 38; t1c 10; t1_canary 6; notebooks 2; bootstrap 6; cymek_checkpoint 7;
   one_shot 9 + 2 torch skips: run_arm feeder-restore wiring across
   flat/curriculum/teacher/self/masked AND continuous-vs-resumed identity —
   both PASS under the torch environment; certificate is the record)
RUN_ARM_FEEDER_STATE_RESTORE = PASS (mid-arm resume restores the EXACT
  data-plane state — tier cursors, teacher cursor, self cursor, carry
  buffer, drawn/placed counters, placed tokens — before any resumed
  consumption; resumed feed byte-identical to uninterrupted execution)
PORTABILITY_SCAN = PASS (no machine-local paths in executable code)
CLEAN_BOOTSTRAP_SIM = PASS (one-shot emulator: fresh session COMPLETE)
STALE_BOOTSTRAP_SIM = PASS (rerun resumes every completed arm — zero
  recomputation; PRE50M failure and arm failure isolated; preflight failure
  exports CITADEL_T1D_FAILURE.zip before any training)
ONE_SHOT_EMULATOR_FRESH = PASS
ONE_SHOT_EMULATOR_RESUME = PASS (phase receipts + plan_sha identity)
DISCONNECT_RECOVERY = PASS (mid-arm checkpoints at 25/50/75% with
  model+optimizer+feeder state; CPU resume-identity proven: N continuous ==
  N/2 + restore + N/2 in tokens, feeder state, and loss trajectory)
TPU_CANARY_PLAN = PASS (canary phase runs the REAL code paths on TPU before
  any arm: MID update; a REAL SCALE2 update - finite loss, backward, verified
  parameter mutation, save/reload with output-identity generation; teacher +
  masked + self feeder paths; checkpoint/reload; producer->finalizer bridge;
  failure aborts pre-arms with a bundle). Calibration shape acceptance is
  now THREE-WAY: MID ordinary + SCALE2 + masked-MID (real Arm E path with
  valid_alphabet_ids/allow-mask/causal loss/update) at the selected batch -
  the fastest candidate failing any variant is rejected in place and the
  next passing shape is selected.
SCHEMA_BOUNDARIES = PASS (terminal arm validator at finalize + verify_bundle
  + session boundary; producer->finalizer probe in every preflight)
FAILURE_BUNDLE = PASS (any phase failure exports environment + preflight +
  phase receipts + traceback + arm receipts; the notebook auto-downloads it)
NOTEBOOK_TORTURE = PASS (one-shot notebook: CELL 0 bootstrap + CELL 1
  RUN EVERYTHING; auto-downloads RESULTS or FAILURE zip)

T1D = READY_FOR_ONE_SHOT_TPU (arms A-F)
PRE50M = READY_FOR_ONE_SHOT_TPU

## OPERATOR'S SELF-KNOWLEDGE HYPOTHESIS (preregistered this cycle)

SELF_KNOWLEDGE_AMENDMENT.md adds Arm F (SELF): identical to the curriculum
arm but every 7th row carries self-knowledge (identity, body, infrastructure,
purpose, motivation, abilities, limits, mission) rendered in the frozen row
grammar. Probes: 96 held-out self-QA rows with disjoint question forms, text
exact-match scoring (never the integer normalizer), untrained baseline +
most-common null on identical rows. Machine rules: SELF_KNOWLEDGE_ACQUIRED
(F probe LCB > untrained + 0.10 AND > null + 0.10) and SELF_PROBE_LEAKAGE
(any non-self arm passing the same bar - probe design is broken, no claim).
The operator's "child that learns exponentially from little data" framing is
recorded as the standing motivation; the causal self-knowledge-accelerates-
learning question needs matched budgets and belongs to main training - this
run tests ACQUISITION and interference descriptively. F budget 2M (adds
~10-15 min, inside the <2 TPU-h ceiling).

## DATA ACCOUNTING (§23 - measured, not guessed)

DATA_UNIQUE_ROWS = 6,416,000 (tiered TRAIN_N sum)
DATA_UNIQUE_BYTES_EST = ~96.8 MB (mean row 15.09 chars, measured)
SCHEDULED_ROWS_ALL_ARMS_EST = ~2,252,486 (all six frozen budgets)
CONSUMABLE_UNIQUE_FRACTION_EST = 0.351
REPLAY_EXPECTED = false at pool level; per-tier drill replay (T0-T2) is
disclosed in PLAN.md as before. DECISION: KEEP the existing corpus —
generating 500 MB-1 GB would be unused scale; unique useful supervision is
the metric. Arm F self rows replay by design (16 fact forms, drilled).

## DOWNLOADS

```text
ITEM | SOURCE/PURPOSE | BYTES | CUMULATIVE BYTES
(none - no pip installs, no datasets, no checkpoints, no artifacts)
TOTAL_DOWNLOADED_GB = 0.0
```

## QUESTIONS_FOR_OPERATOR

```text
NONE
```

## BIGGEST BLOCKER

NONE REMAINING THAT LOCAL VALIDATION CAN ADDRESS. Estimated one-shot TPU
runtime: ~75-115 minutes (six arms + PRE50M + canary), hard ceiling <2 TPU-h.

## NEXT ACTION

Fresh Colab TPU -> run CELL 0 -> run CELL 1 (RUN EVERYTHING) once -> return
CITADEL_T1D_RESULTS.zip (or CITADEL_T1D_FAILURE.zip, which now carries the
exact phase + gate + traceback so no debugging round-trip is needed).

## CYMEK_REQUIRED_CHANGE

```text
NONE FOUND this cycle
```

## T0 / T1 / T1B / T1C (history)

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

