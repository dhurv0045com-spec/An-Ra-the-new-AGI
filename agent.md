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

CURRENT_CITADEL_SHA: see `git log origin/citadel -1` (this handover commit sits
on the T1D schema hotfix cycle)
AUDITED_CYMEK_SHA: `28bf57a0d299a2c13a99fe0046616c00a1b8530c`
RUNTIME_PIN_SHA: `28bf57a0d299a2c13a99fe0046616c00a1b8530c` (live origin/cymek
HEAD == pin, re-verified)

CYMEK_ALIGNMENT = PASS (pin == live HEAD)
RUNTIME_BOOTSTRAP_FRESH_CLONE = PASS (exact-SHA bootstrap unchanged; Cell 0
prints + enforces EXPECTED_CYMEK_SHA before Cell A)
LOCAL_TESTS = 69/69 PASS across all 6 files
  (t1d 38 — was 32, +6 schema-contract regressions; t1c 10; t1_canary 6;
   notebooks 2; bootstrap 6; cymek_checkpoint 7)
POST_TRAIN_ARM_SIMULATION = PASS
FULL_SESSION_SIMULATION = PASS (green + PRE50M-failure; malformed scientific
  receipts are demoted to IMPLEMENTATION_FAILURE at the session boundary)
PRODUCER->FINALIZER SCHEMA = PASS (exact legacy producer shape flows the full
  bridge: normalize -> finalizer -> validator -> classifier -> curves)
PREFINAL RECOVERY = PASS (hash-verified snapshot; finalization-only rerun;
  corrupt/mismatch/missing-checkpoint refused; finalizer exception retains
  the sidecar — expensive arms are never lost to a receipt bug again)
NOTEBOOK_TORTURE = PASS (Cell D now reloads execution modules before the
  session — no stale in-memory code on the operator's open Colab kernel)
NOTEBOOK CELL D RELOAD = PASS
PACKING_TORTURE = PASS
CHECKPOINT_CONTRACT = PASS (7/7 against pinned Cymek 28bf57a)
PRE50M_FAIL_CLOSED_MATRIX = PASS (24 mutations)
BUNDLE_FAILURE_SURVIVAL = PASS
T1D = READY_FOR_REAL_TPU_VALIDATION
PRE50M = READY_FOR_REAL_TPU_VALIDATION

## REAL TPU FAILURE RECORD (2026-09-05, must not be repeated or hidden)

```text
T1D attempt: IMPLEMENTATION_FAILURE (arms A and B, after expensive training)
cause: run_arm stored untrained results as dev_tN/test_tN while
       build_arm_receipt/classify/curves read tN — scientific gate died on
       KeyError: 't1' in the PURE post-training finalizer
A/B:   INVALID — do not count scientifically; must rerun (no valid prefinal
       snapshots existed then; nothing reconstructed from checkpoints)
C/D/E: not completed
TPU itself: NOT implicated — the failure was a pure receipt-schema mismatch
session aborted on the 2nd infra failure, per policy
```

Schema hotfix cycle (implementation-only; no scientific content changed):
1. Canonical untrained contract at the source: run_arm now produces
   untrained_test[tN] (receipt "untrained") + untrained_dev[tN] (explicit
   separate block); DEV and TEST each still evaluated exactly once per arm
   (a dictionary alias is not an observation).
2. normalize_untrained_receipt: pure defense-in-depth — accepts canonical
   t0-t4 or legacy test_t0-t4, validates every summary
   (correct/total/accuracy/wilson_lcb/wilson_ucb, total>=0, accuracy finite
   [0,1]), fails ARM_SCHEMA_INVALID instead of any raw KeyError.
3. validate_arm_receipt terminal validator: full scientific contract
   enforced BEFORE ARM_X.json is finalized, inside verify_bundle(), AND at
   the session boundary — malformed scientific receipts are demoted to
   IMPLEMENTATION_FAILURE (recorded on disk with schema_defects) instead of
   crashing the classifier.
4. ARM_<tag>.prefinal.json recovery sidecar: written after training + TEST
   evals + checkpoint + reload identity; self-hash protected; on rerun,
   finalization-only resume (NO retraining, NO device); corrupt/mismatched/
   checkpoint-missing snapshots archived aside, never trusted.
5. should_skip_arm: IMPLEMENTATION_FAILURE is NEVER completion (marker or
   not) — the arm retries after software repair; orphan checkpoints from
   failed sessions are archived as forensic artifacts, never clobbered.
6. Cell D reloads calculator_eval/tiered_data/t1c_run/t1d_run/pre50m right
   before the session; prints the arm receipt schema.
7. Preflight gained the producer->finalizer schema gate (exact legacy
   producer shape through the full bridge; FAIL => READY_FOR_T1D=NO).

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

