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
on the green-signal certification pass)
AUDITED_CYMEK_SHA: `28bf57a0d299a2c13a99fe0046616c00a1b8530c`
RUNTIME_PIN_SHA: `28bf57a0d299a2c13a99fe0046616c00a1b8530c` (live origin/cymek
HEAD == pin, re-verified this cycle; RUNTIME_AMENDMENT_001.md unchanged)

CYMEK_ALIGNMENT = PASS (pin == live HEAD; zero reconciliation needed)
RUNTIME_BOOTSTRAP_FRESH_CLONE = PASS (exact-SHA bootstrap unchanged; Cell 0
now also prints + enforces EXPECTED_CYMEK_SHA before Cell A)
LOCAL_TESTS = 63/63 PASS across all 6 files
  (t1d 32 — was 22, +10 new regression tests; t1c 10; t1_canary 6;
   notebooks 2; bootstrap 6; cymek_checkpoint 7)
POST_TRAIN_ARM_SIMULATION = PASS (full no-TPU arm flow: real rows + real
  feeder consumption -> nulls t0-t4 -> diagnostics -> lift -> gate ->
  receipt + marker serialization -> classification; would have caught the
  nulls_per_tier bug AND the train-memorization n=1 bug)
FULL_SESSION_SIMULATION = PASS (no-TPU session: green PRE50M -> BUNDLE VALID;
  PRE50M implementation failure -> arms preserved, fail-closed decision,
  BUNDLE VALID with ready_for_50m_training=false)
NOTEBOOK_TORTURE = PASS
PACKING_TORTURE = PASS
CHECKPOINT_CONTRACT = PASS (7/7 against pinned Cymek 28bf57a)
PRE50M_FAIL_CLOSED_MATRIX = PASS (24 independent single-condition mutations
  each force ready=false with a precise blocking reason)
BUNDLE_FAILURE_SURVIVAL = PASS (every required PRE50M artifact exists as an
  explicit IMPLEMENTATION_FAILURE receipt on failure; NEXT_50M_DECISION
  fail-closed; ZIP builds + verifies)
T1D = READY_FOR_REAL_TPU_VALIDATION
PRE50M = READY_FOR_REAL_TPU_VALIDATION

Green-signal repair cycle (independent-audit defects fixed + regression-tested):
1. nulls_per_tier: per-tier assignment moved INSIDE the loop (was: only t4
   written; scientific gate + cross-arm classifier crashed KeyError t0-t3
   AFTER expensive training). Mechanical `set == t0..t4` check + schema
   validator (validate_null_block) now runs before any receipt is written.
2. Train memorization: frozen FIRST-200 candidates per tier fixed before
   training; post-training verification against the feeder's EXACT consumed
   prefix (train_memorization_plan); only verified-consumed rows are scored;
   INSUFFICIENT_CONSUMPTION when n < LIFT_MIN_N=200 — FIRST_TRAIN_LIFT_TIER
   can never fire on n < 200 (was: sampled n=1 from a zero-consumption
   feeder, invalidating the 200-row diagnostic).
3. BUDGET_LIMITED: reads the REAL intermediates schema
   (inter[cp][f"t{tier}"]["exact"]) with the frozen aggregation mean DEV
   exact over tiers 1-4 at the final two preregistered checkpoints (was:
   read a nonexistent top-level dev_exact key — rule could never fire).
   No TEST at intermediate checkpoints. Tests: rising fires; flat/declining/
   high-test never fire.
4. PRE50M decision fail-closed: ready requires ALL conditions (target type +
   value verified from Cymek, fit, safe shape, positive finite throughput,
   smoke PASS, finite loss, nonzero finite gradients, parameter mutation,
   production transaction, checkpoint compat, reload identity, moments
   preserved, continued update, writer fence rejected-as-required, data
   interface PASS, packing PASS, token accounting: capacity >= real >=
   loss-bearing >= 0, padding == capacity-real, cumulative tokens match the
   production TrainingState). 24-mutation negative matrix green.
5. Bundle failure survival + durable arm receipts: every arm writes exactly
   one terminal ARM_X.json (incl. IMPLEMENTATION_FAILURE); PRE50M failure
   writes explicit failure receipts for every missing artifact; Cell E/F
   survive and the ZIP verifies.
6. Calibration SCALE2 guard: failed candidate marked IN PLACE (no duplicate
   selectable dicts); selected shape is the next SCALE2-passing one.
7. verify_bundle: unknown statuses rejected (incl. PRE50M placeholders must
   exist); bundled checkpoint SHAs verified against receipts.

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

