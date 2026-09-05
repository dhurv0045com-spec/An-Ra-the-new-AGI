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

CURRENT_CITADEL_SHA: `2e1c233` at cycle start (handover commit itself on top;
see `git log origin/citadel -1` for the live tip)
AUDITED_CYMEK_SHA: `28bf57a0d299a2c13a99fe0046616c00a1b8530c`
RUNTIME_PIN_SHA: `28bf57a0d299a2c13a99fe0046616c00a1b8530c` (see
`docs/citadel/experiments/T1D/RUNTIME_AMENDMENT_001.md`)

CYMEK_ALIGNMENT = PASS (T1D surface byte-identical pin→HEAD; delta is additive
packaging/data-pipeline/registry/eval files only)
RUNTIME_BOOTSTRAP_FRESH_CLONE = PASS (old pin resolved from a clone lacking
any cymek ref — the reported failure mode is dead)
LOCAL_TESTS = 53/53 PASS (full suite below)
NOTEBOOK_TORTURE = PASS
PACKING_TORTURE = PASS
CHECKPOINT_CONTRACT = PASS (real Cymek transaction path: publish/restore/
fence/inventory/corruption, 7/7 green against pinned code)
PRE50M_DECISION_LOGIC = PASS (per-condition failure matrix green)

T1D = READY_FOR_REAL_TPU_VALIDATION
PRE50M = READY_FOR_REAL_TPU_VALIDATION

Final hardening cycle: exact-SHA bootstrap rewrite (6/6 hermetic git
regressions), Cymek production checkpoint adapter (no duplication) with the
PRE50M smoke rewired through real transactions, arm status schema corrected
to SCIENTIFIC_PASS/FAIL + TIMEBOX_ABORT, calibration OOM isolation + SCALE2
verification, loss-alignment/predictor/template/packing torture tests,
classifier + decision torture matrices, bundle pre-download verification,
notebook hardened (both verdicts, lift tiers, blocking reasons).

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

None remaining that local validation can address — next is the operator run.

## NEXT ACTION

Run `notebooks/citadel_colab_t1d.ipynb` once from Cell 0 through F in a
fresh/restarted Colab TPU session and return `CITADEL_T1D_RESULTS.zip`.

## CYMEK_REQUIRED_CHANGE

```text
NONE FOUND this cycle (checkpoint, data-interface, and contract surfaces
checked against current Cymek; Citadel adapts at its own layer only)
```
