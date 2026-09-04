# AGENT.md — Citadel handover (machine-readable status for the operator's agent)

> Convention (binding): rewritten at the END of every Citadel work cycle,
> committed to the `citadel` branch ONLY, then `git push origin citadel`.
> Other branches are read-only audit inputs — never modified, never pushed.
> A CPU/CUDA run is NEVER a TPU result. No fabricated device results.
> Preregistration and results never share a commit. Download ceiling <10 GB;
> this cycle: 0 bytes.
>
> STANDING OPERATOR POLICY: prefer batched preregistered experiment suites
> over repeated round-trips. When multiple known hypotheses fit one compute
> session, test them together. Tiny experiments are automated internal
> gates/preflight only, never repeated operator-facing work. Bigger only when
> information value justifies it.

## STATUS

Citadel SHA: `f6668e7` (rebased onto hotfix tip; to be pushed this cycle)
Pinned Cymek runtime SHA: `298c91ac04f756f0833a7edcf63e73af3d5af688` (unchanged)

T1C operator run reached the real Colab TPU and successfully reused calibration
(shape 256×32, measured 3867 tok/s; preregistered auto-scale correctly halved
budgets) plus the 108.9 MB data manifest, then failed before training arm A.
Observed error: `prompt too long for fixed buffer: 'subtract 1353 from 1269 ='`;
arm A and B both hit the same deterministic evaluator defect and the session
correctly aborted on the second infrastructure failure. This is NOT a T1C
scientific result and creates no PASS/FAIL claim for any arm.

Root cause: evaluator used the training static length L=32 for greedy generation.
The full supervised row fit L=32, but a 25-token word-template prompt still
needs MAX_ANSWER_TOKENS=8 writable positions (required=33). The old manifest
`max_row_chars <= 32` invariant was therefore insufficient for generation.

Hotfixes pushed:
1. `calculator_eval.py`: dedicated static eval shape raised to L=64; new
   `validate_generation_capacity()` checks prompt + full generation headroom
   before XLA; teacher-forced answer CE now has an explicit row-length guard;
   exact failing prompt is a selftest regression and L=32 is asserted to fail.
2. `t1c_preflight.py`: now materializes/scans the complete DEV + all TEST slices
   (152,500 rows), runs evaluator selftests, validates every prompt against the
   fixed generation buffer, records max prompt / required headroom / full-row
   geometry, and removes the duplicate import. The same class of defect should
   now stop at preflight before any arm or TEST execution.

Independent hardening merged on top: notebook torture test
(`tests/test_citadel_notebooks.py`) proves for all 6 notebooks that every cell
compiles, no name is used before definition, TPU metadata is set, and every
receipt key touched exists in the producing schema. Revalidated green against
the hotfixed tree (see handover validation below).

No T0-critical semantics, model architecture, optimizer, T1C arm definitions,
data splits, success gates, or scientific thresholds changed. T1C
preregistration remains applicable; this is an implementation-only repair.

## T0 / T1 (history)

```text
T0: PASS (unchanged)
T1: FAIL (loss-learned, exact-flat; historical result unchanged)
```

## T1B

```text
SUPERSEDED_BY_T1C (preserved, unexecuted)
```

## T1C

```text
READY_FOR_OPERATOR_RERUN_AFTER_HOTFIX
```

The failed A/B attempts did not produce scientific arm receipts/markers before
the exception. Existing Colab session calibration and data manifest are safe to
reuse; arm execution should restart from A after code refresh.

## DOWNLOADS

```text
ITEM | SOURCE/PURPOSE | BYTES | CUMULATIVE BYTES
(none)
TOTAL_DOWNLOADED_GB = 0.0
```

## QUESTIONS FOR OPERATOR

```text
NONE
```

## BIGGEST BLOCKER

Need operator to refresh the Citadel checkout/module cache and rerun T1C from
preflight; only real TPU execution can reveal further hardware/compiler issues.

## NEXT ACTION

In the existing Colab TPU runtime: restart the Python session (do NOT delete the
runtime/disk), rerun notebook Cell 0, verify Citadel SHA is this hotfix tip or
newer, rerun Cell A and require READY_FOR_T1C=YES, then run Cell D. Calibration
and DATA_MANIFEST may be reused. Finish E/F and return CITADEL_T1C_RESULTS.zip.

## Latest hotfix commits

```text
2a123f2 test(citadel): preflight all T1C eval buffer invariants
3ee6597 fix(citadel): give T1C generation safe fixed-buffer headroom
```
