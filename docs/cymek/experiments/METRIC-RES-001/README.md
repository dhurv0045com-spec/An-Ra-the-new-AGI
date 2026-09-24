# METRIC-RES-001

## Purpose

Decide the branch's single largest open question with the smallest possible
expenditure of compute: is S5 identity/copy capability **forming-but-unmeasured**
or **genuinely absent**?

## Why this experiment and not another

`docs/cymek/experiments/FORMATION-MUX-001/RETURNED_BUNDLE_AUDIT.md` recorded the
disposition after the S5 run: all 24 arms came back at identity formation AUC
0.000–0.008, so the mechanism contrasts had no dynamic range and no conclusion
about extra-row decay, trainability, or denominator participation was licensed.
Its item 2 names the next single action verbatim:

> a metric-resolution audit on the PRESERVED final checkpoints (token-level
> diagnostics; no new training required)

`docs/cymek/experiments/FORMATION-MUX-001/observed/S5_KAGGLE_2026-09-15/POSTMORTEM.md`
independently identifies the prime suspect: the identity family uses an
all-or-nothing sequence-exact endpoint, so a model emitting the first two of
three answer tokens scores 0.0, while the 1-token termination family resolves
immediately (and did, at 1.00).

`tools/formation_diag_001.py` does **not** implement this. It trains from
scratch on the v1 surface. The prescribed checkpoint-only path did not exist.
This experiment supplies it.

## What it does

- Loads the preserved S5 `resume.pt` files read-only via the audited
  model-only restore path. No optimizer is reconstructed. No backward pass is
  executed. Zero training steps.
- Re-scores all four CS-MECH-002 arms across all four seed bundles (16
  checkpoints) on the identity family, plus the termination family as a
  positive control.
- Adds longest-common-prefix, which the S5 endpoint could not express.
- Re-adjudicates the three frozen mechanism contrasts in token-accuracy space.

## Integrity

The termination family is a **positive control**: S5 measured 1.00 exact+valid-EOS.
A valid instrument must reproduce that. If it does not, the run reports
`INSTRUMENT_INVALID` and licenses no conclusion.

Every checkpoint is identity-verified on load (experiment, arm, seed bundle,
protocol SHA, surface SHA). Sealed rows are never loaded, regenerated, scored,
or persisted. The aggregation stage fails closed if any shard receipt claims
training steps or sealed contact.

## Decision tree

Frozen in `PREREGISTRATION.json` before any outcome, applied in this order:

| Branch | Condition | Next action |
|---|---|---|
| `INSTRUMENT_INVALID` | termination control < 0.90 | fix the instrument; no conclusion |
| `FORMING_BUT_UNMEASURED` | token acc >= 0.80, exact < 0.30, LCP >= 0.50 | endpoint was the bottleneck; re-adjudicate contrasts, then fix the endpoint |
| `OPTIMIZATION_CHOKED` | token acc < 0.30, clip_fraction >= 0.99 | S6 LR/schedule bracket before any mechanism re-run |
| `OUTPUT_COMPETITION` | shared-only rescue >= 0.10, token acc < 0.80 | denominator probe on a resolved instrument |
| `MIXTURE_OR_DATA_LIMITED` | 0.30 <= token acc < 0.80 | baseline-gate exposure/mixture ladder |
| `GENUINELY_ABSENT` | token acc < 0.30, termination reproduced | abandon vocabulary/mechanism work; go to P35-TRANSFER-004 |
| `INCONCLUSIVE` | none matched | preserve evidence, re-plan |

## Claim ceiling

Diagnostic re-adjudication only. The S5 NULLs stand permanently as executed. A
positive result means the **instrument** hid the signal, not that the S5 verdict
was wrong. No architecture promotion, no tokenizer or vocabulary change, no
scale authorization, no cognition or AGI claim.

## Running it

**Colab T4** — `notebooks/CYMEK_METRIC_RES_001_COLAB_T4.ipynb`, 20–40 min.
**Kaggle T4 x2** — `notebooks/CYMEK_METRIC_RES_001_KAGGLE_T4X2.ipynb`, 10–20 min,
2-way sharded.

Both notebooks require the S5 `resume.pt` files. They are **not** in the S5
result ZIP; they remain in the original S5 run's Kaggle Output tree. Layout:

```
FORMATION_MUX_001/
  PUBLIC_SURFACE_MANIFEST.json
  CS-MECH-002/
    M0_STANDARD/S1..S4/resume.pt
    M1_EXTRA_NO_DECAY/S1..S4/resume.pt
    M2_EXTRA_FROZEN/S1..S4/resume.pt
    M3_EXTRA_FROZEN_MASKED/S1..S4/resume.pt
```

On Kaggle, attach that Output as Input. On Colab, copy the tree to
`MyDrive/CYMEK/FORMATION_MUX_001/`.

## Operational notes

Colab 2026 Pro+ runtimes are documented to terminate background executions
after 3–10 hours with no error log, and to survive longer while the browser tab
stays open. This audit is short, so the risk is small, but keep the tab open.

Colab's Drive mount drops around the 10-hour mark with
`OSError: [Errno 107] Transport endpoint is not connected`. The operator
therefore writes all receipts to local disk first and syncs to Drive with four
bounded retries, so a mount blip cannot destroy completed evidence.
