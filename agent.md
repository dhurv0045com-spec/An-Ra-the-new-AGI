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

Citadel SHA: `168be62` (local; to be pushed this cycle)
Pinned Cymek runtime SHA: `298c91ac04f756f0833a7edcf63e73af3d5af688` (unchanged)

T1C built this cycle: one-session 4-arm discriminator on ~100 MB unique
synthetic arithmetic (indexed generator, production loss seam for the
answer-objective arm, principled 3.7M MID scale arm, calibration-selected
shape, resume markers, machine-evaluated cross-arm rules, one result bundle).

## T0 / T1 (history)

```text
T0: PASS (unchanged, still applicable — no T0-critical semantics touched)
T1: FAIL (loss-learned, exact-flat; correct H2-direction reading stands)
```

## T1B

```text
SUPERSEDED_BY_T1C (preserved, unexecuted — T1C answers with cleaner contrasts)
```

## T1C

```text
READY_FOR_OPERATOR_RUN
```

Arms A (control) / B (answer-objective) / C (narrow-data) / D (MID scale),
8M cap tokens each, Q1–Q4 contrasts preregistered, TEST-family 32 new frozen
observations total (all reported), repetition disclosed, ceiling <2 TPU-h.

## DOWNLOADS

```text
ITEM | SOURCE/PURPOSE | BYTES | CUMULATIVE BYTES
(none — no pip installs, no datasets, no checkpoints, no artifacts)
TOTAL_DOWNLOADED_GB = 0.0
```

## QUESTIONS FOR OPERATOR

```text
NONE
```

## BIGGEST BLOCKER

Need operator to execute the preregistered T1C session (the discriminator for
H_FORMAT / H_OBJECTIVE / H_DATA / H_BUDGET / H_SCALE / H_ARCH).

## NEXT ACTION

Run `notebooks/citadel_colab_t1c.ipynb` end-to-end once and return the result bundle.

## Validation this cycle (local, 0 downloads)

14/14 unit tests (T1 6/6 incl. notebook references; T1C 8/8 incl. Decimal
Wilson cross-check, eval-slice leakage zeros, every classifier rule, MID
structural rules); MID receipt 3,737,472 verified against real Cymek
contracts; t1c_preflight all-green except hardware gate (correct NO);
compileall clean; fail-closed intact; generator 200k rows/2.4s (5M ≈ 60 s).

## Commit log (latest first, citadel only)

```text
168be62 feat(citadel): add T1C session notebook
d850d2b test(citadel): T1C preflight and unit tests
0d1676a feat(citadel): extend evaluator for arrow and letter rows
731b5bc feat(citadel): add T1C arm runner with scale and objective contrasts
36da48d feat(citadel): add indexed arithmetic corpus generator
4bc9afd research(citadel): preregister T1C batched discriminator
```
