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

## STATUS

Citadel code tip described here: `fc9e7d2` (handover commit itself on top;
see `git log origin/citadel -1` for the live tip — the two differ by docs only)
Pinned Cymek runtime SHA: `298c91ac04f756f0833a7edcf63e73af3d5af688` (re-verified;
`origin/cymek` moved +2 but T1D surface byte-identical — additive packaging only)

T1D + PRE50M built: one session runs five lift-off arms AND the PRE50M
systems certification (SCALE2 smoke, data interface, packing, buckets, compile
audit, throughput curve, memory, checkpoint compat, NEXT_50M_DECISION).
"50M checkpoint" resolved from Cymek sources: 50M training TOKENS milestone
(training_spec.py:223-224), not parameters — no 50M-param spec exists, so no
new model is built; the smoke certifies the path on SCALE2 7.4M.

## T0 / T1 / T1B / T1C (history)

```text
T0: PASS (unchanged, still applicable)
T1: FAIL (loss-learned, exact-flat; historical result unchanged)
T1B: SUPERSEDED_BY_T1C (preserved, unexecuted)
T1C: EXECUTED — 4 arms FAIL, cross-arm INCONCLUSIVE (mode collapse, no
memorization; objective/data/2.3x scale moved nothing at 4M)
```

## T1D

```text
READY_FOR_OPERATOR_RUN
```

Arms A/B/C/D/E; MID 3.7M (A,B,C,E) + SCALE2 7.4M (D); ~130 MB unique data;
8/8/8/4/4M cap-token budgets with halve-on-slow rule; ~60–110 min est.

## PRE50M

```text
READY_FOR_OPERATOR_VALIDATION
```

Smoke (SCALE2, ≤10 updates, easy sanity + forward/backward/opt/checkpoint/
reload/resume + numerics), data interface, packing matrix, 5-bucket compile
audit, throughput curve (MID+SCALE2 → 10M/50M/100M/1B estimates), memory FIT
arithmetic, grad-accum NOT_REQUIRED (recorded), checkpoint-compat check,
NEXT_50M_DECISION machine gate. Grad accumulation NOT implemented (condition
false — would be dead code).

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

Need operator to execute the preregistered T1D+PRE50M session (one run answers
both the arithmetic question and the 50M-readiness question).

## NEXT ACTION

Run `notebooks/citadel_colab_t1d.ipynb` ONCE from Cell 0 through F and return
`CITADEL_T1D_RESULTS.zip`.

## Validation this cycle (local, 0 downloads)

36/36 unit tests (18 T1D incl. packing adversarial, teacher band, PRE50M
estimators/decider, masked vocab, resume dry-run; 10 T1C; 6 T1; 2 notebook);
tiered manifest at scale dried (132.6 MB, max row 31, leakage fatal-empty);
MID+SCALE2 receipts vs real Cymek contracts; preflight ideal (correct NO at
hardware); fail-closed intact; checkpoint module repaired + verified.

## Commit log (latest first, citadel only)

```text
fc9e7d2 test(citadel): PRE50M estimators, packing adversarial, teacher band
eb6dd18 feat(citadel): wire PRE50M phase and extended arm diagnostics
23d2490 research(citadel): preregister PRE50M systems certification
```
