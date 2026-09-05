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

Citadel SHA: `8cb3be1` (local; to be pushed this cycle)
Pinned Cymek runtime SHA: `298c91ac04f756f0833a7edcf63e73af3d5af688` (re-verified;
`origin/cymek` moved +2 but T1D surface byte-identical — additive packaging only)

T1D built: one-session 5-arm lift-off discriminator (A flat / B curriculum /
C teacher / D 7.4M scale / E masked-vocab diagnostic) on the tiered ladder
corpus (~130 MB unique, band-isolated splits, micro-teacher tasks, deterministic
packing, tier lift-off curves, machine-evaluated rules, one bundle).

## T0 / T1 (history)

```text
T0: PASS (unchanged, still applicable)
T1: FAIL (loss-learned, exact-flat; historical result unchanged)
```

## T1B / T1C (history)

```text
T1B: SUPERSEDED_BY_T1C (preserved, unexecuted)
T1C: EXECUTED — 4 arms FAIL, cross-arm INCONCLUSIVE (valid-integer mode
collapse; no memorization; objective/data/2.3x scale moved nothing at 4M)
```

## T1D

```text
READY_FOR_OPERATOR_RUN
```

Arms A/B/C/D/E; models MINI-pattern MID 3.7M (A,B,C,E) + SCALE2 7.4M (D);
~130 MB unique data; 8/8/8/4/4M cap-token budgets with halve-on-slow rule;
~60–110 min est, ceiling <2 TPU-h.

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

Need operator to execute the preregistered T1D session (floor vs curriculum
vs teacher vs representation — the T1C zeros left this unanswered).

## NEXT ACTION

Run `notebooks/citadel_colab_t1d.ipynb` end-to-end once and return
`CITADEL_T1D_RESULTS.zip`.

## Validation this cycle (local, 0 downloads)

33/33 unit tests (15 T1D incl. band isolation, verdict logic, feeder/packing
invariants, classifier rules; 10 T1C; 6 T1; 2 notebook); full tiered manifest
dried at scale (132.6 MB, 58 s, max row 31, dup ≤ 0.29 disclosed, leakage
fatal-empty); MID+SCALE2 receipts verified against real Cymek contracts;
preflight ideal (correct NO at hardware); fail-closed intact.

## Commit log (latest first, citadel only)

```text
8cb3be1 test(citadel): T1D unit tests and session notebook
ed6c2a2 feat(citadel): add T1D arm runner and preflight
5369fe2 research(citadel): preregister T1D lift-off discriminator
```
