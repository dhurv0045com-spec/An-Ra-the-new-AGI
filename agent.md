# AGENT.md — Citadel handover (machine-readable status for the operator's agent)

> Convention (binding): rewritten at the END of every Citadel work cycle,
> committed to the `citadel` branch ONLY, then `git push origin citadel`.
> Other branches are read-only audit inputs — never modified, never pushed.
> A CPU/CUDA run is NEVER a TPU result. No fabricated device results.
> Preregistration and results never share a commit. Download ceiling <10 GB;
> this cycle: 0 bytes.

## STATUS

Citadel SHA: `2988980` (local; to be pushed this cycle)
Pinned Cymek runtime SHA: `298c91ac04f756f0833a7edcf63e73af3d5af688`
(= `origin/cymek`, unmoved this cycle — T1 keeps the exact T0-certified runtime)

This cycle (T1 scientific validity + turnkey): T0 stays PASS (no T0-critical
semantic changed: `one_update.py`/`xla_backend` stepping untouched;
`environment.py` gained only additive `pjrt_device_env`; model/objective/
optimizer/checkpoint paths byte-identical via the pinned runtime — no
re-certification required). Delivered, all locally validated: T1
AMENDMENT_001 (frozen pin/gate/ladder), generation-based `calculator_eval.py`,
dev-gated `calculator_train.py` rewrite, `calculator_preflight` command,
turnkey Colab T1 cells, 6/6 unit tests green (independent Decimal Wilson
reference, data invariants, nulls, notebook-reference integrity).

## T0

```text
PASS (unchanged, still applicable)
```

## T1

```text
READY_FOR_OPERATOR_RUN
```

## T1 AMENDMENT

`docs/citadel/experiments/T1/AMENDMENT_001.md` (commit `4a53ad7`).
Pin `4abeaeb` → T0-certified `298c91a`; reason recorded; no T1 results exist.

## PRIMARY METRIC

Held-out TEST exact-match from static-shape greedy generation
(prompt `"<a> <op> <b> ="`, strict integer normalization), Wilson 95% interval.
TEST observed exactly twice total (untrained baseline + trained final); DEV
drives all ladder escalation [5,20,100,200].

## SUCCESS GATE (frozen, machine-evaluated)

1. trained_acc > untrained_acc AND trained_LCB > untrained_UCB
2. trained_LCB > strongest heuristic-null accuracy (4 mechanical nulls)
3. trained_acc − untrained_acc ≥ 0.10 (>2× sampling noise at n=500)
4. final_loss < first_loss AND trained_test_CE < untrained_test_CE
5. pre_reload_prediction_sha256 == post_reload_prediction_sha256

## TEST-SET POLICY

TEST never drives decisions. Confirmed: no TEST observation exists anywhere
(no T1 execution yet). First two TEST observations will be the preregistered
untrained baseline and the single trained final.

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

Need operator to execute the preregistered T1 Colab run.

## NEXT ACTION

On the Colab TPU runtime with updated `origin/citadel`: run notebook T1 cells
A–F sequentially (preflight → data receipt → train → summary → reload display
→ export); transfer back the exact `TPU_CALCULATOR_CHECKPOINT.json` file.

## Commit log (latest first, citadel only)

```text
2988980 feat(citadel): extend colab notebook with turnkey T1 section
27ef457 feat(citadel): add T1 preflight command; record pjrt_device_env
19335dd feat(citadel): wire dev-gated T1 training ladder
7464da7 test(citadel): notebook-reference integrity tests
4a53ad7 research(citadel): preregister T1 amendment 001
b99ef12 feat(citadel): add generation-based calculator evaluator
```
