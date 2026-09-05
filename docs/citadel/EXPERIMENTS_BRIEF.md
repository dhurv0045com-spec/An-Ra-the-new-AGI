# Citadel experiments — brief record (for future reference)

One line per experiment: what was tested, what happened, what it means.
Evidence: `docs/citadel/tpu_receipts/` + plans in `docs/citadel/experiments/`.
Rule: loss moving is not learning; only held-out answer generation counts.

## T0 — one-update certification (Colab TPU, MINI_SPEC 1.6M)

Proved the real Cymek stack trains on TPU: forward → finite loss (10.12) →
backward → optimizer step → params changed → checkpoint save/reload identical.
Verdict: **PASS**. Standing reference for "plumbing works".

## T1 — calculator canary (Colab TPU, MINI_SPEC, 200 updates, 4k rows)

Loss 10.10→2.85 and dev CE 10.08→2.91, but held-out exact-match 0/500 before
and after; reload identical. Verdict: **FAIL** (honest null result).
Meaning: optimization without answer-rule movement (H2 direction). Exposed
that whole-row CE can collapse while answer accuracy stays zero.

## T1B — batched scale ladder (same 4k rows to 20M tokens)

Preregistered but **SUPERSEDED, unexecuted** — T1C answers its question with
cleaner contrasts instead of 160-epoch replay. Preserved, not deleted.

## T1C — batched discriminator (Colab TPU, one session, 2026-09-04)

Four arms, same seed/init, budget fixed at 4M cap tokens each (auto-scale
halved 8M→4M at measured 3782 tok/s, as preregistered):

| Arm | Model | Objective | Data | Loss | Test exact | Train exact | Verdict |
|-----|-------|-----------|------|------|------------|-------------|---------|
| A | MINI 1.6M | whole-row CE | rich 6.5M rows | 10.07→1.87 | 0/500 | 0/500 | FAIL |
| B | MINI 1.6M | answer-only CE | rich 6.5M rows | 10.06→1.90 | 0/500 | 0/500 | FAIL |
| C | MINI 1.6M | answer-only CE | narrow 4k rows | 10.07→1.34 | 0/500 | 6/500 | FAIL |
| D | MID 3.7M | answer-only CE | rich 6.5M rows | 10.01→1.61 | 0/500 | 0/500 | FAIL |

All reloads identical. Strongest heuristic null 2.7% (copy-first-operand),
beaten by no arm. Cross-arm verdict: **INCONCLUSIVE** (no contrast rule fired —
correct, since nothing moved anywhere).

What the diagnostics say (the valuable part):
- **Not a format failure.** Every generation stops at MAX_TOKENS emitting
  valid integers (` 1110000`, ` 2220000`, …): shape learned, digits wrong.
  The model distribution-matches without computing.
- **No memorization either.** Train exact ≈ 0 even on seen rows — the model
  fits loss statistics, not instances.
- **Objective, data, and 2.3× scale all moved nothing** at 4M tokens.
  Combined with T1, the pattern is: loss-learning without any exact-match
  movement across objectives, corpora, and scales tried so far.

## Standing open questions for the next design

1. Is answer generation below a capacity/budget floor (needs bigger model or
   far more tokens), or is whole-row/answer CE the wrong teacher for exact
   symbolic output?
2. Would explicit answer-format supervision (e.g. digit-level weighting,
   shorter answers-first curriculum) unlock what flat CE does not?
3. At what scale, if any, does train exact first lift off zero?

## T1D — ARITHMETIC LIFT-OFF (preregistered, pending execution)

One session, five arms on the tiered ladder corpus (~130 MB unique):
A flat control / B frozen easy→hard curriculum / C + micro-teacher rows /
D 7.4M scale / E masked-vocab diagnostic. Primary readout is the tier
lift-off curve (FIRST_TRAIN/TEST_LIFT_TIER), not one global zero. Discriminates
floor vs curriculum vs teacher vs representation. No results yet.
Same session also runs PRE50M systems certification (SCALE2 smoke, data
interface, packing, buckets, throughput curve, memory, checkpoint compat) and
emits NEXT_50M_DECISION.json — the 50M-token milestone gate. ("50 million
checkpoint" = 50M training tokens per Cymek training_spec; no 50M-param spec
exists, so no new model is built for it.)

## Proven infrastructure (reusable)

Pinned Cymek runtime bootstrap, PJRT/XLA backend shim, fail-closed probing,
generation evaluator with Wilson + nulls + reload-hash gate, preflight gates,
resume markers, one-bundle session export. All validated on real TPU runs.
