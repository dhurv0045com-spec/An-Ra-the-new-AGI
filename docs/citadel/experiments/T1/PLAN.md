# T1 — Preregistration: calculator checkpoint on single TPU device

Status: **PREREGISTERED — NO RESULTS** (written before any TPU execution).
Branch: `citadel`. Prerequisite: T0 PASS (`TPU_ONE_UPDATE.json` +
`TPU_RESUME.json`). Source pins: Cymek `origin/cymek@4abeaeb`.

## Question

Through the real single-device TPU training infrastructure, does the smallest
useful An-Ra/Cymek-compatible model visibly learn arithmetic (train→held-out)
on the deterministic calculator canary?

## Existing evidence

- T0 (prereq): one certified TPU update exists — or T1 is BLOCKED.
- CUDA training-behavior evidence is reference only.
- Calculator canary is deterministic with known train/held-out split mechanics.

## Primary hypothesis

H1: held-out arithmetic accuracy rises from near-null untrained baseline to a
visibly above-null trained value with a falling loss curve, at tiny token cost.

## Competing hypotheses

- H2 (plumbing without learning): loss falls on train but held-out stays at
  null → memorization/format failure, not rule extraction.
- H3 (scale floor): neither train nor held-out moves → tiny model/budget below
  the learning floor even for arithmetic; ladder spec, not data, is next.

## Why this experiment

First proof that the real An-Ra stack learns on TPU. Checkpoint becomes the
experimental substrate for later work. No corpus, no scale, no sealed touch.

## Independent variable

Exactly one: training (untrained init vs post-training checkpoint), same model,
same TPU path, same eval.

## Fixed variables

- Model: same tiny spec as T0 (SHA pinned at run); init seed recorded.
- TPU path: identical shims as certified T0; bucket 512 fixed-batch.
- Data: fixed calculator train/held-out splits with hash-recorded seeds.
- Optimizer/schedule: identical between the measured endpoints.

## Models/checkpoints

Init SHA, final checkpoint SHA, reload-accuracy hash. Checkpoints are
artifacts, not committed.

## Data

Deterministic calculator generator; train/held-out seed disjointness asserted;
hashes recorded. No external corpus. No sealed fixtures.

## Controls

- Untrained-model baseline on identical eval (null reference).
- Heuristic nulls over identical instances (copy/format priors as applicable).
- Reload-identity: post-save reload accuracy equals pre-save accuracy.

## Metrics

Primary first: held-out exact-match accuracy (Wilson 95% interval).
Secondary: train loss curve, train accuracy, tokens consumed, updates, wall
time, tokens/sec, checkpoint hash, reload accuracy.

## Statistical treatment

Wilson 95% intervals on rates; pre/post comparison on identical held-out set
(paired). No cross-seed claim from a single run.

## Success threshold

Held-out LCB above untrained baseline AND above strongest heuristic null AND
loss curve decreasing AND reload-identical. (Exact numeric gate set at run
from the recorded untrained baseline; amendment if baseline surprises.)

## Failure threshold

Held-out within null of baseline after the fixed budget → classify via
training-set fit (H2 vs H3) before any further interpretation.

## Confound checks

1. T0 certification still valid for the exact code path used.
2. Train/eval seed disjointness mechanically asserted.
3. Token ledger exact (no duplication).
4. `xm.mark_step` ordering unchanged from T0.
5. No threshold/metric/seed change after result inspection.

## Compute budget

Ceiling: 2.0 TPU-hours. Expected well under. Storage: one tiny checkpoint + receipts.

## Stop condition

One train-to-budget run + one eval battery + receipts. Then stop. Replication
seeds only if H1 holds and a promotion case is being built.

## Possible outcomes

1. H1 → M3 (8-device) planning begins; checkpoint becomes the house experimental substrate.
2. H2 → template-diversity/format diagnosis; no scale-up.
3. H3 → parameter/token ladder discriminator, not data work.
4. Degenerate (loss non-decreasing, ledgers inconsistent) → IMPLEMENTATION_FAILURE; re-validate T0 path first.
