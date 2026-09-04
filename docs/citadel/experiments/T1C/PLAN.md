# T1C — Preregistration: large batched learning discriminator (one session)

Status: **PREREGISTERED — NO RESULTS**. Branch `citadel`. New experiment (not
an amendment). T1 (`FAIL`: loss 10.10→2.85, exact 0/500) and T1B (preregistered,
SUPERSEDED operationally — see §A11) are preserved history. Cymek runtime pin:
`298c91ac04f756f0833a7edcf63e73af3d5af688` (unchanged; T0-certified).
No T1C results exist anywhere; no threshold tuned on data.

## Why T1C supersedes T1B operationally

T1B replays one 4000-row corpus to 20M tokens (160 epochs). T1 already showed
loss-learning without exact-match movement at 1.6 epochs; T1B's design cannot
tell replay-memorization from rule-learning cleanly and spends most of a
session re-reading 50 kB of unique text. T1C instead varies, in ONE session,
the three leading explanations (objective shape, data uniqueness/diversity,
parameter scale) with controlled contrasts on substantially unique data.

## Question (one session, four preregistered comparisons)

Q1 OBJECTIVE: does answer-focused supervision improve held-out answer
generation vs whole-row CE (A vs B)?
Q2 DATA: does a much larger/diverse unique corpus improve generalization vs a
narrow repetitive corpus (B vs C)?
Q3 SCALE: does modest parameter scaling improve answer generation under the
same data/objective (B vs D)?
Q4 MEMORIZATION: does any arm fit TRAIN while failing TEST (all arms)?

## Arms (exactly one variable per contrast)

Same tokenizer, batch/sequence shape (from calibration, all arms), optimizer
(AdamW 0.9/0.95/1e-8, wd 0.1, LR 3e-4), init-seed convention (fixed seed per
arm role, recorded), pinned runtime, single XLA device.

```text
ARM A — CONTROL:            MINI_SPEC 1,647,104 | whole-row CE | rich data | 8M cap tokens
ARM B — ANSWER OBJECTIVE:   MINI_SPEC           | answer-only CE (eligible-mask seam) | rich data | 8M
ARM C — NARROW DATA:        MINI_SPEC           | answer-only CE | 4000-row canary (replay disclosed) | 8M
ARM D — SCALE:              MID_SPEC 3,737,472  | answer-only CE | rich data | 8M
```

A vs B → objective effect. B vs C → uniqueness/diversity effect. B vs D →
scale effect. Budgets frozen (auto-scale rule §A8 only). Intermediate DEV
observations at 10/25/50/100% per arm (automated, no TEST, no interaction).

## Models

MINI_SPEC (T0-certified). MID_SPEC (new, same dense-decoder family — NOT new
architecture): layers 4, width 128, query_heads 8, kv_heads 4, head_dim 16,
ffn 256, vocab 24576, ctx 4096, rope_base 10000, RMSNorm 1e-5, tied, QK-norm
affine, no bias, dropout 0. Expected receipt total 3,737,472; validated at
runtime by cymek `ModelSpec.assert_valid()` (receipt mismatch aborts the arm).
P35/250M explicitly out of scope.

## Objective arms

Whole-row CE = T1 semantics (prompt supervised, PAD excluded). Answer-only CE
= production `causal_lm_loss` with `eligible` mask covering exactly the answer
token span (existing seam, no fork, production path intact). Both record
supervised-answer vs supervised-whole counts every arm.

## Data (synthetic, deterministic, indexed — no downloads)

Generator `arith_data` v1: pure function `row(split, i)` (O(1) memory, hashable,
resumable). 4 templates (canon `a op b = c`, compact `a+b=c`, arrow
`a op b -> c`, words e.g. `add A and B = C`); ops +−×÷ (exact division);
difficulty metadata (digits, carries, borrows, magnitude, template_id).
Physical target ~100 MB (≈6.5M rows TRAIN + eval slices); manifest in git
(version, code hash, counts, split hashes, byte/token estimates, max row
length ≤ 32 asserted, audit-sample duplicate rate); corpus ephemeral (never
committed). Every op family spans ~10^6+ unique (a,b) combos (mult/div caps
match split magnitude — no collapsed families). Splits (structurally disjoint
ranges/templates):
TRAIN (0–999, canon/compact/arrow), DEV (1000–1999, canon/compact+words),
TEST-CORE (2000–2999, canon; PRIMARY gate slice), TEST-TEMPLATE (same range,
words; unseen format), TEST-RANGE (100000–999999, canon), TEST-COMPOSITION
(words × shifted 3000–9999 range: format+magnitude combination unseen in
TRAIN — genuine composition probe, not a relabeled range shift). Leakage: exact + commutative-key +
operand-pair + template-key checks on all eval slices + train audit sample;
all must be 0 (else IMPLEMENTATION_FAILURE, no training). Designed exception,
informational only: TEST-CORE × TEST-TEMPLATE commutative keys overlap by
construction — the transfer probe IS "same facts, unseen format" (exact
strings still cannot collide).

## Splits observational accounting

TEST-family observed per arm: untrained baseline (4 slices) + trained final
(4 slices) = 8 frozen observations × 4 arms = 32 new. Plus T1's 2 and T1B's 0
executed. All reported. DEV drives nothing (budgets fixed).

## Metrics per arm (§20 set)

Train/answer/prompt CE, train exact, DEV exact, all 4 TEST exacts (Wilson),
stop histogram, NON_ALPHABET/empty/numeric rates, prediction-length
distribution, 20 deterministic samples, prediction SHA, grad-norm stats,
mutation proof, tok/s, compile/wall times, unique-vs-scheduled-vs-replay
counts. Reload-hash gate per arm.

## Cross-arm rules (CROSS_ARM_SUMMARY.json, machine-evaluated)

Per-arm PASS = T1 5-rule gate adapted (own baseline, TEST-CORE, own nulls).
CAPABILITY_LEARNED: any arm passes. OBJECTIVE_LIMITED: B−A test ≥ 0.10 with
non-overlapping intervals. DATA_LIMITED: B−C ≥ 0.10 likewise. SCALE_LIMITED:
D−B ≥ 0.10 likewise. MEMORIZATION: any arm train−test ≥ 0.30 with test < 0.05.
FORMAT_FAILURE: any arm (NON_ALPHABET+empty) > 50% with test < 0.05.
BUDGET_LIMITED: test still rising ≥ 0.05 between 50%→100% checkpoints with
test < 0.10. Else INCONCLUSIVE. Multiple labels may fire; a FAILING suite
that eliminates hypotheses is success (§38).

## Session mechanics

Throughput calibration first (3 shapes, few steps each, correctness +
accounting verified; max tok/s wins; recorded). Budgets frozen at 8M/arm with
one deterministic rule: if calibrated MINI rate < 5000 tok/s, all budgets
halve (recorded; no peeking). Ceiling <2 TPU-h, target ~45–90 min. Resume via
per-arm receipt+checkpoint+marker (hash-verified skip; TEST accounting
preserved). Per-arm infra failure → error receipt, continue; ≥2 → abort.
Bundle `CITADEL_T1C_RESULTS.zip` (manifests, calibration, 4 arm receipts,
summary; checkpoints exported separately if large).

## Interpretation

At most: this setup learns/fails controlled arithmetic transformations under
these conditions. No C1, no AGI claims. T1B stays preregistered-but-unexecuted
(preserved; T1C answers its question with cleaner contrasts).
