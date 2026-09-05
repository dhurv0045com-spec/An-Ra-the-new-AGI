# T1D — Preregistration: ARITHMETIC LIFT-OFF (floor vs curriculum vs teacher)

Status: **PREREGISTERED — NO RESULTS**. Branch `citadel`. New experiment.
T0 PASS, T1 FAIL, T1C 4×FAIL/INCONCLUSIVE stand as history (see
`docs/citadel/EXPERIMENTS_BRIEF.md`). Cymek runtime pin:
`298c91ac04f756f0833a7edcf63e73af3d5af688` (re-verified: only additive
tokenizer change since certification; T1D surface byte-identical).
No T1D results exist anywhere; nothing tuned on data.

## Question

Why does exact-match never lift off? One session discriminates H_FLOOR (below
model/budget floor even for memorization), H_CURRICULUM (flat multi-digit
exposure too hard; needs easy→hard), H_TEACHER (answer CE hides algorithmic
structure; needs digit/subproblem supervision). Optional H_REPR (99.85% of the
output vocabulary can never be correct) via diagnostic Arm E.

## Arms (one variable per contrast; all else frozen)

Same tokenizer, calibrated static shape (all arms), AdamW 0.9/0.95/1e-8 wd 0.1
LR 3e-4, fixed seeds (recorded), pinned runtime, single XLA device.

```text
ARM A — FLAT CONTROL:   MID 3.7M | answer CE | tiered pool, uniform sampling | 8M cap tokens
ARM B — CURRICULUM:     MID 3.7M | answer CE | same pool, frozen easy→hard schedule | 8M
ARM C — TEACHER:        MID 3.7M | answer CE + micro-teacher rows (60/40 row split; exact token split recorded per arm) | same schedule as B | 8M
ARM D — SCALE:          SCALE2 7.4M | answer CE | same schedule as B | 4M
ARM E — REPR DIAG:      MID 3.7M | answer CE, softmax masked to calculator alphabet | same as B | 4M
```

A vs B = curriculum. C vs B = teacher. D vs B = scale. E vs B = representation
(diagnostic only — never production-capability evidence). Budgets frozen;
auto-scale rule: if calibrated MINI rate < 5000 tok/s, all budgets halve
(recorded, no peeking). Per-arm 45-min wall time-box (abort arm, continue).

## Models

MID 3.7M (T1C-certified pattern). SCALE2 ~7.4M (new, same dense-decoder
family — NOT new architecture): layers 8, width 192, query_heads 12,
kv_heads 6, head_dim 16, ffn 384, vocab 24576, ctx 4096, rope 10000,
RMSNorm 1e-5, tied, QK-norm affine, no bias, dropout 0. Expected receipt total
7,378,368; enforced at runtime by cymek `assert_valid()` + receipt check
(mismatch aborts the arm). P35 explicitly out (unjustified for this question).

## Corpus: tiered difficulty ladder (indexed, deterministic, ~100–300 MB)

Generator `tiered_data` v1: pure `row(tier, i)` + teacher-row generator, O(1)
memory, hashable, resumable. Ops +−×÷ (exact division); 4 templates
(canon/compact/arrow/words); every row carries op/sizes/answer/tier/
carry/borrow/template/index/split metadata. Frozen tiers:
T0 trivial (x+0, x−0, x*1, x/1); T1 single-digit; T2 easy two-digit
(no-carry/no-borrow, 2d×1d mult, small exact div); T3 compositional two-digit
(carry/borrow, 2d×2d mult, larger div); T4 three/four-digit mixed.
Physical target ~100 MB unique (≈7M rows; supports more without re-design);
manifest in git (version, code hash, tier counts/hashes, bytes, dup rate,
max row length ≤ 64 asserted); corpus ephemeral. T0/T1 tiers are inherently
small (single-digit space is finite) — per-tier replay factors disclosed,
never hidden. T2's no-carry pair space is likewise small (audit dup ~0.29):
accepted and disclosed — T2 drills basics where repetition is normal, arm
receipts record exact per-tier consumption, and bulk uniqueness comes from
T3/T4. Splits use frozen operand BANDS per tier (BANDS table in code):
TEST rows come from ranges TRAIN never consumes, so T2+ held-out claims are
structural. Tiers 0/1 cannot be band-isolated (tiny spaces) — their slices
are labeled memorization probes; the zero-leak gate (`leakage_verdict`)
applies only to pairs with all tiers ≥ 2.

## Curriculum (frozen)

B/C/D/E schedule by % of arm budget: 0–15% tiers 0–1; 15–35% tiers 1–2;
35–60% tiers 2–3; 60–100% tiers 1–4 mixture (weights frozen in code).
Arm A samples tiers 0–4 uniformly from step 0 (same families, same budget).
No adaptation, no TEST peeking, no duration tuning.

## Teacher rows (train-only, never evaluated)

Canonical machine notation, short, generative-checked: `digadd 7 8 carry0 =
digit5 carry1`, `digsub 3 8 borrow0 = digit5 borrow1`, `singlemul 7 8 = 56`,
`divmicro 72 8 = 9`. Arm C: ~60% ordinary curriculum rows + ~40% teacher rows
by row count (exact token split recorded per arm). Prompts never
supervised anywhere (answer-only CE + teacher-target-only masks via the
production `eligible` seam; per-arm supervision ledger: prompt/answer/teacher/
padding/loss-bearing counts).

## Packing (deterministic, segment-isolated)

First-fit rows into fixed training sequences (calibrated L) with segment IDs;
loss never crosses example boundaries (production `packed_layout` semantics,
block-diagonal masks). Receipts track capacity/real/loss-bearing/padding
tokens + packing efficiency. Eval stays unpacked (one prompt per sequence).

## Lift-off curves (the primary readout, not one global zero)

Per arm, exact-match + Wilson + answer-CE on frozen slices: TRAIN/DEV/TEST ×
tiers 0–4, plus per-operation breakdown and 20 deterministic samples per tier.
FIRST_TRAIN_LIFT_TIER / FIRST_TEST_LIFT_TIER: easiest non-T0 tier with tier
accuracy ≥ 0.20 on a ≥200-row sample (LCB ≈ 0.15; frozen). Untrained baselines
measured on the identical slices. Heuristic nulls per TEST slice.

## Observation policy

DEV at 25/50/75/100% per arm (automated). TEST exactly once per arm at frozen
budget (5 arms → 5 new TEST-family observations + existing history; all
preregistered, all reported, Holm over per-arm PASS claims). No adaptive arms.

## Interpretation rules (machine-evaluated)

CURRICULUM_HELPED: mean(tiers1–4) B−A ≥ 0.15 with B ahead on ≥3 tiers.
TEACHER_HELPED: C−B same form. SCALE_HELPED: D−B same form.
BELOW_FIT_FLOOR: train exact < 0.05 on tiers 0–2 in ALL arms.
GENERALIZATION_LIMITED: any arm train−test ≥ 0.30 with test < 0.05.
COMPLEXITY_FRONTIER: tiers 1–2 ≥ 0.50 while tier 4 < 0.05 in any arm.
FORMAT_FAILURE: any arm with >50% non-content stops (NON_ALPHABET/PAD) and
test < 0.05 — generation broken, not a learning result.
BUDGET_LIMITED: DEV still rising ≥ 0.05 between the last two checkpoints with
pooled test < 0.10 in any arm (DEV-based: TEST is observed once by design).
CAPABILITY_LIFTED: any arm pooled-tiers-1–4 LCB > max(null, untrained UCB) +
0.10 with reload identity. REPRESENTATION_LIMITED: E−B ≥ 0.15 (diagnostic
only). Else INCONCLUSIVE. A failing suite that eliminates hypotheses is
success.

## Session mechanics

Calibration (few static shapes, correctness + accounting verified, max tok/s
wins) → manifest → arms A–E with per-arm receipt+checkpoint+marker (resume by
hash-verified skip; TEST accounting preserved) → cross-arm summary → one
bundle `CITADEL_T1D_RESULTS.zip` (+ binaries per cap rule). Per-arm infra
failure → error receipt, continue; ≥2 → abort. Ceiling <2 TPU-h, target
~60–110 min. Checkpoints outside git; receipts/hashes committed later.

## What T1D is not

No C1, no 5B work, no multi-device goal, no AGI claims. At most: this setup
learns/fails controlled arithmetic under these conditions.
