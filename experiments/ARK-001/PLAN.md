# ARK-001 — MICRO LIFT-OFF MAPPING (preregistered before execution)

Status: PREREGISTERED. Written before any training run. Metrics frozen below.

## Objective

Measure the FIRST exact-match lift-off point — dose (steps / exposures / tokens)
at which train exact-match rises from zero — as a function of task simplicity,
vocabulary representation, and model scale, on the simplest symbolic tasks, with
answer-only CE. No branch has ever measured this (citadel T1C: train exact ≈ 0
everywhere; open question 3 "at what scale does train exact first lift off").

## Prior knowledge (branches)

- citadel T1/T1C: loss 10.1→1.3-1.9 but train AND test exact ≈ 0 across
  objectives, corpora (6.5M rich / 4k narrow), 1.6M→3.7M scale, 4M tokens (TPU).
- T1D designed but unexecuted; its arms (curriculum/teacher/scale) presuppose
  lift-off is reachable within their budget on mixed tiers — unmeasured.
- cymek: certified training substrate; no capability claims.

## Novelty gap

No branch measured: (a) lift-off existence/dose on a single trivial family,
(b) per-position digit accuracy decomposition, (c) vocabulary contrast.

## Hypotheses (competing)

- **H-FLOOR** (H1/H2/H9): even single-digit addition fails to reach high train
  exact at micro scale within a generous budget → the pathology is upstream of
  curriculum/teacher; T1D arms B/C would be premature.
- **H-GRADIENT** (H3/H4/H6): single-digit lifts off quickly with flat CE; dose
  explodes crossing into two-digit no-carry → the tier boundary is the real
  bottleneck; T1D arms B/C target it correctly.
- **H-REPR** (H5): a compact task vocabulary reaches lift-off at far lower dose
  than the frozen 24,576 byte-level vocabulary at matched decoder size.

## Arms (one variable per contrast; seeds fixed before run)

All arms: decoder-only transformer, 4 layers, width 128, 4 heads, ffn 512,
RoPE 10000, RMSNorm(1e-5), tied embedding, no bias, dropout 0, AdamW
(0.9/0.95/1e-8, wd 0.1), answer-token-only CE, greedy exact-match eval.
Prompt surface: canon template `a + b = c` (citadel-compatible).

1. **T1-COMPACT** — single-digit add (a,b ∈ 0..9; all 100 combos), compact
   vocab (17 symbols), pool of 100 rows replayed; dose-response to repetition.
2. **T1-BYTE** — identical, but byte-level 24,576-vocab embedding (frozen cymek
   tokenizer). Parameter overhead recorded as part of the representation cost.
3. **T2-COMPACT** — two-digit no-carry add, band-split: train tens{1..5}+ones
   constraints, test tens{8..9} (structural holdout, citadel-style bands).
4. **T0-COMPACT** — trivial tier (x+0, x*1), 6000-value band split, absolute
   floor sanity.
5. **T1-COMPACT-LARGE** — T1 task, 2x width (256): micro-scale sensitivity.

## Metrics (frozen)

- train exact (greedy, on the trained pool), test exact (band-held-out where
  defined), per-position digit accuracy, loss, steps and tokens and exposures
  at first lift-off (train exact ≥ 0.9), throughput, parameter count.
- Baselines BEFORE model results: majority-answer frequency; input-independent
  answer predictor (predict marginals from BOS only); lookup-table ceiling (100%).

## Predictions

- H-FLOOR: arms 1/4 stay < 0.9 train exact after full budget.
- H-GRADIENT: arm 1 lifts off at ≤ 2k steps; arm 3 does not lift off within
  budget (dose explodes across the tier boundary).
- H-REPR: arm 2 needs ≥ 3x the dose of arm 1, or fails within budget.

## Falsification / stop conditions

- Any arm hitting its 12-minute wall-clock box aborts that arm (recorded).
- If arm 1 fails where arm 5 succeeds → scale sensitivity at micro (record dose
  ratio); if both fail → H-FLOOR up; if 1 and 3 both lift off → H-GRADIENT down,
  widen the task boundary in ARK-002.

## Budget

Local CPU (shared venv, torch 2.14). ≤ 12 min wall per arm, ≤ 60 min total.
No TPU. No long training. Seeds: 13 (all arms); second seed 29 only for any arm
whose verdict decides a hypothesis (replication).

## Novelty test

Result counts as NEW for this program if the lift-off dose (or its nonexistence
at generous dose) on the simplest family is measured here for the first time —
citadel's receipts contain no such measurement. It is a REPRODUCTION only if it
merely re-confirms "exact stays 0" without the dose/per-position/vocab
decomposition.

## Next decision table

- Arm 1 lifts off, arm 3 does not → ARK-002: dose-response across the T1→T2
  boundary (digit count sweep) to locate the threshold; informs T1D curriculum.
- Arm 1 fails → ARK-002: optimization pathology triage (LR sweep, init, vocab
  reduction, answer-length 1) before any TPU session.
- Arm 2 ≫ arm 1 dose → ARK-003: representation reduction for symbolic training.
- Any lift-off → mechanistic probes (per-position curves already collected).
