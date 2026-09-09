# CYR-GPU-010 — LONG 4L/128W CAPABILITY EMERGENCE + STRUCTURAL REASONING DIAGNOSTICS

## Status

EXECUTABLE CANDIDATE — do not run until a separate hash-bound preregistration commit exists and Cell 0 passes.

## Why this experiment exists

CYR-GPU-009 completed on Tesla T4 using the real Cymek V5 TINY proxy (2L/64w, 1,647,104 parameters). Both seeds reached near-perfect candidate-free train-probe accuracy but neither reached held-out candidate-free G90 by 2,000,000 real tokens / 15,632 optimizer updates. The retention fork correctly never executed. The immediate bottleneck is capability formation/generalization, not retention.

Live Arkenstone at `59e1b805b7d93b7f2e1e9d3ea66b34c4fabca9c8` changes the next question:

- ARK-002B replicated a memorize-first -> generalize-later transition on a 4L/128w Micro subject. Sustained G90 appeared around 12k steps for one fresh seed; another was still climbing at 18k.
- ARK-003 did not demonstrate acceleration from curriculum or teacher suffixes; flat training was the only arm with meaningful OOD lift in the wall box.
- ARK-015 demonstrated a strong non-arithmetic invariance-retention effect: NARROW_HIGH failed 8/8 while NARROW_LOW and AUGMENTED_HIGH failed 0/8.
- ARK-016 was inconclusive because the T2 event rate was too low.
- ARK-017 is preregistered but unexecuted; it is not evidence yet.

## Primary question

Does Cymek's real V5 4L/128w `RESEARCH_SMALL` proxy show the same delayed memorize -> structural-generalization transition inside the Arkenstone 9k–18k update regime with the production 24,576 tokenizer and candidate-free generation?

## Subject

Preferred proxy: canonical Cymek `RESEARCH_SMALL`: 4 layers, width 128, Q4/KV2, head dimension 32, FFN 512, context 512, QK norm, tied embeddings, real `v5_model.core.initialize()`, frozen production tokenizer.

Hardware-only fallback: if RESEARCH_SMALL cannot fit the 18k-update box inside 170 science minutes, use TINY for a 36k-update long-dose diagnostic. TINY fallback has a lower claim ceiling.

## Data and prior receipt

Use the exact existing Cymek T2 four-way split / manifest / contamination audit. STANDARD held-out first-operand tens bands are 6–7; training bands are 1–5. The returned V9 ZIP SHA256 is:

`dc15f14d3bc81551b1f0b00285faa4b23c9e68f1341405377959a7aba108f216`

## Acquisition

Fresh fixed seed `3101`.

- HIGH LR `1e-3`
- AdamW / real V5 backend unchanged
- batch rows 16
- RESEARCH_SMALL maximum 18,000 optimizer updates
- TINY fallback maximum 36,000 updates
- candidate-free eval every 200 updates
- total hard wall 175 minutes
- packaging reserve 5 minutes

Milestones: M99 train-probe >= .99; G50 DEV_CONTROLLER >= .50; G90 onset DEV_CONTROLLER >= .90; G90 confirmation = three consecutive DEV_CONTROLLER evals >= .90. Save immutable M99, G50, G90 and final checkpoints.

## Structural reasoning battery

Only `STANDARD` contributes to the held-out capability gate. Everything else is diagnostic.

1. `STANDARD` — frozen DEV_MEASUREMENT T2.
2. `COMMUTED` — same semantic facts with operands reversed; diagnostic only because the grammar has commutative closure.
3. `LOCALITY` — paired one-digit counterfactual interventions; score whether predicted sum changes by the exact same delta plus both-exact rate.
4. `RENDERING` — unseen natural-language surface form.
5. `THREE_DIGIT` — deterministic no-carry three-digit additions; length/compositional extrapolation diagnostic only.

Also report per-digit accuracy. This carries forward ARK-003's observation that counterfactual locality co-emerged with OOD lift-off.

## Optional post-G90 stress

Only if G90 is confirmed and hardware time remains. From the exact G90 model+optimizer checkpoint and identical semantic-example stream, run two HIGH-LR arms:

- `NARROW_HIGH`: canonical rendering only.
- `SUPPORT_HIGH_1OF16`: same semantic IDs and HIGH LR, but exactly one of each 16-row batch is rendered with operands commuted.

Stress steps are determined only from remaining wall and measured updates/sec, bounded to 1,200–3,000 steps per arm. This is an exploratory Cymek analogue of ARK-015's invariance-support finding; it does not assume ARK-017's unexecuted mechanism attribution.

## Outcomes

- `G90_WITH_COUNTERFACTUAL_LOCALITY`: sustained G90 plus final locality consistency >= .80.
- `G90_WITHOUT_STRONG_LOCALITY_EVIDENCE`: held-out exact generalization without strong locality evidence.
- `NO_SUSTAINED_G90_IN_BOX`: no sustained G90 by the fixed scale-matched box.

A positive result is still single-seed controlled arithmetic development evidence. It does not establish AGI, broad reasoning, TPU equivalence, natural-language benefit, or permission to change the 500M recipe.
