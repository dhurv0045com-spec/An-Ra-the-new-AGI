# FAILURE TAXONOMY

**Phase 2 · 2026-09-13.** All important failures reclassified; the right-hand column names the wrong conclusion a mislabel would produce.

Classes: `ENGINEERING_FAILURE` · `OPTIMIZATION_FAILURE` · `DATA_FAILURE` · `REPRESENTATION_FAILURE` · `EVALUATION_FAILURE` · `GENERALIZATION_FAILURE` · `RETENTION_FAILURE` · `SCALING_FAILURE` · `MECHANISM_FALSIFICATION` · `INCONCLUSIVE_FAILURE`.

## A. Engineering failures (no scientific content — ever)

| Failure | Class | Mislabel risk if wrong |
|---|---|---|
| CYR-GPU-004 fork-contract defect (no parent restore) | ENGINEERING_FAILURE | "retention policies don't work on V5" — would have poisoned the whole retention line |
| CYR-GPU-007 decision-wrapper self-recursion | ENGINEERING_FAILURE | same |
| CYR-GPU-008 all-or-nothing wall rejection | ENGINEERING_FAILURE | "V5 can't retain under Colab constraints" |
| R1C optimizer-constructor TypeError | ENGINEERING_FAILURE | "R1C science failed pre-execution" |
| R1C CLIP_BREACH float32 reduction-order abort | ENGINEERING_FAILURE | "softmax masking destabilizes training" — dangerously seductive mislabel; the abort was numerics, not the treatment |
| ARK-020 V4 A1 phase-boundary resume crash + partial-identity resume | ENGINEERING_FAILURE | "Guardian arms crash = controller broken" |
| T4 CUDA SDPA backward nondeterminism | ENGINEERING_FAILURE | "exact-resume impossible on GPU" |
| XLA per-microstep all-reduce gradient scaling | ENGINEERING_FAILURE | "multi-replica training diverges" |
| Cursor/counter restart on resume; window-vs-loss mismatch; test-pack drift | ENGINEERING_FAILURE | false token-accounting → false dose conclusions |
| Demand-planner / resume-parent defects (found live) | ENGINEERING_FAILURE | silent curriculum corruption |
| ARK-001 ByteVocab answer-encoding bug | ENGINEERING_FAILURE | "byte vocab can't learn" (was refuted *with* the repair, not because of it) |

## B. Optimization failures (real training dynamics, real information)

| Failure | Class | Mislabel risk |
|---|---|---|
| Native BF16 AdamW moments clip-norm overshoot | OPTIMIZATION_FAILURE | "BF16 unusable" (only the state layout is; compute parity passed) |
| EMA-0.999 / WD-removal fail to stabilize | OPTIMIZATION_FAILURE (feeding MECHANISM_FALSIFICATION) | "post-G90 decay is an engineering artifact" |
| 1e-4 LR insufficient to prevent collapse | OPTIMIZATION_FAILURE | "LR doesn't matter" |

## C. Data failures

| Failure | Class | Mislabel risk |
|---|---|---|
| Tiered arithmetic corpus: latest-position shortcut 1.000/tier; 530 verbatim cross-split docs; 13.5% duplication; 0.004× supply (CITADEL-DATA-001) | DATA_FAILURE + EVALUATION_FAILURE | **the dangerous one:** a future model exploiting the shortcut would read as "lift-off"; conversely T1D's null could be misread as "arithmetic is unlearnable at scale" when part of the surface is degenerate. Nulls survive; positives would not be interpretable |
| T1D teacher pool diversity exhaustion (10–13 unique rows replayed ~2,500–3,260×) | DATA_FAILURE | "teacher supervision doesn't help" vs "the teacher had nothing left to teach" |
| Production 5B corpus never materialized | DATA_FAILURE (blocker) | "the Core can't learn language" — no evidence exists either way |
| EXP v10/v11 stale-candidates contamination | DATA_FAILURE + EVALUATION_FAILURE | composition claims were fiction, not falsifications |

## D. Representation failures

| Failure | Class | Mislabel risk |
|---|---|---|
| Production 24,576 rep: M99 + 0% held-out + 0/48 sealed at 100% exposure | REPRESENTATION_FAILURE | "the model/data/objective is broken" — geometry/objective/data were controlled; representation is the implicated bundle |
| V19/V1024/V24576 ≈ 0 at 128k rows (both seeds) | REPRESENTATION_FAILURE (edge regimes) | "small vocab is best" |

## E. Evaluation failures (instrument wrong, not model wrong)

| Failure | Class | Mislabel risk |
|---|---|---|
| Calibrated scorers select fewest-token role 1.000 (15/15 CUDA cells) | EVALUATION_FAILURE | any assisted-selection "capability" measured before certification |
| Scoring fixture v1 structural leak | EVALUATION_FAILURE | bias-screen results pre-repair are void |
| E0 v0.3.0 positional shortcut (81.77% bag-of-words) | EVALUATION_FAILURE | generator-tier capability claims |
| COMMUTED=100% read as commutation invariance | EVALUATION_FAILURE | invariance claims from confounded probes |
| ARK-009 composite query+order diagnostic | EVALUATION_FAILURE | "query conditioning absent" claims from non-orthogonal probes |
| T1D self-knowledge probe contract invalid (57/96 targets > 8 tokens) | EVALUATION_FAILURE | "model lacks self-knowledge" |
| T1C prose 0/500 vs raw 0/1,000 denominators | EVALUATION_FAILURE | miscited effect sizes |
| Readiness gate v1 false green | EVALUATION_FAILURE | mechanism studies on a floor substrate |
| X1-REAL imbalance (always-negative 0.9733) | EVALUATION_FAILURE + MECHANISM_FALSIFICATION | "the model has a self-model" |

## F. Generalization failures (the science worked; the capability didn't transfer)

| Failure | Class | Mislabel risk |
|---|---|---|
| T1-series/T1D: no exact lift-off at 2–8M tokens under any arm | GENERALIZATION_FAILURE (bounded) | "arithmetic is impossible at these budgets" — the arms/budget scope must travel with the claim |
| BRAMASTRA fresh-world 17.2–25.8%, rendering 0/128 | GENERALIZATION_FAILURE | "the lab is broken" |
| BRAMASTRA depth-two ≈ one-step (Δ≈0–0.02) | GENERALIZATION_FAILURE (of the teaching intervention) | "deeper curricula are useless" — budget-2 scope |
| T1D teacher primitives 0.515 with composition ≈ 0 | GENERALIZATION_FAILURE | "primitives are useless" |

## G. Retention / mechanism failures

| Failure | Class | Mislabel risk |
|---|---|---|
| NARROW_HIGH erodes invariance 8/8 at canonical 1.0 | RETENTION_FAILURE | "the capability was never there" — canonical accuracy said it was |
| ARK-013 every arm loses T2 without replay | RETENTION_FAILURE | "LOW LR is useless" (it's a regime boundary, not a refutation of LOW protection) |
| ARK-016 1/12 qualifying events | INCONCLUSIVE_FAILURE | "caps don't work" (event starvation, not refutation) |
| ARK-012 TIME_NOT_STATE_SCREEN | INCONCLUSIVE_FAILURE | "0.85 is the threshold" |
| BRM "this-regime replay" rejected | MECHANISM_FALSIFICATION (provenance-weak, G17) | conflation with treatment-exact replay (which protects) would overgeneralize the null |

## H. Inconclusive by design (must stay inconclusive)

| Failure | Class | Mislabel risk |
|---|---|---|
| CYR-GPU-009 <22% exposure | INCONCLUSIVE_FAILURE | "V5 TINY can't generalize" — dose, not capability |
| ARK-019 V3.1 primary verdict (SKILL_B never formed) | INCONCLUSIVE_FAILURE for the controller question / GENERALIZATION_FAILURE for SKILL_B dose | "Guardians don't work" — the reference couldn't acquire either |
| learned-discovery n.s.; depth-two null | INCONCLUSIVE_FAILURE (underpowered scale) | "controllers/curricula are dead" |

## I. Scaling failures

| Failure | Class | Mislabel risk |
|---|---|---|
| T1C arm D 2.3× scale: still 0/1,000 | SCALING_FAILURE (bounded, confounded) | "scale never helps" — POSTMORTEM 5's budget confound blocks exactly that |
| Corpus supply 0.004× of 500M demand | SCALING_FAILURE (readiness) | "500M is ready to schedule" |
