# ARK-003 — GENERALIZATION ACCELERATION (preregistered; PLAN committed+pushed before training)

## Central question
Can structured training cause structural-band OOD computation to emerge with
substantially fewer post-memorization tokens than flat answer-only CE?

## Task and dataset (frozen)
T2 no-carry two-digit addition, ARK-002B frozen manifest
(split_sha256 0dd930569704..., 500 train / 197 test, zero commutation overlap).
TASK_MANIFEST.json in this directory binds the split. Generator audit in the
manifest's `semantics` block (bands, cardinalities, ones-pair overlap note).

## Arms (identical Micro 4L/128w model, identical 20-token vocab, identical
optimizer/budget/steps/box; ONLY the training stream differs)
- A FLAT: answer-only CE on the frozen T2 pool (baseline; identical config to
  ARK-002B runs).
- B CURRICULUM: first 25% of steps draw from the frozen T1 pool (single-digit
  add, 100 combos), remaining 75% from T2. Same total steps.
- C ALIGNED TEACHER: 40% of drawn rows carry a decomposition suffix
  `; <ua>+<ub>=<sum> ; <ta>+<tb>=<tens-sum>` appended after the answer,
  supervised. Decomposition is mechanically derived from the SAME row.
- D UNALIGNED EASY-DATA: same 40% suffix rate, same format, but the suffix
  digits are drawn from a DIFFERENT random training row (rotationally offset),
  breaking alignment while matching format/length/token statistics.
Vocabulary: ';' added for ALL arms (20 tokens) so parameter counts are equal.
Token accounting: supervised-token counts recorded per arm; all arms run the
same number of optimizer steps on the same batch size.

## Racing design (preregistered)
Stage 1 screen: all four arms, screening seed 29 (init 29 / order 29).
Stage 2 replicate: arms whose tokens_to_G90 or post_mem_delay_90 improve >= 2x
over A with final OOD within 5pp of A are re-run on fresh seeds 47 and 101.
No post-result seed selection.

## Primary endpoint
tokens_to_G90 (sustained G90: >=3 consecutive evals >= 0.90) and
post_mem_delay_90. Secondary: M99, G50, G95, OOD-AUC after M99.

## Transfer suite (behavioral, preregistered)
- S0: historical band OOD (manifest test set) — every eval.
- S1 COMPOSITION: test-band rows whose ones-pair (ua,ub) is absent from the
  train ones-pair set — evaluated at phase checkpoints (M99/G90/final).
- S2 LENGTH TRANSFER: 3-digit no-carry rows (hundreds digit 1..8, all columns
  no-carry; elementary column facts covered by training) — at phase checkpoints.
  Failure here is reported honestly as mechanism scope.
- S3 COUNTERFACTUAL LOCALITY: perturb only ub (ones of b) within no-carry
  constraint; score = fraction where the ones output adapts correctly AND the
  tens output is invariant. Computed every eval on a fixed 100-row set.
- Developmental phases: model state saved at M99 / G50 / G90 / final
  (checkpoints/<arm>/<phase>.pt, not committed); behavioral readouts recorded
  in the trajectory at every eval.

## Success criterion (preregistered)
>= 2x reduction in tokens_to_G90 or post_mem_delay_90 vs A, with final S0 OOD
within 5pp of A, equal parameters, equal optimizer steps, no leakage. One-seed
results are SCREENING only; REPLICATED requires stage 2.

## Red team (post-results)
Independent adversarial review: leakage, seed luck, checkpoint cherry-picking,
token accounting asymmetry, teacher-answer leakage (suffix contains the gold
answer digits — C sees the decomposition OF ITS OWN ROW; D sees the same volume
of decomposition tokens for other rows; the eval never includes suffixes),
format effects, early stopping asymmetry.

## Novelty discipline
Grokking, curriculum, and decomposition supervision are known in the
literature. New-for-program content: measured causal effect of aligned vs
unaligned decomposition on the post-memorization delay at the identified
binding bottleneck, with counterfactual locality as the behavioral signature.
