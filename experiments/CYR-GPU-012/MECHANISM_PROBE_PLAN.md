# CYR-GPU-012 MECHANISM PROBE — preregistration (POSTHOC, EVAL-ONLY)

Status at registration: PREREGISTERED_BEFORE_EXECUTION. No probe numbers observed.
Subject: FINAL compact checkpoint, model_sha256 94e82920eaf03089630b85b9061bd0fa680b629c0bcd526e58bf301712127a99.
No training, no optimizer state, no endpoint change. PRODUCTION_PROMOTION_FORBIDDEN.

## Motivation and prior observations (declared, not hidden)

The full-exposure run scored 52/52 both-exact on COMMUTATION_MATCHED_BAND (same-band
pairs, ones-sum<=9, excluded from all original partitions) while final controller was
0/64 and STANDARD 0/85. Prior spacing diagnostic (6 prompts, post-hoc, already
published in RESULT.md) saw the four sampled in-band examples break without spaces.
The original CARRY and other structural families were already evaluated. F1/F2 below are newly specified families; prior results make this a post-hoc diagnostic, not independent confirmatory evidence.

## Question

Is the in-band commutation success evidence of internal arithmetic structure, or of a
band-local no-carry mapping that never composes? "Cognition-like structure" here means
only: correct novel combinations requiring carry or cross-band operand composition.

## Families (constructed and frozen before execution)

F1 CARRY_MATCHED_BAND: unordered distinct pairs a<b, both operands in the SAME trained
band t in {1,2,3,4}, ones-sum>=10 (forces carry), total<=99. Excludes canonical pairs
present in ANY original partition (train/dev_controller/dev_measurement/sealed_reserved).
Evaluated in BOTH orders (base/reversed), same schema as COMMUTATION_MATCHED_BAND.

F2 CROSS_BAND_NO_CARRY: unordered distinct pairs a<b, a in band 1-2 (10-29), b in band
3-4 (30-49) — both trained bands, DIFFERENT bands — ones-sum<=9 (no carry), total<=99.
Same exclusions; both orders. Separates cross-band operand composition from carry.

F3 STANDARD_LOGIT_FORENSICS: for all 85 original STANDARD prompts, single forward pass;
record first-answer-position entropy (bits, base 2), top-1 margin (l1-l2), top-1 symbol,
and correctness label from the official generation. Classification only, no scoring gate.

F4 SPACING_ALL_STANDARD: all 85 STANDARD prompts evaluated with original spacing and
with spaces removed; exact-with-EOS rates for both variants.

F5 CONSISTENCY_CANARY (abort gate, not a result): reload checkpoint, replay the 253
registered rows (104 matched-band + 85 STANDARD + 64 controller) and require exact
match with saved full01 predictions at batch sizes 1 and 32. Any mismatch aborts the
probe as INVALID before F1-F4 results are examined.

## Scoring

F1/F2: paired content_consistency, eos_valid_consistency, both_exact_with_eos per
closure.score_pairs (constant-wrong yields consistency 1, both-exact 0).
F3/F4: descriptive distributions and rates.

## Interpretation (registered before execution)

- F1 both_exact >= .9: carry structure exists inside trained bands (would partially
  revise the lookup hypothesis; still not a G90 or generalized-arithmetic claim).
- F1 both_exact <= .1 with matched-band 52/52 intact: band-local no-carry mapping;
  the commutation success is consistent with memorized band tables, not composition.
- F2 both_exact >= .9: cross-band composition exists (unexpected given controller 0%).
- F2 both_exact <= .1: no cross-band composition.
- F3: report median entropy and margin separately for correct vs incorrect rows;
  sharp-wrong (low entropy, wrong top-1) vs indecision (near-uniform) is descriptive.
- F4: report both rates; large gap quantifies surface-format dependence.
All outcomes are POSTHOC_DIAGNOSTIC. No verdict change, no authorization change,
no cause established for the V11 trajectory discrepancy by this probe alone.

## Budget and safety

Eval-only on local RTX 4050 (idle 0 MiB, <=51C preflight). Fewer than ~600 generation
rows total, 8-token greedy cap, batch 32; expected wall < 2 minutes. Abort if GPU
temperature >= 85C or free RAM < 2 GiB. Output outside git at
C:/Users/ankit/cyr012-evidence/probe01/MECHANISM_PROBE.json. Focused tests
(contamination exclusion, band rules, determinism, entropy reference values) must pass
before execution.
