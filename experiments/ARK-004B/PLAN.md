# ARK-004B — COLUMN-CONSISTENCY INTERVENTION (preregistered design; plan-only commit)

Gate basis: ARK-004A Case A — tens-column selectivity is a SUPPORTED (TENTATIVE,
n=4) precursor that predicts G90 timing (LOO 4/4, beats time/loss baselines).
This plan is committed BEFORE execution; a later run executes it verbatim.

## Hypothesis (evidence-derived, neutral name)
If early tens-column factorization causes earlier structural generalization,
then training that directly encourages consistent, column-local behavior under
counterfactual perturbation should shorten the post-memorization delay
(post_mem_delay_90) without hurting final OOD.

## Arms (identical model/vocab/optimizer/data/steps; ONLY the objective differs)
- A BASELINE: answer-only CE (exact ARK-002B config, frozen manifest).
- B COLUMN-CONSISTENCY (aligned intervention): CE + lambda * consistency loss.
  For frozen in-band counterfactual pairs (ones-perturbed and tens-perturbed),
  the model is trained so that (i) the changed column's prediction matches the
  perturbed target and (ii) the unchanged column's prediction distribution
  stays close to the clean row's (KL term). lambda frozen before execution;
  pairs drawn only from TRAINING-band rows (never the OOD test band).
- C SHAM CONSISTENCY (misaligned control): identical machinery and compute,
  but the consistency pairs are rotated (perturbation applied to a DIFFERENT
  row than the prediction target), breaking alignment.
- D WEIGHT-DECAY CONTROL (known simple control): baseline CE with the best
  justified simple alternative if preregistration-time evidence supports one;
  otherwise omitted and recorded as omitted.
Extra forward passes are limited to the SAME counterfactual batch for B and C
(compute matched); steps/tokens/wall all recorded.

## Accounting (non-negotiable; corrects ARK-003's confound)
Primary endpoint tokens_to_G90 and post_mem_delay_90 measured under
STEP-MATCHED runs (all arms same optimizer-step count; wall time and
model-token equivalents reported on separate axes). Success requires >=2x
delay reduction on >=2 fresh seeds (preregistered seed list at execution-time
commit), final S0 OOD within 5pp of A, no memorization regression.

## Causal-credit test (Triquetra discipline)
The intervention claims: precursor moves earlier AND capability moves earlier,
and across runs the precursor shift predicts the capability shift. If
capability improves without the precursor moving, effect = real,
explanation = not established.

## What this does not justify
No AGI claim, no core promotion (mission section 12 gate), no architecture
change. A null result writes Case B's negative and moves to replay/
weight-decay/unique-data controls.
