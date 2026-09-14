# FORMATION-MUX-001 — Amendment 1 (PRE-EXECUTION)

Status: **PROSPECTIVE / NO OFFICIAL GPU OUTCOMES OBSERVED**

Supersedes the executable details of science commit `36ab16a4950d582fcc26b239dfc8c0ba816911bb` where they conflict with this amendment. The earlier commit remains preserved as the failed pre-execution design record.

## Why this amendment exists

A live pre-execution audit found defects that would either invalidate the intended causal contrasts or prevent the Kaggle launcher from running correctly. No official FORMATION-MUX-001 arm has been executed on Kaggle T4 x2, and no sealed FORMATION-MUX-001 outcome has been observed.

## Repairs frozen before execution

1. **Shared/extra-row boundary**
   - shared/active rows are `0..4095`;
   - extra rows are `4096..24575`;
   - every latent grammar/content/answer token remains inside the shared region.
   - The earlier `0..127` boundary is invalid and must not be used.

2. **Optimizer / global clipping order**
   - frozen-row gradient masking occurs after backward and before the global clip;
   - the row-aware embedding AdamW step occurs only through the optimizer view called by `ProductionTrainingBackend.finish_update()`;
   - therefore the tied embedding update uses the same globally clipped gradient boundary as the rest of the model.

3. **CS-MECH-002 primary measurement**
   - primary formation metric is computed on the `identity` family only;
   - per-family behavior remains secondary diagnostics.

4. **REP-FORM-003A exposure matching**
   - equal optimizer-step count is retired as the primary matching rule;
   - primary budget is fixed processed non-padding token positions per arm;
   - full examples remain indivisible; the endpoint may overshoot the target by at most one batch;
   - the actual processed-token discrepancy must remain within the preregistered tolerance or the comparison is `INCONCLUSIVE_EXPOSURE_MISMATCH`;
   - formation evaluation is indexed by processed-token exposure, not raw update count.

5. **Sealed finalization**
   - sealed scoring uses `identity exact + valid EOS` as the endpoint;
   - CS-MECH-002 reports all three preregistered contrasts separately;
   - final paired verdicts combine the frozen development formation-AUC deltas with one-shot sealed endpoint gaps;
   - sealed endpoint scores are never substituted for formation AUC.

6. **Checkpoint evaluation restore**
   - sealed evaluation restores verified model weights directly from the arm checkpoint;
   - optimizer topology is not reconstructed merely to score a checkpoint.

7. **Kaggle launcher identity**
   - the canonical notebook must fetch/verify the later operator commit separately from the frozen science commit;
   - it must not check out a science commit and then attempt to execute an operator that does not exist in that commit.

8. **Calibration**
   - calibration uses dedicated non-scientific seeds;
   - both T4s are exercised concurrently;
   - train/evaluation/checkpoint timings are measured from the executable rather than replaced with fixed guessed constants;
   - calibration cannot write an official verdict.

9. **Tokenizer requirement**
   - official REP-FORM-003A execution requires the frozen production V=24,576 tokenizer to load successfully before science starts;
   - missing `r0_*` token IDs are a global pre-science failure, never a silent fallback.

## Scientific question unchanged

The campaign still contains two independent experiments in one operational run:

- `CS-MECH-002`: isolate extra-row weight decay, extra-row trainability/update evolution, and denominator participation at fixed physical V=24,576.
- `REP-FORM-003A`: compare production-BPE versus isomorphic rendering at fixed physical V=24,576.

The two experiments keep separate preregistrations, RNG namespaces, development aggregates, sealed identities, verdicts, and claim ceilings.

## Claim ceiling

This amendment authorizes only development-scale mechanism/representation evidence on the declared 8L/256w surface. It does not authorize a production vocabulary change, PRE500M, 250M/500M training, general cognition claims, or AGI claims.
