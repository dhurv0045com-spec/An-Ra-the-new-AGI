# FORMATION-MUX-001 — Amendment 1B (PRE-EXECUTION clip isolation)

Status: **PROSPECTIVE / NO OFFICIAL FORMATION-MUX GPU OUTCOMES OBSERVED**

A second code-level audit of the repaired S2 implementation found one avoidable coupling in the `M1_EXTRA_NO_DECAY` versus `M2_EXTRA_FROZEN` contrast: S2 zeroed the frozen extra-row gradients **before** the global gradient clip. That changed the clip norm seen by shared/core parameters at update 1 even when all parameter bytes were still matched, so the contrast moved both (a) extra-row update eligibility and (b) the global clip denominator.

This amendment removes that coupling before execution.

## Frozen intervention

For `M2_EXTRA_FROZEN`:

- extra rows 4096..24575 remain physically present and participate in the full training softmax denominator;
- autograd is allowed to compute their gradients;
- those gradients participate in the same whole-model global L2 clip as M1;
- the row-aware optimizer owns only rows 0..4095, so extra rows receive **no parameter update, no Adam moment evolution, and no weight decay**;
- extra rows therefore remain byte-identical to initialization despite their diagnostic gradient being retained through the clip.

For `M3_EXTRA_FROZEN_MASKED` the same optimizer freeze applies, while the declared denominator intervention removes extra logits from the training denominator. Any resulting difference in gradient field / clip norm is treated as a downstream consequence of the denominator intervention, not a second direct treatment.

## Consequence for causal interpretation

- `M0 - M1`: direct intervention = extra-row weight decay.
- `M1 - M2`: direct intervention = extra-row optimizer/update eligibility (including Adam-state evolution), with global clipping semantics held common.
- `M2 - M3`: direct intervention = extra-row denominator participation.

No official science has run under S1 or S2. Science Commit S3 must bind this amendment before Kaggle execution.
