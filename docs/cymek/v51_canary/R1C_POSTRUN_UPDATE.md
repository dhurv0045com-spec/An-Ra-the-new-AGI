# R1C POST-RUN UPDATE FOR V5.1 CANARY

**Date:** 2026-09-13  
**R1C status:** `COMPLETE` (24/24)  
**Verdict:** `SOFTMAX_COMPETITION_NOT_SUFFICIENT`  
**Bundle SHA-256:** `2a9359e49f792f962774e42b4e5825b93fb7c91ddcad8f530c1c6f16c6f9bf9e`

R1C finished after the V5.1 canary branch was designed. The preregistered primary `MASK_4096 - FULL_24576` formation-AUC gaps were `[-0.139792, -0.242215, -0.019377, -0.040138]`, mean `-0.110381`. Functional and structural primary tests both say inactive-softmax competition is not a sufficient explanation of the earlier physical class-space effect.

## Effect on this branch

The existing canary result is **not invalidated**. Task 3 intentionally used the canonical full-softmax path and prohibited EXPERIMENT_ONLY masking from leaking into the default execution path. R1C strengthens that conservative boundary: there is no evidence to promote a masked-4096 output treatment into the canary default.

The canary's `CANARY_FAIL_FORMATION` verdict also remains exactly what it was: an execution/formation result for the specified development instrument, not an output-space mechanism experiment.

## What changes for canary-v2 / later work

- Keep full canonical output behavior as the default unless a new preregistered experiment says otherwise.
- Do not add MASK_4096 as a production fix for the failed canary.
- Do not interpret R1C as proof that physical 24,576 is optimal.
- The next representation decision should come from actual physical-class-space transfer (`V4096` vs `V24576`) on clean attack-screened / production-tokenizer surfaces.
- PRE500M and 500M remain unauthorized.

This note is informational and does not rewrite the frozen Task-3 canary executable or its preregistered result.
