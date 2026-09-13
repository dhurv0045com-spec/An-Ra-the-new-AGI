# R1C POST-RUN UPDATE — V5.1 CONSEQUENCES

**Date:** 2026-09-13  
**Experiment:** `CYR-GPU-014-R1C`  
**Status:** `COMPLETE`, 24/24 arms  
**Verdict:** `SOFTMAX_COMPETITION_NOT_SUFFICIENT`

Source bundle SHA-256: `2a9359e49f792f962774e42b4e5825b93fb7c91ddcad8f530c1c6f16c6f9bf9e`.

## Result that matters for V5.1

R1C held the physical tied embedding/output matrix at 24,576 classes for every arm and altered training-time output competition. The preregistered primary `MASK_4096 - FULL_24576` formation-AUC gaps were:

`[-0.139792, -0.242215, -0.019377, -0.040138]`

Mean gap: `-0.110381`. `MASK_4096` sustained >50% in 1/4 seeds; `FULL_24576` in 0/4. Functional and structural primary tests both mark the mechanism unsupported / not sufficient. The inactive-partition-mass rescue test is false.

## Architecture consequence

This result **does not justify Candidate B / masked-output promotion**. The strong hypothesis that inactive-softmax competition alone carries the earlier physical class-space effect is falsified.

Candidate A therefore remains the conservative V5.1 canary architecture: full canonical 24,576 output path, no experimental mask in the default training contract.

However, R1C also **does not prove that 24,576 is the optimal physical vocabulary/output geometry**. The earlier class-space experiments changed the actual tied matrix size; R1C did not. Vocabulary size / physical output geometry therefore remains blocked pending a true transfer experiment with actual physical V4096 versus V24576 geometry.

## Decision tree update

Before R1C:

- if masking reproduced V4096 formation, a masked-output mechanism could become a live candidate;
- otherwise the search moved toward physical geometry / parameterization / representation.

After R1C:

- the masking branch is rejected for production promotion;
- the live next branch is `CS-TRANSFER-001` with actual physical class-space variation on clean attack-screened / production-tokenizer transfer surfaces.

## What stays blocked

R1C explicitly does not authorize:

- production tokenizer/vocabulary change;
- PRE500M promotion;
- 500M training;
- broad-reasoning claim;
- AGI claim.

The V5.1 overall verdict remains `READY_FOR_CANARY`, not `READY_FOR_500M`.
