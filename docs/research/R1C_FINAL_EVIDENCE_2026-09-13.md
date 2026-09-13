# CYR-GPU-014-R1C — FINAL EVIDENCE UPDATE

**Date:** 2026-09-13  
**Experiment:** `CYR-GPU-014-R1C`  
**Status:** `COMPLETE` (24/24 arms)  
**Verdict:** `SOFTMAX_COMPETITION_NOT_SUFFICIENT`

## Provenance

Operator bundle: `CYMEK_R1C_SOFTMAX_MECHANISM_RESULTS.zip`  
Bundle SHA-256: `2a9359e49f792f962774e42b4e5825b93fb7c91ddcad8f530c1c6f16c6f9bf9e`  
`DECISION.json` SHA-256: `abfe0ca730e0b2a50e5d1004c2fe7d061dcd2162e4e5271c1dac0db0b985e389`  
`CAMPAIGN.json` SHA-256: `d05a11d4a52affc44db85b0194fa05237a7fd5eeda2e54d189ae9659f1554dd0`  
Data split SHA-256: `0dd9305697045b0fbf4e7f268b46a4d7276e4794af5d78b60e999df914ae4236`

Execution environment recorded by the bundle: Tesla T4, CUDA, deterministic algorithms enabled, TF32 disabled.

## What was tested

All six arms retained the same physical 24,576-way tied embedding/output matrix. The experiment varied training-time output treatment:

`FULL_24576`, `MASK_19`, `MASK_4096`, `MASK_8192`, `MASK_16384`, `OFFSET_EQ4096`.

Four matched model/order seeds were run for 3,000 updates per arm: 24 mandatory arms / 72,000 optimizer updates total.

The preregistered primary question was whether inactive-class competition inside the full 24,576 softmax is sufficient to explain the earlier physical class-space formation effect.

## Primary result

Paired formation-AUC gaps (`MASK_4096 - FULL_24576`) by seed:

- seed 3711: `-0.1397923875`
- seed 3712: `-0.2422145329`
- seed 3713: `-0.0193771626`
- seed 3714: `-0.0401384083`

Mean gap: **`-0.1103806228`**.

`MASK_4096` sustained >50% on 1/4 seeds; `FULL_24576` on 0/4. The preregistered mechanism threshold was not met on either the functional full-vocabulary endpoint or the structural active-only diagnostic. `inactive_partition_mass_rescue` is false for both.

Across-arm functional mean formation AUC:

| Arm | Mean AUC |
|---|---:|
| FULL_24576 | 0.210035 |
| MASK_16384 | 0.181315 |
| MASK_19 | 0.151211 |
| MASK_8192 | 0.140657 |
| MASK_4096 | 0.099654 |
| OFFSET_EQ4096 | 0.084429 |

Intermediate masks occasionally form on individual seeds, especially MASK_8192, but the behavior is heterogeneous and does not support the preregistered `MASK_4096` mechanism claim.

## Belief update

**Falsified / materially downgraded:**

> The earlier physical V4096 advantage is primarily caused by inactive classes competing in the full softmax denominator during training.

R1C shows that removing that competition while preserving the physical 24,576 matrix does not reproduce the physical V4096 result.

**Not falsified:** physical class-space size affects formation. R1/R1B changed the actual tied matrix geometry; R1C did not.

The remaining causal candidates therefore move toward physical embedding/output geometry, parameter count and initialization statistics, optimization geometry, representation/tokenization, or interactions among those variables.

## Decision consequences

1. Do **not** promote masked 4096 output training as a production mechanism.
2. Do **not** conclude that FULL_24576 is optimal; R1C was a sufficiency test, not an optimality proof.
3. The next decisive representation experiment should change the **actual physical class space** and test transfer on clean surfaces: `CS-TRANSFER-001` / matched V4096 versus V24576, with production-tokenizer rendering or another attack-screened task.
4. A tokenizer/vocabulary change remains unauthorized until transfer evidence exists.
5. PRE500M and 500M remain unauthorized. The bundle explicitly records `production_tokenizer_change_authorized: false`, `pre500m_authorized: false`, and `training_500m_authorized: false`.
6. Claim ceiling remains `CONTROLLED_DEVELOPMENT_MECHANISM_ONLY`; no broad-reasoning or AGI claim is authorized.

This update supersedes earlier repository text that described R1C as `NOT_EXECUTED` or as launch-blocked.
