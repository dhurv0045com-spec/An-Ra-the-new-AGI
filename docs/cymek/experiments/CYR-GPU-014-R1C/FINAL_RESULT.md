# CYR-GPU-014-R1C — FINAL RESULT

**Execution status:** `COMPLETE`  
**Verdict:** `SOFTMAX_COMPETITION_NOT_SUFFICIENT`  
**Completed arms:** 24/24 = 4 matched seeds × 6 treatments × 3,000 updates  
**Claim ceiling:** `CONTROLLED_DEVELOPMENT_MECHANISM_ONLY`

## Source provenance

Operator result bundle: `CYMEK_R1C_SOFTMAX_MECHANISM_RESULTS.zip`  
Bundle SHA-256: `2a9359e49f792f962774e42b4e5825b93fb7c91ddcad8f530c1c6f16c6f9bf9e`  
`DECISION.json` SHA-256: `abfe0ca730e0b2a50e5d1004c2fe7d061dcd2162e4e5271c1dac0db0b985e389`  
`CAMPAIGN.json` SHA-256: `d05a11d4a52affc44db85b0194fa05237a7fd5eeda2e54d189ae9659f1554dd0`

The bundle records CUDA execution on a Tesla T4 with deterministic algorithms enabled and TF32 disabled. The frozen data split is `0dd9305697045b0fbf4e7f268b46a4d7276e4794af5d78b60e999df914ae4236`.

## Preregistered primary test

Primary pair: `MASK_4096` versus `FULL_24576`, with the physical 24,576-way tied embedding/output matrix held fixed. The hypothesis was that removing inactive-class softmax competition at training time would materially rescue formation.

Observed paired formation-AUC gaps (`MASK_4096 - FULL_24576`) across the four matched seeds:

`[-0.139792, -0.242215, -0.019377, -0.040138]`

Mean paired gap: **-0.110381**.

`MASK_4096` sustained >50% on only **1/4** seeds; `FULL_24576` did so on **0/4**. The preregistered positive mechanism threshold was not met. Both the functional full-vocabulary endpoint and the structural active-only diagnostic therefore return `supported: false` / `not_sufficient: true`.

## Across-arm functional summary

| Arm | Mean formation AUC | Mean endpoint | Mean peak-3 | Seeds with sustained >50% |
|---|---:|---:|---:|---:|
| FULL_24576 | 0.210035 | 0.297059 | 0.412745 | 0/4 |
| MASK_16384 | 0.181315 | 0.217647 | 0.291176 | 1/4 |
| MASK_19 | 0.151211 | 0.070588 | 0.235294 | 0/4 |
| MASK_8192 | 0.140657 | 0.326471 | 0.317647 | 2/4 |
| MASK_4096 | 0.099654 | 0.141176 | 0.183333 | 1/4 |
| OFFSET_EQ4096 | 0.084429 | 0.076471 | 0.158824 | 0/4 |

The individual intermediate-mask successes are seed-dependent and do not rescue the preregistered mechanism claim. `inactive_partition_mass_rescue` is false for both functional and structural diagnostics.

## Scientific interpretation

R1C **falsifies the strong claim that inactive-softmax competition alone is sufficient to explain the earlier class-space formation effect**. A training-time mask applied inside the same 24,576 physical matrix does not reproduce the actual V4096 advantage seen in the earlier class-space experiments.

This pushes the causal search toward factors that R1C deliberately held fixed: physical embedding/output geometry, parameterization and initialization statistics, optimization geometry, representation/tokenization, or interactions among them. It does **not** prove that the 24,576-class production head is optimal, and it does **not** invalidate the earlier observation that physical class-space size changes formation.

The clean next test is therefore a real physical-class-space transfer experiment (`CS-TRANSFER-001`): matched V4096 versus V24576 (and only justified intermediate controls) on clean attack-screened tasks / production-tokenizer rendering, rather than another masked-softmax proxy.

## Authorization consequences

The final decision explicitly leaves all of these false:

- `production_tokenizer_change_authorized`
- `pre500m_authorized`
- `training_500m_authorized`
- `broad_reasoning_claim_authorized`
- `agi_claim_authorized`

So R1C closes one mechanism question, but **does not authorize scaling or a tokenizer/vocabulary change**.
