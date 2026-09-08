# CYR-GPU-003 — P35-SCALE LR-RETENTION PROTECTION REPLICATION

## Question
Does the LR-retention protection law (discovered at 0.8M micro scale,
ARK-006/007) hold at P35 scale (35.4M params) with the production
24,576-token vocabulary?

## Arms
- HIGH: lr = 1e-3 (continued from acquisition)
- LOW: lr = 1e-5 (applied at G90 confirmation)
Only variable: learning rate after G90.

## Model
Cymek P35 spec: 16 layers, width 384, 6 query heads, 3 KV heads, head_dim 64,
ffn 1024, vocab 24,576, context 4096. Parameters: 35,411,328. Tied embeddings.
QK-norm affine. No bias. Dropout 0.

## Data
Frozen ARK-002B manifest: 500 train / 197 test two-digit no-carry addition,
structural tens-band holdout, zero commutation overlap.

## Frozen parameters
- Acquisition seeds: 707, 808
- LR: 1e-3, batch 64
- Acquisition: max 16000 steps or G90 confirmation
- Post-confirmation: 8000 steps per arm
- Eval every 200 steps
- G90: 3 consecutive evals ≥ 0.90 (onset = first in streak;
  confirmation = eval at which streak becomes knowable)

## Metrics
RET90, RET50, GENERALIZATION_AREA, T_COLLAPSE_90, FINAL_OOD,
parameter displacement from G90 checkpoint.

## Verdict rules
- REPLICATED_PROTECTION: LOW RET90 > HIGH RET90 on both seeds;
  HIGH shows measurable decay (RET90 < 0.7 on at least one seed)
- INCONCLUSIVE: both arms stable or both decay identically
- REFUTED: LOW decays faster than HIGH

## What will NOT be claimed
No AGI claim. No universal law. No production schedule recommendation
until replicated at P35 scale and tested on non-arithmetic transfer.
