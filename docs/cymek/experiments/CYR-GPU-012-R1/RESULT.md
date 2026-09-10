# CYR-GPU-012 / R1 — FINAL RESULT

**Status:** EXECUTED / COMPLETE  
**Campaign:** CYR-GPU-012-R1  
**Bundle SHA-256:** `a22b538396a3d0957a60a27f39b0cf3dd3b20874585b4c15a03207224f613d29`  
**GPU:** Tesla T4  
**Torch:** 2.11.0+cu128  
**Wall time:** 5860.10 s = 97.67 min  
**Claim ceiling:** controlled development mechanism evidence only.

The uploaded result ZIP passed CRC validation and contained all expected JSON campaign artifacts. The campaign receipt reports `COMPLETE`. All executed arms reached the frozen endpoint of **8,000 optimizer updates / 512,000 semantic row presentations** at batch 64. The ARK-002B split identity matched the frozen source split `0dd9305697045b0fbf4e7f268b46a4d7276e4794af5d78b60e999df914ae4236`.

## Primary preregistered decision

`MIXED_OR_INTERMEDIATE_REPRESENTATION_EFFECT`

This is the correct preregistered verdict because the primary V19 arm did not satisfy the preregistered minimum compact signal (`>=0.45`) in this seed, while the optional intermediate V4096 arm produced a very large positive result.

## Endpoint results

| Arm | Vocabulary classes | Parameters | STANDARD exact | DEV controller exact | M99 confirm | G50 confirm | Relative parameter displacement |
|---|---:|---:|---:|---:|---:|---:|---:|
| `S1_CHAR_V19` | 19 | 987,392 | **12.94%** | 23.44% | 1,600 | not reached | 0.7275 |
| `S1_CHAR_V4096` | 4,096 | 1,509,248 | **100.00%** | 98.44% | 1,600 | 800 | 2.0047 |
| `S1_CHAR_V24576` | 24,576 | 4,130,688 | **0.00%** | 0.00% | 2,200 | not reached | 3.1027 |

The V4096 arm reached 100% STANDARD by the 400-update / 25,600-row measurement and remained predominantly near ceiling thereafter, ending at 100%. V24576 remained at 0% STANDARD at the fixed endpoint. V19 showed weak, unstable late partial lift, ending at 12.94%.

## What is demonstrated

**DEMONSTRATED:** with the active arithmetic token IDs, segmentation, semantic stream, Cymek V5 block geometry, batch size, optimizer family/scalars, model seed, order seed, non-embedding initialization, and first 19 embedding rows matched, changing the declared tied embedding/output class-space size can change held-out capability formation dramatically.

**DEMONSTRATED:** under this one prospective seed and 512k-row endpoint, V4096 strongly outperformed both V19 and V24576 on the developmental STANDARD set.

**DEMONSTRATED:** the relationship is not monotonic in vocabulary size in this run. The observed ordering was V4096 >> V19 > V24576.

**DEMONSTRATED:** smaller parameter displacement is not sufficient to explain better held-out capability. V4096 moved much farther than V19 while generalizing far better.

## What is not demonstrated

**NOT_DEMONSTRATED:** that 4,096 is an optimal vocabulary size in general.

**NOT_DEMONSTRATED:** that the effect transfers to natural-language tokenization, larger models, broad reasoning, or production training.

**NOT_DEMONSTRATED:** that inactive-softmax competition is the unique causal mechanism. Cymek ties input embeddings and output weights, so vocabulary expansion jointly changes tied embedding/output class-space burden.

**NOT_DEMONSTRATED:** multi-seed replication. Hardware calibration resolved this campaign to one primary seed plus the optional V4096 arm.

**NOT_AUTHORIZED:** production tokenizer change, PRE500M, 500M training, broad reasoning claim, or AGI claim.

## Important interpretation

The result falsifies the simple story `smaller vocabulary -> better capability`. It also falsifies the simple story `more parameters/classes -> better capability` within this controlled setup. A more plausible working hypothesis is a **non-monotonic representation/optimization regime** in which the number of inactive tied output classes changes gradient competition, hidden-state margins, and/or embedding-output geometry; this remains a hypothesis until replicated and measured directly.

Historical CYR-GPU-011 must remain separate evidence: its V19 subject reached 56.47% STANDARD at ~44.89% of the ARK exposure box, whereas the fresh R1 V19 subject ended at 12.94% at a similar early-regime exposure. That seed-to-seed variation is itself a reason not to promote V4096 from a single subject.

## Next experiment

The highest-information follow-up is **R1B: replicated vocabulary-response curve + class-competition diagnostics**. It should map several vocabulary sizes around the observed intermediate optimum while preserving exact active tokenization and matched initialization, and directly record active-vs-inactive probability mass / margin and embedding-gradient partition. The goal is to distinguish a reproducible response curve from a one-seed resonance and to turn the qualitative class-space hypothesis into a measurable mechanism.
