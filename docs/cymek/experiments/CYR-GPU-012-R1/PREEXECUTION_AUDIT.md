# CYR-GPU-012 / R1 — PREEXECUTION AUDIT

**Audit status:** STATIC SOURCE AUDIT PASS / OPERATOR CUDA TESTS STILL REQUIRED  
**Scientific status:** NOT EXECUTED  
**Frozen executable:** `d6953af2b0af64439dd9fd9ac0b2bc4987ed9c97`  
**Claim ceiling:** controlled development mechanism evidence only.

## Purpose

R1 is the causal follow-up to CYR-GPU-011. It asks whether the size of the tied embedding/output class space can by itself suppress early structural capability formation when the active arithmetic representation is held fixed.

The primary matched intervention is `CHAR_V19` versus `CHAR_V24576`. Both arms use the same 19 active token IDs, identical character segmentation, identical examples and order, batch 64, 8,000 updates / 512,000 semantic row presentations, the same Cymek V5 4L/128w block geometry, AdamW constants, causal answer+EOS objective, model seed, all shared non-embedding parameters, and the first 19 embedding rows. The larger arm differs by the additional tied embedding/output classes.

## Static checks completed

1. **Frozen data identity.** R1 reuses the exact CYR-GPU-011 copy of the commutation-free ARK-002B manifest: split SHA256 `0dd9305697045b0fbf4e7f268b46a4d7276e4794af5d78b60e999df914ae4236`, 500 TRAIN / 64 DEV_CONTROLLER / 85 DEV_MEASUREMENT / 48 previously-consumed holdout rows.
2. **No sealed overclaim.** The CYR-GPU-011 holdout has already been consumed, so R1 labels itself developmental and does not treat those rows as a fresh sealed confirmation.
3. **Matched initialization.** The V24576 arm explicitly copies all non-embedding parameters and the first 19 embedding rows from a separately constructed V19 reference using the same seed, and records SHA256 receipts. Extra rows remain the manipulated capacity/class-space term.
4. **Fixed semantic exposure.** Every scientific arm must reach exactly 8,000 updates at batch 64 = 512,000 semantic row presentations. The inherited G90 early-stop is disabled for R1 so all primary arms reach the same endpoint.
5. **Candidate-free behavioral metric.** Primary metric is DEV_MEASUREMENT STANDARD complete exact with valid EOS at the fixed endpoint. Loss is not the promotion criterion.
6. **Fail-visible inactive outputs.** In V4096/V24576 arms, a prediction of any inactive output ID >=19 is rendered as `<inactive:ID>` and therefore cannot disappear during decode and accidentally receive exact-match credit. A dedicated regression test is frozen with the executable.
7. **Runtime resolution is outcome-blind.** Calibration sees only throughput/VRAM/generation speed. It may choose two complete matched primary seed pairs if conservatively safe, otherwise one complete pair. It may not reduce exposure or drop one side of the primary comparison.
8. **Three-hour protection.** The scientific campaign has a hard wall of 175 minutes including a 5-minute packaging reserve. The resolver uses a 1.35 safety factor and refuses execution if even one complete matched primary pair cannot fit. Notebook setup/calibration occurs before that campaign timer and is normally a few additional minutes.
9. **Durability.** Completed arm wrappers are reusable only when experiment identity, vocab, seed, order, update count and semantic endpoint match exactly. Incompatible artifacts abort rather than overwrite. Result/failure evidence is packaged to Google Drive.
10. **Claim firewall.** No R1 outcome can directly authorize a production tokenizer change, PRE500M, 500M training, broad reasoning, or AGI.

## Runtime gates that must pass in Colab

Before scientific updates, the launcher must:

- read `PREREGISTRATION.json` and `RUN_READINESS.json` from the live branch;
- checkout the exact frozen executable commit;
- verify every frozen executable blob with `git hash-object`;
- run `py_compile` on the R1 core and runner;
- run the R1 and inherited CYR-GPU-011 unit-test suites;
- require CUDA;
- calibrate V19/V4096/V24576 at fixed batch 64;
- run the outcome-blind resolver; and
- print `R1 PREEXECUTION GATE: PASS` before Cell 1 is scientifically admissible.

These runtime checks have **not** been claimed as executed by this static audit. A failure in any one is a fail-closed preexecution result, not permission to modify the protocol after seeing outcomes.

## Decision interpretation

A strong replicated R1 result requires both prospective matched seed pairs to show a compact signal >=0.45 and a V19−V24576 endpoint gap >=0.30. A replicated absolute gap <=0.10 with compact signal >=0.45 argues that output-class burden is not the primary explanation at this endpoint. Intermediate/mixed outcomes remain unresolved rather than being forced into a binary story.

Even a strong R1 positive isolates the **joint tied embedding/output class-space burden**. Because Cymek ties input embeddings and output projection, R1 does not yet distinguish input-capacity effects from softmax competition; that would be a later factorization experiment.

## Verdict

**READY FOR OPERATOR COLAB CUDA PREEXECUTION GATE.** The experiment is preregistered, source-frozen, causally narrower than CYR-GPU-011, protected against inactive-output scoring artifacts, fixed-exposure, fail-closed, resumable at completed-arm granularity, and bounded to a 175-minute scientific campaign wall. Scientific result status remains **NOT_EXECUTED** until a returned bundle is independently audited.
