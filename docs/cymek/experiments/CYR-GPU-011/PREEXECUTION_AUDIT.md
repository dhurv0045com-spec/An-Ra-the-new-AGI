# CYR-GPU-011 — PREEXECUTION AUDIT

Status: executable candidate under validation. No scientific execution has occurred.

## Hard findings and closure

| ID | Finding | Severity | Closure |
|---|---|---:|---|
| A01 | V9 semantic exposure was <22% of ARK-002B maximum | scientific | V11 records row presentations and targets ARK's 1,152,000-row box |
| A02 | Original V10 batch16 × 18k was only 25% of ARK exposure | scientific | V10 superseded before execution |
| A03 | A fixed 18k cap would still underdose V11 at batch32/16 | scientific | scoped V11 target becomes 18k/36k/72k for batch64/32/16 |
| A04 | V10 used an approximate regenerated task split | scientific | exact frozen ARK-002B manifest copied and hash-bound |
| A05 | ARK compact training supervises an answer-prefix BOS, but Cymek causal loss hard-excludes BOS targets | scientific | do **not** fake a match; compact bridge keeps canonical Cymek objective and documents this residual difference |
| A06 | An attempted extra-BOS renderer would insert an unsupervised token before the answer | fatal science | removed before freeze; regression requires canonical single-BOS Cymek rendering |
| A07 | A compact-positive/production-null result could be overcalled when production received less dose | scientific | final decision is semantic-exposure aware; underexposed null cannot be called divergence |
| A08 | Controller-only sustained G90 could overstate capability on a small control set | scientific | qualified G90 additionally requires final DEV_MEASUREMENT STANDARD >=.90 exact-with-EOS |
| A09 | Early V11 runner redundantly passed AdamW betas/eps/wd to a constructor that intentionally exposes only LR | fatal runtime | scoped compatibility accepts only exact canonical constants, delegates to canonical optimizer, and restores immediately |
| A10 | Global optimizer monkeypatch would be too invasive | engineering | replaced with context-managed process-local compatibility; regression checks restoration |
| A11 | Old resolver could fail the whole run if a desired campaign did not fit | reliability | V11 is progressive; calibration selects exposure-maximizing batch and wall clock truncates honestly |
| A12 | Reasoning diagnostics could be collapsed into an unjustified headline score | interpretation | diagnostics remain orthogonal named metrics; no aggregate reasoning score |
| A13 | ARK-017/018 could be mistaken for evidence because plans/data exist | evidence | final live Arkenstone audit still classifies both unexecuted |
| A14 | V11's stored audit SHA became stale as Arkenstone advanced 11 commits | provenance | rebound to live Arkenstone `c16718a7841c3cc3eba2b4b2c0388a0e36c0b530`; delta contains ARK-018 implementation/preexecution material, not a result |
| A15 | BRAMASTRA advanced beyond the older audit and added D02 development evidence | provenance | bound live BRAMASTRA `415250f44179f3310e5dc55addb21722290604fa`; D02 reviewed as orthogonal weak/null matched-teaching evidence with no V11 plan change |
| A16 | Research checkpoints existed on Drive but top-level V11 receipts exposed only paths | evidence durability | milestone/final receipts now carry durable path plus model/optimizer SHA-256 and counters from the checkpoint receipt |
| A17 | Returned ZIP exposed aggregate generation scores but not the final candidate-free rows | evidence auditability | final controller, final structural batteries, and sealed measurement now retain row-level generated outputs plus canonical prediction SHA-256 receipts |
| A18 | Structural battery used a 12-token decode cap while the controller gate used 8 | scientific comparability | unified V11 candidate-free capability/structural generation to a fixed 8-token cap |
| A19 | Replication could launch from controller G90 even when larger final measurement failed | compute/science | second production seed now launches only after sustained controller G90 **and** final STANDARD >=.90, with >=40 science minutes left |

## Cross-branch audit identity

Final prefreeze read-only heads reviewed:

- Arkenstone: `c16718a7841c3cc3eba2b4b2c0388a0e36c0b530`
- BRAMASTRA: `415250f44179f3310e5dc55addb21722290604fa`
- canonical Cymek: `28bf57a0d299a2c13a99fe0046616c00a1b8530c`
- Citadel: `1d27f9b0d770e30577de0a8671c909cb783b4ff1`
- Triquetra: `f23f0af42d90847cf1d2c244160c8203d1995b33`

Arkenstone's post-audit delta adds ARK-018 implementation/preexecution files and updates its experiment ledger, which still says ARK-017 and ARK-018 are not executed. BRAMASTRA's new D02 matched-teaching campaign is a small CPU development comparison; its chief review explicitly withholds a reliable multi-step advantage and any accelerator/model-promotion claim. Neither changes the preregistered V11 question.

## Exact data identity

Source: `experiments/ARK-002B/TASK_MANIFEST.json` on the audited Arkenstone lineage.

- split SHA256: `0dd9305697045b0fbf4e7f268b46a4d7276e4794af5d78b60e999df914ae4236`
- Git blob: `6c46fdf90139526b00e9041af2d511ed0ac24270`
- train: 500
- test: 197
- train/test canonical-pair overlap: 0

V11 deterministically partitions the 197 original test rows by content hash into 64 DEV_CONTROLLER / 85 DEV_MEASUREMENT / 48 SEALED_RESERVED. The role-manifest hash is generated from the actual partition and returned in results.

## Model identity

Both bridges use real `v5_model.core.initialize()` and Cymek `ModelSpec` geometry:

- 4 layers
- width 128
- Q4/KV2
- head dimension 32
- FFN 512
- context 512
- QK norm and tied embeddings

Only vocabulary size changes between V11's two Cymek bridges:

- COMPACT_BRIDGE: 19 symbols, exact 987,392 parameters
- PRODUCTION_BRIDGE: 24,576 vocabulary, exact 4,130,688 parameters

Parameter counts are mechanically checked at model construction.

## Remaining intentional differences from ARK-002B

V11 is a bridge, not an exact reproduction. Compared with Arkenstone Micro, Cymek retains:

- its canonical causal objective: BOS targets are excluded, whereas ARK's answer helper supervised a BOS prefix;
- V5 attention/GQA and QK-normalization implementation;
- Cymek initialization;
- Cymek optimizer parameter grouping, including no-decay normalization/QK scales;
- Cymek production backend and CUDA precision path;
- different fixed model/order seeds.

These are documented residual factors. A compact null cannot identify which one is causal.

## Evaluation firewall

Training sees only the 500 frozen train rows. DEV_CONTROLLER may control G50/G90 timing. DEV_MEASUREMENT cannot alter optimization; it validates the final G90 claim and hosts structural diagnostics. SEALED_RESERVED is measured only after the scientific decision when wall time allows.

The same eight-token candidate-free generation cap is used for controller qualification and structural/final prediction receipts. Final row-level predictions are retained for independent re-scoring; periodic traces remain aggregate to avoid unnecessary result bloat.

## Runtime / evidence safety

- full scientific execution refuses non-CUDA device;
- hardware calibration includes optimizer step and candidate-free generation;
- no all-or-nothing feasibility error is required for production stage;
- batch32/16 can target the same semantic box with 36k/72k updates;
- hard campaign wall 175 min; packaging reserve 5 min;
- Google Drive output is used by the operator notebook;
- research checkpoint directories retain `model.bin`, `optimizer.bin`, `counters.json`, and `receipt.json`; top-level receipts carry the model/optimizer hashes without copying tensors into the returned ZIP;
- exceptions package `FAILURE.json` plus partial evidence before re-raising when possible;
- bundle SHA256 is verified by Cell 2.

## Local/CI validation scope

Allowed evidence before Colab is deterministic only:

- compile core/runner/operator/notebook;
- exact manifest and task-firewall tests;
- exact parameter-count tests;
- canonical compact sequence/objective test;
- semantic-exposure resolver tests;
- exposure-aware verdict tests;
- scoped optimizer restoration/drift rejection;
- final prediction-receipt/hash and checkpoint-receipt contracts;
- one-update actual V5 CPU compact plumbing smoke.

This validation is not scientific training and is not GPU/TPU evidence.

## Claim boundary

Even a replicated production-bridge G90 result is controlled-task development evidence. `broad_reasoning_claim_authorized=false`, `production_promotion_authorized=false`, `pre500m_authorized=false`, and `training_500m_authorized=false` remain hard requirements.
