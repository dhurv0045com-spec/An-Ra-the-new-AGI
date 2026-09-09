# CYR-GPU-011 — PREEXECUTION AUDIT

Status: executable candidate under validation. No scientific execution has occurred.

## Hard findings and closure

| ID | Finding | Severity | Closure |
|---|---|---:|---|
| A01 | V9 semantic exposure was <22% of ARK-002B maximum | scientific | V11 records row presentations and targets ARK's 1,152,000-row box |
| A02 | Original V10 batch16 × 18k was only 25% of ARK exposure | scientific | V10 superseded before execution |
| A03 | A fixed 18k cap would still underdose V11 at batch32/16 | scientific | scoped V11 target becomes 18k/36k/72k for batch64/32/16 |
| A04 | V10 used an approximate regenerated task split | scientific | exact frozen ARK-002B manifest copied and hash-bound |
| A05 | ARK compact training supervises a second BOS before answer; normal Cymek render does not | scientific | compact updates reproduce ARK answer-prefix BOS in a scoped research renderer; production bridge retains normal Cymek semantics |
| A06 | A compact-positive/production-null result could be overcalled when production received less dose | scientific | final decision is semantic-exposure aware; underexposed null cannot be called divergence |
| A07 | Controller-only sustained G90 could overstate capability on a small control set | scientific | qualified G90 additionally requires final DEV_MEASUREMENT STANDARD >=.90 exact-with-EOS |
| A08 | Early V11 runner redundantly passed AdamW betas/eps/wd to a constructor that intentionally exposes only LR | fatal runtime | scoped compatibility accepts only exact canonical constants, delegates to canonical optimizer, and restores immediately |
| A09 | Global optimizer monkeypatch would be too invasive | engineering | replaced with context-managed process-local compatibility; regression checks restoration |
| A10 | Old resolver could fail the whole run if a desired campaign did not fit | reliability | V11 is progressive; calibration selects exposure-maximizing batch and wall clock truncates honestly |
| A11 | Reasoning diagnostics could be collapsed into an unjustified headline score | interpretation | diagnostics remain orthogonal named metrics; no aggregate reasoning score |
| A12 | ARK-017/018 could be mistaken for evidence because plans/data exist | evidence | both classified unexecuted until raw RESULT/receipt exists |

## Exact data identity

Source: `experiments/ARK-002B/TASK_MANIFEST.json` on live Arkenstone audit.

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

Only vocabulary size changes:

- COMPACT_BRIDGE: 19 symbols, exact 987,392 parameters
- PRODUCTION_BRIDGE: 24,576 vocabulary, exact 4,130,688 parameters

Parameter counts are mechanically checked at model construction.

## Remaining intentional differences from ARK-002B

V11 is a bridge, not an exact reproduction. Compared with Arkenstone Micro, Cymek retains:

- its V5 attention/GQA and QK-normalization implementation;
- Cymek initialization;
- Cymek optimizer parameter grouping, including no-decay normalization/QK scales;
- Cymek production backend and CUDA precision path;
- different fixed model/order seeds.

These are documented residual factors. A compact null cannot identify which one is causal.

## Evaluation firewall

Training sees only the 500 frozen train rows. DEV_CONTROLLER may control G50/G90 timing. DEV_MEASUREMENT cannot alter optimization; it is used to validate a G90 claim and for structural diagnostics. SEALED_RESERVED is measured only after the scientific decision when wall time allows.

## Runtime / evidence safety

- full scientific execution refuses non-CUDA device;
- hardware calibration includes optimizer step and candidate-free generation;
- no all-or-nothing feasibility error is required for production stage;
- hard wall 175 min; packaging reserve 5 min;
- Google Drive output is used by the operator notebook;
- exceptions package `FAILURE.json` plus partial evidence before re-raising;
- bundle SHA256 is verified by Cell 2.

## Local/CI validation scope

Allowed evidence before Colab is deterministic only:

- compile core/runner/operator/notebook;
- exact manifest and task-firewall tests;
- exact parameter-count tests;
- compact objective/BOS sequence test;
- semantic-exposure resolver tests;
- exposure-aware verdict tests;
- scoped optimizer restoration/drift rejection;
- one-update actual V5 CPU compact plumbing smoke.

This validation is not scientific training and is not GPU/TPU evidence.

## Claim boundary

Even a replicated production-bridge G90 result is controlled-task development evidence. `broad_reasoning_claim_authorized=false`, `production_promotion_authorized=false`, `pre500m_authorized=false`, and `training_500m_authorized=false` remain hard requirements.
